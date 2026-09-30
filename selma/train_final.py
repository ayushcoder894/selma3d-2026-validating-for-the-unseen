"""The single long run for the final submission.

Differences from train.py, each tied to a measured failure (see README "One-run plan"):

* plain BCE (pos_weight 1): the old pos_weight=5 inflated every probability, and the
  thresholds tuned to undo it on seen families do not transfer to unseen structures;
* GPU zoom/rotation, bias-field and low-res augmentation for unseen-structure
  generalisation (gpu_aug.py);
* no gradient accumulation: v8 made only 10k optimizer updates in 3.7 h;
* 50% foreground-centred crops instead of 70%, for more background exposure;
* EMA weights for validation and export, to damp checkpoint-selection noise;
* warm start from v8's final weights, so its 3.7 h of training is not relearned.

The learning-rate schedule is driven by wall-clock progress towards a fixed deadline, not
by an epoch count. Validation time, restarts and speed changes then cannot push the run
past its window: it always ends on time with a fully annealed learning rate.
"""
import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import CATEGORIES, Patches, load_index
from data import Z_HI, Z_LO
from gpu_aug import augment_batch
from window_norm import PerWindowNorm, encoder_modules, normalize_per_window
from losses import SelmaLoss, deep_supervision_loss
from monitor import monitor_validate
from train import build_model


class EMA:
    """Exponential moving average kept as a model copy.

    DynUNet's state_dict exposes the same tensors under two names (skip_layers aliases
    the encoder), so averaging state_dict entries would double memory and snapshot size.
    named_parameters() de-duplicates; the copy's state_dict() keeps the aliasing.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.updates = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # Short ramp so the average is not dominated by the starting weights.
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        src = dict(model.named_parameters())
        dst = [p for _, p in self.module.named_parameters()]
        new = [src[n].detach() for n, _ in self.module.named_parameters()]
        torch._foreach_mul_(dst, d)
        torch._foreach_add_(dst, new, alpha=1 - d)
        bufs = dict(model.named_buffers())
        for n, b in self.module.named_buffers():
            b.copy_(bufs[n])


def lr_at(progress, peak, warmup, floor):
    if progress < warmup:
        return peak * max(progress / warmup, 0.02)
    t = min(1.0, (progress - warmup) / max(1e-9, 1 - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * t))


def atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    try:
        torch.save(obj, tmp)
        tmp.replace(path)
        return True
    except Exception as exc:  # a failed save must never kill a multi-hour run
        tmp.unlink(missing_ok=True)
        free = shutil.disk_usage(Path(path).parent).free / 2 ** 20
        print(f"  WARNING: could not write {path} ({exc}); {free:.0f} MB free", flush=True)
        return False


def backup(src, backup_dir):
    """Best-effort copy to the external drive, which drops intermittently."""
    if not backup_dir:
        return
    try:
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, Path(backup_dir) / Path(src).name)
    except Exception as exc:
        print(f"  (backup of {Path(src).name} skipped: {exc})", flush=True)


HEAD_PREFIXES = ("output_block", "deep_supervision_heads")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/v11")
    ap.add_argument("--holdout", default="selma/holdout_v11.json")
    ap.add_argument("--init", default="runs/v8/fold0/last.pt")
    ap.add_argument("--resume", default="")
    ap.add_argument("--deadline", type=float, required=True,
                    help="unix time at which training must be finished")
    ap.add_argument("--peak-lr", type=float, default=3e-4)
    ap.add_argument("--floor-lr", type=float, default=2e-6)
    ap.add_argument("--warmup", type=float, default=0.015, help="fraction of the run")
    ap.add_argument("--weight-decay", type=float, default=3e-5)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--p-fg", type=float, default=0.5)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--w-cldice", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=250, help="iterations per epoch")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patch", type=int, nargs=3, default=[96, 96, 96])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-epochs", type=int, nargs="+", default=[8, 14],
                    help="early validation epochs, to catch a broken run quickly")
    ap.add_argument("--val-every", type=int, default=15)
    ap.add_argument("--save-every", type=int, default=3)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    ap.add_argument("--min-sizes", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--backup-dir", default="")
    ap.add_argument("--no-gpu-aug", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="train the decoder only, keeping the SSL encoder fixed")
    ap.add_argument("--heads-warmup", type=float, default=0.0,
                    help="train only the output heads until this fraction of the run "
                         "(linear-probe-then-fine-tune: a random head would otherwise "
                         "distort pretrained features through its early gradients)")
    ap.add_argument("--encoder-lr-mult", type=float, default=1.0,
                    help="learning-rate multiplier for the encoder (e.g. 0.1: stay near SSL)")
    ap.add_argument("--norm", choices=["percentile", "zscore"], default="percentile",
                    help="input normalisation; zscore reads raw volumes (data/prep_raw)")
    ap.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                    help="sgd: momentum 0.99, Nesterov, poly decay (standard for 3D U-Nets)")
    ap.add_argument("--schedule", choices=["cosine", "poly"], default="cosine",
                    help="poly: lr = peak * (1 - progress)^0.9, no warm-up")
    ap.add_argument("--uniform-sampling", action="store_true",
                    help="sample volumes uniformly instead of balancing the 4 categories")
    ap.add_argument("--per-window-norm", action="store_true",
                    help="normalise each patch / sliding window to its own percentiles")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    index = load_index()
    spec = json.loads(Path(a.holdout).read_text())
    holdout = set(spec["stems"])
    # Leave-family-out runs also withhold volumes that are not monitored.
    exclude = set(spec.get("exclude", []))
    tr = [r for r in index if r["stem"] not in holdout and r["stem"] not in exclude]
    va = [r for r in index if r["stem"] in holdout]
    assert len(va) == len(holdout), "holdout stems missing from index"
    # An empty holdout ("stems": []) trains on every volume; snapshots are still saved
    # on the validation schedule, just without metrics.
    print(f"train {len(tr)} volumes / holdout {len(va)} volumes")
    for c in CATEGORIES:
        print(f"  {c:20s} train {sum(r['category'] == c for r in tr):3d}"
              f"  holdout {sum(r['category'] == c for r in va):3d}")

    model = build_model().to(dev)
    if a.freeze_encoder:
        for module in encoder_modules(model):
            for prm in module.parameters():
                prm.requires_grad_(False)
    trainable = [prm for prm in model.parameters() if prm.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in trainable):,} of "
          f"{sum(p.numel() for p in model.parameters()):,}", flush=True)
    enc_ids = {id(prm) for mod in encoder_modules(model) for prm in mod.parameters()}
    groups = [
        {"params": [q for q in trainable if id(q) in enc_ids], "lr_mult": a.encoder_lr_mult},
        {"params": [q for q in trainable if id(q) not in enc_ids], "lr_mult": 1.0},
    ]
    groups = [g for g in groups if g["params"]]
    if a.optimizer == "sgd":
        opt = torch.optim.SGD(groups, lr=a.peak_lr, momentum=0.99, nesterov=True,
                              weight_decay=a.weight_decay)
    else:
        opt = torch.optim.AdamW(groups, lr=a.peak_lr, weight_decay=a.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    crit = SelmaLoss(w_cldice=a.w_cldice, pos_weight=a.pos_weight).to(dev)

    start_time, epoch, best, history = time.time(), 0, -1.0, []
    if a.resume:
        st = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        # The saved state restores the old per-group LR multipliers; the command line wins,
        # so a run can be resumed with a different --encoder-lr-mult.
        for g_new, g_opt in zip(groups, opt.param_groups):
            if g_opt.get("lr_mult") != g_new["lr_mult"]:
                print(f"  lr_mult {g_opt.get('lr_mult')} -> {g_new['lr_mult']} on resume",
                      flush=True)
            g_opt["lr_mult"] = g_new["lr_mult"]
        scaler.load_state_dict(st["scaler"])
        ema = EMA(model, a.ema_decay)
        ema.module.load_state_dict(st["ema"])
        ema.updates = int(st["ema_updates"])
        start_time, epoch = float(st["start_time"]), int(st["epoch"])
        best, history = float(st["best"]), list(st["history"])
        print(f"resumed {a.resume} at epoch {epoch}, best {best:.4f}", flush=True)
    else:
        if a.init and a.init.lower() != "none":
            init = torch.load(a.init, map_location="cpu", weights_only=False)
            src = init["model"] if "model" in init else init
            # Pretraining checkpoints may carry extra output channels (self-supervised
            # heads); transfer every tensor whose shape matches and re-initialise the rest.
            own = model.state_dict()
            keep = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
            skipped = sorted(set(own) - set(keep))
            model.load_state_dict(keep, strict=False)
            print(f"initialised from {a.init} ({len(keep)} tensors; re-initialised: "
                  f"{', '.join(skipped) if skipped else 'none'})", flush=True)
        else:
            print("random initialisation (training from scratch)", flush=True)
        ema = EMA(model, a.ema_decay)

    span = a.deadline - start_time
    if span <= 0:
        raise SystemExit("deadline is before the run start")
    print(f"run window {span / 3600:.2f} h, deadline "
          f"{time.strftime('%H:%M', time.localtime(a.deadline))}", flush=True)

    ds = Patches(tr, patch=tuple(a.patch), steps=a.steps * a.batch, p_fg=a.p_fg,
                 seed=1000 + epoch, norm=a.norm, uniform=a.uniform_sampling)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True,
                    persistent_workers=True, drop_last=True)

    def progress():
        return min(1.0, max(0.0, (time.time() - start_time) / span))

    def evaluate(tag):
        # Validation inference leaves large cached blocks; at 7.6 of 8 GB the WDDM driver
        # then pages VRAM to system RAM and the next epoch ran 3x slower (299 s vs 92 s).
        torch.cuda.empty_cache()
        net = PerWindowNorm(ema.module) if a.per_window_norm else ema.module
        m = monitor_validate(net, va, tuple(a.patch), dev, norm=a.norm)
        torch.cuda.empty_cache()
        return m

    def save_latest():
        # Keep the previous resume point too: if the newest one is ever unreadable, the
        # supervisor falls back to it instead of restarting the run from scratch.
        if (out / "latest.pt").exists():
            try:
                shutil.copyfile(out / "latest.pt", out / "latest_prev.pt")
            except Exception as exc:
                print(f"  (could not keep latest_prev.pt: {exc})", flush=True)
        atomic_save({
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(), "ema": ema.module.state_dict(),
            "ema_updates": ema.updates, "start_time": start_time, "epoch": epoch,
            "best": best, "history": history, "args": vars(a),
        }, out / "latest.pt")

    trainable_ids = {id(q) for q in trainable}
    heads_state = [False]
    done = progress() >= 1.0
    while not done:
        t0, losses = time.time(), []
        model.train()
        for batch in dl:
            p = progress()
            if p >= 1.0:
                done = True
                break
            lr = (a.peak_lr * (1 - p) ** 0.9 if a.schedule == "poly"
                  else lr_at(p, a.peak_lr, a.warmup, a.floor_lr))
            for g in opt.param_groups:
                g["lr"] = lr * g.get("lr_mult", 1.0)
            heads_only = p < a.heads_warmup
            if heads_only != heads_state[0]:
                # Frozen parameters get no gradient, so SGD leaves their momentum
                # buffers untouched until they are released.
                for name, prm in model.named_parameters():
                    if not name.startswith(HEAD_PREFIXES):
                        prm.requires_grad_(not heads_only and id(prm) in trainable_ids)
                heads_state[0] = heads_only
                print(f"  {'heads-only warmup' if heads_only else 'all layers released'}"
                      f" at progress {p * 100:.1f}%", flush=True)
            x = batch["image"].to(dev, non_blocking=True)
            y = batch["label"].to(dev, non_blocking=True)
            tub = batch["tubular"].to(dev, non_blocking=True)
            if not a.no_gpu_aug:
                with torch.no_grad():
                    x, y = augment_batch(
                        x, y, clamp=(Z_LO, Z_HI) if a.norm == "zscore" else (-0.5, 2.0))
            if a.per_window_norm:
                with torch.no_grad():
                    x = normalize_per_window(x)
            with torch.autocast("cuda"):
                loss = deep_supervision_loss(crit, model(x), y, tub)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            lv = loss.item()
            if not math.isfinite(lv):
                raise RuntimeError(f"non-finite loss {lv} at epoch {epoch + 1}")
            losses.append(lv)
        if not losses:
            break
        epoch += 1
        line = (f"epoch {epoch:4d}  loss {np.mean(losses):.4f}  lr {lr:.2e}  "
                f"{time.time() - t0:.0f}s  progress {progress() * 100:5.1f}%")

        if not va and (epoch % a.val_every == 0 or done):
            snap = {"model": ema.module.state_dict(), "epoch": epoch, "patch": a.patch,
                    "per_window_norm": bool(a.per_window_norm), "norm": a.norm}
            atomic_save(snap, out / f"ema_e{epoch:04d}.pt")
            backup(out / f"ema_e{epoch:04d}.pt", a.backup_dir)
            line += "  (snapshot saved, no holdout)"
        elif va and (epoch in a.val_epochs or epoch % a.val_every == 0 or done):
            m = evaluate("ema")
            iso, con = m["isolated"], m["contiguous"]
            line += (f"  | iso D={iso['dice']:.3f} PQ={iso['panoptic_quality']:.3f}"
                     f"  cont D={con['dice']:.3f} cD={con['cldice']:.3f}"
                     f"  || score {m['challenge_score']:.4f} thr {m['threshold']:.2f}"
                     f" min {m['minimum_component_size']}")
            per = "  ".join(
                f"{c.split('_')[0][:5]}/{c.split('_')[1][:4]} D={m[c]['dice']:.3f}"
                + (f" PQ={m[c]['panoptic_quality']:.3f}" if "panoptic_quality" in m[c] else "")
                + (f" cD={m[c]['cldice']:.3f}" if "cldice" in m[c] else "")
                for c in CATEGORIES if c in m)
            history.append(dict(epoch=epoch, progress=progress(), **m))
            (out / "history.json").write_text(json.dumps(history, indent=1))
            snap = {"model": ema.module.state_dict(), "metrics": m,
                    "per_window_norm": bool(a.per_window_norm), "norm": a.norm,
                    "threshold": m["threshold"],
                    "minimum_component_size": m["minimum_component_size"],
                    "epoch": epoch, "patch": a.patch}
            atomic_save(snap, out / f"ema_e{epoch:04d}.pt")
            backup(out / f"ema_e{epoch:04d}.pt", a.backup_dir)
            if m["challenge_score"] > best:
                best = m["challenge_score"]
                atomic_save(snap, out / "best_ema.pt")
                backup(out / "best_ema.pt", a.backup_dir)
                line += "  *"
            line += "\n      " + per
        print(line, flush=True)

        if epoch % a.save_every == 0 or done:
            save_latest()
        if (out / "STOP").exists():
            print("STOP file found - saving and exiting", flush=True)
            save_latest()
            return

    save_latest()
    final = {"model": ema.module.state_dict(), "epoch": epoch,
             "per_window_norm": bool(a.per_window_norm), "norm": a.norm,
             "patch": a.patch}
    if history:
        final.update(metrics=history[-1], threshold=history[-1]["threshold"],
                     minimum_component_size=history[-1]["minimum_component_size"])
    atomic_save(final, out / "final_ema.pt")
    backup(out / "final_ema.pt", a.backup_dir)
    print(f"=== TRAINING FINISHED: {epoch} epochs, best holdout {best:.4f} ===", flush=True)


if __name__ == "__main__":
    main()
