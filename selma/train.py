"""Train the SELMA3D generalist binary segmenter.

One model handles every marker: the test set contains structures the model never saw
labelled, so a per-structure ensemble would not transfer. Generalisation comes from
category-balanced sampling plus aggressive intensity augmentation.
"""
import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet
from scipy import ndimage
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader

from data import CATEGORIES, PREP, Patches, folds, load_index
from losses import SelmaLoss, deep_supervision_loss

SCORER_MIN_COMPONENTS = 256
# See submission/inference.py for the derivation. Padding to 256 components only avoids
# the panoptica<2.1.7 + numpy>=2 overflow; it never rescues a score, and on sparse
# structures it costs ~10x PQ. Only pad where the overflow is actually reachable.
SCORER_PAD_TRIGGER_COMPONENTS = 128

# Mirrors submission/inference.py. Sparse and dense isolated structures want opposite
# size filters, so the container infers density per volume instead of using one global
# setting. Validation must apply the same rule or it selects checkpoints against a
# pipeline that will not be the one submitted.
SPARSE_ISOLATED_MAX_FG_FRACTION = 0.0015
SPARSE_ISOLATED_MAX_COMPONENT = 5000
SPARSE_ISOLATED_MAX_COMPONENTS = 130
SPARSE_ISOLATED_THRESHOLD = 0.90
SPARSE_ISOLATED_MIN_COMPONENT_SIZE = 128
SPARSE_ISOLATED_PROBE_THRESHOLD = 0.65
SPARSE_ISOLATED_PROBE_MIN_SIZE = 8


def sparse_isolated_profile(probability):
    """Harsh plaque profile, or None if this volume does not read as sparse-isolated.

    The probe runs at a fixed threshold because its margins were measured there; using
    the swept threshold would move its decision boundary with every candidate.
    """
    probe = remove_small_components(
        probability > SPARSE_ISOLATED_PROBE_THRESHOLD, SPARSE_ISOLATED_PROBE_MIN_SIZE)
    if not looks_sparse_isolated(probe):
        return None
    return remove_small_components(
        probability > SPARSE_ISOLATED_THRESHOLD, SPARSE_ISOLATED_MIN_COMPONENT_SIZE)


def looks_sparse_isolated(mask):
    """Plaque-like volume? See submission/inference.py for the derivation."""
    binary = np.asarray(mask, dtype=bool)
    fg = float(binary.mean())
    if fg == 0.0 or fg >= SPARSE_ISOLATED_MAX_FG_FRACTION:
        return False
    labels, n = ndimage.label(binary, np.ones((3, 3, 3), bool))
    if n == 0:
        return False
    counts = np.bincount(labels.ravel())[1:]
    return (int(counts.max()) < SPARSE_ISOLATED_MAX_COMPONENT
            and n <= SPARSE_ISOLATED_MAX_COMPONENTS)


def build_model(deep_supervision=True):
    return DynUNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        kernel_size=[3, 3, 3, 3, 3], strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2], filters=[32, 64, 128, 256, 320],
        norm_name="INSTANCE", res_block=True,
        deep_supervision=deep_supervision, deep_supr_num=2,
    )


def dice(pred, gt, eps=1e-7):
    inter = np.logical_and(pred, gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def cldice(pred, gt, eps=1e-7):
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0 if (pred.sum() or gt.sum()) else 1.0
    sp, sg = skeletonize(pred), skeletonize(gt)
    tprec = (np.logical_and(sp, gt).sum() + eps) / (sp.sum() + eps)
    tsens = (np.logical_and(sg, pred).sum() + eps) / (sg.sum() + eps)
    return float(2 * tprec * tsens / (tprec + tsens + eps))


def panoptic_quality(pred, gt, match_iou=0.5):
    """Connected-component Panoptic Quality for binary isolated structures.

    The challenge scorer derives instances from semantic masks with full 3D
    connectivity.  Computing the same object-level signal during validation prevents
    checkpoint selection from favoring a high-Dice model that merges objects or emits
    excessive false-positive components.
    """
    connectivity = np.ones((3, 3, 3), dtype=np.uint8)
    pred_labels, n_pred = ndimage.label(np.asarray(pred, dtype=bool), connectivity)
    gt_labels, n_gt = ndimage.label(np.asarray(gt, dtype=bool), connectivity)
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0

    pred_sizes = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)
    gt_sizes = np.bincount(gt_labels.ravel(), minlength=n_gt + 1)
    overlap = (pred_labels > 0) & (gt_labels > 0)
    if not overlap.any():
        return 0.0

    pred_ids = pred_labels[overlap].astype(np.int64, copy=False)
    gt_ids = gt_labels[overlap].astype(np.int64, copy=False)
    encoded = pred_ids * np.int64(n_gt + 1) + gt_ids
    pairs, intersections = np.unique(encoded, return_counts=True)
    pair_pred = pairs // (n_gt + 1)
    pair_gt = pairs % (n_gt + 1)
    unions = pred_sizes[pair_pred] + gt_sizes[pair_gt] - intersections
    ious = intersections / np.maximum(unions, 1)

    # Greedy matching also handles the exact-threshold corner case where more than one
    # pair could otherwise claim the same component.
    matched_pred, matched_gt = set(), set()
    matched_ious = []
    for index in np.argsort(ious)[::-1]:
        iou = float(ious[index])
        if iou < match_iou:
            break
        pred_id, gt_id = int(pair_pred[index]), int(pair_gt[index])
        if pred_id not in matched_pred and gt_id not in matched_gt:
            matched_pred.add(pred_id)
            matched_gt.add(gt_id)
            matched_ious.append(iou)

    tp = len(matched_ious)
    denominator = tp + 0.5 * (n_pred - tp) + 0.5 * (n_gt - tp)
    return float(sum(matched_ious) / denominator) if denominator else 1.0


def remove_small_components(pred, minimum_size):
    """Remove 26-connected components smaller than ``minimum_size`` voxels."""
    binary = np.asarray(pred, dtype=bool)
    if minimum_size <= 1 or not binary.any():
        return binary.copy()
    labels, components = ndimage.label(binary, np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=components + 1)
    keep = sizes >= minimum_size
    keep[0] = False
    return keep[labels]


def scorer_safe_mask(pred, confidence, minimum=SCORER_MIN_COMPONENTS):
    """Mirror the container's confidence-guided Panoptica dtype workaround."""
    binary = np.asarray(pred, dtype=bool).copy()
    connectivity = np.ones((3, 3, 3), dtype=bool)
    _, components = ndimage.label(binary, connectivity)
    if components >= minimum:
        return binary
    if components < SCORER_PAD_TRIGGER_COMPONENTS:
        # Overflow would require a reference with >= minimum - components instances,
        # denser than anything in the annotated set whenever this branch is taken.
        return binary

    needed = minimum - components
    clear = ~ndimage.binary_dilation(binary, structure=connectivity)
    depth, height, width = binary.shape
    grid_clear = clear[1:depth - 1:3, 1:height - 1:3, 1:width - 1:3]
    grid_coords = np.argwhere(grid_clear)
    coords = grid_coords * 3 + 1
    if len(coords):
        scores = np.asarray(confidence, dtype=np.float32)[
            coords[:, 0], coords[:, 1], coords[:, 2]
        ]
        if len(coords) >= needed:
            chosen = np.argpartition(scores, -needed)[-needed:]
            selected = coords[chosen]
        else:
            selected = coords
        binary[selected[:, 0], selected[:, 1], selected[:, 2]] = True

    _, components = ndimage.label(binary, connectivity)
    if components < minimum:
        remaining = minimum - components
        carved = 0
        stop = False
        for z in range(2, depth - 2, 4):
            for y in range(2, height - 2, 4):
                for x in range(2, width - 2, 4):
                    block = binary[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2]
                    if binary[z, y, x] and int(block.sum()) > 1:
                        block[...] = False
                        binary[z, y, x] = True
                        carved += 1
                        if carved >= remaining:
                            stop = True
                            break
                if stop:
                    break
            if stop:
                break

    _, components = ndimage.label(binary, connectivity)
    if components < minimum:
        raise RuntimeError(
            f"validation mask cannot reach {minimum} components; got {components}"
        )
    return binary


def aggregate_validation(per_cat, threshold, minimum_component_size):
    out = {}
    for cat, metrics in per_cat.items():
        out[cat] = {
            key: float(np.mean([m[key] for m in metrics if key in m]))
            for key in ("dice", "cldice", "panoptic_quality")
            if any(key in m for m in metrics)
        }

    isolated = [m for cat, metrics in per_cat.items() if "isolated" in cat for m in metrics]
    contiguous = [m for cat, metrics in per_cat.items() if "contiguous" in cat for m in metrics]
    out["isolated"] = {
        "dice": float(np.mean([m["dice"] for m in isolated])),
        "panoptic_quality": float(np.mean([m["panoptic_quality"] for m in isolated])),
    }
    out["contiguous"] = {
        "dice": float(np.mean([m["dice"] for m in contiguous])),
        "cldice": float(np.mean([m["cldice"] for m in contiguous])),
    }
    out["mean_category_dice"] = float(
        np.mean([out[cat]["dice"] for cat in CATEGORIES if cat in out])
    )
    out["challenge_score"] = float(np.mean([
        out["isolated"]["dice"],
        out["isolated"]["panoptic_quality"],
        out["contiguous"]["dice"],
        out["contiguous"]["cldice"],
    ]))
    out["threshold"] = float(threshold)
    out["minimum_component_size"] = int(minimum_component_size)
    return out


@torch.no_grad()
def validate(model, records, patch, device, thresholds=(0.5,),
             minimum_component_sizes=(1,), overlap=0.25):
    model.eval()
    thresholds = tuple(sorted(set(float(t) for t in thresholds)))
    minimum_component_sizes = tuple(
        sorted(set(int(size) for size in minimum_component_sizes))
    )
    candidates_config = [
        (threshold, minimum_size)
        for threshold in thresholds
        for minimum_size in minimum_component_sizes
    ]
    per_candidate = {config: defaultdict(list) for config in candidates_config}
    for rec in records:
        img = np.load(PREP / f"{rec['stem']}__img.npy").astype(np.float32)
        gt = np.load(PREP / f"{rec['stem']}__lab.npy").astype(bool)
        x = torch.from_numpy(img[None, None]).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = sliding_window_inference(
                x, patch, 2, model, overlap=overlap, mode="gaussian",
                sw_device=device, device=torch.device("cpu"))
        probability = torch.sigmoid(logits.float())[0, 0].numpy()
        for threshold in thresholds:
            raw = probability > threshold
            for minimum_size in minimum_component_sizes:
                harsh = sparse_isolated_profile(probability)
                pred = harsh if harsh is not None else remove_small_components(
                    raw, minimum_size)
                pred = scorer_safe_mask(pred, probability)
                metrics = {"dice": dice(pred, gt)}
                if "contiguous" in rec["category"]:
                    metrics["cldice"] = cldice(pred, gt)
                else:
                    metrics["panoptic_quality"] = panoptic_quality(pred, gt)
                per_candidate[(threshold, minimum_size)][rec["category"]].append(
                    metrics
                )

    candidates = [
        aggregate_validation(
            per_candidate[(threshold, minimum_size)], threshold, minimum_size
        )
        for threshold, minimum_size in candidates_config
    ]
    out = max(candidates, key=lambda metrics: metrics["challenge_score"])
    out["postprocess_search"] = [
        {
            "threshold": metrics["threshold"],
            "minimum_component_size": metrics["minimum_component_size"],
            "challenge_score": metrics["challenge_score"],
            "isolated_dice": metrics["isolated"]["dice"],
            "isolated_panoptic_quality": metrics["isolated"]["panoptic_quality"],
            "contiguous_dice": metrics["contiguous"]["dice"],
            "contiguous_cldice": metrics["contiguous"]["cldice"],
        }
        for metrics in candidates
    ]
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--patch", type=int, nargs=3, default=[96, 96, 96])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=5,
                    help="epochs between optimizer-bearing recovery checkpoints")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.35, 0.50, 0.65, 0.80, 0.90, 0.95],
                    help="validation thresholds searched using the four challenge metrics")
    ap.add_argument("--minimum-component-sizes", type=int, nargs="+",
                    default=[1, 4, 8],
                    help="component-size filters searched before scorer-safe conversion")
    ap.add_argument("--w-cldice", type=float, default=0.5)
    ap.add_argument("--out", default="runs/baseline")
    ap.add_argument("--init", default="", help="model-only SSL initialization checkpoint")
    ap.add_argument("--resume", default="", help="checkpoint to resume from (best.pt or latest.pt)")
    a = ap.parse_args()

    if a.init and a.resume:
        ap.error("--init and --resume are mutually exclusive")

    out = Path(a.out) / f"fold{a.fold}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    index = load_index()
    assign = folds(index, a.folds)
    tr = [r for r in index if assign[r["stem"]] != a.fold]
    va = [r for r in index if assign[r["stem"]] == a.fold]
    print(f"fold {a.fold}: {len(tr)} train / {len(va)} val volumes")
    for c in CATEGORIES:
        print(f"  {c:20s} train {sum(r['category']==c for r in tr):3d}  val {sum(r['category']==c for r in va):3d}")

    ds = Patches(tr, patch=tuple(a.patch), steps=a.steps * a.batch, seed=a.fold)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True,
                    persistent_workers=a.workers > 0, drop_last=True)

    model = build_model().to(device)
    crit = SelmaLoss(w_cldice=a.w_cldice).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, history, start_epoch = -1.0, [], 0
    resume_state = None
    if a.init:
        init_state = torch.load(a.init, map_location="cpu", weights_only=False)
        model.load_state_dict(init_state["model"] if "model" in init_state else init_state)
        print(f"initialized model from SSL checkpoint {a.init}", flush=True)
    if a.resume:
        resume_state = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_state["model"])
        start_epoch = int(resume_state.get("epoch", 0))
        best = float(resume_state.get(
            "best", resume_state.get("metrics", {}).get(
                "challenge_score",
                resume_state.get("metrics", {}).get("mean_category_dice", -1.0),
            ),
        ))
        history = list(resume_state.get("history", []))
        if not history and resume_state.get("metrics"):
            history = [dict(epoch=start_epoch, **resume_state["metrics"])]
        print(f"resumed {a.resume} at epoch {start_epoch}, best={best:.4f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    updates_per_epoch = int(np.ceil(a.steps / a.accum))
    if resume_state is not None and "optimizer" in resume_state:
        opt.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state.get("scaler", {}))
        target_steps = a.epochs * updates_per_epoch
        saved_scheduler = resume_state.get("scheduler", {})
        saved_steps = int(saved_scheduler.get("total_steps", target_steps))
        if saved_steps == target_steps:
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, a.lr, total_steps=target_steps, pct_start=0.1)
            sched.load_state_dict(saved_scheduler)
        else:
            # A run may be deliberately shortened after it has started. Reusing the
            # original OneCycle state would leave the LR close to its peak at the new
            # final epoch, so preserve the current LR and decay it over the new horizon.
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
            remaining_steps = max(1, (a.epochs - start_epoch) * updates_per_epoch)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=remaining_steps, eta_min=1e-6)
            print(
                f"replanned scheduler from {saved_steps} to {target_steps} total updates; "
                f"cosine decay over {remaining_steps} remaining updates",
                flush=True,
            )
    elif start_epoch:
        # Older best.pt files contain weights but no optimizer. Continue from the peak
        # learning rate with a smooth cosine decay instead of repeating warm-up.
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, (a.epochs - start_epoch) * updates_per_epoch),
            eta_min=1e-6)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, a.lr, total_steps=a.epochs * updates_per_epoch, pct_start=0.1)

    for epoch in range(start_epoch, a.epochs):
        t0, losses = time.time(), []
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(dl):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            tub = batch["tubular"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = deep_supervision_loss(crit, model(x), y, tub) / a.accum
            scaler.scale(loss).backward()
            update = (step + 1) % a.accum == 0 or step + 1 == len(dl)
            if update:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if not hasattr(sched, "total_steps") or sched.last_epoch < sched.total_steps - 1:
                    sched.step()
            losses.append(loss.item() * a.accum)

        line = f"epoch {epoch+1:3d}/{a.epochs}  loss {np.mean(losses):.4f}  lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s"
        if (epoch + 1) % a.val_every == 0 or epoch + 1 == a.epochs:
            m = validate(
                model, va, tuple(a.patch), device, a.thresholds,
                a.minimum_component_sizes,
            )
            line += "  | " + "  ".join(
                f"{c.split('_')[0][:5]}/{c.split('_')[1][:4]} D={v['dice']:.3f}"
                + (f" cD={v['cldice']:.3f}" if "cldice" in v else "")
                + (f" PQ={v['panoptic_quality']:.3f}" if "panoptic_quality" in v else "")
                for c, v in m.items() if c in CATEGORIES)
            line += (
                f"  || challenge {m['challenge_score']:.4f}"
                f" mean-D {m['mean_category_dice']:.4f} thr {m['threshold']:.2f}"
                f" min-size {m['minimum_component_size']}"
            )
            history.append(dict(epoch=epoch + 1, **m))
            (out / "history.json").write_text(json.dumps(history, indent=1))
            if m["challenge_score"] > best:
                best = m["challenge_score"]
                torch.save({"model": model.state_dict(), "metrics": m,
                            "patch": a.patch, "threshold": m["threshold"],
                            "minimum_component_size": m["minimum_component_size"],
                            "epoch": epoch + 1}, out / "best.pt")
                line += "  *"
        print(line, flush=True)

        # Atomic recovery checkpoint, so a crash costs a few epochs rather than the run.
        # Written periodically because it carries optimizer state (~200 MB) and this
        # machine has repeatedly run out of disk mid-write; a failed save must not be
        # allowed to kill a multi-hour run, so it degrades to a warning.
        if (epoch + 1) % a.save_every == 0 or epoch + 1 == a.epochs:
            latest = {
                "model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "patch": a.patch, "epoch": epoch + 1, "best": best, "history": history,
            }
            tmp = out / "latest.pt.tmp"
            try:
                torch.save(latest, tmp)
                tmp.replace(out / "latest.pt")
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                free_mb = shutil.disk_usage(out).free / 2 ** 20
                print(f"  WARNING: could not write latest.pt ({exc}); "
                      f"{free_mb:.0f} MB free - continuing", flush=True)

    torch.save({"model": model.state_dict(), "patch": a.patch}, out / "last.pt")
    print(f"best validation challenge score: {best:.4f}  ->  {out/'best.pt'}")


if __name__ == "__main__":
    main()
