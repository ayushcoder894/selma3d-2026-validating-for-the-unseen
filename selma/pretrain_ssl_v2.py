"""Self-supervised pretraining v2: signal-weighted masked reconstruction on z-scored input.

Why v1 underdelivered: light-sheet crops are ~97% near-constant background, so a masked
intensity reconstruction loss is dominated by background voxels that a constant predictor
already gets right, and little gradient teaches the encoder about the structures being
segmented. v2 keeps masked reconstruction but makes structure dominate the objective:

* inputs are z-scored per volume and clipped to [-3, 25], exactly as the supervised recipe
  (selma/data.py) so the pretrained encoder transfers without an input-distribution shift;
* half of the crops are centred on signal voxels (z > 2) rather than random background;
* the reconstruction loss weights structure voxels (target z > 1.5) ten times more than
  background, and the network regresses z-scores directly (linear output).

Each loaded 300^3 patch yields several crops before the next is decompressed, since
reading a ~30 MB .nii.gz per crop would starve the GPU.

    python selma/pretrain_ssl_v2.py --out runs/ssl_v2 --hours 2.5
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from data import Z_HI, Z_LO
from train import build_model

SIGNAL_CENTRE_Z = 2.0      # crop centres drawn from voxels this bright (in SD units)
SIGNAL_WEIGHT_Z = 1.5      # voxels above this count as structure in the loss
SIGNAL_WEIGHT = 10.0       # structure voxel weight relative to background


def zscore_volume(a):
    nz = a[a > 0]
    if nz.size == 0:
        return np.zeros(a.shape, np.float32)
    mean = float(nz.astype(np.float64).mean())
    std = max(float(nz.astype(np.float64).std()), 1e-3)
    return np.clip((a.astype(np.float32) - np.float32(mean)) / np.float32(std),
                   Z_LO, Z_HI).astype(np.float32)


class SignalCrops(IterableDataset):
    """Endless stream of (corrupted, target, mask) crops; structures sampled uniformly."""

    def __init__(self, root, records, patch=96, crops_per_volume=6, p_signal=0.5,
                 mask_range=(0.5, 0.75), block=12, seed=0):
        self.root = Path(root)
        self.patch = patch
        self.cpv = crops_per_volume
        self.p_signal = p_signal
        self.mask_range = mask_range
        self.block = block
        self.seed = seed
        self.by_structure = defaultdict(list)
        for rec in records:
            self.by_structure[rec["structure"]].append(rec)
        self.structures = sorted(self.by_structure)

    def _crop(self, rng, vol, signal):
        p = self.patch
        shape = np.array(vol.shape)
        if signal is not None and len(signal) and rng.random() < self.p_signal:
            c = signal[rng.integers(len(signal))] + rng.integers(-p // 4, p // 4 + 1, 3)
        else:
            c = rng.integers(0, shape)
        start = np.clip(c - p // 2, 0, np.maximum(shape - p, 0))
        crop = vol[tuple(slice(int(s), int(s) + p) for s in start)]
        pad = [(0, max(0, p - s)) for s in crop.shape]
        if any(b for _, b in pad):
            crop = np.pad(crop, pad, mode="reflect" if min(crop.shape) > 1 else "constant")
        for ax in range(3):
            if rng.random() < 0.5:
                crop = np.flip(crop, ax)
        if rng.random() < 0.5:
            crop = np.rot90(crop, int(rng.integers(1, 4)), axes=(1, 2))
        return np.ascontiguousarray(crop, dtype=np.float32)

    def _mask(self, rng):
        g = math.ceil(self.patch / self.block)
        coarse = rng.random((g, g, g)) < rng.uniform(*self.mask_range)
        m = coarse.repeat(self.block, 0).repeat(self.block, 1).repeat(self.block, 2)
        return m[:self.patch, :self.patch, :self.patch].astype(np.float32)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng((self.seed, wid, int(time.time() * 1000) % 2**31))
        while True:
            structure = self.structures[rng.integers(len(self.structures))]
            rec = self.by_structure[structure][rng.integers(len(self.by_structure[structure]))]
            try:
                vol = zscore_volume(sitk.GetArrayFromImage(
                    sitk.ReadImage(str(self.root / rec["path"]))))
            except Exception:
                continue
            sig = np.argwhere(vol[::2, ::2, ::2] > SIGNAL_CENTRE_Z) * 2
            if len(sig) > 20000:
                sig = sig[rng.choice(len(sig), 20000, replace=False)]
            for _ in range(self.cpv):
                target = self._crop(rng, vol, sig)
                mask = self._mask(rng)
                x = target.copy()
                if rng.random() < 0.5:
                    x[mask > 0] = 0.0                       # background level in z units
                else:
                    noise = rng.normal(float(target.mean()), max(0.1, float(target.std())),
                                       target.shape).astype(np.float32)
                    x[mask > 0] = noise[mask > 0]
                x = x * rng.uniform(0.85, 1.15) + rng.normal(0, rng.uniform(0.0, 0.2), x.shape
                                                             ).astype(np.float32)
                yield {"image": torch.from_numpy(np.clip(x, Z_LO, Z_HI)[None]),
                       "target": torch.from_numpy(target[None]),
                       "mask": torch.from_numpy(mask[None])}


def weighted_loss(pred, target, mask):
    w = 1.0 + (SIGNAL_WEIGHT - 1.0) * (target > SIGNAL_WEIGHT_Z).float()
    err = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0)
    masked = (err * w * mask).sum() / (w * mask).sum().clamp_min(1.0)
    return masked + 0.1 * (err * w).mean() / w.mean()


def deep_loss(outputs, target, mask):
    if outputs.ndim == target.ndim:
        return weighted_loss(outputs, target, mask)
    heads = outputs.unbind(1)
    weights = [1 / 2 ** i for i in range(len(heads))]
    total, loss = sum(weights), 0.0
    for wgt, head in zip(weights, heads):
        t, m = target, mask
        if head.shape[2:] != target.shape[2:]:
            t = F.interpolate(target, head.shape[2:], mode="trilinear", align_corners=False)
            m = F.interpolate(mask, head.shape[2:], mode="nearest")
        loss = loss + wgt / total * weighted_loss(head.float(), t, m)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--manifest", default="manifest_pretrain_200.json")
    ap.add_argument("--out", default="runs/ssl_v2")
    ap.add_argument("--hours", type=float, default=2.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--init", default="", help="optional warm start")
    ap.add_argument("--patch", type=int, default=96, help="cubic crop size")
    ap.add_argument("--crops-per-volume", type=int, default=6)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    records = json.loads((Path(a.data) / a.manifest).read_text())
    records = [r for r in records if (Path(a.data) / r["path"]).exists()]
    print(f"{len(records)} unlabelled patches, "
          f"{len({r['structure'] for r in records})} structure types", flush=True)
    ds = SignalCrops(a.data, records, patch=a.patch, crops_per_volume=a.crops_per_volume)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)
    model = build_model().to(dev)
    if a.init:
        st = torch.load(a.init, map_location="cpu", weights_only=False)
        model.load_state_dict(st.get("model", st))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    start, budget = time.time(), a.hours * 3600
    step, losses, t0 = 0, [], time.time()
    for batch in dl:
        prog = (time.time() - start) / budget
        if prog >= 1:
            break
        lr = a.lr * min(1.0, prog / 0.02) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        for g in opt.param_groups:
            g["lr"] = max(lr, 1e-6)
        x = batch["image"].to(dev, non_blocking=True)
        y = batch["target"].to(dev, non_blocking=True)
        m = batch["mask"].to(dev, non_blocking=True)
        with torch.autocast("cuda"):
            out_ = model(x)
        loss = deep_loss(out_.float(), y, m)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss))
        step += 1
        if step % 250 == 0:
            print(f"step {step:6d}  loss {np.mean(losses):.4f}  lr {lr:.2e}  "
                  f"{(time.time() - t0) / 250:.2f}s/step  progress {prog * 100:5.1f}%", flush=True)
            losses, t0 = [], time.time()
        if step % 2000 == 0:
            torch.save({"model": model.state_dict(), "step": step, "norm": "zscore",
                        "ssl": "signal-weighted masked reconstruction v2"}, out / "latest.pt")
    torch.save({"model": model.state_dict(), "step": step, "norm": "zscore",
                "ssl": "signal-weighted masked reconstruction v2"}, out / "final.pt")
    print(f"=== SSL v2 FINISHED: {step} steps ===", flush=True)


if __name__ == "__main__":
    main()
