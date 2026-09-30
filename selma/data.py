"""Patch sampling for SELMA3D.

Two things drive the design:

* The final rank is the mean rank across the four structure categories, so each category
  is sampled with equal probability regardless of how many volumes it contains.
* Foreground fraction ranges from 0.01% (plaques) to 26% (axons). Uniform random crops
  would be nearly pure background for the sparse markers, so crop centres are drawn from
  precomputed foreground coordinates most of the time.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

PREP = Path(__file__).resolve().parent.parent / "data" / "prep"
RAW = Path(__file__).resolve().parent.parent / "data" / "prep_raw"

# z-score clipping range, chosen from our own data: labelled foreground has its 99th
# percentile at z = 7.8-20.7 across families (nerves highest), and only 0.01-1.3% of
# foreground exceeds 20, so [-3, 25] keeps >99% of every family's foreground unsaturated.
# The percentile clip used before saturated 44% of nerve foreground.
Z_LO, Z_HI = -3.0, 25.0
_STATS = None


def volume_stats():
    global _STATS
    if _STATS is None:
        _STATS = json.loads((RAW / "stats.json").read_text())
    return _STATS


def load_raw(stem, mmap=True):
    return np.load(RAW / f"{stem}__raw.npy", mmap_mode="r" if mmap else None)


def zscore(a, mean, std):
    """Standardise with the volume's non-zero statistics and clip to [Z_LO, Z_HI]."""
    return np.clip((np.asarray(a, np.float32) - np.float32(mean)) / np.float32(std),
                   Z_LO, Z_HI).astype(np.float32)
CATEGORIES = ["dense_isolated", "sparse_isolated", "dense_contiguous", "sparse_contiguous"]


def load_index():
    return json.loads((PREP / "index.json").read_text())


def folds(index, n_folds=5, seed=0):
    """Stratify by family so every fold sees every marker."""
    rng = np.random.default_rng(seed)
    assign = {}
    for fam in sorted(set(r["family"] for r in index)):
        stems = sorted(r["stem"] for r in index if r["family"] == fam)
        rng.shuffle(stems)
        for i, s in enumerate(stems):
            assign[s] = i % n_folds
    return assign


class Patches(Dataset):
    def __init__(self, index, patch=(96, 96, 96), steps=250, p_fg=0.7, train=True, seed=0,
                 norm="percentile", uniform=False):
        self.index = index
        self.patch = np.array(patch)
        self.steps = steps
        self.p_fg = p_fg
        self.train = train
        self.seed = seed
        self._worker_rng = None
        self.by_cat = {c: [r for r in index if r["category"] == c] for c in CATEGORIES}
        self.by_cat = {c: v for c, v in self.by_cat.items() if v}
        self.cats = sorted(self.by_cat)
        self._cache = {}
        # norm="zscore": crops come from the raw volumes, standardised per volume.
        # uniform=True samples volumes uniformly instead of balancing the 4 categories.
        self.norm = norm
        self.uniform = uniform

    def __len__(self):
        return self.steps

    def _arrays(self, stem):
        if stem not in self._cache:
            self._cache[stem] = (
                load_raw(stem) if self.norm == "zscore"
                else np.load(PREP / f"{stem}__img.npy", mmap_mode="r"),
                np.load(PREP / f"{stem}__lab.npy", mmap_mode="r"),
                np.load(PREP / f"{stem}__fg.npy"),
            )
        return self._cache[stem]

    def _crop(self, rng, img, lab, fg):
        shape = np.array(img.shape)
        p = np.minimum(self.patch, shape)
        if len(fg) and rng.random() < self.p_fg:
            centre = fg[rng.integers(len(fg))].astype(np.int64)
            centre = centre + rng.integers(-p // 4, p // 4 + 1)
        else:
            centre = rng.integers(0, shape)
        start = np.clip(centre - p // 2, 0, np.maximum(shape - p, 0))
        sl = tuple(slice(int(s), int(s + q)) for s, q in zip(start, p))
        return np.asarray(img[sl], np.float32), np.asarray(lab[sl], np.float32)

    def _normalise_crop(self, stem, img):
        if self.norm != "zscore":
            return img
        st = volume_stats()[stem]
        return zscore(img, st["mean"], st["std"])

    def _pad(self, img, lab):
        # Image and label must be padded the same way. All 16 vessel volumes are 50 slices
        # deep, so every vessel patch is ~48% padding; reflecting the image while
        # zero-filling the label taught the network that mirrored vessels are background.
        # Inference reflect-pads thin volumes too (submission/inference.py, monitor.py).
        pad = [(0, max(0, q - s)) for s, q in zip(img.shape, self.patch)]
        if any(b for _, b in pad):
            mode = "reflect" if min(img.shape) > 1 else "constant"
            img = np.pad(img, pad, mode=mode)
            lab = np.pad(lab, pad, mode=mode)
        return img, lab

    def _augment(self, rng, img, lab):
        for ax in range(3):
            if rng.random() < 0.5:
                img, lab = np.flip(img, ax), np.flip(lab, ax)
        if rng.random() < 0.5:                       # in-plane 90 deg rotation
            k = int(rng.integers(1, 4))
            img, lab = np.rot90(img, k, (1, 2)), np.rot90(lab, k, (1, 2))
        img, lab = np.ascontiguousarray(img), np.ascontiguousarray(lab)

        # Intensity augmentation is what carries the model across unseen markers.
        if rng.random() < 0.3:
            img = gaussian_filter(img, sigma=rng.uniform(0.4, 1.2))
        if rng.random() < 0.5:
            img = np.power(np.clip(img, 0, 1), rng.uniform(0.6, 1.7))
        if rng.random() < 0.5:
            img = img * rng.uniform(0.75, 1.3) + rng.uniform(-0.12, 0.12)
        if rng.random() < 0.3:
            # NumPy's normal() defaults to float64, doubling the temporary allocation.
            # Keep augmentation in float32; this matters when several DataLoader workers
            # prefetch 96^3 patches concurrently on a memory-constrained workstation.
            noise = rng.normal(0, rng.uniform(0.01, 0.06), img.shape).astype(np.float32)
            img = img + noise
        if rng.random() < 0.15:                      # simulate light-sheet stripe artefact
            axis = int(rng.integers(1, 3))
            n = img.shape[axis]
            stripe = 1 + rng.normal(0, 0.08, n).astype(np.float32)
            img = img * stripe.reshape([-1 if i == axis else 1 for i in range(3)])
        return np.clip(img, -0.5, 1.5).astype(np.float32), lab

    def _augment_z(self, rng, img, lab):
        """Augmentation for z-scored crops: the [0, 1] assumptions above (gamma on a
        clipped image, tiny absolute noise, a final [-0.5, 1.5] clip) would destroy
        standardised intensities, so gamma works on a per-crop min-max rescale."""
        for ax in range(3):
            if rng.random() < 0.5:
                img, lab = np.flip(img, ax), np.flip(lab, ax)
        if rng.random() < 0.5:
            k = int(rng.integers(1, 4))
            img, lab = np.rot90(img, k, (1, 2)), np.rot90(lab, k, (1, 2))
        img, lab = np.ascontiguousarray(img), np.ascontiguousarray(lab)
        if rng.random() < 0.2:
            img = gaussian_filter(img, sigma=rng.uniform(0.5, 1.0))
        if rng.random() < 0.15:
            img = img * rng.uniform(0.75, 1.25)                      # brightness
        if rng.random() < 0.15:                                      # contrast
            m = img.mean()
            img = (img - m) * rng.uniform(0.75, 1.25) + m
        if rng.random() < 0.3:                                       # gamma
            lo, hi = float(img.min()), float(img.max())
            if hi > lo:
                u = (img - lo) / (hi - lo)
                g = rng.uniform(0.7, 1.5)
                if rng.random() < 0.15:                              # inverted gamma
                    u = 1 - np.power(1 - u, g)
                else:
                    u = np.power(u, g)
                img = u * (hi - lo) + lo
        if rng.random() < 0.15:
            img = img + rng.normal(0, rng.uniform(0.0, 0.3), img.shape).astype(np.float32)
        return np.clip(img, Z_LO, Z_HI).astype(np.float32), lab

    def __getitem__(self, i):
        wi = torch.utils.data.get_worker_info()
        if self.train:
            if self._worker_rng is None:
                self._worker_rng = np.random.default_rng(
                    (self.seed, wi.id if wi else 0, torch.initial_seed() % 2**31)
                )
            rng = self._worker_rng
        else:
            rng = np.random.default_rng((self.seed, i))
        if self.uniform:
            rec = self.index[rng.integers(len(self.index))]
            cat = rec["category"]
        else:
            cat = self.cats[rng.integers(len(self.cats))]
            rec = self.by_cat[cat][rng.integers(len(self.by_cat[cat]))]
        img, lab, fg = self._arrays(rec["stem"])
        img, lab = self._crop(rng, img, lab, fg)
        img = self._normalise_crop(rec["stem"], img)
        if self.train:
            img, lab = (self._augment_z(rng, img, lab) if self.norm == "zscore"
                        else self._augment(rng, img, lab))
        img, lab = self._pad(img, lab)
        return {
            "image": torch.from_numpy(img[None].copy()),
            "label": torch.from_numpy(lab[None].copy()),
            "tubular": torch.tensor(float("contiguous" in cat)),
            "category": cat,
        }
