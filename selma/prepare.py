"""Preprocess the annotated set into memmap-friendly .npy + an index with sampling metadata.

Raw volumes are uint16 with per-volume intensity scales differing by ~80x, so each is
percentile-normalised here once rather than on every epoch. Foreground coordinates are
subsampled up front because plaque volumes are ~0.01% foreground - finding positives at
train time by argwhere would dominate the step budget.
"""
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path("data/SELMA3D2026_training_annotated")
OUT = Path("data/prep")

# family -> (challenge scoring category, source folder)
FAMILY = {
    "cfos_neuron":      ("dense_isolated",     "isolated_structures"),
    "cell_nuclei":      ("dense_isolated",     "isolated_structures"),
    "AD_plaques":       ("sparse_isolated",    "isolated_structures"),
    "blood_vessel":     ("dense_contiguous",   "contiguous_structures"),
    "peripheral_nerve": ("sparse_contiguous",  "contiguous_structures"),
    "axon":             ("sparse_contiguous",  "contiguous_structures"),
}
MAX_FG_COORDS = 20000


def family_of(name):
    for k in FAMILY:
        if k != "axon" and name.startswith(k):
            return k
    return "axon"


def normalise(a):
    """Percentile clip to [0,1]. Robust to the 80x scale spread across markers."""
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    if hi <= lo:
        lo, hi = float(a.min()), float(max(a.max(), a.min() + 1))
    return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1).astype(np.float16)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for cat in ["isolated_structures", "contiguous_structures"]:
        for p in sorted((ROOT / cat / "raw").glob("*.mha")):
            gtp = ROOT / cat / "gt" / p.name.replace("_raw", "_gt")
            if not gtp.exists():
                gtp = ROOT / cat / "gt" / p.name
            fam = family_of(p.name)
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
            lab = (sitk.GetArrayFromImage(sitk.ReadImage(str(gtp))) > 0).astype(np.uint8)
            assert img.shape == lab.shape, (p.name, img.shape, lab.shape)

            stem = p.stem
            np.save(OUT / f"{stem}__img.npy", normalise(img))
            np.save(OUT / f"{stem}__lab.npy", lab)

            fg = np.argwhere(lab)
            if len(fg) > MAX_FG_COORDS:
                fg = fg[np.random.default_rng(0).choice(len(fg), MAX_FG_COORDS, replace=False)]
            np.save(OUT / f"{stem}__fg.npy", fg.astype(np.int16))

            index.append(dict(stem=stem, family=fam, category=FAMILY[fam][0],
                              shape=list(img.shape), n_fg=int(lab.sum()),
                              fg_frac=float(lab.mean())))
            print(f"  {stem:44s} {str(img.shape):20s} {fam:18s} fg={lab.mean()*100:6.3f}%")

    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(f"\n{len(index)} volumes -> {OUT}")
    for c in sorted(set(r["category"] for r in index)):
        sub = [r for r in index if r["category"] == c]
        empty = sum(1 for r in sub if r["n_fg"] == 0)
        print(f"  {c:20s} {len(sub):3d} volumes" + (f"  ({empty} with EMPTY gt)" if empty else ""))


if __name__ == "__main__":
    main()
