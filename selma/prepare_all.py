"""Rebuild every training artefact from the original annotated .mha files.

Used on a fresh machine (e.g. the A100 server), where only the 2.5 GB of .mha files are
uploaded. Produces exactly what selma/prepare.py and the raw staging step produce locally:

  data/prep/<stem>__img.npy   percentile-normalised image (float16, legacy recipe)
  data/prep/<stem>__lab.npy   binary label (uint8)
  data/prep/<stem>__fg.npy    subsampled foreground coordinates (int16)
  data/prep/index.json        per-volume metadata
  data/prep_raw/<stem>__raw.npy  raw uint16 image (z-score recipe)
  data/prep_raw/stats.json    non-zero mean / std per volume

    python selma/prepare_all.py --src data/SELMA3D2026_training_annotated
"""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent

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


def percentile_normalise(a):
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    if hi <= lo:
        lo, hi = float(a.min()), float(max(a.max(), a.min() + 1))
    return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "SELMA3D2026_training_annotated"))
    a = ap.parse_args()
    src = Path(a.src)
    prep, raw_dir = ROOT / "data" / "prep", ROOT / "data" / "prep_raw"
    prep.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    index, stats = [], {}
    for cat in ["isolated_structures", "contiguous_structures"]:
        for p in sorted((src / cat / "raw").glob("*.mha")):
            gtp = src / cat / "gt" / p.name.replace("_raw", "_gt")
            if not gtp.exists():
                gtp = src / cat / "gt" / p.name
            fam = family_of(p.name)
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
            lab = (sitk.GetArrayFromImage(sitk.ReadImage(str(gtp))) > 0).astype(np.uint8)
            assert img.shape == lab.shape, (p.name, img.shape, lab.shape)
            stem = p.stem
            np.save(prep / f"{stem}__img.npy", percentile_normalise(img))
            np.save(prep / f"{stem}__lab.npy", lab)
            fg = np.argwhere(lab)
            if len(fg) > MAX_FG_COORDS:
                fg = fg[np.random.default_rng(0).choice(len(fg), MAX_FG_COORDS, replace=False)]
            np.save(prep / f"{stem}__fg.npy", fg.astype(np.int16))
            np.save(raw_dir / f"{stem}__raw.npy", img.astype(np.uint16))
            nz = img[img > 0].astype(np.float64)
            stats[stem] = {"mean": float(nz.mean()), "std": float(max(nz.std(), 1e-3)),
                           "p005": float(np.percentile(img, 0.5)),
                           "p995": float(np.percentile(img, 99.5))}
            index.append(dict(stem=stem, family=fam, category=FAMILY[fam][0],
                              shape=list(img.shape), n_fg=int(lab.sum()),
                              fg_frac=float(lab.mean())))
            print(f"  {stem:44s} {str(img.shape):20s} {fam}", flush=True)
    (prep / "index.json").write_text(json.dumps(index, indent=1))
    (raw_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"\n{len(index)} volumes prepared")


if __name__ == "__main__":
    main()
