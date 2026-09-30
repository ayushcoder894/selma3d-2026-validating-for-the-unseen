"""Mask-shape features that tell isolated from contiguous volumes, measured on segmentations.

Training volumes: ground-truth masks. Preliminary-test volumes: v15's predicted masks
(probability > 0.45, the container's own output), i.e. exactly what a router inside the
container would see. Features per volume (components >= 20 voxels):
  elong_w   size-weighted mean elongation (SimpleITK principal-axis ratio)
  frac_el   fraction of foreground in components with elongation > 3
  largest   fraction of foreground in the largest component

    python tools/router_mask_features.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "selma"))
from data import PREP, load_index  # noqa: E402

PRELIM_PROB = Path(__file__).resolve().parent.parent / "runs" / "prelim_v15"
PRELIM = Path(__file__).resolve().parent.parent / "data" / "prelim_test"


def mask_features(mask):
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    lab = sitk.ConnectedComponent(img, True)
    st = sitk.LabelShapeStatisticsImageFilter()
    st.Execute(lab)
    sizes, elong = [], []
    for l in st.GetLabels():
        n = st.GetNumberOfPixels(l)
        if n < 20:
            continue
        sizes.append(n)
        elong.append(st.GetElongation(l))
    if not sizes:
        return dict(elong_w=0.0, frac_el=0.0, largest=0.0, ncomp=0)
    sizes, elong = np.array(sizes, float), np.array(elong, float)
    tot = sizes.sum()
    return dict(elong_w=float((sizes * elong).sum() / tot),
                frac_el=float(sizes[elong > 3].sum() / tot),
                largest=float(sizes.max() / tot), ncomp=int(len(sizes)))


def main():
    rows = []
    for r in load_index():
        gt = np.load(PREP / f"{r['stem']}__lab.npy") > 0
        rows.append(dict(name=r["stem"], set="train-gt",
                         cat="cont" if "contiguous" in r["category"] else "iso", **mask_features(gt)))
    for sub, cat in (("isolated_structures", "iso"), ("contiguous_structures", "cont")):
        for p in sorted((PRELIM / sub).glob("*.mha")):
            prob = np.load(PRELIM_PROB / f"{p.stem}__prob.npy").astype(np.float32)
            rows.append(dict(name=p.stem, set="prelim-v15", cat=cat, **mask_features(prob > 0.45)))
    (ROOT / "runs" / "router_mask_features.json").write_text(json.dumps(rows, indent=1))
    for key in ("elong_w", "frac_el", "largest"):
        vals = sorted(set(r[key] for r in rows))
        best = None
        for cut in vals:
            err = sum((r[key] > cut) != (r["cat"] == "cont") for r in rows)
            perr = sum((r[key] > cut) != (r["cat"] == "cont") for r in rows if r["set"] == "prelim-v15")
            if best is None or err < best[0]:
                best = (err, cut, perr)
        err, cut, perr = best
        print(f"\n{key}: contiguous if > {cut:.3f}   errors {err}/{len(rows)} (prelim {perr}/26)")
        for r in sorted(rows, key=lambda r: r[key]):
            wrong = (r[key] > cut) != (r["cat"] == "cont")
            if wrong:
                print(f"   WRONG {r['set']:10s} {r['cat']:4s} {r['name'][:36]:36s} {key} {r[key]:.3f}  "
                      f"(elong_w {r['elong_w']:.2f} frac_el {r['frac_el']:.2f} largest {r['largest']:.2f} n {r['ncomp']})")
    print("\nprelim volumes:")
    for r in rows:
        if r["set"] == "prelim-v15":
            print(f"   {r['cat']:4s} {r['name'][:36]:36s} elong_w {r['elong_w']:5.2f}  frac_el {r['frac_el']:.2f}  "
                  f"largest {r['largest']:.2f}  n {r['ncomp']}")


if __name__ == "__main__":
    main()
