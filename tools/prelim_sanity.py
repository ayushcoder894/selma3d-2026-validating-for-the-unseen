"""Run the packaged container pipeline over every preliminary-test volume.

There are no labels, so this cannot score anything. It catches the failures that would
wreck a test-set average: an empty mask on an unseen structure type (DSC 0), runaway
foreground, the plaque probe firing where it should not, zero-padding leaking into the
output, or a crash. Uses submission/inference.py and submission/resources exactly as the
image will.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submission"))
import inference as C  # noqa: E402

TEST = Path(__file__).resolve().parent.parent / "data" / "prelim_test"


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "prelim_sanity.json"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C.RESOURCE_PATH = ROOT / "submission" / "resources"
    models, thr, min_size, sp_thr, sp_min = C.load_models(dev)
    rows = []
    for sub in ["isolated_structures", "contiguous_structures"]:
        for p in sorted((TEST / sub).glob("*.mha")):
            raw = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
            t = time.time()
            mask = C.predict(raw, models, dev, thr, min_size, sp_thr, sp_min)
            dt = time.time() - t
            pad = C.zero_plane_mask(raw)
            leak = int(mask[pad].sum()) if pad is not None else 0
            _, n = ndimage.label(mask > 0, np.ones((3, 3, 3), bool))
            row = dict(volume=p.stem, kind=sub.split("_")[0], fg_pct=float((mask > 0).mean() * 100),
                       components=int(n), seconds=round(dt, 1), padding_leak=leak,
                       dtype=str(mask.dtype), values=[int(v) for v in np.unique(mask)[:4]])
            rows.append(row)
            print(f"{p.stem[:36]:36s} {row['kind'][:4]} fg {row['fg_pct']:7.3f}%  comps "
                  f"{n:5d}  {dt:5.1f}s  leak {leak}", flush=True)
    out.write_text(json.dumps(rows, indent=1))
    empty = [r["volume"] for r in rows if r["fg_pct"] == 0]
    leaks = [r["volume"] for r in rows if r["padding_leak"]]
    print(f"\n{len(rows)} volumes, {sum(r['seconds'] for r in rows):.0f}s total; "
          f"EMPTY masks: {empty or 'none'}; padding leaks: {leaks or 'none'}")


if __name__ == "__main__":
    main()
