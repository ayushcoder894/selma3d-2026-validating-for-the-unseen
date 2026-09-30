"""Run a packaged resources folder through the container code on the 26 preliminary images.

Prints each volume's routing decision and foreground, and compares foreground with v15
(the model that scored 0.616) by structure type. A contiguous group that predicts much
less than v15 on gut nerve / lymphatic is the failure that sank v17.

    python tools/prelim_route_check.py <resources_dir> [ref/prelim_sanity_v15.json]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submission"))
import inference as C  # noqa: E402

TEST = ROOT / "data" / "prelim_test"
TYPES = {"gut nerve": "gutnerve", "lymphatic": "lymphatic", "axonal": "axonal",
         "vessel": "blood_vessel", "periph nerve": "peripheral", "microglia": "microglia",
         "protein": "protein", "nuclei": "nulei", "c-Fos": "cfos", "plaques": "plaques"}


def main():
    C.RESOURCE_PATH = Path(sys.argv[1])
    ref = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "ref" / "prelim_sanity_v15.json").read_text())
    ref = {r["volume"]: r["fg_pct"] for r in ref}
    dev = torch.device("cuda")
    models, thr, ms, st, sm = C.load_models(dev)
    decision = {}
    original = C.looks_contiguous

    def spy(mask):
        r = original(mask)
        spy.last = r
        return r
    C.looks_contiguous = spy
    fg, wrong, t0 = {}, 0, time.time()
    for sub, cat in (("isolated_structures", "iso"), ("contiguous_structures", "cont")):
        for p in sorted((TEST / sub).glob("*.mha")):
            spy.last = False
            raw = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
            m = C.predict(raw, models, dev, thr, ms, st, sm)
            fg[p.stem] = float((m > 0).mean() * 100)
            routed = "cont" if spy.last else "iso"
            wrong += routed != cat
            print(f"ROUTE {'OK   ' if routed == cat else 'WRONG'} {cat:4s}->{routed:4s} {p.stem[:34]:34s} "
                  f"fg {fg[p.stem]:.3f}%  (v15 {ref.get(p.stem, float('nan')):.3f}%)", flush=True)
    print(f"\nrouting errors: {wrong}/26    time {time.time() - t0:.0f}s")
    print("foreground relative to v15 by type (v17 was 0.55-0.73x on unseen types):")
    for name, key in TYPES.items():
        vs = [v for v in fg if key in v]
        a = np.mean([fg[v] for v in vs])
        b = np.mean([ref[v] for v in vs])
        print(f"   {name:13s} {a:6.3f}% vs v15 {b:6.3f}%   x{a / b:.2f}")


if __name__ == "__main__":
    main()
