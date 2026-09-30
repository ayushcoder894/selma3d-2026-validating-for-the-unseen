"""Tune the contiguous route on the v11 holdout with models that never saw it.

For every holdout volume: the isolated group's probability (e.g. v15a) decides the route
exactly as the container does; for the contiguous volumes, Dice and clDice are scored for
each (blend, threshold), where blend 1.0 = contiguous group only (candidate A) and 0.5 =
average of the two groups (candidate B). "anch" rows threshold each volume where the routed
map segments as much as the isolated group does at 0.45 (route.json "anchor": true).

    python tools/tune_route.py --iso ckpt/v15a.pt --cont runs/B128_hold/final_ema.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "selma"))
sys.path.insert(0, str(ROOT / "submission"))
from data import PREP, load_index, load_raw  # noqa: E402
from train import cldice, dice, remove_small_components  # noqa: E402
import inference as C  # noqa: E402

THRESHOLDS = [0.25, 0.35, 0.45, 0.55, 0.65]
BLENDS = [1.0, 0.75, 0.5]


def load(paths, dev):
    models = []
    for ck in paths:
        s = torch.load(ck, map_location="cpu", weights_only=False)
        n = C.build_model()
        n.load_state_dict(s["model"])
        n = n.to(dev).eval()
        n.selma_norm = s.get("norm", "percentile")
        n.selma_patch = tuple(int(v) for v in s.get("patch", C.PATCH))
        models.append(n)
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", nargs="+", required=True)
    ap.add_argument("--cont", nargs="+", required=True)
    ap.add_argument("--holdout", default=str(ROOT / "selma" / "holdout_v11.json"))
    a = ap.parse_args()
    dev = torch.device("cuda")
    iso, cont = load(a.iso, dev), load(a.cont, dev)
    hold = set(json.loads(Path(a.holdout).read_text())["stems"])
    table, routing = {}, []
    for rec in [r for r in load_index() if r["stem"] in hold]:
        img = np.asarray(load_raw(rec["stem"], mmap=False))
        gt = np.load(PREP / f"{rec['stem']}__lab.npy").astype(bool)
        pad = C.zero_plane_mask(img)
        pi = C.predict_probability(img, iso, dev)
        if pad is not None:
            pi[pad] = 0.0
        routed = C.looks_contiguous(pi > 0.45)
        truth = "contiguous" in rec["category"]
        routing.append((rec["stem"], truth, routed))
        if not truth:
            continue
        pc = C.predict_probability(img, cont, dev)
        if pad is not None:
            pc[pad] = 0.0
        inside = slice(None) if pad is None else ~pad
        fraction = float((pi[inside] > 0.45).mean())
        for b in BLENDS:
            p = b * pc + (1 - b) * pi
            for t in THRESHOLDS:
                m = remove_small_components(p > t, 4).astype(bool)
                table.setdefault((b, t), []).append((dice(m, gt), cldice(m, gt)))
            t = float(np.clip(np.quantile(p[inside].ravel()[::4], 1.0 - fraction), 0.05, 0.95)) if fraction > 0 else 0.45
            m = remove_small_components(p > t, 4).astype(bool)
            table.setdefault((b, -1.0), []).append((dice(m, gt), cldice(m, gt)))
        base = remove_small_components(pi > 0.45, 4).astype(bool)
        table.setdefault(("iso-only", 0.45), []).append((dice(base, gt), cldice(base, gt)))
        print(f"  {rec['stem'][:40]:40s} done", flush=True)
    print("\nrouting on the holdout (truth -> routed):")
    for s, t, r in routing:
        print(f"   {'OK   ' if t == r else 'WRONG'} {s[:40]:40s} {'cont' if t else 'iso '} -> {'cont' if r else 'iso '}")
    print("\ncontiguous holdout volumes: mean Dice / clDice / mean by (blend, threshold)")
    rows = []
    for k, v in table.items():
        v = np.array(v)
        rows.append((k, v[:, 0].mean(), v[:, 1].mean(), v.mean()))
    for k, d, c, m in sorted(rows, key=lambda r: -r[3]):
        thr = "anch" if k[1] < 0 else f"{k[1]:.2f}"
        print(f"   blend {str(k[0]):8s} thr {thr:4s}   Dice {d:.3f}  clDice {c:.3f}  mean {m:.4f}")


if __name__ == "__main__":
    main()
