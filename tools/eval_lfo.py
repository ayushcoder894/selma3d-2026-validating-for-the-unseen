"""Score a leave-family-out model on the families it never trained on.

This is the proxy for the challenge test set, which is mostly unseen structure types:
our usual holdout contains only trained families and badly overestimated v11 (holdout
0.6386, preliminary test 0.537). c-Fos neurons stand in for unseen cell types (microglia),
peripheral nerves for unseen fibre networks (gut nerve).

Runs through submission/inference.py (flip TTA, overlap, optional scale TTA, per-window
normalisation when the checkpoint asks for it), probe disabled. Sweeps the threshold and
a blend with image intensity: p = (1 - w) * network + w * smoothed normalised intensity.
On held-out nerves a plain intensity threshold beat the network 0.45 vs 0.20 Dice; on
held-out c-Fos it lost 0.13 vs 0.32. Only a weight that helps both is worth shipping.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "selma"))
sys.path.insert(0, str(ROOT / "submission"))
from data import PREP, load_index, load_raw  # noqa: E402
from train import cldice, dice, panoptic_quality, remove_small_components  # noqa: E402
import inference as C  # noqa: E402

THRESHOLDS = [0.3, 0.45, 0.6, 0.75]
BLENDS = [0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-size", type=int, default=8)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0],
                    help="scale test-time augmentation factors")
    ap.add_argument("--blends", type=float, nargs="+", default=BLENDS)
    a = ap.parse_args()

    dev = torch.device("cuda")
    C.TTA_SCALES = list(a.scales)
    models = []
    for ck in a.checkpoint:
        state = torch.load(ck, map_location="cpu", weights_only=False)
        net = C.build_model()
        net.load_state_dict(state["model"])
        net = net.to(dev).eval()
        if state.get("per_window_norm"):
            net = C.PerWindowNorm(net).eval()
        net.selma_norm = state.get("norm", "percentile")
        net.selma_patch = tuple(int(v) for v in state.get("patch", C.PATCH))
        models.append(net)
        print(f"{Path(ck).name}: per_window_norm={bool(state.get('per_window_norm'))}")

    idx = load_index()
    cfos = [r for r in idx if r["family"] == "cfos_neuron"]
    nerves = sorted((r for r in idx if r["family"] == "peripheral_nerve"),
                    key=lambda r: r["fg_frac"])[::3][:6]
    rows = []
    for rec in cfos + nerves:
        img = np.asarray(load_raw(rec["stem"], mmap=False))
        gt = np.load(PREP / f"{rec['stem']}__lab.npy").astype(bool)
        prob = C.predict_probability(img, models, dev)
        pad = C.zero_plane_mask(img)
        if pad is not None:
            prob[pad] = 0.0
        cont = rec["family"] == "peripheral_nerve"
        intensity = gaussian_filter(C.normalise(img), 1.0) if any(a.blends) else None
        res = {}
        for w in a.blends:
            pb = (1 - w) * prob + w * intensity if w else prob
            for t in THRESHOLDS:
                pred = remove_small_components(pb > t, 1 if cont else a.min_size)
                m = {"dice": dice(pred, gt)}
                if cont:
                    m["cldice"] = cldice(pred, gt)
                else:
                    m["pq"] = panoptic_quality(pred, gt)
                res[(w, t)] = m
        rows.append({"family": rec["family"], "res": res})
        best_t = max(THRESHOLDS, key=lambda t: sum(res[(0.0, t)].values()))
        print(f"  {rec['stem'][:34]:34s} network-only best@{best_t}: " +
              " ".join(f"{k}={v:.3f}" for k, v in res[(0.0, best_t)].items()), flush=True)

    print("\nunseen-family score by blend weight w (best threshold per w):")
    summary = {}
    for w in a.blends:
        for t in THRESHOLDS:
            c = [r["res"][(w, t)] for r in rows if r["family"] == "cfos_neuron"]
            n = [r["res"][(w, t)] for r in rows if r["family"] == "peripheral_nerve"]
            s = dict(cfos_dice=np.mean([m["dice"] for m in c]),
                     cfos_pq=np.mean([m["pq"] for m in c]),
                     nerve_dice=np.mean([m["dice"] for m in n]),
                     nerve_cldice=np.mean([m["cldice"] for m in n]))
            s["score"] = float(np.mean(list(s.values())))
            summary[f"w{w}_t{t}"] = {k: float(v) for k, v in s.items()}
        bt = max(THRESHOLDS, key=lambda t: summary[f"w{w}_t{t}"]["score"])
        b = summary[f"w{w}_t{bt}"]
        print(f"  w={w:<4} t={bt}: score {b['score']:.4f}  cfos D {b['cfos_dice']:.3f} "
              f"PQ {b['cfos_pq']:.3f}  nerve D {b['nerve_dice']:.3f} clD {b['nerve_cldice']:.3f}")
    best = max(summary, key=lambda k: summary[k]["score"])
    print(f"\nBEST unseen-family score {summary[best]['score']:.4f} at {best}")
    Path(a.out).write_text(json.dumps({"checkpoint": a.checkpoint, "scales": a.scales,
                                       "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
