"""Is the late-training drop on unseen families a calibration shift or a ranking loss?

For each leave-family-out checkpoint the probability map of every unseen volume is scored
four ways: at the fixed threshold, at the per-volume oracle threshold (best Dice any
threshold could reach: separates ranking quality from calibration), at the threshold
whose foreground equals the ground truth's (calibration oracle), and at the threshold
whose foreground equals the earliest checkpoint's (label-free matching). The foreground
ratio prediction / ground truth shows whether training drifts towards under- or
over-segmentation of structure types it never saw.

    python tools/calib_lfo.py --checkpoint runs/lfo_e/ema_e0008.pt runs/lfo_e/final_ema.pt --out runs/calib_lfo.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "selma"))
sys.path.insert(0, str(ROOT / "submission"))
from data import PREP, load_index, load_raw  # noqa: E402
from train import cldice, dice, panoptic_quality, remove_small_components  # noqa: E402
import inference as C  # noqa: E402

FIXED = 0.5
SWEEP = np.round(np.arange(0.05, 0.96, 0.05), 2)


def load(path, dev):
    st = torch.load(path, map_location="cpu", weights_only=False)
    net = C.build_model()
    net.load_state_dict(st["model"])
    net = net.to(dev).eval()
    net.selma_norm = st.get("norm", "percentile")
    net.selma_patch = tuple(int(v) for v in st.get("patch", C.PATCH))
    return net


def threshold_for(prob, fraction):
    """Threshold whose foreground fraction equals `fraction`."""
    if fraction <= 0:
        return 1.0
    return float(np.quantile(prob[::2, ::2, ::2], 1.0 - fraction))


def measure(prob, gt, cont, t):
    pred = remove_small_components(prob > t, 1 if cont else 8)
    m = {"t": float(t), "dice": dice(pred, gt), "fg_ratio": float(pred.sum() / max(gt.sum(), 1))}
    m["cldice" if cont else "pq"] = cldice(pred, gt) if cont else panoptic_quality(pred, gt)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    idx = load_index()
    cfos = [r for r in idx if r["family"] == "cfos_neuron"]
    nerves = sorted((r for r in idx if r["family"] == "peripheral_nerve"), key=lambda r: r["fg_frac"])[::3][:6]
    nets = [load(p, dev) for p in a.checkpoint]
    rows = []
    for rec in cfos + nerves:
        t0 = time.time()
        img = np.asarray(load_raw(rec["stem"], mmap=False))
        gt = np.load(PREP / f"{rec['stem']}__lab.npy").astype(bool)
        pad = C.zero_plane_mask(img)
        cont = rec["family"] == "peripheral_nerve"
        row = {"stem": rec["stem"], "family": rec["family"], "gt_frac": float(gt.mean()), "ckpt": []}
        ref_frac = None
        for net in nets:
            with torch.no_grad():
                prob = C.predict_probability(img, [net], dev)
            if pad is not None:
                prob[pad] = 0.0
            dices = [dice(prob > t, gt) for t in SWEEP]
            r = {"fixed": measure(prob, gt, cont, FIXED),
                 "oracle": measure(prob, gt, cont, SWEEP[int(np.argmax(dices))]),
                 "gt_matched": measure(prob, gt, cont, threshold_for(prob, gt.mean()))}
            frac = float((prob > FIXED).mean())
            if ref_frac is None:
                ref_frac = frac
            r["ref_matched"] = measure(prob, gt, cont, threshold_for(prob, ref_frac))
            r["mean_prob_fg"] = float(prob[gt].mean()) if gt.any() else 0.0
            r["mean_prob_bg"] = float(prob[~gt].mean())
            row["ckpt"].append(r)
            del prob
            torch.cuda.empty_cache()
        rows.append(row)
        print(f"{rec['stem'][:32]:32s} " + " | ".join(
            f"D {c['fixed']['dice']:.3f} orc {c['oracle']['dice']:.3f} gtm {c['gt_matched']['dice']:.3f} "
            f"refm {c['ref_matched']['dice']:.3f} fg x{c['fixed']['fg_ratio']:.2f}" for c in row["ckpt"])
            + f"  ({time.time() - t0:.0f}s)", flush=True)
        Path(a.out).write_text(json.dumps({"checkpoints": a.checkpoint, "rows": rows}, indent=1))

    second = {"cfos_neuron": "pq", "peripheral_nerve": "cldice"}
    print("\nfamily-mean per checkpoint (dice / second metric / median foreground ratio vs GT)")
    for fam, key in second.items():
        fr = [r for r in rows if r["family"] == fam]
        for i, p in enumerate(a.checkpoint):
            parts = []
            for mode in ("fixed", "oracle", "gt_matched", "ref_matched"):
                d = np.mean([r["ckpt"][i][mode]["dice"] for r in fr])
                s = np.mean([r["ckpt"][i][mode][key] for r in fr])
                parts.append(f"{mode} {d:.3f}/{s:.3f}")
            ratio = np.median([r["ckpt"][i]["fixed"]["fg_ratio"] for r in fr])
            sep = np.mean([r["ckpt"][i]["mean_prob_fg"] - r["ckpt"][i]["mean_prob_bg"] for r in fr])
            print(f"  {fam:16s} {Path(p).stem:12s} " + "  ".join(parts) + f"  fg x{ratio:.2f}  sep {sep:.3f}")


if __name__ == "__main__":
    main()
