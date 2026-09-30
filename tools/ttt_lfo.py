"""Test-time self-training on unseen structure families (leave-family-out proxy).

A model trained without c-Fos and peripheral nerves segments those volumes, then adapts
to each one on its own confident predictions: voxels above HI become foreground targets,
voxels below LO background, everything in between is ignored. A copy of the network is
fine-tuned for a few steps on crops of that single volume and predicts again. Both
predictions are scored against the real labels, so the gain (or damage) of per-volume
self-training on genuinely unseen structure types is measured directly.

    python tools/ttt_lfo.py --checkpoint runs/lfo_e/final_ema.pt --out runs/ttt_lfo_e.json

--holdout selma/holdout_v11.json instead scores the contiguous volumes of that holdout (for a
model trained without them, e.g. the 128^3 B128_hold expert).
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "selma"))
sys.path.insert(0, str(ROOT / "submission"))
from data import PREP, load_index, load_raw  # noqa: E402
from train import cldice, dice, panoptic_quality, remove_small_components  # noqa: E402
import inference as C  # noqa: E402

THRESHOLDS = [0.3, 0.45, 0.6, 0.75]


def crops(z, prob, hi, lo, n, patch, rng):
    """n (image, target, weight) crops; half centred on confident foreground."""
    pos = np.argwhere(prob > hi)
    shape = np.array(z.shape)
    out = []
    for i in range(n):
        if len(pos) and i % 2 == 0:
            c = pos[rng.integers(len(pos))] + rng.integers(-patch // 4, patch // 4 + 1, 3)
        else:
            c = rng.integers(0, shape)
        s = np.clip(c - patch // 2, 0, np.maximum(shape - patch, 0))
        sl = tuple(slice(int(a), int(a) + patch) for a in s)
        x, p = z[sl], prob[sl]
        pad = [(0, max(0, patch - d)) for d in x.shape]
        if any(b for _, b in pad):
            x = np.pad(x, pad, mode="reflect")
            p = np.pad(p, pad, mode="constant", constant_values=0.5)
        t = (p > hi).astype(np.float32)
        w = ((p > hi) | (p < lo)).astype(np.float32)
        for ax in range(3):
            if rng.random() < 0.5:
                x, t, w = np.flip(x, ax), np.flip(t, ax), np.flip(w, ax)
        out.append((np.ascontiguousarray(x), np.ascontiguousarray(t), np.ascontiguousarray(w)))
    return out


def adapt(net, img, prob, dev, steps, lr, hi, lo, patch=96, batch=2, seed=0):
    rng = np.random.default_rng(seed)
    model = copy.deepcopy(net).eval()          # instance norm: eval == train behaviour
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    z = C.zscore_normalise(img).astype(np.float32)
    for step in range(steps):
        cs = crops(z, prob, hi, lo, batch, patch, rng)
        x = torch.from_numpy(np.stack([c[0] for c in cs])[:, None]).to(dev)
        t = torch.from_numpy(np.stack([c[1] for c in cs])[:, None]).to(dev)
        w = torch.from_numpy(np.stack([c[2] for c in cs])[:, None]).to(dev)
        with torch.autocast("cuda"):
            logit = model(x)
        logit = logit.float()
        if logit.ndim == 6:
            logit = logit[:, 0]
        bce = (F.binary_cross_entropy_with_logits(logit, t, reduction="none") * w).sum() / w.sum().clamp_min(1)
        pr = torch.sigmoid(logit) * w
        dl = 1 - (2 * (pr * t).sum() + 1) / (pr.sum() + (t * w).sum() + 1)
        loss = bce + dl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    model.selma_norm = net.selma_norm
    model.selma_patch = net.selma_patch
    return model


def score(prob, gt, cont):
    res = {}
    for t in THRESHOLDS:
        pred = remove_small_components(prob > t, 1 if cont else 8)
        m = {"dice": dice(pred, gt)}
        m["cldice" if cont else "pq"] = cldice(pred, gt) if cont else panoptic_quality(pred, gt)
        res[t] = m
    return res


def summarise(rows, key):
    out = {}
    for t in THRESHOLDS:
        c = [r[key][t] for r in rows if r["family"] == "cfos_neuron"]
        n = [r[key][t] for r in rows if r["family"] == "peripheral_nerve"]
        s = dict(cfos_dice=np.mean([m["dice"] for m in c]), cfos_pq=np.mean([m["pq"] for m in c]),
                 nerve_dice=np.mean([m["dice"] for m in n]), nerve_cldice=np.mean([m["cldice"] for m in n]))
        s["score"] = float(np.mean(list(s.values())))
        out[t] = {k: float(v) for k, v in s.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hi", type=float, default=0.85)
    ap.add_argument("--lo", type=float, default=0.10)
    ap.add_argument("--holdout", default=None, help="score this holdout's contiguous volumes instead")
    a = ap.parse_args()
    dev = torch.device("cuda")
    st = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    net = C.build_model()
    net.load_state_dict(st["model"])
    net = net.to(dev).eval()
    net.selma_norm = st.get("norm", "percentile")
    net.selma_patch = tuple(int(v) for v in st.get("patch", C.PATCH))
    idx = load_index()
    if a.holdout:
        hold = set(json.loads(Path(a.holdout).read_text())["stems"])
        volumes = [r for r in idx if r["stem"] in hold and r["category"].endswith("contiguous")]
    else:
        cfos = [r for r in idx if r["family"] == "cfos_neuron"]
        nerves = sorted((r for r in idx if r["family"] == "peripheral_nerve"), key=lambda r: r["fg_frac"])[::3][:6]
        volumes = cfos + nerves
    rows = []
    for rec in volumes:
        t0 = time.time()
        img = np.asarray(load_raw(rec["stem"], mmap=False))
        gt = np.load(PREP / f"{rec['stem']}__lab.npy").astype(bool)
        pad = C.zero_plane_mask(img)
        p0 = C.predict_probability(img, [net], dev)
        if pad is not None:
            p0[pad] = 0.0
        model = adapt(net, img, p0, dev, a.steps, a.lr, a.hi, a.lo, patch=net.selma_patch[0])
        with torch.no_grad():
            p1 = C.predict_probability(img, [model], dev)
        if pad is not None:
            p1[pad] = 0.0
        del model
        torch.cuda.empty_cache()
        cont = rec["category"].endswith("contiguous")
        r = {"family": rec["family"], "base": score(p0, gt, cont), "ttt": score(p1, gt, cont)}
        rows.append(r)
        b, tt = r["base"][0.6], r["ttt"][0.6]
        print(f"  {rec['stem'][:32]:32s} base@0.6 " + " ".join(f"{k}={v:.3f}" for k, v in b.items())
              + "  ->  ttt " + " ".join(f"{k}={v:.3f}" for k, v in tt.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    if a.holdout:
        for key in ("base", "ttt"):
            print(f"{key:5s} " + "  ".join(f"t{t}: D {np.mean([r[key][t]['dice'] for r in rows]):.3f} "
                                           f"clD {np.mean([r[key][t]['cldice'] for r in rows]):.3f}" for t in THRESHOLDS))
        return
    base, ttt = summarise(rows, "base"), summarise(rows, "ttt")
    for name, s in (("baseline", base), ("self-trained", ttt)):
        bt = max(s, key=lambda t: s[t]["score"])
        v = s[bt]
        print(f"{name:13s} best t={bt}: score {v['score']:.4f}  cfos D {v['cfos_dice']:.3f} PQ {v['cfos_pq']:.3f}  "
              f"nerve D {v['nerve_dice']:.3f} clD {v['nerve_cldice']:.3f}   | per-threshold "
              + " ".join(f"{t}:{s[t]['score']:.4f}" for t in THRESHOLDS))
    Path(a.out).write_text(json.dumps({"args": vars(a), "baseline": {str(k): v for k, v in base.items()},
                                       "ttt": {str(k): v for k, v in ttt.items()}}, indent=1))


if __name__ == "__main__":
    main()
