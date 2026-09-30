"""Holdout validation shaped like the final post-processing pipeline.

The submission applies one shared (threshold, size filter) to every volume except those
the density probe flags as sparse-isolated, which get their own profile. Monitoring
mirrors that: plaques pick their best configuration, everything else shares one. The old
validate() in train.py instead applied a hard-coded plaque profile tuned for the
pos_weight=5 model, which wipes out plaques once the probabilities are calibrated and
would make a better model look worse.

Both profiles are selected on the same volumes they are scored on, so the absolute
number is optimistic; it is for comparing checkpoints under an identical rule.
"""
from collections import defaultdict

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from data import CATEGORIES, PREP, load_raw, volume_stats, zscore
from train import cldice, dice, panoptic_quality, remove_small_components

SHARED_THRESHOLDS = (0.15, 0.25, 0.35, 0.5)
SHARED_MIN_SIZES = (1, 16)          # applied to dense isolated; contiguous keeps min 1
SPARSE_THRESHOLDS = (0.2, 0.35, 0.5, 0.65, 0.8)
SPARSE_MIN_SIZES = (8, 32, 64)


def reflect_pad(x, pad):
    """F.pad(..., mode="reflect") for right-side pads of any size.

    torch reflects at most (axis length - 1) voxels per call, so thin volumes padded to a
    larger patch (a 50-slice slab to 112) are mirrored repeatedly, as np.pad does in
    training. ``pad`` is torch order: (left, right) pairs from the last axis backwards.
    """
    rights = list(pad[1::2])                     # last axis first
    for j, need in enumerate(rights):
        axis = x.ndim - 1 - j
        while need > 0:
            k = min(need, x.shape[axis] - 1)
            if k <= 0:
                step = [0] * (2 * (x.ndim - 2))
                step[2 * j + 1] = need
                return torch.nn.functional.pad(x, step, mode="replicate")
            step = [0] * (2 * (x.ndim - 2))
            step[2 * j + 1] = k
            x = torch.nn.functional.pad(x, step, mode="reflect")
            need -= k
    return x


def _metrics(pred, gt, contiguous):
    m = {"dice": dice(pred, gt)}
    if contiguous:
        m["cldice"] = cldice(pred, gt)
    else:
        m["panoptic_quality"] = panoptic_quality(pred, gt)
    return m


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


@torch.no_grad()
def _image(stem, norm):
    if norm == "zscore":
        st = volume_stats()[stem]
        return zscore(load_raw(stem, mmap=False), st["mean"], st["std"])
    return np.load(PREP / f"{stem}__img.npy").astype(np.float32)


def monitor_validate(model, records, patch, device, overlap=0.25, norm="percentile"):
    was_training = model.training
    model.eval()
    grids = defaultdict(dict)   # stem -> {(t, m): metrics}
    cats = {}
    for rec in records:
        img = _image(rec["stem"], norm)
        gt = np.load(PREP / f"{rec['stem']}__lab.npy").astype(bool)
        x = torch.from_numpy(img[None, None])
        shape = x.shape[2:]
        pad = []
        for i in range(2, -1, -1):
            pad += [0, max(0, patch[i] - shape[i])]
        if any(pad):
            # Same convention as training and the container: reflect thin volumes.
            x = reflect_pad(x, pad)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = sliding_window_inference(
                x.to(device), patch, 1, model,
                overlap=overlap, mode="gaussian", sw_device=device,
                device=torch.device("cpu"))
        prob = torch.sigmoid(logits.float())[0, 0, :shape[0], :shape[1], :shape[2]].numpy()
        cat = rec["category"]
        cats[rec["stem"]] = cat
        if cat == "sparse_isolated":
            grid = [(t, m) for t in SPARSE_THRESHOLDS for m in SPARSE_MIN_SIZES]
        elif cat == "dense_isolated":
            grid = [(t, m) for t in SHARED_THRESHOLDS for m in SHARED_MIN_SIZES]
        else:
            grid = [(t, 1) for t in SHARED_THRESHOLDS]
        for t, m in grid:
            pred = remove_small_components(prob > t, m)
            grids[rec["stem"]][(t, m)] = _metrics(pred, gt, "contiguous" in cat)

    sparse = [s for s, c in cats.items() if c == "sparse_isolated"]
    best_sparse, sparse_cfg = None, None
    if sparse:
        def sparse_score(cfg):
            rows = [grids[s][cfg] for s in sparse]
            return (_mean(rows, "dice") + _mean(rows, "panoptic_quality")) / 2
        sparse_cfg = max(((t, m) for t in SPARSE_THRESHOLDS for m in SPARSE_MIN_SIZES),
                         key=sparse_score)
        best_sparse = sparse_cfg

    candidates = []
    for t in SHARED_THRESHOLDS:
        for m in SHARED_MIN_SIZES:
            per_cat = defaultdict(list)
            for stem, cat in cats.items():
                if cat == "sparse_isolated":
                    cfg = best_sparse
                elif cat == "dense_isolated":
                    cfg = (t, m)
                else:
                    cfg = (t, 1)
                per_cat[cat].append(grids[stem][cfg])
            iso = per_cat["dense_isolated"] + per_cat["sparse_isolated"]
            con = per_cat["dense_contiguous"] + per_cat["sparse_contiguous"]
            # Leave-family-out holdouts lack whole categories; looking them up above
            # created empty entries.
            out = {c: {k: _mean(v, k) for k in v[0]} for c, v in per_cat.items() if v}
            out["isolated"] = {"dice": _mean(iso, "dice"),
                               "panoptic_quality": _mean(iso, "panoptic_quality")}
            out["contiguous"] = {"dice": _mean(con, "dice"), "cldice": _mean(con, "cldice")}
            out["challenge_score"] = float(np.mean([
                out["isolated"]["dice"], out["isolated"]["panoptic_quality"],
                out["contiguous"]["dice"], out["contiguous"]["cldice"]]))
            out["threshold"], out["minimum_component_size"] = t, m
            candidates.append(out)
    best = max(candidates, key=lambda o: o["challenge_score"])
    if sparse_cfg is not None:
        best["sparse_threshold"], best["sparse_minimum_component_size"] = sparse_cfg
    if was_training:
        model.train()
    return best
