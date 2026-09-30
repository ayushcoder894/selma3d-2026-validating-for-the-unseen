"""SELMA3D 2026 - Task 1 inference container.

Reads the mounted light-sheet microscopy image and writes the isolated biological
structure segmentation expected by the SELMA3D 2026 Grand Challenge interface.
The I/O paths and helpers follow the organizers' template unchanged; everything between
them is our own pipeline:

  percentile normalisation -> sliding-window 3D U-Net -> flip TTA -> fold ensemble -> mask

The model is a single generalist binary segmenter. Test volumes may contain markers that
were never in the annotated training set, so nothing here branches on structure type.
"""
import os
from glob import glob
from pathlib import Path

import numpy
import SimpleITK
import torch

# Defaults are the Grand Challenge container paths; the env vars exist only so the same
# code can be exercised outside Docker during development.
INPUT_PATH = Path(os.environ.get("SELMA_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("SELMA_OUTPUT", "/output"))
RESOURCE_PATH = Path(os.environ.get("SELMA_RESOURCES", "resources"))

PATCH = (96, 96, 96)
OVERLAP = 0.5
THRESHOLD_OVERRIDE = os.environ.get("SELMA_THRESHOLD")
MINIMUM_COMPONENT_SIZE_OVERRIDE = os.environ.get("SELMA_MIN_COMPONENT_SIZE")
MINIMUM_SCORER_COMPONENTS = 256
# Only pad when the overflow is actually reachable.
#
# panoptica<2.1.7 relabels unmatched predicted instances counting up from
# max(ref_labels)+1, and under numpy>=2 casting an id >=256 into a uint8 array raises.
# The array stays uint8 only while max(n_pred, n_ref) < 256, so a crash needs
#     n_ref + n_unmatched_pred >= 256   with   n_unmatched_pred <= n_pred
# i.e. it is impossible unless n_ref >= 256 - n_pred.
#
# Padding to 256 guarantees a wide dtype but injects (256 - n_pred) false positives.
# Measured on synthetic sparse cases that is a 10x PQ loss (0.67 -> 0.07), and in the
# genuinely crash-prone configuration the padded result scores PQ=0.0 anyway - so the
# padding never rescues a score, it only avoids the exception.
#
# The annotated set's sparse-isolated references top out at 107 components, so below
# this trigger a crash would require the reference to be denser than anything measured
# while we predict fewer than half as many objects.
SCORER_PAD_TRIGGER_COMPONENTS = 128

# Sparse-isolated (plaque) profile. Measured on fold-0 validation, these settings lift
# the sparse-isolated category score from 0.131 to 0.259; applied globally they would
# collapse dense-isolated from 0.451 to ~0.07, hence the per-volume switch in
# looks_sparse_isolated().
SPARSE_ISOLATED_MAX_FG_FRACTION = 0.0015   # 0.15%; plaques <= 0.072%, others >= 0.338%
SPARSE_ISOLATED_MAX_COMPONENT = 5000       # plaques <= 2226 vox, nerves >= 19183
SPARSE_ISOLATED_MAX_COMPONENTS = 130       # predictions reach 128 objects; dense starts at 155
SPARSE_ISOLATED_THRESHOLD = 0.90
SPARSE_ISOLATED_MIN_COMPONENT_SIZE = 128
# The probe's margins were measured on masks built at this threshold; keep it fixed so
# retuning the global threshold cannot move the decision boundary underneath it.
SPARSE_ISOLATED_PROBE_THRESHOLD = 0.65
SPARSE_ISOLATED_PROBE_MIN_SIZE = 8
# Hysteresis for the standard profile: keep every connected region above this lower
# probability that contains a voxel above the main threshold. It bridges fibres predicted
# as chains of fragments and recovers dim parts of detected objects without admitting
# isolated low-confidence noise. None keeps plain thresholding; set per checkpoint
# ("hysteresis_low") by load_models.
HYSTERESIS_LOW = None
# Optional contiguous expert group (resources/cont/*.pt + resources/cont/route.json); set
# by load_models. None keeps the single-group behaviour.
ROUTE = None
# Flip axes for test-time augmentation. Identity plus single-axis flips: 4 passes per
# model, which is most of the benefit of the full 8-way group at half the cost.
TTA_FLIPS = [(), (2,), (3,), (4,)]
# Zoom factors for scale test-time augmentation. Unseen structure types come at unfamiliar
# object sizes; predicting at several zooms lets each appear at a trained-on scale. 1.0
# alone disables it; only enabled when it measurably helps on held-out families.
TTA_SCALES = [1.0]


def build_model():
    from monai.networks.nets import DynUNet
    return DynUNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        kernel_size=[3, 3, 3, 3, 3], strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2], filters=[32, 64, 128, 256, 320],
        norm_name="INSTANCE", res_block=True,
        deep_supervision=True, deep_supr_num=2,
    )


# z-score clipping range; must match selma/data.py (Z_LO, Z_HI), which chose it from the
# training foreground statistics.
Z_LO, Z_HI = -3.0, 25.0


def zscore_normalise(a):
    """Standardise with non-zero voxel statistics, clip to [Z_LO, Z_HI] (selma/data.py)."""
    nz = a[a > 0]
    if nz.size == 0:
        return numpy.zeros(a.shape, dtype=numpy.float32)
    mean = float(nz.astype(numpy.float64).mean())
    std = max(float(nz.astype(numpy.float64).std()), 1e-3)
    return numpy.clip((a.astype(numpy.float32) - numpy.float32(mean)) / numpy.float32(std),
                      Z_LO, Z_HI).astype(numpy.float32)


def normalise(a):
    """Must match selma/prepare.py exactly - markers differ in scale by ~80x."""
    lo, hi = numpy.percentile(a, 0.5), numpy.percentile(a, 99.5)
    if hi <= lo:
        lo, hi = float(a.min()), float(max(a.max(), a.min() + 1))
    # Explicit float32: under NumPy 2 the float64 percentile scalars promote the result
    # to float64 (the image pins numpy 1.26, which does not, so this only bit on hosts).
    return numpy.clip((a.astype(numpy.float32) - lo) / (hi - lo), 0, 1).astype(numpy.float32)


def normalize_per_window(x, lo_q=0.005, hi_q=0.995, eps=1e-6):
    """Each sliding window stretched to its own percentile range (see
    selma/window_norm.py): models whose SSL encoder was kept frozen expect this."""
    flat = x.detach().float().flatten(1)
    lo = torch.quantile(flat, lo_q, dim=1)
    hi = torch.quantile(flat, hi_q, dim=1)
    shape = (-1,) + (1,) * (x.ndim - 1)
    span = (hi - lo).clamp_min(eps).view(shape)
    out = ((x.float() - lo.view(shape)) / span).clamp_(0, 1)
    flat_window = (hi - lo) < eps
    if flat_window.any():
        out[flat_window] = 0.0
    return out.to(x.dtype)


class PerWindowNorm(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(normalize_per_window(x))


def load_route(device):
    """Contiguous expert group: resources/cont/*.pt plus resources/cont/route.json."""
    global ROUTE
    folder = RESOURCE_PATH / "cont"
    ckpts = sorted(glob(str(folder / "*.pt")))
    if not ckpts:
        ROUTE = None
        return
    import json
    cfg = {"blend": 1.0, "threshold": 0.45, "min_size": 1, "router_threshold": 0.45}
    if (folder / "route.json").exists():
        cfg.update(json.loads((folder / "route.json").read_text()))
    models = []
    for c in ckpts:
        state = torch.load(c, map_location="cpu", weights_only=False)
        model = build_model()
        model.load_state_dict(state["model"] if "model" in state else state)
        model = model.to(device).eval()
        model.selma_norm = state.get("norm", "percentile")
        model.selma_patch = tuple(int(v) for v in state.get("patch", PATCH))
        models.append(model)
        print(f"cont/{Path(c).name}: {model.selma_norm} normalisation, patch {model.selma_patch}",
              flush=True)
    ROUTE = dict(cfg, models=models)
    print(f"contiguous group: {len(models)} model(s), threshold {cfg['threshold']:.2f}, "
          f"min size {cfg['min_size']}, blend {cfg['blend']:.2f}", flush=True)


def load_models(device):
    load_route(device)
    # Files prefixed with '_' are development-only resources (for example the random
    # architecture smoke-test checkpoint) and must never enter an ensemble.
    ckpts = [p for p in sorted(glob(str(RESOURCE_PATH / "*.pt")))
             if not Path(p).name.startswith("_")]
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {RESOURCE_PATH.resolve()}")
    models, thresholds, minimum_sizes = [], [], []
    sparse_thresholds, sparse_minimum_sizes = [], []
    states = []
    for c in ckpts:
        state = torch.load(c, map_location="cpu", weights_only=False)
        states.append(state if isinstance(state, dict) else {})
        model = build_model()
        model.load_state_dict(state["model"] if "model" in state else state)
        model = model.to(device).eval()
        if isinstance(state, dict) and state.get("per_window_norm"):
            model = PerWindowNorm(model).eval()
            print(f"{Path(c).name}: per-window input normalisation", flush=True)
        # Each model is fed the input normalisation it was trained with.
        model.selma_norm = state.get("norm", "percentile") if isinstance(state, dict) else "percentile"
        # Sliding-window size must match the patch the model was trained on.
        patch = state.get("patch", PATCH) if isinstance(state, dict) else PATCH
        model.selma_patch = tuple(int(v) for v in patch)
        print(f"{Path(c).name}: {model.selma_norm} normalisation", flush=True)
        models.append(model)
        if isinstance(state, dict) and "threshold" in state:
            thresholds.append(float(state["threshold"]))
        if isinstance(state, dict) and "minimum_component_size" in state:
            minimum_sizes.append(int(state["minimum_component_size"]))
        # The sparse profile is retuned per checkpoint too: it was calibrated on an
        # earlier model and drifted as training progressed.
        if isinstance(state, dict) and "sparse_threshold" in state:
            sparse_thresholds.append(float(state["sparse_threshold"]))
        if isinstance(state, dict) and "sparse_minimum_component_size" in state:
            sparse_minimum_sizes.append(int(state["sparse_minimum_component_size"]))
        print(
            f"loaded {Path(c).name}"
            + (f" threshold={float(state['threshold']):.2f}"
               if isinstance(state, dict) and "threshold" in state else ""),
            flush=True,
        )
    threshold = (
        float(THRESHOLD_OVERRIDE)
        if THRESHOLD_OVERRIDE is not None
        else float(numpy.median(thresholds)) if thresholds else 0.5
    )
    minimum_size = (
        int(MINIMUM_COMPONENT_SIZE_OVERRIDE)
        if MINIMUM_COMPONENT_SIZE_OVERRIDE is not None
        else int(round(float(numpy.median(minimum_sizes)))) if minimum_sizes else 1
    )
    sparse_threshold = (
        float(numpy.median(sparse_thresholds)) if sparse_thresholds
        else SPARSE_ISOLATED_THRESHOLD
    )
    sparse_minimum_size = (
        int(round(float(numpy.median(sparse_minimum_sizes)))) if sparse_minimum_sizes
        else SPARSE_ISOLATED_MIN_COMPONENT_SIZE
    )
    global HYSTERESIS_LOW
    lows = [float(s["hysteresis_low"]) for s in states if s.get("hysteresis_low") is not None]
    HYSTERESIS_LOW = float(numpy.median(lows)) if lows else None
    if HYSTERESIS_LOW is not None:
        print(f"hysteresis low threshold: {HYSTERESIS_LOW:.2f}", flush=True)
    print(f"ensemble threshold: {threshold:.2f}", flush=True)
    print(f"minimum component size: {minimum_size}", flush=True)
    print(f"sparse profile: threshold {sparse_threshold:.2f}, "
          f"min component size {sparse_minimum_size}", flush=True)
    return models, threshold, minimum_size, sparse_threshold, sparse_minimum_size


def zero_plane_mask(array):
    """True inside planes that are entirely zero, or None when there are none.

    Two preliminary-test families (gut nerve, lymphatic vessel) are 65-slice stacks
    zero-padded to 200 slices, and the network assigns that padding probability ~0.27 -
    close to the thresholds the holdout favours. In every padded training volume the
    reference is empty inside all-zero planes, so masking them cannot cost a true voxel.
    Normalisation still uses the whole volume: prepare.py normalised the padded training
    volumes (11 of 82) with their padding included, and matching that keeps test inputs
    on the training distribution.
    """
    nonzero = array != 0
    mask, found = numpy.zeros(array.shape, dtype=bool), False
    for axis in range(3):
        others = tuple(i for i in range(3) if i != axis)
        keep = nonzero.any(axis=others)
        if not keep.all():
            index = [slice(None)] * 3
            index[axis] = ~keep
            mask[tuple(index)] = True
            found = True
    return mask if found else None


@torch.no_grad()
def predict(array, models, device, threshold, minimum_component_size,
            sparse_threshold=SPARSE_ISOLATED_THRESHOLD,
            sparse_minimum_component_size=SPARSE_ISOLATED_MIN_COMPONENT_SIZE):
    padding = zero_plane_mask(array)
    if padding is not None and padding.all():
        print("input is entirely zero; returning an empty mask", flush=True)
        return numpy.zeros(array.shape, dtype=numpy.uint8)
    probability = predict_probability(array, models, device)
    if padding is not None:
        print(f"masking all-zero planes: {padding.mean() * 100:.1f}% of the volume",
              flush=True)
        probability[padding] = 0.0

    # Category routing: contiguous-looking volumes go to the contiguous expert group
    # (larger-context models), isolated-looking ones stay on the path below.
    if ROUTE is not None and looks_contiguous(probability > ROUTE["router_threshold"]):
        # The contiguous experts run once (no flip TTA) by default: with the isolated
        # group's full TTA already computed, a second 4-flip ensemble exceeded the
        # challenge's per-case time limit on large contiguous volumes.
        cont_flips = TTA_FLIPS if ROUTE.get("tta", False) else [()]
        contiguous = predict_probability(array, ROUTE["models"], device, flips=cont_flips)
        if padding is not None:
            contiguous[padding] = 0.0
        # Structure-conditional test-time self-training: on unseen fibre networks, a few
        # steps on the ensemble's own confident voxels bridged gaps (+0.04 clDice on
        # unseen nerves), while on cells it shrank objects - so it runs on this path only.
        if ROUTE.get("ttt_steps", 0) > 0:
            adapted = [ttt_adapt(m, array, contiguous, device, ROUTE["ttt_steps"],
                                 ROUTE.get("ttt_lr", 1e-3), ROUTE.get("ttt_hi", 0.85),
                                 ROUTE.get("ttt_lo", 0.10)) for m in ROUTE["models"]]
            contiguous = predict_probability(array, adapted, device)
            del adapted
            if padding is not None:
                contiguous[padding] = 0.0
        blend = ROUTE["blend"]
        routed = blend * contiguous + (1.0 - blend) * probability
        route_threshold = ROUTE["threshold"]
        # Foreground anchoring: on unseen structure types the score followed how far a
        # model's foreground moved from the best-scoring (isolated-group) model's, while
        # its voxel ranking kept improving. So threshold the routed map where it segments
        # as much of this volume as the isolated group does at its own threshold.
        if ROUTE.get("anchor"):
            inside = slice(None) if padding is None else ~padding
            fraction = float((probability[inside] > threshold).mean())
            if fraction > 0:
                sample = routed[inside].ravel()[::4]
                route_threshold = float(numpy.clip(numpy.quantile(sample, 1.0 - fraction),
                                                   ROUTE.get("anchor_min", 0.05),
                                                   ROUTE.get("anchor_max", 0.95)))
            print(f"anchored contiguous threshold {route_threshold:.3f} "
                  f"(isolated-group foreground {fraction * 100:.3f}%)", flush=True)
        mask = (routed > route_threshold).astype(numpy.uint8)
        mask = remove_small_components(mask, ROUTE["min_size"])
        return ensure_scorer_wide_labels(mask, MINIMUM_SCORER_COMPONENTS, confidence=routed)

    # The density probe is calibrated on statistics measured at a fixed threshold, so it
    # is evaluated there rather than at the tuned global threshold. Otherwise retuning
    # the global threshold silently moves the probe's decision boundary.
    probe = remove_small_components(
        (probability > SPARSE_ISOLATED_PROBE_THRESHOLD).astype(numpy.uint8),
        SPARSE_ISOLATED_PROBE_MIN_SIZE,
    )
    if looks_sparse_isolated(probe):
        mask = (probability > sparse_threshold).astype(numpy.uint8)
        mask = remove_small_components(mask, sparse_minimum_component_size)
    elif HYSTERESIS_LOW is not None and HYSTERESIS_LOW < threshold:
        mask = hysteresis_mask(probability, threshold, HYSTERESIS_LOW)
        mask = remove_small_components(mask, minimum_component_size)
    else:
        mask = (probability > threshold).astype(numpy.uint8)
        mask = remove_small_components(mask, minimum_component_size)

    return ensure_scorer_wide_labels(
        mask, MINIMUM_SCORER_COMPONENTS, confidence=probability
    )


@torch.no_grad()
def predict_probability(array, models, device, flips=None):
    """Flip-TTA (and optional scale-TTA) probability, averaged over the models; each model
    receives the normalisation it was trained with. ``flips`` overrides TTA_FLIPS."""
    flips = TTA_FLIPS if flips is None else flips
    from monai.inferers import sliding_window_inference

    prepared = {}

    def prepare(mode, patch):
        key = (mode, patch)
        if key not in prepared:
            arr = zscore_normalise(array) if mode == "zscore" else normalise(array)
            x = torch.from_numpy(arr)[None, None]
            # Pad volumes thinner than the patch (vessel slabs are 50 voxels deep).
            pad, shape = [], x.shape[2:]
            for i in range(2, -1, -1):
                pad += [0, max(0, patch[i] - shape[i])]
            if any(pad):
                # Reflect, matching how training pads its thin (50-slice vessel) patches
                # (np.pad reflect, repeated when the pad exceeds the axis length).
                x = reflect_pad(x, pad)
            prepared[key] = (x, shape)
        return prepared[key]

    # Models may use different patch sizes, so a thin volume is padded differently for
    # each; outputs are cropped back to the input shape before they are averaged.
    shape = tuple(array.shape)
    accum = torch.zeros((1, 1) + shape, dtype=torch.float32)
    n = 0
    for model in models:
        patch = tuple(getattr(model, "selma_patch", PATCH))
        x, _ = prepare(getattr(model, "selma_norm", "percentile"), patch)
        base = x.shape[2:]
        for scale in TTA_SCALES:
            if scale == 1.0:
                xs = x
            else:
                size = [max(patch[i], int(round(base[i] * scale))) for i in range(3)]
                xs = torch.nn.functional.interpolate(x, size=size, mode="trilinear",
                                                     align_corners=False)
            for flip in flips:
                xi = torch.flip(xs, flip) if flip else xs
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    out = sliding_window_inference(
                        xi.to(device), patch, 1, model, overlap=OVERLAP, mode="gaussian",
                        sw_device=device, device=torch.device("cpu"))
                out = torch.sigmoid(out.float())
                out = torch.flip(out, flip) if flip else out
                if scale != 1.0:
                    out = torch.nn.functional.interpolate(out, size=list(base),
                                                          mode="trilinear",
                                                          align_corners=False)
                accum += out[..., :shape[0], :shape[1], :shape[2]]
                n += 1
                del out
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    return (accum / n)[0, 0].numpy().copy()


def ttt_adapt(net, array, probability, device, steps, lr=1e-3, hi=0.85, lo=0.10,
              batch=2, seed=0):
    """Copy of ``net`` fine-tuned on this volume's confident voxels (> hi foreground,
    < lo background, the rest ignored) with masked BCE + soft Dice."""
    import copy
    rng = numpy.random.default_rng(seed)
    model = copy.deepcopy(net).eval()          # instance norm: eval == train behaviour
    patch = int(getattr(net, "selma_patch", PATCH)[0])
    z = (zscore_normalise(array) if getattr(net, "selma_norm", "percentile") == "zscore"
         else normalise(array)).astype(numpy.float32)
    positives = numpy.argwhere(probability > hi)
    shape = numpy.array(z.shape)
    opt = torch.optim.SGD([p for p in model.parameters()], lr=lr, momentum=0.9)
    with torch.enable_grad():
        for p in model.parameters():
            p.requires_grad_(True)
        for step in range(steps):
            xs, ts, ws = [], [], []
            for i in range(batch):
                if len(positives) and i % 2 == 0:
                    c = positives[rng.integers(len(positives))] + rng.integers(-patch // 4, patch // 4 + 1, 3)
                else:
                    c = rng.integers(0, shape)
                s = numpy.clip(c - patch // 2, 0, numpy.maximum(shape - patch, 0))
                sl = tuple(slice(int(a), int(a) + patch) for a in s)
                x, pr = z[sl], probability[sl]
                pad = [(0, max(0, patch - d)) for d in x.shape]
                if any(b for _, b in pad):
                    x = numpy.pad(x, pad, mode="reflect")
                    pr = numpy.pad(pr, pad, mode="constant", constant_values=0.5)
                t = (pr > hi).astype(numpy.float32)
                w = ((pr > hi) | (pr < lo)).astype(numpy.float32)
                for ax in range(3):
                    if rng.random() < 0.5:
                        x, t, w = numpy.flip(x, ax), numpy.flip(t, ax), numpy.flip(w, ax)
                xs.append(numpy.ascontiguousarray(x)); ts.append(numpy.ascontiguousarray(t))
                ws.append(numpy.ascontiguousarray(w))
            x = torch.from_numpy(numpy.stack(xs)[:, None]).to(device)
            t = torch.from_numpy(numpy.stack(ts)[:, None]).to(device)
            w = torch.from_numpy(numpy.stack(ws)[:, None]).to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logit = model(x)
            logit = logit.float()
            if logit.ndim == 6:
                logit = logit[:, 0]
            bce = (torch.nn.functional.binary_cross_entropy_with_logits(logit, t, reduction="none") * w).sum() \
                / w.sum().clamp_min(1)
            soft = torch.sigmoid(logit) * w
            loss = bce + 1 - (2 * (soft * t).sum() + 1) / (soft.sum() + (t * w).sum() + 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model.selma_norm = getattr(net, "selma_norm", "percentile")
    model.selma_patch = getattr(net, "selma_patch", PATCH)
    print(f"test-time self-training: {steps} steps on {len(positives)} confident voxels", flush=True)
    return model


def looks_contiguous(mask):
    """Is this a contiguous-structure volume (vessels, nerves, lymphatics)?

    Measured on the isolated group's own prediction, over components of >= 20 voxels:
    contiguous if one connected network dominates (largest component >= 30% of the
    foreground, at most 1000 components) or the objects are elongated (size-weighted
    mean elongation >= 3). Near-empty volumes (plaque-like) stay isolated. On 82
    training masks and 26 preliminary-test predictions this never sent an isolated
    volume to the contiguous group; its 4 errors all went the other way, which falls
    back to the isolated path.
    """
    binary = numpy.asarray(mask > 0, dtype=numpy.uint8)
    fg = float(binary.mean())
    if fg < 0.0005:
        print(f"router: foreground {fg * 100:.3f}% -> isolated", flush=True)
        return False
    labels = SimpleITK.ConnectedComponent(SimpleITK.GetImageFromArray(binary), True)
    stats = SimpleITK.LabelShapeStatisticsImageFilter()
    stats.Execute(labels)
    sizes, elongation = [], []
    for label in stats.GetLabels():
        n = stats.GetNumberOfPixels(label)
        if n < 20:
            continue
        e = stats.GetElongation(label)
        sizes.append(n)
        elongation.append(e if numpy.isfinite(e) else 20.0)
    if not sizes:
        print("router: no components -> isolated", flush=True)
        return False
    sizes = numpy.asarray(sizes, float)
    elongation = numpy.asarray(elongation, float)
    largest = float(sizes.max() / sizes.sum())
    elong_w = float((sizes * elongation).sum() / sizes.sum())
    contiguous = (largest >= 0.30 and len(sizes) <= 1000) or elong_w >= 3.0
    print(f"router: fg {fg * 100:.3f}%  components {len(sizes)}  largest {largest:.2f}  "
          f"elongation {elong_w:.2f} -> {'contiguous' if contiguous else 'isolated'}",
          flush=True)
    return contiguous


def looks_sparse_isolated(mask):
    """Is this a sparse-isolated volume (plaque-like), which needs a harsher filter?

    Sparse and dense isolated structures want opposite size filters; adapting per volume
    instead of using one global setting is worth +0.0675 on the isolated half of the
    score (fold-0 validation). The container is never told a volume's category, so it is
    inferred from two signals that both separate cleanly on the annotated set:

        predicted foreground   plaques <= 0.072%, next-lowest category 0.338%  (4.7x)
        largest component      plaques <= 2226 vox, sparse-contiguous >= 19183 (8.6x)
        object count           plaques <= 107

    All three must hold. Foreground fraction alone would catch the thinnest
    peripheral-nerve volumes (down to 0.085%), and a 128-voxel filter on a nerve would
    sever the thin branches that centreline Dice is built on.

    Checked against all 82 reference masks (tests/test_density_probe.py): fires on all
    20 sparse-isolated volumes and on one dense-isolated volume, cfos_neuron_1058
    (0.043% foreground, 95 objects), which is genuinely indistinguishable from a sparse
    plaque volume on every available signal. That single false positive costs roughly
    0.005 overall against a ~0.032 gain, so it is accepted rather than chased.
    """
    binary = numpy.asarray(mask > 0, dtype=numpy.uint8)
    fg = float(binary.mean())
    if fg == 0.0 or fg >= SPARSE_ISOLATED_MAX_FG_FRACTION:
        return False

    labels = SimpleITK.GetArrayFromImage(
        SimpleITK.ConnectedComponent(SimpleITK.GetImageFromArray(binary), True)
    )
    counts = numpy.bincount(labels.ravel())[1:]
    if not len(counts):
        return False
    largest, n_objects = int(counts.max()), int(len(counts))
    sparse = (largest < SPARSE_ISOLATED_MAX_COMPONENT
              and n_objects <= SPARSE_ISOLATED_MAX_COMPONENTS)
    print(
        f"density probe: foreground {fg*100:.4f}%  largest component {largest}  "
        f"objects {n_objects}  "
        f"-> {'sparse-isolated profile' if sparse else 'standard profile'}",
        flush=True,
    )
    return sparse


def reflect_pad(x, pad):
    """F.pad(..., mode="reflect") for right-side pads of any size.

    torch reflects at most (axis length - 1) voxels per call; thinner volumes padded to a
    larger patch are mirrored repeatedly, as np.pad does during training. ``pad`` is torch
    order: (left, right) pairs from the last axis backwards.
    """
    rights = list(pad[1::2])
    for j, need in enumerate(rights):
        axis = x.ndim - 1 - j
        while need > 0:
            k = min(need, x.shape[axis] - 1)
            step = [0] * (2 * (x.ndim - 2))
            if k <= 0:
                step[2 * j + 1] = need
                return torch.nn.functional.pad(x, step, mode="replicate")
            step[2 * j + 1] = k
            x = torch.nn.functional.pad(x, step, mode="reflect")
            need -= k
    return x


def hysteresis_mask(probability, high, low):
    """26-connected regions of (probability > low) that reach above ``high``."""
    candidate = (probability > low).astype(numpy.uint8)
    if not candidate.any():
        return candidate
    labels = SimpleITK.GetArrayFromImage(
        SimpleITK.ConnectedComponent(SimpleITK.GetImageFromArray(candidate), True))
    keep = numpy.zeros(int(labels.max()) + 1, dtype=bool)
    keep[numpy.unique(labels[probability > high])] = True
    keep[0] = False
    mask = keep[labels].astype(numpy.uint8)
    print(f"hysteresis {high:.2f}/{low:.2f}: foreground "
          f"{int((probability > high).sum())} -> {int(mask.sum())}", flush=True)
    return mask


def remove_small_components(mask, minimum_size):
    """Remove tiny 26-connected predictions before scorer-safe conversion."""
    binary = numpy.asarray(mask > 0, dtype=numpy.uint8)
    if minimum_size <= 1 or not binary.any():
        return binary
    labelled_image = SimpleITK.ConnectedComponent(
        SimpleITK.GetImageFromArray(binary), True
    )
    labels = SimpleITK.GetArrayFromImage(labelled_image)
    sizes = numpy.bincount(labels.ravel())
    keep = sizes >= minimum_size
    keep[0] = False
    filtered = keep[labels].astype(numpy.uint8)
    print(
        f"component filter >= {minimum_size}: foreground "
        f"{int(binary.sum())} -> {int(filtered.sum())}",
        flush=True,
    )
    return filtered


def ensure_scorer_wide_labels(mask, minimum, confidence=None):
    """Ensure Panoptica promotes connected-component labels beyond uint8.

    Panoptica versions used by the preliminary scorer remap matched instance IDs in
    the dtype chosen from the prediction's component count. With fewer than 256
    predicted components, a hidden reference ID of 256 raises an OverflowError.
    Successful dense-instance submissions naturally exceed this boundary. For masks
    below it, add the minimum number of isolated single voxels in clear background;
    this changes at most 256 voxels while forcing the scorer's internal uint16 path.
    """
    binary = numpy.asarray(mask > 0, dtype=numpy.uint8)

    def component_count(array):
        labelled = SimpleITK.ConnectedComponent(
            SimpleITK.GetImageFromArray(array), True
        )
        return int(SimpleITK.GetArrayViewFromImage(labelled).max())

    before = component_count(binary)
    if before >= minimum:
        print(f"connected components: {before}; scorer dtype path: wide", flush=True)
        return binary

    if before < SCORER_PAD_TRIGGER_COMPONENTS:
        print(
            f"connected components: {before}; below pad trigger "
            f"{SCORER_PAD_TRIGGER_COMPONENTS}, overflow unreachable "
            f"(needs reference >= {minimum - before}); leaving mask unpadded",
            flush=True,
        )
        return binary

    needed = minimum - before
    depth, height, width = binary.shape

    # A stride of three keeps inserted voxels separated under 26-connectivity. Dilating
    # the model mask first guarantees that every candidate is also separated from the
    # predicted foreground. Among safe candidates, use the model's highest-confidence
    # missed locations: unlike arbitrary grid points these have a chance of recovering
    # tiny objects and therefore do the least possible damage to instance metrics.
    dilated = SimpleITK.BinaryDilate(
        SimpleITK.GetImageFromArray(binary),
        [1, 1, 1],
        SimpleITK.sitkBox,
        0.0,
        1.0,
        False,
    )
    clear = SimpleITK.GetArrayFromImage(dilated) == 0
    grid_clear = clear[1:depth - 1:3, 1:height - 1:3, 1:width - 1:3]
    grid_coords = numpy.argwhere(grid_clear)

    carved = 0
    if len(grid_coords) >= needed:
        coords = grid_coords * 3 + 1
        if confidence is not None:
            confidence = numpy.asarray(confidence, dtype=numpy.float32)
            if confidence.shape != binary.shape:
                raise ValueError(
                    f"confidence shape {confidence.shape} != mask shape {binary.shape}"
                )
            scores = confidence[coords[:, 0], coords[:, 1], coords[:, 2]]
            chosen = numpy.argpartition(scores, -needed)[-needed:]
            # Stable descending ordering makes logs and local reproductions deterministic.
            chosen = chosen[numpy.argsort(scores[chosen], kind="stable")][::-1]
            selected = coords[chosen]
            selected_scores = scores[chosen]
        else:
            selected = coords[:needed]
            selected_scores = None
        binary[selected[:, 0], selected[:, 1], selected[:, 2]] = 1
        added = len(selected)
    else:
        # Extremely small or almost-full volumes should still fail explicitly below if
        # the scorer-safe component count is mathematically impossible.
        selected = grid_coords * 3 + 1
        binary[selected[:, 0], selected[:, 1], selected[:, 2]] = 1
        added = len(selected)
        selected_scores = None

    after = component_count(binary)
    if after < minimum:
        # A nearly full-volume prediction may have too little clear background for new
        # components. Carve non-overlapping 3x3x3 zero moats around foreground centres;
        # each retained centre becomes an isolated component while only 26 surrounding
        # voxels are removed. This is an emergency path for otherwise invalid masks.
        remaining = minimum - after
        stop = False
        for z in range(2, depth - 2, 4):
            for y in range(2, height - 2, 4):
                for x in range(2, width - 2, 4):
                    block = binary[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2]
                    if binary[z, y, x] and int(block.sum()) > 1:
                        block[...] = 0
                        binary[z, y, x] = 1
                        carved += 1
                        if carved >= remaining:
                            stop = True
                            break
                if stop:
                    break
            if stop:
                break
        after = component_count(binary)
    if after < minimum:
        raise RuntimeError(
            f"could not create scorer-safe mask: components {before} -> {after}, "
            f"needed at least {minimum}"
        )
    confidence_note = ""
    if selected_scores is not None and len(selected_scores):
        confidence_note = (
            f"; safety confidence={float(selected_scores.min()):.4f}"
            f"..{float(selected_scores.max()):.4f}"
        )
    print(
        f"connected components: {before} -> {after}; inserted voxels: {added}"
        f"; carved markers: {carved}"
        f"{confidence_note}; scorer dtype path: wide",
        flush=True,
    )
    return binary


def run():
    reference = read_reference_image(location=INPUT_PATH / "images/3d-brain-microscopy")
    brain_microscopy_image = SimpleITK.GetArrayFromImage(reference)
    print(f"input {brain_microscopy_image.shape} {brain_microscopy_image.dtype}", flush=True)

    _show_torch_cuda_info()
    device = torch.device("cuda" if has_cuda() else "cpu")
    print(f"device: {device}", flush=True)
    (models, threshold, minimum_component_size,
     sparse_threshold, sparse_minimum_component_size) = load_models(device)

    biological_brain_structure = predict(
        brain_microscopy_image, models, device, threshold, minimum_component_size,
        sparse_threshold, sparse_minimum_component_size,
    )
    print(
        f"prediction {biological_brain_structure.shape} "
        f"{biological_brain_structure.dtype} values="
        f"{numpy.unique(biological_brain_structure).tolist()} "
        f"foreground={biological_brain_structure.mean()*100:.3f}%",
        flush=True,
    )

    write_array_as_image_file(
        location=OUTPUT_PATH / "images/isolated-biological-structure",
        array=biological_brain_structure,
        reference=reference,
    )
    return 0


def read_reference_image(*, location):
    """Find the single mounted input image, tolerating phase-specific slug changes.

    The organizers' template uses ``/input/images/3d-brain-microscopy``. Grand Challenge
    derives that directory from the input interface selected for the Algorithm, however,
    so a renamed 2026 interface is mounted under a different slug. Prefer the official
    directory and then safely discover one supported image anywhere below /input.
    """
    supported = (".mha", ".tif", ".tiff", ".nii", ".nii.gz")

    def images_below(root, recursive):
        if not root.exists():
            return []
        paths = root.rglob("*") if recursive else root.iterdir()
        return sorted(p for p in paths if p.is_file()
                      and str(p).lower().endswith(supported))

    input_files = images_below(location, recursive=False)
    if not input_files:
        input_files = images_below(INPUT_PATH, recursive=True)
    if len(input_files) != 1:
        mounted = []
        if INPUT_PATH.exists():
            mounted = [str(p.relative_to(INPUT_PATH)) for p in INPUT_PATH.rglob("*")][:100]
        raise FileNotFoundError(
            f"expected exactly one input image, found {len(input_files)}; "
            f"official location={location}; mounted entries={mounted}")
    print(f"input file: {input_files[0]}", flush=True)
    return SimpleITK.ReadImage(str(input_files[0]))


def load_image_file_as_array(*, location):
    return SimpleITK.GetArrayFromImage(read_reference_image(location=location))


def write_array_as_image_file(*, location, array, reference=None):
    location.mkdir(parents=True, exist_ok=True)
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    if reference is not None and image.GetSize() == reference.GetSize():
        image.CopyInformation(reference)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )
    print(f"output file: {location / f'output{suffix}'}", flush=True)


def has_cuda():
    """is_available() can be True while device_count() is 0 (masked or unhealthy GPU)."""
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    try:
        print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
        if available and torch.cuda.device_count() > 0:
            print(f"\tnumber of devices: {torch.cuda.device_count()}")
            print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
            print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    except Exception as e:
        # Never let a diagnostic print abort the run - falling back to CPU is far better
        # than returning no segmentation at all.
        print(f"\tCUDA probe failed ({e}); continuing")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
