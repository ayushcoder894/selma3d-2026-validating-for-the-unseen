"""Copy trained fold checkpoints into submission/resources/ for the container build.

Weights are stored as float16: the container ensembles several folds and the image ships
with every one of them, so halving checkpoint size directly halves the upload.
"""
import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/baseline")
    ap.add_argument("--checkpoint", default=None, nargs="+",
                    help="package these checkpoints as fold0.pt, fold1.pt, ... (an ensemble)"
                         " instead of globbing <runs>/fold*/best.pt")
    ap.add_argument("--resources", default=str(ROOT / "submission" / "resources"),
                    help="destination resource directory for staged or final packaging")
    ap.add_argument("--fp32", action="store_true", help="keep full precision weights")
    ap.add_argument("--clean", action="store_true", help="remove existing checkpoints first")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the packaged inference threshold")
    ap.add_argument("--minimum-component-size", type=int, default=None,
                    help="override the packaged connected-component size filter")
    ap.add_argument("--sparse-threshold", type=float, default=None,
                    help="probability threshold for the sparse-isolated (plaque) profile")
    ap.add_argument("--sparse-minimum-component-size", type=int, default=None,
                    help="size filter for the sparse-isolated (plaque) profile")
    ap.add_argument("--hysteresis-low", type=float, default=None,
                    help="grow standard-profile objects into (prob > low) regions")
    a = ap.parse_args()

    runs = ROOT / a.runs
    resources = Path(a.resources)
    resources.mkdir(parents=True, exist_ok=True)
    # Development smoke tests may leave a random ``_dummy.pt`` in resources.  It must
    # never be ensembled with a trained fold.  ``--clean`` additionally removes stale
    # fold checkpoints when rebuilding an archive from a different run.
    (resources / "_dummy.pt").unlink(missing_ok=True)
    if a.clean:
        for old in resources.glob("*.pt"):
            old.unlink()

    if a.checkpoint:
        checkpoints = [Path(c) for c in a.checkpoint]
    else:
        checkpoints = sorted(runs.glob("fold*/best.pt"))
    if not checkpoints:
        raise SystemExit(f"no trained fold checkpoints found under {runs}")

    total = 0
    for k, ckpt in enumerate(checkpoints):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        weights = state["model"]
        if not a.fp32:
            weights = {k: (v.half() if v.is_floating_point() else v) for k, v in weights.items()}
        dst = resources / (f"fold{k}.pt" if a.checkpoint else f"{ckpt.parent.name}.pt")
        packaged = {"model": weights}
        if a.threshold is not None:
            packaged["threshold"] = float(a.threshold)
        elif "threshold" in state:
            packaged["threshold"] = float(state["threshold"])
        if a.minimum_component_size is not None:
            packaged["minimum_component_size"] = int(a.minimum_component_size)
        elif "minimum_component_size" in state:
            packaged["minimum_component_size"] = int(state["minimum_component_size"])
        # The sparse-isolated profile is calibrated against a specific checkpoint, so it
        # travels with the weights rather than being hardcoded in the container.
        if a.sparse_threshold is not None:
            packaged["sparse_threshold"] = float(a.sparse_threshold)
        elif "sparse_threshold" in state:
            packaged["sparse_threshold"] = float(state["sparse_threshold"])
        if a.sparse_minimum_component_size is not None:
            packaged["sparse_minimum_component_size"] = int(a.sparse_minimum_component_size)
        elif "sparse_minimum_component_size" in state:
            packaged["sparse_minimum_component_size"] = int(state["sparse_minimum_component_size"])
        # The input convention travels with the weights: a model trained on per-window
        # normalised inputs silently degrades if the container feeds it anything else.
        if state.get("per_window_norm"):
            packaged["per_window_norm"] = True
        if state.get("norm"):
            packaged["norm"] = state["norm"]
        if a.hysteresis_low is not None:
            packaged["hysteresis_low"] = float(a.hysteresis_low)
        if state.get("patch"):
            packaged["patch"] = [int(v) for v in state["patch"]]
        if "metrics" in state:
            packaged["metrics"] = state["metrics"]
        torch.save(packaged, dst)
        mb = dst.stat().st_size / 2 ** 20
        total += mb
        m = state.get("metrics", {})
        print(f"  {ckpt.name:14s} epoch {state.get('epoch','?'):>4}  "
              f"challenge {m.get('challenge_score', float('nan')):.4f}  "
              f"threshold {packaged.get('threshold', 0.5):.2f}  "
              f"min-size {packaged.get('minimum_component_size', 1)}  "
              f"-> {dst.name} ({mb:.0f} MB)")

    n = len(list(resources.glob("*.pt")))
    print(f"\n{n} checkpoint(s) in {resources}, {total:.0f} MB total")


if __name__ == "__main__":
    main()
