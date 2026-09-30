"""Qualitative figures: v15a on held-out volumes (with labels) and v15 on unseen-type preliminary test volumes."""
import os
import sys
from pathlib import Path
import numpy as np, SimpleITK as sitk, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
# folders with packaged weights (see submission/resources/README.md): v15a alone, and v15a + v15b
RES_V15A = Path(os.environ.get("RES_V15A", ROOT / "weights" / "v15a_only"))
RES_V15 = Path(os.environ.get("RES_V15", ROOT / "weights" / "v15"))
sys.path.insert(0, str(ROOT / "selma")); sys.path.insert(0, str(ROOT / "submission"))
from data import PREP, load_raw  # noqa
from train import dice  # noqa
import inference as C  # noqa
OUT = ROOT / "paper" / "fig"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
dev = torch.device("cuda")
def mip(v, ax, d=24):
    c = v.shape[ax] // 2; lo, hi = max(0, c - d // 2), min(v.shape[ax], c + d // 2)
    return np.take(v, range(lo, hi), axis=ax).max(axis=ax)
def st(m):
    lo, hi = np.percentile(m, [1, 99.7]); return np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1)
def overlay(m, g, col):
    rgb = np.dstack([m, m, m]) * 0.75; rgb[g] = col; return rgb

C.RESOURCE_PATH = RES_V15A
models, thr, ms, sth, sms = C.load_models(dev)
hold = [("cfos_neuron_patchvolume_753", "c-Fos neurons"), ("blood_vessel_patchvolume_003", "Blood vessels"),
        ("Human_NEFH_transition_raw", "Axons (NEFH)")]
fig, axs = plt.subplots(3, 3, figsize=(8.4, 8.6), dpi=170)
for i, (stem, name) in enumerate(hold):
    img = np.asarray(load_raw(stem, mmap=False)); gt = np.load(PREP / f"{stem}__lab.npy").astype(bool)
    pred = C.predict(img, models, dev, thr, ms, sth, sms) > 0
    d = dice(pred, gt); ax_ = int(np.argmin(img.shape)); dd = min(24, img.shape[ax_])
    m = st(mip(img.astype(np.float32), ax_, dd))
    axs[i, 0].imshow(m, cmap="gray"); axs[i, 0].set_ylabel(name)
    axs[i, 1].imshow(overlay(m, mip(gt, ax_, dd), [1.0, 0.45, 0.1]))
    axs[i, 2].imshow(overlay(m, mip(pred, ax_, dd), [0.2, 0.6, 1.0])); axs[i, 2].set_xlabel(f"Dice {d:.3f}")
    print(stem, f"dice {d:.3f}", flush=True)
for j, t in enumerate(["image", "ground truth", "v15a prediction"]): axs[0, j].set_title(t)
for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(OUT / "qual_holdout.png"); plt.close(fig)

C.RESOURCE_PATH = RES_V15
models, thr, ms, sth, sms = C.load_models(dev)
T = ROOT / "data" / "prelim_test"
unseen = [("contiguous_structures/gutnerve_patchvolume_60_1.mha", "Gut nerve"),
          ("contiguous_structures/lymphatic_vessel_patchvolume_70_0.mha", "Lymphatic vessels"),
          ("isolated_structures/microglia_patchvolume_1874.mha", "Microglia"),
          ("isolated_structures/protein_kidneyvolume_1.mha", "Organ protein (kidney)")]
fig, axs = plt.subplots(2, 4, figsize=(12, 6.4), dpi=170)
for j, (rel, name) in enumerate(unseen):
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(T / rel)))
    pred = C.predict(img, models, dev, thr, ms, sth, sms) > 0
    ax_ = int(np.argmin(img.shape)); dd = min(24, img.shape[ax_])
    m = st(mip(img.astype(np.float32), ax_, dd))
    axs[0, j].imshow(m, cmap="gray"); axs[0, j].set_title(name)
    axs[1, j].imshow(overlay(m, mip(pred, ax_, dd), [0.2, 0.6, 1.0]))
    print(rel, f"fg {pred.mean()*100:.3f}%", flush=True)
axs[0, 0].set_ylabel("image (slab MIP)"); axs[1, 0].set_ylabel("v15 prediction")
for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(OUT / "qual_unseen.png"); plt.close(fig)
print("done")
