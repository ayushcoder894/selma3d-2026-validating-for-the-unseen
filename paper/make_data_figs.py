"""Dataset figures for the paper: annotated examples with labels, and the 17 unlabelled pool types."""
import json, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "selma"))
from data import PREP, load_index, load_raw  # noqa: E402
OUT = ROOT / "paper" / "fig"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

def slab_mip(v, axis=0, depth=24):
    c = v.shape[axis] // 2
    lo, hi = max(0, c - depth // 2), min(v.shape[axis], c + depth // 2)
    return np.take(v, range(lo, hi), axis=axis).max(axis=axis)

def stretch(img):
    lo, hi = np.percentile(img, [1, 99.7])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

idx = {r["stem"]: r for r in load_index()}
plq = max((r for r in idx.values() if r["family"] == "AD_plaques"), key=lambda r: r["fg_frac"])["stem"]
pick = [(plq, "Amyloid plaques", "isolated"),
        ("cell_nuclei_patchvolume_002", "Cell nuclei", "isolated"),
        ("cfos_neuron_patchvolume_753", "c-Fos neurons", "isolated"),
        ("blood_vessel_patchvolume_003", "Blood vessels", "contiguous"),
        ("peripheral_nerve_patchvolume_366", "Peripheral nerves", "contiguous"),
        ("Human_NEFH_transition_raw", "Axons (NEFH)", "contiguous")]
fig, axs = plt.subplots(2, 6, figsize=(13, 4.9), dpi=170)
for j, (stem, title, cat) in enumerate(pick):
    img = np.asarray(load_raw(stem, mmap=False)).astype(np.float32)
    lab = np.load(PREP / f"{stem}__lab.npy").astype(bool)
    ax_ = int(np.argmin(img.shape))                     # project along the thinnest axis
    depth = img.shape[ax_] if "plaques" in stem else min(24, img.shape[ax_])
    m = stretch(slab_mip(img, ax_, depth)); g = slab_mip(lab, ax_, depth)
    axs[0, j].imshow(m, cmap="gray"); axs[0, j].set_title(f"{title}\n({cat})", fontsize=9.5)
    rgb = np.dstack([m, m, m]) * 0.75; rgb[g] = [1.0, 0.45, 0.1]
    axs[1, j].imshow(rgb)
for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
axs[0, 0].set_ylabel("image (slab MIP)"); axs[1, 0].set_ylabel("annotation")
fig.tight_layout(); fig.savefig(OUT / "dataset_annotated.png"); plt.close(fig)
print("annotated done", flush=True)

DATA = ROOT / "data"
man = json.load(open(DATA / "manifest_pretrain_200.json"))
cands = {}
for x in man:
    cands.setdefault((x["category"], x["structure"]), []).append(x["path"])
def contrast(m):
    lo, mid, hi = np.percentile(m, [5, 50, 99.8])
    return (hi - mid) / (mid - lo + hi * 0.05 + 1e-6)
first = {}
names = {"Ab_plaques": "Aβ plaques", "Collagen_Chondrocytes": "Collagen / chondrocytes", "GFAP_astrocytes": "GFAP astrocytes",
         "Sox9_Chondrogenic_cells": "Sox9 chondrogenic cells", "TH_neurons": "TH neurons", "c-Fos_brain_cells": "c-Fos cells",
         "cell_nucleus": "Cell nuclei", "Beta3tubulin_cranial_nerve": "β3-tubulin cranial nerve", "CTIP2_neurons": "CTIP2 neurons",
         "NF_peripheral_nerve": "NF peripheral nerve", "P75_nerves_selected": "P75 nerves", "PGP_peripheral_nerve": "PGP peripheral nerve",
         "SMA_artery": "SMA arteries", "TH_sympathetic_nerve": "TH sympathetic nerve", "axonal_markers": "Axonal markers",
         "lyve1_lymphatic_vessel": "LYVE1 lymphatic vessels", "vessel": "Blood vessels"}
keys = sorted(cands, key=lambda k: (k[0] != "isolated", k[1].lower()))
fig, axs = plt.subplots(3, 6, figsize=(13, 7.8), dpi=150)
def pool_stretch(img):
    lo, hi = np.percentile(img, [2, 99.9])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
for a, k in zip(axs.ravel(), keys):
    best = None
    for pth in cands[k][:8]:
        v = sitk.GetArrayFromImage(sitk.ReadImage(str(DATA / pth))).astype(np.float32)
        m = slab_mip(v, 0, 60)
        sc = contrast(m)
        if best is None or sc > best[0]:
            best = (sc, m)
    a.imshow(pool_stretch(best[1]), cmap="gray")
    a.set_title(f"{names.get(k[1], k[1])}\n[{k[0]}]", fontsize=8.6)
for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
axs.ravel()[-1].axis("off")
fig.tight_layout(); fig.savefig(OUT / "dataset_pool.png"); plt.close(fig)
print("pool done")
