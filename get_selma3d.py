#!/usr/bin/env python3
"""
SELMA3D 2026 - Task 1 subset downloader (EBI BioStudies, plain HTTPS, stdlib only).

Task 1 data lives in four BioStudies accessions:
  S-BIAD2110  isolated   annotated   (fine-tuning)     38 raw/gt pairs   ~0.86 GB
  S-BIAD2106  contiguous annotated   (fine-tuning)     44 raw/gt pairs   ~1.54 GB
  S-BIAD3281  isolated   unannotated (SSL pretraining) 17,470 patches    ~568 GB
  S-BIAD3277  contiguous unannotated (SSL pretraining) 19,600 patches    ~627 GB

Patches are individual .nii.gz files, so any subset can be taken.

Usage
  python get_selma3d.py plan --per-type 200          # size estimate, no download
  python get_selma3d.py annotated                    # 2.4 GB, get this first
  python get_selma3d.py pretrain --per-type 200      # ~112 GB, all 17 types
  python get_selma3d.py pretrain --per-type 400 --only matched
  python get_selma3d.py pretrain --per-type 100 --only c-Fos_brain_cells,vessel

Re-running resumes: files whose size already matches the listing are skipped.
"""
import argparse
import json
import random
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FIRE = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD"
ACC = {
    "3281": FIRE + "/281/S-BIAD3281/Files/",
    "3277": FIRE + "/277/S-BIAD3277/Files/",
    "2110": FIRE + "/110/S-BIAD2110/Files/",
    "2106": FIRE + "/106/S-BIAD2106/Files/",
}

# name, accession, file-list, category, n_patches, GB, matched = has an annotated/eval counterpart
UNANN = [
    ("c-Fos_brain_cells",         "3281", "c-Fos_brain_cells_patches",           "isolated",   5400, 180.4, True),
    ("cell_nucleus",              "3281", "cell_nucleus_patches",                "isolated",   3500, 126.2, True),
    ("Ab_plaques",                "3281", "Ab_plaques_patches",                  "isolated",   1470,  45.8, True),
    ("TH_neurons",                "3281", "TH_neurons_patches",                  "isolated",   2500,  81.1, False),
    ("GFAP_astrocytes",           "3281", "GFAP_astrocytes_patches",             "isolated",   2400,  79.1, False),
    ("Collagen_Chondrocytes",     "3281", "Collagen_Chondrocytes_patches",       "isolated",   1100,  29.9, False),
    ("Sox9_Chondrogenic_cells",   "3281", "Sox9_Chondrogenic_cells_patches",     "isolated",   1100,  27.3, False),
    ("vessel",                    "3277", "brain_vessel_patches",                "contiguous", 2700, 102.2, True),
    ("axonal_markers",            "3277", "axonal_markers_patches",              "contiguous", 2500,  67.9, True),
    ("P75_nerves_selected",       "3277", "P75_nerves_selected_patches",         "contiguous", 3150,  94.4, True),
    ("PGP_peripheral_nerve",      "3277", "PGP_peripheral_nerve_patches",        "contiguous", 1300,  38.6, True),
    ("NF_peripheral_nerve",       "3277", "NF_peripheral_nerve_patches",         "contiguous", 1050,  28.1, True),
    ("CTIP2_neurons",             "3277", "CTIP2_neurons_patches",               "contiguous", 3300, 108.4, False),
    ("TH_sympathetic_nerve",      "3277", "TH_sympathetic_nerve_patches",        "contiguous", 2250,  70.4, False),
    ("lyve1_lymphatic_vessel",    "3277", "lyve1_lymphatic_vessel_patches",      "contiguous", 1300,  45.9, False),
    ("Beta3tubulin_cranial_nerve","3277", "Beta3tubulin_cranial_nerve_patches",  "contiguous", 1200,  39.8, False),
    ("SMA_artery",                "3277", "SMA_artery_patches",                  "contiguous",  850,  29.6, False),
]

ANN = [("2110", n) for n in ["isolated_raw_cfos", "isolated_gt_cfos",
                             "isolated_raw_nuclei", "isolated_gt_nuclei",
                             "isolated_raw_plaque", "isolated_gt_plaque"]] + \
      [("2106", n) for n in ["contiguous_raw_vessel", "contiguous_gt_vessel",
                             "contiguous_raw_nerve", "contiguous_gt_nerve",
                             "contiguous_raw_axon", "contiguous_gt_axon"]]


def get_json(url, cache):
    if cache.exists():
        return json.loads(cache.read_text())
    with urllib.request.urlopen(url, timeout=180) as r:
        data = r.read()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    return json.loads(data)


def filelist(acc, name, root):
    return get_json(ACC[acc] + name + ".json", root / "_filelists" / (acc + "_" + name + ".json"))


def stratify(files, k, seed=0):
    """Take k files spread evenly across the sample sub-directories, deterministically."""
    if k is None or k >= len(files):
        return list(files)
    groups = {}
    for f in files:
        groups.setdefault("/".join(f["path"].split("/")[:-1]), []).append(f)
    rng = random.Random(seed)
    for g in groups.values():
        g.sort(key=lambda f: f["path"])
        rng.shuffle(g)
    keys = sorted(groups)
    out, i = [], 0
    while len(out) < k:
        g = groups[keys[i % len(keys)]]
        if g:
            out.append(g.pop())
        elif all(not groups[x] for x in keys):
            break
        i += 1
    return out


def fetch(job):
    url, dest, size = job
    if dest.exists() and dest.stat().st_size == size:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        # A server that closes the connection early still ends the read "cleanly";
        # without this check a truncated file would be counted as downloaded.
        if tmp.stat().st_size != size:
            raise IOError(f"truncated: {tmp.stat().st_size} of {size} bytes")
        tmp.replace(dest)
        return size
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print("  FAIL " + dest.name + ": " + str(e), file=sys.stderr)
        return -1


def run(jobs, workers, label):
    todo = [j for j in jobs if not (j[1].exists() and j[1].stat().st_size == j[2])]
    total = sum(j[2] for j in todo)
    print("%s: %d/%d files to fetch, %.1f GB" % (label, len(todo), len(jobs), total / 2 ** 30))
    done = bad = 0
    with ThreadPoolExecutor(workers) as ex:
        for i, got in enumerate(ex.map(fetch, todo), 1):
            if got < 0:
                bad += 1
            else:
                done += got
            if i % 25 == 0 or i == len(todo):
                print("  %d/%d  %.1f/%.1f GB" % (i, len(todo), done / 2 ** 30, total / 2 ** 30), flush=True)
    print("%s: done%s" % (label, (" (%d failures - rerun to retry)" % bad) if bad else ""))


def select(only):
    if not only:
        return UNANN
    if only == "matched":
        return [s for s in UNANN if s[6]]
    if only in ("isolated", "contiguous"):
        return [s for s in UNANN if s[3] == only]
    want = set(x.strip() for x in only.split(","))
    picked = [s for s in UNANN if s[0] in want]
    missing = want - set(s[0] for s in picked)
    if missing:
        sys.exit("unknown structure(s): %s\nvalid: %s" % (sorted(missing), [s[0] for s in UNANN]))
    return picked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["plan", "annotated", "pretrain"])
    p.add_argument("--out", default="data")
    p.add_argument("--per-type", type=int, default=200, help="patches per structure (use 99999 for all)")
    p.add_argument("--only", default="", help="'matched', 'isolated', 'contiguous', or comma-separated names")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    root = Path(a.out)

    if a.mode == "plan":
        picked, tot, npatch = select(a.only), 0.0, 0
        print("%-30s%-11s%7s%7s%9s  %s" % ("structure", "cat", "have", "take", "GB", "eval-matched"))
        for name, acc, fl, cat, n, gb, m in picked:
            k = min(a.per_type, n)
            g = gb * k / n
            tot += g
            npatch += k
            print("%-30s%-11s%7d%7d%9.1f  %s" % (name, cat, n, k, g, "yes" if m else ""))
        print("\n%d patches, %.1f GB  (+2.4 GB annotated)" % (npatch, tot))
        return

    if a.mode == "annotated":
        jobs = []
        for acc, name in ANN:
            for f in filelist(acc, name, root):
                jobs.append((ACC[acc] + urllib.parse.quote(f["path"]), root / f["path"], f["size"]))
        run(jobs, a.workers, "annotated")
        return

    manifest, jobs = [], []
    for name, acc, fl, cat, n, gb, m in select(a.only):
        files = filelist(acc, fl, root)
        sub = stratify(files, a.per_type, a.seed)
        print("  %-30s %5d/%d patches  %6.1f GB" % (name, len(sub), len(files),
                                                    sum(f["size"] for f in sub) / 2 ** 30))
        for f in sub:
            jobs.append((ACC[acc] + urllib.parse.quote(f["path"]), root / f["path"], f["size"]))
            manifest.append({"structure": name, "category": cat, "path": f["path"], "size": f["size"]})
    root.mkdir(parents=True, exist_ok=True)
    (root / ("manifest_pretrain_%d.json" % a.per_type)).write_text(json.dumps(manifest, indent=1))
    random.Random(1).shuffle(jobs)  # balanced partial downloads
    run(jobs, a.workers, "pretrain")


if __name__ == "__main__":
    main()
