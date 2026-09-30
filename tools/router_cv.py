"""Cross-validate the morphology router's thresholds instead of fitting them on all 108 volumes.

Leave-one-out and leave-one-structure-type-out: the rule's thresholds are grid-searched on
the remaining volumes (an isolated volume sent to the contiguous experts costs 3, the
reverse costs 1 because it only reproduces the baseline), then applied to the held-out ones.

    python tools/router_cv.py
"""
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
rows = json.loads((ROOT / "runs" / "router_mask_features.json").read_text())
def structure_type(name):
    n = name.lower()
    for key, t in (("nef", "neurofilament"), ("_pv_", "neurofilament"), ("nulei", "nuclei"), ("nuclei", "nuclei"),
                   ("protein", "organ protein")):
        if key in n:
            return t
    return name.split("_patchvolume")[0].split("_patch")[0]


for r in rows:
    r["type"] = structure_type(r["name"])

GRID = list(itertools.product([0.2, 0.25, 0.3, 0.35, 0.4, 0.5], [500, 1000, 2000, 10 ** 9],
                              [2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 10 ** 9], [0.0, 0.0005]))
DEFAULT = (0.3, 1000, 3.0, 0.0005)


def route(r, p):
    lg, nc, el, fg = p
    if r["fg"] < fg:
        return "iso"
    return "cont" if (r["largest"] >= lg and r["ncomp"] <= nc) or r["elong_w"] >= el else "iso"


def cost(rs, p):
    c = 0
    for r in rs:
        g = route(r, p)
        c += 3 if (r["cat"] == "iso" and g == "cont") else (1 if g != r["cat"] else 0)
    return c


def fit(rs):
    best = min(cost(rs, p) for p in GRID)
    ties = [p for p in GRID if cost(rs, p) == best]
    # among equally good rules prefer the one closest to the shipped default
    return min(ties, key=lambda p: sum(a != b for a, b in zip(p, DEFAULT)))


def report(name, splits):
    harm = safe = 0
    wrong = []
    for held, train in splits:
        p = fit(train)
        for r in held:
            g = route(r, p)
            if g != r["cat"]:
                harm += r["cat"] == "iso"
                safe += r["cat"] == "cont"
                wrong.append(f"{r['name'][:34]} ({r['cat']}->{g})")
    n = sum(len(h) for h, _ in splits)
    print(f"{name}: {harm + safe}/{n} wrong  (iso->cont harmful {harm}, cont->iso safe {safe})")
    for w in wrong:
        print("    " + w)
    return harm, safe


print(f"fitted on all 108: default rule cost {cost(rows, DEFAULT)}, best grid rule {fit(rows)} cost {cost(rows, fit(rows))}")
report("leave-one-volume-out", [([r], [s for s in rows if s is not r]) for r in rows])
types = sorted(set(r["type"] for r in rows))
print("structure types:", dict(Counter(r["type"] for r in rows)))
report("leave-one-structure-type-out", [([r for r in rows if r["type"] == t], [r for r in rows if r["type"] != t]) for t in types])
