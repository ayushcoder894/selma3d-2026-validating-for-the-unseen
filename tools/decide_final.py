"""Compare routed candidates from their preliminary-image check logs (prelim_route_check.py).

The calibration finding: preliminary-test score fell whenever a model's foreground on UNSEEN
structure types moved away from v15's (v11 x1.19 -> .537, v17 x0.75 -> .573, v15 .616).
A candidate is preferred when it routes all 26 volumes correctly and its unseen-type
foreground stays close to v15 (seen types may move: they are covered by the holdout).

    python tools/decide_final.py logs/route_FA.log logs/route_FB.log [...]
"""
import math
import re
import sys
from pathlib import Path

UNSEEN = ["gut nerve", "lymphatic", "axonal", "microglia", "protein"]
SEEN = ["vessel", "periph nerve", "nuclei", "c-Fos", "plaques"]


def parse(path):
    text = Path(path).read_text(errors="ignore")
    err = re.search(r"routing errors: (\d+)/26", text)
    ratios = {m[1].strip(): float(m[2]) for m in re.finditer(r"^\s+(\S[\w -]*?)\s+[\d.]+% vs v15\s+[\d.]+%\s+x([\d.]+)", text, re.M)}
    return (int(err[1]) if err else None), ratios


def main():
    rows = []
    for p in sys.argv[1:]:
        err, r = parse(p)
        unseen = [r[k] for k in UNSEEN if k in r]
        dev = math.exp(sum(abs(math.log(v)) for v in unseen) / len(unseen)) if unseen else float("nan")
        rows.append((Path(p).stem, err, r, dev))
        print(f"{Path(p).stem:12s} routing errors {err}/26   unseen-type foreground drift x{dev:.3f} (geometric mean |log ratio|)")
        print("    unseen: " + "  ".join(f"{k} x{r[k]:.2f}" for k in UNSEEN if k in r))
        print("    seen:   " + "  ".join(f"{k} x{r[k]:.2f}" for k in SEEN if k in r))
    ok = [x for x in rows if x[1] == 0]
    if ok:
        best = min(ok, key=lambda x: x[3])
        print(f"\nclosest to v15 on unseen types with correct routing: {best[0]} (drift x{best[3]:.3f})")
        for name, _, r, _ in ok:
            low = [k for k in UNSEEN if k in r and r[k] < 0.8]
            high = [k for k in UNSEEN if k in r and r[k] > 1.25]
            if low or high:
                print(f"  warning {name}: under-segments {low or '-'}; over-segments {high or '-'} (v17 was 0.55-0.73x)")
    else:
        print("\nno candidate routed all 26 volumes correctly")


if __name__ == "__main__":
    main()
