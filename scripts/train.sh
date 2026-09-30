#!/usr/bin/env bash
# Auto-resuming training on a Linux GPU server (A100).
#
#   bash run_a100.sh <run_dir> <hours> <holdout_json> [extra train_final.py args]
#
# The deadline is fixed on first launch (stored in <run_dir>/DEADLINE); relaunching the
# same command after a crash or disconnect resumes from <run_dir>/latest.pt.
set -u
cd "$(dirname "$0")/.."
OUT=${1:?run dir}; HOURS=${2:?hours}; HOLDOUT=${3:?holdout json}; shift 3
mkdir -p "$OUT"
[ -f "$OUT/DEADLINE" ] || python -c "import time,sys; print(int(time.time() + float(sys.argv[1]) * 3600))" "$HOURS" > "$OUT/DEADLINE"
DEADLINE=$(cat "$OUT/DEADLINE")
for attempt in $(seq 1 40); do
    if [ -f "$OUT/latest.pt" ]; then START=(--resume "$OUT/latest.pt"); else START=(--init none); fi
    echo "=== attempt $attempt $(date '+%F %T') ${START[*]} ===" >> "$OUT/train.log"
    python selma/train_final.py --out "$OUT" --holdout "$HOLDOUT" --deadline "$DEADLINE" \
        --backup-dir "" --workers 8 "${START[@]}" "$@" >> "$OUT/train.log" 2>&1
    code=$?
    echo "=== attempt $attempt exited $code $(date '+%F %T') ===" >> "$OUT/train.log"
    [ "$code" -eq 0 ] && exit 0
    sleep 20
done
