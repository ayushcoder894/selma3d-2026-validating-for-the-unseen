#!/usr/bin/env bash
# End-to-end reproduction of our SELMA3D 2026 submissions. Run from anywhere; each stage can
# be run on its own:
#
#   bash scripts/reproduce.sh data       # download annotated data + 3,400-patch pretraining pool (~110 GB)
#   bash scripts/reproduce.sh prepare    # preprocess the 82 annotated patches into data/prep*
#   bash scripts/reproduce.sh ssl        # signal-weighted masked-reconstruction pretraining
#   bash scripts/reproduce.sh v15        # fine-tune the two 96^3 ensemble members (v15a, v15b)
#   bash scripts/reproduce.sh experts    # fine-tune the two 128^3 contiguous experts (routed entry only)
#   bash scripts/reproduce.sh containers # package weights and build both Grand Challenge containers
#
# Training is time-budgeted: the learning-rate schedule follows wall-clock progress towards a
# deadline fixed at first launch. Our runs: SSL ~3 h and each v15 member 4 h on an RTX 4060
# Laptop GPU (8 GB); experts ~3 h each on an A100. On a faster GPU, shorten the hours to reach a
# similar number of epochs (v15a: 174, v15b: 187, experts: ~50; 250 iterations per epoch).
set -eu
cd "$(dirname "$0")/.."
STAGE=${1:?stage: data | prepare | ssl | v15 | experts | containers}
HOURS_SSL=${HOURS_SSL:-3}
HOURS_V15=${HOURS_V15:-4}
HOURS_EXPERT=${HOURS_EXPERT:-3}

# Fine-tuning recipe E (identical for all four networks apart from patch size and data split).
RECIPE="--norm zscore --optimizer sgd --schedule poly --peak-lr 1e-2 --floor-lr 2e-6 --warmup 0.015 \
  --weight-decay 3e-5 --ema-decay 0.999 --p-fg 0.33 --pos-weight 1 --w-cldice 0 --uniform-sampling \
  --steps 250 --batch 2 --val-every 20 --init runs/ssl_v2/final.pt"

case "$STAGE" in
  data)
    python get_selma3d.py annotated --out data
    python get_selma3d.py pretrain --out data --per-type 200 --seed 0     # writes data/manifest_pretrain_200.json
    ;;
  prepare)
    python selma/prepare_all.py --src data/SELMA3D2026_training_annotated
    ;;
  ssl)
    python selma/pretrain_ssl_v2.py --data data --manifest manifest_pretrain_200.json \
        --out runs/ssl_v2 --hours "$HOURS_SSL" --batch 2 --patch 96
    ;;
  v15)
    # v15a keeps the 9-volume held-out split for validation; v15b uses all 82 volumes.
    bash scripts/train.sh runs/v15a "$HOURS_V15" selma/holdout_v11.json  $RECIPE --patch 96 96 96
    bash scripts/train.sh runs/v15b "$HOURS_V15" selma/holdout_none.json $RECIPE --patch 96 96 96
    ;;
  experts)
    bash scripts/train.sh runs/B128_all1 "$HOURS_EXPERT" selma/holdout_none.json $RECIPE --patch 128 128 128
    bash scripts/train.sh runs/B128_all2 "$HOURS_EXPERT" selma/holdout_none.json $RECIPE --patch 128 128 128
    ;;
  containers)
    # Task 2 entry: v15 only.  Task 1 entry: v15 + routed 128^3 experts with foreground anchoring.
    bash scripts/build_container.sh v15    runs/v15a/final_ema.pt runs/v15b/final_ema.pt
    bash scripts/build_container.sh routed runs/v15a/final_ema.pt runs/v15b/final_ema.pt -- \
        runs/B128_all1/final_ema.pt runs/B128_all2/final_ema.pt
    ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
