#!/usr/bin/env bash
# The recorded historical final-fit command; this script does not bypass
# the parent project's required pilot admission for future long fits.
set -euo pipefail
cd "$(dirname "$0")/.."
fold="${1:?fold 0 through 4 is required}"
case "$fold" in 0|1|2|3|4) ;; *) exit 2 ;; esac
py="${FEC1_PYTHON:-.venv/bin/python}"
root=my_experiments/candidates/fec1_dct24_more_train_20260929
"$py" my_experiments/train_fec1_structured_correction.py \
  --data "$root/data_fold${fold}" --output "$root/train_fold${fold}" \
  --decoder dct --epochs 36 --batch-size 128 --width 384 --channels 24 \
  --geometry-weight 0.003 --loss-version symmetric --geometry-components both \
  --scalar-reference "$root/data_fold${fold}/scalar_reference.pt" \
  --score-workers 4 --observation-encoder spatial_attention --geometry-batch 16 \
  --geometry-warmup 4 --exact-every 6 --selection-shots 64 --patience-checks 3 \
  --seed 260905 --device cuda
