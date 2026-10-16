#!/usr/bin/env bash
# Historical preparation recipe. Run only in a fresh data/artifact installation.
set -euo pipefail
cd "$(dirname "$0")/.."
py="${FEC1_PYTHON:-.venv/bin/python}"
cache=data/cache
art=artifacts/training
mkdir -p "$art"
"$py" -m fec1.data.cache --input-dir data/diii_d_train \
  --output-dir "$cache" --frames-per-shot 0 --workers 16 --uncompressed
"$py" scripts/prepare_pca.py --workers 8 \
  --output "$art/source_pca.joblib"
"$py" -m fec1.training.source_rows --folds 5 --workers 8 \
  --train-stride 4 --estimators 96 --min-samples-leaf 4 --max-features 0.7 \
  --output "$art/source_rows.npz"
"$py" -m fec1.training.inputs \
  --oof "$art/source_rows.npz" --cache-dir "$cache" \
  --output-dir "$art/inputs"
"$py" -m fec1.training.targets \
  --oof "$art/source_rows.npz" \
  --base-artifact "$art/source_pca.joblib" \
  --output "$art/target_normalized.npy" \
  --pca-output "$art/pca_basis.npz"
"$py" -m fec1.training.topology --all-train-stride 4 \
  --workers 7 --output "$art/topology.npz"
"$py" -m fec1.training.contours \
  --reference "$art/topology.npz" \
  --angles 64 --contour-points 128 --workers 8 \
  --output "$art/contours.npz"
