#!/usr/bin/env bash
# Historical preparation recipe. Run only in a fresh data/artifact installation.
set -euo pipefail
cd "$(dirname "$0")/.."
py="${FEC1_PYTHON:-.venv/bin/python}"
cache=my_experiments/neuralgs_real_d3d_full_frame_cache
art=my_experiments/artifacts
mkdir -p "$art"
"$py" train_neuralgs.py prepare --input-dir data/diii_d_train \
  --output-dir "$cache" --frames-per-shot 0 --workers 16 --uncompressed
"$py" scripts/prepare_pca.py --workers 8 \
  --output "$art/fec1_full_frame_pca64_extra_trees.joblib"
"$py" my_experiments/generate_fec1_train_oof_base.py --folds 5 --workers 8 \
  --train-stride 4 --estimators 96 --min-samples-leaf 4 --max-features 0.7 \
  --output "$art/fec1_train_stride4_oof_base.npz"
"$py" my_experiments/prepare_fec1_topology_native_inputs.py \
  --oof "$art/fec1_train_stride4_oof_base.npz" --cache-dir "$cache" \
  --output-dir "$art/fec1_topology_native_inputs"
"$py" my_experiments/prepare_fec1_score_aligned_targets.py \
  --oof "$art/fec1_train_stride4_oof_base.npz" \
  --base-artifact "$art/fec1_full_frame_pca64_extra_trees.joblib" \
  --output "$art/fec1_train_stride4_target_psi_n.npy" \
  --pca-output "$art/fec1_train_pca64_basis.npz"
"$py" my_experiments/extract_fec1_train_exact_topology.py --all-train-stride 4 \
  --workers 7 --output "$art/fec1_train_all_frame_stride4_exact_topology.npz"
"$py" my_experiments/extract_fec1_exact_contour_radius.py \
  --reference "$art/fec1_train_all_frame_stride4_exact_topology.npz" \
  --angles 64 --contour-points 128 --workers 8 \
  --output "$art/fec1_train_stride4_exact_contour_radius.npz"
