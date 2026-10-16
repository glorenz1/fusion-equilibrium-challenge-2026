#!/usr/bin/env bash
#SBATCH --job-name=fec1-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${FEC1_PYTHON:?set absolute Python executable}"
: "${FEC1_RUN_DIR:?set absolute prepared run directory}"
: "${FEC1_REVIEWED_COMMIT:?set reviewed Git commit}"
test "$(git rev-parse HEAD)" = "$FEC1_REVIEWED_COMMIT"
test -z "$(git status --porcelain)"
"$FEC1_PYTHON" -m fec1 verify-source
"$FEC1_PYTHON" -m fec1 train --run "$FEC1_RUN_DIR" --device cuda \
  --workers "${SLURM_CPUS_PER_TASK:-8}" --resume
"$FEC1_PYTHON" -m fec1 verify-source
test "$(git rev-parse HEAD)" = "$FEC1_REVIEWED_COMMIT"
test -z "$(git status --porcelain)"
