#!/usr/bin/env bash
#SBATCH --job-name=fec1-scratch-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${FEC1_PYTHON:?set absolute Python executable}"
: "${FEC1_REVIEWED_COMMIT:?set reviewed Git commit}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
test "$(git rev-parse HEAD)" = "$FEC1_REVIEWED_COMMIT"
test -z "$(git status --porcelain)"
"$FEC1_PYTHON" -m fec1 verify-source
"$FEC1_PYTHON" -m pytest -q
"$FEC1_PYTHON" -m pytest -q -m integration --basetemp="outputs/scratch-test-${SLURM_JOB_ID:?}"
"$FEC1_PYTHON" -m fec1 verify-source
test "$(git rev-parse HEAD)" = "$FEC1_REVIEWED_COMMIT"
test -z "$(git status --porcelain)"
