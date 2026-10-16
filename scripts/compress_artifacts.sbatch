#!/usr/bin/env bash
#SBATCH --job-name=fec1-compress
#SBATCH --account=acf-utk0011
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${FEC1_PYTHON:?set the absolute Python executable}"
: "${FEC1_REVIEWED_COMMIT:?set the reviewed commit}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test "$(git rev-parse HEAD)" = "$FEC1_REVIEWED_COMMIT"
test -z "$(git status --porcelain)"
"$FEC1_PYTHON" fec1.py verify-source
"$FEC1_PYTHON" -m pytest tests/test_artifact_storage.py -q
"$FEC1_PYTHON" fec1.py pack-artifacts \
  --destination "outputs/compression-${SLURM_JOB_ID:?}/weights" \
  --workers 5 --level 6
"$FEC1_PYTHON" fec1.py verify-source
test "$(git rev-parse HEAD)" = "$FEC1_REVIEWED_COMMIT"
test -z "$(git status --porcelain)"
