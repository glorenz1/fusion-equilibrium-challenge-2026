#!/usr/bin/env bash
#SBATCH --job-name=fec1-reproduce
#SBATCH --account=acf-utk0011
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${FEC1_PYTHON:?set the absolute Python executable}"
: "${FEC1_REVIEWED_COMMIT:?set the reviewed repository commit}"
: "${FEC1_INPUTS:?set the public parquet directory}"
: "${FEC1_REFERENCE:?set the retained prediction archive}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test "$(git rev-parse HEAD)" = "$FEC1_REVIEWED_COMMIT"
test -z "$(git status --porcelain)"
"$FEC1_PYTHON" -m fec1 verify-source
run="outputs/replay-${SLURM_JOB_ID:?}"
mkdir -p "$run"
"$FEC1_PYTHON" - "$run/environment.json" <<'PY'
import importlib.metadata as m, json, platform, subprocess, sys
from pathlib import Path
import torch
Path(sys.argv[1]).write_text(json.dumps(dict(
    python=sys.version, host=platform.node(), cuda=torch.version.cuda,
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    packages={n:m.version(n) for n in ['torch','numpy','scikit-learn','scipy','pandas','pyarrow']}
),indent=2)+'\n')
PY
"$FEC1_PYTHON" -m pytest -q
"$FEC1_PYTHON" -m fec1 predict --inputs "$FEC1_INPUTS" --output "$run" \
  --device cpu --workers 4 --batch-size 128 --max-shots "${FEC1_MAX_SHOTS:-1}"
"$FEC1_PYTHON" -m fec1 compare "$run/diii_d_public_test.npz" "$FEC1_REFERENCE" \
  --report "$run/parity.json"
"$FEC1_PYTHON" -m fec1 verify-source
test "$(git rev-parse HEAD)" = "$FEC1_REVIEWED_COMMIT"
test -z "$(git status --porcelain)"
