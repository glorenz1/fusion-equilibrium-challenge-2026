# FEC1: reproduce the DIII-D 0.9931 ensemble

This repository preserves the five-fold spatial-attention DCT24 model reported
as **S = 0.9931 on Codabench on 2026-09-29**. Its separate, local pooled outer-OOF
score is **0.990843 on 5,677 training shots**. These scores use different
populations. Public hidden labels are unavailable; the public score is a
historical reference, not a score recomputed by this repository.

The implementation is copied byte-for-byte from incumbent source snapshot
`625e6f3c0e67b70f7f9c28526e8e3972b7457b9f`. It includes preprocessing, tree
carriers, neural architecture, calibration, training and exact scoring code.
The selected fold epochs are **36 / 30 / 30 / 36 / 36**. This is the DIII-D
model; the unrelated MAST prediction archive included in the original upload
is outside this reproduction.

## Setup

Use Python 3.11 on Linux. Allow at least 32 GB RAM for inference (64 GB
recommended), about 35 GB for model artifacts, and additional space for input
data and output arrays. The original fits requested 96 GiB and a V100S GPU.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python fec1.py verify-source
.venv/bin/python -m pytest -q
```

`requirements.txt` pins the environment used for reproduction checks;
`repro/environment.json` records the complete installed package inventory.
It is not a recovered lockfile from September's training jobs. PyTorch's CUDA
build in that environment is `2.5.1+cu124`; CPU inference is also supported.

## Install the retained weights

The ten exact model artifacts are bundled as losslessly compressed Zstandard
parts under `weights/`, tracked with **Git LFS**. Install Git LFS before cloning
on a new machine, then fetch and restore the models:

```bash
git lfs install
git lfs pull
.venv/bin/python fec1.py unpack-artifacts
```

Restoration checks every compressed part and verifies the SHA-256 of every
restored file against [`repro/artifacts.json`](repro/artifacts.json). It writes
ordinary original `.joblib` and `.pt` files to the ignored
`artifacts/incumbent/` directory. Use `--models /another/fresh/directory` to
restore elsewhere; existing installations are never overwritten.

Compression changes storage and transfer size only. Unpack once before
inference: the model code, predictions, loading behavior and RAM requirements
then match the original files. Allow 36.9 GB for the restored models in addition
to the compressed bundle. The pinned `zstandard` dependency is in requirements.

This existing checkout has a verified Git LFS helper installed locally in its
Git metadata, so its ordinary `git push` uploads the compressed objects. That
installation does not change global Git configuration. Other machines should
install [Git LFS](https://git-lfs.com/) normally. Sufficient GitHub LFS storage
and bandwidth are required; no model files are uploaded by the packaging job.

As an alternative, import the retained originals from an existing starter checkout:

```bash
.venv/bin/python fec1.py import-artifacts --source /path/to/fusion-equilibrium-challenge-starter --copy
.venv/bin/python fec1.py verify-artifacts
```

Omit `--copy` to create local symlinks and avoid duplicating the large files.
Those symlinks depend on the retained source checkout. Use `--copy` for an
independent installation. Import checks all hashes before writing files;
verification reads every artifact and can take several minutes.

See [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) for packaging and the training
recipe. Obtain the official DIII-D parquet inputs separately.

## Generate predictions

Run from anywhere; paths are explicit and the code does not import the parent
starter repository:

```bash
.venv/bin/python fec1.py predict \
  --inputs /path/to/diii_d_public_test \
  --output outputs/public-replay --device cuda --workers 8 --batch-size 128
```

Use `--device cpu` without a GPU. The result is
`outputs/public-replay/diii_d_public_test.npz`, with `shot_NNNN_psirz`,
`shot_NNNN_q95` and `shot_NNNN_betaN` float32 arrays. Shot numbering follows
sorted parquet filenames. Use a fresh output directory for every run.

For a bounded replay, add `--max-shots 1`. Compare with the retained submission:

```bash
.venv/bin/python fec1.py compare \
  outputs/public-replay/diii_d_public_test.npz \
  /path/to/original/diii_d_public_test.npz --report outputs/parity.json
```

Subset comparison is intended only for the **first N sorted input shots**;
arbitrary shot subsets cannot be compared by these positional archive keys.
All five members participate even in a one-shot replay. Numerical arrays are
compared, since ZIP timestamps and compression are not reproducible identifiers.

See [`docs/VALIDATION.md`](docs/VALIDATION.md) for executed checks and limits.
The allocated CPU replay passed all 31 tests and matched all five ensemble
members' combined predictions on the first public shot (240 frames), with
maximum absolute difference `9.54e-7` from the retained submission.
