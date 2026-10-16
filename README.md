# FEC1: reproduce the DIII-D 0.9931 ensemble

This repository preserves the five-fold spatial-attention DCT24 model reported
as **S = 0.9931 on Codabench on 2026-09-29**. Its separate, local pooled outer-OOF
score is **0.990843 on 5,677 training shots**. These scores use different
populations. Public hidden labels are unavailable; the public score is a
historical reference, not a score recomputed by this repository.

The implementation is derived from incumbent source snapshot
`625e6f3c0e67b70f7f9c28526e8e3972b7457b9f` and organized in the `fec1`
package. It includes preprocessing, tree carriers, neural architecture,
calibration, training and exact scoring code. The original source hashes and
symbol-level migration record are retained under `repro/`.
The selected fold epochs are **36 / 30 / 30 / 36 / 36**. This is the DIII-D
model; the unrelated MAST prediction archive included in the original upload
is outside this reproduction.

## Setup

Use Python 3.11 on Linux. Allow at least 32 GB RAM for inference (64 GB
recommended), about 35 GB for model artifacts, and additional space for input
data and output arrays. The original fits requested 96 GiB and a V100S GPU.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m fec1 verify-source
.venv/bin/python -m pytest -q
```

`requirements.txt` pins the environment used for reproduction checks;
`repro/environment.json` records the complete installed package inventory.
It is not a recovered lockfile from September's training jobs. PyTorch's CUDA
build in that environment is `2.5.1+cu124`; CPU inference is also supported.

## Install the retained weights

Git contains code, tests, configurations, hashes, split identities and historical
reports. The approximately 35 GB of binary artifacts are kept in a separate,
ignored directory. A Git push alone does **not** distribute these weights.
No public artifact download URL is currently configured.

Running the retained pretrained ensemble requires these exact ten artifacts;
training is not needed once they are installed. A fresh Git clone alone cannot
run that ensemble. To retrain instead, obtain the official labelled parquet
data, retain the committed shot splits, and follow the full preparation and
training recipe. Retained weights are not required for training from scratch,
but all carrier and scalar-reference stages must be rebuilt before the final
field fits. A new fit is not guaranteed to reproduce the exact checkpoints or
the historical public score.

From the existing starter checkout, import the exact ten files listed in
[`repro/artifacts.json`](repro/artifacts.json):

```bash
.venv/bin/python -m fec1 import-artifacts --source /path/to/fusion-equilibrium-challenge-starter --copy
.venv/bin/python -m fec1 verify-artifacts
```

Omit `--copy` to create local symlinks and avoid duplicating the large files.
Those symlinks depend on the retained source checkout. Use `--copy` for an
independent installation. Import checks all hashes before writing files;
verification reads every artifact and can take several minutes.

For another machine, transfer `artifacts/incumbent/` with symlinks dereferenced,
preserving its paths, then run `verify-artifacts`. See
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) for artifact packaging and the
training recipe. Obtain the official DIII-D parquet inputs separately.

## Generate predictions

Run from the repository root. Input/output paths are explicit, and the code
does not import the parent starter repository:

```bash
.venv/bin/python -m fec1 predict \
  --inputs /path/to/diii_d_public_test \
  --output outputs/public-replay --device cuda --workers 8 --batch-size 128
```

Use `--device cpu` without a GPU. The result is
`outputs/public-replay/diii_d_public_test.npz`, with `shot_NNNN_psirz`,
`shot_NNNN_q95` and `shot_NNNN_betaN` float32 arrays. Shot numbering follows
sorted parquet filenames. Use a fresh output directory for every run.

For a bounded replay, add `--max-shots 1`. Compare with the retained submission:

```bash
.venv/bin/python -m fec1 compare \
  outputs/public-replay/diii_d_public_test.npz \
  /path/to/original/diii_d_public_test.npz --report outputs/parity.json
```

Subset comparison is intended only for the **first N sorted input shots**;
arbitrary shot subsets cannot be compared by these positional archive keys.
All five members participate even in a one-shot replay. Numerical arrays are
compared, since ZIP timestamps and compression are not reproducible identifiers.

See [`docs/VALIDATION.md`](docs/VALIDATION.md) for executed checks and limits.
The reorganized package passed 47 tests and an allocated CPU replay of all
five ensemble members on the first public shot (240 frames). Its maximum
absolute difference from the retained submission was `9.54e-7`.

## Repository layout

`fec1/` contains the model and its required data, training, inference and scoring
code. `config/` holds preprocessing settings; `scripts/` holds launch recipes;
`tests/` holds checks; and `repro/` holds provenance, fixed splits and historical
evidence. Generated data, weights and outputs are ignored. There is no runtime
dependency on `my_experiments`, NeuralGS or the parent checkout. See
[`docs/LAYOUT.md`](docs/LAYOUT.md) for the migration details.
