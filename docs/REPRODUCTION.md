# Model and artifact provenance

Each member first predicts a PCA field, mean/log-RMS coordinates, q95 and betaN
with its nested ExtraTrees carrier. A neural model uses 696 contextual features,
192 masked spatial tokens (five values plus R/Z coordinates), base field and
base scalars. Its spatial attention and width-384 network predict coordinates
and 575 nonconstant coefficients of a 24-by-24 DCT basis on a 65-by-65 grid.
A separately retained neural scalar predictor supplies q95 and betaN.

Before ensembling, apply each member's saved shape, coordinate and scalar
strengths. Renormalize the calibrated shape. Average the five calibrated fields
and scalars with equal weight 0.2 in the original fold order. Scalar strengths
are [1,1]; shape strength is 0.75 for folds 0–3 and 1.0 for fold 4.
Coordinate strengths differ by fold.
The exact values and epochs are in `repro/artifacts.json`.

All-frame deployment preserves four interleaved stride-4 sequences. Compute
the contextual difference and whole-sequence mean separately for each phase.
Computing context across consecutive all-frame samples changes this model.
Public preprocessing reads an explicit projection of permitted input columns
and calls `preprocess_row(..., include_targets=False)`.

## Distributing weights separately

The ten files total 36.9 GB (34.4 GiB). Each tree carrier is about
7.3 GB; each neural checkpoint is about 30 MB and includes its scalar predictor.
The carriers contain three 96-tree forests per fold. The field forest stores
64-output prediction vectors at every tree node; these dense arrays account
for most of the size. Neural checkpoints total only 152 MB across five folds.
Use the manifest hashes as artifact identities. No credentials, prediction
upload tokens, private leaderboard labels or raw challenge data belong in Git.

Create a portable archive after importing or linking the retained artifacts:

```bash
tar -chf artifacts/fec1-incumbent.tar -C artifacts incumbent
```

`-h` dereferences local symlinks. This archive is uncompressed and stays in the
ignored artifact directory. Transfer it through your chosen artifact store,
alongside `repro/artifacts.json`. Alternatively, copy the `incumbent/` directory
directly, preserving its paths and dereferencing symlinks. On a new checkout:

```bash
mkdir -p artifacts
tar -xf /path/to/fec1-incumbent.tar -C artifacts
.venv/bin/python -m fec1 verify-artifacts
```

Only deserialize trusted artifacts. The inference entry point verifies their
SHA-256 hashes before `torch.load` or `joblib.load`.

## Historical training recipe

Exact reproduction of the **deployed model** uses the retained checkpoint
bytes. A new training run is not promised to reproduce those bytes or the
displayed public score: the old environment was not fully locked, GPU kernels
can differ, and the public scorer's hidden labels are unavailable.

This checkout retains the required training implementation and immutable shot
identities. `repro/splits/shots.json` contains the original ordered shot/split
assignments and source frame counts: 5,677 train, 689 development, 675 frozen
audit. `repro/splits/foldN.json` preserves the expanded-training, 64-shot
selection and untouched outer-shot roles plus each row digest. Historical
reports and arguments are under `repro/history/`.

Run the commands below from the repository root in a fresh installation.
Obtain the official labelled parquet files separately, under `data/diii_d_train/`,
with their original filenames. Install the runtime requirements first. The
original training allocation requested 96 GiB RAM and a V100S GPU; the carrier
fits also need substantial RAM and disk space. No retained weights are required
for these rebuild stages.

1. `bash scripts/prepare_training.sh` prepares all frames, a train-only initial
   PCA, stride-4 OOF source rows, spatial inputs, normalized targets, exact
   topology and contour radii. Generated caches go under `data/cache/` and
   intermediate arrays under `artifacts/training/`. Its PCA helper preserves
   sampling, ordering, seeds and PCA arguments while omitting the unrelated
   historical full-frame forest and its dependency on development part files.
   The nested deployment carriers refit their own PCA and trees.
2. For each fold, fit the nested carrier, its pooled scalar-reference model,
   and the final spatial-attention field model, in that order:

   ```bash
   export FEC1_PYTHON="$PWD/.venv/bin/python"
   for fold in 0 1 2 3 4; do
     "$FEC1_PYTHON" -m fec1.training.carriers --fold "$fold" \
       --output "artifacts/training/base_fold$fold"
     "$FEC1_PYTHON" -m fec1.training.fit \
       --data "artifacts/training/base_fold$fold" \
       --output "artifacts/training/scalar_fold$fold" \
       --decoder dct --epochs 36 --batch-size 128 \
       --geometry-weight 0.003 --device cuda
     "$FEC1_PYTHON" -m fec1.training.expanded --fold "$fold" \
       --seed 260905 --selection-shots 64 \
       --output "artifacts/training/data_fold$fold"
     bash scripts/train_fold.sh "$fold"
     ln -s "train_fold$fold" "artifacts/training/dct_fold$fold"
   done
   ```

   Carrier defaults are five outer folds, three inner folds, seed 260905,
   PCA64 and 96 trees. Every correction-training row has an out-of-shot base
   prediction. The first neural fit uses the default pooled encoder and legacy
   geometry loss; its scalar predictor is frozen for the final field fit.
   Expanded training retains 64 selection shots and preserves outer identities.
   `train_fold.sh` records the historical final fit: width 384, spatial attention,
   DCT24, symmetric geometry weight .003, warmup 4, exact scoring every 6 epochs,
   patience 3, batch 128, seed 260905 plus fold. Historical selected epochs were
   36/30/30/36/36; a fresh fit must report its actual selections.
3. Pool each outer-fold prediction once and evaluate development separately:

   ```bash
   .venv/bin/python -m fec1.evaluation --root artifacts/training \
     --output artifacts/evaluation --folds 5 --device cuda \
     --batch-size 128 --score-workers 4
   ```

   The DCT ensemble and equal weights are fixed before opening development
   targets. This historical evaluation must not be treated as a new promotion
   gate or tuned using held labels. The archived OOF result is S .990843,
   R2_psi .999156, R2_qb .989019, D_LCFS .012771, consistency .971159.
4. Generate public predictions from the **newly trained** artifacts:

   ```bash
   .venv/bin/python -m fec1.inference --inputs /path/to/diii_d_public_test \
     --selected-root artifacts/training --base-root artifacts/training \
     --output outputs/retrained-public --device cuda --workers 8 --batch-size 128
   ```

   This lower-level command accepts newly trained artifacts. The `python -m fec1
   predict` facade instead verifies the retained incumbent's exact artifact
   hashes and is intended for replay of that model.

These commands are historical documentation, not a newly launched fit.
Later continuation methods and rejected experiments are not included as
retained improvements. Before any new long field-correction fit, follow the
parent project's reviewed epoch-zero/pilot admission and joint role-1/role-2
gates. Preserve all protected shot identities. This reproduction does not
reopen any scientifically concluded experiment.
