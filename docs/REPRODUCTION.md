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
are [1,1] and shape strength is 0.75; coordinate strengths differ by fold.
The exact values and epochs are in `repro/artifacts.json`.

All-frame deployment preserves four interleaved stride-4 sequences. Compute
the contextual difference and whole-sequence mean separately for each phase.
Computing context across consecutive all-frame samples changes this model.
Public preprocessing reads an explicit projection of permitted input columns
and calls `preprocess_row(..., include_targets=False)`.

## Distributing weights separately

The ten files have a total size of about 35 GB. Each tree carrier is about
7.3 GB; each neural checkpoint is about 30 MB and includes its scalar predictor.
Use the manifest hashes as artifact identities. No credentials, prediction
upload tokens, private leaderboard labels or raw challenge data belong in Git.

Create a portable archive after importing or linking the retained artifacts:

```bash
tar -chf fec1-incumbent.tar -C artifacts incumbent
```

`-h` dereferences local symlinks. Transfer the archive through your chosen
artifact store, alongside `repro/artifacts.json`. On a new checkout:

```bash
mkdir -p artifacts
tar -xf /path/to/fec1-incumbent.tar -C artifacts
.venv/bin/python fec1.py verify-artifacts
```

Only deserialize trusted artifacts. The inference entry point verifies their
SHA-256 hashes before `torch.load` or `joblib.load`.

## Historical training recipe

Exact reproduction of the **deployed model** uses the retained checkpoint
bytes. A new training run is not promised to reproduce those bytes or the
displayed public score: the old environment was not fully locked, GPU kernels
can differ, and the public scorer's hidden labels are unavailable.

This checkout includes the original training implementation and all immutable
shot identities. `my_experiments/causal_hybrid_encoder_v2_dataset/manifest.json`
contains the original ordered shot/split assignments, projected to the two
fields consumed by `_split_records`: 5,677 train, 689 development, 675 frozen
audit. `repro/splits/foldN.json` preserves the actual expanded-training,
64-shot selection and untouched outer-shot roles plus each row digest.
Historical reports and arguments are under `repro/history/`.

The complete pipeline, in dependency order, is:

1. Prepare every released DIII-D training frame with `train_neuralgs.py prepare
   --input-dir data/diii_d_train --output-dir
   my_experiments/neuralgs_real_d3d_full_frame_cache --frames-per-shot 0
   --workers 16 --uncompressed`. The frozen configuration is committed beside
   that cache location. Data must retain the original input filenames.
2. Run `scripts/prepare_pca.py --output
   my_experiments/artifacts/fec1_full_frame_pca64_extra_trees.joblib` to create
   the initial train-only PCA cache, then
   `my_experiments/generate_fec1_train_oof_base.py --folds 5 --workers 8
   --train-stride 4 --estimators 96 --min-samples-leaf 4 --max-features 0.7
   --output my_experiments/artifacts/fec1_train_stride4_oof_base.npz`.
   The helper preserves the original PCA sampling, ordering, seeds and fit
   arguments while omitting the old unrelated full-frame tree fit and its
   post-fit dependency on development part files. The source-row builders
   consume only this initial PCA; nested deployment carriers refit their own
   PCA and trees. No deployed prediction depends on the omitted models.
3. Materialize spatial inputs with `prepare_fec1_topology_native_inputs.py`,
   normalized targets with `prepare_fec1_score_aligned_targets.py`, exact
   topology with `extract_fec1_train_exact_topology.py --all-train-stride 4`,
   and contour radii with `extract_fec1_exact_contour_radius.py --angles 64
   --contour-points 128`. `scripts/prepare_training.sh` supplies the exact paths.
4. Fit five nested bases with `prepare_fec1_structured_crossfit.py --fold N
   --output my_experiments/candidates/fec1_structured_crossfit_v1/base_foldN`.
   Defaults: five outer folds, three inner folds, seed 260905, PCA64,
   96 trees. Inner predictions are out-of-shot for every correction-training
   row; outer shots are excluded from every corresponding fit.
5. Fit the original DCT scalar-reference model for each nested base with
   `train_fec1_structured_correction.py --decoder dct --epochs 36 --batch-size
   128 --geometry-weight 0.003 --device cuda`, using that base and output
   `my_experiments/candidates/fec1_structured_crossfit_v1/dct_foldN`.
   Its default encoder is pooled and its default geometry loss is legacy.
   These scalar-reference weights are frozen for the final field fits.
6. `prepare_fec1_dct24_more_train_20260929.py --fold N --seed 260905
   --selection-shots 64 --output
   my_experiments/candidates/fec1_dct24_more_train_20260929/data_foldN`
   moves formerly unused inner selection shots into fitting, preserves
   out-of-sample base predictions, and writes `scalar_reference.pt`.
7. Run the final fit with the arguments recorded in each
   `repro/history/foldN/result.json`. `scripts/train_fold.sh N` expresses this
   historical command. This fits a fresh spatial-attention DCT24 field model
   for up to 36 epochs, symmetric geometry loss weight .003, warmup 4,
   exact scoring every 6 epochs, patience 3, batch 128, seed 260905+N.
   The scalar predictor remains fixed. Epochs 36/30/30/36/36 were selected.
8. Alias `base_foldN` to `data_foldN` and `dct_foldN` to `train_foldN`, then
   run `evaluate_fec1_structured_crossfit.py --root ROOT --output ROOT/evaluation
   --folds 5 --families dct --device cuda --batch-size 128 --score-workers 4`.
   This pools exactly one outer-fold prediction per training row, freezes the
   ensemble and evaluates development separately. It must not be treated as
   a new promotion gate or tuned using held labels. The archived OOF result
   is S .990843, R2_psi .999156, R2_qb .989019, D_LCFS .012771,
   consistency .971159.

These commands are historical documentation, not a newly launched fit.
Later continuation methods and rejected experiments are not included as
retained improvements. Before any new long field-correction fit, follow the
parent project's reviewed epoch-zero/pilot admission and joint role-1/role-2
gates. Preserve all protected shot identities. This reproduction does not
reopen any scientifically concluded experiment.
