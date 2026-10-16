# Reproduction validation

The CPU suite passes **28 tests** in the existing Python 3.11 environment.
It covers the frozen source checksums, all-member provenance and calibration,
whole-shot role isolation, complete OOF shot coverage, base-fit label isolation,
zero initialization, missing tokens, calibration, scalar freezing, finite
gradients, symmetric geometry, four-phase deployment context, target-column
projection and artifact/archive corruption detection.

A bounded allocated-node replay is specified in `scripts/replay.sbatch`.
It records the reviewed Git commit and runtime, checks source and clean-tree
preconditions, runs the suite, hashes all ten artifacts, preprocesses the first
sorted public parquet shot, predicts every frame with all five members and
compares the arrays against the retained public submission. Its postflight
checks source identity and Git cleanliness again. Logs and arrays are ignored
by Git; summary evidence is retained under `repro/validation/` when complete.

Numerical parity tolerances are `atol=2e-6`, `rtol=2e-5`, accounting for CPU/GPU
floating-point differences. This is prediction parity, not an estimate of the
public leaderboard score or a new held-label model selection. No retraining,
protected role-1/role-2 scoring or public submission is performed by this check.
