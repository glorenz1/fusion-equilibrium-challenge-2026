# Reproduction validation

The reorganized package passes **47 tests** in the existing Python 3.11
environment. The original source layout passed 31 tests in the historical replay
below. The current suite also checks package isolation, documented module entry
points, retained mathematical definitions, source frame counts and a synthetic
perfect-prediction replay through the exact scorer.
It covers the frozen source checksums, all-member provenance and calibration,
whole-shot role isolation, complete OOF shot coverage, base-fit label isolation,
zero initialization, missing tokens, calibration, scalar freezing, finite
gradients, symmetric geometry, four-phase deployment context, target-column
projection, artifact import/copy behavior, the historical PCA initialization,
and artifact/archive corruption detection.

A bounded allocated-node replay is specified in `scripts/replay.sbatch`.
It records the reviewed Git commit and runtime, checks source and clean-tree
preconditions, runs the suite, hashes all ten artifacts, preprocesses the first
sorted public parquet shot, predicts every frame with all five members and
compares the arrays against the retained public submission. Its postflight
checks source identity and Git cleanliness again. Logs and arrays are ignored
by Git; permanent evidence is retained under `repro/validation/`.

## Reorganized package replay

Slurm job **6417606** completed with exit code **0:0** in **5:30** on CPU node
`clr1450`, using the same four-CPU, 32 GiB allocation. Peak RSS was
12,053,044 KiB. The reviewed code commit was
`14f612115db6971664ff629c0eff8595f48aa4e2`.

All 47 protected source files matched before and after execution, the Git tree
was clean at both checks, all ten artifact hashes passed, and all **47 tests**
passed in 84.80 seconds. All five retained members predicted all 240 frames of
`d3d_shot_0047b7837b.parquet`. psirz, q95 and betaN passed numerical comparison
against the retained public archive, with maximum absolute error
**9.5367431640625e-07**. This covers one of 874 public shots. A separate syntax-tree
audit confirms that 67 retained mathematical definitions match their originals,
excluding relocation-only imports, documentation and a renamed configuration
annotation.

A direct comparison with the earlier layout replay also passed (maximum
absolute error 5.96e-08 across the different CPU nodes).

Permanent launch details, runtime, raw log, comparison results and decisions
are in `repro/validation/layout_20261016/`. The earlier layout's evidence below
remains intact. Neither replay trains a new model or recomputes the public score.

## Executed allocated-node replay

Slurm job **6417455** completed successfully with exit code **0:0** in **4:59**
on CPU node `il1227`. Effective allocation: short partition/QOS, four requested
CPUs, 32 GiB, 15 minutes. Peak reported RSS was 8,208,272 KiB. The reviewed
code commit was `a14ed8004e7adbbca04b82b7d66df1ded9c7f12b`.

| Check | Result |
| --- | --- |
| Frozen source | All 105 files matched before and after replay |
| Git source tree | Clean before and after replay |
| Model artifacts | All ten full-file checksums matched |
| CPU tests | 31 passed in 87.61 seconds |
| Raw input | `d3d_shot_0047b7837b.parquet`, 240 frames |
| Ensemble | All five retained members, original calibrations and order |
| Outputs compared | psirz, q95, betaN; all arrays passed |
| Maximum absolute difference | 0.00000095367431640625 |
| Original reference | SHA-256 matched the original deployment bundle audit |

This replay covers **one of 874 public shots**, with every frame of that shot.
It is an end-to-end smoke replay, not a complete public-archive regeneration.
The full generation command is in the README; omit `--max-shots` to process
the entire input directory. Retained checkpoint bytes, immutable source hashes
and public submission provenance are also preserved in the manifests.

`repro/validation/report.json` records decisions and scope; `environment.json`,
`replay.log`, `parity.json`, `submission_reference.json` and `commands.json`
preserve the runtime, raw output, comparison and launch details. Two earlier
submissions were canceled while pending, before execution, and are recorded.

To repeat on this cluster, from the repository root:

```bash
mkdir -p logs
export FEC1_PYTHON="$PWD/.venv/bin/python"
export FEC1_REVIEWED_COMMIT="$(git rev-parse HEAD)"
export FEC1_INPUTS=/path/to/diii_d_public_test
export FEC1_REFERENCE=/path/to/retained/diii_d_public_test.npz
sbatch --export=ALL scripts/replay.sbatch
```

Use a clean tracked source tree. The artifact directory, logs and outputs are
ignored. For a full archive replay, set `FEC1_MAX_SHOTS=0` and request a longer
time limit with `sbatch --time=...` appropriate to your hardware.

Numerical parity tolerances are `atol=2e-6`, `rtol=2e-5`, accounting for CPU/GPU
floating-point differences. This is prediction parity, not an estimate of the
public leaderboard score or a new held-label model selection. No retraining,
protected role-1/role-2 scoring or public submission is performed by this check.
