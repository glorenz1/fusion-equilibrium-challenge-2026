# Reproduction validation

The CPU suite passes **31 tests** in the existing Python 3.11 environment.
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
