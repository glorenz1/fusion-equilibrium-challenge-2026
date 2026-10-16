# Validation

## Standalone workflow: 2026-10-16

Slurm job **6421846** passed on CPU node `il1227` at reviewed commit
`af844c3d64e62218f56758fcb277ab659d5aebb6`. It completed in **1:39** with exit
code **0:0**, using four requested CPUs and 16 GiB. Peak RSS was 981,732 KiB.
All 59 protected source/config/identity files matched before and after execution,
and the Git tree was clean at both checks.

| Check | Result |
| --- | --- |
| Unit and regression suite | 56 passed in 3.56 seconds |
| Complete fresh-checkout integration | 1 passed in 91.32 seconds |
| Independent copy | Only `fec1/` and `config/`; no retained artifacts, data, research history or Git metadata |
| Raw test data | 23 newly generated synthetic parquet shots: 20 train, 3 development |
| Preparation | All seven stages completed, producing 40 training rows |
| Training | Five carriers and ten neural fits from scratch; one optimizer update per neural fit |
| Generated deployment model | All ten newly generated files hashed and verified |
| Evaluation and prediction | Complete held-shot metrics and five-member public-format prediction completed |
| Independent scoring | Exactly matched the development score reported by evaluation |
| Resume | Skipped completed stages; every deployment file retained its modification time |
| Real dataset acquisition | All 7,041 upstream training identities matched the committed split; one 11 MB unlabelled public file downloaded and SHA-256 verified |

This is a software integration check on reduced synthetic models. It does not
estimate challenge performance. The complete 7,041-shot real-data fit has **not**
been rerun here, and the public score has not been recomputed.

The run record, model manifest (hashes only), exact commands, environment,
raw stage logs, synthetic evaluation/rescoring results and download probe are
preserved under `repro/validation/scratch_20261016/`. Generated binary artifacts
remain ignored in `outputs/`; none are required by a fresh clone.

The first integration attempt, job 6421824 at `eaaa83f`, passed all 56 unit
tests and all preparation/training/evaluation stages, then failed because the
synthetic fixture omitted `coil_input_column`. The fix adds the field, checks
the schema before fitting, and preserves partial smoke predictions before a
retry. The failed attempt's logs and outcome remain recorded. This was an
implementation failure in a software fixture, not a scientific model result.

Run the quick suite with `python -m pytest -q`; run the full synthetic check with
`python -m pytest -q -m integration`. `scripts/validate.sbatch` runs both with a
clean pinned-commit preflight and postflight.

## Historical retained-weight checks

The earlier replay records below preserve evidence for the original trained
model and previous layouts. They do not require a new user to possess those
weights to run the standalone training workflow.

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
