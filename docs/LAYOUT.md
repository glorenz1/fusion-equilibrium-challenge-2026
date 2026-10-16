# Repository and run layout

| Location | Responsibility |
| --- | --- |
| `fec1/cli.py`, `pipeline.py`, `download.py` | Raw-data download and the complete standalone workflow |
| `fec1/model.py`, `basis.py`, `carrier.py` | Neural correction, calibration, DCT basis and tree carriers |
| `fec1/data/` | Input projection, repairs, magnetic features and preprocessing |
| `fec1/training/` | PCA, OOF source rows, nested carriers, targets and neural fits |
| `fec1/inference.py`, `evaluation.py` | Prediction and held-shot evaluation |
| `fec1/scoring/`, `losses/` | Exact scorer, required DIII-D mask and geometry losses |
| `config/` | Pinned upstream dataset revision and preprocessing settings |
| `tests/`, `scripts/` | Unit/integration checks and launch helpers |
| `repro/` | Fixed shot identities, historical provenance and permanent validation evidence |
| `data/`, `runs/`, `outputs/`, `logs/` | Ignored raw inputs, generated runs, predictions and logs |

A normal run owns its cache, preparation arrays and generated models. The
workflow sets its paths using `FEC1_RUN_DIR` in child processes; users supply
`--run`. The unconfigured low-level module defaults remain for compatibility.
Use the top-level commands to obtain run validation, logging and model checksums.

The complete synthetic integration test copies only `fec1/` and `config/` into
a temporary checkout, omitting all prior artifacts, data, research history and
Git metadata. It creates raw parquet and exercises preparation, fitting,
evaluation, prediction, rescoring and resume in that independent copy.

The broad original `my_experiments`, NeuralGS and solver trees are unnecessary.
Required preprocessing helpers and numerical definitions were extracted into
the package. `repro/layout_migration.json`, `legacy_source_manifest.json` and
`math_signatures.json` retain the original migration evidence. Core numerical
checks protect the retained architecture/scoring mathematics; training workflow
and provenance checks are covered separately. `source_manifest.json` protects
the current runtime and fixed inputs to the recipe.

Historical paths in `repro/history` and `repro/artifacts.json` are records of
the original trained model. They are not prerequisites for this workflow.
Legacy `import-artifacts`, `verify-artifacts` and prediction without `--run`
remain available for an owner replaying that historical model; those commands
require retained files. The main README's workflow uses none of them.
