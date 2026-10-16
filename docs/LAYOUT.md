# Package layout and migration

The runtime and training implementation live in one Python package:

| Location | Responsibility |
| --- | --- |
| `fec1/model.py`, `basis.py`, `carrier.py` | Neural correction, calibration, DCT basis and nested tree carriers |
| `fec1/data/` | Input projection, repairs, magnetic features, preprocessing and cache preparation |
| `fec1/training/` | Source rows, nested carriers, targets, expanded folds and neural fits |
| `fec1/inference.py`, `evaluation.py` | Public prediction and frozen local evaluation |
| `fec1/scoring/` | Exact DIII-D scorer, vendored scorer primitives and required mask |
| `fec1/losses/` | Normalization and the original geometry objectives |
| `config/` | The three settings actually consumed by preprocessing |
| `scripts/`, `tests/` | Launch recipes and regression checks |
| `repro/` | Artifact/source hashes, immutable splits, migration records and historical evidence |
| `data/`, `artifacts/`, `outputs/`, `logs/` | Ignored local inputs, model binaries, caches and run outputs |

The original broad source copy imported unrelated encoder, solver, experiment
and MAST model code indirectly through shared helpers. Those trees have been
removed. Necessary functions from `my_experiments` now have descriptive modules
in `fec1/`. The NeuralGS name originally concealed shared preprocessing:
magnetic-input rendering, token construction, input repairs and scalar feature
construction. Those helpers are retained in `fec1/data/`; the NeuralGS solver,
network and training system are unnecessary for this ensemble.

`python -m fec1` is the main entry point. Root `fec1.py` remains a small
compatibility wrapper. Run preparation and training modules with `python -m`
from the repository root; launch scripts set that working directory themselves.
Runtime imports do not depend on the parent checkout or its experiment trees.

`repro/layout_migration.json` maps retained code to its original sources.
`repro/legacy_source_manifest.json` preserves the old source hashes;
`repro/source_manifest.json` protects the current runtime files.
`repro/math_signatures.json` checks 67 retained definitions against their
original syntax trees after excluding imports, documentation and a renamed
configuration annotation. This supplements numerical prediction replay; it
does not substitute for retraining the complete pipeline.

Historical paths remain in provenance reports and `repro/artifacts.json`:
its `source` paths locate the existing uncompressed files when importing from
the original starter checkout. These are records and import locations, not
runtime module dependencies. Historical research evidence is retained.

This reorganization changes paths and removes unused branches, including an
unrelated implicit-scorer training mode and experimental family selection.
It preserves the retained model's mathematics, checkpoint keys, calibration,
ensemble order and shot identities. The training split manifest now also
retains original frame counts required by topology extraction. No new model
was trained or selected as part of this cleanup. Weights and carriers remain
uncompressed, ignored and uncommitted.
