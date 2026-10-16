import ast
import hashlib
import importlib
import json
import pkgutil
import sys

import pytest

import fec1
from fec1.paths import ROOT


class CanonicalMath(ast.NodeTransformer):
    """Ignore relocation-only imports, documentation and a renamed annotation."""

    def visit_ImportFrom(self, node):
        return None

    def visit_Import(self, node):
        return None

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "PreprocessingConfig":
            node.id = "NeuralGSConfig"
        return node


def test_retained_core_definitions_match_original_mathematics():
    record = json.loads((ROOT / "repro/math_signatures.json").read_text())
    assert len(record["checks"]) == 67
    for check in record["checks"]:
        tree = ast.parse((ROOT / check["path"]).read_text())
        definition = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == check["symbol"]
        )
        canonical = ast.dump(CanonicalMath().visit(definition), include_attributes=False)
        assert hashlib.sha256(canonical.encode()).hexdigest() == check["canonical_sha256"], check


def test_package_imports_without_legacy_model_trees():
    for module in pkgutil.walk_packages(fec1.__path__, prefix="fec1."):
        if module.name != "fec1.__main__":
            importlib.import_module(module.name)
    for name in ["my_experiments", "neuralgs", "FEC1", "src"]:
        assert name not in sys.modules
        assert not (ROOT / name).exists()


def test_shot_metadata_includes_required_frame_counts():
    records = json.loads((ROOT / "repro/splits/shots.json").read_text())["records"]
    assert len(records) == 7041
    assert all(
        isinstance(r["source_frame_count"], int) and r["source_frame_count"] > 0 for r in records
    )
    assert (
        sum((r["source_frame_count"] + 3) // 4 for r in records if r["split"] == "train") == 316200
    )


@pytest.mark.parametrize(
    "module",
    [
        "fec1.data.cache",
        "fec1.training.carriers",
        "fec1.training.source_rows",
        "fec1.training.inputs",
        "fec1.training.targets",
        "fec1.training.topology",
        "fec1.training.contours",
        "fec1.training.expanded",
        "fec1.training.fit",
        "fec1.inference",
        "fec1.evaluation",
    ],
)
def test_documented_entry_points_can_start_without_data(module, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit) as result:
        importlib.import_module(module).main()
    assert result.value.code == 0


@pytest.mark.parametrize("loss", ["legacy", "symmetric"])
def test_original_training_modes_reach_data_loading(tmp_path, monkeypatch, loss):
    from fec1.training.fit import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit",
            "--data",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "output"),
            "--decoder",
            "dct",
            "--loss-version",
            loss,
        ],
    )
    with pytest.raises(FileNotFoundError, match="ready.json"):
        main()


def test_relocated_official_scorer_scores_a_perfect_synthetic_prediction():
    import numpy as np
    from fec1.scoring.archive import _parallel_official_score

    with np.load(ROOT / "fec1/scoring/masks/d3d_envelope.npz") as mask:
        r, z = mask["grid_R"], mask["grid_Z"]
    rr, zz = np.meshgrid(r, z, indexing="xy")
    fields = []
    for i in range(3):
        normalized = ((rr - 1.7 - 0.015 * i) / (0.45 + 0.005 * i)) ** 2
        normalized += ((zz - 0.05 - 0.01 * i) / (0.7 + 0.01 * i)) ** 2
        fields.append(0.15 * (1 - normalized))
    field = np.asarray(fields, dtype=np.float32)
    q95 = np.asarray([2.0, 2.2, 2.4], dtype=np.float32)
    beta = np.asarray([0.7, 0.8, 0.9], dtype=np.float32)
    result = _parallel_official_score(
        dict(
            field=field,
            target_field=field.copy(),
            q95=q95,
            target_q95=q95.copy(),
            betaN=beta,
            target_betaN=beta.copy(),
            shot_index=np.zeros(3, dtype=int),
            shot_name=np.asarray(["synthetic"] * 3),
            source_index=np.arange(3),
            reference_source="synthetic-test-only",
        ),
        workers=1,
    )
    assert result["S"] == 1.0
    assert result["r2_psi"] == 1.0
    assert result["lcfs_fail_frac"] == 0.0
