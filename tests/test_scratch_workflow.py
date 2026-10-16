import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fec1 import pipeline
from fec1.cli import compare_archives
from fec1.download import download_file
from fec1.paths import ROOT
from fec1.provenance import sha256, verify_records
from fec1.scoring.archive import validate_development_predictions


def test_download_verifies_contents_and_never_replaces_existing_corruption(tmp_path):
    source = tmp_path / "source.parquet"
    destination = tmp_path / "downloaded.parquet"
    source.write_bytes(b"raw-shot")
    task = (source.as_uri(), destination, source.stat().st_size, sha256(source))
    download_file(task)
    assert destination.read_bytes() == source.read_bytes()
    destination.write_bytes(b"modified")
    with pytest.raises(ValueError, match="existing file differs"):
        download_file(task)
    destination.unlink()
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_file((*task[:3], "0" * 64))
    assert not destination.exists()
    assert not destination.with_suffix(".parquet.part").exists()


def test_comparison_rejects_missing_outputs_and_nonprefix_shots(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    x = np.ones(2, dtype=np.float32)
    np.savez(a, shot_0000_q95=x)
    np.savez(
        b,
        shot_0000_q95=x,
        shot_0000_betaN=x,
        shot_0000_psirz=np.zeros((2, 65, 65), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="complete"):
        compare_archives(a, b)
    values = {key.replace("0000", "0001"): value for key, value in np.load(b).items()}
    np.savez(a, **values)
    np.savez(b, **values)
    with pytest.raises(ValueError, match="first N"):
        compare_archives(a, b)


def test_frozen_calibration_changes_supervised_loss_before_geometry():
    from fec1.model import StructuredCorrection
    from fec1.training.fit import objective

    torch.manual_seed(2)
    model = StructuredCorrection(6, order=4, width=32, channels=8)
    base, scalar = torch.randn(2, 65, 65), torch.ones(2, 2)
    inputs = (
        torch.randn(2, 6),
        base,
        scalar,
        torch.randn(2, 7, 5),
        torch.randn(2, 7, 2),
        torch.ones(2, 7, dtype=torch.bool),
    )
    output = model(*inputs)
    output["field"] = output["field"] + 0.25
    output["coordinate"] = output["coordinate"] + torch.tensor([0.25, 0.0])
    one = dict(shape=1.0, coordinate=[1.0, 1.0], scalar=[1.0, 1.0])
    zero = dict(shape=0.0, coordinate=[0.0, 0.0], scalar=[0.0, 0.0])
    common = dict(base_field=base, base_scalar=scalar, fixed_scalars=True)
    _, identity = objective(
        model, output, output["field"].detach(), scalar, frozen_strengths=one, **common
    )
    loss, unchanged_carrier = objective(
        model, output, output["field"].detach(), scalar, frozen_strengths=zero, **common
    )
    assert unchanged_carrier["coordinate"] > identity["coordinate"] + 0.01
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def prediction_fixture():
    return dict(
        sample_index=np.arange(2),
        frame_index=np.arange(2),
        shot_name=np.asarray(["development"] * 2),
        psirz=np.zeros((2, 65, 65), dtype=np.float32),
        q95=np.ones(2, dtype=np.float32),
        betaN=np.ones(2, dtype=np.float32),
    )


@pytest.mark.parametrize("change", ["train", "duplicate_frame", "missing_frame", "nonfinite"])
def test_local_scoring_rejects_invalid_or_nonheld_predictions(change):
    records = [
        dict(shot="development", split="development", source_frame_count=2),
        dict(shot="train", split="train", source_frame_count=2),
    ]
    prediction = prediction_fixture()
    validate_development_predictions(prediction, records)
    if change == "train":
        prediction["shot_name"] = np.asarray(["train"] * 2)
    elif change == "duplicate_frame":
        prediction["frame_index"][:] = 0
    elif change == "missing_frame":
        prediction = {key: value[:1] for key, value in prediction.items()}
    else:
        prediction["q95"][0] = np.nan
    with pytest.raises(ValueError):
        validate_development_predictions(prediction, records)


def initialize_test_run(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "environment", lambda: {})
    raw, run = tmp_path / "raw", tmp_path / "run"
    raw.mkdir()
    (raw / "shot.parquet").write_bytes(b"input")
    manifest = {"records": [dict(shot="shot", split="train", source_frame_count=8)]}
    with pipeline.run_lock(run):
        pipeline.initialize(run, raw, smoke_manifest=manifest)
    return raw, run


def test_resume_rejects_changed_raw_input_contents(tmp_path, monkeypatch):
    raw, run = initialize_test_run(tmp_path, monkeypatch)
    (raw / "shot.parquet").write_bytes(b"other")
    with pytest.raises(ValueError, match="changed file"):
        pipeline.initialize(run, raw, resume=True)


def test_failed_stage_preserves_partial_outputs_before_retry(tmp_path, monkeypatch):
    _, run = initialize_test_run(tmp_path, monkeypatch)

    def fail(*args, **kwargs):
        (run / "result").write_bytes(b"partial")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(pipeline.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="failed"):
        pipeline.stage(run, "example", "irrelevant", [], ["result"], 1, False)

    def success(*args, **kwargs):
        (run / "result").write_bytes(b"complete")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, "run", success)
    pipeline.stage(run, "example", "irrelevant", [], ["result"], 1, True)
    assert (run / "failed/example-1/result").read_bytes() == b"partial"
    assert (run / "result").read_bytes() == b"complete"
    assert json.loads((run / "run.json").read_text())["stages"]["example"]["attempt"] == 2


@pytest.mark.integration
def test_fresh_checkout_trains_predicts_and_scores_without_retained_artifacts(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    # Deliberately omit repro/, scripts/, artifacts/, data/ and the original checkout's Git history.
    for name in ("fec1", "config"):
        shutil.copytree(ROOT / name, checkout / name, ignore=shutil.ignore_patterns("__pycache__"))
    run = tmp_path / "run"
    command = [sys.executable, "-B", "-m", "fec1", "smoke", "--run", str(run), "--workers", "1"]
    log = tmp_path / "smoke.log"
    with log.open("w") as output:
        completed = subprocess.run(
            command, cwd=checkout, stdout=output, stderr=subprocess.STDOUT, timeout=1800
        )
    assert completed.returncode == 0, log.read_text()[-18000:]
    record = json.loads((run / "run.json").read_text())
    assert record["profile"] == "synthetic_smoke"
    assert len(record["stages"]) == 28  # seven preparation, four per fold, one evaluation
    assert all(stage["status"] == "complete" for stage in record["stages"].values())
    model = json.loads((run / "model.json").read_text())
    assert len(model["artifacts"]) == 10
    verify_records(run / "artifacts", model["artifacts"])
    assert (run / "public-smoke/diii_d_public_test.npz").is_file()
    result = json.loads((run / "evaluation/result.json").read_text())
    rescored = json.loads((run / "rescored.json").read_text())
    assert result["development_score"] == rescored["score"]
    assert rescored["data_policy"] == "synthetic_smoke_only"
    assert result["public_score_recomputed"] is False
    timestamps = {
        r["path"]: (run / "artifacts" / r["path"]).stat().st_mtime_ns for r in model["artifacts"]
    }
    with log.open("a") as output:
        resumed = subprocess.run(
            [*command, "--resume"],
            cwd=checkout,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    assert resumed.returncode == 0, log.read_text()[-18000:]
    assert timestamps == {
        r["path"]: (run / "artifacts" / r["path"]).stat().st_mtime_ns for r in model["artifacts"]
    }
