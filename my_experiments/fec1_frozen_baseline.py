"""Immutable identity checks for the selected FEC1 development baseline."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "my_experiments/candidates/fec1_oof_temporal_blend"
BASELINE_REPORT = BASELINE_DIR / "fec1_oof_temporal_blend_crossvalidated.json"
BASELINE_PREDICTION = BASELINE_DIR / "fec1_oof_temporal_blend_full_development.npz"
BASELINE_REPORT_SHA256 = "2704a8bb84c3193f88e48f4e6345e7532c5d045a313e3a81947db6317e715842"
BASELINE_PREDICTION_SHA256 = "8d5a2d51f2c29068740364732d97752124f6122b07fa674f50f55f115bc4323d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected_path: Path, expected_sha256: str) -> None:
    if path.resolve() != expected_path.resolve():
        raise RuntimeError(f"expected frozen FEC1 baseline {expected_path}, got {path}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"frozen FEC1 baseline hash mismatch for {path}: {actual}"
        )


def verify_frozen_baseline_report(path: Path) -> None:
    _verify(path, BASELINE_REPORT, BASELINE_REPORT_SHA256)


def verify_frozen_baseline_prediction_if_selected(path: Path) -> None:
    if path.resolve() == BASELINE_PREDICTION.resolve():
        _verify(path, BASELINE_PREDICTION, BASELINE_PREDICTION_SHA256)
