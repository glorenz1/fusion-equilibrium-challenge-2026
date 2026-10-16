"""Shared locations for source, immutable metadata and generated artifacts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
MANIFEST = ROOT / "repro/splits/shots.json"
CONFIG = ROOT / "config/preprocessing.json"
ARTIFACTS = ROOT / "artifacts/training"
SOURCE = ARTIFACTS / "source_rows.npz"
SPATIAL = ARTIFACTS / "inputs"
