"""Shared locations for source, immutable metadata and generated artifacts."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ["FEC1_RUN_DIR"]).resolve() if os.environ.get("FEC1_RUN_DIR") else None
CACHE = RUN / "cache" if RUN else ROOT / "data/cache"
MANIFEST = RUN / "splits.json" if RUN else ROOT / "repro/splits/shots.json"
CONFIG = ROOT / "config/preprocessing.json"
ARTIFACTS = RUN / "artifacts" if RUN else ROOT / "artifacts/training"
SOURCE = ARTIFACTS / "source_rows.npz"
SPATIAL = ARTIFACTS / "inputs"
SYNTHETIC = bool(
    RUN
    and (RUN / "run.json").exists()
    and json.loads((RUN / "run.json").read_text())["profile"] == "synthetic_smoke"
)
DATA_POLICY = "synthetic_smoke_only" if SYNTHETIC else "real_diii_d_only"
