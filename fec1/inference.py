#!/usr/bin/env python3
"""Export the frozen five-fold DCT24 FEC1 ensemble on DIII-D public inputs."""

from __future__ import annotations

from fec1.paths import ROOT

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
import os
from pathlib import Path
import zipfile

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


from fec1.data.projection import safe_input_columns
from fec1.model import (
    FrozenScalarPredictor,
    StructuredCorrection,
    calibrated_prediction,
)
from fec1.data.features import _features
from fec1.carrier import diagnostic_context, predict_base
from fec1.data.config import PreprocessingConfig
from fec1.data.records import preprocess_row

SELECTED = ROOT / "artifacts/incumbent"
BASES = ROOT / "artifacts/incumbent"
CONFIG = ROOT / "config/preprocessing.json"
PUBLIC = ROOT / "data/diii_d_public_test"


def _read_public(path: Path) -> dict:
    columns = safe_input_columns(tuple(pq.ParquetFile(path).schema_arrow.names))
    frame = pd.read_parquet(path, columns=list(columns))
    if len(frame) != 1:
        raise RuntimeError(f"expected one shot in {path}")
    return frame.iloc[0].to_dict()


def prepare_one(task: tuple[str, str, dict]) -> tuple[str, int]:
    source, destination, config_dict = task
    path = Path(source)
    out = Path(destination)
    if out.is_file():
        with np.load(out, allow_pickle=False) as cached:
            if cached["shot_name"].item() == path.stem:
                return path.name, len(cached["features"])
        raise RuntimeError(f"stale public input cache: {out}")
    config = PreprocessingConfig.from_dict(config_dict)
    prepared = preprocess_row(_read_public(path), config, include_targets=False)
    feature = _features(prepared)
    arrays = {
        "shot_name": np.asarray(path.stem),
        "features": feature,
        "token_values": np.nan_to_num(
            prepared["token_values"], nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32),
        "token_coords": np.nan_to_num(
            prepared["token_coords"], nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32),
        "token_mask": np.asarray(prepared["token_mask"], dtype=bool),
    }
    if feature.shape[1] * 3 != 696 or arrays["token_values"].shape[1:] != (192, 5):
        raise RuntimeError(f"DCT input schema changed for {path.name}")
    if not np.isfinite(feature).all():
        raise RuntimeError(f"non-finite features for {path.name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, out)
    return path.name, len(feature)


def prepare(paths: list[Path], cache: Path, workers: int) -> None:
    metadata = json.loads(CONFIG.read_text())
    config = metadata["config"]
    tasks = [(str(p), str(cache / f"{p.stem}.npz"), config) for p in paths]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (name, count) in enumerate(pool.map(prepare_one, tasks), 1):
            if index == 1 or index % 25 == 0 or index == len(paths):
                print(
                    json.dumps(
                        {
                            "stage": "prepare",
                            "shots": index,
                            "total": len(paths),
                            "last": name,
                            "frames": count,
                        }
                    ),
                    flush=True,
                )


def _context(features: np.ndarray) -> np.ndarray:
    context = np.empty((len(features), features.shape[1] * 3), dtype=np.float32)
    for phase in range(4):
        rows = np.arange(phase, len(features), 4)
        if len(rows):
            context[rows] = diagnostic_context(features[rows])
    return context


def _load_model(fold: int, device: torch.device, selected_root: Path):
    checkpoint = torch.load(
        selected_root / f"dct_fold{fold}/checkpoint.pt", map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("smoke_only")
        or not checkpoint.get("nested_shot_crossfit")
        or checkpoint.get("development_used")
        or checkpoint.get("frozen_audit_used")
        or checkpoint["model_config"]["decoder"] != "dct"
    ):
        raise RuntimeError(f"unsafe DCT checkpoint in fold {fold}")
    model = StructuredCorrection(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    fixed = FrozenScalarPredictor(checkpoint["fixed_scalar_checkpoint"]).to(device)
    return checkpoint, model, fixed


def _write_array(archive: zipfile.ZipFile, name: str, values: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    archive.writestr(f"{name}.npy", buffer.getvalue())


def predict(
    paths: list[Path],
    cache: Path,
    output: Path,
    *,
    batch_size: int,
    device_name: str,
    workers: int,
    selected_root: Path,
    base_root: Path,
) -> None:
    device = torch.device(device_name)
    torch.set_num_threads(max(1, min(workers, 8)))
    counts = []
    for path in paths:
        with np.load(cache / f"{path.stem}.npz", allow_pickle=False) as saved:
            counts.append(len(saved["features"]))
    starts = np.concatenate(([0], np.cumsum(counts)))
    fields_path = output / "ensemble_field.npy"
    scalars_path = output / "ensemble_scalar.npy"
    if fields_path.exists() or scalars_path.exists():
        raise FileExistsError("ensemble accumulators already exist; use a fresh output directory")
    field_sum = np.lib.format.open_memmap(
        fields_path, mode="w+", dtype=np.float32, shape=(int(starts[-1]), 65, 65)
    )
    scalar_sum = np.lib.format.open_memmap(
        scalars_path, mode="w+", dtype=np.float32, shape=(int(starts[-1]), 2)
    )
    field_sum[:] = 0
    scalar_sum[:] = 0
    for fold in range(5):
        checkpoint, model, fixed_scalar = _load_model(fold, device, selected_root)
        base = joblib.load(base_root / f"base_fold{fold}/base.joblib")
        if base.get("development_used") or base.get("frozen_audit_used"):
            raise RuntimeError(f"unsafe base checkpoint in fold {fold}")
        for shot_index, path in enumerate(paths):
            with np.load(cache / f"{path.stem}.npz", allow_pickle=False) as saved:
                feature = saved["features"].astype(np.float32)
                tokens = saved["token_values"].astype(np.float32)
                coords = saved["token_coords"].astype(np.float32)
                mask = saved["token_mask"].astype(bool)
            base_field, base_scalar = predict_base(base, feature)
            context = _context(feature)
            start = int(starts[shot_index])
            with torch.inference_mode():
                for offset in range(0, len(feature), batch_size):
                    stop = min(offset + batch_size, len(feature))
                    inputs = (
                        context[offset:stop],
                        base_field[offset:stop],
                        base_scalar[offset:stop],
                        tokens[offset:stop],
                        coords[offset:stop],
                        mask[offset:stop],
                    )
                    tensors = tuple(torch.as_tensor(np.array(a), device=device) for a in inputs)
                    result = model(*tensors)
                    scalar = fixed_scalar(*tensors)
                    corrected, corrected_scalar = calibrated_prediction(
                        base_field[offset:stop],
                        base_scalar[offset:stop],
                        result["field"].cpu().numpy(),
                        scalar.cpu().numpy(),
                        checkpoint["correction_strengths"],
                    )
                    if not np.isfinite(corrected).all() or not np.isfinite(corrected_scalar).all():
                        raise RuntimeError(f"non-finite prediction for {path.name}")
                    field_sum[start + offset : start + stop] += corrected / 5.0
                    scalar_sum[start + offset : start + stop] += corrected_scalar / 5.0
            if shot_index == 0 or (shot_index + 1) % 25 == 0 or shot_index + 1 == len(paths):
                print(
                    json.dumps(
                        {
                            "stage": "predict",
                            "fold": fold,
                            "shots": shot_index + 1,
                            "total": len(paths),
                            "last": path.name,
                        }
                    ),
                    flush=True,
                )
        field_sum.flush()
        scalar_sum.flush()
        del base, model, fixed_scalar, checkpoint
    output_archive = output / "diii_d_public_test.npz"
    with zipfile.ZipFile(
        output_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4, allowZip64=True
    ) as archive:
        for shot_index, path in enumerate(paths):
            start, stop = int(starts[shot_index]), int(starts[shot_index + 1])
            prefix = f"shot_{shot_index:04d}"
            _write_array(archive, f"{prefix}_psirz", np.asarray(field_sum[start:stop]))
            _write_array(archive, f"{prefix}_q95", np.asarray(scalar_sum[start:stop, 0]))
            _write_array(archive, f"{prefix}_betaN", np.asarray(scalar_sum[start:stop, 1]))
    print(
        json.dumps(
            {
                "stage": "complete",
                "archive": str(output_archive.resolve()),
                "shots": len(paths),
                "frames": int(starts[-1]),
                "bytes": output_archive.stat().st_size,
            }
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "predict", "all"), default="all")
    parser.add_argument("--max-shots", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selected-root", type=Path, default=SELECTED)
    parser.add_argument("--base-root", type=Path, default=BASES)
    parser.add_argument("--inputs", type=Path, default=PUBLIC)
    args = parser.parse_args()
    paths = sorted(args.inputs.glob("*.parquet"))
    if args.max_shots:
        paths = paths[: args.max_shots]
    if not paths:
        raise RuntimeError("DIII-D public input directory is empty")
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "inputs"
    if args.stage in ("prepare", "all"):
        prepare(paths, cache, args.workers)
    if args.stage in ("predict", "all"):
        predict(
            paths,
            cache,
            args.output,
            batch_size=args.batch_size,
            device_name=args.device,
            workers=args.workers,
            selected_root=args.selected_root,
            base_root=args.base_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
