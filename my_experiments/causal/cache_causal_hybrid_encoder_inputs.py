#!/usr/bin/env python3
"""Cache causal ``CausalHybridEncoderV1`` inputs for the frozen DIII-D bridge.

Only measurement, timing, and machine-geometry columns are loaded from the raw
parquet files.  EFIT targets and target-derived scalar columns are excluded at
the parquet boundary.  Each supervised bridge frame is preceded by a short,
strictly causal EFIT-frame history so the time-decayed GRU is trained rather
than bypassed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from FEC1.decoder import decoder_frame_from_encoder
from FEC1.encoder.common_inputs import (
    MODEL_SPATIAL_CHANNELS,
    SHARED_SCALAR_CHANNELS,
    THOMSON_TOKEN_FEATURES,
    build_common_encoder_inputs,
)
from src.measurement.coil_calculation import calculate_coil_fields
from src.measurement.read_measurements import read_measurements


DEFAULT_BRIDGE = ROOT / "my_experiments/d3d_spatial_current_bridge_wide"
DEFAULT_RAW = ROOT / "data/diii_d_train"
DEFAULT_OUTPUT = ROOT / "my_experiments/causal_hybrid_encoder_inputs"

_SAFE_FIXED_COLUMNS = {
    "source",
    "efit_times",
    "efit_grid_R",
    "efit_grid_Z",
    "magnetics_time",
    "magnetics_plasma_current_times",
}
_SAFE_PREFIXES = ("coil_", "thomson_")
_FORBIDDEN_EXACT = {
    "efit_psirz",
    "efit_beta_n",
    "efit_li",
    "efit_q95",
    "efit_r_axis",
    "efit_z_axis",
    "efit_lcfs_n",
    "efit_lcfs_r",
    "efit_lcfs_z",
    "magnetics_dsep",
    "magnetics_dsep_times",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_input_columns(column_names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return the explicit target-free parquet projection used by the cache."""

    selected = []
    for name in column_names:
        if name in _FORBIDDEN_EXACT:
            continue
        if (
            name in _SAFE_FIXED_COLUMNS
            or name.startswith(_SAFE_PREFIXES)
            or name.startswith("magnetics_")
        ):
            selected.append(name)
    forbidden = _FORBIDDEN_EXACT.intersection(selected)
    if forbidden or any(
        name.startswith("efit_") and name not in _SAFE_FIXED_COLUMNS
        for name in selected
    ):
        raise RuntimeError(f"target columns crossed the input firewall: {forbidden}")
    required = {
        "source",
        "efit_times",
        "efit_grid_R",
        "efit_grid_Z",
        "magnetics_plasma_current",
        "coil_input_column",
    }
    missing = required.difference(selected)
    if missing:
        raise RuntimeError(f"raw parquet is missing required inputs: {sorted(missing)}")
    return tuple(selected)


def causal_history_indices(
    target_index: int,
    frame_count: int,
    history_frames: int,
    history_stride: int,
) -> np.ndarray:
    """Return increasing indices ending at the supervised frame."""

    if frame_count < 1:
        raise ValueError("a shot must have at least one EFIT frame")
    if not 0 <= target_index < frame_count:
        raise IndexError("target frame is outside the shot")
    if history_frames < 1 or history_stride < 1:
        raise ValueError("history length and stride must be positive")
    start = target_index - history_stride * (history_frames - 1)
    return np.arange(max(start, 0), target_index + 1, history_stride, dtype=np.int64)


def _left_pad(values: np.ndarray, length: int) -> np.ndarray:
    result = np.zeros((length, *values.shape[1:]), dtype=values.dtype)
    result[-len(values) :] = values
    return result


def build_one_input(
    raw_path: Path,
    target_index: int,
    *,
    history_frames: int,
    history_stride: int,
) -> dict[str, np.ndarray | float | int | tuple[str, ...]]:
    schema = tuple(pq.ParquetFile(raw_path).schema_arrow.names)
    columns = safe_input_columns(schema)
    frame = pd.read_parquet(raw_path, columns=list(columns))
    if len(frame) != 1:
        raise ValueError(f"expected one shot row in {raw_path}, found {len(frame)}")
    row = frame.iloc[0].to_dict()
    measurements = read_measurements(
        row,
        current_interpolation="previous",
        thomson_interpolation="previous",
        strict_coordinates=True,
    )
    indices = causal_history_indices(
        int(target_index),
        len(measurements.times_ms),
        history_frames,
        history_stride,
    )
    fields = calculate_coil_fields(
        row,
        measurements,
        frame_indices=indices,
        canonicalize_current_direction=False,
        strict_geometry=False,
    )
    inputs = build_common_encoder_inputs(measurements, fields)
    if inputs.spatial_channel_names != MODEL_SPATIAL_CHANNELS:
        raise RuntimeError("spatial channel schema changed")
    if inputs.shared_scalars.feature_names != SHARED_SCALAR_CHANNELS:
        raise RuntimeError("shared scalar schema changed")
    if inputs.thomson_tokens.feature_names != THOMSON_TOKEN_FEATURES:
        raise RuntimeError("Thomson token schema changed")
    valid_count = len(indices)
    sequence_mask = np.zeros(history_frames, dtype=bool)
    sequence_mask[-valid_count:] = True
    times = np.asarray(inputs.raster.times_ms, dtype=np.float64)
    delta = np.zeros(valid_count, dtype=np.float32)
    if valid_count > 1:
        delta[1:] = np.diff(times).astype(np.float32)
    frame_inputs = decoder_frame_from_encoder(inputs, valid_count - 1)
    return {
        "spatial_values": _left_pad(inputs.spatial_values, history_frames),
        "shared_scalars": _left_pad(inputs.shared_scalars.values, history_frames),
        "thomson_values": _left_pad(inputs.thomson_tokens.values, history_frames),
        "thomson_attention_mask": _left_pad(
            inputs.thomson_tokens.attention_mask, history_frames
        ),
        "sequence_mask": sequence_mask,
        "delta_time_ms": _left_pad(delta[:, None], history_frames)[:, 0],
        "active_psi_Wb_per_rad": frame_inputs.active_psi_Wb_per_rad.astype(
            np.float32
        ),
        "plasma_current_magnitude_A": float(
            frame_inputs.plasma_current_magnitude_A
        ),
        "target_time_ms": float(frame_inputs.time_ms),
        "history_source_indices": _left_pad(indices[:, None], history_frames)[:, 0],
        "loaded_columns": columns,
        "field_approximations": tuple(fields.approximations),
    }


def _pad_tokens(values: np.ndarray, maximum_tokens: int) -> np.ndarray:
    result = np.zeros(
        (*values.shape[:-2], maximum_tokens, values.shape[-1]), dtype=values.dtype
    )
    result[..., : values.shape[-2], :] = values
    return result


def _pad_token_mask(values: np.ndarray, maximum_tokens: int) -> np.ndarray:
    result = np.zeros((*values.shape[:-1], maximum_tokens), dtype=bool)
    result[..., : values.shape[-1]] = values
    return result


def cache_part(args: argparse.Namespace) -> int:
    part_index = args.part_index
    if part_index is None:
        part_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if not 0 <= part_index < args.parts:
        raise ValueError("part-index must lie in [0, parts)")
    bridge_path = args.bridge_dir / "bridge.npz"
    manifest_path = args.bridge_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with np.load(bridge_path, allow_pickle=False) as bridge:
        shot_names = bridge["shot_name"].astype(str)
        frame_indices = bridge["frame_index"].astype(np.int64)
    chosen = np.arange(len(shot_names), dtype=np.int64)[part_index :: args.parts]
    if args.limit is not None:
        chosen = chosen[: args.limit]
    if not len(chosen):
        raise RuntimeError("selected cache shard is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.monotonic()
    loaded_columns: tuple[str, ...] | None = None
    approximations: set[str] = set()
    for local, source_index in enumerate(chosen, start=1):
        shot = shot_names[source_index]
        path = args.raw_dir / f"{shot}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        item = build_one_input(
            path,
            int(frame_indices[source_index]),
            history_frames=args.history_frames,
            history_stride=args.history_stride,
        )
        if loaded_columns is None:
            loaded_columns = item.pop("loaded_columns")
        elif item.pop("loaded_columns") != loaded_columns:
            raise RuntimeError("input parquet schema changed within a cache shard")
        approximations.update(item.pop("field_approximations"))
        rows.append(item)
        print(
            f"part={part_index}/{args.parts} sample={local}/{len(chosen)} "
            f"shot={shot} frame={frame_indices[source_index]}",
            flush=True,
        )
    maximum_tokens = max(row["thomson_values"].shape[-2] for row in rows)
    output = parts_dir / f"part_{part_index:04d}.npz"
    np.savez_compressed(
        output,
        source_index=chosen,
        shot_name=shot_names[chosen],
        frame_index=frame_indices[chosen],
        spatial_values=np.asarray([row["spatial_values"] for row in rows]),
        shared_scalars=np.asarray([row["shared_scalars"] for row in rows]),
        thomson_values=np.asarray(
            [_pad_tokens(row["thomson_values"], maximum_tokens) for row in rows]
        ),
        thomson_attention_mask=np.asarray(
            [
                _pad_token_mask(row["thomson_attention_mask"], maximum_tokens)
                for row in rows
            ]
        ),
        sequence_mask=np.asarray([row["sequence_mask"] for row in rows]),
        delta_time_ms=np.asarray([row["delta_time_ms"] for row in rows]),
        active_psi_Wb_per_rad=np.asarray(
            [row["active_psi_Wb_per_rad"] for row in rows]
        ),
        plasma_current_magnitude_A=np.asarray(
            [row["plasma_current_magnitude_A"] for row in rows]
        ),
        target_time_ms=np.asarray([row["target_time_ms"] for row in rows]),
        history_source_indices=np.asarray(
            [row["history_source_indices"] for row in rows]
        ),
    )
    report = {
        "complete": True,
        "part_index": part_index,
        "parts": args.parts,
        "samples": len(rows),
        "history_frames": args.history_frames,
        "history_stride": args.history_stride,
        "maximum_thomson_tokens": maximum_tokens,
        "strictly_causal_interpolation": True,
        "target_columns_loaded": False,
        "loaded_columns": list(loaded_columns or ()),
        "field_strict_geometry": False,
        "field_approximations": sorted(approximations),
        "bridge_manifest_sha256": sha256_file(manifest_path),
        "bridge_dataset_sha256": sha256_file(bridge_path),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "elapsed_seconds": time.monotonic() - started,
        "mast_inputs_used": False,
        "mast_targets_used": False,
        "bridge_complete": bool(manifest.get("complete")),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def assemble(args: argparse.Namespace) -> int:
    parts_dir = args.output_dir / "parts"
    paths = [parts_dir / f"part_{index:04d}.npz" for index in range(args.parts)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing {len(missing)} cache parts; first={missing[0]}")
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            parts.append({key: archive[key].copy() for key in archive.files})
    source_index = np.concatenate([part["source_index"] for part in parts])
    order = np.argsort(source_index)
    sorted_index = source_index[order]
    if len(np.unique(sorted_index)) != len(sorted_index):
        raise RuntimeError("cache parts contain duplicate source indices")
    bridge_path = args.bridge_dir / "bridge.npz"
    with np.load(bridge_path, allow_pickle=False) as bridge:
        bridge_count = len(bridge["shot_name"])
        bridge_names = bridge["shot_name"].astype(str)
        bridge_frames = bridge["frame_index"].astype(np.int64)
    if not args.allow_partial and not np.array_equal(
        sorted_index, np.arange(bridge_count, dtype=np.int64)
    ):
        raise RuntimeError("assembled cache does not cover every bridge frame")
    maximum_tokens = max(part["thomson_values"].shape[-2] for part in parts)
    token_values = np.concatenate(
        [_pad_tokens(part["thomson_values"], maximum_tokens) for part in parts]
    )[order]
    token_mask = np.concatenate(
        [
            _pad_token_mask(part["thomson_attention_mask"], maximum_tokens)
            for part in parts
        ]
    )[order]
    ordinary_keys = (
        "shot_name",
        "frame_index",
        "spatial_values",
        "shared_scalars",
        "sequence_mask",
        "delta_time_ms",
        "active_psi_Wb_per_rad",
        "plasma_current_magnitude_A",
        "target_time_ms",
        "history_source_indices",
    )
    packed = {
        key: np.concatenate([part[key] for part in parts])[order]
        for key in ordinary_keys
    }
    packed.update(
        source_index=sorted_index,
        thomson_values=token_values,
        thomson_attention_mask=token_mask,
    )
    if not np.array_equal(packed["shot_name"].astype(str), bridge_names[sorted_index]):
        raise RuntimeError("assembled shot names do not align with bridge labels")
    if not np.array_equal(packed["frame_index"], bridge_frames[sorted_index]):
        raise RuntimeError("assembled frame indices do not align with bridge labels")
    output = args.output_dir / "inputs.npz"
    np.savez_compressed(output, **packed)
    report = {
        "complete": not args.allow_partial,
        "samples": len(sorted_index),
        "bridge_samples": bridge_count,
        "parts": args.parts,
        "history_frames": int(packed["spatial_values"].shape[1]),
        "maximum_thomson_tokens": maximum_tokens,
        "target_columns_loaded": False,
        "strictly_causal_interpolation": True,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "bridge_dataset_sha256": sha256_file(bridge_path),
        "mast_inputs_used": False,
        "mast_targets_used": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-dir", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parts", type=int, default=32)
    parser.add_argument("--part-index", type=int)
    parser.add_argument("--history-frames", type=int, default=8)
    parser.add_argument("--history-stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.parts < 1:
        raise ValueError("parts must be positive")
    return assemble(args) if args.assemble else cache_part(args)


if __name__ == "__main__":
    raise SystemExit(main())
