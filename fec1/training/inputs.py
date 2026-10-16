#!/usr/bin/env python3
"""Materialize exact train-only grids and spatial tokens for FEC1 training."""

from __future__ import annotations


import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


ARRAY_SPECS = {
    "target_field": (np.float32, (65, 65)),
    "vacuum_field": (np.float32, (65, 65)),
    "token_values": (np.float32, (192, 5)),
    "token_coords": (np.float32, (192, 2)),
    "token_mask": (np.bool_, (192,)),
}


def _atomic_memmaps(
    output_dir: Path,
    rows: int,
) -> tuple[dict[str, np.memmap], dict[str, Path]]:
    arrays: dict[str, np.memmap] = {}
    temporary: dict[str, Path] = {}
    for name, (dtype, tail) in ARRAY_SPECS.items():
        final = output_dir / f"{name}.npy"
        path = final.with_name(f"{final.name}.tmp-{os.getpid()}")
        temporary[name] = path
        arrays[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(rows, *tail))
    return arrays, temporary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shots", type=int, default=0, help="smoke-test only")
    args = parser.parse_args()
    started = time.monotonic()

    with np.load(args.oof, allow_pickle=False) as saved:
        for key, expected in (
            ("data_policy", "real_diii_d_only"),
            ("same_shot_targets_used", False),
            ("development_used", False),
            ("frozen_audit_used", False),
        ):
            if key not in saved.files or saved[key].item() != expected:
                raise RuntimeError(f"unsafe OOF provenance: {key}")
        names = saved["shot_name"].astype(str)
        frames = saved["frame_index"].astype(np.int64)
        train_stride = int(saved["train_stride"].item())
    if args.max_shots > 0:
        selected_shots = np.unique(names)[: args.max_shots]
        selected = np.isin(names, selected_shots)
        names, frames = names[selected], frames[selected]
    if len(names) == 0:
        raise RuntimeError("no OOF rows selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, temporary = _atomic_memmaps(args.output_dir, len(names))
    token_sum = np.zeros(7, dtype=np.float64)
    token_sumsq = np.zeros(7, dtype=np.float64)
    token_count = 0
    try:
        shots = np.unique(names)
        for count, shot in enumerate(shots, start=1):
            rows = np.flatnonzero(names == shot)
            local_frames = frames[rows]
            path = args.cache_dir / f"{shot}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as saved:
                required = (
                    "target_psi",
                    "vacuum_psi",
                    "token_values",
                    "token_coords",
                    "token_mask",
                )
                missing = set(required) - set(saved.files)
                if missing:
                    raise RuntimeError(f"{path} misses {sorted(missing)}")
                if np.any(local_frames < 0) or np.any(local_frames >= len(saved["target_psi"])):
                    raise RuntimeError(f"invalid frame index for {shot}")
                target = np.asarray(saved["target_psi"])[local_frames].astype(np.float32)
                vacuum = np.asarray(saved["vacuum_psi"])[local_frames].astype(np.float32)
                token_values = np.asarray(saved["token_values"])[local_frames].astype(np.float32)
                token_coords = np.asarray(saved["token_coords"])[local_frames].astype(np.float32)
                token_mask = np.asarray(saved["token_mask"])[local_frames].astype(bool)
            if target.shape[1:] != (65, 65) or vacuum.shape != target.shape:
                raise RuntimeError(f"invalid spatial grids for {shot}")
            if token_values.shape[1:] != (192, 5):
                raise RuntimeError(f"invalid token values for {shot}")
            if token_coords.shape[1:] != (192, 2) or token_mask.shape[1:] != (192,):
                raise RuntimeError(f"invalid token geometry for {shot}")

            clean_values = np.nan_to_num(token_values, nan=0.0, posinf=0.0, neginf=0.0)
            clean_coords = np.nan_to_num(token_coords, nan=0.0, posinf=0.0, neginf=0.0)
            arrays["target_field"][rows] = target
            arrays["vacuum_field"][rows] = vacuum
            arrays["token_values"][rows] = clean_values
            arrays["token_coords"][rows] = clean_coords
            arrays["token_mask"][rows] = token_mask

            token = np.concatenate((clean_values, clean_coords), axis=-1)
            valid_token = token[token_mask]
            token_sum += valid_token.sum(axis=0, dtype=np.float64)
            token_sumsq += np.square(valid_token.astype(np.float64)).sum(axis=0)
            token_count += len(valid_token)
            if count == 1 or count % 100 == 0 or count == len(shots):
                print(f"materialized {count}/{len(shots)} shots", flush=True)

        for array in arrays.values():
            array.flush()
        del arrays
        for name, path in temporary.items():
            os.replace(path, args.output_dir / f"{name}.npy")
    finally:
        for path in temporary.values():
            if path.exists():
                path.unlink()

    token_mean = token_sum / max(token_count, 1)
    token_variance = token_sumsq / max(token_count, 1) - np.square(token_mean)
    token_scale = np.sqrt(np.maximum(token_variance, 1.0e-12))
    np.savez_compressed(
        args.output_dir / "index.npz",
        shot_name=names.astype("U19"),
        frame_index=frames.astype(np.int32),
        train_stride=np.asarray(train_stride, dtype=np.int32),
        data_policy=np.asarray("real_diii_d_only"),
        input_only_except_train_targets=np.asarray(True),
        development_used=np.asarray(False),
        frozen_audit_used=np.asarray(False),
    )
    report = {
        "oof": str(args.oof.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "shots": int(len(np.unique(names))),
        "rows": int(len(names)),
        "train_stride": train_stride,
        "token_count": int(token_count),
        "token_mean": token_mean.tolist(),
        "token_scale": token_scale.tolist(),
        "arrays": {name: str((args.output_dir / f"{name}.npy").resolve()) for name in ARRAY_SPECS},
        "data_policy": "real_diii_d_only",
        "development_used": False,
        "frozen_audit_used": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
