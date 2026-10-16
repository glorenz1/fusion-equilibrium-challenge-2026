#!/usr/bin/env python3
"""Prepare train-only score-aligned targets for the temporal FEC1 residual model.

The source OOF archive fixes the exact shot/frame row order.  This program
copies only the matching normalized-flux teachers from immutable training-shot
caches into a memory-mapped array and extracts the train-only PCA decoder into
a compact archive.  Development and frozen-audit shot files are never opened.
"""

from __future__ import annotations
from fec1.paths import DATA_POLICY

from fec1.paths import CACHE


import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pca-output", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="float16 is sufficient for the smooth topology losses and halves I/O",
    )
    args = parser.parse_args()
    started = time.monotonic()

    with np.load(args.oof, allow_pickle=False) as saved:
        for key, expected in (
            ("data_policy", DATA_POLICY),
            ("same_shot_targets_used", False),
            ("development_used", False),
            ("frozen_audit_used", False),
        ):
            if key not in saved.files or saved[key].item() != expected:
                raise RuntimeError(f"unsafe OOF provenance: {key}")
        names = np.asarray(saved["shot_name"]).astype(str)
        frames = np.asarray(saved["frame_index"], dtype=np.int64)
        shape_width = int(np.asarray(saved["target_shape"]).shape[1])
    unique = np.unique(names)
    from fec1.data.features import _split_records

    if set(unique) != set(_split_records()[0]):
        raise RuntimeError(f"unexpected immutable training-shot identities; found {len(unique)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if args.dtype == "float16" else np.float32
    target = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=dtype,
        shape=(len(names), 65, 65),
    )
    grid_r = grid_z = None
    for count, shot in enumerate(unique, start=1):
        rows = np.flatnonzero(names == shot)
        path = args.cache_dir / f"{shot}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as saved:
            if "target_psi_n" not in saved.files:
                raise RuntimeError(f"{path.name} lacks target_psi_n")
            local_frames = frames[rows]
            if local_frames.min(initial=0) < 0 or local_frames.max(initial=0) >= len(
                saved["target_psi_n"]
            ):
                raise RuntimeError(f"{shot}: OOF frame index is out of bounds")
            values = np.asarray(saved["target_psi_n"])[local_frames]
            target[rows] = np.clip(values, -0.5, 2.0).astype(dtype)
            observed_r = np.asarray(saved["grid_r"], dtype=np.float32)
            observed_z = np.asarray(saved["grid_z"], dtype=np.float32)
        if grid_r is None:
            grid_r, grid_z = observed_r, observed_z
        elif not (np.array_equal(grid_r, observed_r) and np.array_equal(grid_z, observed_z)):
            raise RuntimeError("DIII-D score-aligned caches do not share one grid")
        if count == 1 or count % 100 == 0 or count == len(unique):
            print(f"score targets {count}/{len(unique)} train shots", flush=True)
    target.flush()
    del target

    artifact = joblib.load(args.base_artifact)
    for key, expected in (
        ("data_policy", DATA_POLICY),
        ("development_used_for_fit", False),
        ("frozen_audit_used", False),
    ):
        if artifact.get(key) != expected:
            raise RuntimeError(f"unsafe PCA artifact provenance: {key}")
    pca = artifact["pca"]
    if int(pca.n_components_) != shape_width:
        raise RuntimeError("PCA width differs from the OOF shape width")
    args.pca_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.pca_output,
        components=np.asarray(pca.components_, dtype=np.float32),
        mean=np.asarray(pca.mean_, dtype=np.float32),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float32),
        grid_r=grid_r,
        grid_z=grid_z,
        data_policy=np.asarray(DATA_POLICY),
        development_used=np.asarray(False),
        frozen_audit_used=np.asarray(False),
    )
    report = {
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "pca_output": str(args.pca_output.resolve()),
        "pca_output_sha256": _sha256(args.pca_output),
        "oof": str(args.oof.resolve()),
        "oof_sha256": _sha256(args.oof),
        "base_artifact": str(args.base_artifact.resolve()),
        "rows": int(len(names)),
        "shots": int(len(unique)),
        "shape": [int(len(names)), 65, 65],
        "dtype": args.dtype,
        "pca_components": shape_width,
        "pca_explained_variance": float(
            np.asarray(pca.explained_variance_ratio_, dtype=np.float64).sum()
        ),
        "data_policy": DATA_POLICY,
        "development_used": False,
        "frozen_audit_used": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
