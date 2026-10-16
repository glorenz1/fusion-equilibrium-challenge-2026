"""Align frozen training-only topology and contour cache rows."""
from __future__ import annotations
from fec1.paths import DATA_POLICY
from pathlib import Path
import numpy as np
from fec1.scoring.vendor.common import CONS_SCALARS as CONSISTENCY_NAMES


def _align_topology(
    path: Path,
    names: np.ndarray,
    frames: np.ndarray,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as saved:
        for key, expected in (
            ("data_policy", DATA_POLICY),
            ("development_used", False),
            ("frozen_audit_used", False),
        ):
            if key not in saved.files or saved[key].item() != expected:
                raise RuntimeError(f"unsafe topology provenance: {key}")
        if saved["names"].astype(str).tolist() != list(CONSISTENCY_NAMES):
            raise RuntimeError("topology target column order changed")
        source_names = saved["shot_name"].astype(str)
        source_frames = saved["frame_index"].astype(np.int64)
        source_target = saved["target"].astype(np.float32)
    if np.array_equal(source_names, names) and np.array_equal(source_frames, frames):
        return source_target
    lookup = {
        (shot, int(frame)): index
        for index, (shot, frame) in enumerate(zip(source_names, source_frames))
    }
    try:
        rows = np.asarray(
            [lookup[(shot, int(frame))] for shot, frame in zip(names, frames)],
            dtype=np.int64,
        )
    except KeyError as error:
        raise RuntimeError("topology archive misses an OOF shot/frame") from error
    return source_target[rows]


def _align_contours(
    path: Path,
    names: np.ndarray,
    frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Align train-only exact LCFS radii to the OOF row order."""
    with np.load(path, allow_pickle=False) as saved:
        for key, expected in (
            ("data_policy", DATA_POLICY),
            ("frozen_audit_used", False),
        ):
            if key not in saved.files or saved[key].item() != expected:
                raise RuntimeError(f"unsafe contour provenance: {key}")
        if "split" in saved.files and np.any(saved["split"].astype(str) != "train"):
            raise RuntimeError("non-train rows entered exact contour supervision")
        source_names = saved["shot_name"].astype(str)
        source_frames = saved["frame_index"].astype(np.int64)
        source_radius = saved["radius"].astype(np.float32)
        angles = saved["angles"].astype(np.float32)
        source_offset = (
            saved["contour_offset"].astype(np.float32) if "contour_offset" in saved.files else None
        )
    if np.array_equal(source_names, names) and np.array_equal(source_frames, frames):
        return source_radius, angles, source_offset
    lookup = {
        (shot, int(frame)): index
        for index, (shot, frame) in enumerate(zip(source_names, source_frames))
    }
    try:
        rows = np.asarray(
            [lookup[(shot, int(frame))] for shot, frame in zip(names, frames)],
            dtype=np.int64,
        )
    except KeyError as error:
        raise RuntimeError("contour archive misses an OOF shot/frame") from error
    return (
        source_radius[rows],
        angles,
        source_offset[rows] if source_offset is not None else None,
    )
