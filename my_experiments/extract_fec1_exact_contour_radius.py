#!/usr/bin/env python3
"""Extract compact polar LCFS signatures from exact FEC1 target flux."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "fusion_scoring"
for path in (ROOT, SCORER):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import CONS_SCALARS  # noqa: E402
from contour import symmetric_hausdorff  # noqa: E402
from derive import derive_frame  # noqa: E402
from lcfs import extract_lcfs  # noqa: E402
from my_experiments.fit_fec1_full_frame_pca import CACHE  # noqa: E402


@lru_cache(maxsize=1)
def _scorer_mask() -> tuple[np.ndarray, np.ndarray]:
    with np.load(ROOT / "fusion_scoring/masks/d3d_envelope.npz") as saved:
        mask = saved["mask_coarse"].astype(bool)
    return mask, mask.astype(np.float64)


def _derive_flux_topology(
    fields: np.ndarray, grid_r: np.ndarray, grid_z: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    mask, mask_f = _scorer_mask()
    values = np.full((len(fields), len(CONS_SCALARS)), np.nan, dtype=np.float64)
    contours: list[np.ndarray | None] = []
    for index, field in enumerate(fields):
        contour = extract_lcfs(
            field, grid_r, grid_z, "DIII-D", mask, mask_f,
            n_iter=22, n_points=512, axis_sign=1.0,
        )
        derived = derive_frame(
            field, grid_r, grid_z, "DIII-D", mask, mask_f,
            contour=contour, with_li=True, axis_sign=1.0,
        )
        values[index] = [derived[name] for name in CONS_SCALARS]
        contours.append(contour)
    return values, contours


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contour_radius(
    contour: np.ndarray,
    axis_r: float,
    axis_z: float,
    angles: np.ndarray,
) -> np.ndarray:
    """Periodic polar interpolation of a star-shaped tokamak boundary."""
    points = np.asarray(contour, dtype=np.float64)
    dr = points[:, 0] - float(axis_r)
    dz = points[:, 1] - float(axis_z)
    theta = np.mod(np.arctan2(dz, dr), 2.0 * np.pi)
    radius = np.sqrt(np.square(dr) + np.square(dz))
    order = np.argsort(theta)
    theta = theta[order]
    radius = radius[order]
    # Collapse numerically duplicated polar angles. Taking the outermost
    # intersection preserves the separatrix lobe at a diverted X-point.
    key = np.round(theta, decimals=8)
    unique, inverse = np.unique(key, return_inverse=True)
    outer = np.full(len(unique), -np.inf, dtype=np.float64)
    np.maximum.at(outer, inverse, radius)
    if len(unique) < 8:
        return np.full(len(angles), np.nan, dtype=np.float64)
    extended_theta = np.concatenate((unique[-1:] - 2.0 * np.pi, unique, unique[:1] + 2.0 * np.pi))
    extended_radius = np.concatenate((outer[-1:], outer, outer[:1]))
    return np.interp(angles, extended_theta, extended_radius)


def canonical_contour_offset(
    contour: np.ndarray, axis_r: float, axis_z: float,
) -> np.ndarray:
    """Fix contour orientation/start and express it relative to the O-point."""
    points = np.asarray(contour, dtype=np.float64)
    signed_area_twice = np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - np.roll(points[:, 0], -1) * points[:, 1]
    )
    if signed_area_twice < 0.0:
        points = points[::-1]
    # Anchor at the outboard-most point; the tiny vertical tie-break avoids a
    # one-index start jitter on nearly vertical outboard segments.
    anchor_score = points[:, 0] - 1.0e-8 * np.abs(points[:, 1] - axis_z)
    points = np.roll(points, -int(np.argmax(anchor_score)), axis=0)
    return points - np.asarray([axis_r, axis_z], dtype=np.float64)


def _one(task: tuple[str, np.ndarray, str, np.ndarray, bool, int]) -> tuple:
    shot, frame_indices, cache_dir, angles, radius_only, contour_points = task
    with np.load(Path(cache_dir) / f"{shot}.npz", allow_pickle=False) as saved:
        fields = np.asarray(saved["target_psi"])[frame_indices].astype(np.float64)
        grid_r = np.asarray(saved["grid_r"], dtype=np.float64)
        grid_z = np.asarray(saved["grid_z"], dtype=np.float64)
    topology, contours = _derive_flux_topology(fields, grid_r, grid_z)
    radius = np.full((len(fields), len(angles)), np.nan, dtype=np.float32)
    contour_offset = (
        None if radius_only
        else np.full((len(fields), contour_points, 2), np.nan, dtype=np.float32)
    )
    reconstruction_distance = np.full(len(fields), np.nan, dtype=np.float32)
    for index, contour in enumerate(contours):
        if contour is None or not np.isfinite(topology[index, :2]).all():
            continue
        radius[index] = contour_radius(
            contour, topology[index, 0], topology[index, 1], angles
        ).astype(np.float32)
        if contour_offset is not None:
            canonical = canonical_contour_offset(
                contour, topology[index, 0], topology[index, 1]
            )
            chosen = (
                np.arange(contour_points, dtype=np.int64) * len(canonical)
                // contour_points
            )
            contour_offset[index] = canonical[chosen].astype(np.float32)
        reconstructed = np.column_stack((
            topology[index, 0] + radius[index] * np.cos(angles),
            topology[index, 1] + radius[index] * np.sin(angles),
        ))
        reconstruction_distance[index] = symmetric_hausdorff(
            reconstructed, contour
        )
    return shot, frame_indices, radius, contour_offset, reconstruction_distance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--angles", type=int, default=64)
    parser.add_argument(
        "--radius-only", action="store_true",
        help="omit the much larger 512-point contour-offset tensor",
    )
    parser.add_argument(
        "--contour-points", type=int, default=512,
        help="uniformly subsample the exact 512-point scorer contour",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-shots", type=int, default=0, help="smoke-test prefix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.contour_points < 16 or args.contour_points > 512:
        parser.error("contour points must lie between 16 and 512")

    with np.load(args.reference, allow_pickle=False) as saved:
        if saved["frozen_audit_used"].item():
            raise RuntimeError("contour reference opened frozen audit")
        values = {
            key: np.asarray(saved[key])
            for key in ("sample_index", "shot_name", "frame_index", "split")
        }
    order = np.argsort(values["sample_index"].astype(np.int64))
    values = {key: value[order] for key, value in values.items()}
    if np.any(values["split"].astype(str) == "frozen_audit"):
        raise RuntimeError("frozen audit entered contour extraction")
    angles = np.linspace(0.0, 2.0 * np.pi, args.angles, endpoint=False)
    names = values["shot_name"].astype(str)
    frames = values["frame_index"].astype(np.int64)
    if args.max_shots > 0:
        selected_shots = np.unique(names)[: args.max_shots]
        keep = np.isin(names, selected_shots)
        values = {key: value[keep] for key, value in values.items()}
        names = values["shot_name"].astype(str)
        frames = values["frame_index"].astype(np.int64)
    tasks = []
    for shot in np.unique(names):
        rows = np.flatnonzero(names == shot)
        tasks.append((
            shot, frames[rows], str(args.cache_dir.resolve()), angles,
            args.radius_only, args.contour_points,
        ))
    with ProcessPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        results = []
        for count, result in enumerate(executor.map(_one, tasks, chunksize=1), start=1):
            results.append(result)
            if count == 1 or count % 100 == 0 or count == len(tasks):
                print(f"exact contour radius {count}/{len(tasks)} shots", flush=True)
    by_shot = {
        shot: (frame, radius, contour, distance)
        for shot, frame, radius, contour, distance in results
    }
    cursors = {shot: 0 for shot in by_shot}
    radii = []
    contours = [] if not args.radius_only else None
    distances = []
    for shot in names:
        cursor = cursors[shot]
        radii.append(by_shot[shot][1][cursor])
        if contours is not None:
            contours.append(by_shot[shot][2][cursor])
        distances.append(by_shot[shot][3][cursor])
        cursors[shot] = cursor + 1
    radius = np.asarray(radii, dtype=np.float32)
    contour_offset = (
        np.asarray(contours, dtype=np.float32) if contours is not None else None
    )
    distance = np.asarray(distances, dtype=np.float32)
    finite = np.isfinite(radius).all(axis=1)
    if contour_offset is not None:
        finite &= np.isfinite(contour_offset).all(axis=(1, 2))
    if finite.mean() < 0.995:
        raise RuntimeError("too many non-finite exact contour signatures")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        sample_index=values["sample_index"].astype(np.int64),
        shot_name=values["shot_name"].astype("U19"),
        frame_index=values["frame_index"].astype(np.int32),
        split=values["split"].astype("U12"),
        radius=radius,
        angles=angles.astype(np.float32),
        reconstruction_distance_m=distance,
        data_policy=np.asarray("real_diii_d_only"),
        mast_data_used=np.asarray(False),
        frozen_audit_used=np.asarray(False),
        label_source=np.asarray("exact scorer LCFS from target flux"),
    )
    if contour_offset is not None:
        payload["contour_offset"] = contour_offset
    np.savez_compressed(args.output, **payload)
    report = {
        "reference": str(args.reference.resolve()),
        "reference_sha256": _sha256(args.reference),
        "output": str(args.output.resolve()),
        "shots": int(len(tasks)),
        "frames": int(len(radius)),
        "angles": args.angles,
        "radius_only": args.radius_only,
        "contour_points": 0 if args.radius_only else args.contour_points,
        "finite_fraction": float(finite.mean()),
        "reconstruction_distance_m": {
            "mean": float(np.nanmean(distance)),
            "median": float(np.nanmedian(distance)),
            "p95": float(np.nanquantile(distance, 0.95)),
            "max": float(np.nanmax(distance)),
        },
        "frozen_audit_used": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
