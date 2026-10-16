"""Exact-scorer DIII-D topology features for postprocessing predicted flux."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "fusion_scoring"
if str(SCORER) not in sys.path:
    sys.path.insert(0, str(SCORER))

from common import CONS_SCALARS
from derive import derive_frame
from lcfs import extract_lcfs
from neuralgs.config import NeuralGSConfig
from neuralgs.data import build_thomson_tokens
from neuralgs.geometry import render_magnetic_inputs


@lru_cache(maxsize=1)
def scorer_mask() -> tuple[np.ndarray, np.ndarray]:
    with np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz") as saved:
        mask = saved["mask_coarse"].astype(bool)
    return mask, mask.astype(np.float64)


def derive_flux_topology(
    flux: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    axis_sign: float,
    n_iter: int = 22,
    n_points: int = 512,
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    mask, mask_f = scorer_mask()
    values = np.full((len(flux), len(CONS_SCALARS)), np.nan, dtype=np.float64)
    contours: list[np.ndarray | None] = []
    for index, frame in enumerate(np.asarray(flux)):
        contour = extract_lcfs(
            frame, r, z, "DIII-D", mask, mask_f,
            n_iter=n_iter, n_points=n_points, axis_sign=axis_sign,
        )
        derived = derive_frame(
            frame, r, z, "DIII-D", mask, mask_f,
            contour=contour, with_li=True, axis_sign=axis_sign,
        )
        values[index] = [derived[name] for name in CONS_SCALARS]
        contours.append(contour)
    return values, contours


def _normalized_coarse(field: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    mean = values.mean(axis=(1, 2), keepdims=True)
    scale = values.std(axis=(1, 2), keepdims=True)
    normalized = (values - mean) / np.maximum(scale, 1.0e-8)
    return normalized[:, ::8, ::8].reshape(len(values), -1)


def _token_summary(row: dict) -> np.ndarray:
    _, values, mask = build_thomson_tokens(row, NeuralGSConfig())
    count = np.maximum(mask.sum(axis=1, keepdims=True), 1)
    mean = np.where(mask[..., None], values, 0.0).sum(axis=1) / count
    variance = (
        np.where(mask[..., None], (values - mean[:, None]) ** 2, 0.0).sum(axis=1)
        / count
    )
    return np.column_stack([mask.mean(axis=1), mean, np.sqrt(variance)])


def build_topology_features(
    row: dict,
    prediction: dict[str, np.ndarray],
    current: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray | None]]:
    r = np.asarray(row["efit_grid_R"], dtype=np.float64)
    z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    flux = np.asarray(prediction["psirz"], dtype=np.float64)
    if current is None:
        current, contours = derive_flux_topology(flux, r, z, axis_sign=1.0)
    else:
        current = np.asarray(current, dtype=np.float64)
        _, contours = derive_flux_topology(flux, r, z, axis_sign=1.0)

    magnetic = render_magnetic_inputs(row)
    vacuum = np.asarray(magnetic.vacuum_psi, dtype=np.float64)
    field_stats = np.column_stack([
        np.arcsinh(flux.mean(axis=(1, 2)) / 0.2),
        np.log1p(flux.std(axis=(1, 2)) / 0.02),
        np.log1p(np.ptp(flux, axis=(1, 2)) / 0.02),
        np.arcsinh(vacuum.mean(axis=(1, 2)) / 0.2),
        np.log1p(vacuum.std(axis=(1, 2)) / 0.02),
        np.log1p(np.ptp(vacuum, axis=(1, 2)) / 0.02),
    ])
    time = np.asarray(row["efit_times"], dtype=np.float64) / 1000.0
    if len(time) > 1:
        dt = np.maximum(np.gradient(time), 1.0e-6)
        topology_rate = np.gradient(current, axis=0) / dt[:, None]
    else:
        topology_rate = np.zeros_like(current)
    features = np.column_stack([
        magnetic.global_features,
        np.log1p(magnetic.ip_abs / 1.0e5),
        np.asarray(prediction["q95"], dtype=np.float64),
        np.asarray(prediction["betaN"], dtype=np.float64),
        current,
        np.nan_to_num(topology_rate, nan=0.0, posinf=0.0, neginf=0.0),
        field_stats,
        _normalized_coarse(flux),
        _normalized_coarse(vacuum),
        _token_summary(row),
    ])
    return (
        np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        current.astype(np.float32),
        contours,
    )


__all__ = [
    "build_topology_features",
    "derive_flux_topology",
    "scorer_mask",
]
