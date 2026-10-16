"""Machine-independent scalar features and a lightweight Ridge inference head."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np


FEATURE_VERSION = 1
DEFAULT_SCALAR_ARTIFACT = (
    Path(__file__).resolve().parent.parent
    / "my_experiments"
    / "artifacts"
    / "neuralgs_scalar_ridge.joblib"
)


def _masked_token_features(
    values: np.ndarray,
    coords: np.ndarray,
    mask: np.ndarray,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
) -> np.ndarray:
    """Reduce variable-length Thomson sets without using machine/channel IDs."""

    count = mask.sum(axis=1).astype(np.float64)
    denom = np.maximum(count, 1.0)[:, None]
    clean = np.where(mask[..., None], values[..., :4], 0.0).astype(np.float64)
    mean = clean.sum(axis=1) / denom
    variance = (
        np.where(mask[..., None], (values[..., :4] - mean[:, None]) ** 2, 0.0).sum(axis=1)
        / denom
    )
    std = np.sqrt(np.maximum(variance, 0.0))
    maximum = np.where(mask[..., None], values[..., :4], -np.inf).max(axis=1)
    minimum = np.where(mask[..., None], values[..., :4], np.inf).min(axis=1)
    empty = count == 0
    maximum[empty] = 0.0
    minimum[empty] = 0.0

    r_span = max(float(grid_r[-1] - grid_r[0]), 1.0e-8)
    z_span = max(float(grid_z[-1] - grid_z[0]), 1.0e-8)
    normalized_coords = np.asarray(coords, dtype=np.float64).copy()
    normalized_coords[..., 0] = (normalized_coords[..., 0] - grid_r[0]) / r_span
    normalized_coords[..., 1] = (normalized_coords[..., 1] - grid_z[0]) / z_span
    pressure_weight = np.expm1(np.clip(values[..., 2], 0.0, 20.0)) * mask
    weight_sum = np.maximum(pressure_weight.sum(axis=1), 1.0e-8)
    centroid = (pressure_weight[..., None] * normalized_coords).sum(axis=1) / weight_sum[:, None]
    width = np.sqrt(
        np.maximum(
            (
                pressure_weight[..., None]
                * (normalized_coords - centroid[:, None]) ** 2
            ).sum(axis=1)
            / weight_sum[:, None],
            0.0,
        )
    )
    centroid[empty] = 0.0
    width[empty] = 0.0
    valid_fraction = (count / max(mask.shape[1], 1))[:, None]
    return np.column_stack([valid_fraction, mean, std, minimum, maximum, centroid, width])


def build_scalar_features(prepared: dict[str, np.ndarray]) -> np.ndarray:
    """Build fixed-size q95/betaN features from the NeuralGS tensor contract."""

    vacuum = np.asarray(prepared["vacuum_psi"], dtype=np.float64)
    global_features = np.asarray(prepared["global_features"], dtype=np.float64)
    values = np.asarray(prepared["token_values"], dtype=np.float64)
    coords = np.asarray(prepared["token_coords"], dtype=np.float64)
    mask = np.asarray(prepared["token_mask"], dtype=bool)
    grid_r = np.asarray(prepared["grid_r"], dtype=np.float64)
    grid_z = np.asarray(prepared["grid_z"], dtype=np.float64)
    n_frames = len(vacuum)

    sample9 = np.unique(np.linspace(0, len(grid_r) - 1, 9).round().astype(int))
    coarse_psi = vacuum[:, sample9][:, :, sample9].reshape(n_frames, -1)
    dpsi_dz, dpsi_dr = np.gradient(vacuum, grid_z, grid_r, axis=(1, 2))
    sample5 = np.unique(np.linspace(0, len(grid_r) - 1, 5).round().astype(int))
    coarse_dz = dpsi_dz[:, sample5][:, :, sample5].reshape(n_frames, -1)
    coarse_dr = dpsi_dr[:, sample5][:, :, sample5].reshape(n_frames, -1)
    vacuum_flat = vacuum.reshape(n_frames, -1)
    vacuum_stats = np.column_stack(
        [
            vacuum_flat.mean(axis=1),
            vacuum_flat.std(axis=1),
            vacuum_flat.min(axis=1),
            vacuum_flat.max(axis=1),
            np.ptp(vacuum_flat, axis=1),
        ]
    )

    source_id = values[..., 4]
    token_parts = []
    for group_mask in (mask, mask & (source_id < 0.0), mask & (source_id > 0.0)):
        token_parts.append(
            _masked_token_features(values, coords, group_mask, grid_r, grid_z)
        )

    grid_values = np.asarray(
        [
            grid_r[0], grid_r[-1], grid_z[0], grid_z[-1],
            grid_r[-1] - grid_r[0], grid_z[-1] - grid_z[0],
            0.5 * (grid_r[0] + grid_r[-1]),
            0.5 * (grid_z[0] + grid_z[-1]),
            (grid_r[-1] - grid_r[0]) / max(grid_z[-1] - grid_z[0], 1.0e-8),
        ],
        dtype=np.float64,
    )
    grid_features = np.broadcast_to(grid_values, (n_frames, len(grid_values)))
    features = np.column_stack(
        [
            global_features,
            coarse_psi,
            coarse_dz,
            coarse_dr,
            vacuum_stats,
            *token_parts,
            grid_features,
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_scalar_artifact(path: Path | str = DEFAULT_SCALAR_ARTIFACT) -> dict:
    artifact = joblib.load(Path(path))
    if artifact.get("feature_version") != FEATURE_VERSION:
        raise ValueError("unsupported NeuralGS scalar feature version")
    return artifact


def predict_prepared_scalars(prepared: dict[str, np.ndarray], artifact: dict) -> dict[str, np.ndarray]:
    features = build_scalar_features(prepared).astype(np.float64)
    if features.shape[1] != artifact["n_features"]:
        raise ValueError("NeuralGS scalar feature width does not match the artifact")
    output = {}
    for name, bounds in (("q95", (0.1, 30.0)), ("betaN", (0.0, 10.0))):
        model = artifact["models"][name]
        values = features @ model["coefficient"] + model["intercept"]
        output[name] = np.clip(values, *bounds).astype(np.float32)
    return output


def artifact_matches_grid(
    prepared: dict[str, np.ndarray], artifact: dict, tolerance: float = 1.0e-4
) -> bool:
    """Fail-closed geometry OOD guard for the DIII-D-only scalar model."""

    reference = artifact.get("metadata", {}).get("training_grid")
    if not reference:
        return False
    grid_r = np.asarray(prepared["grid_r"], dtype=np.float64)
    grid_z = np.asarray(prepared["grid_z"], dtype=np.float64)
    observed = np.asarray(
        [grid_r[0], grid_r[-1], len(grid_r), grid_z[0], grid_z[-1], len(grid_z)],
        dtype=np.float64,
    )
    expected = np.asarray(
        [
            reference["r_min"], reference["r_max"], reference["n_r"],
            reference["z_min"], reference["z_max"], reference["n_z"],
        ],
        dtype=np.float64,
    )
    scale = np.maximum(np.abs(expected), 1.0)
    return bool(np.all(np.abs(observed - expected) <= float(tolerance) * scale))


@lru_cache(maxsize=2)
def cached_scalar_artifact(path: str) -> dict:
    return load_scalar_artifact(path)
