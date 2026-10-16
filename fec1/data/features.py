"""Frozen diagnostic features, field normalization and shot identities."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from fec1.paths import MANIFEST


def _coordinates(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(field, dtype=np.float32)
    mean = values.mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
    centered = values - mean[:, None, None]
    rms = np.sqrt(np.mean(np.square(centered), axis=(1, 2), dtype=np.float64)).astype(np.float32)
    rms = np.maximum(rms, 1.0e-8)
    return mean, rms, (centered / rms[:, None, None]).astype(np.float32)


def _centered_context(values: np.ndarray, width: int = 8) -> tuple[np.ndarray, np.ndarray]:
    values64 = np.asarray(values, dtype=np.float64)
    count = len(values64)
    mean = np.empty_like(values64)
    std = np.empty_like(values64)
    left = (width - 1) // 2
    for target in range(count):
        start = target - left
        stop = start + width
        if start < 0:
            stop -= start
            start = 0
        if stop > count:
            start = max(0, start - (stop - count))
            stop = count
        local = values64[start:stop]
        mean[target] = local.mean(axis=0)
        std[target] = local.std(axis=0)
    return mean.astype(np.float32), std.astype(np.float32)


def _features(saved: dict[str, np.ndarray]) -> np.ndarray:
    scalar = build_scalar_features(saved)
    canonical = np.asarray(saved["canonical_global_features"], dtype=np.float32)
    context_mean, context_std = _centered_context(canonical)
    features = np.column_stack((canonical, context_mean, context_std, scalar))
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _read(path: Path, include_targets: bool = True) -> dict[str, np.ndarray]:
    input_keys = (
        "vacuum_psi",
        "global_features",
        "canonical_global_features",
        "token_coords",
        "token_values",
        "token_mask",
        "grid_r",
        "grid_z",
    )
    target_keys = ("target_psi", "target_q95", "target_betaN")
    with np.load(path, allow_pickle=False) as saved:
        keys = input_keys + target_keys if include_targets else input_keys
        return {key: np.asarray(saved[key]) for key in keys}


def _pca_sample(task: tuple[str, int]) -> np.ndarray:
    path_text, per_shot = task
    data = _read(Path(path_text), include_targets=True)
    count = len(data["target_psi"])
    chosen = np.unique(np.linspace(0, count - 1, min(per_shot, count)).round().astype(np.int64))
    return _coordinates(data["target_psi"][chosen])[2]


def _split_records() -> tuple[list[str], set[str], set[str]]:
    manifest = json.loads(MANIFEST.read_text())
    train = [record["shot"] for record in manifest["records"] if record["split"] == "train"]
    development = {
        record["shot"] for record in manifest["records"] if record["split"] == "development"
    }
    frozen = {record["shot"] for record in manifest["records"] if record["split"] == "frozen_audit"}
    if set(train) & development or set(train) & frozen or development & frozen:
        raise RuntimeError("immutable shot split overlaps")
    return train, development, frozen


def _trees(estimators: int, leaf: int, max_features: float, seed: int):
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=estimators,
        min_samples_leaf=leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=seed,
    )


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    total = np.square(y - y.mean()).sum(dtype=np.float64)
    return float(1.0 - np.square(y - p).sum(dtype=np.float64) / max(total, 1.0e-30))


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
        np.where(mask[..., None], (values[..., :4] - mean[:, None]) ** 2, 0.0).sum(axis=1) / denom
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
            (pressure_weight[..., None] * (normalized_coords - centroid[:, None]) ** 2).sum(axis=1)
            / weight_sum[:, None],
            0.0,
        )
    )
    centroid[empty] = 0.0
    width[empty] = 0.0
    valid_fraction = (count / max(mask.shape[1], 1))[:, None]
    return np.column_stack([valid_fraction, mean, std, minimum, maximum, centroid, width])


def build_scalar_features(prepared: dict[str, np.ndarray]) -> np.ndarray:
    """Build fixed-size q95/betaN features from the FEC1 tensor contract."""

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
        token_parts.append(_masked_token_features(values, coords, group_mask, grid_r, grid_z))

    grid_values = np.asarray(
        [
            grid_r[0],
            grid_r[-1],
            grid_z[0],
            grid_z[-1],
            grid_r[-1] - grid_r[0],
            grid_z[-1] - grid_z[0],
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
