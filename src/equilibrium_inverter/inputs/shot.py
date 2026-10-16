"""Target-free shot preparation through :mod:`src.measurement`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from neuralgs.geometry import MU0
from src.measurement.coil_calculation import CoilFields, calculate_coil_fields
from src.measurement.read_measurements import SynchronizedMeasurements, read_measurements

from ..config import EquilibriumInverterConfig


@dataclass(frozen=True)
class PreparedShot:
    """Calibrated target-free inputs for one shot.

    Thomson arrays use ``[frame, token, feature]`` with raw features
    ``(R_m, Z_m, Te_eV, ne_m3, pressure_Pa, age_ms)``.  Invalid values remain
    zero-filled underneath an explicit mask.
    """

    measurements: SynchronizedMeasurements
    active_fields: CoilFields
    frame_indices: np.ndarray
    thomson_values: np.ndarray
    thomson_valid: np.ndarray
    tf_current_A: np.ndarray
    toroidal_field_response_Tm_per_A: float

    @property
    def source(self) -> str:
        return self.measurements.source

    @property
    def frames(self) -> int:
        return len(self.frame_indices)


@dataclass(frozen=True)
class TorchShot:
    """Device-resident shot tensors consumed by proposal and forward models."""

    times_ms: torch.Tensor
    active_psi_Wb_per_rad: torch.Tensor
    active_b_r_T: torch.Tensor
    active_b_z_T: torch.Tensor
    plasma_current_A: torch.Tensor
    tf_current_A: torch.Tensor
    toroidal_F_Tm: torch.Tensor
    toroidal_response_valid: torch.Tensor
    flux_scale_Wb_per_rad: torch.Tensor
    pressure_scale_Pa: torch.Tensor
    thomson_coordinates_m: torch.Tensor
    thomson_features: torch.Tensor
    thomson_pressure_Pa: torch.Tensor
    thomson_valid: torch.Tensor
    global_features: torch.Tensor
    frame_valid: torch.Tensor

    @property
    def frames(self) -> int:
        return len(self.times_ms)


def _thomson_matrix(
    measurements: SynchronizedMeasurements,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    systems = list(measurements.thomson.values())
    frames = len(selected)
    if not systems:
        return np.zeros((frames, 1, 6), dtype=np.float32), np.zeros((frames, 1), dtype=bool)
    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for system in systems:
        count = len(system.names)
        coordinates = np.broadcast_to(
            np.asarray(system.coordinates_m, dtype=np.float64)[None],
            (frames, count, 2),
        )
        block = np.concatenate(
            [
                coordinates,
                np.asarray(system.te_eV[selected], dtype=np.float64)[..., None],
                np.asarray(system.ne_m3[selected], dtype=np.float64)[..., None],
                np.asarray(system.pressure_Pa[selected], dtype=np.float64)[..., None],
                np.asarray(system.age_ms[selected], dtype=np.float64)[..., None],
            ],
            axis=-1,
        )
        valid = np.asarray(system.valid[selected], dtype=bool)
        block[~valid] = 0.0
        values.append(block)
        masks.append(valid)
    return np.concatenate(values, axis=1).astype(np.float32), np.concatenate(masks, axis=1)


def _tf_current(measurements: SynchronizedMeasurements, selected: np.ndarray) -> np.ndarray:
    candidates = (
        ("magnetics_bcoil",)
        if measurements.source == "DIII-D"
        else ("magnetics_tf_current",)
    )
    for name in candidates:
        if name in measurements.currents.names:
            return np.abs(measurements.currents.column(name)[selected]).astype(np.float64)
    return np.zeros(len(selected), dtype=np.float64)


def _toroidal_field_response(row: Mapping[str, object]) -> float:
    """Read physical TF geometry without consulting a machine identifier."""

    if "toroidal_field_F_per_A" in row:
        response = float(row["toroidal_field_F_per_A"])
    elif "tf_turns" in row:
        response = MU0 * float(row["tf_turns"]) / (2.0 * np.pi)
    else:
        return float("nan")
    if not np.isfinite(response) or response <= 0.0:
        raise ValueError("toroidal-field response must be finite and positive")
    return response


def prepare_shot(
    row: Mapping[str, object],
    config: EquilibriumInverterConfig | None = None,
    *,
    frame_indices: np.ndarray | None = None,
) -> PreparedShot:
    """Read one row without inspecting any EFIT target column."""

    cfg = config or EquilibriumInverterConfig()
    measurements = read_measurements(
        row,
        current_interpolation="previous" if cfg.causal_measurements else "linear",
        thomson_interpolation="previous" if cfg.causal_measurements else "nearest",
    )
    if frame_indices is None:
        selected = np.arange(len(measurements.times_ms), dtype=np.int64)
    else:
        selected = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        if not len(selected):
            raise ValueError("frame_indices cannot be empty")
        if selected.min() < 0 or selected.max() >= len(measurements.times_ms):
            raise IndexError("frame index is outside the synchronized shot")
    active = calculate_coil_fields(
        row,
        measurements,
        frame_indices=selected,
        canonicalize_current_direction=True,
        strict_geometry=cfg.strict_geometry,
    )
    thomson, valid = _thomson_matrix(measurements, selected)
    return PreparedShot(
        measurements=measurements,
        active_fields=active,
        frame_indices=selected,
        thomson_values=thomson,
        thomson_valid=valid,
        tf_current_A=_tf_current(measurements, selected),
        toroidal_field_response_Tm_per_A=_toroidal_field_response(row),
    )


def to_torch_shot(
    prepared: PreparedShot,
    *,
    length_scale_m: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> TorchShot:
    """Normalize a prepared shot using fixed physical similarity scales."""

    selected = prepared.frame_indices
    measurements = prepared.measurements
    ip = np.abs(np.asarray(measurements.plasma_current_A[selected], dtype=np.float64))
    finite_ip = ip[np.isfinite(ip)]
    floor = max(
        0.02 * float(np.percentile(finite_ip, 95.0)) if len(finite_ip) else 0.0,
        1.0e3,
    )
    safe_ip = np.maximum(ip, floor)
    flux_scale = MU0 * safe_ip * float(length_scale_m)
    pressure_scale = MU0 * (safe_ip / float(length_scale_m)) ** 2

    raw = np.asarray(prepared.thomson_values, dtype=np.float64)
    grid_r = np.asarray(measurements.grid_r_m, dtype=np.float64)
    grid_z = np.asarray(measurements.grid_z_m, dtype=np.float64)
    r_span = max(float(grid_r[-1] - grid_r[0]), 1.0e-8)
    z_span = max(float(grid_z[-1] - grid_z[0]), 1.0e-8)
    coordinates = raw[..., :2]
    r_normalized = 2.0 * (coordinates[..., 0] - grid_r[0]) / r_span - 1.0
    z_normalized = 2.0 * (coordinates[..., 1] - grid_z[0]) / z_span - 1.0
    token_features = np.stack(
        [
            r_normalized,
            z_normalized,
            np.log1p(np.maximum(raw[..., 2], 0.0) / 100.0),
            np.log1p(np.maximum(raw[..., 3], 0.0) / 1.0e18),
            np.log1p(np.maximum(raw[..., 4], 0.0) / 1.0e3),
            np.log1p(np.maximum(raw[..., 5], 0.0) / 10.0),
        ],
        axis=-1,
    )

    times_ms = np.asarray(measurements.times_ms[selected], dtype=np.float64)
    if len(times_ms) > 1:
        dip_dt = np.gradient(ip, times_ms / 1000.0)
        phase = 2.0 * (times_ms - times_ms[0]) / max(times_ms[-1] - times_ms[0], 1.0e-6) - 1.0
    else:
        dip_dt = np.zeros_like(ip)
        phase = np.zeros_like(ip)
    active = np.asarray(prepared.active_fields.psi_active_Wb_per_rad, dtype=np.float64)
    active_normalized = active / flux_scale[:, None, None]
    active_mean = np.nanmean(active_normalized, axis=(1, 2))
    active_std = np.nanstd(active_normalized, axis=(1, 2))
    response = float(prepared.toroidal_field_response_Tm_per_A)
    response_valid = np.isfinite(response) and response > 0.0
    toroidal_f = (
        np.abs(prepared.tf_current_A) * response
        if response_valid
        else np.zeros(len(selected), dtype=np.float64)
    )
    toroidal_similarity = toroidal_f / np.maximum(MU0 * safe_ip, 1.0e-8)
    global_features = np.column_stack(
        [
            np.log1p(toroidal_similarity),
            np.full(len(ip), float(response_valid)),
            np.arcsinh(dip_dt / np.maximum(safe_ip, floor)),
            phase,
            active_mean,
            active_std,
            np.full(len(ip), np.sqrt(grid_r[0] * grid_r[-1]) / length_scale_m),
            np.full(len(ip), r_span / z_span),
        ]
    )

    def tensor(values: np.ndarray | list[float], tensor_dtype: torch.dtype = dtype) -> torch.Tensor:
        return torch.as_tensor(values, device=device, dtype=tensor_dtype)

    try:
        ip_column = measurements.currents.index("magnetics_plasma_current")
        ip_valid = np.asarray(
            measurements.currents.valid[selected, ip_column], dtype=bool
        )
    except KeyError:
        ip_valid = np.zeros(len(selected), dtype=bool)
    frame_valid = (
        np.asarray(prepared.active_fields.frame_valid, dtype=bool)
        & ip_valid
        & np.isfinite(ip)
        & np.isfinite(active).all(axis=(1, 2))
    )
    return TorchShot(
        times_ms=tensor(times_ms),
        active_psi_Wb_per_rad=tensor(active),
        active_b_r_T=tensor(prepared.active_fields.b_r_T),
        active_b_z_T=tensor(prepared.active_fields.b_z_T),
        plasma_current_A=tensor(ip),
        tf_current_A=tensor(prepared.tf_current_A),
        toroidal_F_Tm=tensor(toroidal_f),
        toroidal_response_valid=tensor(
            np.full(len(selected), response_valid, dtype=bool), torch.bool
        ),
        flux_scale_Wb_per_rad=tensor(flux_scale),
        pressure_scale_Pa=tensor(pressure_scale),
        thomson_coordinates_m=tensor(coordinates),
        thomson_features=tensor(token_features),
        thomson_pressure_Pa=tensor(raw[..., 4]),
        thomson_valid=tensor(prepared.thomson_valid, torch.bool),
        global_features=tensor(global_features),
        frame_valid=tensor(frame_valid, torch.bool),
    )
