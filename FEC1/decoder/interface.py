"""Validated handoff from audited encoder inputs to the physical decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neuralgs.geometry import MU0
from FEC1.encoder.common_inputs import CommonEncoderInputs

from .physical import SpatialDecoderState


@dataclass(frozen=True)
class DecoderFrameInputs:
    active_psi_Wb_per_rad: np.ndarray
    plasma_current_magnitude_A: float
    grid_r_m: np.ndarray
    grid_z_m: np.ndarray
    current_direction_sign: float
    source: str
    time_ms: float

    def restore_source_flux_convention(self, canonical_psi: np.ndarray) -> np.ndarray:
        return self.current_direction_sign * np.asarray(canonical_psi)


def decoder_frame_from_encoder(
    inputs: CommonEncoderInputs,
    frame_index: int,
) -> DecoderFrameInputs:
    """Recover exact-scale decoder inputs from one target-free encoder frame."""

    index = int(frame_index)
    if index < 0 or index >= len(inputs.raster.times_ms):
        raise IndexError("frame_index is outside the encoder input")
    shared = inputs.shared_scalars
    required = (
        "active_psi_mean_norm",
        "active_field_frame_valid",
        "plasma_current_MA",
        "plasma_current_valid",
        "plasma_current_coverage",
        "plasma_current_fresh",
    )
    values = {name: float(shared.feature(name)[index]) for name in required}
    if values["active_field_frame_valid"] < 0.5:
        raise ValueError("active coil field is invalid for this frame")
    if any(values[name] < 0.5 for name in (
        "plasma_current_valid", "plasma_current_coverage", "plasma_current_fresh"
    )):
        raise ValueError("plasma current is invalid, uncovered, or stale")
    ip = abs(values["plasma_current_MA"]) * 1.0e6
    flux_scale = MU0 * ip * float(inputs.raster.length_scale_m)
    centered = np.asarray(
        inputs.raster.channel("active_psi_centered_norm")[index], dtype=np.float64
    )
    active = flux_scale * (centered + values["active_psi_mean_norm"])
    if not np.isfinite(active).all() or ip <= 0.0:
        raise FloatingPointError("decoder frame contains invalid physical values")
    return DecoderFrameInputs(
        active_psi_Wb_per_rad=active,
        plasma_current_magnitude_A=ip,
        grid_r_m=np.asarray(inputs.raster.grid_r_m, dtype=np.float64),
        grid_z_m=np.asarray(inputs.raster.grid_z_m, dtype=np.float64),
        current_direction_sign=float(inputs.raster.current_direction_sign),
        source=inputs.raster.source,
        time_ms=float(inputs.raster.times_ms[index]),
    )


def _to_numpy(values: Any) -> np.ndarray:
    """Convert NumPy or an inference-time tensor without importing torch."""

    item = values
    if hasattr(item, "detach"):
        item = item.detach()
    if hasattr(item, "cpu"):
        item = item.cpu()
    if hasattr(item, "numpy"):
        item = item.numpy()
    return np.asarray(item)


def spatial_decoder_state_from_encoder(
    output: Any,
    *,
    batch_index: int = 0,
    frame_index: int = 0,
) -> SpatialDecoderState:
    """Extract one deployable state from ``CausalHybridEncoderOutput``."""

    batch = int(batch_index)
    frame = int(frame_index)
    try:
        current_mass = _to_numpy(output.current_mass[batch, frame])
        harmonics = _to_numpy(output.harmonics[batch, frame])
        gauge = _to_numpy(output.gauge[batch, frame]).reshape(1)
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("output does not satisfy the causal encoder contract") from exc
    return SpatialDecoderState(
        current_mass=current_mass[None],
        harmonic_coefficients=harmonics[None],
        gauge_Wb_per_rad=gauge,
    )
