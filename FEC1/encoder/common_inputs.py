"""Route the audited raster into the DIII-D Challenge-1 encoder.

Challenge 1 is an intra-machine problem.  The trainable FEC1 boundary therefore
fails closed on every source except DIII-D and deliberately retains DIII-D
spatial shortcuts that were excluded from the old transfer model.  In
particular, the primary CNN sees the real field-validity footprint and the
spatially aligned core/edge Thomson maps in addition to the physical field.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from src.measurement.coil_calculation import CoilFields
from src.measurement.read_measurements import SynchronizedMeasurements

from .common_raster import (
    CommonRaster,
    _measurement_indices,
    _safe_age_feature,
    build_common_raster,
)


SUPPORTED_ENCODER_SOURCE = "DIII-D"

D3D_MODEL_SPATIAL_CHANNELS = (
    "active_psi_centered_norm",
    "active_b_r_asinh",
    "active_b_z_asinh",
    "active_field_frame_valid",
    "active_field_spatial_valid",
    "canonical_r",
    "canonical_z",
    "cylindrical_r_over_length",
    "cylindrical_z_over_length",
    "thomson_core_log1p_pressure_Pa",
    "thomson_core_valid_support",
    "thomson_edge_log1p_pressure_Pa",
    "thomson_edge_valid_support",
)

# Compatibility name used by the cache/training scripts.  Its contract is now
# explicitly DIII-D-only rather than a lowest-common-denominator machine set.
MODEL_SPATIAL_CHANNELS = D3D_MODEL_SPATIAL_CHANNELS

BASE_SCALAR_CHANNELS = (
    "active_psi_mean_norm",
    "active_field_frame_valid",
    "plasma_current_MA",
    "plasma_current_valid",
    "plasma_current_coverage",
    "plasma_current_fresh",
    "plasma_current_age_log1p",
    "toroidal_current_log1p_abs",
    "toroidal_current_valid",
    "toroidal_current_coverage",
    "toroidal_current_fresh",
    "toroidal_current_age_log1p",
    "solenoid_current_signed_log1p",
    "solenoid_current_valid",
    "solenoid_current_coverage",
    "solenoid_current_fresh",
    "solenoid_current_age_log1p",
    "delta_time_log1p_ms",
    "grid_length_log1p_m",
    "grid_z_half_span_over_length",
    "grid_z_center_over_length",
)

D3D_PF_CURRENT_NAMES = tuple(
    f"magnetics_F{index}{side}"
    for index in range(1, 10)
    for side in ("A", "B")
)

D3D_PF_SCALAR_CHANNELS = tuple(
    feature
    for name in D3D_PF_CURRENT_NAMES
    for feature in (
        f"d3d_{name}_signed_log1p",
        f"d3d_{name}_valid",
        f"d3d_{name}_coverage",
        f"d3d_{name}_fresh",
        f"d3d_{name}_age_log1p",
    )
)

SHARED_SCALAR_CHANNELS = (*BASE_SCALAR_CHANNELS, *D3D_PF_SCALAR_CHANNELS)

# Retained in the diagnostic object only.  It is identically absent for the
# supported DIII-D source and never enters the Challenge-1 model.
AUXILIARY_SCALAR_CHANNELS = (
    "auxiliary_current_signed_log1p",
    "auxiliary_current_valid",
    "auxiliary_current_coverage",
    "auxiliary_current_fresh",
    "auxiliary_current_age_log1p",
)

THOMSON_TOKEN_FEATURES = (
    "canonical_r",
    "canonical_z",
    "log1p_te_eV",
    "log1p_ne_1e18_m3",
    "log1p_pressure_Pa",
    "valid",
    "quality_valid",
    "coverage_valid",
    "fresh",
    "age_log1p",
    "system_core",
    "system_edge",
)


@dataclass(frozen=True)
class CommonScalarInputs:
    values: np.ndarray
    feature_names: tuple[str, ...]

    def feature(self, name: str) -> np.ndarray:
        try:
            index = self.feature_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return self.values[:, index]


@dataclass(frozen=True)
class CommonThomsonTokens:
    """Variable-length, coordinate-aware tokens for a masked set encoder."""

    values: np.ndarray
    attention_mask: np.ndarray
    feature_names: tuple[str, ...]
    token_names: tuple[str, ...]
    token_systems: tuple[str, ...]
    times_ms: np.ndarray

    def feature(self, name: str) -> np.ndarray:
        try:
            index = self.feature_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return self.values[:, :, index]


@dataclass(frozen=True)
class CommonEncoderInputs:
    """One shot routed into spatial, scalar, and Thomson set branches."""

    raster: CommonRaster
    spatial_values: np.ndarray
    spatial_channel_names: tuple[str, ...]
    shared_scalars: CommonScalarInputs
    auxiliary_scalars: CommonScalarInputs
    thomson_tokens: CommonThomsonTokens

    @property
    def field_spatial_validity(self) -> np.ndarray:
        """Diagnostic mask excluded from the primary spatial CNN."""

        return self.raster.channel("active_field_spatial_valid")

    def manifest(self) -> dict[str, object]:
        return {
            "raster": self.raster.manifest(),
            "spatial_channel_names": list(self.spatial_channel_names),
            "shared_scalar_names": list(self.shared_scalars.feature_names),
            "auxiliary_scalar_names": list(self.auxiliary_scalars.feature_names),
            "thomson_token_features": list(self.thomson_tokens.feature_names),
            "thomson_token_count": len(self.thomson_tokens.token_names),
            "routing": {
                "supported_source": SUPPORTED_ENCODER_SOURCE,
                "thomson_splats_in_primary_cnn": True,
                "auxiliary_currents_in_shared_branch": False,
                "field_spatial_mask_in_primary_cnn": True,
            },
        }


@dataclass(frozen=True)
class CommonEncoderFrameBatch:
    """Stacked frames with padded Thomson tokens and per-sample geometry."""

    spatial_values: np.ndarray
    shared_scalars: np.ndarray
    auxiliary_scalars: np.ndarray
    thomson_values: np.ndarray
    thomson_attention_mask: np.ndarray
    field_spatial_validity: np.ndarray
    grid_r_m: np.ndarray
    grid_z_m: np.ndarray
    current_direction_sign: np.ndarray
    times_ms: np.ndarray
    sources: tuple[str, ...]
    spatial_channel_names: tuple[str, ...]
    shared_scalar_names: tuple[str, ...]
    auxiliary_scalar_names: tuple[str, ...]
    thomson_feature_names: tuple[str, ...]


def _select_raster_channels(
    raster: CommonRaster,
    names: tuple[str, ...],
) -> np.ndarray:
    return np.stack([raster.channel(name) for name in names], axis=1).astype(
        np.float32, copy=False
    )


def _scalar_channels(
    raster: CommonRaster,
    names: tuple[str, ...],
) -> CommonScalarInputs:
    columns = []
    for name in names:
        channel = np.asarray(raster.channel(name), dtype=np.float64)
        reference = channel[:, :1, :1]
        if not np.allclose(channel, reference, rtol=0.0, atol=1.0e-7):
            raise ValueError(f"raster channel {name!r} is not spatially constant")
        columns.append(reference[:, 0, 0])
    values = np.column_stack(columns).astype(np.float32)
    return CommonScalarInputs(values=values, feature_names=names)


def _d3d_pf_scalar_channels(
    measurements: SynchronizedMeasurements,
    times_ms: np.ndarray,
) -> CommonScalarInputs:
    indices = _measurement_indices(measurements.times_ms, times_ms)
    columns: list[np.ndarray] = []
    for name in D3D_PF_CURRENT_NAMES:
        if name not in measurements.currents.names:
            columns.extend(
                np.zeros(len(indices), dtype=np.float64) for _ in range(5)
            )
            continue
        column = measurements.currents.index(name)
        current = measurements.currents.values_si[indices, column]
        valid = measurements.currents.valid[indices, column]
        coverage = measurements.currents.coverage_valid[indices, column]
        fresh = measurements.currents.fresh[indices, column]
        encoded = np.sign(current) * np.log1p(np.abs(current) / 1.0e4)
        encoded = np.where(valid, encoded, 0.0)
        age = _safe_age_feature(
            measurements.currents.age_ms[indices, column],
            coverage,
            measurements.max_current_age_ms,
        )
        columns.extend(
            (
                encoded,
                valid.astype(np.float64),
                coverage.astype(np.float64),
                fresh.astype(np.float64),
                age,
            )
        )
    return CommonScalarInputs(
        values=np.column_stack(columns).astype(np.float32),
        feature_names=D3D_PF_SCALAR_CHANNELS,
    )


def build_common_thomson_tokens(
    measurements: SynchronizedMeasurements,
    times_ms: np.ndarray | None = None,
) -> CommonThomsonTokens:
    """Build coordinate-aware tokens without rasterizing chord orientation."""

    if times_ms is None:
        selected_times = np.asarray(measurements.times_ms, dtype=np.float64)
    else:
        selected_times = np.asarray(times_ms, dtype=np.float64).ravel()
    indices = _measurement_indices(measurements.times_ms, selected_times)
    grid_r = np.asarray(measurements.grid_r_m, dtype=np.float64)
    grid_z = np.asarray(measurements.grid_z_m, dtype=np.float64)
    feature_parts: list[np.ndarray] = []
    present_parts: list[np.ndarray] = []
    token_names: list[str] = []
    token_systems: list[str] = []

    for system in ("core", "edge"):
        item = measurements.thomson.get(system)
        if item is None:
            continue
        coordinates = np.asarray(item.coordinates_m, dtype=np.float64)
        xi = 2.0 * (coordinates[:, 0] - grid_r[0]) / (grid_r[-1] - grid_r[0]) - 1.0
        eta = 2.0 * (coordinates[:, 1] - grid_z[0]) / (grid_z[-1] - grid_z[0]) - 1.0
        present = (
            np.isfinite(xi)
            & np.isfinite(eta)
            & (xi >= -1.0)
            & (xi <= 1.0)
            & (eta >= -1.0)
            & (eta <= 1.0)
        )
        # A masked token must still be tensor-safe when callers intentionally
        # disable strict coordinate validation upstream.
        xi = np.where(present, xi, 0.0)
        eta = np.where(present, eta, 0.0)
        count = len(coordinates)
        frame_count = len(indices)
        xi_frames = np.broadcast_to(xi, (frame_count, count))
        eta_frames = np.broadcast_to(eta, (frame_count, count))
        valid = np.asarray(item.valid[indices], dtype=bool) & present[None, :]
        quality = np.asarray(item.quality_valid[indices], dtype=bool) & present[None, :]
        coverage = np.asarray(item.coverage_valid[indices], dtype=bool) & present[None, :]
        fresh = np.asarray(item.fresh[indices], dtype=bool) & coverage
        te = np.where(valid, np.log1p(np.maximum(item.te_eV[indices], 0.0)), 0.0)
        ne = np.where(
            valid,
            np.log1p(np.maximum(item.ne_m3[indices], 0.0) / 1.0e18),
            0.0,
        )
        pressure = np.where(
            valid,
            np.log1p(np.maximum(item.pressure_Pa[indices], 0.0)),
            0.0,
        )
        age = _safe_age_feature(
            item.age_ms[indices], coverage, measurements.max_thomson_age_ms
        )
        system_core = np.full((frame_count, count), float(system == "core"))
        system_edge = np.full((frame_count, count), float(system == "edge"))
        feature_parts.append(
            np.stack(
                [
                    xi_frames,
                    eta_frames,
                    te,
                    ne,
                    pressure,
                    valid.astype(np.float64),
                    quality.astype(np.float64),
                    coverage.astype(np.float64),
                    fresh.astype(np.float64),
                    age,
                    system_core,
                    system_edge,
                ],
                axis=-1,
            )
        )
        present_parts.append(present)
        token_names.extend(item.names)
        token_systems.extend([system] * count)

    if feature_parts:
        values = np.concatenate(feature_parts, axis=1).astype(np.float32)
        present = np.concatenate(present_parts)
    else:
        values = np.zeros(
            (len(indices), 0, len(THOMSON_TOKEN_FEATURES)), dtype=np.float32
        )
        present = np.zeros(0, dtype=bool)
    attention_mask = np.broadcast_to(present, values.shape[:2]).copy()
    if not np.isfinite(values).all():
        raise FloatingPointError("common Thomson tokens contain non-finite values")
    return CommonThomsonTokens(
        values=values,
        attention_mask=attention_mask,
        feature_names=THOMSON_TOKEN_FEATURES,
        token_names=tuple(token_names),
        token_systems=tuple(token_systems),
        times_ms=selected_times.copy(),
    )


def build_common_encoder_inputs(
    measurements: SynchronizedMeasurements,
    active_fields: CoilFields,
    **raster_kwargs,
) -> CommonEncoderInputs:
    """Build DIII-D-only Challenge-1 model inputs.

    ``build_common_raster`` remains useful for machine diagnostics, but the
    trainable FEC1 contract intentionally has no MAST execution path.
    """

    if measurements.source != SUPPORTED_ENCODER_SOURCE:
        raise ValueError(
            "FEC1 Challenge-1 encoder supports DIII-D only; "
            f"received {measurements.source!r}"
        )

    raster = build_common_raster(measurements, active_fields, **raster_kwargs)
    spatial = _select_raster_channels(raster, MODEL_SPATIAL_CHANNELS)
    base = _scalar_channels(raster, BASE_SCALAR_CHANNELS)
    pf = _d3d_pf_scalar_channels(measurements, raster.times_ms)
    shared = CommonScalarInputs(
        values=np.concatenate([base.values, pf.values], axis=1),
        feature_names=SHARED_SCALAR_CHANNELS,
    )
    auxiliary = _scalar_channels(raster, AUXILIARY_SCALAR_CHANNELS)
    tokens = build_common_thomson_tokens(measurements, raster.times_ms)
    return CommonEncoderInputs(
        raster=raster,
        spatial_values=spatial,
        spatial_channel_names=MODEL_SPATIAL_CHANNELS,
        shared_scalars=shared,
        auxiliary_scalars=auxiliary,
        thomson_tokens=tokens,
    )


def stack_common_encoder_frames(
    inputs: Sequence[CommonEncoderInputs],
    frame_indices: Sequence[int] | None = None,
) -> CommonEncoderFrameBatch:
    """Stack one frame per DIII-D shot, padding only Thomson token count."""

    items = tuple(inputs)
    if not items:
        raise ValueError("at least one CommonEncoderInputs object is required")
    unsupported = sorted(
        {item.raster.source for item in items}
        - {SUPPORTED_ENCODER_SOURCE}
    )
    if unsupported:
        raise ValueError(
            "FEC1 Challenge-1 batches are DIII-D-only; received "
            + ", ".join(unsupported)
        )
    if frame_indices is None:
        indices = (0,) * len(items)
    else:
        indices = tuple(int(index) for index in frame_indices)
        if len(indices) != len(items):
            raise ValueError("frame_indices must have one entry per input shot")
    reference = items[0]
    spatial_shape = reference.spatial_values.shape[1:]
    for item, index in zip(items, indices):
        if index < 0:
            raise IndexError("negative frame indices are not accepted by the collator")
        if index >= len(item.raster.times_ms):
            raise IndexError("frame index is outside one input shot")
        if item.spatial_channel_names != reference.spatial_channel_names:
            raise ValueError("spatial channel contracts differ")
        if item.shared_scalars.feature_names != reference.shared_scalars.feature_names:
            raise ValueError("shared scalar contracts differ")
        if item.auxiliary_scalars.feature_names != reference.auxiliary_scalars.feature_names:
            raise ValueError("auxiliary scalar contracts differ")
        if item.thomson_tokens.feature_names != reference.thomson_tokens.feature_names:
            raise ValueError("Thomson token contracts differ")
        if item.spatial_values.shape[1:] != spatial_shape:
            raise ValueError("spatial raster shapes differ")

    batch_size = len(items)
    maximum_tokens = max(item.thomson_tokens.values.shape[1] for item in items)
    token_features = len(reference.thomson_tokens.feature_names)
    thomson_values = np.zeros(
        (batch_size, maximum_tokens, token_features), dtype=np.float32
    )
    thomson_mask = np.zeros((batch_size, maximum_tokens), dtype=bool)
    for batch_index, (item, frame_index) in enumerate(zip(items, indices)):
        token_count = item.thomson_tokens.values.shape[1]
        thomson_values[batch_index, :token_count] = item.thomson_tokens.values[
            frame_index
        ]
        thomson_mask[batch_index, :token_count] = item.thomson_tokens.attention_mask[
            frame_index
        ]

    return CommonEncoderFrameBatch(
        spatial_values=np.stack(
            [item.spatial_values[index] for item, index in zip(items, indices)]
        ).astype(np.float32, copy=False),
        shared_scalars=np.stack(
            [item.shared_scalars.values[index] for item, index in zip(items, indices)]
        ).astype(np.float32, copy=False),
        auxiliary_scalars=np.stack(
            [item.auxiliary_scalars.values[index] for item, index in zip(items, indices)]
        ).astype(np.float32, copy=False),
        thomson_values=thomson_values,
        thomson_attention_mask=thomson_mask,
        field_spatial_validity=np.stack(
            [item.field_spatial_validity[index] for item, index in zip(items, indices)]
        ).astype(np.float32, copy=False),
        grid_r_m=np.stack([item.raster.grid_r_m for item in items]),
        grid_z_m=np.stack([item.raster.grid_z_m for item in items]),
        current_direction_sign=np.asarray(
            [item.raster.current_direction_sign for item in items], dtype=np.float32
        ),
        times_ms=np.asarray(
            [item.raster.times_ms[index] for item, index in zip(items, indices)],
            dtype=np.float64,
        ),
        sources=tuple(item.raster.source for item in items),
        spatial_channel_names=reference.spatial_channel_names,
        shared_scalar_names=reference.shared_scalars.feature_names,
        auxiliary_scalar_names=reference.auxiliary_scalars.feature_names,
        thomson_feature_names=reference.thomson_tokens.feature_names,
    )
