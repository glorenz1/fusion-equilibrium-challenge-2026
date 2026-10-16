"""Build one target-free spatial input contract for DIII-D and MAST.

The two machines have different coil inventories and physical ``(R, Z)``
domains.  This module removes those superficial differences without pretending
that their geometry is identical:

* active currents are represented by their calibrated physical fields;
* machine-specific global current names are mapped to shared physical roles;
* Thomson points are splatted in normalized grid coordinates with explicit
  quality, coverage, freshness, and age channels; and
* physical scale and cylindrical metric channels remain available to the
  encoder and decoder.

No equilibrium target is accepted by this API.  Causality is selected upstream
by calling :func:`src.measurement.read_measurements.read_measurements` with
``current_interpolation="previous"`` and ``thomson_interpolation="previous"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from neuralgs.geometry import MU0
from src.measurement.coil_calculation import CoilFields
from src.measurement.read_measurements import (
    CurrentMeasurements,
    SynchronizedMeasurements,
    ThomsonMeasurements,
)


COMMON_RASTER_VERSION = "2.0.0"
DEFAULT_RASTER_HEIGHT = 65
DEFAULT_RASTER_WIDTH = 65
DEFAULT_THOMSON_SIGMA_PIXELS = 2.0
DEFAULT_THOMSON_CUTOFF_SIGMA = 3.0
DEFAULT_CURRENT_FLOOR_A = 1.0e4
DEFAULT_CURRENT_LOG_SCALE_A = 1.0e4
DEFAULT_FIELD_ASINH_SCALE = 0.1
DEFAULT_FIELD_ASINH_CLIP = 5.0

_CURRENT_ROLES: dict[str, dict[str, str | None]] = {
    "DIII-D": {
        "toroidal": "magnetics_bcoil",
        "solenoid": "magnetics_ECOILA",
        "auxiliary": None,
    },
    "MAST": {
        "toroidal": "magnetics_tf_current",
        "solenoid": "magnetics_sol_current",
        "auxiliary": "magnetics_efps_current",
    },
}

_BASE_CHANNELS = (
    "active_psi_centered_norm",
    "active_psi_mean_norm",
    "active_b_r_asinh",
    "active_b_z_asinh",
    "active_field_frame_valid",
    "active_field_spatial_valid",
    "plasma_current_MA",
    "plasma_current_valid",
    "plasma_current_coverage",
    "plasma_current_fresh",
    "plasma_current_age_log1p",
)

_ROLE_STATUS_SUFFIXES = (
    "current_valid",
    "current_coverage",
    "current_fresh",
    "current_age_log1p",
)

_ROLE_CHANNELS = tuple(
    channel
    for role in ("toroidal", "solenoid", "auxiliary")
    for channel in (
        f"{role}_{'current_log1p_abs' if role == 'toroidal' else 'current_signed_log1p'}",
        *(f"{role}_{suffix}" for suffix in _ROLE_STATUS_SUFFIXES),
    )
)

_TIME_AND_GEOMETRY_CHANNELS = (
    "delta_time_log1p_ms",
    "canonical_r",
    "canonical_z",
    "cylindrical_r_over_length",
    "cylindrical_z_over_length",
    "grid_length_log1p_m",
    "grid_z_half_span_over_length",
    "grid_z_center_over_length",
)

_THOMSON_CHANNEL_SUFFIXES = (
    "log1p_te_eV",
    "log1p_ne_1e18_m3",
    "log1p_pressure_Pa",
    "valid_support",
    "quality_support",
    "coverage_support",
    "fresh_support",
    "age_log1p",
)

COMMON_RASTER_CHANNELS = (
    *_BASE_CHANNELS,
    *_ROLE_CHANNELS,
    *_TIME_AND_GEOMETRY_CHANNELS,
    *(
        f"thomson_{system}_{suffix}"
        for system in ("core", "edge")
        for suffix in _THOMSON_CHANNEL_SUFFIXES
    ),
)


@dataclass(frozen=True)
class CommonRaster:
    """Canonical encoder canvas and the physical mapping needed to decode it.

    ``values`` has shape ``(time, channel, z, r)``.  ``grid_r_m`` and
    ``grid_z_m`` are the physical coordinates represented by the common
    canvas for this shot; they intentionally differ between machines.
    """

    values: np.ndarray
    channel_names: tuple[str, ...]
    times_ms: np.ndarray
    canonical_r: np.ndarray
    canonical_z: np.ndarray
    grid_r_m: np.ndarray
    grid_z_m: np.ndarray
    source: str
    current_direction_sign: float
    length_scale_m: float
    unit_schema: str
    max_current_age_ms: float | None
    max_thomson_age_ms: float | None
    field_quadrature_order: int
    field_strict_geometry: bool
    field_approximations: tuple[str, ...]
    thomson_sigma_pixels: float
    thomson_cutoff_sigma: float
    current_floor_A: float
    field_asinh_scale: float
    field_asinh_clip: float
    version: str = COMMON_RASTER_VERSION

    def channel_index(self, name: str) -> int:
        try:
            return self.channel_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc

    def channel(self, name: str) -> np.ndarray:
        """Return one named channel with shape ``(time, z, r)``."""

        return self.values[:, self.channel_index(name)]

    def manifest(self) -> dict[str, object]:
        """Return the serializable part of the input contract."""

        return {
            "version": self.version,
            "source": self.source,
            "shape": list(self.values.shape),
            "channel_names": list(self.channel_names),
            "current_direction_sign": self.current_direction_sign,
            "length_scale_m": self.length_scale_m,
            "unit_schema": self.unit_schema,
            "max_current_age_ms": self.max_current_age_ms,
            "max_thomson_age_ms": self.max_thomson_age_ms,
            "field_quadrature_order": self.field_quadrature_order,
            "field_strict_geometry": self.field_strict_geometry,
            "field_approximations": list(self.field_approximations),
            "thomson_sigma_pixels": self.thomson_sigma_pixels,
            "thomson_cutoff_sigma": self.thomson_cutoff_sigma,
            "current_floor_A": self.current_floor_A,
            "field_asinh_scale": self.field_asinh_scale,
            "field_asinh_clip": self.field_asinh_clip,
            "grid_r_range_m": [float(self.grid_r_m[0]), float(self.grid_r_m[-1])],
            "grid_z_range_m": [float(self.grid_z_m[0]), float(self.grid_z_m[-1])],
        }


def _validate_grid(name: str, values: np.ndarray) -> np.ndarray:
    grid = np.asarray(values, dtype=np.float64).ravel()
    if len(grid) < 2 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing with at least two points")
    return grid


def _measurement_indices(
    measurement_times_ms: np.ndarray,
    field_times_ms: np.ndarray,
) -> np.ndarray:
    measurement_times = np.asarray(measurement_times_ms, dtype=np.float64).ravel()
    field_times = np.asarray(field_times_ms, dtype=np.float64).ravel()
    if not len(field_times):
        return np.empty(0, dtype=np.int64)
    right = np.searchsorted(measurement_times, field_times)
    clipped = np.clip(right, 0, len(measurement_times) - 1)
    previous = np.clip(right - 1, 0, len(measurement_times) - 1)
    choose_previous = (
        np.abs(measurement_times[previous] - field_times)
        < np.abs(measurement_times[clipped] - field_times)
    )
    indices = np.where(choose_previous, previous, clipped)
    tolerance = 32.0 * np.finfo(np.float64).eps * np.maximum(
        1.0, np.abs(field_times)
    )
    if np.any(np.abs(measurement_times[indices] - field_times) > tolerance):
        raise ValueError("coil-field times are not a subset of measurement times")
    return indices.astype(np.int64)


def _resample_rectilinear(
    values: np.ndarray,
    source_r: np.ndarray,
    source_z: np.ndarray,
    target_r: np.ndarray,
    target_z: np.ndarray,
) -> np.ndarray:
    """Separable linear interpolation for arrays ending in ``(Z, R)``."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape[-2:] != (len(source_z), len(source_r)):
        raise ValueError("array trailing dimensions do not match the source grid")
    flat = array.reshape((-1, len(source_z), len(source_r)))
    radial = np.empty((len(flat), len(source_z), len(target_r)), dtype=np.float64)
    for sample in range(len(flat)):
        for z_index in range(len(source_z)):
            radial[sample, z_index] = np.interp(
                target_r, source_r, flat[sample, z_index]
            )
    output = np.empty((len(flat), len(target_z), len(target_r)), dtype=np.float64)
    for sample in range(len(flat)):
        for r_index in range(len(target_r)):
            output[sample, :, r_index] = np.interp(
                target_z, source_z, radial[sample, :, r_index]
            )
    return output.reshape((*array.shape[:-2], len(target_z), len(target_r)))


def _broadcast_frames(values: np.ndarray, height: int, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape((-1, 1, 1))
    return np.broadcast_to(array, (len(array), height, width)).copy()


def _inpaint_nearest(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill small invalid spatial regions without exposing hard zero holes.

    The validity channel remains authoritative.  This fill only prevents a
    convolution from treating a conductor-mask edge as a physical field edge.
    Entirely invalid frames remain zero.
    """

    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if array.shape != mask.shape or array.ndim != 3:
        raise ValueError("inpainting expects matching (time, Z, R) arrays")
    output = np.zeros_like(array)
    for frame in range(len(array)):
        if not np.any(mask[frame]):
            continue
        indices = distance_transform_edt(
            ~mask[frame],
            return_distances=False,
            return_indices=True,
        )
        output[frame] = array[frame][tuple(indices)]
    return output


def _safe_age_feature(
    ages_ms: np.ndarray,
    coverage: np.ndarray,
    reference_ms: float | None,
) -> np.ndarray:
    scale = max(float(reference_ms) if reference_ms is not None else 1.0, 1.0e-6)
    age = np.asarray(ages_ms, dtype=np.float64)
    valid = np.asarray(coverage, dtype=bool) & np.isfinite(age) & (age >= 0.0)
    result = np.zeros_like(age, dtype=np.float64)
    result[valid] = np.log1p(age[valid] / scale)
    return result


def _signed_log_current(values_A: np.ndarray) -> np.ndarray:
    values = np.asarray(values_A, dtype=np.float64)
    return np.sign(values) * np.log1p(np.abs(values) / DEFAULT_CURRENT_LOG_SCALE_A)


def _current_role_frames(
    currents: CurrentMeasurements,
    source: str,
    role: str,
    indices: np.ndarray,
    current_direction_sign: float,
    age_reference_ms: float | None,
) -> tuple[np.ndarray, ...]:
    key = _CURRENT_ROLES[source][role]
    count = len(indices)
    zeros = np.zeros(count, dtype=np.float64)
    if key is None or key not in currents.names:
        return zeros, zeros, zeros, zeros, zeros
    column = currents.index(key)
    values = currents.values_si[indices, column]
    valid = currents.valid[indices, column]
    coverage = currents.coverage_valid[indices, column]
    fresh = currents.fresh[indices, column]
    if role == "toroidal":
        encoded = np.log1p(np.abs(values) / DEFAULT_CURRENT_LOG_SCALE_A)
    else:
        encoded = _signed_log_current(current_direction_sign * values)
    encoded = np.where(valid, encoded, 0.0)
    age = _safe_age_feature(
        currents.age_ms[indices, column], coverage, age_reference_ms
    )
    return (
        encoded,
        valid.astype(np.float64),
        coverage.astype(np.float64),
        fresh.astype(np.float64),
        age,
    )


def _thomson_weights(
    measurements: ThomsonMeasurements,
    source_r: np.ndarray,
    source_z: np.ndarray,
    canonical_r: np.ndarray,
    canonical_z: np.ndarray,
    sigma_pixels: float,
    cutoff_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(measurements.coordinates_m, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Thomson coordinates must have shape (channel, 2)")
    xi = 2.0 * (coordinates[:, 0] - source_r[0]) / (source_r[-1] - source_r[0]) - 1.0
    eta = 2.0 * (coordinates[:, 1] - source_z[0]) / (source_z[-1] - source_z[0]) - 1.0
    coordinate_valid = (
        np.isfinite(xi)
        & np.isfinite(eta)
        & (xi >= -1.0)
        & (xi <= 1.0)
        & (eta >= -1.0)
        & (eta <= 1.0)
    )
    xx, yy = np.meshgrid(canonical_r, canonical_z)
    dx_pixels = (xx[None] - xi[:, None, None]) * 0.5 * (len(canonical_r) - 1)
    dy_pixels = (yy[None] - eta[:, None, None]) * 0.5 * (len(canonical_z) - 1)
    distance_squared = dx_pixels**2 + dy_pixels**2
    weights = np.exp(-0.5 * distance_squared / sigma_pixels**2)
    weights[distance_squared > (cutoff_sigma * sigma_pixels) ** 2] = 0.0
    weights[~coordinate_valid] = 0.0
    return weights, coordinate_valid


def _weighted_average(
    weights: np.ndarray,
    mask: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    selected = weights * np.asarray(mask, dtype=bool)[:, None, None]
    support = np.sum(selected, axis=0)
    numerator = np.sum(selected * np.asarray(values, dtype=np.float64)[:, None, None], axis=0)
    result = np.zeros_like(support)
    np.divide(numerator, support, out=result, where=support > 1.0e-12)
    return result


def _support(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = weights * np.asarray(mask, dtype=bool)[:, None, None]
    return 1.0 - np.exp(-np.sum(selected, axis=0))


def _thomson_system_frames(
    measurements: ThomsonMeasurements | None,
    indices: np.ndarray,
    source_r: np.ndarray,
    source_z: np.ndarray,
    canonical_r: np.ndarray,
    canonical_z: np.ndarray,
    sigma_pixels: float,
    cutoff_sigma: float,
    age_reference_ms: float | None,
) -> tuple[np.ndarray, ...]:
    shape = (len(indices), len(canonical_z), len(canonical_r))
    if measurements is None:
        return tuple(np.zeros(shape, dtype=np.float64) for _ in _THOMSON_CHANNEL_SUFFIXES)
    weights, coordinate_valid = _thomson_weights(
        measurements,
        source_r,
        source_z,
        canonical_r,
        canonical_z,
        sigma_pixels,
        cutoff_sigma,
    )
    channels = [np.zeros(shape, dtype=np.float64) for _ in _THOMSON_CHANNEL_SUFFIXES]
    for output_frame, measurement_frame in enumerate(indices):
        valid = coordinate_valid & measurements.valid[measurement_frame]
        quality = coordinate_valid & measurements.quality_valid[measurement_frame]
        coverage = coordinate_valid & measurements.coverage_valid[measurement_frame]
        fresh = coverage & measurements.fresh[measurement_frame]
        te = np.log1p(np.maximum(measurements.te_eV[measurement_frame], 0.0))
        ne = np.log1p(
            np.maximum(measurements.ne_m3[measurement_frame], 0.0) / 1.0e18
        )
        pressure = np.log1p(
            np.maximum(measurements.pressure_Pa[measurement_frame], 0.0)
        )
        age = _safe_age_feature(
            measurements.age_ms[measurement_frame], coverage, age_reference_ms
        )
        channels[0][output_frame] = _weighted_average(weights, valid, te)
        channels[1][output_frame] = _weighted_average(weights, valid, ne)
        channels[2][output_frame] = _weighted_average(weights, valid, pressure)
        channels[3][output_frame] = _support(weights, valid)
        channels[4][output_frame] = _support(weights, quality)
        channels[5][output_frame] = _support(weights, coverage)
        channels[6][output_frame] = _support(weights, fresh)
        channels[7][output_frame] = _weighted_average(weights, coverage, age)
    return tuple(channels)


def build_common_raster(
    measurements: SynchronizedMeasurements,
    active_fields: CoilFields,
    *,
    height: int = DEFAULT_RASTER_HEIGHT,
    width: int = DEFAULT_RASTER_WIDTH,
    thomson_sigma_pixels: float = DEFAULT_THOMSON_SIGMA_PIXELS,
    thomson_cutoff_sigma: float = DEFAULT_THOMSON_CUTOFF_SIGMA,
    current_floor_A: float = DEFAULT_CURRENT_FLOOR_A,
    field_asinh_scale: float = DEFAULT_FIELD_ASINH_SCALE,
    field_asinh_clip: float = DEFAULT_FIELD_ASINH_CLIP,
) -> CommonRaster:
    """Render calibrated measurements into the shared DIII-D/MAST canvas.

    Active fields are canonicalized to the plasma-current direction if the
    supplied :class:`CoilFields` has not already been canonicalized.  The
    returned ``current_direction_sign`` must be retained so a downstream
    decoder can restore the source convention.
    """

    if measurements.source not in _CURRENT_ROLES:
        raise ValueError(f"unsupported source: {measurements.source}")
    if height < 2 or width < 2:
        raise ValueError("common raster height and width must both be at least two")
    if thomson_sigma_pixels <= 0.0 or thomson_cutoff_sigma <= 0.0:
        raise ValueError("Thomson sigma and cutoff must be positive")
    if current_floor_A <= 0.0:
        raise ValueError("current_floor_A must be positive")
    if field_asinh_scale <= 0.0 or field_asinh_clip <= 0.0:
        raise ValueError("field asinh scale and clip must be positive")

    source_r = _validate_grid("measurement grid R", measurements.grid_r_m)
    source_z = _validate_grid("measurement grid Z", measurements.grid_z_m)
    field_r = _validate_grid("coil-field grid R", active_fields.grid_r_m)
    field_z = _validate_grid("coil-field grid Z", active_fields.grid_z_m)
    if not np.array_equal(source_r, field_r) or not np.array_equal(source_z, field_z):
        raise ValueError("measurement and active-field physical grids differ")
    expected_field_shape = (
        len(active_fields.times_ms),
        len(source_z),
        len(source_r),
    )
    for name, values in (
        ("psi", active_fields.psi_active_Wb_per_rad),
        ("B_R", active_fields.b_r_T),
        ("B_Z", active_fields.b_z_T),
    ):
        if np.asarray(values).shape != expected_field_shape:
            raise ValueError(f"active {name} has shape {np.asarray(values).shape}, expected {expected_field_shape}")
    if np.asarray(active_fields.frame_valid).shape != (len(active_fields.times_ms),):
        raise ValueError("active-field frame_valid has the wrong shape")
    if np.asarray(active_fields.field_valid_mask).shape != (len(source_z), len(source_r)):
        raise ValueError("active-field spatial validity mask has the wrong shape")

    measurement_indices = _measurement_indices(
        measurements.times_ms, active_fields.times_ms
    )
    frame_count = len(measurement_indices)
    canonical_r = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    canonical_z = np.linspace(-1.0, 1.0, height, dtype=np.float64)
    target_r = np.linspace(source_r[0], source_r[-1], width, dtype=np.float64)
    target_z = np.linspace(source_z[0], source_z[-1], height, dtype=np.float64)
    rr, zz = np.meshgrid(target_r, target_z)
    length_scale = 0.5 * float(source_r[-1] - source_r[0])
    z_center = 0.5 * float(source_z[0] + source_z[-1])
    z_half_span = 0.5 * float(source_z[-1] - source_z[0])
    direction = float(measurements.current_direction_sign)
    if not np.isfinite(direction) or abs(direction) != 1.0:
        raise ValueError("current_direction_sign must be finite and equal to -1 or +1")

    ip_column = measurements.currents.index("magnetics_plasma_current")
    ip = direction * measurements.plasma_current_A[measurement_indices]
    ip_valid = measurements.currents.valid[measurement_indices, ip_column]
    ip_coverage = measurements.currents.coverage_valid[measurement_indices, ip_column]
    ip_fresh = measurements.currents.fresh[measurement_indices, ip_column]
    ip_age = _safe_age_feature(
        measurements.currents.age_ms[measurement_indices, ip_column],
        ip_coverage,
        measurements.max_current_age_ms,
    )
    current_scale = np.maximum(np.abs(ip), current_floor_A)
    flux_scale = MU0 * current_scale * length_scale
    field_scale = MU0 * current_scale / length_scale

    canonicalization = 1.0 if active_fields.canonicalized else direction
    spatial_valid = np.asarray(active_fields.field_valid_mask, dtype=bool)
    frame_valid = np.asarray(active_fields.frame_valid, dtype=bool)
    psi_native = canonicalization * np.asarray(
        active_fields.psi_active_Wb_per_rad, dtype=np.float64
    )
    b_r_native = canonicalization * np.asarray(active_fields.b_r_T, dtype=np.float64)
    b_z_native = canonicalization * np.asarray(active_fields.b_z_T, dtype=np.float64)
    finite_native = (
        np.isfinite(psi_native) & np.isfinite(b_r_native) & np.isfinite(b_z_native)
    )
    field_mask_native = (
        frame_valid[:, None, None]
        & spatial_valid[None, :, :]
        & finite_native
    )
    psi_mean = np.zeros(frame_count, dtype=np.float64)
    for frame in range(frame_count):
        if np.any(field_mask_native[frame]):
            psi_mean[frame] = np.mean(psi_native[frame][field_mask_native[frame]])
    effective_frame_valid = frame_valid & np.any(field_mask_native, axis=(1, 2))
    psi_centered_native = _inpaint_nearest(
        psi_native - psi_mean[:, None, None], field_mask_native
    )
    b_r_native = _inpaint_nearest(b_r_native, field_mask_native)
    b_z_native = _inpaint_nearest(b_z_native, field_mask_native)
    spatial_field_mask = _resample_rectilinear(
        field_mask_native.astype(np.float64), source_r, source_z, target_r, target_z
    )
    spatial_field_mask = np.clip(spatial_field_mask, 0.0, 1.0)
    psi_centered = _resample_rectilinear(
        psi_centered_native, source_r, source_z, target_r, target_z
    ) / flux_scale[:, None, None]
    b_r_dimensionless = _resample_rectilinear(
        b_r_native, source_r, source_z, target_r, target_z
    ) / field_scale[:, None, None]
    b_z_dimensionless = _resample_rectilinear(
        b_z_native, source_r, source_z, target_r, target_z
    ) / field_scale[:, None, None]
    b_r = np.clip(
        np.arcsinh(b_r_dimensionless / field_asinh_scale),
        -field_asinh_clip,
        field_asinh_clip,
    )
    b_z = np.clip(
        np.arcsinh(b_z_dimensionless / field_asinh_scale),
        -field_asinh_clip,
        field_asinh_clip,
    )

    channels: list[np.ndarray] = [
        psi_centered,
        _broadcast_frames(psi_mean / flux_scale, height, width),
        b_r,
        b_z,
        _broadcast_frames(effective_frame_valid.astype(np.float64), height, width),
        spatial_field_mask,
        _broadcast_frames(np.where(ip_valid, ip / 1.0e6, 0.0), height, width),
        _broadcast_frames(ip_valid.astype(np.float64), height, width),
        _broadcast_frames(ip_coverage.astype(np.float64), height, width),
        _broadcast_frames(ip_fresh.astype(np.float64), height, width),
        _broadcast_frames(ip_age, height, width),
    ]
    for role in ("toroidal", "solenoid", "auxiliary"):
        for values in _current_role_frames(
            measurements.currents,
            measurements.source,
            role,
            measurement_indices,
            direction,
            measurements.max_current_age_ms,
        ):
            channels.append(_broadcast_frames(values, height, width))

    times = np.asarray(active_fields.times_ms, dtype=np.float64)
    delta_time = np.zeros(frame_count, dtype=np.float64)
    if frame_count > 1:
        delta_time[1:] = np.diff(times)
    channels.extend(
        [
            _broadcast_frames(np.log1p(np.maximum(delta_time, 0.0)), height, width),
            np.broadcast_to(canonical_r[None, None, :], (frame_count, height, width)).copy(),
            np.broadcast_to(canonical_z[None, :, None], (frame_count, height, width)).copy(),
            np.broadcast_to((rr / length_scale)[None], (frame_count, height, width)).copy(),
            np.broadcast_to(((zz - z_center) / length_scale)[None], (frame_count, height, width)).copy(),
            np.full((frame_count, height, width), np.log1p(length_scale), dtype=np.float64),
            np.full((frame_count, height, width), z_half_span / length_scale, dtype=np.float64),
            np.full((frame_count, height, width), z_center / length_scale, dtype=np.float64),
        ]
    )

    for system in ("core", "edge"):
        channels.extend(
            _thomson_system_frames(
                measurements.thomson.get(system),
                measurement_indices,
                source_r,
                source_z,
                canonical_r,
                canonical_z,
                thomson_sigma_pixels,
                thomson_cutoff_sigma,
                measurements.max_thomson_age_ms,
            )
        )

    values = np.stack(channels, axis=1)
    if values.shape != (frame_count, len(COMMON_RASTER_CHANNELS), height, width):
        raise AssertionError("common raster channel contract and implementation diverged")
    if not np.isfinite(values).all():
        raise FloatingPointError("common raster contains non-finite values")
    return CommonRaster(
        values=values.astype(np.float32),
        channel_names=COMMON_RASTER_CHANNELS,
        times_ms=times.copy(),
        canonical_r=canonical_r.astype(np.float32),
        canonical_z=canonical_z.astype(np.float32),
        grid_r_m=target_r.astype(np.float64),
        grid_z_m=target_z.astype(np.float64),
        source=measurements.source,
        current_direction_sign=direction,
        length_scale_m=length_scale,
        unit_schema=measurements.unit_schema,
        max_current_age_ms=measurements.max_current_age_ms,
        max_thomson_age_ms=measurements.max_thomson_age_ms,
        field_quadrature_order=int(active_fields.quadrature_order),
        field_strict_geometry=bool(active_fields.strict_geometry),
        field_approximations=tuple(active_fields.approximations),
        thomson_sigma_pixels=float(thomson_sigma_pixels),
        thomson_cutoff_sigma=float(thomson_cutoff_sigma),
        current_floor_A=float(current_floor_A),
        field_asinh_scale=float(field_asinh_scale),
        field_asinh_clip=float(field_asinh_clip),
    )
