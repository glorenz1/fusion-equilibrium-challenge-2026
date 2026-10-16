#!/usr/bin/env python3
"""Calibrate and synchronize challenge inputs without reading EFIT targets.

The public row schema mixes dense and gapped time bases, machine-specific
current conventions, ragged Thomson arrays, and per-shot chord geometry.  This
module turns one raw row into a finite, mask-carrying representation on the
row's ``efit_times`` axis.  It intentionally does not inspect any EFIT target.

All returned times are milliseconds, coordinates are metres, temperatures are
electron-volts, densities are m^-3, pressure is pascals, and currents are SI
amperes or ampere-turns as recorded in ``CurrentMeasurements.units``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from .data_fixes import (
    align_mast_thomson_core,
    fix_d3d_ip_times,
    fix_d3d_thomson_edge_padding,
)
from neuralgs.geometry import canonical_current_sign

ELECTRON_CHARGE_C = 1.602176634e-19
InterpolationMode = Literal["linear", "previous", "nearest"]
ThomsonInterpolationMode = Literal["previous", "nearest"]
DEFAULT_UNIT_SCHEMA = "fusion-challenge-v1.1-kA"
MEASUREMENT_READER_VERSION = "1.0.0"
DEFAULT_MAX_CURRENT_AGE_MS = 2.0
DEFAULT_MAX_THOMSON_AGE_MS = 50.0


@dataclass(frozen=True)
class CurrentConvention:
    storage_unit: str
    physical_unit: str
    scale_to_si: float


_D3D_PF_KEYS = tuple(
    f"magnetics_F{index}{side}"
    for index in range(1, 10)
    for side in ("A", "B")
)
_MAST_PF_KEYS = tuple(
    f"magnetics_p{index}{side}_current"
    for index in range(2, 7)
    for side in ("l", "u")
)
_SUPPORTED_CURRENT_KEYS = {
    "DIII-D": set(_D3D_PF_KEYS)
    | {"magnetics_ECOILA", "magnetics_bcoil", "magnetics_plasma_current"},
    "MAST": set(_MAST_PF_KEYS)
    | {
        "magnetics_sol_current",
        "magnetics_tf_current",
        "magnetics_efps_current",
        "magnetics_plasma_current",
    },
}


def current_convention(
    source: str,
    key: str,
    unit_schema: str = DEFAULT_UNIT_SCHEMA,
) -> CurrentConvention:
    """Return an explicit release-specific convention; never infer by magnitude."""

    if source not in _SUPPORTED_CURRENT_KEYS or key not in _SUPPORTED_CURRENT_KEYS[source]:
        raise KeyError(f"unsupported {source} current channel: {key}")
    physical_unit = "A-turn" if (
        (source == "DIII-D" and key in _D3D_PF_KEYS)
        or (source == "MAST" and key in _MAST_PF_KEYS)
    ) else "A"
    if unit_schema == "fusion-challenge-v1.1-kA":
        storage_unit = "kA-turn" if physical_unit == "A-turn" else "kA"
        return CurrentConvention(storage_unit, physical_unit, 1000.0)
    if unit_schema == "explicit-SI":
        return CurrentConvention(physical_unit, physical_unit, 1.0)
    raise ValueError(
        f"unknown unit schema {unit_schema!r}; expected "
        f"{DEFAULT_UNIT_SCHEMA!r} or 'explicit-SI'"
    )


def calibration_schema_manifest(
    unit_schema: str = DEFAULT_UNIT_SCHEMA,
) -> dict[str, object]:
    """Return the complete, canonical calibration contract for provenance.

    Keeping this generated from :func:`current_convention` prevents the audit
    record and the code path used by inference from drifting apart.
    """

    currents: dict[str, dict[str, dict[str, object]]] = {}
    for source in sorted(_SUPPORTED_CURRENT_KEYS):
        currents[source] = {}
        for key in sorted(_SUPPORTED_CURRENT_KEYS[source]):
            convention = current_convention(source, key, unit_schema)
            currents[source][key] = {
                "storage_unit": convention.storage_unit,
                "physical_unit": convention.physical_unit,
                "scale_to_si": convention.scale_to_si,
            }
    return {
        "reader_version": MEASUREMENT_READER_VERSION,
        "unit_schema": unit_schema,
        "time_unit": "ms",
        "coordinate_unit": "m",
        "temperature_unit": "eV",
        "density_unit": "m^-3",
        "pressure_unit": "Pa",
        "currents": currents,
        "corrections": {
            "DIII-D.magnetics_plasma_current_times": "data_fixes.fix_d3d_ip_times:v1.1.0",
            "DIII-D.thomson_edge_trailing_padding": "data_fixes.fix_d3d_thomson_edge_padding:v1.1.0",
            "MAST.thomson_core_channel_0": "data_fixes.align_mast_thomson_core:v1.1.0",
        },
        "synchronization_defaults": {
            "currents": "linear",
            "thomson": "nearest",
            "extrapolation": False,
            "max_current_age_ms": DEFAULT_MAX_CURRENT_AGE_MS,
            "max_thomson_age_ms": DEFAULT_MAX_THOMSON_AGE_MS,
            "thomson_pairing": "one shared laser-pulse row for Te/ne and all channels",
        },
    }


def calibration_schema_sha256(unit_schema: str = DEFAULT_UNIT_SCHEMA) -> str:
    """Hash the canonical calibration contract used by this module."""

    payload = json.dumps(
        calibration_schema_manifest(unit_schema),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SynchronizedSignal:
    """One signal sampled on the requested time axis."""

    values: np.ndarray
    valid: np.ndarray
    coverage_valid: np.ndarray
    fresh: np.ndarray
    age_ms: np.ndarray


@dataclass(frozen=True)
class CurrentMeasurements:
    """Current channels synchronized to ``efit_times``.

    ``values_si[:, j]`` has the unit in ``units[j]``.  PF channels whose
    upstream value is total excitation use ``"A-turn"``; per-turn currents,
    plasma current, and TF current use ``"A"``.
    """

    names: tuple[str, ...]
    storage_units: tuple[str, ...]
    units: tuple[str, ...]
    scale_to_si: np.ndarray
    values_si: np.ndarray
    valid: np.ndarray
    coverage_valid: np.ndarray
    fresh: np.ndarray
    age_ms: np.ndarray

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc

    def column(self, name: str) -> np.ndarray:
        return self.values_si[:, self.index(name)]

    def matrix(self, names: list[str] | tuple[str, ...]) -> np.ndarray:
        if not names:
            return np.empty((len(self.values_si), 0), dtype=np.float64)
        return self.values_si[:, [self.index(name) for name in names]]


@dataclass(frozen=True)
class ThomsonMeasurements:
    """One coordinate-tagged Thomson subsystem on ``efit_times``."""

    system: str
    names: tuple[str, ...]
    coordinates_m: np.ndarray
    te_eV: np.ndarray
    ne_m3: np.ndarray
    pressure_Pa: np.ndarray
    valid: np.ndarray
    quality_valid: np.ndarray
    coverage_valid: np.ndarray
    fresh: np.ndarray
    age_ms: np.ndarray
    acquisition_time_ms: np.ndarray


@dataclass(frozen=True)
class SynchronizedMeasurements:
    """Calibrated inputs for one shot, all aligned to EFIT timestamps."""

    source: str
    times_ms: np.ndarray
    grid_r_m: np.ndarray
    grid_z_m: np.ndarray
    currents: CurrentMeasurements
    thomson: Mapping[str, ThomsonMeasurements]
    plasma_current_A: np.ndarray
    current_direction_sign: float
    input_current_scale_to_si: float
    unit_schema: str
    max_current_age_ms: float | None
    max_thomson_age_ms: float | None


def _as_profiles(raw: object) -> np.ndarray:
    """Convert Arrow/pandas nested profiles into ``time x channel``."""

    array = np.asarray(raw)
    if array.ndim == 2 and array.dtype != object:
        return array.astype(np.float64, copy=False)
    return np.stack([np.asarray(profile, dtype=np.float64) for profile in raw])


def _clean_time_series(times: object, values: object) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(times, dtype=np.float64).ravel()
    value = np.asarray(values, dtype=np.float64).ravel()
    count = min(len(time), len(value))
    finite = np.isfinite(time[:count]) & np.isfinite(value[:count])
    time, value = time[:count][finite], value[:count][finite]
    if not len(time):
        return time, value
    order = np.argsort(time, kind="stable")
    time, value = time[order], value[order]
    # Keep the last observation when an acquisition contains duplicate stamps.
    _, reverse_index = np.unique(time[::-1], return_index=True)
    keep = np.sort(len(time) - 1 - reverse_index)
    return time[keep], value[keep]


def synchronize_signal(
    times: object,
    values: object,
    query_ms: object,
    *,
    mode: InterpolationMode = "linear",
    max_age_ms: float | None = None,
    extrapolate: bool = False,
    fill_value: float = 0.0,
) -> SynchronizedSignal:
    """Sample one finite time series and retain validity and sample age.

    Masking each signal independently is essential for early MAST shots, whose
    ``magnetics_time`` is a union of two acquisition grids containing NaN holes.
    ``previous`` is causal; ``linear`` and ``nearest`` may use a future sample.
    """

    query = np.asarray(query_ms, dtype=np.float64).ravel()
    time, value = _clean_time_series(times, values)
    output = np.full(len(query), float(fill_value), dtype=np.float64)
    valid = np.zeros(len(query), dtype=bool)
    age = np.full(len(query), np.inf, dtype=np.float64)
    if not len(time):
        return SynchronizedSignal(
            output,
            valid,
            valid.copy(),
            valid.copy(),
            np.zeros(len(query), dtype=np.float64),
        )

    right = np.searchsorted(time, query, side="left")
    left_index = np.clip(right - 1, 0, len(time) - 1)
    right_index = np.clip(right, 0, len(time) - 1)
    left_age = np.abs(query - time[left_index])
    right_age = np.abs(time[right_index] - query)
    nearest_index = np.where(left_age <= right_age, left_index, right_index)
    age = np.minimum(left_age, right_age)

    if mode == "linear":
        if len(time) == 1:
            output[:] = value[0]
            valid = np.isclose(query, time[0]) | bool(extrapolate)
        else:
            output = np.interp(query, time, value, left=value[0], right=value[-1])
            valid = (query >= time[0]) & (query <= time[-1])
            if extrapolate:
                valid[:] = True
    elif mode == "previous":
        indices = np.searchsorted(time, query, side="right") - 1
        present = indices >= 0
        output[present] = value[np.clip(indices[present], 0, len(time) - 1)]
        age[present] = query[present] - time[indices[present]]
        valid = present & (query <= time[-1])
        if extrapolate:
            valid = present
    elif mode == "nearest":
        output = value[nearest_index]
        valid = (query >= time[0]) & (query <= time[-1])
        if extrapolate:
            valid[:] = True
    else:
        raise ValueError(f"unsupported interpolation mode: {mode}")

    coverage_valid = valid.copy()
    fresh = np.ones(len(query), dtype=bool)
    if max_age_ms is not None:
        if max_age_ms < 0:
            raise ValueError("max_age_ms must be nonnegative")
        fresh = age <= float(max_age_ms)
        valid &= fresh
    output[~valid] = float(fill_value)
    return SynchronizedSignal(output, valid, coverage_valid, fresh, age)


def _ordered_current_names(row: Mapping[str, object], source: str) -> tuple[str, ...]:
    joins = np.asarray(row.get("coil_input_column", []), dtype=str)
    names = [key for key in dict.fromkeys(joins.tolist()) if key]
    extras = (
        ("magnetics_plasma_current", "magnetics_bcoil")
        if source == "DIII-D"
        else ("magnetics_plasma_current", "magnetics_tf_current", "magnetics_efps_current")
    )
    for key in extras:
        if key in row and key not in names:
            names.append(key)
    return tuple(names)


def _signal_times(row: Mapping[str, object], source: str, key: str) -> object:
    if key == "magnetics_plasma_current" and source == "DIII-D":
        return fix_d3d_ip_times(row)
    return row.get(f"{key}_times", row["magnetics_time"])


def _read_currents(
    row: Mapping[str, object],
    source: str,
    query: np.ndarray,
    mode: InterpolationMode,
    unit_schema: str,
    max_age_ms: float | None,
) -> tuple[CurrentMeasurements, float]:
    names = _ordered_current_names(row, source)
    conventions = tuple(current_convention(source, key, unit_schema) for key in names)
    values = np.zeros((len(query), len(names)), dtype=np.float64)
    valid = np.zeros_like(values, dtype=bool)
    coverage_valid = np.zeros_like(values, dtype=bool)
    fresh = np.zeros_like(values, dtype=bool)
    age = np.full_like(values, np.inf)
    for column, (key, convention) in enumerate(zip(names, conventions)):
        if key not in row:
            continue
        synchronized = synchronize_signal(
            _signal_times(row, source, key),
            row[key],
            query,
            mode=mode,
            max_age_ms=max_age_ms,
        )
        values[:, column] = convention.scale_to_si * synchronized.values
        valid[:, column] = synchronized.valid
        coverage_valid[:, column] = synchronized.coverage_valid
        fresh[:, column] = synchronized.fresh
        age[:, column] = synchronized.age_ms
    return CurrentMeasurements(
        names=names,
        storage_units=tuple(item.storage_unit for item in conventions),
        units=tuple(item.physical_unit for item in conventions),
        scale_to_si=np.asarray([item.scale_to_si for item in conventions], dtype=np.float64),
        values_si=values,
        valid=valid,
        coverage_valid=coverage_valid,
        fresh=fresh,
        age_ms=age,
    ), float(conventions[0].scale_to_si) if conventions else 1.0


def _chord_coordinates(
    row: Mapping[str, object],
    source: str,
    system: str,
    channels: int,
    *,
    strict: bool,
) -> tuple[tuple[str, ...], np.ndarray]:
    all_names = np.asarray(row.get("thomson_chord_name", []), dtype=str)
    all_r = np.asarray(row.get("thomson_chord_R", []), dtype=np.float64)
    all_z = np.asarray(row.get("thomson_chord_Z", []), dtype=np.float64)
    count = min(len(all_names), len(all_r), len(all_z))
    lower = np.char.lower(all_names[:count])
    selected = np.char.find(lower, "core") >= 0
    if system == "edge":
        selected = (np.char.find(lower, "edge") >= 0) | (
            np.char.find(lower, "tangential") >= 0
        )
    names = all_names[:count][selected]
    coordinates = np.column_stack([all_r[:count][selected], all_z[:count][selected]])
    if len(coordinates) == channels:
        return tuple(names.tolist()), coordinates

    # MAST has a fully specified horizontal system even if chord-name metadata
    # is unavailable.  DIII-D needs the supplied per-shot two-coordinate chord
    # table; guessing the missing coordinate would silently corrupt geometry.
    if source == "MAST":
        spatial_key = "thomson_core_R" if system == "core" else "thomson_edge_spatial"
        radius = np.asarray(row.get(spatial_key, []), dtype=np.float64)
        if len(radius) >= channels:
            radius = radius[-channels:]
            fallback_names = tuple(f"{system}_{index}" for index in range(channels))
            return fallback_names, np.column_stack([radius, np.zeros(channels)])
    if strict:
        raise ValueError(
            f"{source} {system} Thomson geometry mismatch: {channels} data "
            f"channels but {len(coordinates)} coordinate-tagged chords"
        )
    fallback_names = tuple(f"{system}_{index}" for index in range(channels))
    coordinates = np.full((channels, 2), np.nan, dtype=np.float64)
    return fallback_names, coordinates


def _synchronize_profile_pairs(
    times: np.ndarray,
    te_profiles: np.ndarray,
    ne_profiles: np.ndarray,
    query: np.ndarray,
    mode: ThomsonInterpolationMode,
    max_age_ms: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select one shared laser-pulse row for both quantities and all channels."""

    if mode not in {"nearest", "previous"}:
        raise ValueError("Thomson profiles must use nearest or previous pulse sampling")
    count = min(len(times), len(te_profiles), len(ne_profiles))
    channels = min(te_profiles.shape[1], ne_profiles.shape[1])
    times = np.asarray(times[:count], dtype=np.float64)
    te_profiles = np.asarray(te_profiles[:count, :channels], dtype=np.float64)
    ne_profiles = np.asarray(ne_profiles[:count, :channels], dtype=np.float64)
    finite_time = np.isfinite(times)
    times, te_profiles, ne_profiles = (
        times[finite_time],
        te_profiles[finite_time],
        ne_profiles[finite_time],
    )
    output_te = np.zeros((len(query), channels), dtype=np.float64)
    output_ne = np.zeros_like(output_te)
    age = np.full(len(query), np.inf, dtype=np.float64)
    acquisition = np.full(len(query), np.nan, dtype=np.float64)
    coverage = np.zeros(len(query), dtype=bool)
    if not len(times):
        return (
            output_te,
            output_ne,
            coverage,
            np.zeros(len(query), dtype=np.float64),
            np.zeros(len(query), dtype=np.float64),
        )
    order = np.argsort(times, kind="stable")
    times, te_profiles, ne_profiles = times[order], te_profiles[order], ne_profiles[order]
    _, reverse_index = np.unique(times[::-1], return_index=True)
    keep = np.sort(len(times) - 1 - reverse_index)
    times, te_profiles, ne_profiles = times[keep], te_profiles[keep], ne_profiles[keep]

    if mode == "nearest":
        right = np.searchsorted(times, query, side="left")
        left_index = np.clip(right - 1, 0, len(times) - 1)
        right_index = np.clip(right, 0, len(times) - 1)
        left_age = np.abs(query - times[left_index])
        right_age = np.abs(times[right_index] - query)
        selected = np.where(left_age <= right_age, left_index, right_index)
        coverage = (query >= times[0]) & (query <= times[-1])
    else:
        selected = np.searchsorted(times, query, side="right") - 1
        coverage = (selected >= 0) & (query <= times[-1])
        selected = np.clip(selected, 0, len(times) - 1)
    acquisition = times[selected]
    age = np.abs(query - acquisition)
    if max_age_ms is not None:
        if max_age_ms < 0:
            raise ValueError("max_age_ms must be nonnegative")
    output_te[coverage] = te_profiles[selected[coverage]]
    output_ne[coverage] = ne_profiles[selected[coverage]]
    acquisition[~coverage] = 0.0
    return output_te, output_ne, coverage, age, acquisition


def _read_thomson_system(
    row: Mapping[str, object],
    source: str,
    system: str,
    query: np.ndarray,
    mode: ThomsonInterpolationMode,
    max_age_ms: float | None,
    strict_coordinates: bool,
) -> ThomsonMeasurements | None:
    time_key = f"thomson_{system}_times"
    te_key = f"thomson_{system}_Te"
    ne_key = f"thomson_{system}_ne"
    if time_key not in row or te_key not in row or ne_key not in row:
        return None
    times = np.asarray(row[time_key], dtype=np.float64)
    if system == "core":
        _, te, ne = align_mast_thomson_core(row)
        te, ne = np.asarray(te, dtype=np.float64), np.asarray(ne, dtype=np.float64)
    elif source == "DIII-D":
        times, te, ne = fix_d3d_thomson_edge_padding(row)
    else:
        te, ne = _as_profiles(row[te_key]), _as_profiles(row[ne_key])
    time_count = min(len(times), len(te), len(ne))
    channels = min(te.shape[1], ne.shape[1])
    times, te, ne = times[:time_count], te[:time_count, :channels], ne[:time_count, :channels]
    names, coordinates = _chord_coordinates(
        row, source, system, channels, strict=strict_coordinates
    )
    channels = min(channels, len(coordinates))
    te, ne, temporal_coverage, pulse_age, acquisition_time = _synchronize_profile_pairs(
        times,
        te[:, :channels],
        ne[:, :channels],
        query,
        mode,
        max_age_ms,
    )
    coordinate_valid = np.isfinite(coordinates[:channels]).all(axis=1)
    quality_valid = (
        np.isfinite(te)
        & np.isfinite(ne)
        & (te > 0.0)
        & (ne > 0.0)
        & coordinate_valid[None, :]
    )
    age = np.broadcast_to(pulse_age[:, None], te.shape).copy()
    coverage_valid = np.broadcast_to(temporal_coverage[:, None], te.shape).copy()
    fresh_frame = (
        np.ones(len(query), dtype=bool)
        if max_age_ms is None
        else pulse_age <= float(max_age_ms)
    )
    fresh = np.broadcast_to(fresh_frame[:, None], te.shape).copy()
    valid = quality_valid & coverage_valid & fresh
    te = np.where(valid, te, 0.0)
    ne = np.where(valid, ne, 0.0)
    pressure = ELECTRON_CHARGE_C * te * ne
    return ThomsonMeasurements(
        system=system,
        names=names[:channels],
        coordinates_m=np.asarray(coordinates[:channels], dtype=np.float64),
        te_eV=te,
        ne_m3=ne,
        pressure_Pa=pressure,
        valid=valid,
        quality_valid=quality_valid,
        coverage_valid=coverage_valid,
        fresh=fresh,
        age_ms=age,
        acquisition_time_ms=acquisition_time,
    )


def read_measurements(
    row: Mapping[str, object],
    *,
    unit_schema: str = DEFAULT_UNIT_SCHEMA,
    current_interpolation: InterpolationMode = "linear",
    thomson_interpolation: ThomsonInterpolationMode = "nearest",
    max_current_age_ms: float | None = DEFAULT_MAX_CURRENT_AGE_MS,
    max_thomson_age_ms: float | None = DEFAULT_MAX_THOMSON_AGE_MS,
    strict_coordinates: bool = True,
) -> SynchronizedMeasurements:
    """Calibrate one challenge row and synchronize all inputs to EFIT time.

    Set ``thomson_interpolation="previous"`` for a strictly causal observer.
    The default nearest-snapshot policy never invents a Thomson profile between
    laser pulses; ``age_ms`` records its temporal displacement from EFIT time.
    """

    source = str(row.get("source", "DIII-D"))
    if source not in {"DIII-D", "MAST"}:
        raise ValueError(f"unsupported source: {source}")
    if "magnetics_plasma_current" not in row:
        raise KeyError("magnetics_plasma_current is required for calibration")
    query = np.asarray(row["efit_times"], dtype=np.float64).ravel()
    grid_r = np.asarray(row["efit_grid_R"], dtype=np.float64).ravel()
    grid_z = np.asarray(row["efit_grid_Z"], dtype=np.float64).ravel()
    for label, values in (("efit_times", query), ("efit_grid_R", grid_r), ("efit_grid_Z", grid_z)):
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"{label} must be nonempty and finite")
    if np.any(np.diff(query) <= 0) or np.any(np.diff(grid_r) <= 0) or np.any(np.diff(grid_z) <= 0):
        raise ValueError("EFIT times and R/Z grids must be strictly increasing")

    currents, scale = _read_currents(
        row,
        source,
        query,
        current_interpolation,
        unit_schema,
        max_current_age_ms,
    )
    if "magnetics_plasma_current" in currents.names:
        plasma_current = currents.column("magnetics_plasma_current")
        direction = canonical_current_sign(plasma_current)
    else:
        plasma_current = np.zeros(len(query), dtype=np.float64)
        direction = 1.0
    thomson: dict[str, ThomsonMeasurements] = {}
    for system in ("core", "edge"):
        measurements = _read_thomson_system(
            row,
            source,
            system,
            query,
            thomson_interpolation,
            max_thomson_age_ms,
            strict_coordinates,
        )
        if measurements is not None:
            thomson[system] = measurements
    return SynchronizedMeasurements(
        source=source,
        times_ms=query,
        grid_r_m=grid_r,
        grid_z_m=grid_z,
        currents=currents,
        thomson=thomson,
        plasma_current_A=plasma_current,
        current_direction_sign=direction,
        input_current_scale_to_si=scale,
        unit_schema=unit_schema,
        max_current_age_ms=max_current_age_ms,
        max_thomson_age_ms=max_thomson_age_ms,
    )


def read_parquet_measurements(path: str | Path, **kwargs) -> SynchronizedMeasurements:
    """Load the single shot in a challenge parquet and call ``read_measurements``."""

    import pandas as pd

    frame = pd.read_parquet(Path(path))
    if len(frame) != 1:
        raise ValueError(f"expected one shot row in {path}, found {len(frame)}")
    return read_measurements(frame.iloc[0].to_dict(), **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="one challenge shot parquet")
    parser.add_argument(
        "--causal",
        action="store_true",
        help="use previous-sample holds instead of future-looking interpolation",
    )
    parser.add_argument(
        "--max-thomson-age-ms", type=float, default=DEFAULT_MAX_THOMSON_AGE_MS
    )
    parser.add_argument(
        "--max-current-age-ms", type=float, default=DEFAULT_MAX_CURRENT_AGE_MS
    )
    parser.add_argument(
        "--current-interpolation",
        choices=["linear", "previous", "nearest"],
        default="linear",
    )
    parser.add_argument(
        "--thomson-interpolation",
        choices=["previous", "nearest"],
        default="nearest",
    )
    parser.add_argument(
        "--unit-schema",
        choices=[DEFAULT_UNIT_SCHEMA, "explicit-SI"],
        default=DEFAULT_UNIT_SCHEMA,
    )
    args = parser.parse_args()
    current_mode: InterpolationMode = (
        "previous" if args.causal else args.current_interpolation
    )
    thomson_mode: ThomsonInterpolationMode = (
        "previous" if args.causal else args.thomson_interpolation
    )
    measurements = read_parquet_measurements(
        args.parquet,
        unit_schema=args.unit_schema,
        current_interpolation=current_mode,
        thomson_interpolation=thomson_mode,
        max_current_age_ms=args.max_current_age_ms,
        max_thomson_age_ms=args.max_thomson_age_ms,
    )
    print(
        f"{measurements.source}: {len(measurements.times_ms)} EFIT times, "
        f"{len(measurements.currents.names)} current channels, "
        f"current scale x{measurements.input_current_scale_to_si:g}"
    )
    for system, values in measurements.thomson.items():
        print(
            f"  Thomson {system}: {len(values.names)} chords, "
            f"{values.valid.mean():.1%} valid synchronized samples"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
