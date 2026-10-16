#!/usr/bin/env python3
"""Numerically converged active-coil ``psi``, ``B_R``, and ``B_Z`` fields.

Each axisymmetric conductor cross-section is integrated with tensor-product
Gauss--Legendre quadrature. Circular-filament ``B_R`` and ``B_Z`` are evaluated
from analytic elliptic-integral expressions rather than finite differences of
``psi``. Strict mode refuses geometry whose released convention is incomplete.

MAST P-coil signals are total ampere-turns, so their enumerated turn responses
are averaged. The MAST solenoid signal is current per turn, so all enumerated
responses are summed. TF coils, plasma current, and passive/vessel currents are
not active poloidal-coil sources and remain explicitly separate.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ellipe, ellipk

from neuralgs.geometry import MU0
from .read_measurements import (
    SynchronizedMeasurements,
    current_convention,
    read_measurements,
)

DEFAULT_QUADRATURE_ORDER = 4
_RESPONSE_CACHE: dict[str, "CoilResponse"] = {}


class UnresolvedGeometryError(ValueError):
    """Raised when the released geometry cannot support an exact calculation."""


@dataclass(frozen=True)
class CoilResponse:
    names: tuple[str, ...]
    units: tuple[str, ...]
    psi_per_si: np.ndarray
    b_r_per_si: np.ndarray
    b_z_per_si: np.ndarray
    quadrature_order: int
    strict_geometry: bool
    approximations: tuple[str, ...]


@dataclass(frozen=True)
class CoilFields:
    """Active-coil fields on a physical ``(Z, R)`` grid."""

    times_ms: np.ndarray
    grid_r_m: np.ndarray
    grid_z_m: np.ndarray
    coil_names: tuple[str, ...]
    current_units: tuple[str, ...]
    currents_si: np.ndarray
    current_valid: np.ndarray
    frame_valid: np.ndarray
    psi_per_si: np.ndarray
    b_r_per_si: np.ndarray
    b_z_per_si: np.ndarray
    psi_active_Wb_per_rad: np.ndarray
    b_r_T: np.ndarray
    b_z_T: np.ndarray
    field_valid_mask: np.ndarray
    canonicalized: bool
    quadrature_order: int
    strict_geometry: bool
    approximations: tuple[str, ...]


@dataclass(frozen=True)
class PassiveFieldComponent:
    """An externally supplied passive/eddy field; never inferred here."""

    psi_Wb_per_rad: np.ndarray
    b_r_T: np.ndarray
    b_z_T: np.ndarray
    provenance: str


@dataclass(frozen=True)
class TotalPoloidalFields:
    psi_Wb_per_rad: np.ndarray
    b_r_T: np.ndarray
    b_z_T: np.ndarray
    active: CoilFields
    passive: PassiveFieldComponent


@dataclass(frozen=True)
class ReferenceComparison:
    metrics: Mapping[str, Mapping[str, float]]
    passed: bool


def circular_loop_response_per_ampere(
    target_r_m: np.ndarray | float,
    target_z_m: np.ndarray | float,
    source_r_m: np.ndarray | float,
    source_z_m: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return analytic filament ``(psi, B_R, B_Z)`` per ampere.

    The formulas use SciPy's parameter convention ``K(m), E(m)`` with
    ``m=k**2``. Evaluation on the current filament is singular and raises
    instead of being silently clipped to a finite value.
    """

    r, z, a, z0 = np.broadcast_arrays(
        np.asarray(target_r_m, dtype=np.float64),
        np.asarray(target_z_m, dtype=np.float64),
        np.asarray(source_r_m, dtype=np.float64),
        np.asarray(source_z_m, dtype=np.float64),
    )
    if np.any(r < 0.0) or np.any(a <= 0.0):
        raise ValueError("target R must be nonnegative and source R must be positive")
    dz = z - z0
    beta2 = (a + r) ** 2 + dz**2
    alpha2 = (a - r) ** 2 + dz**2
    scale = np.maximum(beta2, 1.0)
    singular = alpha2 <= 32.0 * np.finfo(np.float64).eps * scale
    if np.any(singular):
        raise FloatingPointError("field requested on an ideal current filament")

    psi = np.zeros_like(r)
    b_r = np.zeros_like(r)
    b_z = np.empty_like(r)
    axis = r <= np.sqrt(np.finfo(np.float64).eps) * np.maximum(a, 1.0)
    b_z[axis] = MU0 * a[axis] ** 2 / (2.0 * beta2[axis] ** 1.5)
    off_axis = ~axis
    if np.any(off_axis):
        ro = r[off_axis]
        ao = a[off_axis]
        dzo = dz[off_axis]
        beta2o = beta2[off_axis]
        alpha2o = alpha2[off_axis]
        m = np.clip(
            4.0 * ao * ro / beta2o,
            np.finfo(np.float64).tiny,
            np.nextafter(1.0, 0.0),
        )
        k = np.sqrt(m)
        first = ellipk(m)
        second = ellipe(m)
        root_beta = np.sqrt(beta2o)
        psi[off_axis] = (
            MU0
            * np.sqrt(ao * ro)
            * ((1.0 - 0.5 * m) * first - second)
            / (np.pi * k)
        )
        common = MU0 / (2.0 * np.pi * root_beta)
        b_r[off_axis] = common * dzo / ro * (
            -first + (ao**2 + ro**2 + dzo**2) * second / alpha2o
        )
        b_z[off_axis] = common * (
            first + (ao**2 - ro**2 - dzo**2) * second / alpha2o
        )
    return psi, b_r, b_z


def poloidal_field_from_psi(
    psi: np.ndarray,
    grid_r_m: np.ndarray,
    grid_z_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference identity check, not the production field path."""

    field = np.asarray(psi, dtype=np.float64)
    grid_r = np.asarray(grid_r_m, dtype=np.float64).ravel()
    grid_z = np.asarray(grid_z_m, dtype=np.float64).ravel()
    if field.shape[-2:] != (len(grid_z), len(grid_r)):
        raise ValueError("psi trailing dimensions do not match the R/Z grid")
    if len(grid_r) < 3 or len(grid_z) < 3:
        raise ValueError("at least three R and Z points are required")
    if np.any(grid_r <= 0.0):
        raise ValueError("axisymmetric field calculation requires R > 0")
    if np.any(np.diff(grid_r) <= 0.0) or np.any(np.diff(grid_z) <= 0.0):
        raise ValueError("R and Z grids must be strictly increasing")
    dpsi_dz, dpsi_dr = np.gradient(
        field, grid_z, grid_r, axis=(-2, -1), edge_order=2
    )
    radius = grid_r.reshape((1,) * (field.ndim - 1) + (len(grid_r),))
    return -dpsi_dz / radius, dpsi_dr / radius


def _geometry_arrays(row: Mapping[str, object]) -> tuple[np.ndarray, ...]:
    joins = np.asarray(row.get("coil_input_column", []), dtype=str)
    arrays = [
        np.asarray(row.get(key, []), dtype=np.float64)
        for key in (
            "coil_R",
            "coil_Z",
            "coil_width",
            "coil_height",
            "coil_angle1",
            "coil_angle2",
        )
    ]
    lengths = {len(joins), *(len(array) for array in arrays)}
    if len(lengths) != 1:
        raise ValueError(f"coil geometry columns have inconsistent lengths: {sorted(lengths)}")
    if not len(joins):
        raise ValueError("coil geometry is empty")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("coil geometry must be finite")
    if np.any(arrays[0] <= 0.0) or np.any(arrays[2] < 0.0) or np.any(arrays[3] < 0.0):
        raise ValueError("coil R must be positive and width/height nonnegative")
    return (joins, *arrays)


def _quadrature_nodes(
    center_r: float,
    center_z: float,
    width: float,
    height: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def one_axis(center: float, span: float) -> tuple[np.ndarray, np.ndarray]:
        if span == 0.0:
            return np.asarray([center]), np.asarray([1.0])
        nodes, weights = leggauss(order)
        return center + 0.5 * span * nodes, 0.5 * weights

    r_nodes, r_weights = one_axis(center_r, width)
    z_nodes, z_weights = one_axis(center_z, height)
    rr, zz = np.meshgrid(r_nodes, z_nodes)
    weights = np.outer(z_weights, r_weights)
    return rr.ravel(), zz.ravel(), weights.ravel()


def _response_cache_key(
    row: Mapping[str, object],
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    order: int,
    strict_geometry: bool,
) -> str:
    digest = hashlib.sha256()
    for key in (
        "coil_input_column",
        "coil_R",
        "coil_Z",
        "coil_width",
        "coil_height",
        "coil_angle1",
        "coil_angle2",
    ):
        array = np.asarray(row.get(key, []))
        payload = array.astype(str).tobytes() if array.dtype.kind in "OUS" else array.tobytes()
        digest.update(payload)
    digest.update(np.asarray(grid_r, dtype=np.float64).tobytes())
    digest.update(np.asarray(grid_z, dtype=np.float64).tobytes())
    digest.update(str(row.get("source", "DIII-D")).encode())
    digest.update(f"{order}:{int(strict_geometry)}".encode())
    return digest.hexdigest()


def build_coil_response(
    row: Mapping[str, object],
    grid_r_m: np.ndarray,
    grid_z_m: np.ndarray,
    *,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    strict_geometry: bool = True,
) -> CoilResponse:
    """Build finite-conductor response maps for every active current input."""

    if quadrature_order < 2:
        raise ValueError("quadrature_order must be at least two")
    grid_r = np.asarray(grid_r_m, dtype=np.float64).ravel()
    grid_z = np.asarray(grid_z_m, dtype=np.float64).ravel()
    if np.any(grid_r <= 0.0) or np.any(np.diff(grid_r) <= 0.0) or np.any(np.diff(grid_z) <= 0.0):
        raise ValueError("physical R/Z grids must be strictly increasing with R > 0")
    cache_key = _response_cache_key(
        row, grid_r, grid_z, quadrature_order, strict_geometry
    )
    if cache_key in _RESPONSE_CACHE:
        return _RESPONSE_CACHE[cache_key]

    source = str(row.get("source", "DIII-D"))
    joins, coil_r, coil_z, width, height, angle1, angle2 = _geometry_arrays(row)
    skewed = (np.abs(angle1) > 1.0e-10) | (np.abs(angle2) > 1.0e-10)
    problems: list[str] = []
    if np.any(skewed):
        keys = sorted(set(joins[skewed].tolist()))
        problems.append(f"unresolved skew convention for {keys}")
    if source == "DIII-D" and "magnetics_ECOILA" in joins:
        problems.append("DIII-D ECOILA turn count and missing ECOILB are unresolved")
    if strict_geometry and problems:
        raise UnresolvedGeometryError("; ".join(problems))

    rr, zz = np.meshgrid(grid_r, grid_z)
    names = tuple(key for key in dict.fromkeys(joins.tolist()) if key)
    psi_responses: list[np.ndarray] = []
    br_responses: list[np.ndarray] = []
    bz_responses: list[np.ndarray] = []
    units: list[str] = []
    for key in names:
        convention = current_convention(source, key)
        units.append(convention.physical_unit)
        indices = np.flatnonzero(joins == key)
        divisor = len(indices) if convention.physical_unit == "A-turn" else 1
        group_psi = np.zeros_like(rr)
        group_br = np.zeros_like(rr)
        group_bz = np.zeros_like(rr)
        for index in indices:
            if skewed[index]:
                nodes_r = np.asarray([coil_r[index]])
                nodes_z = np.asarray([coil_z[index]])
                weights = np.asarray([1.0])
            else:
                nodes_r, nodes_z, weights = _quadrature_nodes(
                    coil_r[index],
                    coil_z[index],
                    width[index],
                    height[index],
                    quadrature_order,
                )
            for node_r, node_z, weight in zip(nodes_r, nodes_z, weights):
                element_psi, element_br, element_bz = circular_loop_response_per_ampere(
                    rr, zz, node_r, node_z
                )
                group_psi += weight * element_psi / divisor
                group_br += weight * element_br / divisor
                group_bz += weight * element_bz / divisor
        psi_responses.append(group_psi)
        br_responses.append(group_br)
        bz_responses.append(group_bz)

    result = CoilResponse(
        names=names,
        units=tuple(units),
        psi_per_si=np.stack(psi_responses),
        b_r_per_si=np.stack(br_responses),
        b_z_per_si=np.stack(bz_responses),
        quadrature_order=quadrature_order,
        strict_geometry=strict_geometry,
        approximations=tuple(problems),
    )
    _RESPONSE_CACHE[cache_key] = result
    return result


def _outside_conductors(
    row: Mapping[str, object], grid_r: np.ndarray, grid_z: np.ndarray
) -> np.ndarray:
    _, coil_r, coil_z, width, height, _, _ = _geometry_arrays(row)
    rr, zz = np.meshgrid(grid_r, grid_z)
    dr = float(np.min(np.diff(grid_r)))
    dz = float(np.min(np.diff(grid_z)))
    outside = np.ones_like(rr, dtype=bool)
    for r0, z0, w, h in zip(coil_r, coil_z, width, height):
        half_r = max(0.5 * w, 0.5 * dr)
        half_z = max(0.5 * h, 0.5 * dz)
        outside &= (np.abs(rr - r0) > half_r) | (np.abs(zz - z0) > half_z)
    return outside


def quadrature_convergence_report(
    row: Mapping[str, object],
    grid_r_m: np.ndarray,
    grid_z_m: np.ndarray,
    *,
    order: int = 4,
    reference_order: int = 8,
    relative_tolerance: float = 1.0e-5,
    strict_geometry: bool = True,
) -> dict[str, object]:
    """Compare every conductor-group response with a higher-order calculation."""

    if reference_order <= order:
        raise ValueError("reference_order must exceed order")
    grid_r = np.asarray(grid_r_m, dtype=np.float64)
    grid_z = np.asarray(grid_z_m, dtype=np.float64)
    trial = build_coil_response(
        row, grid_r, grid_z, quadrature_order=order, strict_geometry=strict_geometry
    )
    reference = build_coil_response(
        row,
        grid_r,
        grid_z,
        quadrature_order=reference_order,
        strict_geometry=strict_geometry,
    )
    mask = _outside_conductors(row, grid_r, grid_z)
    per_coil: dict[str, object] = {}
    passed = True
    for index, name in enumerate(trial.names):
        components: dict[str, object] = {}
        for label, candidate, target in (
            ("psi", trial.psi_per_si[index], reference.psi_per_si[index]),
            ("b_r", trial.b_r_per_si[index], reference.b_r_per_si[index]),
            ("b_z", trial.b_z_per_si[index], reference.b_z_per_si[index]),
        ):
            delta = (candidate - target)[mask]
            target_values = target[mask]
            denominator = max(float(np.linalg.norm(target_values)), np.finfo(float).tiny)
            relative_l2 = float(np.linalg.norm(delta) / denominator)
            components[label] = {
                "relative_l2": relative_l2,
                "max_abs": float(np.max(np.abs(delta))),
            }
            passed &= relative_l2 <= relative_tolerance
        per_coil[name] = components
    return {
        "order": order,
        "reference_order": reference_order,
        "relative_tolerance": relative_tolerance,
        "outside_grid_points": int(mask.sum()),
        "per_coil": per_coil,
        "passed": bool(passed),
    }


def calculate_coil_fields(
    row: Mapping[str, object],
    measurements: SynchronizedMeasurements | None = None,
    *,
    frame_indices: np.ndarray | None = None,
    canonicalize_current_direction: bool = False,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    strict_geometry: bool = True,
) -> CoilFields:
    """Render active-coil fields using signed, synchronized physical currents."""

    if measurements is None:
        measurements = read_measurements(row)
    row_source = str(row.get("source", "DIII-D"))
    if measurements.source != row_source:
        raise ValueError("measurement source does not match row source")
    response = build_coil_response(
        row,
        measurements.grid_r_m,
        measurements.grid_z_m,
        quadrature_order=quadrature_order,
        strict_geometry=strict_geometry,
    )
    columns = [measurements.currents.index(name) for name in response.names]
    currents = measurements.currents.values_si[:, columns]
    current_valid = measurements.currents.valid[:, columns]
    frame_valid = current_valid.all(axis=1) & np.isfinite(currents).all(axis=1)
    units = tuple(measurements.currents.units[index] for index in columns)
    if units != response.units:
        raise ValueError("measurement and coil-response current units differ")
    if canonicalize_current_direction:
        currents = measurements.current_direction_sign * currents
    psi = np.einsum("tc,czr->tzr", currents, response.psi_per_si, optimize=True)
    b_r = np.einsum("tc,czr->tzr", currents, response.b_r_per_si, optimize=True)
    b_z = np.einsum("tc,czr->tzr", currents, response.b_z_per_si, optimize=True)
    # Invalid synchronized currents are zero-filled at the measurement boundary
    # for tensor safety.  A zero-filled field would look plausible, so make it
    # unusable unless every required active-coil current is valid.
    psi[~frame_valid] = np.nan
    b_r[~frame_valid] = np.nan
    b_z[~frame_valid] = np.nan

    if frame_indices is None:
        selected = np.arange(len(measurements.times_ms), dtype=np.int64)
    else:
        selected = np.asarray(frame_indices, dtype=np.int64).ravel()
        if len(selected) and (
            selected.min() < 0 or selected.max() >= len(measurements.times_ms)
        ):
            raise IndexError("frame index is outside efit_times")
    return CoilFields(
        times_ms=measurements.times_ms[selected],
        grid_r_m=np.asarray(measurements.grid_r_m, dtype=np.float64),
        grid_z_m=np.asarray(measurements.grid_z_m, dtype=np.float64),
        coil_names=response.names,
        current_units=units,
        currents_si=currents[selected],
        current_valid=current_valid[selected],
        frame_valid=frame_valid[selected],
        psi_per_si=response.psi_per_si,
        b_r_per_si=response.b_r_per_si,
        b_z_per_si=response.b_z_per_si,
        psi_active_Wb_per_rad=psi[selected],
        b_r_T=b_r[selected],
        b_z_T=b_z[selected],
        field_valid_mask=_outside_conductors(
            row,
            np.asarray(measurements.grid_r_m, dtype=np.float64),
            np.asarray(measurements.grid_z_m, dtype=np.float64),
        ),
        canonicalized=bool(canonicalize_current_direction),
        quadrature_order=response.quadrature_order,
        strict_geometry=response.strict_geometry,
        approximations=response.approximations,
    )


def combine_with_passive_field(
    active: CoilFields,
    passive: PassiveFieldComponent,
) -> TotalPoloidalFields:
    """Combine an explicitly sourced passive field with the active result."""

    if not passive.provenance.strip():
        raise ValueError("passive-field provenance must be nonempty")
    expected = active.psi_active_Wb_per_rad.shape
    arrays = (passive.psi_Wb_per_rad, passive.b_r_T, passive.b_z_T)
    if any(np.asarray(array).shape != expected for array in arrays):
        raise ValueError(f"passive fields must all have shape {expected}")
    return TotalPoloidalFields(
        psi_Wb_per_rad=active.psi_active_Wb_per_rad + passive.psi_Wb_per_rad,
        b_r_T=active.b_r_T + passive.b_r_T,
        b_z_T=active.b_z_T + passive.b_z_T,
        active=active,
        passive=passive,
    )


def compare_reference_fields(
    calculated: CoilFields,
    *,
    reference_psi: np.ndarray,
    reference_b_r: np.ndarray,
    reference_b_z: np.ndarray,
    mask: np.ndarray | None = None,
    relative_tolerance: float = 1.0e-5,
    absolute_tolerance: float = 1.0e-9,
) -> ReferenceComparison:
    """Compare against independent solver or vacuum-shot reference arrays."""

    metrics: dict[str, Mapping[str, float]] = {}
    passed = True
    for label, candidate, reference in (
        ("psi", calculated.psi_active_Wb_per_rad, reference_psi),
        ("b_r", calculated.b_r_T, reference_b_r),
        ("b_z", calculated.b_z_T, reference_b_z),
    ):
        candidate_array = np.asarray(candidate, dtype=np.float64)
        reference_array = np.asarray(reference, dtype=np.float64)
        if candidate_array.shape != reference_array.shape:
            raise ValueError(f"{label} reference shape differs from calculated fields")
        selected = (
            np.ones(candidate_array.shape, dtype=bool)
            if mask is None
            else np.broadcast_to(mask, candidate_array.shape)
        )
        delta = candidate_array[selected] - reference_array[selected]
        reference_values = reference_array[selected]
        relative_l2 = float(
            np.linalg.norm(delta)
            / max(float(np.linalg.norm(reference_values)), np.finfo(float).tiny)
        )
        max_abs = float(np.max(np.abs(delta)))
        passed &= relative_l2 <= relative_tolerance and max_abs <= absolute_tolerance
        metrics[label] = {"relative_l2": relative_l2, "max_abs": max_abs}
    return ReferenceComparison(metrics=metrics, passed=bool(passed))


def compare_filament_with_freegs(
    source_r_m: float,
    source_z_m: float,
    target_r_m: np.ndarray,
    target_z_m: np.ndarray,
) -> ReferenceComparison:
    """Optional independent filament comparison; requires ``freegs``."""

    try:
        from freegs.gradshafranov import Greens, GreensBr, GreensBz
    except ImportError as exc:
        raise RuntimeError(
            "FreeGS is not installed; install it in a validation environment "
            "before claiming independent agreement"
        ) from exc
    psi, b_r, b_z = circular_loop_response_per_ampere(
        target_r_m, target_z_m, source_r_m, source_z_m
    )
    references = (
        np.asarray(Greens(source_r_m, source_z_m, target_r_m, target_z_m)),
        np.asarray(GreensBr(source_r_m, source_z_m, target_r_m, target_z_m)),
        np.asarray(GreensBz(source_r_m, source_z_m, target_r_m, target_z_m)),
    )
    metrics: dict[str, Mapping[str, float]] = {}
    passed = True
    for label, candidate, reference in zip(
        ("psi", "b_r", "b_z"), (psi, b_r, b_z), references
    ):
        delta = candidate - reference
        relative_l2 = float(
            np.linalg.norm(delta)
            / max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
        )
        max_abs = float(np.max(np.abs(delta)))
        metrics[label] = {"relative_l2": relative_l2, "max_abs": max_abs}
        # FreeGS differentiates B with a fixed 1 mm stencil, so its B reference
        # is intentionally held to a looser tolerance than its analytic psi.
        passed &= relative_l2 <= (1.0e-10 if label == "psi" else 2.0e-4)
    return ReferenceComparison(metrics=metrics, passed=bool(passed))


def calculate_parquet_coil_fields(path: str | Path, **kwargs) -> CoilFields:
    import pandas as pd

    frame = pd.read_parquet(Path(path))
    if len(frame) != 1:
        raise ValueError(f"expected one shot row in {path}, found {len(frame)}")
    row = frame.iloc[0].to_dict()
    return calculate_coil_fields(row, read_measurements(row), **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="one challenge shot parquet")
    parser.add_argument("--canonicalize-current-direction", action="store_true")
    parser.add_argument("--quadrature-order", type=int, default=DEFAULT_QUADRATURE_ORDER)
    parser.add_argument("--allow-documented-approximations", action="store_true")
    parser.add_argument("--output", type=Path, help="optional compressed NPZ output")
    args = parser.parse_args()
    fields = calculate_parquet_coil_fields(
        args.parquet,
        canonicalize_current_direction=args.canonicalize_current_direction,
        quadrature_order=args.quadrature_order,
        strict_geometry=not args.allow_documented_approximations,
    )
    print(
        f"{len(fields.times_ms)} frames, {len(fields.coil_names)} groups, "
        f"Gauss order {fields.quadrature_order}, "
        f"valid fields={fields.frame_valid.sum()}/{len(fields.frame_valid)}, "
        f"max |Bpol|={np.nanmax(np.hypot(fields.b_r_T, fields.b_z_T)):.4g} T"
    )
    for approximation in fields.approximations:
        print(f"WARNING: {approximation}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            times_ms=fields.times_ms,
            grid_r_m=fields.grid_r_m,
            grid_z_m=fields.grid_z_m,
            coil_names=np.asarray(fields.coil_names),
            current_units=np.asarray(fields.current_units),
            currents_si=fields.currents_si,
            current_valid=fields.current_valid,
            frame_valid=fields.frame_valid,
            psi_active_Wb_per_rad=fields.psi_active_Wb_per_rad,
            b_r_T=fields.b_r_T,
            b_z_T=fields.b_z_T,
            field_valid_mask=fields.field_valid_mask,
            approximations=np.asarray(fields.approximations),
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
