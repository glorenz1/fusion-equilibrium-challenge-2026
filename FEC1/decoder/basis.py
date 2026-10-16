"""Machine-grid-native current-response and external-field decoder bases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from sklearn.utils.extmath import randomized_svd

from neuralgs.geometry import plasma_green_matrix


DEFAULT_GREEN_MODE_COUNT = 128
DEFAULT_HARMONIC_MODE_COUNT = 64
DEFAULT_GREEN_MODE_SEED = 20260828


def trapezoid_weights(coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64).ravel()
    if len(values) < 2 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0.0):
        raise ValueError("coordinates must be finite and strictly increasing")
    weights = np.empty_like(values)
    weights[0] = 0.5 * (values[1] - values[0])
    weights[-1] = 0.5 * (values[-1] - values[-2])
    if len(values) > 2:
        weights[1:-1] = 0.5 * (values[2:] - values[:-2])
    return weights


def grid_weights(grid_r_m: np.ndarray, grid_z_m: np.ndarray) -> np.ndarray:
    return np.outer(trapezoid_weights(grid_z_m), trapezoid_weights(grid_r_m)).reshape(-1)


def weighted_center_columns(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != len(vector):
        raise ValueError("values must be pixel-by-column and match weights")
    means = vector @ matrix / np.sum(vector)
    return matrix - means[None, :]


def _constraint_basis(columns: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(np.asarray(columns, dtype=np.float64), mode="reduced")
    diagonal = np.abs(np.diag(r))
    if not len(diagonal):
        return np.empty((len(columns), 0), dtype=np.float64)
    keep = diagonal > max(float(np.max(diagonal)) * 1.0e-11, 1.0e-13)
    return q[:, keep]


@dataclass(frozen=True)
class GreenModeBasis:
    """Dominant zero-net-current directions of a machine's Green operator."""

    grid_r_m: np.ndarray
    grid_z_m: np.ndarray
    source_r_m: np.ndarray
    source_z_m: np.ndarray
    source_weights_m2: np.ndarray
    source_modes_per_m2: np.ndarray
    flux_fields_Wb_per_rad_per_A: np.ndarray
    singular_values: np.ndarray
    coarse_constraint_rank: int
    source_stride: int
    softening_cells: float
    seed: int

    def manifest(self) -> dict[str, object]:
        quadrature = self.source_weights_m2.reshape(
            len(self.source_z_m), len(self.source_r_m)
        )
        net = np.sum(self.source_modes_per_m2 * quadrature[None], axis=(1, 2))
        return {
            "construction": "zero-net coarse-orthogonal Green SVD",
            "mode_count": len(self.source_modes_per_m2),
            "source_shape": list(self.source_modes_per_m2.shape[1:]),
            "output_shape": list(self.flux_fields_Wb_per_rad_per_A.shape[1:]),
            "coarse_constraint_rank": self.coarse_constraint_rank,
            "source_stride": self.source_stride,
            "softening_cells": self.softening_cells,
            "seed": self.seed,
            "maximum_absolute_net_current_per_unit_coefficient": float(
                np.max(np.abs(net), initial=0.0)
            ),
            "singular_values": self.singular_values.tolist(),
        }


def build_zero_net_green_basis(
    grid_r_m: np.ndarray,
    grid_z_m: np.ndarray,
    coarse_source_atoms: np.ndarray,
    *,
    mode_count: int = DEFAULT_GREEN_MODE_COUNT,
    source_stride: int = 2,
    softening_cells: float = 0.45,
    seed: int = DEFAULT_GREEN_MODE_SEED,
    power_iterations: int = 9,
) -> GreenModeBasis:
    """Build flux-observable current modes while preserving measured ``I_p``.

    Source coordinates are quadrature-whitened. The total-current direction
    and the coarse positive-simplex tangent are removed before the Green
    operator SVD, so every returned mode has zero integral and adds genuinely
    new decoder capacity.
    """

    if mode_count < 1:
        raise ValueError("mode_count must be positive")
    grid_r = np.asarray(grid_r_m, dtype=np.float64).ravel()
    grid_z = np.asarray(grid_z_m, dtype=np.float64).ravel()
    green, source_r, source_z, source_weights = plasma_green_matrix(
        grid_r,
        grid_z,
        source_stride=source_stride,
        softening_cells=softening_cells,
    )
    green = np.asarray(green, dtype=np.float64)
    source_r = np.asarray(source_r, dtype=np.float64)
    source_z = np.asarray(source_z, dtype=np.float64)
    source_weights = np.asarray(source_weights, dtype=np.float64).reshape(-1)
    atoms = np.asarray(coarse_source_atoms, dtype=np.float64)
    expected_source_shape = (len(source_z), len(source_r))
    if atoms.ndim != 3 or atoms.shape[1:] != expected_source_shape or len(atoms) < 2:
        raise ValueError(
            f"coarse_source_atoms must have shape (K, {expected_source_shape[0]}, "
            f"{expected_source_shape[1]}) with K>=2"
        )
    source_quadrature = source_weights.reshape(expected_source_shape)
    masses = np.sum(atoms * source_quadrature[None], axis=(1, 2))
    if not np.isfinite(masses).all() or np.any(np.abs(masses) < 1.0e-20):
        raise ValueError("every coarse source atom must have finite nonzero current")
    atoms = atoms / masses[:, None, None]
    root_source = np.sqrt(source_weights)
    constraint_columns = [root_source]
    constraint_columns.extend(
        root_source * (atoms[index] - atoms[0]).reshape(-1)
        for index in range(1, len(atoms))
    )
    constraints = _constraint_basis(np.column_stack(constraint_columns))
    output_weights = grid_weights(grid_r, grid_z)
    normalized_green = (
        np.sqrt(output_weights)[:, None]
        * weighted_center_columns(green / root_source[None, :], output_weights)
    )
    normalized_green -= (
        normalized_green @ constraints
    ) @ constraints.T
    maximum_rank = min(normalized_green.shape) - 1
    if mode_count > maximum_rank:
        raise ValueError(f"mode_count {mode_count} exceeds available rank {maximum_rank}")
    _, singular, right = randomized_svd(
        normalized_green,
        n_components=mode_count,
        n_iter=power_iterations,
        random_state=seed,
        power_iteration_normalizer="LU",
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    right = right[order]
    source_modes = (right / root_source[None, :]).reshape(
        mode_count, *expected_source_shape
    )
    field_modes = (
        green @ source_modes.reshape(mode_count, -1).T
    ).T.reshape(mode_count, len(grid_z), len(grid_r))
    net = np.sum(source_modes * source_quadrature[None], axis=(1, 2))
    weighted_modes = root_source[:, None] * source_modes.reshape(mode_count, -1).T
    overlap = constraints.T @ weighted_modes
    if np.max(np.abs(net), initial=0.0) > 1.0e-10:
        raise FloatingPointError("Green current modes do not preserve total current")
    if np.max(np.abs(overlap), initial=0.0) > 1.0e-9:
        raise FloatingPointError("Green current modes overlap the coarse current state")
    return GreenModeBasis(
        grid_r_m=grid_r,
        grid_z_m=grid_z,
        source_r_m=source_r,
        source_z_m=source_z,
        source_weights_m2=source_weights,
        source_modes_per_m2=source_modes,
        flux_fields_Wb_per_rad_per_A=field_modes,
        singular_values=singular,
        coarse_constraint_rank=int(constraints.shape[1]),
        source_stride=int(source_stride),
        softening_cells=float(softening_cells),
        seed=int(seed),
    )


def _gs_dirichlet_matrix(grid_r: np.ndarray, grid_z: np.ndarray) -> scipy.sparse.csc_matrix:
    nr, nz = len(grid_r), len(grid_z)
    dr = float(np.mean(np.diff(grid_r)))
    dz = float(np.mean(np.diff(grid_z)))
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    def index(iz: int, ir: int) -> int:
        return (iz - 1) * (nr - 2) + (ir - 1)

    for iz in range(1, nz - 1):
        for ir in range(1, nr - 1):
            row = index(iz, ir)
            coefficients = {
                (iz, ir): -2.0 / dr**2 - 2.0 / dz**2,
                (iz, ir + 1): 1.0 / dr**2 - 1.0 / (2.0 * grid_r[ir] * dr),
                (iz, ir - 1): 1.0 / dr**2 + 1.0 / (2.0 * grid_r[ir] * dr),
                (iz + 1, ir): 1.0 / dz**2,
                (iz - 1, ir): 1.0 / dz**2,
            }
            for (neighbor_z, neighbor_r), coefficient in coefficients.items():
                if 1 <= neighbor_z < nz - 1 and 1 <= neighbor_r < nr - 1:
                    rows.append(row)
                    columns.append(index(neighbor_z, neighbor_r))
                    values.append(coefficient)
    shape = ((nz - 2) * (nr - 2),) * 2
    return scipy.sparse.csc_matrix((values, (rows, columns)), shape=shape)


def _boundary_modes(nz: int, nr: int, count: int) -> np.ndarray:
    perimeter: list[tuple[int, int]] = []
    perimeter.extend((0, ir) for ir in range(nr))
    perimeter.extend((iz, nr - 1) for iz in range(1, nz))
    perimeter.extend((nz - 1, ir) for ir in range(nr - 2, -1, -1))
    perimeter.extend((iz, 0) for iz in range(nz - 2, 0, -1))
    phase = np.arange(len(perimeter), dtype=np.float64) / len(perimeter)
    modes = np.zeros((count, nz, nr), dtype=np.float64)
    for mode in range(count):
        frequency = mode // 2 + 1
        boundary = (
            np.cos(2.0 * np.pi * frequency * phase)
            if mode % 2 == 0
            else np.sin(2.0 * np.pi * frequency * phase)
        )
        for value, (iz, ir) in zip(boundary, perimeter):
            modes[mode, iz, ir] = value
    return modes


def _harmonic_extension(
    boundary_fields: np.ndarray,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
) -> np.ndarray:
    fields = np.asarray(boundary_fields, dtype=np.float64)
    count, nz, nr = fields.shape
    dr = float(np.mean(np.diff(grid_r)))
    dz = float(np.mean(np.diff(grid_z)))
    solver = scipy.sparse.linalg.factorized(_gs_dirichlet_matrix(grid_r, grid_z))
    rhs = np.zeros(((nz - 2) * (nr - 2), count), dtype=np.float64)

    def index(iz: int, ir: int) -> int:
        return (iz - 1) * (nr - 2) + (ir - 1)

    for iz in range(1, nz - 1):
        for ir in range(1, nr - 1):
            row = index(iz, ir)
            if ir == 1:
                rhs[row] -= (1.0 / dr**2 + 1.0 / (2.0 * grid_r[ir] * dr)) * fields[:, iz, 0]
            if ir == nr - 2:
                rhs[row] -= (1.0 / dr**2 - 1.0 / (2.0 * grid_r[ir] * dr)) * fields[:, iz, -1]
            if iz == 1:
                rhs[row] -= fields[:, 0, ir] / dz**2
            if iz == nz - 2:
                rhs[row] -= fields[:, -1, ir] / dz**2
    result = fields.copy()
    result[:, 1:-1, 1:-1] = solver(rhs).T.reshape(count, nz - 2, nr - 2)
    return result


@dataclass(frozen=True)
class HarmonicBasis:
    grid_r_m: np.ndarray
    grid_z_m: np.ndarray
    flux_fields_Wb_per_rad_per_coefficient: np.ndarray

    def manifest(self) -> dict[str, object]:
        return {
            "construction": "rectangular-boundary homogeneous Grad-Shafranov modes",
            "mode_count": len(self.flux_fields_Wb_per_rad_per_coefficient),
            "output_shape": list(self.flux_fields_Wb_per_rad_per_coefficient.shape[1:]),
        }


def build_harmonic_basis(
    grid_r_m: np.ndarray,
    grid_z_m: np.ndarray,
    *,
    mode_count: int = DEFAULT_HARMONIC_MODE_COUNT,
    raw_multiplier: int = 4,
) -> HarmonicBasis:
    """Build weighted-orthonormal homogeneous external/passive field modes."""

    if mode_count < 0:
        raise ValueError("mode_count cannot be negative")
    grid_r = np.asarray(grid_r_m, dtype=np.float64).ravel()
    grid_z = np.asarray(grid_z_m, dtype=np.float64).ravel()
    if mode_count == 0:
        fields = np.empty((0, len(grid_z), len(grid_r)), dtype=np.float64)
        return HarmonicBasis(grid_r, grid_z, fields)
    raw_count = max(raw_multiplier * mode_count, mode_count + 16)
    raw = _harmonic_extension(
        _boundary_modes(len(grid_z), len(grid_r), raw_count),
        grid_r,
        grid_z,
    )
    weights = grid_weights(grid_r, grid_z)
    flat = raw.reshape(raw_count, -1).T
    weighted = np.sqrt(weights)[:, None] * weighted_center_columns(flat, weights)
    q, r = np.linalg.qr(weighted, mode="reduced")
    diagonal = np.abs(np.diag(r))
    keep = diagonal > max(float(np.max(diagonal)) * 1.0e-10, 1.0e-13)
    q = q[:, keep]
    if q.shape[1] < mode_count:
        raise RuntimeError(f"only {q.shape[1]} independent harmonic modes are available")
    fields = (
        q[:, :mode_count] / np.sqrt(weights)[:, None]
    ).T.reshape(mode_count, len(grid_z), len(grid_r))
    return HarmonicBasis(grid_r, grid_z, fields)
