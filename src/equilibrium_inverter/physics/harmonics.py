"""Homogeneous Grad--Shafranov modes for unmodelled external fields."""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def _trapezoid_weights(values: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(coordinates) < 3 or np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("harmonic grids must be increasing and contain at least 3 points")
    result = np.empty_like(coordinates)
    result[0] = 0.5 * (coordinates[1] - coordinates[0])
    result[-1] = 0.5 * (coordinates[-1] - coordinates[-2])
    result[1:-1] = 0.5 * (coordinates[2:] - coordinates[:-2])
    return result


def _dirichlet_matrix(grid_r: np.ndarray, grid_z: np.ndarray) -> scipy.sparse.csc_matrix:
    nr, nz = len(grid_r), len(grid_z)
    dr = float(np.mean(np.diff(grid_r)))
    dz = float(np.mean(np.diff(grid_z)))
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    def index(iz: int, ir: int) -> int:
        return (iz - 1) * (nr - 2) + ir - 1

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
    size = (nz - 2) * (nr - 2)
    return scipy.sparse.csc_matrix((values, (rows, columns)), shape=(size, size))


def _boundary_modes(nz: int, nr: int, count: int) -> np.ndarray:
    perimeter: list[tuple[int, int]] = []
    perimeter.extend((0, ir) for ir in range(nr))
    perimeter.extend((iz, nr - 1) for iz in range(1, nz))
    perimeter.extend((nz - 1, ir) for ir in range(nr - 2, -1, -1))
    perimeter.extend((iz, 0) for iz in range(nz - 2, 0, -1))
    phase = np.arange(len(perimeter), dtype=np.float64) / len(perimeter)
    fields = np.zeros((count, nz, nr), dtype=np.float64)
    for mode in range(count):
        frequency = mode // 2 + 1
        boundary = (
            np.cos(2.0 * np.pi * frequency * phase)
            if mode % 2 == 0
            else np.sin(2.0 * np.pi * frequency * phase)
        )
        for value, (iz, ir) in zip(boundary, perimeter):
            fields[mode, iz, ir] = value
    return fields


def _extend_harmonics(
    boundary_fields: np.ndarray,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
) -> np.ndarray:
    fields = np.asarray(boundary_fields, dtype=np.float64)
    count, nz, nr = fields.shape
    dr = float(np.mean(np.diff(grid_r)))
    dz = float(np.mean(np.diff(grid_z)))
    solve = scipy.sparse.linalg.factorized(_dirichlet_matrix(grid_r, grid_z))
    rhs = np.zeros(((nz - 2) * (nr - 2), count), dtype=np.float64)

    def index(iz: int, ir: int) -> int:
        return (iz - 1) * (nr - 2) + ir - 1

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
    result[:, 1:-1, 1:-1] = solve(rhs).T.reshape(count, nz - 2, nr - 2)
    return result


def build_harmonic_basis(
    grid_r_m: np.ndarray,
    grid_z_m: np.ndarray,
    mode_count: int,
) -> np.ndarray:
    """Return weighted-orthonormal fields satisfying ``Delta* psi = 0``."""

    grid_r = np.asarray(grid_r_m, dtype=np.float64).reshape(-1)
    grid_z = np.asarray(grid_z_m, dtype=np.float64).reshape(-1)
    if mode_count < 0:
        raise ValueError("mode_count cannot be negative")
    if mode_count == 0:
        return np.empty((0, len(grid_z), len(grid_r)), dtype=np.float32)
    raw_count = max(4 * mode_count, mode_count + 8)
    raw = _extend_harmonics(
        _boundary_modes(len(grid_z), len(grid_r), raw_count), grid_r, grid_z
    )
    weights = np.outer(_trapezoid_weights(grid_z), _trapezoid_weights(grid_r)).reshape(-1)
    flat = raw.reshape(raw_count, -1).T
    means = weights @ flat / weights.sum()
    centered = flat - means[None]
    weighted = np.sqrt(weights)[:, None] * centered
    q, r = np.linalg.qr(weighted, mode="reduced")
    diagonal = np.abs(np.diag(r))
    keep = diagonal > max(float(diagonal.max(initial=0.0)) * 1.0e-10, 1.0e-13)
    q = q[:, keep]
    if q.shape[1] < mode_count:
        raise RuntimeError("insufficient independent harmonic boundary modes")
    fields = (q[:, :mode_count] / np.sqrt(weights)[:, None]).T
    return fields.reshape(mode_count, len(grid_z), len(grid_r)).astype(np.float32)
