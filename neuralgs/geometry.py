"""Machine-independent axisymmetric magnetic-field rendering."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.special import ellipe, ellipk

from data_fixes import fix_d3d_ip_times

MU0 = 4.0e-7 * np.pi
TF_TURNS = {"DIII-D": 144.0, "MAST": 24.0}
_COIL_BASIS_CACHE: dict[str, tuple[list[str], np.ndarray]] = {}


def finite_interp(times, values, query) -> np.ndarray:
    """Interpolate a signal after removing holes in a union time grid."""

    t = np.asarray(times, dtype=np.float64).ravel()
    y = np.asarray(values, dtype=np.float64).ravel()
    q = np.asarray(query, dtype=np.float64).ravel()
    n = min(len(t), len(y))
    good = np.isfinite(t[:n]) & np.isfinite(y[:n])
    if not good.any():
        return np.zeros(len(q), dtype=np.float64)
    t, y = t[:n][good], y[:n][good]
    order = np.argsort(t)
    t, y = t[order], y[order]
    t, unique = np.unique(t, return_index=True)
    y = y[unique]
    if len(t) == 1:
        return np.full(len(q), y[0], dtype=np.float64)
    return np.interp(q, t, y, left=y[0], right=y[-1])


def current_scale_to_ampere(row: dict) -> float:
    """Handle both the current kA release and a possible future SI release."""

    ip = np.asarray(row.get("magnetics_plasma_current", []), dtype=np.float64)
    finite = np.abs(ip[np.isfinite(ip)])
    if not len(finite):
        return 1.0
    return 1000.0 if np.nanpercentile(finite, 99.0) < 1.0e5 else 1.0


def signal_at_efit(row: dict, key: str, amp_scale: float = 1.0) -> np.ndarray:
    query = np.asarray(row["efit_times"], dtype=np.float64)
    if not key or key not in row:
        return np.zeros(len(query), dtype=np.float64)
    if key == "magnetics_plasma_current" and str(row.get("source", "DIII-D")) == "DIII-D":
        times = fix_d3d_ip_times(row)
    else:
        times = row.get(f"{key}_times", row["magnetics_time"])
    return amp_scale * finite_interp(times, row[key], query)


def canonical_current_sign(ip: np.ndarray) -> float:
    """Return one shot-global sign, avoiding orientation changes on noisy ramp frames.

    EFIT stores DIII-D flux with a fixed axis orientation even though the corpus
    contains both plasma-current directions.  Canonicalizing all poloidal currents
    by the dominant Ip sign lets the learned profile model use one orientation; the
    official scorer handles the one remaining fold-global sign convention.
    """

    ip = np.asarray(ip, dtype=np.float64)
    finite = ip[np.isfinite(ip)]
    if not len(finite):
        return 1.0
    cutoff = 0.2 * np.nanpercentile(np.abs(finite), 95.0)
    active = finite[np.abs(finite) >= cutoff]
    value = float(np.nanmedian(active if len(active) else finite))
    return -1.0 if value < 0 else 1.0


def loop_psi_per_ampere(
    target_r: np.ndarray,
    target_z: np.ndarray,
    source_r: np.ndarray | float,
    source_z: np.ndarray | float,
    softening: float = 0.0,
) -> np.ndarray:
    """Poloidal flux ``psi=R*A_phi`` from circular filaments, per ampere.

    Inputs broadcast.  ``softening`` regularizes the logarithmic self-field when
    this function is used to integrate a finite plasma-current cell.
    """

    r = np.maximum(np.asarray(target_r, dtype=np.float64), 1.0e-7)
    a = np.maximum(np.asarray(source_r, dtype=np.float64), 1.0e-7)
    dz = np.asarray(target_z, dtype=np.float64) - np.asarray(source_z, dtype=np.float64)
    denom = (a + r) ** 2 + dz**2 + float(softening) ** 2
    k2 = np.clip(4.0 * a * r / denom, 1.0e-12, 1.0 - 1.0e-9)
    k = np.sqrt(k2)
    bracket = (1.0 - 0.5 * k2) * ellipk(k2) - ellipe(k2)
    return MU0 * np.sqrt(a * r) * bracket / (np.pi * k)


def _geometry_cache_key(row: dict, grid_r: np.ndarray, grid_z: np.ndarray) -> str:
    digest = hashlib.sha1()
    for key in (
        "coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height",
        "coil_angle1", "coil_angle2",
    ):
        arr = np.asarray(row.get(key, []))
        digest.update(arr.astype(str).tobytes() if arr.dtype.kind in "OUS" else arr.tobytes())
    digest.update(np.asarray(grid_r, dtype=np.float64).tobytes())
    digest.update(np.asarray(grid_z, dtype=np.float64).tobytes())
    digest.update(str(row.get("source", "DIII-D")).encode())
    return digest.hexdigest()


def _rectangle_filaments(r0: float, z0: float, width: float, height: float, detailed: bool):
    """Small tensor-product quadrature for a finite conductor rectangle."""

    nq = 3 if detailed and max(abs(width), abs(height)) > 0.04 else 1
    if nq == 1:
        return np.asarray([r0]), np.asarray([z0])
    # Midpoint quadrature avoids putting a filament exactly on a rectangle edge.
    u = (np.arange(nq, dtype=np.float64) + 0.5) / nq - 0.5
    rr, zz = np.meshgrid(r0 + width * u, z0 + height * u)
    return rr.ravel(), zz.ravel()


def coil_basis(row: dict, grid_r: np.ndarray, grid_z: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Return one physical flux basis per current input column."""

    cache_key = _geometry_cache_key(row, grid_r, grid_z)
    cached = _COIL_BASIS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    joins = np.asarray(row.get("coil_input_column", []), dtype=str)
    coil_r = np.asarray(row.get("coil_R", []), dtype=np.float64)
    coil_z = np.asarray(row.get("coil_Z", []), dtype=np.float64)
    width = np.asarray(row.get("coil_width", np.zeros(len(coil_r))), dtype=np.float64)
    height = np.asarray(row.get("coil_height", np.zeros(len(coil_r))), dtype=np.float64)
    angle1 = np.asarray(row.get("coil_angle1", np.zeros(len(coil_r))), dtype=np.float64)
    angle2 = np.asarray(row.get("coil_angle2", np.zeros(len(coil_r))), dtype=np.float64)
    n = min(len(joins), len(coil_r), len(coil_z), len(width), len(height), len(angle1), len(angle2))
    joins, coil_r, coil_z, width, height, angle1, angle2 = (
        joins[:n], coil_r[:n], coil_z[:n], width[:n], height[:n], angle1[:n], angle2[:n]
    )
    rr, zz = np.meshgrid(grid_r, grid_z)
    source = str(row.get("source", "DIII-D"))
    keys: list[str] = []
    fields: list[np.ndarray] = []
    for key in dict.fromkeys(joins.tolist()):
        if not key:
            continue
        indices = np.flatnonzero(joins == key)
        # MAST P-coil signals are total ampere-turns but geometry is one row per
        # turn.  Solenoid current is per turn, so all 656 elements must be summed.
        group_divisor = len(indices) if source == "MAST" and key.startswith("magnetics_p") else 1
        basis = np.zeros_like(rr, dtype=np.float64)
        for index in indices:
            if not (np.isfinite(coil_r[index]) and np.isfinite(coil_z[index])):
                continue
            # EFIT's two-angle skew convention describes six DIII-D
            # parallelograms, but the release does not include the polygon
            # vertices.  A center filament is invariant to that unresolved
            # convention; axis-aligned rectangles use finite-area quadrature.
            unskewed = abs(angle1[index]) < 1.0e-8 and abs(angle2[index]) < 1.0e-8
            qr, qz = _rectangle_filaments(
                coil_r[index], coil_z[index], width[index], height[index],
                source == "DIII-D" and unskewed,
            )
            element = np.zeros_like(rr, dtype=np.float64)
            for fr, fz in zip(qr, qz):
                element += loop_psi_per_ampere(rr, zz, fr, fz)
            basis += element / max(len(qr), 1) / group_divisor
        keys.append(key)
        fields.append(basis)
    matrix = np.stack(fields).astype(np.float32) if fields else np.empty((0, len(grid_z), len(grid_r)), np.float32)
    result = (keys, matrix)
    _COIL_BASIS_CACHE[cache_key] = result
    return result


@dataclass
class MagneticInputs:
    vacuum_psi: np.ndarray
    ip_abs: np.ndarray
    global_features: np.ndarray
    canonical_global_features: np.ndarray
    tf_ampere: np.ndarray
    tf_turns: np.ndarray
    canonical_sign: float


def render_magnetic_inputs(
    row: dict,
    frame_indices: np.ndarray | None = None,
) -> MagneticInputs:
    """Render canonical PF vacuum flux and fixed-scale global inputs."""

    query = np.asarray(row["efit_times"], dtype=np.float64)
    if frame_indices is None:
        selected = np.arange(len(query), dtype=np.int64)
    else:
        selected = np.asarray(frame_indices, dtype=np.int64).ravel()
        if len(selected) and (selected.min() < 0 or selected.max() >= len(query)):
            raise IndexError("frame index is outside efit_times")
    grid_r = np.asarray(row["efit_grid_R"], dtype=np.float64)
    grid_z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    scale = current_scale_to_ampere(row)
    source = str(row.get("source", "DIII-D"))
    if source not in TF_TURNS:
        raise ValueError(f"unsupported machine for TF turns: {source!r}")
    ip = signal_at_efit(row, "magnetics_plasma_current", scale)
    sign = canonical_current_sign(ip)

    keys, basis = coil_basis(row, grid_r, grid_z)
    if len(keys):
        currents = np.column_stack([signal_at_efit(row, key, scale) for key in keys])
        vacuum = np.einsum("tc,czr->tzr", sign * currents[selected], basis, optimize=True)
    else:
        vacuum = np.zeros((len(selected), len(grid_z), len(grid_r)), dtype=np.float64)

    tf_key = "magnetics_bcoil" if source == "DIII-D" else "magnetics_tf_current"
    sol_key = "magnetics_ECOILA" if source == "DIII-D" else "magnetics_sol_current"
    tf = np.abs(signal_at_efit(row, tf_key, scale))
    sol = sign * signal_at_efit(row, sol_key, scale)
    ip_abs = np.abs(ip)
    time_s = query / 1000.0
    if len(query) > 1:
        dt = np.maximum(np.gradient(time_s), 1.0e-6)
        dip_dt = np.gradient(ip_abs) / dt
    else:
        dip_dt = np.zeros(1)
    global_features = np.column_stack(
        [
            np.log1p(ip_abs / 1.0e5),
            np.log1p(tf / 1.0e4),
            np.sign(sol) * np.log1p(np.abs(sol) / 1.0e4),
            np.sign(dip_dt) * np.log1p(np.abs(dip_dt) / 1.0e5),
            np.clip(time_s / 10.0, -2.0, 2.0),
            np.log1p(tf / np.maximum(ip_abs, 1.0e3)),
        ]
    )
    # Learned cross-device closure features must not reveal absolute device
    # size, current scale, shot duration, or machine-specific coil-current
    # conventions. Normalize the analytic vacuum flux by its similarity scale
    # mu0*Ip*R and retain only a relative current-ramp rate and shot phase.
    r_ref = np.sqrt(max(float(grid_r[0] * grid_r[-1]), 1.0e-8))
    flux_scale = np.maximum(MU0 * ip_abs[selected] * r_ref, 1.0e-3)
    canonical_vacuum = np.clip(vacuum / flux_scale[:, None, None], -4.0, 4.0)
    flat_vacuum = canonical_vacuum.reshape(len(selected), -1)
    finite_ip = ip_abs[np.isfinite(ip_abs)]
    active_current_floor = max(
        0.05 * float(np.nanpercentile(finite_ip, 95.0)) if len(finite_ip) else 0.0,
        1.0e3,
    )
    relative_ramp = dip_dt[selected] / np.maximum(ip_abs[selected], active_current_floor)
    duration = float(np.nanmax(time_s) - np.nanmin(time_s)) if len(time_s) else 0.0
    if duration > 1.0e-8:
        phase = 2.0 * (time_s[selected] - np.nanmin(time_s)) / duration - 1.0
    else:
        phase = np.zeros(len(selected), dtype=np.float64)
    canonical_global_features = np.column_stack(
        [
            flat_vacuum.mean(axis=1),
            flat_vacuum.std(axis=1),
            flat_vacuum.min(axis=1),
            flat_vacuum.max(axis=1),
            np.sign(relative_ramp) * np.log1p(np.abs(relative_ramp)),
            np.clip(phase, -1.0, 1.0),
        ]
    )
    return MagneticInputs(
        vacuum_psi=np.asarray(vacuum, dtype=np.float32),
        ip_abs=np.asarray(ip_abs[selected], dtype=np.float32),
        global_features=np.asarray(global_features[selected], dtype=np.float32),
        canonical_global_features=np.asarray(
            canonical_global_features, dtype=np.float32
        ),
        tf_ampere=np.asarray(tf[selected], dtype=np.float32),
        tf_turns=np.full(len(selected), TF_TURNS[source], dtype=np.float32),
        canonical_sign=sign,
    )


def plasma_green_matrix(
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    source_stride: int = 2,
    softening_cells: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the free-space Green operator from coarse J_phi cells to the full grid.

    The returned matrix maps current density in A/m^2 to psi in Wb/rad and
    already includes trapezoidal ``dR*dZ`` quadrature weights.
    """

    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    source_indices = np.arange(0, len(r), source_stride, dtype=int)
    if source_indices[-1] != len(r) - 1:
        source_indices = np.append(source_indices, len(r) - 1)
    sr, sz = r[source_indices], z[source_indices]
    trr, tzz = np.meshgrid(r, z)
    srr, szz = np.meshgrid(sr, sz)
    dr = float(np.mean(np.diff(sr)))
    dz = float(np.mean(np.diff(sz)))
    softening = softening_cells * np.sqrt(dr * dr + dz * dz)
    green = loop_psi_per_ampere(
        trr.reshape(-1, 1), tzz.reshape(-1, 1),
        srr.reshape(1, -1), szz.reshape(1, -1), softening=softening,
    )
    wr = np.ones(len(sr)); wz = np.ones(len(sz))
    wr[[0, -1]] = 0.5; wz[[0, -1]] = 0.5
    weights = (wz[:, None] * wr[None, :] * dr * dz).reshape(-1)
    green *= weights[None, :]
    return green.astype(np.float32), sr.astype(np.float32), sz.astype(np.float32), weights.astype(np.float32)
