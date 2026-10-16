"""MAST scalar estimates derived from submitted flux and measured inputs."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "fusion_scoring"
if str(SCORER) not in sys.path:
    sys.path.insert(0, str(SCORER))

from contour import encloses_point, find_contours, grid_to_physical, is_closed, resample_closed
from common import AXIS_SIGN
from derive import _bilinear
from lcfs import extract_lcfs
from o_point import find_o_point
from neuralgs.geometry import current_scale_to_ampere, signal_at_efit


MU0 = 4.0e-7 * np.pi
TF_TURNS = {
    # DIII-D: 24 bundles with six turns each.
    "DIII-D": 144,
    # MAST: 24 single-turn centre-column/return-limb circuits.
    "MAST": 24,
}


@lru_cache(maxsize=2)
def scorer_mask(machine: str) -> tuple[np.ndarray, np.ndarray]:
    filename = "d3d_envelope.npz" if machine == "DIII-D" else "mast_envelope.npz"
    with np.load(SCORER / "masks" / filename) as saved:
        mask = saved["mask_coarse"].astype(bool)
    return mask, mask.astype(np.float64)


def flux_surface_contour(
    frame: np.ndarray,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    normalized_level: float = 0.95,
    machine: str = "MAST",
    axis_sign: float | None = None,
    use_machine_envelope: bool = True,
) -> np.ndarray | None:
    """Return the axis-enclosing normalized-poloidal-flux contour."""

    psi = np.asarray(frame, dtype=np.float64)
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    if use_machine_envelope:
        mask, mask_f = scorer_mask(machine)
    else:
        # Strict zero-shot inference cannot depend on a target-machine mask
        # measured outside the submitted row. The grid boundary itself is the
        # generic open-surface bracket used by the same contour algorithm.
        mask = mask_f = None
    sign = float(AXIS_SIGN[machine] if axis_sign is None else axis_sign)
    lcfs = extract_lcfs(
        psi, r, z, machine, mask, mask_f,
        n_iter=18, n_points=256, axis_sign=sign,
    )
    if lcfs is None:
        return None
    _, _, iz, ir, _ = find_o_point(sign * psi, r, z, mask)
    axis = float(psi[iz, ir])
    boundary = float(np.median(_bilinear(psi, r, z, lcfs[:, 0], lcfs[:, 1])))
    level = axis + float(normalized_level) * (boundary - axis)
    candidates = []
    for contour_rc in find_contours(psi, level):
        if len(contour_rc) < 8 or not is_closed(contour_rc):
            continue
        if not encloses_point(contour_rc, iz, ir):
            continue
        candidates.append(contour_rc)
    if not candidates:
        return None
    contour_rc = max(candidates, key=len)
    return resample_closed(grid_to_physical(contour_rc, r, z), 256)


def safety_factor_integral(
    frame: np.ndarray,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    normalized_level: float = 0.95,
    machine: str = "MAST",
    axis_sign: float | None = None,
    use_machine_envelope: bool = True,
) -> float:
    """Compute ``(1/2pi) integral dl/(R |grad psi|)`` on a flux surface."""

    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    psi = np.asarray(frame, dtype=np.float64)
    contour = flux_surface_contour(
        psi,
        r,
        z,
        normalized_level,
        machine=machine,
        axis_sign=axis_sign,
        use_machine_envelope=use_machine_envelope,
    )
    if contour is None:
        return np.nan
    dpsi_dz, dpsi_dr = np.gradient(psi, z, r, edge_order=2)
    gradient = np.hypot(dpsi_dr, dpsi_dz)
    grad_c = _bilinear(gradient, r, z, contour[:, 0], contour[:, 1])
    following = np.roll(contour, -1, axis=0)
    segments = np.sqrt(np.sum((following - contour) ** 2, axis=1))
    grad_next = np.roll(grad_c, -1)
    inverse_grad = 0.5 * (
        1.0 / np.maximum(grad_c, 1.0e-9)
        + 1.0 / np.maximum(grad_next, 1.0e-9)
    )
    radius = np.maximum(0.5 * (contour[:, 0] + following[:, 0]), 0.05)
    return float(np.sum(segments * inverse_grad / radius) / (2.0 * np.pi))


def safety_factor_proxy(
    row: dict,
    flux: np.ndarray,
    frame_indices: np.ndarray | None = None,
    axis_sign: float | None = None,
    use_machine_envelope: bool = True,
) -> np.ndarray:
    """Vacuum-toroidal-field q95 proxy; one global scale remains to calibrate."""

    scale = current_scale_to_ampere(row)
    machine = str(row.get("source", "MAST"))
    tf_key = "magnetics_bcoil" if machine == "DIII-D" else "magnetics_tf_current"
    tf = np.abs(signal_at_efit(row, tf_key, scale))
    r = np.asarray(row["efit_grid_R"], dtype=np.float64)
    z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    integral = np.asarray(
        [
            safety_factor_integral(
                frame,
                r,
                z,
                machine=machine,
                axis_sign=axis_sign,
                use_machine_envelope=use_machine_envelope,
            )
            for frame in np.asarray(flux)
        ],
        dtype=np.float64,
    )
    if len(tf) != len(np.asarray(row["efit_times"])):
        tf = np.interp(
            np.asarray(row["efit_times"], dtype=np.float64),
            np.asarray(row["magnetics_time"], dtype=np.float64),
            np.asarray(row[tf_key], dtype=np.float64) * scale,
        )
    if frame_indices is not None:
        tf = tf[np.asarray(frame_indices, dtype=np.int64)]
    if len(tf) != len(integral):
        raise ValueError("flux frames and selected toroidal-field frames do not align")
    return tf * integral


def toroidal_flux_per_ampere(machine: str) -> float:
    """Return the ideal axisymmetric ``F=R*Bphi`` produced per TF ampere."""

    if machine not in TF_TURNS:
        raise ValueError(f"unsupported machine: {machine!r}")
    return float(MU0 * TF_TURNS[machine] / (2.0 * np.pi))


def physics_q95(
    row: dict,
    flux: np.ndarray,
    frame_indices: np.ndarray | None = None,
    numerical_scale: float = 1.0,
    axis_sign: float | None = None,
    use_machine_envelope: bool = True,
) -> np.ndarray:
    """Estimate q95 from submitted flux and the machine's physical TF turns."""

    machine = str(row.get("source", "MAST"))
    proxy = safety_factor_proxy(
        row,
        flux,
        frame_indices=frame_indices,
        axis_sign=axis_sign,
        use_machine_envelope=use_machine_envelope,
    )
    return numerical_scale * toroidal_flux_per_ampere(machine) * proxy


def physics_q95_prepared(
    prepared: dict[str, np.ndarray],
    flux: np.ndarray,
    *,
    machine: str,
    numerical_scale: float = 1.0,
    axis_sign: float | None = None,
    use_machine_envelope: bool = False,
) -> np.ndarray:
    """Estimate q95 from aligned prepared inputs without reopening a raw row."""

    fields = np.asarray(flux, dtype=np.float64)
    tf = np.asarray(prepared["tf_ampere"], dtype=np.float64).reshape(-1)
    r = np.asarray(prepared["grid_r"], dtype=np.float64)
    z = np.asarray(prepared["grid_z"], dtype=np.float64)
    if len(fields) != len(tf):
        raise ValueError("flux frames and prepared TF current do not align")
    integral = np.asarray(
        [
            safety_factor_integral(
                frame,
                r,
                z,
                machine=machine,
                axis_sign=axis_sign,
                use_machine_envelope=use_machine_envelope,
            )
            for frame in fields
        ],
        dtype=np.float64,
    )
    return numerical_scale * toroidal_flux_per_ampere(machine) * np.abs(tf) * integral


def normalized_poloidal_flux(
    frame: np.ndarray,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    *,
    machine: str,
    axis_sign: float | None = None,
    use_machine_envelope: bool = False,
) -> np.ndarray | None:
    """Return a native-grid 0-at-axis, 1-at-boundary flux coordinate."""

    psi = np.asarray(frame, dtype=np.float64)
    r = np.asarray(grid_r, dtype=np.float64)
    z = np.asarray(grid_z, dtype=np.float64)
    sign = float(AXIS_SIGN[machine] if axis_sign is None else axis_sign)
    contour = flux_surface_contour(
        psi,
        r,
        z,
        machine=machine,
        axis_sign=sign,
        use_machine_envelope=use_machine_envelope,
    )
    if contour is None:
        return None
    mask = scorer_mask(machine)[0] if use_machine_envelope else None
    _, _, iz, ir, _ = find_o_point(sign * psi, r, z, mask)
    axis = float(psi[iz, ir])
    boundary = float(np.median(_bilinear(psi, r, z, contour[:, 0], contour[:, 1])))
    span = boundary - axis
    if not np.isfinite(span) or abs(span) < 1.0e-10:
        return None
    coordinate = (psi - axis) / span
    return coordinate if np.isfinite(coordinate).all() else None


def physical_scalars_from_prepared(
    prepared: dict[str, np.ndarray],
    flux: np.ndarray,
    *,
    machine: str,
    q_scale: float = 1.0,
    beta_scale: float = 1.0,
    axis_sign: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute target-free native q95 and electron-pressure betaN proxies."""

    from my_experiments.neuralgs.validate_beta_physics import electron_beta_n_proxy

    fields = np.asarray(flux, dtype=np.float64)
    q95 = physics_q95_prepared(
        prepared,
        fields,
        machine=machine,
        numerical_scale=q_scale,
        axis_sign=axis_sign,
        use_machine_envelope=False,
    )
    beta_n = np.full(len(fields), np.nan, dtype=np.float64)
    r = np.asarray(prepared["grid_r"], dtype=np.float64)
    z = np.asarray(prepared["grid_z"], dtype=np.float64)
    for index, frame in enumerate(fields):
        psi_n = normalized_poloidal_flux(
            frame,
            r,
            z,
            machine=machine,
            axis_sign=axis_sign,
            use_machine_envelope=False,
        )
        if psi_n is None:
            continue
        beta_n[index] = beta_scale * electron_beta_n_proxy(
            psi_n,
            np.asarray(prepared["token_coords"])[index],
            np.asarray(prepared["token_values"])[index],
            np.asarray(prepared["token_mask"])[index],
            float(np.asarray(prepared["ip_abs"])[index]),
            float(np.asarray(prepared["tf_ampere"])[index]),
            r,
            z,
            machine=machine,
        )
    return q95, beta_n


def fit_affine_proxy(proxy: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(proxy) & np.isfinite(target)
    x = np.asarray(proxy, dtype=np.float64)[valid]
    y = np.asarray(target, dtype=np.float64)[valid]
    design = np.column_stack([x, np.ones(len(x))])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def apply_affine_proxy(
    proxy: np.ndarray,
    slope: float,
    intercept: float,
    fallback: np.ndarray | float,
) -> np.ndarray:
    prediction = slope * np.asarray(proxy, dtype=np.float64) + intercept
    return np.where(np.isfinite(prediction), prediction, fallback)
