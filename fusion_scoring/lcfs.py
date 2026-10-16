"""Last-closed-flux-surface (LCFS) extraction from a 65x65 poloidal flux map.

The scorer never receives an LCFS contour from participants -- it extracts one from the predicted
flux map with the SAME published procedure it uses on the ground-truth flux map. Because both
sides pass through identical code, a perfect prediction reproduces the ground-truth contour
exactly and scores D_LCFS = 0 by construction (this is why `lcfs_reference: extracted` is the
default; see build_ref.py / evaluation.md).

Definition of "last closed flux surface" used here: on phi = AXIS_SIGN * psi (so the magnetic
axis is always a maximum of phi), the LCFS is the contour at the LOWEST flux level that still
encloses the axis with a single closed curve lying inside the machine's plasma envelope. Lowering
the level further makes the surface open onto the wall or split at an x-point -- that transition
defines the boundary. We binary-search that critical level.
"""
from __future__ import annotations

import numpy as np

from contour import (
    encloses_point,
    find_contours,
    grid_to_physical,
    is_closed,
    resample_closed,
)
from o_point import find_o_point


def _in_mask(contour_rc: np.ndarray, mask_f: np.ndarray | None) -> bool:
    """All contour points fall inside the (float) envelope mask, bilinearly sampled."""
    if mask_f is None:
        return True
    nr, nc = mask_f.shape
    rows = np.clip(contour_rc[:, 0], 0, nr - 1)
    cols = np.clip(contour_rc[:, 1], 0, nc - 1)
    r0 = np.floor(rows).astype(int); c0 = np.floor(cols).astype(int)
    r1 = np.minimum(r0 + 1, nr - 1); c1 = np.minimum(c0 + 1, nc - 1)
    fr = rows - r0; fc = cols - c0
    v = (mask_f[r0, c0] * (1 - fr) * (1 - fc) + mask_f[r1, c0] * fr * (1 - fc)
         + mask_f[r0, c1] * (1 - fr) * fc + mask_f[r1, c1] * fr * fc)
    return bool(np.all(v > 0.5))


def _best_contour_at(phi: np.ndarray, level: float, iz: int, ir: int,
                     mask_f: np.ndarray | None):
    """Closed, axis-enclosing, in-envelope contour at `level`, or None."""
    for c in find_contours(phi, level):
        if len(c) < 4 or not is_closed(c):
            continue
        if not encloses_point(c, iz, ir):
            continue
        if not _in_mask(c, mask_f):
            continue
        return c
    return None


def _extract_one_sign(phi: np.ndarray, R: np.ndarray, Z: np.ndarray,
                      mask_coarse: np.ndarray | None, mask_f: np.ndarray | None,
                      n_iter: int, n_points: int):
    if not np.isfinite(phi).all():
        phi = np.nan_to_num(phi, nan=np.nanmin(phi) - 1.0, posinf=np.nanmax(phi),
                            neginf=np.nanmin(phi))
    # Magnetic axis = strongest interior LOCAL maximum of phi (the O-point), not a global
    # argmax: on ~1.4% of MAST frames the coil-driven flux at the envelope edge exceeds the
    # flux at the plasma axis, and a global argmax walks off to the mask rim, after which the
    # bisection finds no closed surface and the frame is lost. See o_point.py.
    mask = mask_coarse if (mask_coarse is not None and mask_coarse.any()) else None
    _, _, iz, ir, _ = find_o_point(phi, R, Z, mask)
    axis_val = float(phi[iz, ir])
    border = np.concatenate([phi[0, :], phi[-1, :], phi[:, 0], phi[:, -1]])
    lo = float(np.min(border))       # low bracket: level that opens onto the wall (invalid)
    hi = axis_val - 1e-9             # high bracket: tight closed loop around the axis (valid)
    if not (hi > lo):
        return None
    best = _best_contour_at(phi, hi, iz, ir, mask_f)
    if best is None:
        return None                  # even a tight loop is not closed/enclosing -> give up
    # Binary-search the lowest still-valid level (the last closed surface).
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        c = _best_contour_at(phi, mid, iz, ir, mask_f)
        if c is not None:
            hi = mid
            best = c
        else:
            lo = mid
    phys = grid_to_physical(best, R, Z)
    return resample_closed(phys, n_points)


def extract_lcfs_with_sign(psi: np.ndarray, R: np.ndarray, Z: np.ndarray, machine: str,
                           mask_coarse: np.ndarray | None = None,
                           mask_f: np.ndarray | None = None,
                           n_iter: int = 22, n_points: int = 512,
                           axis_sign: float | None = None):
    """Extract the LCFS, and report the flux sign under which it was found.

    Returns ``(contour, sign)``: an (n_points, 2) physical (R, Z) polygon and the sign `s` such
    that `s * psi` puts the magnetic axis at a maximum, or ``(None, sign)`` on failure.

    `axis_sign` pins the orientation: the map is read ONLY that way up, and a frame with no
    closed axis-enclosing surface in that orientation fails outright. This is what the scorer
    passes, using the flux sign it determined once for the whole fold.

    With `axis_sign=None` (the default, and what the reference builder uses) the machine's own
    convention is tried first and the opposite sign is a fallback. Do NOT rely on that fallback
    to score a sign-inverted submission: it is first-success, not best-fit, and on a NOISY
    inverted map the wrong orientation frequently does yield some closed contour around a noise
    maximum -- so the search stops there and returns a spurious boundary. Measured on a 30%-noise
    inverted submission, that path gave D_LCFS = 1.0 (total failure) and Consistency = 0 even
    though R2_psi was 0.913. Hence the fold-global sign: convention is decided from the pooled
    flux residual, which is robust, and then imposed here.

    Pinning the sign does NOT tighten scoring for a correctly-signed submission, i.e. it costs
    nobody the fallback they used to get. Swept over 413 frames per machine at noise levels
    0/0.1/0.3/0.6/1.0 x std(psi), the machine's own orientation succeeded on EVERY frame -- the
    fallback never fired on a correctly-signed map, so removing it there is a no-op. Confirmed
    end-to-end: a correctly-signed noisy submission scores bit-identically under pre-v3.1 and
    v3.1 code on every term.
    """
    from common import AXIS_SIGN
    if axis_sign is not None:
        s = float(axis_sign)
        out = _extract_one_sign(s * np.asarray(psi, dtype=np.float64), R, Z,
                                mask_coarse, mask_f, n_iter, n_points)
        return out, s
    sign = AXIS_SIGN.get(machine, 1.0)
    for s in (sign, -sign):
        out = _extract_one_sign(s * np.asarray(psi, dtype=np.float64), R, Z,
                                mask_coarse, mask_f, n_iter, n_points)
        if out is not None:
            return out, s
    return None, sign


def extract_lcfs(psi: np.ndarray, R: np.ndarray, Z: np.ndarray, machine: str,
                 mask_coarse: np.ndarray | None = None, mask_f: np.ndarray | None = None,
                 n_iter: int = 22, n_points: int = 512, axis_sign: float | None = None):
    """Extract the LCFS as an (n_points, 2) physical (R, Z) contour, or None on failure.

    Thin wrapper over `extract_lcfs_with_sign` that drops the sign. With `axis_sign` left None
    this is byte-for-byte the pre-v3.1 behaviour, which is what the reference builder relies on.
    """
    return extract_lcfs_with_sign(psi, R, Z, machine, mask_coarse, mask_f,
                                  n_iter, n_points, axis_sign)[0]


def major_radius(contour_xy: np.ndarray) -> float:
    """Geometric major radius (Rmax + Rmin) / 2 of a physical (R, Z) contour."""
    R = contour_xy[:, 0]
    return 0.5 * (float(np.max(R)) + float(np.min(R)))
