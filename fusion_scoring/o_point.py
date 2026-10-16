"""Locate the magnetic axis as a LOCAL maximum (the plasma O-point), not a global argmax.

Why not a global argmax
-----------------------
The magnetic axis is a *local* maximum of phi = AXIS_SIGN * psi -- an O-point of the poloidal
flux. A global argmax inside the plasma envelope coincides with it only when the plasma's own
flux dominates everything else inside the envelope. On ~1.4% of MAST frames it does not: the
coil-driven flux at the envelope edge is HIGHER than the flux at the plasma axis, so argmax
walks off to the mask boundary and the returned "axis" is the envelope edge, after which the
LCFS bisection seeded from it finds no closed surface. EFIT's reported axis on those same
frames is a true local maximum of psi 98.8% of the time -- the plasma is really there.

The method
----------
Enumerate interior local maxima of phi and take the strongest, ignoring the envelope boundary:

    1. erode the envelope mask by `edge_cells` so candidates cannot sit on its rim
    2. a candidate is a cell >= all 8 neighbours (a discrete elliptic critical point)
    3. return the candidate with the largest phi, refined to sub-pixel by a parabolic fit

Falls back to the global argmax when no interior local maximum exists (genuinely empty frames),
so behaviour is unchanged wherever the plain argmax was already right.
"""
from __future__ import annotations

import numpy as np


def _erode(mask, cells):
    """Binary-erode a boolean mask by `cells` in 4-connectivity (no scipy dependency)."""
    m = mask.copy()
    for _ in range(int(cells)):
        e = m.copy()
        e[1:, :] &= m[:-1, :]
        e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        m = e
    return m


def local_maxima(phi, mask=None, edge_cells=2):
    """Boolean array marking cells that are >= all 8 neighbours, inside the eroded mask."""
    p = np.where(np.isfinite(phi), phi, -np.inf)
    ok = np.ones_like(p, bool)
    ok[0, :] = ok[-1, :] = ok[:, 0] = ok[:, -1] = False
    if mask is not None:
        ok &= _erode(mask.astype(bool), edge_cells)
    c = p[1:-1, 1:-1]
    ge = np.ones_like(c, bool)
    for dz in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if dz == 0 and dr == 0:
                continue
            ge &= c >= p[1 + dz:p.shape[0] - 1 + dz, 1 + dr:p.shape[1] - 1 + dr]
    out = np.zeros_like(ok)
    out[1:-1, 1:-1] = ge
    return out & ok


def _subpixel(phi, iz, ir, R, Z):
    """Parabolic sub-pixel refinement about an integer critical point."""
    def d1(a, b, c):
        den = a - 2.0 * b + c
        return 0.0 if den == 0 else float(np.clip(0.5 * (a - c) / den, -1.0, 1.0))
    dz = d1(phi[iz - 1, ir], phi[iz, ir], phi[iz + 1, ir]) if 0 < iz < phi.shape[0] - 1 else 0.0
    dr = d1(phi[iz, ir - 1], phi[iz, ir], phi[iz, ir + 1]) if 0 < ir < phi.shape[1] - 1 else 0.0
    return (float(R[ir]) + dr * float(R[1] - R[0]),
            float(Z[iz]) + dz * float(Z[1] - Z[0]))


def find_o_point(phi, R, Z, mask=None, edge_cells=2):
    """Return (R_axis, Z_axis, iz, ir, used_local) for the plasma O-point of phi (axis = MAX).

    used_local is False when we had to fall back to the global argmax (no interior local max).
    """
    p = np.where(np.isfinite(phi), phi, -np.inf)
    lm = local_maxima(p, mask, edge_cells)
    if lm.any():
        cand = np.where(lm, p, -np.inf)
        iz, ir = np.unravel_index(np.argmax(cand), p.shape)
        used_local = True
    else:                                            # fallback = plain global argmax
        m = np.where(mask, p, -np.inf) if mask is not None else p
        iz, ir = np.unravel_index(np.argmax(m), p.shape)
        used_local = False
    Ra, Za = _subpixel(p, int(iz), int(ir), R, Z)
    return Ra, Za, int(iz), int(ir), used_local
