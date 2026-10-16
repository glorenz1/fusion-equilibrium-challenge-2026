"""psi -> scalar derivations f_j for the consistency metric.

Each f_j maps a single-frame poloidal flux map psi(Z,R) to an equilibrium scalar, using the SAME
code on the ground-truth psi (at ref-build time, ref_builder/build_ref.py) and on the predicted
psi (at scoring time, parallel.py). That symmetry is the metric's guarantee: a perfect psi gives
f_j(psi_pred) == f_j(psi_gt) exactly, so every consistency term scores 1 by construction --
exactly how D_LCFS already treats the boundary. A scalar can therefore no longer be earned by a
regression head that is decoupled from the submitted flux map.

The seven derived scalars (CONS_SCALARS in common.py):
    R_axis, Z_axis   magnetic axis = strongest interior local maximum of phi (o_point.py),
                     parabolic sub-pixel refinement (~mm-level, so the term is non-trivial)
    kappa            (Zmax - Zmin) / (Rmax - Rmin) of the LCFS
    tri_top/tri_bot  (Rgeo - R_at_Zmax|Zmin) / a of the LCFS
    volume           2*pi * first area-moment of the LCFS polygon about the Z axis
    li               internal inductance, ITER li(2) form <Bp^2>_V / (mu0 Ip / lp)^2 with
                     Bp = |grad psi| / R and mu0 Ip = loop-integral of Bp dl (Ampere); this is
                     the convention EFIT's `LI` reports (Lao et al., FST 48 (2005), a-file docs)

`dsep` was an eighth scalar in v2 and is dropped in v3. Two independent reasons: the DIII-D
label it was validated against is the a-file X-point<->limiter clearance, not a drsep (corr
-0.041 over 207,402 frames -- the wrong target, not a numerical bug), and the saddle-finding
extraction never worked on single-null DIII-D, where the secondary separatrix routinely sits
off-grid. The corpus is now diverted-only, which removes the classification role it also played.

CONVENTION INDEPENDENCE: because f_j is applied identically to both psi maps, absolute
normalisation (psi units, the 2*pi in Bp, mu0) cancels out of the consistency ratio. The forms
above were validated against fetched EFIT labels on the full test folds (see
codabench/SCALAR_DERIVATION_VALIDATION.md): correlations +0.95..+0.99 on the well-posed frames.

SIGN INDEPENDENCE (v3.1): six of the seven scalars are already invariant under psi -> -psi --
kappa/tri_top/tri_bot/volume are read off the LCFS polygon, which lcfs.py recovers under either
sign, and li depends on psi only through |grad psi|. Measured over 436 frames on both machines,
their error under a global sign flip is exactly 0. R_axis/Z_axis are the exception: locating an
O-point requires knowing which way is "up" in flux. Hence the `axis_sign` argument -- the scorer
passes the orientation in which the prediction's own LCFS was found, so a submission written in
the other machine's flux convention is read correctly instead of yielding an axis ~1 m adrift.

Every function returns np.nan on failure (degenerate frame, no closed surface, no x-point pair),
never raises -- the reduction masks nan targets and mean-substitutes nan predictions.

Deviation from the validated prototype (metric_v2_prototype/derive_scalars.py): point-in-polygon
uses a dependency-free even-odd ray cast instead of matplotlib.path, and the saddle scan is
vectorised. Both are semantically identical (up to measure-zero boundary-point ties) and keep the
scorer image free of extra hot-loop dependencies.
"""
from __future__ import annotations

import numpy as np

from common import AXIS_SIGN
from lcfs import extract_lcfs
from o_point import find_o_point

SCALAR_NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]

# Physical-plausibility ranges. A derived value outside its range means the extraction FAILED on
# that frame (a pathological LCFS/saddle from a degraded psi) -- return nan so the reduction
# treats it as a penalised failure rather than letting one absurd outlier dominate pooled R2.
# (R_axis/Z_axis are unbounded here -- they are kept in range by the grid and never blow up.)
PLAUSIBLE = {
    "kappa": (0.8, 4.0),      # tokamak elongation
    "tri_top": (-1.0, 1.5),   # triangularity
    "tri_bot": (-1.0, 1.5),
    "volume": (0.0, 100.0),   # m^3
    "li": (0.1, 5.0),         # internal inductance
}


def _clamp_to_nan(out):
    """Map physically-impossible derived values to nan (a failed extraction, not a real scalar)."""
    for k, (lo, hi) in PLAUSIBLE.items():
        v = out.get(k, np.nan)
        if np.isfinite(v) and not (lo <= v <= hi):
            out[k] = np.nan
    return out


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _phi(psi, machine, axis_sign=None):
    """Signed flux with the magnetic axis as a maximum; nan-filled to a low value.

    `axis_sign` overrides the machine's reference convention. None (the default, and what the
    reference builder always uses) means AXIS_SIGN[machine].
    """
    p = np.asarray(psi, dtype=np.float64)
    s = AXIS_SIGN.get(machine, 1.0) if axis_sign is None else float(axis_sign)
    phi = s * p
    if not np.isfinite(phi).all():
        lo = np.nanmin(phi) if np.isfinite(phi).any() else 0.0
        phi = np.nan_to_num(phi, nan=lo - 1.0, posinf=np.nanmax(phi), neginf=lo - 1.0)
    return phi


def magnetic_axis(psi, R, Z, machine, mask_coarse=None, axis_sign=None):
    """(R_axis, Z_axis, iz, ir): sub-pixel O-point position + its integer peak cell."""
    phi = _phi(psi, machine, axis_sign)
    mask = np.asarray(mask_coarse, bool) if mask_coarse is not None else None
    if mask is not None and not mask.any():
        mask = None
    Ra, Za, iz, ir, _ = find_o_point(phi, np.asarray(R, float), np.asarray(Z, float), mask)
    return Ra, Za, iz, ir


def _shape_from_contour(C):
    """Elongation, upper/lower triangularity and toroidal volume from an (N,2) (R,Z) polygon."""
    R = C[:, 0]
    Z = C[:, 1]
    Rmax, Rmin = float(R.max()), float(R.min())
    Zmax, Zmin = float(Z.max()), float(Z.min())
    width = Rmax - Rmin
    if width <= 0:
        return dict(kappa=np.nan, tri_top=np.nan, tri_bot=np.nan, volume=np.nan)
    a = 0.5 * width
    Rgeo = 0.5 * (Rmax + Rmin)
    R_at_Zmax = float(R[int(np.argmax(Z))])   # R of the highest point
    R_at_Zmin = float(R[int(np.argmin(Z))])   # R of the lowest point
    kappa = (Zmax - Zmin) / width
    tri_top = (Rgeo - R_at_Zmax) / a
    tri_bot = (Rgeo - R_at_Zmin) / a

    # Toroidal volume V = 2*pi * integral_A R dR dZ, via the polygon first-moment about Z:
    #   M = (1/6) sum (R_i + R_{i+1})(R_i Z_{i+1} - R_{i+1} Z_i);   V = 2*pi*|M|.
    R2 = np.append(R, R[0])
    Z2 = np.append(Z, Z[0])
    cross = R2[:-1] * Z2[1:] - R2[1:] * Z2[:-1]
    M = (1.0 / 6.0) * np.sum((R2[:-1] + R2[1:]) * cross)
    volume = float(2.0 * np.pi * abs(M))
    return dict(kappa=float(kappa), tri_top=float(tri_top), tri_bot=float(tri_bot), volume=volume)


def _points_in_polygon(rq, zq, C):
    """Even-odd ray-cast membership of points (rq, zq) in the closed polygon C (N,2)."""
    r1 = C[:, 0]; z1 = C[:, 1]
    r2 = np.roll(r1, -1); z2 = np.roll(z1, -1)
    inside = np.zeros(rq.shape, dtype=bool)
    for e in range(len(r1)):
        za, zb = z1[e], z2[e]
        if za == zb:
            continue
        cross = (za > zq) != (zb > zq)
        if not cross.any():
            continue
        xint = r1[e] + (zq - za) * (r2[e] - r1[e]) / (zb - za)
        inside ^= cross & (rq < xint)
    return inside


def _poloidal_field(psi, R, Z, machine):
    """Bp = |grad psi| / R on the grid (units follow psi; constant factors cancel in the metric)."""
    p = _phi(psi, machine)   # sign flip does not change |grad|; keeps nan handling in one place
    dpsi_dZ, dpsi_dR = np.gradient(p, Z, R)          # axis0 = Z, axis1 = R
    Rg = np.broadcast_to(np.asarray(R, float)[None, :], p.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        Bp = np.sqrt(dpsi_dR ** 2 + dpsi_dZ ** 2) / Rg
    return Bp


def _bilinear(field, R, Z, rq, zq):
    """Bilinear sample of field[iz,ir] at physical query points (rq, zq)."""
    nZ, nR = field.shape
    fr = np.interp(rq, R, np.arange(nR))
    fz = np.interp(zq, Z, np.arange(nZ))
    r0 = np.clip(np.floor(fr).astype(int), 0, nR - 2)
    z0 = np.clip(np.floor(fz).astype(int), 0, nZ - 2)
    tr = fr - r0
    tz = fz - z0
    f00 = field[z0, r0]; f01 = field[z0, r0 + 1]
    f10 = field[z0 + 1, r0]; f11 = field[z0 + 1, r0 + 1]
    return (f00 * (1 - tr) * (1 - tz) + f01 * tr * (1 - tz)
            + f10 * (1 - tr) * tz + f11 * tr * tz)


def internal_inductance(psi, R, Z, machine, C):
    """Scored li = ITER li(2) = <Bp^2>_V / (mu0 Ip / lp)^2, all pieces functionals of psi.

    Omega = grid cells inside the LCFS polygon C; dV = 2*pi R dR dZ;
    mu0 Ip = loop-integral of Bp dl around C (Ampere); lp = poloidal perimeter of C.
    li(2) is the normalisation EFIT's `LI` reports (a-file: "li with normalization average
    poloidal magnetic defined through Ampere's law"); derived-vs-label R2 = 0.97 (DIII-D).
    """
    if C is None or len(C) < 8:
        return np.nan
    Rv = np.asarray(R, float)
    Zv = np.asarray(Z, float)
    Bp = _poloidal_field(psi, R, Z, machine)
    RR, ZZ = np.meshgrid(Rv, Zv)
    inside = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
    if inside.sum() < 4:
        return np.nan
    dR = float(np.mean(np.diff(Rv)))
    dZ = float(np.mean(np.diff(Zv)))
    dV = 2.0 * np.pi * RR * dR * dZ
    Bp2 = np.where(np.isfinite(Bp), Bp, 0.0) ** 2
    num = float(np.sum(Bp2[inside] * dV[inside]))     # integral Bp^2 dV
    V = float(np.sum(dV[inside]))
    if V <= 0:
        return np.nan
    Bp2_V = num / V                                   # <Bp^2>_V

    Bp_on_C = _bilinear(Bp, Rv, Zv, C[:, 0], C[:, 1])
    Cc = np.vstack([C, C[0]])
    seg = np.sqrt(np.sum(np.diff(Cc, axis=0) ** 2, axis=1))
    Bp_mid = 0.5 * (Bp_on_C + np.roll(Bp_on_C, -1))
    G = float(np.sum(Bp_mid * seg))                   # mu0 Ip
    lp = float(np.sum(seg))                           # poloidal perimeter
    if G <= 0 or lp <= 0:
        return np.nan
    return float(Bp2_V / (G / lp) ** 2)


# --------------------------------------------------------------------------------------------
# top-level: derive all seven scalars for one frame
# --------------------------------------------------------------------------------------------
def derive_frame(psi, R, Z, machine, mask_coarse=None, mask_f=None,
                 contour="auto", n_points=512, with_li=True, axis_sign=None):
    """Return {name: value} for the seven psi-derived scalars of one (nZ,nR) frame.

    `contour` is the LCFS polygon to use: "auto" extracts one here (ref-build path); an (N,2)
    array reuses a contour the caller already extracted (the scoring worker shares the D_LCFS
    extraction); None means extraction is known to have failed -- the contour-based scalars
    become nan (a penalised failure) without re-attempting it. Values are nan wherever the
    derivation fails; `with_li` skips the one field-based scalar for cheap passes.

    `axis_sign` overrides the flux orientation used to locate the magnetic axis, for scoring a
    submission whose global psi sign differs from the machine's reference convention. It is the
    ONLY sign-sensitive derivation here: kappa/tri_top/tri_bot/volume come from the contour and
    li from |grad psi|, all of which are invariant under psi -> -psi (measured: error exactly 0
    over 436 frames on both machines). None keeps the reference convention -- the reference
    builder always leaves it None so ground-truth targets never depend on prediction-side logic.
    """
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)
    out = {k: np.nan for k in SCALAR_NAMES}

    R_axis, Z_axis, iz, ir = magnetic_axis(psi, R, Z, machine, mask_coarse, axis_sign)
    out["R_axis"], out["Z_axis"] = R_axis, Z_axis

    if isinstance(contour, str) and contour == "auto":
        C = extract_lcfs(psi, R, Z, machine, mask_coarse, mask_f, n_points=n_points)
    else:
        C = contour
    if C is not None and len(C) >= 8:
        out.update(_shape_from_contour(C))
        if with_li:
            out["li"] = internal_inductance(psi, R, Z, machine, C)
    return _clamp_to_nan(out)
