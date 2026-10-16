"""Official per-shot reference and residual calculations used by FEC1."""
from __future__ import annotations
import numpy as np
from fec1.scoring.vendor.common import (
    AXIS_SIGN,
    CONS_SCALARS,
    N_CONS,
    N_SCALARS,
    PSI_SIGNS,
    SCALARS,
)
from fec1.scoring.vendor.contour import symmetric_hausdorff
from fec1.scoring.vendor.derive import derive_frame
from fec1.scoring.vendor.lcfs import extract_lcfs, extract_lcfs_with_sign, major_radius


MACHINE = "DIII-D"  # the only machine with released ground truth


N_POINTS = 512


N_ITER = 22


LI_IDX = CONS_SCALARS.index("li")


def build_reference(psi_gt, R, Z, mask_coarse, mask_f):
    """The targets the scorer compares against: per-frame LCFS, the seven derived scalars, rgeo.

    This is what the organizers precompute into the reference bundle. Running the same
    `extract_lcfs` / `derive_frame` on the ground truth is what makes a perfect prediction score
    exactly 1 — both sides go through identical code.
    """
    T = psi_gt.shape[0]
    contours, rgeos = [], []
    cons = np.full((T, N_CONS), np.nan)
    for k in range(T):
        c = extract_lcfs(psi_gt[k], R, Z, MACHINE, mask_coarse, mask_f, n_points=N_POINTS)
        contours.append(c)
        if c is not None:
            rgeos.append(major_radius(c))
        vals = derive_frame(psi_gt[k], R, Z, MACHINE, mask_coarse, mask_f, contour=c)
        for j, name in enumerate(CONS_SCALARS):
            cons[k, j] = vals[name]
    # derive_frame already NaNs out physically implausible values, so finiteness is the mask.
    return contours, cons, np.isfinite(cons), (float(np.mean(rgeos)) if rgeos else float("nan"))


def psi_residuals(psi_gt, pred, mean_psi):
    """SS_res under both candidate global flux signs (see metrics.py on sign invariance)."""
    g = psi_gt.astype(np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if p.shape != g.shape:
        r = g - mean_psi
        ss = float(np.sum(r * r))
        return {s: ss for s in PSI_SIGNS}, 0, True
    nf = ~np.isfinite(p)
    out = {}
    for s in PSI_SIGNS:
        q = np.where(nf, mean_psi, s * p) if nf.any() else s * p
        r = g - q
        out[s] = float(np.sum(r * r))
    return out, int(nf.sum()), False


def score_shot(gt, ref, pred, R, Z, mask_coarse, mask_f, psi_sign, means):
    """One shot's contribution. Mirrors the platform's per-shot worker."""
    mean_psi, mean_scal, mean_cons = means
    contours, cons_gt, cmask, rgeo = ref
    psi_gt = gt["psi"]
    T = psi_gt.shape[0]
    # Two different signs compose: submission -> stored convention (psi_sign), and stored
    # convention -> "axis is a maximum of phi" (AXIS_SIGN), which the extractors require.
    axis_sign = AXIS_SIGN[MACHINE] * psi_sign

    ss_res_psi, nonfinite, missing = psi_residuals(psi_gt, pred["psirz"], mean_psi)
    ss_tot_shot = float(np.sum((psi_gt.astype(np.float64) - mean_psi) ** 2))
    shot_r2 = {
        s: (1.0 - ss_res_psi[s] / ss_tot_shot if ss_tot_shot > 0 else 0.0) for s in PSI_SIGNS
    }

    ss_res_scal = np.zeros(N_SCALARS)
    for j, name in enumerate(SCALARS):
        g = np.asarray(gt[name], dtype=np.float64)
        defined = np.isfinite(g)
        if not defined.any():
            continue
        p = np.asarray(pred.get(name), dtype=np.float64) if pred.get(name) is not None else None
        if p is None or p.shape != (T,):
            ss_res_scal[j] = float(np.sum((g[defined] - mean_scal[j]) ** 2))
        else:
            pj = np.where(np.isfinite(p[defined]), p[defined], mean_scal[j])
            ss_res_scal[j] = float(np.sum((g[defined] - pj) ** 2))

    ss_res_cons = np.zeros(N_CONS)
    cons_frames = np.zeros(N_CONS, dtype=np.int64)
    cons_fails = np.zeros(N_CONS, dtype=np.int64)
    ds, n_fail, n_frames = [], 0, 0
    ref_dropped = sum(1 for c in contours if c is None)

    if missing:
        n_frames = T - ref_dropped
        n_fail = n_frames
        ds = [1.0] * n_frames
        for j in range(N_CONS):
            m = cmask[:, j]
            if m.any():
                ss_res_cons[j] = float(np.sum((cons_gt[m, j] - mean_cons[j]) ** 2))
                cons_frames[j] = cons_fails[j] = int(m.sum())
    else:
        psi_pred = np.asarray(pred["psirz"], dtype=np.float64)
        for k in range(T):
            ref_c, need = contours[k], cmask[k]
            if ref_c is None and not need.any():
                continue
            ex, _ = extract_lcfs_with_sign(
                psi_pred[k], R, Z, MACHINE, mask_coarse, mask_f, n_iter=N_ITER, axis_sign=axis_sign
            )
            if ref_c is not None:
                n_frames += 1
                if ex is None:
                    n_fail += 1
                    ds.append(1.0)
                else:
                    d = symmetric_hausdorff(ex, np.asarray(ref_c, dtype=np.float64))
                    ds.append(min(1.0, d / rgeo if rgeo > 0 else 1.0))
            if need.any():
                vals = derive_frame(
                    psi_pred[k],
                    R,
                    Z,
                    MACHINE,
                    mask_coarse,
                    mask_f,
                    contour=ex,
                    with_li=bool(need[LI_IDX]),
                    axis_sign=axis_sign,
                )
                for j in np.nonzero(need)[0]:
                    v = vals[CONS_SCALARS[j]]
                    cons_frames[j] += 1
                    if not np.isfinite(v):
                        v = mean_cons[j]
                        cons_fails[j] += 1
                    dv = cons_gt[k, j] - v
                    ss_res_cons[j] += dv * dv

    return {
        "ss_res_psi": ss_res_psi,
        "psi_pixels": int(psi_gt.size),
        "psi_nonfinite": nonfinite,
        "ss_res_scal": ss_res_scal,
        "ss_res_cons": ss_res_cons,
        "cons_frames": cons_frames,
        "cons_fails": cons_fails,
        "shot_r2_psi": shot_r2,
        "shot_dlcfs": (float(np.mean(ds)) if ds else None),
        "lcfs_frames": n_frames,
        "lcfs_fails": n_fail,
        "ref_lcfs_dropped": ref_dropped,
        "missing": missing,
    }
