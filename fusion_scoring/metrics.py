"""Metric reduction: pooled R2 for the flux map, the two value scalars, and the eight
consistency scalars, plus the composite S (metric v2).

All R2 here are *pooled* (a single ratio-of-sums over every element of a machine's test set),
not per-shot averages. To make that streamable, the reference bundle precomputes the total sum
of squares from ground truth alone (SS_tot) -- for the value scalars over the stored EFIT
labels, for the consistency scalars over f_j(psi_gt) -- and each shot worker only accumulates
the residual sum of squares SS_res from its own predictions; the parent forms
R2 = 1 - SS_res / SS_tot. A perfect psi therefore scores R2_psi = 1, D_LCFS = 0 and every
R2_cons_j = 1 by construction.

Sign invariance (v3.1): the fold's global flux sign is chosen ONCE, by the probe pass in
parallel.score_config, and imposed on every shot before any of this runs -- so `finalize_machine`
reads it off the accumulator rather than choosing it. Workers still report SS_res under both
signs, but only so the report can show what the submission would have scored unflipped. Being a
single fold-global bit is what makes it ungameable: a submission cannot pick a favourable sign
per shot, and one bit chosen over ~10^8 pixels carries no exploitable noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common import (CH2_GATE_D3D_S, CONS_SCALARS, N_CONS, N_SCALARS,
                    PSI_SIGNS, SCALARS, W_CONS, W_LCFS, W_PSI, W_QB)


def ss_tot(sum_: float, sumsq: float, n: float) -> float:
    """Total sum of squares about the pooled mean:  Σy² - (Σy)²/n."""
    if n <= 0:
        return 0.0
    return float(sumsq - (sum_ * sum_) / n)


@dataclass
class Accum:
    """Machine-level accumulator; shot workers contribute into it, parent finalizes."""
    # SS_res under each candidate global flux sign, summed over the fold. Keeping both and
    # deciding once here -- rather than per shot in the worker -- is what makes the sign a
    # single fold-global bit rather than 1,206 independent coin flips.
    ss_res_psi: dict = field(default_factory=lambda: {s: 0.0 for s in PSI_SIGNS})
    # The fold's global flux sign, decided by the pass-1 probe in parallel.score_config and set
    # there before finalize. Every orientation-sensitive term was already computed under it.
    psi_sign: float = 1.0
    psi_pixels: int = 0
    psi_nonfinite: int = 0
    ss_res_scal: np.ndarray = field(default_factory=lambda: np.zeros(N_SCALARS))
    ss_res_cons: np.ndarray = field(default_factory=lambda: np.zeros(N_CONS))
    cons_frames: np.ndarray = field(default_factory=lambda: np.zeros(N_CONS, dtype=np.int64))
    cons_fails: np.ndarray = field(default_factory=lambda: np.zeros(N_CONS, dtype=np.int64))
    shot_r2_psi: list = field(default_factory=list)   # per-shot diagnostic
    shot_dlcfs: list = field(default_factory=list)     # per-shot mean D_LCFS (valid shots only)
    lcfs_frames: int = 0
    lcfs_fails: int = 0
    ref_lcfs_dropped: int = 0                          # frames with no reference contour at all
    n_shots_scored: int = 0
    n_shots_missing: int = 0

    def add(self, part: dict) -> None:
        for s in PSI_SIGNS:
            self.ss_res_psi[s] += part["ss_res_psi"][s]
        self.psi_pixels += part["psi_pixels"]
        self.psi_nonfinite += part["psi_nonfinite"]
        self.ss_res_scal = self.ss_res_scal + part["ss_res_scal"]
        self.ss_res_cons = self.ss_res_cons + part["ss_res_cons"]
        self.cons_frames = self.cons_frames + part["cons_frames"]
        self.cons_fails = self.cons_fails + part["cons_fails"]
        if part["shot_r2_psi"] is not None:
            self.shot_r2_psi.append(part["shot_r2_psi"])   # {sign: r2}; resolved in finalize
        if part["shot_dlcfs"] is not None:
            self.shot_dlcfs.append(part["shot_dlcfs"])
        self.lcfs_frames += part["lcfs_frames"]
        self.lcfs_fails += part["lcfs_fails"]
        self.ref_lcfs_dropped += part["ref_lcfs_dropped"]
        self.n_shots_scored += 1
        self.n_shots_missing += int(part["missing"])


def finalize_machine(acc: Accum, ref_stats: dict) -> dict:
    """Turn accumulated residuals + precomputed SS_tot into the machine's leaderboard row."""
    tot_psi = ss_tot(ref_stats["psi_sum"], ref_stats["psi_sumsq"], ref_stats["psi_n"])
    # Sign invariance: one bit for the whole fold, fixed by the pass-1 probe. It must be the
    # same bit the LCFS/derivation pass ran under, so it is taken from the accumulator rather
    # than re-minimised here -- re-deciding it now could disagree with what was scored.
    psi_sign = acc.psi_sign
    ss_res_psi = acc.ss_res_psi[psi_sign]
    r2_psi = 1.0 - ss_res_psi / tot_psi if tot_psi > 0 else 0.0
    r2_psi_unflipped = 1.0 - acc.ss_res_psi[1.0] / tot_psi if tot_psi > 0 else 0.0
    # The sign came from a subsample; these sums are the whole fold. They disagree only for a
    # submission with essentially no flux skill, where the two residuals are nearly equal --
    # but if it happens, r2_psi comes out below r2_psi_unflipped, which reads as a scorer bug.
    # Surface it instead. (Not re-decided here: the LCFS/derivation pass already ran under
    # acc.psi_sign, so changing it now would make the reported terms mutually inconsistent.)
    sign_disagrees = acc.ss_res_psi[psi_sign] > acc.ss_res_psi[-psi_sign]

    # value scalars {q95, betaN}: mean pooled R2 vs the stored EFIT labels
    r2_each = np.zeros(N_SCALARS)
    for j in range(N_SCALARS):
        tot_j = ss_tot(ref_stats["scal_sum"][j], ref_stats["scal_sumsq"][j], ref_stats["scal_n"][j])
        r2_each[j] = 1.0 - acc.ss_res_scal[j] / tot_j if tot_j > 0 else 0.0
    r2_qb = float(np.mean(r2_each))

    # consistency scalars: pooled R2 of f_j(psi_pred) vs f_j(psi_gt); a scalar with no valid
    # frames in the whole fold (SS_tot = 0) is excluded from the mean, not scored as 0 or 1.
    r2_cons = np.full(N_CONS, np.nan)
    for j in range(N_CONS):
        tot_j = ss_tot(ref_stats["cons_sum"][j], ref_stats["cons_sumsq"][j], ref_stats["cons_n"][j])
        if tot_j > 0:
            r2_cons[j] = 1.0 - acc.ss_res_cons[j] / tot_j
    scored = np.isfinite(r2_cons)
    consistency = float(np.mean(np.maximum(0.0, r2_cons[scored]))) if scored.any() else 0.0

    d_lcfs = float(np.mean(acc.shot_dlcfs)) if acc.shot_dlcfs else 1.0

    s = (W_PSI * max(0.0, r2_psi)
         + W_QB * max(0.0, r2_qb)
         + W_LCFS * (1.0 - min(1.0, d_lcfs))
         + W_CONS * consistency)

    cons_valid = int(acc.cons_frames.sum())
    per_shot = [d[psi_sign] for d in acc.shot_r2_psi]
    out = {
        "S": round(s, 6),
        "r2_psi": round(r2_psi, 6),
        "r2_qb": round(r2_qb, 6),
        "dlcfs": round(d_lcfs, 6),
        "consistency": round(consistency, 6),
        # The global flux sign the submission was read under, and what R2_psi would have been
        # without the normalization -- so a flip is always visible, never silent.
        "psi_sign": int(psi_sign),
        "psi_sign_flipped": bool(psi_sign < 0),
        "psi_sign_probe_disagrees": bool(sign_disagrees),
        "r2_psi_unflipped": round(r2_psi_unflipped, 6),
        "r2_psi_per_shot_mean": round(float(np.mean(per_shot)), 6) if per_shot else 0.0,
        "lcfs_fail_frac": round(acc.lcfs_fails / acc.lcfs_frames, 6) if acc.lcfs_frames else 0.0,
        "ref_lcfs_dropped": int(acc.ref_lcfs_dropped),
        "psi_nonfinite_frac": round(acc.psi_nonfinite / acc.psi_pixels, 8) if acc.psi_pixels else 0.0,
        "cons_fail_frac": round(float(acc.cons_fails.sum()) / cons_valid, 6) if cons_valid else 0.0,
        "n_shots": acc.n_shots_scored,
        "n_shots_missing": acc.n_shots_missing,
        # per-scalar detail for the report (unclipped R2; nan = not scored on this fold)
        "r2_each": {SCALARS[j]: round(float(r2_each[j]), 6) for j in range(N_SCALARS)},
        "r2_cons_each": {CONS_SCALARS[j]: (round(float(r2_cons[j]), 6)
                                           if np.isfinite(r2_cons[j]) else None)
                         for j in range(N_CONS)},
        "cons_fail_each": {CONS_SCALARS[j]: (round(float(acc.cons_fails[j]) / acc.cons_frames[j], 6)
                                             if acc.cons_frames[j] else 0.0)
                           for j in range(N_CONS)},
    }
    return out


def g_ratio(s_mast: float, s_d3d: float) -> float:
    """Award-2 cross-machine ratio S_MAST / S_D3D; 0 unless DIII-D S clears CH2_GATE_D3D_S."""
    if s_d3d < CH2_GATE_D3D_S:
        return 0.0
    return round(s_mast / s_d3d, 6)
