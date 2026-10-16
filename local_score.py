#!/usr/bin/env python3
"""
Score your model locally, on held-out DIII-D training shots, with the real metric.

`validate_submission.py` checks that your file has the right *shape*. This scores it. Without it
you can compute R²ψ yourself easily enough, but `D_LCFS` and `Consistency` — 30% of the composite
— need contour extraction and the seven ψ-derived functionals, so most people fly blind on the
part of the score where a plausible-looking flux map and a good one come apart.

    S = 0.55·R²ψ  +  0.15·R²{q95,βN}  +  0.10·(1 − D_LCFS)  +  0.20·Consistency

`fusion_scoring/` holds the competition scorer's own modules, copied unmodified, so the
functionals here are byte-identical to the ones Codabench runs. What differs is the *data*: this
scores held-out training shots you provide, the leaderboard scores withheld test shots. Treat the
number as a faithful proxy for model selection, not a leaderboard prediction.

    # sanity-check the harness itself (should print S = 1.0 then S = 0.0)
    uv run python local_score.py --mode perfect --n-shots 3
    uv run python local_score.py --mode zeros   --n-shots 3

    # score your model, on shots it was not trained on
    uv run python local_score.py --n-shots 20 --skip 7000

    # or score a .npz you already built
    uv run python local_score.py --pred my_preds.npz --n-shots 20 --skip 7000

By default it imports `your_model_predict` from submission_skeleton.py — the same function that
builds your real submission. Note the training rows it receives DO contain `efit_*` ground truth:
if your model reads those, this score is meaningless. Use only the inputs listed in that
function's docstring.

Memory: shots are held in RAM (~20 MB per DIII-D shot). Start around --n-shots 10-20.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# The vendored scorer uses flat imports (`from common import ...`) exactly as it does on the
# platform. Keeping the files byte-identical means syncing them is a plain copy, so they cannot
# quietly drift away from what actually scores your submission.
sys.path.insert(0, str(HERE / "fusion_scoring"))

from common import (AXIS_SIGN, CONS_SCALARS, N_CONS, N_SCALARS,  # noqa: E402
                    PSI_SIGNS, SCALARS, SCORING_VERSION)
from contour import symmetric_hausdorff                          # noqa: E402
from derive import derive_frame                                  # noqa: E402
from lcfs import extract_lcfs, extract_lcfs_with_sign, major_radius  # noqa: E402
from metrics import Accum, finalize_machine                      # noqa: E402

REPO_ID = "Sophelio/fusion-equilibrium-challenge"
MACHINE = "DIII-D"          # the only machine with released ground truth
N_POINTS = 512
N_ITER = 22
LI_IDX = CONS_SCALARS.index("li")


# --------------------------------------------------------------------------- ground truth

def load_shots(n_shots: int, skip: int) -> list[dict]:
    """Stream `n_shots` DIII-D training shots, keeping ψ and the two value scalars."""
    from datasets import load_dataset
    ds = load_dataset(REPO_ID, "diii_d_train", split="train", streaming=True)
    out = []
    for i, row in enumerate(ds):
        if i < skip:
            continue
        if len(out) >= n_shots:
            break
        psi = np.asarray(row["efit_psirz"], dtype=np.float32)
        if psi.ndim != 3:                       # nested list -> (T, 65, 65)
            psi = np.stack([np.asarray(f, dtype=np.float32) for f in row["efit_psirz"]])
        out.append({
            "row": row,
            "psi": psi,
            "q95": np.asarray(row["efit_q95"], dtype=np.float64),
            "betaN": np.asarray(row["efit_beta_n"], dtype=np.float64),
        })
        print(f"  loaded shot {i}  T={psi.shape[0]}")
    if not out:
        raise SystemExit(f"No shots loaded (skip={skip} past the end of the split?)")
    return out


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


# --------------------------------------------------------------------------- predictions

def predict(shots: list[dict], mode: str, pred_npz: Path | None) -> list[dict]:
    if pred_npz is not None:
        z = np.load(pred_npz, allow_pickle=False)
        return [{"psirz": z[f"shot_{i:04d}_psirz"].astype(np.float64),
                 "q95": z[f"shot_{i:04d}_q95"].astype(np.float64),
                 "betaN": z[f"shot_{i:04d}_betaN"].astype(np.float64)}
                for i in range(len(shots))]
    if mode == "perfect":       # harness self-check: must score S = 1.0
        return [{"psirz": s["psi"].astype(np.float64), "q95": s["q95"], "betaN": s["betaN"]}
                for s in shots]
    if mode == "zeros":         # harness self-check: must score S = 0.0
        return [{"psirz": np.zeros_like(s["psi"], dtype=np.float64),
                 "q95": np.zeros_like(s["q95"]), "betaN": np.zeros_like(s["betaN"])}
                for s in shots]
    from submission_skeleton import your_model_predict
    return [your_model_predict(s["row"], MACHINE) for s in shots]


# --------------------------------------------------------------------------- scoring

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
    shot_r2 = {s: (1.0 - ss_res_psi[s] / ss_tot_shot if ss_tot_shot > 0 else 0.0)
               for s in PSI_SIGNS}

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
            ex, _ = extract_lcfs_with_sign(psi_pred[k], R, Z, MACHINE, mask_coarse, mask_f,
                                           n_iter=N_ITER, axis_sign=axis_sign)
            if ref_c is not None:
                n_frames += 1
                if ex is None:
                    n_fail += 1
                    ds.append(1.0)
                else:
                    d = symmetric_hausdorff(ex, np.asarray(ref_c, dtype=np.float64))
                    ds.append(min(1.0, d / rgeo if rgeo > 0 else 1.0))
            if need.any():
                vals = derive_frame(psi_pred[k], R, Z, MACHINE, mask_coarse, mask_f, contour=ex,
                                    with_li=bool(need[LI_IDX]), axis_sign=axis_sign)
                for j in np.nonzero(need)[0]:
                    v = vals[CONS_SCALARS[j]]
                    cons_frames[j] += 1
                    if not np.isfinite(v):
                        v = mean_cons[j]
                        cons_fails[j] += 1
                    dv = cons_gt[k, j] - v
                    ss_res_cons[j] += dv * dv

    return {
        "ss_res_psi": ss_res_psi, "psi_pixels": int(psi_gt.size), "psi_nonfinite": nonfinite,
        "ss_res_scal": ss_res_scal, "ss_res_cons": ss_res_cons,
        "cons_frames": cons_frames, "cons_fails": cons_fails,
        "shot_r2_psi": shot_r2, "shot_dlcfs": (float(np.mean(ds)) if ds else None),
        "lcfs_frames": n_frames, "lcfs_fails": n_fail, "ref_lcfs_dropped": ref_dropped,
        "missing": missing,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-shots", type=int, default=10, help="held-out shots to score (default 10)")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip this many train shots first — use it to score shots you did NOT "
                         "train on")
    ap.add_argument("--mode", choices=["model", "perfect", "zeros"], default="model",
                    help="perfect/zeros verify the harness itself (S must be 1.0 / 0.0)")
    ap.add_argument("--pred", type=Path, help="score this .npz instead of calling your model")
    args = ap.parse_args()

    mask = np.load(HERE / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]
    mask_coarse = mask["mask_coarse"].astype(bool)
    mask_f = mask_coarse.astype(np.float64)

    print(f"Fusion Equilibrium Challenge — local scorer (metric v{SCORING_VERSION}, {MACHINE})")
    print(f"Loading {args.n_shots} training shots (skip={args.skip})...")
    shots = load_shots(args.n_shots, args.skip)

    print("Building reference targets from ground truth (LCFS + 7 derived scalars per frame)...")
    refs, psi_sum, psi_sumsq, psi_n = [], 0.0, 0.0, 0.0
    scal_sum, scal_sumsq, scal_n = np.zeros(N_SCALARS), np.zeros(N_SCALARS), np.zeros(N_SCALARS)
    cons_sum, cons_sumsq, cons_n = np.zeros(N_CONS), np.zeros(N_CONS), np.zeros(N_CONS)
    for si, s in enumerate(shots):
        ref = build_reference(s["psi"], R, Z, mask_coarse, mask_f)
        refs.append(ref)
        g = s["psi"].astype(np.float64)
        fin = np.isfinite(g)
        psi_sum += float(g[fin].sum()); psi_sumsq += float((g[fin] ** 2).sum()); psi_n += int(fin.sum())
        for j, name in enumerate(SCALARS):
            v = np.asarray(s[name], dtype=np.float64)
            v = v[np.isfinite(v)]
            scal_sum[j] += v.sum(); scal_sumsq[j] += (v ** 2).sum(); scal_n[j] += v.size
        cons_gt, cmask = ref[1], ref[2]
        for j in range(N_CONS):
            v = cons_gt[cmask[:, j], j]
            cons_sum[j] += v.sum(); cons_sumsq[j] += (v ** 2).sum(); cons_n[j] += v.size
        print(f"  shot {si}: reference built")

    ref_stats = {"psi_sum": psi_sum, "psi_sumsq": psi_sumsq, "psi_n": psi_n,
                 "scal_sum": scal_sum, "scal_sumsq": scal_sumsq, "scal_n": scal_n,
                 "cons_sum": cons_sum, "cons_sumsq": cons_sumsq, "cons_n": cons_n}
    mean_psi = psi_sum / psi_n if psi_n else 0.0
    means = (mean_psi,
             np.where(scal_n > 0, scal_sum / np.where(scal_n > 0, scal_n, 1), 0.0),
             np.where(cons_n > 0, cons_sum / np.where(cons_n > 0, cons_n, 1), 0.0))

    print(f"Generating predictions (mode={args.mode})...")
    preds = predict(shots, args.mode, args.pred)

    # The flux sign is ONE bit for the whole set, not per shot — a submission must not be able to
    # pick a favourable orientation shot by shot. Decide it from the ψ residuals, then impose it
    # on every orientation-sensitive derivation below.
    totals = {s: 0.0 for s in PSI_SIGNS}
    for s_, p in zip(shots, preds):
        rr, _, _ = psi_residuals(s_["psi"], p["psirz"], mean_psi)
        for sg in PSI_SIGNS:
            totals[sg] += rr[sg]
    psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])
    if psi_sign < 0:
        print("  note: your flux is sign-inverted vs the DIII-D convention — normalized for you, "
              "exactly as the leaderboard does")

    print("Scoring...")
    acc = Accum()
    acc.psi_sign = psi_sign
    for si, (s_, ref, p) in enumerate(zip(shots, refs, preds)):
        acc.add(score_shot(s_, ref, p, R, Z, mask_coarse, mask_f, psi_sign, means))
        print(f"  shot {si} scored")

    res = finalize_machine(acc, ref_stats)

    print("\n" + "=" * 62)
    print(f"  COMPOSITE S = {res['S']:.4f}      ({len(shots)} held-out {MACHINE} shots)")
    print("=" * 62)
    for label, key, w in [("R2_psi", "r2_psi", 0.55), ("R2_{q95,betaN}", "r2_qb", 0.15),
                          ("1 - D_LCFS", "dlcfs", 0.10), ("Consistency", "consistency", 0.20)]:
        v = (1.0 - min(1.0, res["dlcfs"])) if key == "dlcfs" else res[key]
        print(f"  {label:>16s}  {v:8.4f}   x {w:.2f}  =  {w * max(0.0, v):.4f}")
    print(f"\n  per-derived-scalar R2 (the Consistency term):")
    for k, v in res["r2_cons_each"].items():
        print(f"      {k:>8s}  {'  n/a' if v is None else f'{v:7.4f}'}")
    print(f"\n  LCFS extraction failed on {res['lcfs_fail_frac']:.1%} of frames; "
          f"derivations on {res['cons_fail_frac']:.1%}")
    print(f"  psi_sign = {res['psi_sign']:+d}")

    if args.mode in ("perfect", "zeros"):
        want = 1.0 if args.mode == "perfect" else 0.0
        ok = abs(res["S"] - want) < 1e-4
        print(f"\n  self-check ({args.mode}): S = {res['S']:.6f}, want {want} -> "
              f"{'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
