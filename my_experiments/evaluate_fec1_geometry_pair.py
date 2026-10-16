#!/usr/bin/env python3
"""Compare controlled same-fold FEC1 corrections with fixed scalar predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def score_contributions(detail):
    def total(prefix):
        return detail[f"ref_{prefix}_sumsq"]-detail[f"ref_{prefix}_sum"]**2/np.maximum(detail[f"ref_{prefix}_n"], 1)
    count = len(detail["shot_name"]) if "shot_name" in detail else len(detail["ss_res_psi"])
    psi_total = float(total("psi"))
    psi = (.55/count-.55*detail["ss_res_psi"]/max(psi_total, 1e-30))
    if psi_total <= 0 or detail["ss_res_psi"].sum() >= psi_total:
        psi = np.zeros(count)
    scalar = .15/count-.15*np.mean(detail["ss_res_scal"]/np.maximum(total("scal"), 1e-30), axis=1)
    if scalar.sum() <= 0:
        scalar = np.zeros(count)
    valid = total("cons") > 0
    ratios = detail["ss_res_cons"][:, valid]/total("cons")[valid]
    active = ratios.sum(0) < 1
    consistency = .20*np.mean((1/count-ratios)*active[None], axis=1)
    lcfs = detail["shot_dlcfs"]
    boundary = .1*np.where(np.isfinite(lcfs), 1-lcfs, 0.)/max(np.isfinite(lcfs).sum(), 1)
    return np.column_stack((psi, scalar, boundary, consistency))


def paired_bootstrap(difference, samples=4000, seed=260906):
    rng = np.random.default_rng(seed)
    scores = np.empty(samples)
    for start in range(0, samples, 100):
        indices = rng.integers(0, len(difference), (min(100, samples-start), len(difference)))
        scores[start:start+len(indices)] = difference[indices].sum(axis=1)
    return {"delta_S": float(difference.sum()), "interval_95": np.quantile(scores, [.025, .975]).tolist(),
            "resamples": samples, "method": "paired shot bootstrap with original pooled normalizations, clipping masks, and failure-substitution means fixed"}


def compare(first, second):
    reports = [json.loads((directory/"result.json").read_text()) for directory in (first, second)]
    a, b = reports
    for key in ("fold", "decoder", "scalar_reference_sha256"):
        if a.get(key) != b.get(key):
            raise ValueError(f"unmatched comparison: {key}")
    if not a.get("scalar_reference_sha256"):
        raise ValueError("comparison requires the same frozen scalar path")
    for key in ("data", "epochs", "batch_size", "width", "channels", "seed", "geometry_weight",
                "geometry_batch", "geometry_warmup", "selection_shots", "exact_every", "patience_checks"):
        if a["training_arguments"].get(key) != b["training_arguments"].get(key):
            raise ValueError(f"unmatched training setting: {key}")
    for report in reports:
        if report["development_used"] or report["frozen_audit_used"] or not report["nested_shot_crossfit"]:
            raise ValueError("unsafe evaluation provenance")
    with np.load(first/"outer_predictions.npz") as pa, np.load(second/"outer_predictions.npz") as pb:
        for key in ("shot_name", "frame_index", "source_row", "q95", "betaN"):
            if not np.array_equal(pa[key], pb[key]):
                raise ValueError(f"population or frozen scalars differ: {key}")
        row_digest = hashlib.sha256(pa["shot_name"].tobytes()+pa["frame_index"].tobytes()).hexdigest()
    with np.load(first/"outer_contributions.npz") as da, np.load(second/"outer_contributions.npz") as db:
        da, db = dict(da), dict(db)
    if not np.array_equal(da["shot_name"], db["shot_name"]):
        raise ValueError("shot contributions are misaligned")
    for key in da:
        if key.startswith("ref_") and not np.array_equal(da[key], db[key]):
            raise ValueError("official reference populations differ")
    difference_terms = score_contributions(db)-score_contributions(da)
    difference = difference_terms.sum(1)
    sa, sb = a["outer_score"], b["outer_score"]
    bootstrap = paired_bootstrap(difference)
    if abs(bootstrap["delta_S"]-(sb["S"]-sa["S"])) > 2e-6:
        raise ValueError("per-shot score deltas do not reproduce official score change")
    order = np.argsort(difference)
    def shot_rows(indices):
        return [{"shot": str(da["shot_name"][i]), "delta_S_contribution": float(difference[i]),
                 "delta_by_term": dict(zip(("flux", "scalars", "lcfs", "consistency"), difference_terms[i].tolist()))}
                for i in indices]
    checks = {"improves_S": sb["S"] > sa["S"],
              "preserves_flux": sb["r2_psi"] >= sa["r2_psi"],
              "improves_lcfs": sb["dlcfs"] < sa["dlcfs"],
              "improves_consistency": sb["consistency"] > sa["consistency"],
              "improves_kappa": sb["r2_cons_each"]["kappa"] > sa["r2_cons_each"]["kappa"],
              "improves_li": sb["r2_cons_each"]["li"] > sa["r2_cons_each"]["li"],
              "positive_bootstrap_lower_bound": bootstrap["interval_95"][0] > 0}
    return {"first": str(first.resolve()), "second": str(second.resolve()),
            "first_score": sa, "second_score": sb, "row_digest": row_digest,
            "bootstrap": bootstrap, "checks": checks, "decisive_improvement": all(checks.values()),
            "worst_shot_changes": shot_rows(order[:20]), "best_shot_changes": shot_rows(order[-20:][::-1]),
            "geometry_component_deltas": {k: sb["r2_cons_each"][k]-v for k, v in sa["r2_cons_each"].items()
                                           if v is not None and sb["r2_cons_each"][k] is not None},
            "scalar_reference_sha256": a["scalar_reference_sha256"],
            "frame_stride": 4, "nested_shot_crossfit": True,
            "development_used": False, "frozen_audit_used": False,
            "smoke_only": a["smoke_only"] or b["smoke_only"],
            "next_step": "boundary_profile_ablation" if all(checks.values()) else "inspect_errors_and_test_recoverability"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first", type=Path, required=True)
    p.add_argument("--second", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.first, args.second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
