#!/usr/bin/env python3
"""Replace the g003 field branch at frozen weights and compare the entire stack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from my_experiments import fec1_g003_reference as reference
from my_experiments.fec1_frozen_baseline import BASELINE_PREDICTION, verify_frozen_baseline_prediction_if_selected
from my_experiments.crossvalidate_fec1_archive_blend import _blend
from my_experiments.evaluate_fec1_geometry_pair import score_contributions, paired_bootstrap
from my_experiments.fit_fec1_full_frame_pca import CACHE, _read, _split_records
from my_experiments.score_fec1_prediction_archive import _parallel_official_score


def aligned_rows(wanted_names, wanted_frames, available_names, available_frames):
    wanted = list(zip(map(str, wanted_names), map(int, wanted_frames)))
    available = list(zip(map(str, available_names), map(int, available_frames)))
    if len(set(wanted)) != len(wanted) or len(set(available)) != len(available):
        raise ValueError("duplicate shot/frame identifiers")
    lookup = {key: i for i, key in enumerate(available)}
    if len(wanted) != len(available) or set(wanted) != set(available):
        raise ValueError("prediction frame populations differ")
    return np.asarray([lookup[key] for key in wanted], dtype=np.int64)


def replacement_field(temporal, branch, alpha):
    if float(alpha) != .5:
        raise ValueError("first g003 repair experiment fixes field alpha at 0.5")
    if temporal.shape != branch.shape or not np.isfinite(branch).all():
        raise ValueError("invalid replacement field")
    return _blend(temporal, branch, alpha)


def build_archive(branch_path, output, training_report):
    reference.verify_report()
    reference.verify_prediction()
    verify_frozen_baseline_prediction_if_selected(BASELINE_PREDICTION)
    if training_report.get("smoke_only") or training_report.get("loss_version") != "symmetric":
        raise RuntimeError("the full comparison requires the real symmetric g003 fit")
    if (training_report.get("development_used") is not False
            or training_report.get("frozen_audit_used") is not False):
        raise RuntimeError("evaluation labels entered candidate training")
    frozen = json.loads(reference.REPORT.read_text())
    alpha = frozen["frozen_final_weights"]["field_alpha_second"]
    with np.load(reference.PREDICTION, allow_pickle=False) as saved:
        metadata = {key: saved[key] for key in ("sample_index", "shot_name", "frame_index", "q95", "betaN")}
        baseline_field = saved["psirz"]
    _, development, _ = _split_records()
    names, frames = metadata["shot_name"].astype(str), metadata["frame_index"]
    if set(names) != set(development):
        raise RuntimeError("comparison must contain exactly the immutable development shots")
    for shot in np.unique(names):
        with np.load(CACHE/f"{shot}.npz", allow_pickle=False) as saved:
            count = len(saved["canonical_global_features"])
        if not np.array_equal(np.sort(frames[names == shot]), np.arange(count)):
            raise RuntimeError("comparison must contain every development frame exactly once")
    with np.load(BASELINE_PREDICTION, allow_pickle=False) as saved:
        rows = aligned_rows(names, frames, saved["shot_name"], saved["frame_index"])
        temporal = saved["psirz"][rows]
    with np.load(reference.BRANCH, allow_pickle=False) as saved:
        rows = aligned_rows(names, frames, saved["shot_name"], saved["frame_index"])
        reconstructed = replacement_field(temporal, saved["psirz"][rows], alpha)
    if not np.array_equal(reconstructed, baseline_field):
        raise RuntimeError("the archived g003 ensemble is not reproduced by its frozen recipe")
    del reconstructed
    with np.load(branch_path, allow_pickle=False) as saved:
        if (saved["frozen_audit_used"].item() or saved["development_used_for_fit"].item()
                or saved["data_policy"].item() != "real_diii_d_only"):
            raise RuntimeError("unsafe candidate prediction provenance")
        rows = aligned_rows(names, frames, saved["shot_name"], saved["frame_index"])
        candidate_field = replacement_field(temporal, saved["psirz"][rows], alpha)
    del temporal
    np.savez_compressed(output/"development_predictions.npz", **metadata, psirz=candidate_field,
                        data_policy=np.asarray("real_diii_d_only"), frozen_audit_used=np.asarray(False),
                        development_used_for_fit=np.asarray(False), field_alpha=np.asarray(alpha),
                        prediction_variant=np.asarray("g003_symmetric_field_replacement_fixed_scalars"))
    # No target labels are read until the final complete-stack predictions exist.
    return metadata, baseline_field, candidate_field


def compare_details(first, second, first_score, second_score):
    for key in first:
        if key == "shot_name" or key.startswith("ref_"):
            if not np.array_equal(first[key], second[key]):
                raise RuntimeError("paired scoring populations differ")
    terms = score_contributions(second)-score_contributions(first)
    delta = terms.sum(1)
    bootstrap = paired_bootstrap(delta)
    if abs(bootstrap["delta_S"]-(second_score["S"]-first_score["S"])) > 2e-6:
        raise RuntimeError("per-shot contributions do not reproduce official score change")
    order = np.argsort(delta)
    return {"bootstrap": bootstrap,
            "delta_by_term": dict(zip(("flux", "scalars", "lcfs", "consistency"), terms.sum(0).tolist())),
            "worst_shots": [{"shot": str(first["shot_name"][i]), "delta_S_contribution": float(delta[i])}
                            for i in order[:20]],
            "best_shots": [{"shot": str(first["shot_name"][i]), "delta_S_contribution": float(delta[i])}
                           for i in order[-20:][::-1]]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branch", type=Path, required=True)
    p.add_argument("--training-report", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    if protocol["field_alpha"] != .5 or not protocol["reuse_exact_g003_scalars"]:
        raise RuntimeError("unexpected frozen comparison recipe")
    training = json.loads(args.training_report.read_text())
    expected = protocol["training_settings"]
    for key, value in expected.items():
        if training["training_arguments"].get(key) != value:
            raise RuntimeError(f"training differs from frozen g003 protocol: {key}")
    from my_experiments.fit_fec1_geometry_aligned_oof_residual import _sha256
    if _sha256(Path(training["artifact"])) != training["artifact_sha256"]:
        raise RuntimeError("candidate artifact changed after training")
    args.output.mkdir(parents=True)
    (args.output/"frozen_recipe.json").write_text(json.dumps({
        "protocol": protocol, "training_report": training,
        "branch": str(args.branch.resolve()), "g003_reference_sha256": reference.PREDICTION_SHA256,
        "development_labels_used_for_new_recipe": False, "historical_development_selection": True,
        "frozen_audit_used": False,
    }, indent=2)+"\n")
    metadata, baseline_field, candidate_field = build_archive(args.branch, args.output, training)
    names, frames = metadata["shot_name"].astype(str), metadata["frame_index"]
    target = np.lib.format.open_memmap(args.output/"target_field.npy", mode="w+", dtype=np.float32,
                                     shape=candidate_field.shape)
    target_scalar = np.empty((len(names), 2), dtype=np.float32)
    for shot in np.unique(names):
        rows = np.flatnonzero(names == shot)
        data = _read(CACHE/f"{shot}.npz", include_targets=True)
        target[rows] = data["target_psi"][frames[rows]]
        target_scalar[rows] = np.column_stack((data["target_q95"][frames[rows]], data["target_betaN"][frames[rows]]))
    target.flush()
    values = {"field": baseline_field, "q95": metadata["q95"], "betaN": metadata["betaN"],
              "target_field": target, "target_q95": target_scalar[:, 0], "target_betaN": target_scalar[:, 1],
              "shot_name": names, "source_index": frames,
              "shot_index": np.unique(names, return_inverse=True)[1], "reference_source": str(CACHE.resolve())}
    cache = ROOT/"my_experiments/artifacts/fec1_development_all_frame_official_reference.pkl"
    first = _parallel_official_score(values, args.workers, cache, args.output/"baseline_contributions.npz")
    archived = json.loads(reference.REPORT.read_text())["full_development_score_after_freezing"]
    for key in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency"):
        if abs(first[key]-archived[key]) > 2e-6:
            raise RuntimeError(f"frozen g003 score did not reproduce: {key}")
    values["field"] = candidate_field
    second = _parallel_official_score(values, args.workers, cache, args.output/"candidate_contributions.npz")
    with np.load(args.output/"baseline_contributions.npz") as a, np.load(args.output/"candidate_contributions.npz") as b:
        paired = compare_details(dict(a), dict(b), first, second)
    checks = {"improves_S": second["S"] > first["S"],
              "preserves_flux": second["r2_psi"] >= first["r2_psi"],
              "improves_lcfs": second["dlcfs"] < first["dlcfs"],
              "improves_consistency": second["consistency"] > first["consistency"],
              "positive_paired_interval": paired["bootstrap"]["interval_95"][0] > 0,
              "unchanged_scalar_score": first["r2_qb"] == second["r2_qb"]}
    result = {"baseline_score": first, "candidate_score": second, "paired": paired, "checks": checks,
              "improves_complete_stack": all(checks.values()), "reaches_0996_on_development": second["S"] >= .996,
              "shots": len(np.unique(names)), "frames": len(names), "field_alpha": .5,
              "exact_baseline_scalar_predictions_reused": True, "historical_development_selection": True,
              "frozen_audit_used": False, "published": False,
              "next_step": "validate_frozen_recipe_with_nested_training_folds_before_final_audit" if all(checks.values())
                           else "retain_g003_and_use_boundary_profile_recoverability_diagnostics"}
    (args.output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
