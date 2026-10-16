#!/usr/bin/env python3
"""Select a two-model FEC1 blend with shot-held-out development predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_experiments.fit_fec1_full_frame_pca import CACHE, _read  # noqa: E402
from my_experiments.fec1_frozen_baseline import (  # noqa: E402
    verify_frozen_baseline_prediction_if_selected,
)
from my_experiments.score_fec1_prediction_archive import (  # noqa: E402
    _parallel_official_score,
)


KEYS = ("sample_index", "shot_name", "frame_index", "psirz", "q95", "betaN")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        if "frozen_audit_used" in saved.files and bool(saved["frozen_audit_used"].item()):
            raise RuntimeError(f"{path} used frozen-audit data")
        values = {key: np.asarray(saved[key]) for key in KEYS}
    order = np.argsort(values["sample_index"].astype(np.int64))
    return {key: value[order] for key, value in values.items()}


def _optimal_alpha(
    first: np.ndarray, second: np.ndarray, target: np.ndarray,
) -> float:
    delta = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    residual = np.asarray(target, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    finite = np.isfinite(delta) & np.isfinite(residual)
    denominator = np.square(delta[finite]).sum(dtype=np.float64)
    if denominator <= 1.0e-30:
        return 0.0
    return float(np.clip(
        np.sum(delta[finite] * residual[finite], dtype=np.float64) / denominator,
        0.0, 1.0,
    ))


def _blend(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    return (
        np.asarray(first, dtype=np.float32)
        + np.float32(alpha) * (
            np.asarray(second, dtype=np.float32) - np.asarray(first, dtype=np.float32)
        )
    ).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--field-alphas", default="0,.05,.10,.15,.20,.30,.50,1")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--calibration-shots", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=260902)
    parser.add_argument("--full-reference-cache", type=Path)
    parser.add_argument("--prediction-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    alphas = [float(value) for value in args.field_alphas.split(",") if value.strip()]
    if args.folds < 2 or not alphas or any(value < 0.0 or value > 1.0 for value in alphas):
        parser.error("folds must be >=2 and field alphas must lie in [0,1]")
    verify_frozen_baseline_prediction_if_selected(args.first)
    started = time.monotonic()
    first = _load(args.first)
    second = _load(args.second)
    for key in ("sample_index", "shot_name", "frame_index"):
        left = first[key].astype(str) if key == "shot_name" else first[key]
        right = second[key].astype(str) if key == "shot_name" else second[key]
        if not np.array_equal(left, right):
            raise RuntimeError(f"prediction archives differ on {key}")
    names = first["shot_name"].astype(str)
    frames = first["frame_index"].astype(np.int64)
    shots = np.unique(names)

    target_field = np.empty_like(first["psirz"], dtype=np.float32)
    target_q95 = np.empty(len(names), dtype=np.float32)
    target_beta_n = np.empty(len(names), dtype=np.float32)
    shot_index = np.empty(len(names), dtype=np.int64)
    for index, shot in enumerate(shots):
        rows = np.flatnonzero(names == shot)
        data = _read(args.cache_dir / f"{shot}.npz", include_targets=True)
        target_field[rows] = data["target_psi"][frames[rows]].astype(np.float32)
        target_q95[rows] = data["target_q95"][frames[rows]].astype(np.float32)
        target_beta_n[rows] = data["target_betaN"][frames[rows]].astype(np.float32)
        shot_index[rows] = index
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(shots):
            print(f"loaded blend targets {index + 1}/{len(shots)} shots", flush=True)

    def score(
        rows: np.ndarray, field_alpha: float, q_alpha: float, beta_alpha: float,
        reference_cache: Path | None,
    ) -> dict[str, object]:
        local_names = names[rows]
        local_groups = np.empty(len(rows), dtype=np.int64)
        for group, shot in enumerate(np.unique(local_names)):
            local_groups[local_names == shot] = group
        values = {
            "field": _blend(first["psirz"][rows], second["psirz"][rows], field_alpha),
            "target_field": target_field[rows],
            "q95": _blend(first["q95"][rows], second["q95"][rows], q_alpha),
            "target_q95": target_q95[rows],
            "betaN": _blend(first["betaN"][rows], second["betaN"][rows], beta_alpha),
            "target_betaN": target_beta_n[rows],
            "shot_index": local_groups,
            "source_index": frames[rows],
            "shot_name": first["shot_name"][rows],
            "reference_source": str(args.cache_dir.resolve()),
        }
        return _parallel_official_score(values, max(args.workers, 1), reference_cache)

    shuffled = np.random.default_rng(args.seed).permutation(shots)
    folds = [part.astype(str) for part in np.array_split(shuffled, args.folds)]
    crossfit_field = np.empty_like(first["psirz"], dtype=np.float32)
    crossfit_q = np.empty(len(names), dtype=np.float32)
    crossfit_beta = np.empty(len(names), dtype=np.float32)
    fold_reports = []
    selected_field, selected_q, selected_beta = [], [], []
    for held_fold, held_shots in enumerate(folds):
        held = np.isin(names, held_shots)
        calibration_pool = shots[~np.isin(shots, held_shots)]
        calibration_selected = np.random.default_rng(
            args.seed + 1000 + held_fold
        ).permutation(calibration_pool)[: args.calibration_shots]
        calibration = np.isin(names, calibration_selected)
        calibration_rows = np.flatnonzero(calibration)
        held_rows = np.flatnonzero(held)
        q_alpha = _optimal_alpha(
            first["q95"][calibration_rows], second["q95"][calibration_rows],
            target_q95[calibration_rows],
        )
        beta_alpha = _optimal_alpha(
            first["betaN"][calibration_rows], second["betaN"][calibration_rows],
            target_beta_n[calibration_rows],
        )
        cache = args.output.with_name(
            f"{args.output.stem}.calibration_fold{held_fold}.reference.pkl"
        )
        trials = []
        for field_alpha in alphas:
            result = score(
                calibration_rows, field_alpha, q_alpha, beta_alpha, cache,
            )
            trials.append({"field_alpha": field_alpha, "score": result})
            print(json.dumps({
                "held_fold": held_fold,
                "field_alpha": field_alpha,
                "calibration_S": result["S"],
            }), flush=True)
        best = max(trials, key=lambda item: float(item["score"]["S"]))
        field_alpha = float(best["field_alpha"])
        held_score = score(held_rows, field_alpha, q_alpha, beta_alpha, None)
        crossfit_field[held_rows] = _blend(
            first["psirz"][held_rows], second["psirz"][held_rows], field_alpha,
        )
        crossfit_q[held_rows] = _blend(
            first["q95"][held_rows], second["q95"][held_rows], q_alpha,
        )
        crossfit_beta[held_rows] = _blend(
            first["betaN"][held_rows], second["betaN"][held_rows], beta_alpha,
        )
        selected_field.append(field_alpha)
        selected_q.append(q_alpha)
        selected_beta.append(beta_alpha)
        fold_reports.append({
            "held_fold": held_fold,
            "calibration_shots": int(len(calibration_selected)),
            "held_shots": int(len(held_shots)),
            "selected": {
                "field_alpha_second": field_alpha,
                "q95_alpha_second": q_alpha,
                "betaN_alpha_second": beta_alpha,
            },
            "calibration_trials": trials,
            "held_out_score": held_score,
        })
        print(json.dumps({
            "held_fold": held_fold,
            "selected_field_alpha": field_alpha,
            "held_out_S": held_score["S"],
        }), flush=True)

    all_rows = np.arange(len(names), dtype=np.int64)
    crossfit_values = {
        "field": crossfit_field,
        "target_field": target_field,
        "q95": crossfit_q,
        "target_q95": target_q95,
        "betaN": crossfit_beta,
        "target_betaN": target_beta_n,
        "shot_index": shot_index,
        "source_index": frames,
        "shot_name": first["shot_name"],
        "reference_source": str(args.cache_dir.resolve()),
    }
    crossfit_score = _parallel_official_score(
        crossfit_values, max(args.workers, 1), args.full_reference_cache,
    )
    final_field_alpha = float(np.median(selected_field))
    final_q_alpha = float(np.mean(selected_q))
    final_beta_alpha = float(np.mean(selected_beta))
    final_score = score(
        all_rows, final_field_alpha, final_q_alpha, final_beta_alpha,
        args.full_reference_cache,
    )

    if args.prediction_output is not None:
        args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.prediction_output,
            sample_index=first["sample_index"].astype(np.int64),
            shot_name=first["shot_name"].astype("U19"),
            frame_index=first["frame_index"].astype(np.int32),
            psirz=_blend(first["psirz"], second["psirz"], final_field_alpha),
            q95=_blend(first["q95"], second["q95"], final_q_alpha),
            betaN=_blend(first["betaN"], second["betaN"], final_beta_alpha),
            data_policy=np.asarray("real_diii_d_only"),
            mast_data_used=np.asarray(False),
            frozen_audit_used=np.asarray(False),
            prediction_variant=np.asarray("shot_crossvalidated_two_archive_blend"),
        )

    report = {
        "first": str(args.first.resolve()),
        "second": str(args.second.resolve()),
        "second_weight_definition": "prediction=(1-alpha)*first+alpha*second",
        "folds": args.folds,
        "field_alpha_grid": alphas,
        "fold_reports": fold_reports,
        "shot_crossfit_score": crossfit_score,
        "frozen_final_weights": {
            "field_alpha_second": final_field_alpha,
            "q95_alpha_second": final_q_alpha,
            "betaN_alpha_second": final_beta_alpha,
        },
        "full_development_score_after_freezing": final_score,
        "prediction_output": (
            str(args.prediction_output.resolve())
            if args.prediction_output is not None else None
        ),
        "data_policy": "real_diii_d_only",
        "frozen_audit_used": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
