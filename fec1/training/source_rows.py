#!/usr/bin/env python3
"""Generate compact shot-OOF FEC1 base predictions on the train split.

The existing full-frame PCA basis is fitted only on immutable training shots.
This script re-fits the supervised regressors inside shot-level folds and stores
held-shot predictions in that shared train-only coordinate system.  Development
and frozen-audit targets are never opened.  The resulting archive is intended
for fitting a second-stage residual model without in-sample base errors.
"""

from __future__ import annotations
from fec1.paths import DATA_POLICY, SYNTHETIC

from fec1.paths import CACHE

from fec1.paths import ARTIFACTS

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
from pathlib import Path
import time

import joblib
import numpy as np


from fec1.data.features import _coordinates, _features, _r2, _read, _split_records, _trees


BASE_ARTIFACT = ARTIFACTS / "source_pca.joblib"
_WORKER_PCA = None


def shot_folds(
    shot_names: np.ndarray,
    folds: int,
    seed: int,
) -> list[np.ndarray]:
    """Return deterministic, disjoint shot folds covering every shot once."""

    shots = np.unique(np.asarray(shot_names).astype(str))
    if folds < 2 or folds > len(shots):
        raise ValueError("fold count must lie between two and the shot count")
    shuffled = np.random.default_rng(seed).permutation(shots)
    result = [part.astype(str) for part in np.array_split(shuffled, folds)]
    combined = np.concatenate(result)
    if len(combined) != len(shots) or set(combined) != set(shots):
        raise AssertionError("shot folds do not partition the shots")
    return result


def _initialize_worker(pca: object) -> None:
    global _WORKER_PCA
    _WORKER_PCA = pca


def _shot_rows(task: tuple[str, str, int]) -> tuple[np.ndarray, ...]:
    shot, path_text, stride = task
    if _WORKER_PCA is None:
        raise RuntimeError("training worker PCA was not initialized")
    data = _read(Path(path_text), include_targets=True)
    frame_index = np.arange(0, len(data["target_psi"]), stride, dtype=np.int32)
    mean, rms, normalized_shape = _coordinates(data["target_psi"][frame_index])
    coefficients = _WORKER_PCA.transform(normalized_shape.reshape(len(frame_index), -1)).astype(
        np.float32
    )
    features = _features(data)[frame_index]
    scalar = np.column_stack(
        (
            data["target_q95"][frame_index],
            data["target_betaN"][frame_index],
        )
    ).astype(np.float32)
    return (
        np.full(len(frame_index), shot, dtype="U19"),
        frame_index,
        features,
        coefficients,
        np.column_stack((mean, np.log(rms))).astype(np.float32),
        scalar,
    )


def _finite(name: str, values: np.ndarray) -> None:
    invalid = int(values.size - np.isfinite(values).sum())
    if invalid:
        raise RuntimeError(f"{name} contains {invalid} non-finite values")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", type=Path, default=BASE_ARTIFACT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--estimators", type=int, default=96)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--max-features", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=260901)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.train_stride < 1 or args.workers < 1 or args.estimators < 1:
        raise ValueError("stride, workers, and estimators must be positive")
    started = time.monotonic()

    train_shots, development_shots, frozen_shots = _split_records()
    if not SYNTHETIC and (
        len(train_shots) != 5677 or len(development_shots) != 689 or len(frozen_shots) != 675
    ):
        raise RuntimeError("immutable DIII-D split counts changed")
    paths = [args.cache_dir / f"{shot}.npz" for shot in train_shots]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

    print(f"loading train-only PCA from {args.base_artifact}", flush=True)
    artifact = joblib.load(args.base_artifact)
    if artifact.get("development_used_for_fit") or artifact.get("frozen_audit_used"):
        raise RuntimeError("base artifact is not train-only")
    if int(artifact.get("train_shots", -1)) != len(train_shots):
        raise RuntimeError("base artifact train-shot count changed")
    pca = artifact["pca"]
    pca_components = int(pca.n_components_)
    del artifact
    gc.collect()

    print(
        f"encoding {len(train_shots)} train shots at stride {args.train_stride}",
        flush=True,
    )
    tasks = [(shot, str(path), args.train_stride) for shot, path in zip(train_shots, paths)]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(pca,),
    ) as executor:
        pieces = list(executor.map(_shot_rows, tasks, chunksize=1))
    shot_name = np.concatenate([piece[0] for piece in pieces])
    frame_index = np.concatenate([piece[1] for piece in pieces])
    features = np.concatenate([piece[2] for piece in pieces])
    target_shape = np.concatenate([piece[3] for piece in pieces])
    target_coordinate = np.concatenate([piece[4] for piece in pieces])
    target_scalar = np.concatenate([piece[5] for piece in pieces])
    del pieces
    for name, values in (
        ("features", features),
        ("target shape", target_shape),
        ("target coordinate", target_coordinate),
        ("target scalar", target_scalar),
    ):
        _finite(name, values)
    print(
        f"encoded rows={len(features)} features={features.shape[1]} "
        f"components={target_shape.shape[1]}",
        flush=True,
    )

    base_shape = np.full_like(target_shape, np.nan)
    base_coordinate = np.full_like(target_coordinate, np.nan)
    base_scalar = np.full_like(target_scalar, np.nan)
    reports: list[dict[str, object]] = []
    folds = shot_folds(shot_name, args.folds, args.seed)
    for fold, held_shots in enumerate(folds):
        held = np.isin(shot_name, held_shots)
        fit = ~held
        print(
            f"fold {fold + 1}/{args.folds}: fit_shots={len(np.unique(shot_name[fit]))} "
            f"fit_rows={int(fit.sum())} held_shots={len(held_shots)} "
            f"held_rows={int(held.sum())}",
            flush=True,
        )
        shape_model = _trees(
            args.estimators,
            args.min_samples_leaf,
            args.max_features,
            args.seed + 100 * fold + 1,
        )
        coordinate_model = _trees(
            args.estimators,
            max(2, args.min_samples_leaf // 2),
            args.max_features,
            args.seed + 100 * fold + 2,
        )
        scalar_model = _trees(
            args.estimators,
            args.min_samples_leaf,
            args.max_features,
            args.seed + 100 * fold + 3,
        )
        for model in (shape_model, coordinate_model, scalar_model):
            model.set_params(n_jobs=args.workers)
        shape_model.fit(features[fit], target_shape[fit])
        coordinate_model.fit(features[fit], target_coordinate[fit])
        scalar_model.fit(features[fit], target_scalar[fit])
        base_shape[held] = shape_model.predict(features[held]).astype(np.float32)
        base_coordinate[held] = coordinate_model.predict(features[held]).astype(np.float32)
        base_scalar[held] = scalar_model.predict(features[held]).astype(np.float32)
        report = {
            "fold": fold,
            "fit_shots": int(len(np.unique(shot_name[fit]))),
            "fit_rows": int(fit.sum()),
            "held_shots": int(len(held_shots)),
            "held_rows": int(held.sum()),
            "shape_coefficient_r2": _r2(target_shape[held], base_shape[held]),
            "coordinate_r2": _r2(target_coordinate[held], base_coordinate[held]),
            "q95_r2": _r2(target_scalar[held, 0], base_scalar[held, 0]),
            "betaN_r2": _r2(target_scalar[held, 1], base_scalar[held, 1]),
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
        del shape_model, coordinate_model, scalar_model
        gc.collect()

    for name, values in (
        ("OOF shape", base_shape),
        ("OOF coordinate", base_coordinate),
        ("OOF scalar", base_scalar),
    ):
        _finite(name, values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        shot_name=shot_name.astype("U19"),
        frame_index=frame_index.astype(np.int32),
        features=features.astype(np.float32),
        target_shape=target_shape.astype(np.float32),
        target_coordinate=target_coordinate.astype(np.float32),
        target_scalar=target_scalar.astype(np.float32),
        base_shape=base_shape.astype(np.float32),
        base_coordinate=base_coordinate.astype(np.float32),
        base_scalar=base_scalar.astype(np.float32),
        train_stride=np.asarray(args.train_stride, dtype=np.int32),
        pca_components=np.asarray(pca_components, dtype=np.int32),
        data_policy=np.asarray(DATA_POLICY),
        same_shot_targets_used=np.asarray(False),
        development_used=np.asarray(False),
        frozen_audit_used=np.asarray(False),
    )
    report = {
        "output": str(args.output.resolve()),
        "base_artifact": str(args.base_artifact.resolve()),
        "data_policy": DATA_POLICY,
        "train_shots": len(train_shots),
        "train_rows": int(len(features)),
        "train_stride": args.train_stride,
        "feature_width": int(features.shape[1]),
        "pca_components": pca_components,
        "folds": args.folds,
        "estimators": args.estimators,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "oof_shape_coefficient_r2": _r2(target_shape, base_shape),
        "oof_coordinate_r2": _r2(target_coordinate, base_coordinate),
        "oof_q95_r2": _r2(target_scalar[:, 0], base_scalar[:, 0]),
        "oof_betaN_r2": _r2(target_scalar[:, 1], base_scalar[:, 1]),
        "fold_reports": reports,
        "same_shot_targets_used": False,
        "global_pca_fit": "immutable_train_split_only",
        "development_used": False,
        "frozen_audit_used": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
