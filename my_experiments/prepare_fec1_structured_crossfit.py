#!/usr/bin/env python3
"""Build a nested, shot-disjoint base for a learned FEC1 correction fold.

Only immutable training shots are used. The old OOF archive supplies diagnostic
features and row identities, never its fitted PCA or base predictions. PCA and
all three regressors are refitted inside each allowed shot partition.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from my_experiments.fit_fec1_full_frame_pca import _coordinates, _split_records

ARTIFACTS = ROOT / "my_experiments/artifacts"
SOURCE = ARTIFACTS / "fec1_train_stride4_oof_base.npz"
SPATIAL = ARTIFACTS / "fec1_topology_native_inputs"


def partitions(names, fold, folds=5, seed=260905, validation_fraction=0.1):
    shots = np.unique(names)
    if not 0 <= fold < folds or len(shots) < 2 * folds:
        raise ValueError("invalid outer fold or insufficient shots")
    shuffled = np.random.default_rng(seed).permutation(shots)
    outer = np.array_split(shuffled, folds)[fold]
    remaining = shuffled[~np.isin(shuffled, outer)]
    count = max(1, int(round(len(remaining) * validation_fraction)))
    validation = remaining[:count]
    role = np.zeros(len(names), dtype=np.int8)
    role[np.isin(names, validation)] = 1
    role[np.isin(names, outer)] = 2
    return role


def diagnostic_context(features):
    """Input-only, within-shot context; identical at training and inference."""
    x = np.asarray(features, dtype=np.float32)
    difference = np.zeros_like(x)
    difference[1:] = x[1:] - x[:-1]
    mean = np.broadcast_to(x.mean(axis=0), x.shape)
    return np.column_stack((x, difference, mean)).astype(np.float32)


def row_digest(names, frames):
    digest = hashlib.sha256()
    for name, frame in zip(names, frames):
        digest.update(f"{name}:{int(frame)}\n".encode())
    return digest.hexdigest()


def predict_base(artifact, features):
    coefficient = artifact["shape"].predict(features).astype(np.float32)
    coordinate = artifact["coordinate"].predict(features).astype(np.float32)
    scalar = artifact["scalar"].predict(features).astype(np.float32)
    shape = artifact["pca"].inverse_transform(coefficient).reshape(-1, 65, 65)
    shape = _coordinates(shape)[2]
    field = coordinate[:, 0, None, None] + np.exp(
        np.clip(coordinate[:, 1], -20, 10)
    )[:, None, None] * shape
    if not np.isfinite(field).all() or not np.isfinite(scalar).all():
        raise RuntimeError("non-finite base prediction")
    return field.astype(np.float32), scalar


def fit_base(features, target_field, source_rows, scalar, names, fit_rows,
             *, components, estimators, workers, seed):
    rng = np.random.default_rng(seed)
    # Balance the representation sample by shot, using only the fit partition.
    sampled = []
    for shot in np.unique(names[fit_rows]):
        rows = fit_rows[names[fit_rows] == shot]
        sampled.extend(rng.choice(rows, min(8, len(rows)), replace=False))
    sampled = np.asarray(sampled, dtype=np.int64)
    shape_sample = _coordinates(target_field[source_rows[sampled]])[2].reshape(len(sampled), -1)
    pca = PCA(n_components=min(components, len(sampled) - 1),
              svd_solver="randomized", random_state=seed)
    pca.fit(shape_sample)
    del shape_sample
    coefficient = np.empty((len(fit_rows), pca.n_components_), np.float32)
    coordinate = np.empty((len(fit_rows), 2), np.float32)
    for start in range(0, len(fit_rows), 512):
        rows = fit_rows[start:start + 512]
        mean, rms, shape = _coordinates(target_field[source_rows[rows]])
        coefficient[start:start + len(rows)] = pca.transform(shape.reshape(len(rows), -1))
        coordinate[start:start + len(rows)] = np.column_stack((mean, np.log(rms)))
    artifact = {"pca": pca, "fit_shots": np.unique(names[fit_rows]).tolist(),
                "development_used": False, "frozen_audit_used": False,
                "data_policy": "real_diii_d_only"}
    for index, (key, target) in enumerate((
        ("shape", coefficient), ("coordinate", coordinate),
        ("scalar", scalar[fit_rows]),
    )):
        model = ExtraTreesRegressor(
            n_estimators=estimators, min_samples_leaf=2 if key == "coordinate" else 4,
            max_features=0.7, n_jobs=workers, random_state=seed + index,
        )
        model.fit(features[fit_rows], target)
        artifact[key] = model
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=260905)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--estimators", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-shots", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"use a fresh output directory: {args.output}")
    started = time.monotonic()
    with np.load(SOURCE, allow_pickle=False) as archive:
        if archive["development_used"].item() or archive["frozen_audit_used"].item():
            raise RuntimeError("unsafe source provenance")
        names = archive["shot_name"].astype(str)
        frames = archive["frame_index"].astype(np.int64)
        features = archive["features"].astype(np.float32)
        scalars = archive["target_scalar"].astype(np.float32)
    with np.load(SPATIAL / "index.npz", allow_pickle=False) as index:
        if not (np.array_equal(names, index["shot_name"].astype(str))
                and np.array_equal(frames, index["frame_index"])):
            raise RuntimeError("spatial cache is not aligned to source archive")
    training, development, frozen = _split_records()
    if set(names) != set(training) or set(names) & (development | frozen):
        raise RuntimeError("source does not match the immutable training split")
    source_rows = np.arange(len(names))
    if args.max_shots:
        chosen = np.random.default_rng(args.seed).permutation(np.unique(names))[:args.max_shots]
        keep = np.isin(names, chosen)
        names, frames, features, scalars, source_rows = (
            value[keep] for value in (names, frames, features, scalars, source_rows)
        )
    role = partitions(names, args.fold, args.folds, args.seed)
    fit_rows = np.flatnonzero(role == 0)
    fit_shots = np.unique(names[fit_rows])
    if not 2 <= args.inner_folds <= len(fit_shots):
        raise ValueError("invalid number of inner folds")
    args.output.mkdir(parents=True)
    context = np.empty((len(names), features.shape[1] * 3), np.float32)
    for shot in np.unique(names):
        rows = np.flatnonzero(names == shot)
        rows = rows[np.argsort(frames[rows])]
        context[rows] = diagnostic_context(features[rows])
    np.save(args.output / "features.npy", context)
    del context
    np.savez_compressed(args.output / "index.npz", shot_name=names, frame_index=frames,
                        source_row=source_rows, role=role, target_scalar=scalars)
    base_field = np.lib.format.open_memmap(args.output / "base_field.npy", mode="w+",
                                          dtype=np.float32, shape=(len(names), 65, 65))
    base_scalar = np.lib.format.open_memmap(args.output / "base_scalar.npy", mode="w+",
                                           dtype=np.float32, shape=(len(names), 2))
    target_field = np.load(SPATIAL / "target_field.npy", mmap_mode="r")
    inner_shots = np.array_split(np.random.default_rng(args.seed + args.fold + 1)
                                .permutation(fit_shots), args.inner_folds)
    reports = []
    written = np.zeros(len(names), dtype=bool)
    for inner, held_shots in enumerate([*inner_shots, None]):
        if held_shots is None:
            local_fit = fit_rows
            predicted_rows = np.flatnonzero(role != 0)
        else:
            is_held = np.isin(names, held_shots)
            local_fit = np.flatnonzero((role == 0) & ~is_held)
            predicted_rows = np.flatnonzero((role == 0) & is_held)
        assert not set(names[local_fit]) & set(names[predicted_rows])
        print(json.dumps({"stage": "base_fit", "outer_fold": args.fold,
                          "inner": inner, "fit_shots": len(np.unique(names[local_fit])),
                          "predict_shots": len(np.unique(names[predicted_rows]))}), flush=True)
        artifact = fit_base(features, target_field, source_rows, scalars, names, local_fit,
                            components=args.components, estimators=args.estimators,
                            workers=args.workers, seed=args.seed + 100 * args.fold + 10 * inner)
        for start in range(0, len(predicted_rows), 1024):
            rows = predicted_rows[start:start + 1024]
            field, scalar = predict_base(artifact, features[rows])
            base_field[rows], base_scalar[rows] = field, scalar
            written[rows] = True
        reports.append({"inner": inner, "fit_shots": artifact["fit_shots"],
                        "predicted_shots": np.unique(names[predicted_rows]).tolist()})
        if held_shots is None:
            joblib.dump(artifact, args.output / "base.joblib", compress=0)
        del artifact
        gc.collect()
        base_field.flush()
        base_scalar.flush()
    if not written.all():
        raise RuntimeError("incomplete nested base archive")
    report = {"status": "complete", "fold": args.fold, "folds": args.folds,
              "seed": args.seed, "rows": len(names), "shots": len(np.unique(names)),
              "row_digest": row_digest(names, frames), "roles": {
                  str(r): np.unique(names[role == r]).tolist() for r in (0, 1, 2)},
              "base_fits": reports, "nested_shot_crossfit": True,
              "shared_target_pca_used": False, "development_used": False,
              "frozen_audit_used": False, "data_policy": "real_diii_d_only",
              "smoke_only": bool(args.max_shots), "elapsed_seconds": time.monotonic() - started}
    (args.output / "ready.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("roles", "base_fits")}), flush=True)


if __name__ == "__main__":
    main()
