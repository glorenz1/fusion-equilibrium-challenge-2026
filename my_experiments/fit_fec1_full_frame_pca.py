#!/usr/bin/env python3
"""Fit a leakage-safe full-frame DIII-D PCA/ExtraTrees equilibrium model.

Only shots assigned to the immutable ``train`` split enter PCA or regression
fitting.  Development shots are opened only after all models are frozen, and
frozen-audit shots are never opened by this script.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CACHE = ROOT / "my_experiments/neuralgs_real_d3d_full_frame_cache"
MANIFEST = ROOT / "my_experiments/causal_hybrid_encoder_v2_dataset/manifest.json"
PARTS = ROOT / "my_experiments/causal_hybrid_encoder_v2_dataset/parts"
_WORKER_PCA = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coordinates(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(field, dtype=np.float32)
    mean = values.mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
    centered = values - mean[:, None, None]
    rms = np.sqrt(
        np.mean(np.square(centered), axis=(1, 2), dtype=np.float64)
    ).astype(np.float32)
    rms = np.maximum(rms, 1.0e-8)
    return mean, rms, (centered / rms[:, None, None]).astype(np.float32)


def _centered_context(values: np.ndarray, width: int = 8) -> tuple[np.ndarray, np.ndarray]:
    values64 = np.asarray(values, dtype=np.float64)
    count = len(values64)
    mean = np.empty_like(values64)
    std = np.empty_like(values64)
    left = (width - 1) // 2
    for target in range(count):
        start = target - left
        stop = start + width
        if start < 0:
            stop -= start
            start = 0
        if stop > count:
            start = max(0, start - (stop - count))
            stop = count
        local = values64[start:stop]
        mean[target] = local.mean(axis=0)
        std[target] = local.std(axis=0)
    return mean.astype(np.float32), std.astype(np.float32)


def _features(saved: dict[str, np.ndarray]) -> np.ndarray:
    from neuralgs.scalar_ridge import build_scalar_features

    scalar = build_scalar_features(saved)
    canonical = np.asarray(saved["canonical_global_features"], dtype=np.float32)
    context_mean, context_std = _centered_context(canonical)
    features = np.column_stack((canonical, context_mean, context_std, scalar))
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _read(path: Path, include_targets: bool = True) -> dict[str, np.ndarray]:
    input_keys = (
        "vacuum_psi", "global_features", "canonical_global_features",
        "token_coords", "token_values", "token_mask", "grid_r", "grid_z",
    )
    target_keys = ("target_psi", "target_q95", "target_betaN")
    with np.load(path, allow_pickle=False) as saved:
        keys = input_keys + target_keys if include_targets else input_keys
        return {key: np.asarray(saved[key]) for key in keys}


def _pca_sample(task: tuple[str, int]) -> np.ndarray:
    path_text, per_shot = task
    data = _read(Path(path_text), include_targets=True)
    count = len(data["target_psi"])
    chosen = np.unique(
        np.linspace(0, count - 1, min(per_shot, count)).round().astype(np.int64)
    )
    return _coordinates(data["target_psi"][chosen])[2]


def _initialize_training_worker(pca: object) -> None:
    global _WORKER_PCA
    _WORKER_PCA = pca


def _training_rows(task: tuple[str, int]) -> tuple[np.ndarray, ...]:
    path_text, stride = task
    if _WORKER_PCA is None:
        raise RuntimeError("training worker PCA was not initialized")
    data = _read(Path(path_text), include_targets=True)
    chosen = np.arange(0, len(data["target_psi"]), stride, dtype=np.int64)
    mean, rms, shape = _coordinates(data["target_psi"][chosen])
    coefficients = _WORKER_PCA.transform(
        shape.reshape(len(shape), -1)
    ).astype(np.float32)
    return (
        _features(data)[chosen],
        coefficients,
        np.column_stack((mean, np.log(rms))).astype(np.float32),
        data["target_q95"][chosen].astype(np.float32),
        data["target_betaN"][chosen].astype(np.float32),
    )


def _split_records() -> tuple[list[str], set[str], set[str]]:
    manifest = json.loads(MANIFEST.read_text())
    train = [record["shot"] for record in manifest["records"] if record["split"] == "train"]
    development = {record["shot"] for record in manifest["records"]
                   if record["split"] == "development"}
    frozen = {record["shot"] for record in manifest["records"]
              if record["split"] == "frozen_audit"}
    if set(train) & development or set(train) & frozen or development & frozen:
        raise RuntimeError("immutable shot split overlaps")
    return train, development, frozen


def _development_metadata(parts: int) -> dict[str, np.ndarray]:
    collected = {key: [] for key in ("sample_index", "shot_name", "frame_index")}
    for index in range(parts):
        with np.load(PARTS / f"part_{index:04d}.npz", allow_pickle=False) as saved:
            mask = np.asarray(saved["split"]).astype(str) == "development"
            for key in collected:
                collected[key].append(np.asarray(saved[key])[mask])
    values = {key: np.concatenate(parts_value) for key, parts_value in collected.items()}
    order = np.argsort(values["sample_index"].astype(np.int64))
    return {key: value[order] for key, value in values.items()}


def _trees(estimators: int, leaf: int, max_features: float, seed: int):
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=estimators,
        min_samples_leaf=leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=seed,
    )


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    total = np.square(y - y.mean()).sum(dtype=np.float64)
    return float(1.0 - np.square(y - p).sum(dtype=np.float64) / max(total, 1.0e-30))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--pca-frames", type=int, default=60000)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--estimators", type=int, default=96)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--max-features", type=float, default=0.7)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    train_shots, development_shots, frozen_shots = _split_records()
    train_paths = [CACHE / f"{shot}.npz" for shot in train_shots]
    missing = [path for path in train_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    print(
        f"split train={len(train_shots)} development={len(development_shots)} "
        f"frozen_audit={len(frozen_shots)}", flush=True,
    )

    from sklearn.decomposition import PCA

    per_shot = max(1, int(np.ceil(args.pca_frames / len(train_paths))))
    print(f"loading train-only PCA sample, up to {per_shot} frames/shot", flush=True)
    with ProcessPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        pieces = list(executor.map(
            _pca_sample,
            [(str(path), per_shot) for path in train_paths],
            chunksize=4,
        ))
    pca_sample = np.concatenate(pieces)
    if len(pca_sample) > args.pca_frames:
        selected = np.random.default_rng(260901).choice(
            len(pca_sample), args.pca_frames, replace=False
        )
        pca_sample = pca_sample[selected]
    pca = PCA(
        n_components=args.pca_components,
        svd_solver="randomized",
        iterated_power=3,
        random_state=260902,
    )
    pca.fit(pca_sample.reshape(len(pca_sample), -1))
    del pca_sample, pieces
    print(
        f"PCA frames={args.pca_frames} explained={pca.explained_variance_ratio_.sum():.8f}",
        flush=True,
    )

    print("encoding full-frame train rows", flush=True)
    # PCA is read-only and small enough to serialize once per worker.  Limit the
    # pool here because every task also opens a multi-megabyte shot archive.
    encode_workers = min(max(args.workers, 1), 8)
    with ProcessPoolExecutor(
        max_workers=encode_workers,
        initializer=_initialize_training_worker,
        initargs=(pca,),
    ) as executor:
        pieces = list(executor.map(
            _training_rows,
            [(str(path), args.train_stride) for path in train_paths],
            chunksize=1,
        ))
    x = np.concatenate([piece[0] for piece in pieces])
    shape_y = np.concatenate([piece[1] for piece in pieces])
    coordinate_y = np.concatenate([piece[2] for piece in pieces])
    q_y = np.concatenate([piece[3] for piece in pieces])
    beta_y = np.concatenate([piece[4] for piece in pieces])
    del pieces
    if not all(np.isfinite(value).all() for value in (x, shape_y, coordinate_y, q_y, beta_y)):
        raise RuntimeError("non-finite full-frame training matrix")
    print(f"training rows={len(x)} features={x.shape[1]}", flush=True)

    shape_model = _trees(
        args.estimators, args.min_samples_leaf, args.max_features, 260903
    )
    shape_model.fit(x, shape_y)
    coordinate_model = _trees(
        args.estimators, max(2, args.min_samples_leaf // 2),
        args.max_features, 260904,
    )
    coordinate_model.fit(x, coordinate_y)
    scalar_model = _trees(
        args.estimators, args.min_samples_leaf, args.max_features, 260905
    )
    scalar_model.fit(x, np.column_stack((q_y, beta_y)))
    del x, shape_y, coordinate_y, q_y, beta_y

    artifact = {
        "version": 1,
        "data_policy": "real_diii_d_only",
        "mast_data_used": False,
        "synthetic_data_used": False,
        "frozen_audit_used": False,
        "development_used_for_fit": False,
        "train_shots": len(train_shots),
        "development_shots_excluded": len(development_shots),
        "frozen_audit_shots_excluded": len(frozen_shots),
        "train_stride": args.train_stride,
        "feature_width": int(shape_model.n_features_in_),
        "pca": pca,
        "shape_model": shape_model,
        "coordinate_model": coordinate_model,
        "scalar_model": scalar_model,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact, compress=1)
    print(f"saved {args.artifact}", flush=True)

    print("predicting untouched development shots", flush=True)
    metadata = _development_metadata(args.parts)
    if set(metadata["shot_name"].astype(str)) != development_shots:
        raise RuntimeError("development metadata does not match immutable split")
    fields = np.empty((len(metadata["sample_index"]), 65, 65), dtype=np.float32)
    q95 = np.empty(len(fields), dtype=np.float32)
    beta_n = np.empty(len(fields), dtype=np.float32)
    target_fields = np.empty_like(fields)
    target_q95 = np.empty(len(fields), dtype=np.float32)
    target_beta = np.empty(len(fields), dtype=np.float32)
    names = metadata["shot_name"].astype(str)
    for count, shot in enumerate(sorted(development_shots), start=1):
        positions = np.flatnonzero(names == shot)
        frame_indices = metadata["frame_index"][positions].astype(np.int64)
        data = _read(CACHE / f"{shot}.npz", include_targets=True)
        feature = _features(data)[frame_indices]
        coefficients = shape_model.predict(feature)
        shape = pca.inverse_transform(coefficients).reshape(-1, 65, 65)
        _, _, shape = _coordinates(shape)
        coordinates = coordinate_model.predict(feature)
        scalar = scalar_model.predict(feature)
        fields[positions] = (
            coordinates[:, 0, None, None]
            + np.exp(coordinates[:, 1])[:, None, None] * shape
        ).astype(np.float32)
        q95[positions] = np.maximum(scalar[:, 0], 1.0e-4)
        beta_n[positions] = np.maximum(scalar[:, 1], 1.0e-4)
        target_fields[positions] = data["target_psi"][frame_indices]
        target_q95[positions] = data["target_q95"][frame_indices]
        target_beta[positions] = data["target_betaN"][frame_indices]
        if count == 1 or count % 50 == 0 or count == len(development_shots):
            print(f"development {count}/{len(development_shots)} shots", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_index=metadata["sample_index"].astype(np.int64),
        shot_name=metadata["shot_name"].astype("U19"),
        frame_index=metadata["frame_index"].astype(np.int32),
        psirz=fields,
        q95=q95,
        betaN=beta_n,
        data_policy=np.asarray("real_diii_d_only"),
        mast_data_used=np.asarray(False),
        frozen_audit_used=np.asarray(False),
        development_used_for_fit=np.asarray(False),
    )
    report = {
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": _sha256(args.artifact),
        "output": str(args.output.resolve()),
        "data_policy": "real_diii_d_only",
        "mast_data_used": False,
        "frozen_audit_used": False,
        "development_used_for_fit": False,
        "train_shots": len(train_shots),
        "train_stride": args.train_stride,
        "pca_frames": args.pca_frames,
        "pca_components": args.pca_components,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "estimators": args.estimators,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "development_frames": len(fields),
        "development_flux_r2": _r2(target_fields, fields),
        "development_q95_r2": _r2(target_q95, q95),
        "development_betaN_r2": _r2(target_beta, beta_n),
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
