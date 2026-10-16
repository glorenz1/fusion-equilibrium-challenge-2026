"""Shot-disjoint PCA/ExtraTrees carriers and within-shot diagnostic context."""
from __future__ import annotations
import hashlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from fec1.data.features import _coordinates


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
    field = (
        coordinate[:, 0, None, None]
        + np.exp(np.clip(coordinate[:, 1], -20, 10))[:, None, None] * shape
    )
    if not np.isfinite(field).all() or not np.isfinite(scalar).all():
        raise RuntimeError("non-finite base prediction")
    return field.astype(np.float32), scalar


def fit_base(
    features,
    target_field,
    source_rows,
    scalar,
    names,
    fit_rows,
    *,
    components,
    estimators,
    workers,
    seed,
):
    rng = np.random.default_rng(seed)
    # Balance the representation sample by shot, using only the fit partition.
    sampled = []
    for shot in np.unique(names[fit_rows]):
        rows = fit_rows[names[fit_rows] == shot]
        sampled.extend(rng.choice(rows, min(8, len(rows)), replace=False))
    sampled = np.asarray(sampled, dtype=np.int64)
    shape_sample = _coordinates(target_field[source_rows[sampled]])[2].reshape(len(sampled), -1)
    pca = PCA(
        n_components=min(components, len(sampled) - 1), svd_solver="randomized", random_state=seed
    )
    pca.fit(shape_sample)
    del shape_sample
    coefficient = np.empty((len(fit_rows), pca.n_components_), np.float32)
    coordinate = np.empty((len(fit_rows), 2), np.float32)
    for start in range(0, len(fit_rows), 512):
        rows = fit_rows[start : start + 512]
        mean, rms, shape = _coordinates(target_field[source_rows[rows]])
        coefficient[start : start + len(rows)] = pca.transform(shape.reshape(len(rows), -1))
        coordinate[start : start + len(rows)] = np.column_stack((mean, np.log(rms)))
    artifact = {
        "pca": pca,
        "fit_shots": np.unique(names[fit_rows]).tolist(),
        "development_used": False,
        "frozen_audit_used": False,
        "data_policy": "real_diii_d_only",
    }
    for index, (key, target) in enumerate(
        (
            ("shape", coefficient),
            ("coordinate", coordinate),
            ("scalar", scalar[fit_rows]),
        )
    ):
        model = ExtraTreesRegressor(
            n_estimators=estimators,
            min_samples_leaf=2 if key == "coordinate" else 4,
            max_features=0.7,
            n_jobs=workers,
            random_state=seed + index,
        )
        model.fit(features[fit_rows], target)
        artifact[key] = model
    return artifact
