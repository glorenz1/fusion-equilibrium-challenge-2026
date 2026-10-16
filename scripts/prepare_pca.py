"""Rebuild the initial train-only PCA without the old development-cache dependency.

The nested carrier preparation refits its own PCA. This initial PCA is used
only by the legacy source-row/teacher cache builders, never by deployment.
Sampling, ordering, random seeds and PCA arguments match fit_fec1_full_frame_pca.
"""
from __future__ import annotations


import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fec1.paths import CACHE

import joblib
import numpy as np
from sklearn.decomposition import PCA
from fec1.data.features import _pca_sample, _split_records


def fit_pca(sample, components=64, frames=60000):
    if len(sample) > frames:
        sample = sample[np.random.default_rng(260901).choice(len(sample), frames, replace=False)]
    return PCA(
        n_components=components, svd_solver="randomized", iterated_power=3, random_state=260902
    ).fit(sample.reshape(len(sample), -1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.workers < 1:
        parser.error("workers must be positive")
    training, development, audit = _split_records()
    if (len(training), len(development), len(audit)) != (5677, 689, 675):
        raise ValueError("immutable split changed")
    per_shot = int(np.ceil(60000 / len(training)))
    tasks = [(str(CACHE / f"{shot}.npz"), per_shot) for shot in training]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        sample = np.concatenate(list(executor.map(_pca_sample, tasks, chunksize=4)))
    pca = fit_pca(sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        dict(
            pca=pca,
            train_shots=len(training),
            data_policy="real_diii_d_only",
            mast_data_used=False,
            synthetic_data_used=False,
            frozen_audit_used=False,
            development_used_for_fit=False,
            preparation="training_only_pca",
        ),
        args.output,
        compress=1,
    )


if __name__ == "__main__":
    main()
