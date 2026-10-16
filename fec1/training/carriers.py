"""Fit nested, shot-disjoint FEC1 carriers."""
from __future__ import annotations
import argparse, gc, json, time
from pathlib import Path
import joblib
import numpy as np
from fec1.paths import SOURCE, SPATIAL
from fec1.data.features import _split_records
from fec1.carrier import partitions, diagnostic_context, row_digest, predict_base, fit_base


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
        if not (
            np.array_equal(names, index["shot_name"].astype(str))
            and np.array_equal(frames, index["frame_index"])
        ):
            raise RuntimeError("spatial cache is not aligned to source archive")
    training, development, frozen = _split_records()
    if set(names) != set(training) or set(names) & (development | frozen):
        raise RuntimeError("source does not match the immutable training split")
    source_rows = np.arange(len(names))
    if args.max_shots:
        chosen = np.random.default_rng(args.seed).permutation(np.unique(names))[: args.max_shots]
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
    np.savez_compressed(
        args.output / "index.npz",
        shot_name=names,
        frame_index=frames,
        source_row=source_rows,
        role=role,
        target_scalar=scalars,
    )
    base_field = np.lib.format.open_memmap(
        args.output / "base_field.npy", mode="w+", dtype=np.float32, shape=(len(names), 65, 65)
    )
    base_scalar = np.lib.format.open_memmap(
        args.output / "base_scalar.npy", mode="w+", dtype=np.float32, shape=(len(names), 2)
    )
    target_field = np.load(SPATIAL / "target_field.npy", mmap_mode="r")
    inner_shots = np.array_split(
        np.random.default_rng(args.seed + args.fold + 1).permutation(fit_shots), args.inner_folds
    )
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
        print(
            json.dumps(
                {
                    "stage": "base_fit",
                    "outer_fold": args.fold,
                    "inner": inner,
                    "fit_shots": len(np.unique(names[local_fit])),
                    "predict_shots": len(np.unique(names[predicted_rows])),
                }
            ),
            flush=True,
        )
        artifact = fit_base(
            features,
            target_field,
            source_rows,
            scalars,
            names,
            local_fit,
            components=args.components,
            estimators=args.estimators,
            workers=args.workers,
            seed=args.seed + 100 * args.fold + 10 * inner,
        )
        for start in range(0, len(predicted_rows), 1024):
            rows = predicted_rows[start : start + 1024]
            field, scalar = predict_base(artifact, features[rows])
            base_field[rows], base_scalar[rows] = field, scalar
            written[rows] = True
        reports.append(
            {
                "inner": inner,
                "fit_shots": artifact["fit_shots"],
                "predicted_shots": np.unique(names[predicted_rows]).tolist(),
            }
        )
        if held_shots is None:
            joblib.dump(artifact, args.output / "base.joblib", compress=0)
        del artifact
        gc.collect()
        base_field.flush()
        base_scalar.flush()
    if not written.all():
        raise RuntimeError("incomplete nested base archive")
    report = {
        "status": "complete",
        "fold": args.fold,
        "folds": args.folds,
        "seed": args.seed,
        "rows": len(names),
        "shots": len(np.unique(names)),
        "row_digest": row_digest(names, frames),
        "roles": {str(r): np.unique(names[role == r]).tolist() for r in (0, 1, 2)},
        "base_fits": reports,
        "nested_shot_crossfit": True,
        "shared_target_pca_used": False,
        "development_used": False,
        "frozen_audit_used": False,
        "data_policy": "real_diii_d_only",
        "smoke_only": bool(args.max_shots),
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "ready.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("roles", "base_fits")}), flush=True
    )


if __name__ == "__main__":
    main()
