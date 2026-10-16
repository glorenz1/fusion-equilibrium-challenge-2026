#!/usr/bin/env python3
"""Pool nested OOF predictions, freeze a family, then evaluate development.

Development targets are opened only after the family and ensemble are frozen
and all development predictions have been written. This script never publishes
a submission or changes the existing frozen baseline.
"""
from __future__ import annotations

from fec1.paths import CACHE


import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from fec1.model import (
    StructuredCorrection,
    calibrated_prediction,
    FrozenScalarPredictor,
    within_shot_neighbors,
)
from fec1.carrier import (
    diagnostic_context,
    predict_base,
)
from fec1.paths import SOURCE, SPATIAL
from fec1.data.features import _features, _read, _split_records
from fec1.scoring.archive import _parallel_official_score


def pooled_score(root, family, folds, output, workers=1):
    with np.load(SOURCE, allow_pickle=False) as source:
        names = source["shot_name"].astype(str)
        frames = source["frame_index"]
        scalar = source["target_scalar"]
    field = np.lib.format.open_memmap(
        output / f"{family}_pooled_field.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(names), 65, 65),
    )
    predicted_scalar = np.empty_like(scalar)
    coverage = np.zeros(len(names), dtype=np.int8)
    fold_reports = []
    for fold in range(folds):
        directory = root / f"{family}_fold{fold}"
        report = json.loads((directory / "result.json").read_text())
        if report["smoke_only"] or not report["nested_shot_crossfit"]:
            raise RuntimeError("smoke or non-crossfit results cannot enter evaluation")
        if report["development_used"] or report["frozen_audit_used"]:
            raise RuntimeError("evaluation labels entered training")
        with np.load(directory / "outer_predictions.npz", allow_pickle=False) as prediction:
            rows = prediction["source_row"].astype(np.int64)
            if (
                len(np.unique(rows)) != len(rows)
                or np.any(rows < 0)
                or np.any(rows >= len(names))
                or np.any(coverage[rows])
            ):
                raise RuntimeError("overlapping or invalid OOF rows")
            if not (
                np.array_equal(names[rows], prediction["shot_name"].astype(str))
                and np.array_equal(frames[rows], prediction["frame_index"])
            ):
                raise RuntimeError("misaligned OOF predictions")
            field[rows] = prediction["psirz"]
            predicted_scalar[rows] = np.column_stack((prediction["q95"], prediction["betaN"]))
            coverage[rows] += 1
        fold_reports.append(report)
    if not np.all(coverage == 1):
        raise RuntimeError("OOF population is incomplete")
    field.flush()
    values = {
        "field": field,
        "q95": predicted_scalar[:, 0],
        "betaN": predicted_scalar[:, 1],
        "target_field": np.load(SPATIAL / "target_field.npy", mmap_mode="r"),
        "target_q95": scalar[:, 0],
        "target_betaN": scalar[:, 1],
        "shot_name": names,
        "source_index": frames,
        "shot_index": np.unique(names, return_inverse=True)[1],
        "reference_source": str(CACHE.resolve()),
    }
    result = _parallel_official_score(values, workers, output / "pooled_reference.pkl")
    return {
        "family": family,
        "score": result,
        "folds": fold_reports,
        "shots": len(np.unique(names)),
        "frames": len(names),
        "frame_stride": 4,
        "nested_shot_crossfit": True,
    }


@torch.no_grad()
def predict_development(
    root, family, folds, output, device, batch_size, split_name="development", workers=1
):
    # Read only row counts and inputs until every prediction is frozen.
    training, development, frozen = _split_records()
    if split_name == "frozen_audit":
        development = frozen
    elif split_name != "development":
        raise ValueError("invalid evaluation split")
    names, frames, lengths = [], [], []
    for shot in sorted(development):
        with np.load(CACHE / f"{shot}.npz", allow_pickle=False) as data:
            count = len(data["canonical_global_features"])
        names.extend([shot] * count)
        frames.extend(range(count))
        lengths.append(count)
    names = np.asarray(names, dtype="U19")
    frames = np.asarray(frames, dtype=np.int32)
    field = np.lib.format.open_memmap(
        output / f"{split_name}_field.npy", mode="w+", dtype=np.float32, shape=(len(names), 65, 65)
    )
    field[:] = 0
    scalar = np.zeros((len(names), 2), np.float32)
    starts = np.concatenate(([0], np.cumsum(lengths)))
    for fold in range(folds):
        checkpoint = torch.load(
            root / f"{family}_fold{fold}/checkpoint.pt", map_location="cpu", weights_only=False
        )
        if (
            checkpoint["smoke_only"]
            or not checkpoint["nested_shot_crossfit"]
            or checkpoint["development_used"]
            or checkpoint["frozen_audit_used"]
        ):
            raise RuntimeError("unsafe correction checkpoint")
        model = StructuredCorrection(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        fixed_scalar = (
            FrozenScalarPredictor(checkpoint["fixed_scalar_checkpoint"]).to(device)
            if checkpoint.get("fixed_scalar_checkpoint") is not None
            else None
        )
        base = joblib.load(root / f"base_fold{fold}/base.joblib")
        if (
            set(base["fit_shots"]) - set(training)
            or base["development_used"]
            or base["frozen_audit_used"]
        ):
            raise RuntimeError("unsafe base checkpoint")
        for shot_number, shot in enumerate(sorted(development)):
            data = _read(CACHE / f"{shot}.npz", include_targets=False)
            feature = _features(data)
            base_field, base_scalar = predict_base(base, feature)
            local_count = len(feature)
            # Match the stride-4 context used for the training cache, while
            # predicting every development frame exactly once per ensemble member.
            context = np.empty((local_count, feature.shape[1] * 3), np.float32)
            sequence_rows = np.empty((local_count, 9), dtype=np.int64)
            sequence_mask = np.empty((local_count, 9), dtype=bool)
            for phase in range(4):
                rows = np.arange(phase, local_count, 4)
                if len(rows):
                    context[rows] = diagnostic_context(feature[rows])
                    neighbors, valid = within_shot_neighbors(np.full(len(rows), shot), rows)
                    sequence_rows[rows], sequence_mask[rows] = rows[neighbors], valid
            start = int(starts[shot_number])
            for offset in range(0, local_count, batch_size):
                rows = np.arange(offset, min(offset + batch_size, local_count))
                inputs = (
                    context[rows],
                    base_field[rows],
                    base_scalar[rows],
                    data["token_values"][rows],
                    data["token_coords"][rows],
                    data["token_mask"][rows],
                )
                if model.config["observation_encoder"] == "temporal_attention":
                    inputs += (feature[sequence_rows[rows]], sequence_mask[rows])
                tensors = tuple(torch.as_tensor(np.array(a), device=device) for a in inputs)
                prediction = model(*tensors)
                if fixed_scalar is not None:
                    prediction["scalar"] = fixed_scalar(*tensors)
                corrected, corrected_scalar = calibrated_prediction(
                    base_field[rows],
                    base_scalar[rows],
                    prediction["field"].cpu().numpy(),
                    prediction["scalar"].cpu().numpy(),
                    checkpoint["correction_strengths"],
                )
                field[start + rows] += corrected / folds
                scalar[start + rows] += corrected_scalar / folds
            if shot_number % 100 == 0:
                print(
                    json.dumps(
                        {
                            "stage": f"{split_name}_inference",
                            "fold": fold,
                            "shot": shot_number,
                            "total": len(development),
                        }
                    ),
                    flush=True,
                )
        del model, base, checkpoint, fixed_scalar
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    field.flush()
    if not np.isfinite(field).all() or not np.isfinite(scalar).all():
        raise RuntimeError("non-finite ensemble predictions")
    np.savez_compressed(
        output / f"{split_name}_predictions.npz",
        sample_index=np.arange(len(names)),
        shot_name=names,
        frame_index=frames,
        psirz=field,
        q95=scalar[:, 0],
        betaN=scalar[:, 1],
        data_policy=np.asarray("real_diii_d_only"),
        frozen_audit_used=np.asarray(split_name == "frozen_audit"),
        evaluation_split=np.asarray(split_name),
        prediction_variant=np.asarray("nested_train_crossfit_ensemble"),
    )
    # Open development labels for scoring after the prediction archive exists.
    target = np.lib.format.open_memmap(
        output / f"{split_name}_target.npy", mode="w+", dtype=np.float32, shape=field.shape
    )
    target_scalar = np.empty_like(scalar)
    for shot_number, shot in enumerate(sorted(development)):
        data = _read(CACHE / f"{shot}.npz", include_targets=True)
        start, stop = starts[shot_number : shot_number + 2]
        target[start:stop] = data["target_psi"]
        target_scalar[start:stop] = np.column_stack((data["target_q95"], data["target_betaN"]))
    target.flush()
    np.save(output / f"{split_name}_target_scalars.npy", target_scalar)
    values = {
        "field": field,
        "q95": scalar[:, 0],
        "betaN": scalar[:, 1],
        "target_field": target,
        "target_q95": target_scalar[:, 0],
        "target_betaN": target_scalar[:, 1],
        "shot_name": names,
        "source_index": frames,
        "shot_index": np.unique(names, return_inverse=True)[1],
        "reference_source": str(CACHE.resolve()),
    }
    return _parallel_official_score(
        values,
        workers,
        output / f"{split_name}_reference.pkl",
        output / f"{split_name}_contributions.npz",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pool final DCT outer-OOF predictions and evaluate development separately."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--score-workers", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pooled = pooled_score(args.root, "dct", args.folds, args.output, args.score_workers)
    selection = dict(
        selected_family="dct",
        ensemble_weights=[1 / args.folds] * args.folds,
        reports={"dct": pooled},
        development_labels_used_for_selection=False,
        frozen_audit_used=False,
    )
    (args.output / "frozen_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    development = predict_development(
        args.root,
        "dct",
        args.folds,
        args.output,
        torch.device(args.device),
        args.batch_size,
        workers=args.score_workers,
    )
    result = dict(
        selected_family="dct",
        pooled_oof_score=pooled["score"],
        development_score=development,
        frozen_audit_used=False,
        published=False,
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
