#!/usr/bin/env python3
"""Fit a deployable neural residual head from train-shot OOF base errors.

The input archive contains base predictions made by models that did not see the
corresponding shot.  Model selection uses a shot-disjoint validation subset;
after selecting an epoch count, the final head is refit on every immutable
training shot.  Development and frozen-audit data are never opened here.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


class OOFResidualHead(nn.Module):
    """Shared nonlinear trunk with separate equilibrium-output heads."""

    def __init__(self, features: int, shape: int, width: int, dropout: float) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(features, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.shape = nn.Linear(width, shape)
        self.coordinate = nn.Linear(width, 2)
        self.scalar = nn.Linear(width, 2)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.trunk(values)
        return self.shape(hidden), self.coordinate(hidden), self.scalar(hidden)


def split_validation_shots(
    shot_name: np.ndarray, fraction: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row masks for a deterministic, shot-disjoint train/validation split."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("validation fraction must lie strictly between zero and one")
    names = np.asarray(shot_name).astype(str)
    shots = np.unique(names)
    if len(shots) < 2:
        raise ValueError("at least two shots are required")
    shuffled = np.random.default_rng(seed).permutation(shots)
    count = min(len(shots) - 1, max(1, int(round(fraction * len(shots)))))
    validation_shots = shuffled[:count]
    validation = np.isin(names, validation_shots)
    training = ~validation
    if set(names[training]) & set(names[validation]):
        raise AssertionError("training and validation shots overlap")
    return training, validation


def correction_strength(prediction: np.ndarray, target: np.ndarray) -> float:
    """Least-squares scale for a predicted correction, bounded for stability."""

    prediction64 = np.asarray(prediction, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    if prediction64.shape != target64.shape:
        raise ValueError("correction prediction and target shapes differ")
    numerator = float(np.sum(prediction64 * target64, dtype=np.float64))
    denominator = float(np.sum(np.square(prediction64), dtype=np.float64))
    return float(np.clip(numerator / max(denominator, 1.0e-30), 0.0, 1.5))


def _batches(rows: np.ndarray, batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start:min(start + batch_size, len(rows))]


def _normalized_batch(
    values: np.ndarray, rows: np.ndarray, mean: np.ndarray, scale: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    normalized = ((values[rows] - mean) / scale).astype(np.float32)
    return torch.from_numpy(normalized).to(device)


def _loss(
    prediction: tuple[torch.Tensor, ...], target: torch.Tensor, shape_width: int,
    coordinate_weight: float, scalar_weight: float,
) -> torch.Tensor:
    shape, coordinate, scalar = prediction
    shape_loss = torch.mean(torch.square(shape - target[:, :shape_width]))
    coordinate_loss = torch.mean(torch.square(
        coordinate - target[:, shape_width:shape_width + 2]
    ))
    scalar_loss = torch.mean(torch.square(scalar - target[:, shape_width + 2:]))
    return shape_loss + coordinate_weight * coordinate_loss + scalar_weight * scalar_loss


def _train_epoch(
    model: OOFResidualHead,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    inputs: np.ndarray,
    targets: np.ndarray,
    rows: np.ndarray,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    shape_width: int,
    batch_size: int,
    coordinate_weight: float,
    scalar_weight: float,
    device: torch.device,
    seed: int,
) -> float:
    model.train()
    shuffled = np.random.default_rng(seed).permutation(rows)
    total = 0.0
    count = 0
    for batch_rows in _batches(shuffled, batch_size):
        x = _normalized_batch(inputs, batch_rows, input_mean, input_scale, device)
        y = _normalized_batch(targets, batch_rows, target_mean, target_scale, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = _loss(
                model(x), y, shape_width, coordinate_weight, scalar_weight,
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("residual training produced a non-finite loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * len(batch_rows)
        count += len(batch_rows)
    return total / max(count, 1)


@torch.no_grad()
def _predict(
    model: OOFResidualHead,
    inputs: np.ndarray,
    rows: np.ndarray,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts = []
    for batch_rows in _batches(rows, batch_size):
        x = _normalized_batch(inputs, batch_rows, input_mean, input_scale, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            predicted = model(x)
        normalized = torch.cat(predicted, dim=1).float().cpu().numpy()
        parts.append(target_mean + target_scale * normalized)
    result = np.concatenate(parts).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("residual prediction contains non-finite values")
    return result


def _statistics(values: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values[rows].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values[rows].std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale, 1.0e-6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--coordinate-weight", type=float, default=0.50)
    parser.add_argument("--scalar-weight", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=260917)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch size must be positive")
    started = time.monotonic()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    required = (
        "shot_name", "features", "target_shape", "target_coordinate",
        "target_scalar", "base_shape", "base_coordinate", "base_scalar",
    )
    with np.load(args.oof, allow_pickle=False) as saved:
        missing = [key for key in required if key not in saved]
        if missing:
            raise RuntimeError(f"OOF archive is missing {missing}")
        for key in ("same_shot_targets_used", "development_used", "frozen_audit_used"):
            if key not in saved or bool(saved[key].item()):
                raise RuntimeError(f"unsafe OOF provenance: {key}")
        names = np.asarray(saved["shot_name"]).astype(str)
        features = np.asarray(saved["features"], dtype=np.float32)
        target_shape = np.asarray(saved["target_shape"], dtype=np.float32)
        target_coordinate = np.asarray(saved["target_coordinate"], dtype=np.float32)
        target_scalar = np.asarray(saved["target_scalar"], dtype=np.float32)
        base_shape = np.asarray(saved["base_shape"], dtype=np.float32)
        base_coordinate = np.asarray(saved["base_coordinate"], dtype=np.float32)
        base_scalar = np.asarray(saved["base_scalar"], dtype=np.float32)
    row_count = len(names)
    if len(np.unique(names)) != 5677:
        raise RuntimeError("OOF archive does not cover all 5,677 training shots")
    if any(len(value) != row_count for value in (
        features, target_shape, target_coordinate, target_scalar,
        base_shape, base_coordinate, base_scalar,
    )):
        raise RuntimeError("OOF arrays have inconsistent row counts")
    inputs = np.column_stack((
        features, base_shape, base_coordinate, base_scalar,
    )).astype(np.float32)
    residual = np.column_stack((
        target_shape - base_shape,
        target_coordinate - base_coordinate,
        target_scalar - base_scalar,
    )).astype(np.float32)
    del features, target_shape, target_coordinate, target_scalar
    del base_shape, base_coordinate, base_scalar
    if not np.isfinite(inputs).all() or not np.isfinite(residual).all():
        raise RuntimeError("OOF residual training arrays contain non-finite values")
    shape_width = residual.shape[1] - 4
    training, validation = split_validation_shots(
        names, args.validation_fraction, args.seed,
    )
    train_rows = np.flatnonzero(training)
    validation_rows = np.flatnonzero(validation)
    input_mean, input_scale = _statistics(inputs, train_rows)
    target_mean, target_scale = _statistics(residual, train_rows)
    print(
        f"OOF rows={row_count} shots={len(np.unique(names))} "
        f"train_rows={len(train_rows)} validation_rows={len(validation_rows)} "
        f"input_width={inputs.shape[1]} shape_width={shape_width}",
        flush=True,
    )

    model = OOFResidualHead(
        inputs.shape[1], shape_width, args.width, args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = np.inf
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(
            model, optimizer, scaler, inputs, residual, train_rows,
            input_mean, input_scale, target_mean, target_scale, shape_width,
            args.batch_size, args.coordinate_weight, args.scalar_weight,
            device, args.seed + epoch,
        )
        validation_prediction = _predict(
            model, inputs, validation_rows, input_mean, input_scale,
            target_mean, target_scale, args.batch_size, device,
        )
        validation_loss = float(np.mean(np.square(
            validation_prediction[:, :shape_width]
            - residual[validation_rows, :shape_width]
        ), dtype=np.float64))
        epoch_report = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_shape_residual_mse": validation_loss,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report), flush=True)
        if validation_loss < best_loss - 1.0e-12:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("residual model selection produced no checkpoint")
    model.load_state_dict(best_state)
    validation_prediction = _predict(
        model, inputs, validation_rows, input_mean, input_scale,
        target_mean, target_scale, args.batch_size, device,
    )
    validation_target = residual[validation_rows]
    strengths = {
        "shape": correction_strength(
            validation_prediction[:, :shape_width], validation_target[:, :shape_width],
        ),
        "coordinate": correction_strength(
            validation_prediction[:, shape_width:shape_width + 2],
            validation_target[:, shape_width:shape_width + 2],
        ),
        "scalar": correction_strength(
            validation_prediction[:, shape_width + 2:],
            validation_target[:, shape_width + 2:],
        ),
    }
    del model, optimizer, scaler, best_state, validation_prediction
    torch.cuda.empty_cache()

    # The selection split chooses only the epoch count and calibration scales.
    # Recompute normalization and fit the deployable state on every train shot.
    all_rows = np.arange(row_count, dtype=np.int64)
    input_mean, input_scale = _statistics(inputs, all_rows)
    target_mean, target_scale = _statistics(residual, all_rows)
    torch.manual_seed(args.seed + 100000)
    final_model = OOFResidualHead(
        inputs.shape[1], shape_width, args.width, args.dropout,
    ).to(device)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    final_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    final_history = []
    for epoch in range(1, best_epoch + 1):
        train_loss = _train_epoch(
            final_model, final_optimizer, final_scaler, inputs, residual, all_rows,
            input_mean, input_scale, target_mean, target_scale, shape_width,
            args.batch_size, args.coordinate_weight, args.scalar_weight,
            device, args.seed + 100000 + epoch,
        )
        final_history.append({"epoch": epoch, "train_loss": train_loss})
        print(json.dumps({"final_epoch": epoch, "train_loss": train_loss}), flush=True)

    artifact = {
        "state_dict": {
            key: value.detach().cpu() for key, value in final_model.state_dict().items()
        },
        "input_mean": torch.from_numpy(input_mean),
        "input_scale": torch.from_numpy(input_scale),
        "target_mean": torch.from_numpy(target_mean),
        "target_scale": torch.from_numpy(target_scale),
        "input_width": int(inputs.shape[1]),
        "shape_width": int(shape_width),
        "width": int(args.width),
        "dropout": float(args.dropout),
        "correction_strength": strengths,
        "data_policy": "real_diii_d_only",
        "training_shots": int(len(np.unique(names))),
        "training_rows": int(row_count),
        "shot_oof_base_predictions": True,
        "all_train_shots_used_for_final_fit": True,
        "development_used": False,
        "frozen_audit_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    report = {
        "output": str(args.output.resolve()),
        "oof": str(args.oof.resolve()),
        "training_shots": artifact["training_shots"],
        "training_rows": row_count,
        "input_width": int(inputs.shape[1]),
        "shape_width": int(shape_width),
        "selected_epochs": best_epoch,
        "best_validation_shape_residual_mse": best_loss,
        "correction_strength": strengths,
        "selection_history": history,
        "final_history": final_history,
        "shot_oof_base_predictions": True,
        "all_train_shots_used_for_final_fit": True,
        "development_used": False,
        "frozen_audit_used": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
