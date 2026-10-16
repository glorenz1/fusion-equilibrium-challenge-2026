#!/usr/bin/env python3
"""Train the train-only, whole-shot, score-aligned FEC1 residual stack.

Model selection is shot-disjoint.  The field checkpoint is selected with the
exact official scorer at a configurable interval; differentiable normalized-
flux, boundary, gradient, curvature, axis, Fourier-LCFS, and topology losses
provide the training signal. q95 and betaN have independent temporal models,
validation criteria, calibration, and final refits.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_experiments.fec1_score_aligned_temporal import (  # noqa: E402
    CONSISTENCY_NAMES,
    ScoreAlignedTemporalResidual,
    TemporalScalarResidual,
    add_spatial_residual,
    decode_pca_field,
    pooled_r2,
    scalar_temporal_loss,
    score_aligned_field_loss,
    valid_time_mask,
)
from my_experiments.fit_fec1_train_oof_residual import (  # noqa: E402
    correction_strength,
    split_validation_shots,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _statistics(values: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values[rows].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values[rows].std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale, 1.0e-6)


def _align_topology(
    path: Path, names: np.ndarray, frames: np.ndarray,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as saved:
        for key, expected in (
            ("data_policy", "real_diii_d_only"),
            ("development_used", False),
            ("frozen_audit_used", False),
        ):
            if key not in saved.files or saved[key].item() != expected:
                raise RuntimeError(f"unsafe topology provenance: {key}")
        if saved["names"].astype(str).tolist() != list(CONSISTENCY_NAMES):
            raise RuntimeError("topology target column order changed")
        source_names = saved["shot_name"].astype(str)
        source_frames = saved["frame_index"].astype(np.int64)
        source_target = saved["target"].astype(np.float32)
    if np.array_equal(source_names, names) and np.array_equal(source_frames, frames):
        return source_target
    lookup = {
        (shot, int(frame)): index
        for index, (shot, frame) in enumerate(zip(source_names, source_frames))
    }
    try:
        rows = np.asarray(
            [lookup[(shot, int(frame))] for shot, frame in zip(names, frames)],
            dtype=np.int64,
        )
    except KeyError as error:
        raise RuntimeError("topology archive misses an OOF shot/frame") from error
    return source_target[rows]


def _align_contours(
    path: Path, names: np.ndarray, frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Align train-only exact LCFS radii to the OOF row order."""
    with np.load(path, allow_pickle=False) as saved:
        for key, expected in (
            ("data_policy", "real_diii_d_only"),
            ("frozen_audit_used", False),
        ):
            if key not in saved.files or saved[key].item() != expected:
                raise RuntimeError(f"unsafe contour provenance: {key}")
        if "split" in saved.files and np.any(saved["split"].astype(str) != "train"):
            raise RuntimeError("non-train rows entered exact contour supervision")
        source_names = saved["shot_name"].astype(str)
        source_frames = saved["frame_index"].astype(np.int64)
        source_radius = saved["radius"].astype(np.float32)
        angles = saved["angles"].astype(np.float32)
        source_offset = (
            saved["contour_offset"].astype(np.float32)
            if "contour_offset" in saved.files else None
        )
    if np.array_equal(source_names, names) and np.array_equal(source_frames, frames):
        return source_radius, angles, source_offset
    lookup = {
        (shot, int(frame)): index
        for index, (shot, frame) in enumerate(zip(source_names, source_frames))
    }
    try:
        rows = np.asarray(
            [lookup[(shot, int(frame))] for shot, frame in zip(names, frames)],
            dtype=np.int64,
        )
    except KeyError as error:
        raise RuntimeError("contour archive misses an OOF shot/frame") from error
    return (
        source_radius[rows],
        angles,
        source_offset[rows] if source_offset is not None else None,
    )


def _align_multirate_context(
    path: Path, names: np.ndarray, frames: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Align target-free full-rate context and return its frozen projector."""
    with np.load(path, allow_pickle=False) as saved:
        for key, expected in (
            ("data_policy", "real_diii_d_only"),
            ("input_only", True),
            ("development_used", False),
            ("frozen_audit_used", False),
        ):
            if key not in saved.files or saved[key].item() != expected:
                raise RuntimeError(f"unsafe multirate context provenance: {key}")
        source_names = saved["shot_name"].astype(str)
        source_frames = saved["frame_index"].astype(np.int64)
        source_context = saved["context"].astype(np.float32)
        projector = {
            key: np.asarray(saved[key]).copy()
            for key in (
                "projection_mean", "projection_scale", "projection_components",
                "offsets", "window_radii", "program_statistics",
            )
        }
    if np.array_equal(source_names, names) and np.array_equal(source_frames, frames):
        return source_context, projector
    lookup = {
        (shot, int(frame)): index
        for index, (shot, frame) in enumerate(zip(source_names, source_frames))
    }
    try:
        rows = np.asarray(
            [lookup[(shot, int(frame))] for shot, frame in zip(names, frames)],
            dtype=np.int64,
        )
    except KeyError as error:
        raise RuntimeError("multirate context misses an OOF shot/frame") from error
    return source_context[rows], projector


def _sequences(
    names: np.ndarray, frames: np.ndarray, selected: np.ndarray,
) -> list[np.ndarray]:
    result = []
    for shot in np.unique(names[selected]):
        rows = np.flatnonzero(selected & (names == shot))
        result.append(rows[np.argsort(frames[rows])])
    return result


class ShotDataset(Dataset):
    def __init__(
        self, arrays: dict[str, np.ndarray], psi_n_path: Path,
        sequences: list[np.ndarray],
    ) -> None:
        self.arrays = arrays
        self.psi_n_path = psi_n_path
        self.sequences = sequences
        self._psi_n = None

    @property
    def psi_n(self):
        if self._psi_n is None:
            self._psi_n = np.load(self.psi_n_path, mmap_mode="r")
        return self._psi_n

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rows = self.sequences[index]
        item = {
            key: torch.from_numpy(np.asarray(value[rows]).copy())
            for key, value in self.arrays.items()
        }
        item["target_psi_n"] = torch.from_numpy(
            np.asarray(self.psi_n[rows], dtype=np.float32)
        )
        item["rows"] = torch.from_numpy(rows.astype(np.int64))
        return item


def _collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    lengths = torch.as_tensor([len(item["rows"]) for item in items], dtype=torch.int64)
    width = int(lengths.max())
    result: dict[str, torch.Tensor] = {"lengths": lengths}
    for key in items[0]:
        tail = items[0][key].shape[1:]
        dtype = items[0][key].dtype
        fill = -1 if key == "rows" else 0
        padded = torch.full((len(items), width, *tail), fill, dtype=dtype)
        for batch, item in enumerate(items):
            padded[batch, : len(item[key])] = item[key]
        result[key] = padded
    return result


def _loader(
    arrays: dict[str, np.ndarray], psi_n_path: Path,
    sequences: list[np.ndarray], batch_shots: int, workers: int, shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ShotDataset(arrays, psi_n_path, sequences),
        batch_size=batch_shots,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _to_device(batch: dict[str, torch.Tensor], device: torch.device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def _flatten_output(
    output: dict[str, torch.Tensor], mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {key: value[mask] for key, value in output.items()}


def _field_batch(
    model: ScoreAlignedTemporalResidual,
    batch: dict[str, torch.Tensor],
    components: torch.Tensor,
    pca_mean: torch.Tensor,
    topology_mean: torch.Tensor,
    topology_scale: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    contour_angles: torch.Tensor | None = None,
    objective_weights: dict[str, float] | None = None,
    consistency_channel_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    output_sequence = model(batch["input"], batch["lengths"])
    mask = valid_time_mask(batch["lengths"], batch["input"].shape[1])
    output = _flatten_output(output_sequence, mask)
    base_shape = batch["base_shape"][mask]
    base_coordinate = batch["base_coordinate"][mask]
    target_shape = batch["target_shape"][mask]
    target_coordinate = batch["target_coordinate"][mask]
    predicted_shape = base_shape + (
        batch["shape_residual_mean"][mask]
        + batch["shape_residual_scale"][mask] * output["shape"]
    )
    predicted_coordinate = base_coordinate + (
        batch["coordinate_residual_mean"][mask]
        + batch["coordinate_residual_scale"][mask] * output["coordinate"]
    )
    predicted_field, _ = decode_pca_field(
        predicted_shape, predicted_coordinate, components, pca_mean
    )
    predicted_field = add_spatial_residual(
        predicted_field,
        predicted_coordinate,
        output["spatial"],
        model.dct_order,
    )
    target_field, _ = decode_pca_field(
        target_shape, target_coordinate, components, pca_mean
    )
    total, parts = score_aligned_field_loss(
        output,
        predicted_field,
        target_field,
        batch["target_psi_n"][mask],
        batch["topology"][mask],
        topology_mean,
        topology_scale,
        grid_r,
        grid_z,
        target_shape_residual_normalized=batch["shape_residual_normalized"][mask],
        target_coordinate_residual_normalized=(
            batch["coordinate_residual_normalized"][mask]
        ),
        exact_target_radius=(
            batch["contour_radius"][mask]
            if "contour_radius" in batch else None
        ),
        contour_angles=contour_angles,
        exact_target_contour_offset=(
            batch["contour_offset"][mask]
            if "contour_offset" in batch else None
        ),
        objective_weights=objective_weights,
        consistency_channel_weights=consistency_channel_weights,
    )
    return total, parts, int(mask.sum())


def _run_field_epoch(
    model: ScoreAlignedTemporalResidual,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    components: torch.Tensor,
    pca_mean: torch.Tensor,
    topology_mean: torch.Tensor,
    topology_scale: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    device: torch.device,
    contour_angles: torch.Tensor | None = None,
    objective_weights: dict[str, float] | None = None,
    consistency_channel_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    frames = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for host_batch in loader:
            batch = _to_device(host_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss, parts, count = _field_batch(
                    model, batch, components, pca_mean, topology_mean,
                    topology_scale, grid_r, grid_z, contour_angles,
                    objective_weights,
                    consistency_channel_weights,
                )
            if not bool(torch.isfinite(loss)):
                detail = {key: float(value.detach()) for key, value in parts.items()}
                raise RuntimeError(f"non-finite score-aligned field loss: {detail}")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * count
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * count
            frames += count
    return {key: value / max(frames, 1) for key, value in totals.items()}


@torch.no_grad()
def _predict_field_rows(
    model: ScoreAlignedTemporalResidual,
    loader: DataLoader,
    components: torch.Tensor,
    pca_mean: torch.Tensor,
    device: torch.device,
    residual_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if residual_scale < 0.0:
        raise ValueError("field residual scale must be nonnegative")
    model.eval()
    row_parts, field_parts = [], []
    for host_batch in loader:
        batch = _to_device(host_batch, device)
        output_sequence = model(batch["input"], batch["lengths"])
        mask = valid_time_mask(batch["lengths"], batch["input"].shape[1])
        output = _flatten_output(output_sequence, mask)
        shape = batch["base_shape"][mask] + residual_scale * (
            batch["shape_residual_mean"][mask]
            + batch["shape_residual_scale"][mask] * output["shape"]
        )
        coordinate = batch["base_coordinate"][mask] + residual_scale * (
            batch["coordinate_residual_mean"][mask]
            + batch["coordinate_residual_scale"][mask] * output["coordinate"]
        )
        field, _ = decode_pca_field(shape, coordinate, components, pca_mean)
        field = add_spatial_residual(
            field,
            coordinate,
            residual_scale * output["spatial"],
            model.dct_order,
        )
        row_parts.append(batch["rows"][mask].cpu().numpy())
        field_parts.append(field.float().cpu().numpy())
    return np.concatenate(row_parts), np.concatenate(field_parts)


def _exact_field_score(
    model: ScoreAlignedTemporalResidual,
    loader: DataLoader,
    arrays: dict[str, np.ndarray],
    names: np.ndarray,
    frames: np.ndarray,
    cache_dir: Path,
    components: torch.Tensor,
    pca_mean: torch.Tensor,
    device: torch.device,
    workers: int,
    residual_scale: float = 1.0,
) -> dict[str, object]:
    from my_experiments.score_fec1_prediction_archive import _parallel_official_score

    rows, prediction = _predict_field_rows(
        model, loader, components, pca_mean, device, residual_scale
    )
    order = np.argsort(rows)
    rows, prediction = rows[order], prediction[order]
    target = np.empty_like(prediction, dtype=np.float32)
    q95 = arrays["target_scalar"][rows, 0].astype(np.float32)
    beta_n = arrays["target_scalar"][rows, 1].astype(np.float32)
    local_names = names[rows]
    local_frames = frames[rows]
    shot_index = np.empty(len(rows), dtype=np.int64)
    for index, shot in enumerate(np.unique(local_names)):
        selected = np.flatnonzero(local_names == shot)
        with np.load(cache_dir / f"{shot}.npz", allow_pickle=False) as saved:
            target[selected] = np.asarray(saved["target_psi"])[local_frames[selected]]
        shot_index[selected] = index
    values = {
        "field": prediction,
        "target_field": target,
        "q95": q95,
        "target_q95": q95,
        "betaN": beta_n,
        "target_betaN": beta_n,
        "shot_index": shot_index,
        "source_index": local_frames.astype(np.int64),
        "shot_name": local_names,
        "reference_source": str(cache_dir.resolve()),
    }
    return _parallel_official_score(values, max(1, workers))


def _field_calibration(
    model: ScoreAlignedTemporalResidual,
    loader: DataLoader,
    arrays: dict[str, np.ndarray],
    names: np.ndarray,
    frames: np.ndarray,
    cache_dir: Path,
    components: torch.Tensor,
    pca_mean: torch.Tensor,
    device: torch.device,
    workers: int,
    scales: list[float],
) -> tuple[float, list[dict[str, object]]]:
    """Choose correction strength using only shot-disjoint OOF validation."""
    reports = []
    for scale in scales:
        score = _exact_field_score(
            model, loader, arrays, names, frames, cache_dir,
            components, pca_mean, device, workers, scale,
        )
        reports.append({"scale": scale, "score": score})
        print(json.dumps({"stage": "field_calibration", **reports[-1]}), flush=True)
    selected = max(reports, key=lambda item: float(item["score"]["S"]))
    return float(selected["scale"]), reports


def _run_scalar_epoch(
    model: TemporalScalarResidual,
    loader: DataLoader,
    key: str,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    losses, targets, predictions = [], [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for host_batch in loader:
            batch = _to_device(host_batch, device)
            valid = valid_time_mask(batch["lengths"], batch["input"].shape[1])
            if training:
                optimizer.zero_grad(set_to_none=True)
            predicted_normalized = model(batch["input"], batch["lengths"])
            target_normalized = batch[f"{key}_residual_normalized"]
            loss = scalar_temporal_loss(
                predicted_normalized, target_normalized, valid
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite {key} temporal scalar loss")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
            corrected = batch[f"base_{key}"] + (
                batch[f"{key}_residual_mean"]
                + batch[f"{key}_residual_scale"] * predicted_normalized
            )
            losses.append(float(loss.detach()) * int(valid.sum()))
            targets.append(batch[f"target_{key}"][valid].detach().cpu())
            predictions.append(corrected[valid].detach().cpu())
    target = torch.cat(targets)
    prediction = torch.cat(predictions)
    return sum(losses) / max(len(target), 1), float(pooled_r2(target, prediction))


@torch.no_grad()
def _scalar_calibration(
    model: TemporalScalarResidual,
    loader: DataLoader,
    key: str,
    device: torch.device,
) -> float:
    predicted, target = [], []
    model.eval()
    for host_batch in loader:
        batch = _to_device(host_batch, device)
        valid = valid_time_mask(batch["lengths"], batch["input"].shape[1])
        normalized = model(batch["input"], batch["lengths"])
        residual = (
            batch[f"{key}_residual_mean"]
            + batch[f"{key}_residual_scale"] * normalized
        )
        predicted.append(residual[valid].cpu().numpy())
        target.append(
            (batch[f"target_{key}"] - batch[f"base_{key}"])[valid].cpu().numpy()
        )
    return correction_strength(np.concatenate(predicted), np.concatenate(target))


def _fit_scalar(
    key: str,
    feature_width: int,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    hidden_width: int,
    layers: int,
    dropout: float,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], int, float, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = TemporalScalarResidual(
        feature_width, hidden_width, layers, dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_r2 = -np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_r2 = _run_scalar_epoch(
            model, train_loader, key, optimizer, device
        )
        val_loss, val_r2 = _run_scalar_epoch(
            model, validation_loader, key, None, device
        )
        record = {
            "epoch": epoch, "train_loss": train_loss, "train_r2": train_r2,
            "validation_loss": val_loss, "validation_r2": val_r2,
        }
        history.append(record)
        print(json.dumps({"stage": key, **record}), flush=True)
        if val_r2 > best_r2 + 1.0e-7:
            best_r2 = val_r2
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"{key} model selection produced no checkpoint")
    model.load_state_dict(best_state)
    strength = _scalar_calibration(model, validation_loader, key, device)
    return best_state, best_epoch, strength, history


def _refit_field(
    feature_width: int,
    shape_width: int,
    loader: DataLoader,
    epochs: int,
    initial_state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    model = ScoreAlignedTemporalResidual(
        feature_width, shape_width, args.hidden_width, args.layers,
        args.dropout, args.fourier_modes, args.profile_modes,
        args.dct_order, args.separate_decoder_branches,
    ).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    for epoch in range(1, epochs + 1):
        metrics = _run_field_epoch(
            model, loader, optimizer, tensors["components"], tensors["pca_mean"],
            tensors["topology_mean"], tensors["topology_scale"],
            tensors["grid_r"], tensors["grid_z"], device,
            tensors.get("contour_angles"),
            tensors["objective_weights"],
            tensors["consistency_channel_weights"],
        )
        history.append({"epoch": epoch, **metrics})
        print(json.dumps({"stage": "field_final", "epoch": epoch, **metrics}), flush=True)
    return deepcopy(model.state_dict()), history


def _refit_scalar(
    key: str,
    feature_width: int,
    loader: DataLoader,
    epochs: int,
    initial_state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    model = TemporalScalarResidual(
        feature_width, args.scalar_hidden_width, args.scalar_layers, args.dropout
    ).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    for epoch in range(1, epochs + 1):
        loss, r2 = _run_scalar_epoch(model, loader, key, optimizer, device)
        history.append({"epoch": epoch, "loss": loss, "r2": r2})
        print(json.dumps({"stage": f"{key}_final", "epoch": epoch, "loss": loss, "r2": r2}), flush=True)
    return deepcopy(model.state_dict()), history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--target-psi-n", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument(
        "--contours", type=Path,
        help="optional exact train-only LCFS radius archive",
    )
    parser.add_argument(
        "--multirate-context", type=Path,
        help="optional target-free adjacent-frame and shot-program context",
    )
    parser.add_argument("--pca-basis", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=56)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-shots", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hidden-width", type=int, default=384)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--scalar-hidden-width", type=int, default=192)
    parser.add_argument("--scalar-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--fourier-modes", type=int, default=8)
    parser.add_argument("--profile-modes", type=int, default=6)
    parser.add_argument("--dct-order", type=int, default=6)
    parser.add_argument(
        "--separate-decoder-branches", action="store_true",
        help="use independent boundary, interior-profile, and field decoders",
    )
    parser.add_argument("--field-objective-weight", type=float, default=1.0)
    parser.add_argument("--boundary-objective-weight", type=float, default=1.0)
    parser.add_argument("--consistency-objective-weight", type=float, default=1.0)
    parser.add_argument("--profile-objective-weight", type=float, default=1.0)
    parser.add_argument(
        "--consistency-channel-weights", default="1,1,1,1,1,1,1",
        help=(
            "comma-separated weights for R_axis,Z_axis,kappa,tri_top,"
            "tri_bot,volume,li"
        ),
    )
    parser.add_argument(
        "--field-calibration-scales", default="0,0.25,0.5,0.75,1.0,1.25",
        help="comma-separated residual strengths selected on shot-disjoint validation",
    )
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--exact-score-every", type=int, default=4)
    parser.add_argument("--exact-score-shots", type=int, default=24)
    parser.add_argument("--exact-score-workers", type=int, default=4)
    parser.add_argument(
        "--final-refit-epochs", type=int, default=4,
        help="fine-tuning epochs on all train shots after held-out selection",
    )
    parser.add_argument("--seed", type=int, default=260917)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-shots", type=int, default=0, help="smoke-test only")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_shots < 1 or args.exact_score_every < 1:
        parser.error("epochs, batch size, and exact-score interval must be positive")
    objective_weights = {
        "field": args.field_objective_weight,
        "boundary": args.boundary_objective_weight,
        "consistency": args.consistency_objective_weight,
        "profile": args.profile_objective_weight,
    }
    if any(not np.isfinite(value) or value < 0.0 for value in objective_weights.values()):
        parser.error("objective weights must be finite and nonnegative")
    if not any(value > 0.0 for value in objective_weights.values()):
        parser.error("at least one objective weight must be positive")
    try:
        consistency_channel_weights = np.asarray(
            [float(value) for value in args.consistency_channel_weights.split(",")],
            dtype=np.float32,
        )
    except ValueError:
        parser.error("consistency channel weights must be comma-separated numbers")
    if (
        consistency_channel_weights.shape != (len(CONSISTENCY_NAMES),)
        or not np.isfinite(consistency_channel_weights).all()
        or np.any(consistency_channel_weights < 0.0)
        or not np.any(consistency_channel_weights > 0.0)
    ):
        parser.error("consistency channel weights must be seven nonnegative finite values")
    try:
        field_calibration_scales = sorted({
            float(value) for value in args.field_calibration_scales.split(",")
            if value.strip()
        })
    except ValueError:
        parser.error("field calibration scales must be comma-separated numbers")
    if (
        not field_calibration_scales
        or any(not np.isfinite(value) or value < 0.0 for value in field_calibration_scales)
    ):
        parser.error("field calibration scales must be finite and nonnegative")
    started = time.monotonic()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    required = (
        "shot_name", "frame_index", "features", "target_shape",
        "target_coordinate", "target_scalar", "base_shape",
        "base_coordinate", "base_scalar",
    )
    with np.load(args.oof, allow_pickle=False) as saved:
        for key, expected in (
            ("data_policy", "real_diii_d_only"),
            ("same_shot_targets_used", False),
            ("development_used", False),
            ("frozen_audit_used", False),
        ):
            if key not in saved.files or saved[key].item() != expected:
                raise RuntimeError(f"unsafe OOF provenance: {key}")
        values = {key: np.asarray(saved[key]) for key in required}
        train_stride = int(saved["train_stride"].item())
    names = values["shot_name"].astype(str)
    frames = values["frame_index"].astype(np.int64)
    if args.max_shots > 0:
        keep_shots = np.unique(names)[: args.max_shots]
        keep = np.isin(names, keep_shots)
        values = {key: value[keep] for key, value in values.items()}
        names, frames = names[keep], frames[keep]
    topology = _align_topology(args.topology, names, frames)
    contour_radius = contour_angles_np = contour_offset = None
    if args.contours is not None:
        contour_radius, contour_angles_np, contour_offset = _align_contours(
            args.contours, names, frames
        )
    multirate_values = multirate_projector = None
    if args.multirate_context is not None:
        multirate_values, multirate_projector = _align_multirate_context(
            args.multirate_context, names, frames
        )
    target_psi_n = np.load(args.target_psi_n, mmap_mode="r")
    if args.max_shots <= 0 and len(target_psi_n) != len(names):
        raise RuntimeError("normalized-flux target rows differ from OOF rows")
    if args.max_shots > 0:
        # Smoke rows are a prefix because shots are selected in sorted OOF order.
        original_names = np.load(args.oof, allow_pickle=False)["shot_name"].astype(str)
        original_rows = np.flatnonzero(np.isin(original_names, np.unique(names)))
        smoke_path = args.target_psi_n.with_name(args.target_psi_n.stem + ".smoke.npy")
        smoke = np.lib.format.open_memmap(
            smoke_path, mode="w+", dtype=np.float16, shape=(len(names), 65, 65)
        )
        smoke[:] = target_psi_n[original_rows]
        smoke.flush()
        del smoke
        target_psi_n_path = smoke_path
    else:
        target_psi_n_path = args.target_psi_n

    with np.load(args.pca_basis, allow_pickle=False) as saved:
        for key, expected in (("development_used", False), ("frozen_audit_used", False)):
            if saved[key].item() != expected:
                raise RuntimeError(f"unsafe PCA basis provenance: {key}")
        components_np = saved["components"].astype(np.float32)
        pca_mean_np = saved["mean"].astype(np.float32)
        grid_r_np = saved["grid_r"].astype(np.float32)
        grid_z_np = saved["grid_z"].astype(np.float32)

    inputs = np.column_stack((
        values["features"], values["base_shape"],
        values["base_coordinate"], values["base_scalar"],
    )).astype(np.float32)
    if multirate_values is not None:
        inputs = np.column_stack((inputs, multirate_values)).astype(np.float32)
    training_mask, validation_mask = split_validation_shots(
        names, args.validation_fraction, args.seed
    )
    train_rows = np.flatnonzero(training_mask)
    validation_rows = np.flatnonzero(validation_mask)
    input_mean, input_scale = _statistics(inputs, train_rows)
    shape_residual = (values["target_shape"] - values["base_shape"]).astype(np.float32)
    coordinate_residual = (
        values["target_coordinate"] - values["base_coordinate"]
    ).astype(np.float32)
    scalar_residual = (values["target_scalar"] - values["base_scalar"]).astype(np.float32)
    shape_mean, shape_scale = _statistics(shape_residual, train_rows)
    coordinate_mean, coordinate_scale = _statistics(coordinate_residual, train_rows)
    scalar_mean, scalar_scale = _statistics(scalar_residual, train_rows)
    topology_mean = np.nanmean(topology[train_rows], axis=0).astype(np.float32)
    topology_scale = np.nanstd(topology[train_rows], axis=0).astype(np.float32)
    topology_scale = np.maximum(topology_scale, 1.0e-5)
    arrays = {
        "input": ((inputs - input_mean) / input_scale).astype(np.float32),
        "base_shape": values["base_shape"].astype(np.float32),
        "base_coordinate": values["base_coordinate"].astype(np.float32),
        "target_shape": values["target_shape"].astype(np.float32),
        "target_coordinate": values["target_coordinate"].astype(np.float32),
        "topology": topology.astype(np.float32),
        "shape_residual_normalized": ((shape_residual - shape_mean) / shape_scale).astype(np.float32),
        "coordinate_residual_normalized": ((coordinate_residual - coordinate_mean) / coordinate_scale).astype(np.float32),
        "shape_residual_mean": np.broadcast_to(shape_mean, shape_residual.shape).astype(np.float32),
        "shape_residual_scale": np.broadcast_to(shape_scale, shape_residual.shape).astype(np.float32),
        "coordinate_residual_mean": np.broadcast_to(coordinate_mean, coordinate_residual.shape).astype(np.float32),
        "coordinate_residual_scale": np.broadcast_to(coordinate_scale, coordinate_residual.shape).astype(np.float32),
    }
    if contour_radius is not None:
        arrays["contour_radius"] = contour_radius.astype(np.float32)
    if contour_offset is not None:
        arrays["contour_offset"] = contour_offset.astype(np.float32)
    for column, key in enumerate(("q95", "betaN")):
        arrays[f"base_{key}"] = values["base_scalar"][:, column].astype(np.float32)
        arrays[f"target_{key}"] = values["target_scalar"][:, column].astype(np.float32)
        arrays[f"{key}_residual_normalized"] = (
            (scalar_residual[:, column] - scalar_mean[column]) / scalar_scale[column]
        ).astype(np.float32)
        arrays[f"{key}_residual_mean"] = np.full(len(names), scalar_mean[column], dtype=np.float32)
        arrays[f"{key}_residual_scale"] = np.full(len(names), scalar_scale[column], dtype=np.float32)

    train_sequences = _sequences(names, frames, training_mask)
    validation_sequences = _sequences(names, frames, validation_mask)
    validation_shots = np.unique(names[validation_mask])
    exact_shots = np.random.default_rng(args.seed + 70001).permutation(
        validation_shots
    )[: args.exact_score_shots]
    exact_mask = validation_mask & np.isin(names, exact_shots)
    exact_sequences = _sequences(names, frames, exact_mask)
    all_sequences = _sequences(names, frames, np.ones(len(names), dtype=bool))
    loader_args = (arrays, target_psi_n_path)
    train_loader = _loader(*loader_args, train_sequences, args.batch_shots, args.workers, True)
    validation_loader = _loader(*loader_args, validation_sequences, args.batch_shots, args.workers, False)
    exact_loader = _loader(*loader_args, exact_sequences, args.batch_shots, args.workers, False)
    all_loader = _loader(*loader_args, all_sequences, args.batch_shots, args.workers, True)
    tensors = {
        "components": torch.from_numpy(components_np).to(device),
        "pca_mean": torch.from_numpy(pca_mean_np).to(device),
        "topology_mean": torch.from_numpy(topology_mean).to(device),
        "topology_scale": torch.from_numpy(topology_scale).to(device),
        "grid_r": torch.from_numpy(grid_r_np).to(device),
        "grid_z": torch.from_numpy(grid_z_np).to(device),
        "objective_weights": objective_weights,
        "consistency_channel_weights": torch.from_numpy(
            consistency_channel_weights
        ).to(device),
    }
    if contour_angles_np is not None:
        tensors["contour_angles"] = torch.from_numpy(contour_angles_np).to(device)

    feature_width = inputs.shape[1]
    shape_width = values["target_shape"].shape[1]
    field_model = ScoreAlignedTemporalResidual(
        feature_width, shape_width, args.hidden_width, args.layers,
        args.dropout, args.fourier_modes, args.profile_modes,
        args.dct_order, args.separate_decoder_branches,
    ).to(device)
    field_optimizer = torch.optim.AdamW(
        field_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    field_history = []
    best_score = -np.inf
    best_field_epoch = 0
    best_field_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_field_epoch(
            field_model, train_loader, field_optimizer,
            tensors["components"], tensors["pca_mean"], tensors["topology_mean"],
            tensors["topology_scale"], tensors["grid_r"], tensors["grid_z"], device,
            tensors.get("contour_angles"),
            tensors["objective_weights"],
            tensors["consistency_channel_weights"],
        )
        validation_metrics = _run_field_epoch(
            field_model, validation_loader, None,
            tensors["components"], tensors["pca_mean"], tensors["topology_mean"],
            tensors["topology_scale"], tensors["grid_r"], tensors["grid_z"], device,
            tensors.get("contour_angles"),
            tensors["objective_weights"],
            tensors["consistency_channel_weights"],
        )
        record: dict[str, object] = {
            "epoch": epoch, "train": train_metrics, "validation": validation_metrics,
        }
        if epoch == 1 or epoch % args.exact_score_every == 0 or epoch == args.epochs:
            exact = _exact_field_score(
                field_model, exact_loader, values, names, frames, args.cache_dir,
                tensors["components"], tensors["pca_mean"], device,
                args.exact_score_workers,
            )
            record["exact_score_perfect_scalars"] = exact
            selection = float(exact["S"])
            if selection > best_score + 1.0e-7:
                best_score = selection
                best_field_epoch = epoch
                best_field_state = deepcopy(field_model.state_dict())
                stale = 0
            else:
                stale += args.exact_score_every
        field_history.append(record)
        print(json.dumps({"stage": "field", **record}), flush=True)
        if stale >= args.patience:
            break
    if best_field_state is None:
        raise RuntimeError("exact-score field selection produced no checkpoint")
    field_model.load_state_dict(best_field_state)
    field_strength, field_calibration = _field_calibration(
        field_model, exact_loader, values, names, frames, args.cache_dir,
        tensors["components"], tensors["pca_mean"], device,
        args.exact_score_workers, field_calibration_scales,
    )

    q_state, q_epoch, q_strength, q_history = _fit_scalar(
        "q95", feature_width, train_loader, validation_loader,
        args.epochs, args.patience, args.learning_rate, args.weight_decay,
        args.scalar_hidden_width, args.scalar_layers, args.dropout,
        device, args.seed + 20000,
    )
    beta_state, beta_epoch, beta_strength, beta_history = _fit_scalar(
        "betaN", feature_width, train_loader, validation_loader,
        args.epochs, args.patience, args.learning_rate, args.weight_decay,
        args.scalar_hidden_width, args.scalar_layers, args.dropout,
        device, args.seed + 30000,
    )
    del field_model, field_optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    field_refit_epochs = max(1, min(args.final_refit_epochs, best_field_epoch))
    q_refit_epochs = max(1, min(args.final_refit_epochs, q_epoch))
    beta_refit_epochs = max(1, min(args.final_refit_epochs, beta_epoch))
    final_field_state, final_field_history = _refit_field(
        feature_width, shape_width, all_loader, field_refit_epochs,
        best_field_state, args, tensors, device,
    )
    final_q_state, final_q_history = _refit_scalar(
        "q95", feature_width, all_loader, q_refit_epochs, q_state, args, device,
    )
    final_beta_state, final_beta_history = _refit_scalar(
        "betaN", feature_width, all_loader, beta_refit_epochs, beta_state, args, device,
    )

    artifact = {
        "version": 2 if args.separate_decoder_branches else 1,
        "field_state_dict": {key: value.cpu() for key, value in final_field_state.items()},
        "q95_state_dict": {key: value.cpu() for key, value in final_q_state.items()},
        "betaN_state_dict": {key: value.cpu() for key, value in final_beta_state.items()},
        "input_mean": torch.from_numpy(input_mean),
        "input_scale": torch.from_numpy(input_scale),
        "shape_residual_mean": torch.from_numpy(shape_mean),
        "shape_residual_scale": torch.from_numpy(shape_scale),
        "coordinate_residual_mean": torch.from_numpy(coordinate_mean),
        "coordinate_residual_scale": torch.from_numpy(coordinate_scale),
        "scalar_residual_mean": torch.from_numpy(scalar_mean),
        "scalar_residual_scale": torch.from_numpy(scalar_scale),
        "topology_mean": torch.from_numpy(topology_mean),
        "topology_scale": torch.from_numpy(topology_scale),
        "contour_angles": (
            torch.from_numpy(contour_angles_np)
            if contour_angles_np is not None else None
        ),
        "pca_components": torch.from_numpy(components_np),
        "pca_mean": torch.from_numpy(pca_mean_np),
        "grid_r": torch.from_numpy(grid_r_np),
        "grid_z": torch.from_numpy(grid_z_np),
        "feature_width": int(feature_width),
        "shape_width": int(shape_width),
        "hidden_width": args.hidden_width,
        "layers": args.layers,
        "scalar_hidden_width": args.scalar_hidden_width,
        "scalar_layers": args.scalar_layers,
        "dropout": args.dropout,
        "fourier_modes": args.fourier_modes,
        "profile_modes": args.profile_modes,
        "dct_order": args.dct_order,
        "separate_decoder_branches": args.separate_decoder_branches,
        "objective_weights": objective_weights,
        "consistency_channel_weights": dict(zip(
            CONSISTENCY_NAMES, consistency_channel_weights.astype(float).tolist()
        )),
        "q95_correction_strength": q_strength,
        "betaN_correction_strength": beta_strength,
        "field_residual_scale": field_strength,
        "field_calibration_scales": field_calibration_scales,
        "training_shots": int(len(np.unique(names))),
        "training_rows": int(len(names)),
        "train_stride": train_stride,
        "shot_oof_base_predictions": True,
        "exact_scorer_selected": True,
        "direct_raster_geometry_loss": True,
        "exact_contour_supervision": args.contours is not None,
        "exact_contour_levelset_supervision": contour_offset is not None,
        "multirate_context": args.multirate_context is not None,
        "development_used": False,
        "frozen_audit_used": False,
        "data_policy": "real_diii_d_only",
    }
    if multirate_projector is not None:
        artifact.update({
            "multirate_projection_mean": torch.from_numpy(
                multirate_projector["projection_mean"].astype(np.float32)
            ),
            "multirate_projection_scale": torch.from_numpy(
                multirate_projector["projection_scale"].astype(np.float32)
            ),
            "multirate_projection_components": torch.from_numpy(
                multirate_projector["projection_components"].astype(np.float32)
            ),
            "multirate_offsets": multirate_projector["offsets"].astype(int).tolist(),
            "multirate_window_radii": (
                multirate_projector["window_radii"].astype(int).tolist()
            ),
            "multirate_program_statistics": (
                multirate_projector["program_statistics"].astype(str).tolist()
            ),
        })
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.artifact)
    report = {
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": _sha256(args.artifact),
        "oof": str(args.oof.resolve()),
        "target_psi_n": str(args.target_psi_n.resolve()),
        "topology": str(args.topology.resolve()),
        "contours": str(args.contours.resolve()) if args.contours is not None else None,
        "multirate_context": (
            str(args.multirate_context.resolve())
            if args.multirate_context is not None else None
        ),
        "training_shots": artifact["training_shots"],
        "training_rows": artifact["training_rows"],
        "validation_shots": int(len(np.unique(names[validation_mask]))),
        "exact_selection_shots": int(len(exact_shots)),
        "exact_selection_shot_names": exact_shots.tolist(),
        "best_field_epoch": best_field_epoch,
        "best_field_exact_score_perfect_scalars": best_score,
        "field_residual_scale": field_strength,
        "field_calibration": field_calibration,
        "separate_decoder_branches": args.separate_decoder_branches,
        "objective_weights": objective_weights,
        "consistency_channel_weights": artifact["consistency_channel_weights"],
        "field_refit_epochs": field_refit_epochs,
        "q95_epoch": q_epoch,
        "q95_refit_epochs": q_refit_epochs,
        "q95_correction_strength": q_strength,
        "betaN_epoch": beta_epoch,
        "betaN_refit_epochs": beta_refit_epochs,
        "betaN_correction_strength": beta_strength,
        "field_history": field_history,
        "q95_history": q_history,
        "betaN_history": beta_history,
        "final_field_history": final_field_history,
        "final_q95_history": final_q_history,
        "final_betaN_history": final_beta_history,
        "data_policy": "real_diii_d_only",
        "development_used": False,
        "frozen_audit_used": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.artifact.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
