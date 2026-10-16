#!/usr/bin/env python3
"""Fit a geometry-aligned variant of the proven FEC1 OOF residual model.

The ExtraTrees base predictions are out-of-fold at shot level.  This script
keeps the successful compact MLP/PCA64 path, but adds differentiable losses
computed from the rendered raster so that LCFS, kappa, volume, axis and li
errors can change the submitted field.  Model selection is performed on
held-out training shots with the official scorer; the deployable model is then
refit on all immutable training shots for the selected number of epochs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_experiments.fec1_conv_autoencoder import (  # noqa: E402
    normalize_shape,
    teacher_normalized_flux,
)
from my_experiments.fec1_direct_geometry import (  # noqa: E402
    direct_raster_geometry_losses,
)
from my_experiments.fec1_symmetric_geometry import symmetric_geometry_losses
from my_experiments.fit_fec1_full_frame_pca import _read  # noqa: E402
from my_experiments.fit_fec1_train_oof_residual import (  # noqa: E402
    OOFResidualHead,
    correction_strength,
    split_validation_shots,
)
from my_experiments.score_fec1_prediction_archive import (  # noqa: E402
    _parallel_official_score,
)
from my_experiments.train_fec1_score_aligned_temporal import (  # noqa: E402
    _align_contours,
    _align_topology,
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


def _batches(rows: np.ndarray, batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start:min(start + batch_size, len(rows))]


def load_geometry_targets(path: Path, preload: bool = False) -> np.ndarray:
    """Read once into RAM when random Lustre page faults dominate GPU batches."""
    started = time.monotonic()
    print(json.dumps({"stage": "load_geometry_start", "path": str(path), "preload": preload}), flush=True)
    values = np.load(path, mmap_mode=None if preload else "r")
    print(json.dumps({"stage": "load_geometry_done", "path": str(path), "preload": preload,
                      "bytes": values.nbytes, "elapsed_seconds": time.monotonic()-started}), flush=True)
    return values


def _standard_loss(
    prediction: tuple[torch.Tensor, ...], target: torch.Tensor, shape_width: int,
    coordinate_weight: float, scalar_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    shape, coordinate, scalar = prediction
    shape_loss = torch.mean(torch.square(shape.float() - target[:, :shape_width]))
    coordinate_loss = torch.mean(torch.square(
        coordinate.float() - target[:, shape_width:shape_width + 2]
    ))
    scalar_loss = torch.mean(torch.square(
        scalar.float() - target[:, shape_width + 2:]
    ))
    total = (
        shape_loss
        + coordinate_weight * coordinate_loss
        + scalar_weight * scalar_loss
    )
    return total, {
        "shape": shape_loss.detach(),
        "coordinate": coordinate_loss.detach(),
        "scalar": scalar_loss.detach(),
    }


def geometry_objective(
    parts: dict[str, torch.Tensor],
    *,
    axis_weight: float,
    lcfs_weight: float,
    shape_weight: float,
    li_weight: float,
) -> torch.Tensor:
    """Combine raster-derived terms with extra weight on weak axis/kappa/li metrics."""

    return (
        axis_weight * parts["direct_axis"]
        + lcfs_weight * parts["direct_lcfs"]
        + shape_weight * parts["direct_geometry"]
        + li_weight * parts["direct_li"]
    )


def raster_geometry_parts(rendered_shape, target_n, topology, contour_radius,
                          grid_r, grid_z, angles, *, loss_version="legacy", target_field=None):
    """Use actual target rasters, not PCA reconstructions, on the symmetric side."""
    if loss_version == "symmetric":
        if target_field is None:
            raise ValueError("symmetric geometry requires aligned actual target fields")
        return symmetric_geometry_losses(
            rendered_shape, normalize_shape(target_field.float()), target_n,
            grid_r, grid_z, angles=angles,
        )
    if loss_version != "legacy":
        raise ValueError("unknown geometry loss version")
    predicted_n, useful = teacher_normalized_flux(rendered_shape, target_n)
    return direct_raster_geometry_losses(
        predicted_n, target_n, useful, topology, grid_r, grid_z,
        exact_target_radius=contour_radius, contour_angles=angles,
    )


def _run_epoch(
    model: OOFResidualHead,
    *,
    inputs: np.ndarray,
    residual: np.ndarray,
    base_shape: np.ndarray,
    topology: np.ndarray,
    contour_radius: np.ndarray,
    target_psi_n: np.ndarray,
    source_rows: np.ndarray,
    rows: np.ndarray,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    components: torch.Tensor,
    pca_mean: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    contour_angles: torch.Tensor,
    geometry_weight: float,
    axis_weight: float,
    lcfs_weight: float,
    shape_geometry_weight: float,
    li_weight: float,
    coordinate_weight: float,
    scalar_weight: float,
    batch_size: int,
    direct_geometry_batch: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    seed: int,
    loss_version: str = "legacy",
    target_field: np.ndarray | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    epoch_started = time.monotonic()
    model.train(training)
    ordered = np.random.default_rng(seed).permutation(rows) if training else rows
    totals = {
        key: 0.0 for key in (
            "loss", "standard", "shape", "coordinate", "scalar", "geometry",
            "direct_axis", "direct_lcfs", "direct_geometry", "direct_li",
        )
    }
    count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_rows in _batches(ordered, batch_size):
            normalized_input = (
                (inputs[batch_rows] - input_mean) / input_scale
            ).astype(np.float32)
            normalized_target = (
                (residual[batch_rows] - target_mean) / target_scale
            ).astype(np.float32)
            x = torch.from_numpy(normalized_input).to(device)
            y = torch.from_numpy(normalized_target).to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(x)
            with torch.autocast(device_type=device.type, enabled=False):
                standard, standard_parts = _standard_loss(
                    prediction, y, base_shape.shape[1],
                    coordinate_weight, scalar_weight,
                )
                direct_count = min(direct_geometry_batch, len(batch_rows))
                direct_rows = batch_rows[:direct_count]
                predicted_shape_normalized = prediction[0][:direct_count].float()
                predicted_shape_residual = (
                    torch.from_numpy(target_mean[:base_shape.shape[1]]).to(device)
                    + torch.from_numpy(target_scale[:base_shape.shape[1]]).to(device)
                    * predicted_shape_normalized
                )
                coefficient = (
                    torch.from_numpy(base_shape[direct_rows]).to(device)
                    + predicted_shape_residual
                )
                flat = coefficient @ components + pca_mean
                rendered_shape = normalize_shape(
                    flat.reshape(direct_count, 65, 65)
                )
                target_n = torch.from_numpy(np.asarray(
                    target_psi_n[source_rows[direct_rows]], dtype=np.float32,
                )).to(device)
                direct = raster_geometry_parts(
                    rendered_shape,
                    target_n,
                    torch.from_numpy(topology[direct_rows]).to(device),
                    torch.from_numpy(contour_radius[direct_rows]).to(device),
                    grid_r,
                    grid_z,
                    contour_angles,
                    loss_version=loss_version,
                    target_field=(torch.as_tensor(np.array(target_field[source_rows[direct_rows]]),
                                                  device=device) if target_field is not None else None),
                )
                geometry = geometry_objective(
                    direct,
                    axis_weight=axis_weight,
                    lcfs_weight=lcfs_weight,
                    shape_weight=shape_geometry_weight,
                    li_weight=li_weight,
                )
                loss = standard + geometry_weight * geometry
            if not bool(torch.isfinite(loss)):
                detail = {
                    **{key: float(value) for key, value in standard_parts.items()},
                    **{key: float(value) for key, value in direct.items()},
                }
                raise RuntimeError(f"non-finite geometry-aligned loss: {detail}")
            if optimizer is not None:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            batch_count = len(batch_rows)
            totals["loss"] += float(loss.detach()) * batch_count
            totals["standard"] += float(standard.detach()) * batch_count
            for key, value in standard_parts.items():
                totals[key] += float(value) * batch_count
            totals["geometry"] += float(geometry.detach()) * batch_count
            for key in ("direct_axis", "direct_lcfs", "direct_geometry", "direct_li"):
                totals[key] += float(direct[key].detach()) * batch_count
            count += batch_count
            if training and (count == batch_count or count % (100 * batch_size) == 0):
                print(json.dumps({"stage": "geometry_training_batch", "seed": seed,
                                  "rows_done": count, "rows_total": len(rows),
                                  "elapsed_seconds": time.monotonic()-epoch_started}), flush=True)
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def _predict_residual(
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
        normalized = ((inputs[batch_rows] - input_mean) / input_scale).astype(np.float32)
        x = torch.from_numpy(normalized).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = torch.cat(model(x), dim=1)
        parts.append(output.float().cpu().numpy())
    normalized = np.concatenate(parts)
    return (target_mean + target_scale * normalized).astype(np.float32)


def _load_exact_targets(
    cache_dir: Path,
    names: np.ndarray,
    frames: np.ndarray,
    rows: np.ndarray,
) -> dict[str, np.ndarray]:
    local_names = names[rows]
    local_frames = frames[rows]
    field = np.empty((len(rows), 65, 65), dtype=np.float32)
    q95 = np.empty(len(rows), dtype=np.float32)
    beta_n = np.empty(len(rows), dtype=np.float32)
    shot_index = np.empty(len(rows), dtype=np.int64)
    for index, shot in enumerate(np.unique(local_names)):
        selected = np.flatnonzero(local_names == shot)
        data = _read(cache_dir / f"{shot}.npz", include_targets=True)
        source = local_frames[selected]
        field[selected] = data["target_psi"][source].astype(np.float32)
        q95[selected] = data["target_q95"][source].astype(np.float32)
        beta_n[selected] = data["target_betaN"][source].astype(np.float32)
        shot_index[selected] = index
    return {
        "field": field,
        "q95": q95,
        "betaN": beta_n,
        "shot_index": shot_index,
        "source_index": local_frames.astype(np.int64),
        "shot_name": local_names,
        "reference_source": str(cache_dir.resolve()),
    }


def _decode_exact_prediction(
    *,
    base_shape: np.ndarray,
    base_coordinate: np.ndarray,
    base_scalar: np.ndarray,
    correction: np.ndarray,
    components: np.ndarray,
    pca_mean: np.ndarray,
    shape_width: int,
    shape_strength: float,
    coordinate_strength: float,
    scalar_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_flat = base_shape @ components + pca_mean
    correction_flat = correction[:, :shape_width] @ components
    shape = (base_flat + shape_strength * correction_flat).reshape(-1, 65, 65)
    centered = shape - shape.mean(axis=(-2, -1), keepdims=True)
    rms = np.sqrt(np.mean(np.square(centered), axis=(-2, -1), keepdims=True))
    normalized_shape = centered / np.maximum(rms, 1.0e-7)
    coordinate = (
        base_coordinate
        + coordinate_strength * correction[:, shape_width:shape_width + 2]
    )
    scalar = base_scalar + scalar_strength * correction[:, shape_width + 2:]
    field = (
        coordinate[:, 0, None, None]
        + np.exp(coordinate[:, 1])[:, None, None] * normalized_shape
    ).astype(np.float32)
    return (
        field,
        np.maximum(scalar[:, 0], 1.0e-4).astype(np.float32),
        np.maximum(scalar[:, 1], 1.0e-4).astype(np.float32),
    )


def _exact_score_sweep(
    *,
    model: OOFResidualHead,
    inputs: np.ndarray,
    residual: np.ndarray,
    base_shape: np.ndarray,
    base_coordinate: np.ndarray,
    base_scalar: np.ndarray,
    rows: np.ndarray,
    names: np.ndarray,
    frames: np.ndarray,
    input_mean: np.ndarray,
    input_scale: np.ndarray,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    components: np.ndarray,
    pca_mean: np.ndarray,
    shape_scales: list[float],
    exact_targets: dict[str, np.ndarray],
    reference_cache: Path,
    batch_size: int,
    workers: int,
    device: torch.device,
    fixed_base_scalars: bool = False,
) -> dict[str, object]:
    correction = _predict_residual(
        model, inputs, rows, input_mean, input_scale,
        target_mean, target_scale, batch_size, device,
    )
    target = residual[rows]
    shape_width = base_shape.shape[1]
    coordinate_strength = correction_strength(
        correction[:, shape_width:shape_width + 2],
        target[:, shape_width:shape_width + 2],
    )
    scalar_strength = correction_strength(
        correction[:, shape_width + 2:], target[:, shape_width + 2:],
    )
    reports = []
    for shape_strength in shape_scales:
        field, q95, beta_n = _decode_exact_prediction(
            base_shape=base_shape[rows],
            base_coordinate=base_coordinate[rows],
            base_scalar=base_scalar[rows],
            correction=correction,
            components=components,
            pca_mean=pca_mean,
            shape_width=shape_width,
            shape_strength=shape_strength,
            coordinate_strength=coordinate_strength,
            scalar_strength=scalar_strength,
        )
        if fixed_base_scalars:
            q95 = np.maximum(base_scalar[rows, 0], 1.0e-4).astype(np.float32)
            beta_n = np.maximum(base_scalar[rows, 1], 1.0e-4).astype(np.float32)
        values = {
            "field": field,
            "target_field": exact_targets["field"],
            "q95": q95,
            "target_q95": exact_targets["q95"],
            "betaN": beta_n,
            "target_betaN": exact_targets["betaN"],
            "shot_index": exact_targets["shot_index"],
            "source_index": exact_targets["source_index"],
            "shot_name": exact_targets["shot_name"],
            "reference_source": exact_targets["reference_source"],
        }
        score = _parallel_official_score(values, max(1, workers), reference_cache)
        reports.append({"shape_strength": shape_strength, "score": score})
    best = max(reports, key=lambda item: float(item["score"]["S"]))
    return {
        "reports": reports,
        "selected_shape_strength": float(best["shape_strength"]),
        "coordinate_strength": coordinate_strength,
        "scalar_strength": scalar_strength,
        "score": best["score"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--target-psi-n", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--contours", type=Path, required=True)
    parser.add_argument("--pca-basis", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--direct-geometry-batch", type=int, default=16)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--coordinate-weight", type=float, default=0.50)
    parser.add_argument("--scalar-weight", type=float, default=0.25)
    parser.add_argument("--geometry-weight", type=float, required=True)
    parser.add_argument("--loss-version", choices=("legacy", "symmetric"), default="legacy")
    parser.add_argument("--target-field-directory", type=Path,
                        help="aligned index.npz and actual target_field.npy for symmetric geometry")
    parser.add_argument("--preload-geometry-targets", action="store_true",
                        help="read geometry target arrays into RAM once to avoid random shared-filesystem reads")
    parser.add_argument("--selection-fixed-base-scalars", action="store_true",
                        help="use the same OOF base scalars at every field-selection checkpoint")
    parser.add_argument("--axis-weight", type=float, default=20.0)
    parser.add_argument("--lcfs-weight", type=float, default=1.0)
    parser.add_argument("--shape-geometry-weight", type=float, default=3.0)
    parser.add_argument("--li-weight", type=float, default=3.0)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--exact-score-every", type=int, default=6)
    parser.add_argument("--exact-score-shots", type=int, default=96)
    parser.add_argument("--exact-score-workers", type=int, default=4)
    parser.add_argument("--shape-scales", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--seed", type=int, default=260906)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-shots", type=int, default=0, help="smoke-test only")
    args = parser.parse_args()
    if args.artifact.exists() or args.artifact.with_suffix(".json").exists():
        raise FileExistsError(f"use a fresh artifact path: {args.artifact}")
    if args.loss_version == "symmetric" and args.target_field_directory is None:
        parser.error("symmetric geometry requires --target-field-directory")
    if (
        args.epochs < 1 or args.patience < 1 or args.batch_size < 1
        or args.direct_geometry_batch < 1 or args.direct_geometry_batch > args.batch_size
        or args.width < 1 or args.exact_score_every < 1 or args.exact_score_workers < 1
    ):
        parser.error("invalid epoch, batch, width, or exact-score settings")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("validation fraction must lie in (0, 1)")
    weights = {
        "coordinate": args.coordinate_weight,
        "scalar": args.scalar_weight,
        "geometry": args.geometry_weight,
        "axis": args.axis_weight,
        "lcfs": args.lcfs_weight,
        "shape_geometry": args.shape_geometry_weight,
        "li": args.li_weight,
    }
    if any(not np.isfinite(value) or value < 0.0 for value in weights.values()):
        parser.error("loss weights must be finite and nonnegative")
    shape_scales = sorted({
        float(value) for value in args.shape_scales.split(",") if value.strip()
    })
    if not shape_scales or any(value < 0.0 for value in shape_scales):
        parser.error("shape scales must be nonnegative")

    started = time.monotonic()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    required = (
        "shot_name", "frame_index", "features", "target_shape",
        "target_coordinate", "target_scalar", "base_shape",
        "base_coordinate", "base_scalar",
    )
    with np.load(args.oof, allow_pickle=False) as saved:
        missing = [key for key in required if key not in saved.files]
        if missing:
            raise RuntimeError(f"OOF archive is missing {missing}")
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
    full_names = values["shot_name"].astype(str)
    full_frames = values["frame_index"].astype(np.int64)
    target_field = None
    if args.target_field_directory is not None:
        with np.load(args.target_field_directory / "index.npz", allow_pickle=False) as index:
            if not (np.array_equal(full_names, index["shot_name"].astype(str))
                    and np.array_equal(full_frames, index["frame_index"])):
                raise RuntimeError("actual target-field cache is not aligned to OOF rows")
        target_field = load_geometry_targets(args.target_field_directory / "target_field.npy",
                                             args.preload_geometry_targets)
        if target_field.shape != (len(full_names), 65, 65):
            raise RuntimeError("actual target-field dimensions differ from OOF population")
    source_rows = np.arange(len(full_names), dtype=np.int64)
    topology = _align_topology(args.topology, full_names, full_frames)
    contour_radius, contour_angles_np, _ = _align_contours(
        args.contours, full_names, full_frames,
    )
    if args.max_shots > 0:
        selected_shots = np.unique(full_names)[:args.max_shots]
        keep = np.isin(full_names, selected_shots)
        source_rows = np.flatnonzero(keep)
        values = {key: value[keep] for key, value in values.items()}
        topology = topology[keep]
        contour_radius = contour_radius[keep]
    names = values["shot_name"].astype(str)
    frames = values["frame_index"].astype(np.int64)
    if args.max_shots <= 0 and len(np.unique(names)) != 5677:
        raise RuntimeError("geometry training does not cover all 5,677 training shots")

    target_psi_n = load_geometry_targets(args.target_psi_n, args.preload_geometry_targets)
    if len(target_psi_n) != len(full_names):
        raise RuntimeError("normalized-flux target rows differ from OOF rows")
    with np.load(args.pca_basis, allow_pickle=False) as saved:
        for key, expected in (
            ("development_used", False), ("frozen_audit_used", False),
        ):
            if key not in saved.files or saved[key].item() != expected:
                raise RuntimeError(f"unsafe PCA basis provenance: {key}")
        components_np = saved["components"].astype(np.float32)
        pca_mean_np = saved["mean"].astype(np.float32)
        grid_r_np = saved["grid_r"].astype(np.float32)
        grid_z_np = saved["grid_z"].astype(np.float32)

    inputs = np.column_stack((
        values["features"], values["base_shape"],
        values["base_coordinate"], values["base_scalar"],
    )).astype(np.float32)
    base_shape = values["base_shape"].astype(np.float32, copy=False)
    base_coordinate = values["base_coordinate"].astype(np.float32, copy=False)
    base_scalar = values["base_scalar"].astype(np.float32, copy=False)
    topology = topology.astype(np.float32, copy=False)
    contour_radius = contour_radius.astype(np.float32, copy=False)
    residual = np.column_stack((
        values["target_shape"] - values["base_shape"],
        values["target_coordinate"] - values["base_coordinate"],
        values["target_scalar"] - values["base_scalar"],
    )).astype(np.float32)
    if not np.isfinite(inputs).all() or not np.isfinite(residual).all():
        raise RuntimeError("geometry training arrays contain non-finite values")
    training_mask, validation_mask = split_validation_shots(
        names, args.validation_fraction, args.seed,
    )
    train_rows = np.flatnonzero(training_mask)
    validation_rows = np.flatnonzero(validation_mask)
    input_mean, input_scale = _statistics(inputs, train_rows)
    target_mean, target_scale = _statistics(residual, train_rows)
    validation_shots = np.unique(names[validation_rows])
    exact_rows = np.empty(0, dtype=np.int64)
    exact_targets = None
    if args.exact_score_shots > 0:
        chosen = np.random.default_rng(args.seed + 70001).permutation(
            validation_shots
        )[:args.exact_score_shots]
        exact_rows = np.flatnonzero(validation_mask & np.isin(names, chosen))
        exact_targets = _load_exact_targets(
            args.cache_dir, names, frames, exact_rows,
        )
    print(
        f"geometry OOF rows={len(names)} shots={len(np.unique(names))} "
        f"train_rows={len(train_rows)} validation_rows={len(validation_rows)} "
        f"exact_rows={len(exact_rows)} geometry_weight={args.geometry_weight}",
        flush=True,
    )

    tensors = {
        "components": torch.from_numpy(components_np).to(device),
        "pca_mean": torch.from_numpy(pca_mean_np).to(device),
        "grid_r": torch.from_numpy(grid_r_np).to(device),
        "grid_z": torch.from_numpy(grid_z_np).to(device),
        "contour_angles": torch.from_numpy(contour_angles_np).to(device),
    }

    def run_epoch(
        model: OOFResidualHead,
        rows: np.ndarray,
        local_input_mean: np.ndarray,
        local_input_scale: np.ndarray,
        local_target_mean: np.ndarray,
        local_target_scale: np.ndarray,
        epoch: int,
        optimizer: torch.optim.Optimizer | None,
        scaler: torch.amp.GradScaler | None,
    ) -> dict[str, float]:
        return _run_epoch(
            model,
            inputs=inputs,
            residual=residual,
            base_shape=base_shape,
            topology=topology,
            contour_radius=contour_radius,
            target_psi_n=target_psi_n,
            source_rows=source_rows,
            rows=rows,
            input_mean=local_input_mean,
            input_scale=local_input_scale,
            target_mean=local_target_mean,
            target_scale=local_target_scale,
            components=tensors["components"],
            pca_mean=tensors["pca_mean"],
            grid_r=tensors["grid_r"],
            grid_z=tensors["grid_z"],
            contour_angles=tensors["contour_angles"],
            geometry_weight=args.geometry_weight,
            axis_weight=args.axis_weight,
            lcfs_weight=args.lcfs_weight,
            shape_geometry_weight=args.shape_geometry_weight,
            li_weight=args.li_weight,
            coordinate_weight=args.coordinate_weight,
            scalar_weight=args.scalar_weight,
            batch_size=args.batch_size,
            direct_geometry_batch=args.direct_geometry_batch,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            seed=args.seed + epoch,
            loss_version=args.loss_version,
            target_field=target_field,
        )

    model = OOFResidualHead(
        inputs.shape[1], values["base_shape"].shape[1], args.width, args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_selection = -np.inf
    best_state = None
    best_epoch = 0
    best_calibration = None
    stale = 0
    history = []
    reference_cache = args.artifact.with_name(
        f"fec1_geometry_selection_seed{args.seed}_reference.pkl"
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_rows, input_mean, input_scale,
            target_mean, target_scale, epoch, optimizer, scaler,
        )
        validation_metrics = run_epoch(
            model, validation_rows, input_mean, input_scale,
            target_mean, target_scale, epoch, None, None,
        )
        record: dict[str, object] = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        evaluate_exact = (
            exact_targets is not None
            and (epoch == 1 or epoch % args.exact_score_every == 0 or epoch == args.epochs)
        )
        if evaluate_exact:
            exact = _exact_score_sweep(
                model=model,
                inputs=inputs,
                residual=residual,
                base_shape=base_shape,
                base_coordinate=base_coordinate,
                base_scalar=base_scalar,
                rows=exact_rows,
                names=names,
                frames=frames,
                input_mean=input_mean,
                input_scale=input_scale,
                target_mean=target_mean,
                target_scale=target_scale,
                components=components_np,
                pca_mean=pca_mean_np,
                shape_scales=shape_scales,
                exact_targets=exact_targets,
                reference_cache=reference_cache,
                batch_size=args.batch_size,
                workers=args.exact_score_workers,
                device=device,
                fixed_base_scalars=args.selection_fixed_base_scalars,
            )
            record["exact_score"] = exact
            selection = float(exact["score"]["S"])
        else:
            selection = -float(validation_metrics["loss"])
        if (exact_targets is None or evaluate_exact) and selection > best_selection + 1.0e-7:
            best_selection = selection
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_calibration = record.get("exact_score")
            stale = 0
        elif evaluate_exact:
            stale += args.exact_score_every
        history.append(record)
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.with_suffix(".progress.json").write_text(json.dumps({
            "stage": "selection", "epoch": epoch, "best_epoch": best_epoch,
            "history": history, "loss_version": args.loss_version,
        }, indent=2)+"\n")
        print(json.dumps({"stage": "selection", **record}), flush=True)
        if exact_targets is not None and stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("geometry-aligned selection produced no checkpoint")
    model.load_state_dict(best_state)
    if best_calibration is None:
        validation_prediction = _predict_residual(
            model, inputs, validation_rows, input_mean, input_scale,
            target_mean, target_scale, args.batch_size, device,
        )
        validation_target = residual[validation_rows]
        shape_width = values["base_shape"].shape[1]
        best_calibration = {
            "selected_shape_strength": correction_strength(
                validation_prediction[:, :shape_width],
                validation_target[:, :shape_width],
            ),
            "coordinate_strength": correction_strength(
                validation_prediction[:, shape_width:shape_width + 2],
                validation_target[:, shape_width:shape_width + 2],
            ),
            "scalar_strength": correction_strength(
                validation_prediction[:, shape_width + 2:],
                validation_target[:, shape_width + 2:],
            ),
            "score": None,
            "reports": [],
        }

    del model, optimizer, scaler, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    all_rows = np.arange(len(names), dtype=np.int64)
    final_input_mean, final_input_scale = _statistics(inputs, all_rows)
    final_target_mean, final_target_scale = _statistics(residual, all_rows)
    torch.manual_seed(args.seed + 100000)
    final_model = OOFResidualHead(
        inputs.shape[1], values["base_shape"].shape[1], args.width, args.dropout,
    ).to(device)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    final_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    final_history = []
    for epoch in range(1, best_epoch + 1):
        metrics = run_epoch(
            final_model, all_rows, final_input_mean, final_input_scale,
            final_target_mean, final_target_scale, 100000 + epoch,
            final_optimizer, final_scaler,
        )
        final_history.append({"epoch": epoch, **metrics})
        args.artifact.with_suffix(".progress.json").write_text(json.dumps({
            "stage": "refit", "epoch": epoch, "selected_epochs": best_epoch,
            "history": final_history, "loss_version": args.loss_version,
        }, indent=2)+"\n")
        print(json.dumps({"stage": "refit", "epoch": epoch, **metrics}), flush=True)

    strengths = {
        "shape": float(best_calibration["selected_shape_strength"]),
        "coordinate": float(best_calibration["coordinate_strength"]),
        "scalar": float(best_calibration["scalar_strength"]),
    }
    artifact = {
        "state_dict": {
            key: value.detach().cpu() for key, value in final_model.state_dict().items()
        },
        "input_mean": torch.from_numpy(final_input_mean),
        "input_scale": torch.from_numpy(final_input_scale),
        "target_mean": torch.from_numpy(final_target_mean),
        "target_scale": torch.from_numpy(final_target_scale),
        "input_width": int(inputs.shape[1]),
        "shape_width": int(values["base_shape"].shape[1]),
        "width": int(args.width),
        "dropout": float(args.dropout),
        "correction_strength": strengths,
        "geometry_weight": float(args.geometry_weight),
        "loss_version": args.loss_version,
        "selection_fixed_base_scalars": args.selection_fixed_base_scalars,
        "smoke_only": args.max_shots > 0,
        "geometry_term_weights": weights,
        "selected_epochs": int(best_epoch),
        "selected_exact_score": best_calibration["score"],
        "data_policy": "real_diii_d_only",
        "training_shots": int(len(np.unique(names))),
        "training_rows": int(len(names)),
        "train_stride": train_stride,
        "shot_oof_base_predictions": True,
        "all_train_shots_used_for_final_fit": True,
        "exact_scorer_selected": exact_targets is not None,
        "development_used": False,
        "frozen_audit_used": False,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.artifact)
    report = {
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": _sha256(args.artifact),
        "oof": str(args.oof.resolve()),
        "training_shots": artifact["training_shots"],
        "training_rows": artifact["training_rows"],
        "geometry_weight": args.geometry_weight,
        "loss_version": args.loss_version,
        "selection_fixed_base_scalars": args.selection_fixed_base_scalars,
        "smoke_only": args.max_shots > 0,
        "training_arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "geometry_term_weights": weights,
        "selected_epochs": best_epoch,
        "selected_exact_score": best_calibration["score"],
        "correction_strength": strengths,
        "selection_history": history,
        "final_history": final_history,
        "shot_oof_base_predictions": True,
        "all_train_shots_used_for_final_fit": True,
        "development_used": False,
        "frozen_audit_used": False,
        "target_score": 0.996,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.artifact.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
