"""Two-stage training utilities for NeuralGS."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import NeuralGSConfig
from .data import (
    PreparedFrameDataset,
    PreparedShotDataset,
    collate_prepared_shots,
    evenly_spaced_frame_indices,
    save_prepared_shot,
)
from .geometry import MU0, plasma_green_matrix
from .model import NeuralGS
from .physics import (
    green_response,
    gs_current_from_flux,
    trapezoid_grid_weights,
    weighted_gauge_project,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_names(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_supervision_losses(
    predicted: torch.Tensor,
    oracle: torch.Tensor,
    green: torch.Tensor,
    source_weights: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Gauge-invariant field and local-current losses for oracle source targets."""

    if predicted.shape != oracle.shape:
        raise ValueError(f"source shapes differ: {predicted.shape} != {oracle.shape}")
    target_weights = trapezoid_grid_weights(grid_r, grid_z).to(predicted)
    field_error = green_response(predicted - oracle, green).reshape(
        len(predicted), len(grid_z), len(grid_r)
    )
    target_field = green_response(oracle, green).reshape_as(field_error)
    field_error = weighted_gauge_project(field_error, target_weights)
    target_field = weighted_gauge_project(target_field, target_weights)
    green_loss = (
        field_error.square() * target_weights[None]
    ).sum() / (
        (target_field.square() * target_weights[None]).sum().clamp_min(1.0e-12)
    )

    quadrature = source_weights.to(predicted).reshape(1, *predicted.shape[-2:])
    current_loss = (
        (predicted - oracle).square() * quadrature
    ).sum() / (
        (oracle.square() * quadrature).sum().clamp_min(1.0e-12)
    )
    ip_scale = (
        oracle.abs().flatten(1).mean(dim=1).clamp_min(1.0)
    )[:, None, None]
    normalized = predicted / ip_scale
    smoothness = (
        torch.diff(normalized, n=2, dim=1).square().mean()
        + torch.diff(normalized, n=2, dim=2).square().mean()
    )
    negative = (
        torch.relu(-predicted).square() * quadrature
    ).sum() / (
        (oracle.square() * quadrature).sum().clamp_min(1.0e-12)
    )
    return {
        "source_green": green_loss,
        "source_current": current_loss,
        "source_smooth": smoothness,
        "source_negative": negative,
    }


def _prepare_one_shot(task: tuple[str, str, dict, int, bool]) -> tuple[str, int]:
    path_text, destination_text, config_values, frames_per_shot, compressed = task
    path = Path(path_text)
    destination = Path(destination_text)
    config = NeuralGSConfig.from_dict(config_values)
    row = pd.read_parquet(path).iloc[0].to_dict()
    if str(row.get("source", "DIII-D")) != "DIII-D":
        raise ValueError(
            f"refusing non-DIII-D training row {path}: Challenge 2 training must remain DIII-D-only"
        )
    selected = evenly_spaced_frame_indices(len(row["efit_times"]), frames_per_shot)
    save_prepared_shot(
        row,
        destination,
        config,
        frame_indices=selected,
        compressed=compressed,
    )
    return destination.name, len(selected)


def prepare_local_shards(
    input_dir: Path,
    output_dir: Path,
    config: NeuralGSConfig,
    skip: int = 0,
    n_shots: int = 0,
    frames_per_shot: int = 0,
    workers: int = 1,
    compressed: bool = True,
) -> list[Path]:
    """Convert raw one-row parquet shots to compact NeuralGS shards."""

    files = sorted(Path(input_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet shots found under {input_dir}")
    selected = files[skip:] if n_shots <= 0 else files[skip : skip + n_shots]
    output_dir.mkdir(parents=True, exist_ok=True)
    written = [output_dir / f"{path.stem}.npz" for path in selected]
    tasks = [
        (
            str(path),
            str(destination),
            config.to_dict(),
            int(frames_per_shot),
            bool(compressed),
        )
        for path, destination in zip(selected, written)
    ]
    workers = max(1, min(int(workers), len(tasks) or 1))
    started = time.monotonic()
    if workers == 1:
        results = map(_prepare_one_shot, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_prepare_one_shot, tasks, chunksize=1)
    frame_count = 0
    try:
        for index, (name, n_frames) in enumerate(results, start=1):
            frame_count += n_frames
            if index == 1 or index % 100 == 0 or index == len(selected):
                print(
                    f"  prepared {index}/{len(selected)}: {name} "
                    f"({time.monotonic() - started:.1f}s)",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    manifest = {
        "data_policy": "real_diii_d_only",
        "source_machines": ["DIII-D"],
        "mast_data_used": False,
        "synthetic_data_used": False,
        "raw_input_dir": str(Path(input_dir).resolve()),
        "raw_input_filenames_sha256": sha256_names(selected),
        "config": config.to_dict(),
        "preparation": {
            "shots": len(written),
            "frames": frame_count,
            "frames_per_shot": int(frames_per_shot),
            "workers": workers,
            "compressed": bool(compressed),
            "elapsed_seconds": time.monotonic() - started,
        },
        "files": [path.name for path in written],
    }
    (output_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return written


def split_shards(cache_dir: Path, n_val_shots: int, seed: int) -> tuple[list[Path], list[Path]]:
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "config.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        listed = manifest.get("files")
        paths = [cache_dir / name for name in listed] if listed else sorted(cache_dir.glob("*.npz"))
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"prepared shard is missing: {missing[0]}")
    else:
        paths = sorted(cache_dir.glob("*.npz"))
    if len(paths) <= n_val_shots:
        raise ValueError(f"need more than {n_val_shots} cached shots, found {len(paths)}")
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[n_val_shots:], paths[:n_val_shots]


def split_shards_from_manifest(
    cache_dir: Path,
    manifest_path: Path,
) -> tuple[list[Path], list[Path]]:
    """Use the locked source-head training/development split, excluding final holdout."""

    payload = json.loads(Path(manifest_path).read_text())
    train_paths = [Path(cache_dir) / name for name in payload["training"]]
    val_paths = [Path(cache_dir) / name for name in payload["audit_development"]]
    missing = [path for path in train_paths + val_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"locked split shard is missing: {missing[0]}")
    final_names = set(payload["final_holdout"])
    if final_names & {path.name for path in train_paths + val_paths}:
        raise RuntimeError("locked final holdout leaked into training/development paths")
    return train_paths, val_paths


def _grid_operators(path: Path, config: NeuralGSConfig, device: torch.device):
    with np.load(path, allow_pickle=False) as shot:
        grid_r = np.asarray(shot["grid_r"], dtype=np.float32)
        grid_z = np.asarray(shot["grid_z"], dtype=np.float32)
    green, source_r, source_z, weights = plasma_green_matrix(
        grid_r, grid_z, config.source_stride, config.green_softening_cells
    )
    return tuple(
        torch.as_tensor(value, device=device)
        for value in (green, source_r, source_z, weights, grid_r, grid_z)
    )


def _load_resume_state(model: NeuralGS, state: dict[str, torch.Tensor]) -> bool:
    """Load a checkpoint, allowing only explicitly neutral opt-in modules.

    Returns whether the spatial head was absent and therefore retained at its
    identity initialization.  All unrelated missing or unexpected parameters
    remain hard errors; ``strict=False`` is not used as a blanket escape hatch.
    """

    state = dict(state)
    projection_key = "set_encoder.input_projection.0.weight"
    if projection_key in state:
        old_weight = state[projection_key]
        new_weight = model.state_dict()[projection_key]
        if (
            model.config.coordinate_tokens
            and old_weight.ndim == 2
            and new_weight.ndim == 2
            and old_weight.shape[0] == new_weight.shape[0]
            and old_weight.shape[1] + 2 == new_weight.shape[1]
        ):
            # Coordinates are appended after all legacy features.  Zero-init
            # their columns so resuming is functionally identical at step 0.
            expanded = new_weight.new_zeros(new_weight.shape)
            expanded[:, : old_weight.shape[1]].copy_(old_weight)
            state[projection_key] = expanded
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = (
        {"spatial_current_head.weight", "spatial_current_head.bias"}
        if model.spatial_current_head is not None
        else set()
    )
    if model.source_head is not None:
        allowed_missing |= {
            "source_head.weight",
            "source_head.bias",
            "source_basis_transform",
        }
    if model.physics_raster is not None:
        allowed_missing |= {
            key for key in model.state_dict() if key.startswith("physics_raster.")
        }
    missing = set(incompatible.missing_keys)
    invalid_missing = not missing.issubset(allowed_missing)
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "incompatible resume checkpoint: "
            f"missing={sorted(missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return bool(missing & {"spatial_current_head.weight", "spatial_current_head.bias"})


def _move_batch(batch: dict, device: torch.device, channel_dropout: float = 0.0) -> dict:
    ignored = {"grid_r", "grid_z", "canonical_current_sign"}
    out = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key not in ignored
    }
    if channel_dropout > 0.0:
        keep = torch.rand_like(out["token_mask"], dtype=torch.float32) >= channel_dropout
        out["token_mask"] = out["token_mask"].bool() & keep
    return out


def _profile_loss(
    model: NeuralGS,
    batch: dict,
    operators,
    source_green_weight: float = 1.0,
    source_current_weight: float = 0.25,
    source_smooth_weight: float = 0.01,
    source_negative_weight: float = 0.002,
    source_coordinate_augmentation: float = 0.0,
    gauge_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    green, source_r, source_z, weights, grid_r, grid_z = operators
    teacher = batch["target_psi_n"]
    augmented = None
    if source_coordinate_augmentation > 0.0:
        with torch.no_grad():
            deployment = model(
                batch,
                green,
                source_r,
                source_z,
                weights,
                grid_r,
                grid_z,
                return_coordinate_trajectory=True,
            )["coordinate_trajectory"][-1]
        strength = source_coordinate_augmentation * torch.rand(
            (len(teacher), 1, 1), device=teacher.device, dtype=teacher.dtype
        )
        augmented = teacher + strength * (deployment - teacher)
    output = model(
        batch, green, source_r, source_z, weights, grid_r, grid_z,
        teacher_psi_n=teacher,
        teacher_token_psi_n=augmented,
        teacher_source_psi_n=augmented,
    )
    if model.source_head is not None and "oracle_current" in batch:
        losses = source_supervision_losses(
            output["current_density"],
            batch["oracle_current"],
            green,
            weights,
            grid_r,
            grid_z,
        )
        loss = (
            source_green_weight * losses["source_green"]
            + source_current_weight * losses["source_current"]
            + source_smooth_weight * losses["source_smooth"]
            + source_negative_weight * losses["source_negative"]
        )
        if gauge_weight > 0.0:
            with torch.no_grad():
                plasma = green_response(output["current_density"].detach(), green).reshape_as(
                    batch["target_psi"]
                )
                field = model.vacuum_gain.detach() * batch["vacuum_psi"] + plasma
                target_gauge = (batch["target_psi"] - field).flatten(1).mean(dim=1)
            predicted_gauge = (
                0.65
                * torch.tanh(model.gauge_head(output["latent"].detach()).squeeze(-1))
                * MU0
                * batch["ip_abs"]
            )
            gauge_scale = (MU0 * batch["ip_abs"]).clamp_min(0.05)
            gauge_loss = ((predicted_gauge - target_gauge) / gauge_scale).square().mean()
            loss = loss + gauge_weight * gauge_loss
            losses["gauge"] = gauge_loss
        return loss, {key: value.detach() for key, value in losses.items()}

    target_j = gs_current_from_flux(batch["target_psi"], grid_r, grid_z)
    target_j = torch.nn.functional.avg_pool2d(target_j.unsqueeze(1), 5, stride=1, padding=2)[:, 0]
    stride = model.config.source_stride
    target_j = torch.relu(target_j[:, ::stride, ::stride])[:, : len(source_z), : len(source_r)]
    target_mask = torch.sigmoid(
        (1.0 - batch["target_psi_n"][:, ::stride, ::stride][:, : len(source_z), : len(source_r)])
        / model.config.lcfs_temperature
    )
    target_j *= target_mask
    integral = (target_j.flatten(1) * weights[None]).sum(dim=1).clamp_min(1.0e-8)
    target_j *= (batch["ip_abs"] / integral)[:, None, None]
    scale = 2.0e6
    source_loss = torch.mean(
        (torch.asinh(output["current_density"] / scale) - torch.asinh(target_j / scale)).square()
        * target_mask
    )
    # Current normalization is hard-coded; this term guards implementation regressions.
    predicted_ip = (output["current_density"].flatten(1) * weights[None]).sum(dim=1)
    current_error = ((predicted_ip - batch["ip_abs"]) / batch["ip_abs"].clamp_min(1.0e4)).square().mean()
    loss = source_loss + 0.05 * current_error
    return loss, {"source": source_loss.detach(), "ip": current_error.detach()}


def normalized_flux_topology_loss(
    prediction: torch.Tensor,
    target_psi_n: torch.Tensor,
) -> torch.Tensor:
    """Compare flux-surface topology independently of gauge and amplitude.

    Target normalized flux supplies soft axis and LCFS bands.  Their predicted
    flux levels define a per-frame affine normalization, after which errors near
    the magnetic axis and separatrix receive extra weight.  This directly trains
    the contour geometry used by the challenge scorer without embedding any
    machine-specific coordinates or labels in inference.
    """

    finite = torch.isfinite(target_psi_n)
    target = torch.nan_to_num(target_psi_n).clamp(-0.5, 2.0)
    useful = finite & (target_psi_n >= -0.5) & (target_psi_n <= 2.0)
    dtype = prediction.dtype
    useful_f = useful.to(dtype)
    axis_band = torch.exp(-0.5 * (target / 0.08).square()) * useful_f
    lcfs_band = torch.exp(-0.5 * ((target - 1.0) / 0.08).square()) * useful_f

    def weighted_level(field: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (field * weights).flatten(1).sum(dim=1) / weights.flatten(1).sum(
            dim=1
        ).clamp_min(1.0e-6)

    axis_level = weighted_level(prediction, axis_band)
    lcfs_level = weighted_level(prediction, lcfs_band)
    target_axis_level = weighted_level(target, axis_band)
    target_lcfs_level = weighted_level(target, lcfs_band)
    raw_span = axis_level - lcfs_level
    orientation = torch.where(
        raw_span.detach() >= 0.0,
        torch.ones_like(raw_span),
        -torch.ones_like(raw_span),
    )
    span = orientation * raw_span.abs().clamp_min(0.02)
    coordinate = (axis_level[:, None, None] - prediction) / span[:, None, None]
    predicted_n = target_axis_level[:, None, None] + coordinate * (
        target_lcfs_level - target_axis_level
    )[:, None, None]

    # The magnetic-axis band covers only ~1.5% of useful pixels on the honest
    # validation fold, versus ~19% for the LCFS band.  Equal-looking pointwise
    # weights therefore let a handful of false O-points destroy axis-derived
    # consistency.  Balance the *integrated* contributions instead.
    focus = useful_f * (0.25 + 25.0 * axis_band + 4.0 * lcfs_band)
    residual = torch.nn.functional.smooth_l1_loss(
        predicted_n,
        target,
        beta=0.08,
        reduction="none",
    )
    surface_loss = (residual * focus).sum() / focus.sum().clamp_min(1.0)

    # A differentiable O-point coordinate term prevents a few spurious extrema
    # from moving the scorer's discrete magnetic axis while barely affecting
    # pixel MSE.  Normalize orientation and span so this remains invariant to
    # flux sign, gauge, and amplitude just like the surface loss above.
    nz, nr = prediction.shape[-2:]
    r_coordinate = torch.linspace(-1.0, 1.0, nr, device=prediction.device, dtype=dtype)
    z_coordinate = torch.linspace(-1.0, 1.0, nz, device=prediction.device, dtype=dtype)
    rr, zz = torch.meshgrid(r_coordinate, z_coordinate, indexing="xy")
    field_scale = prediction.detach().flatten(1).std(dim=1).clamp_min(1.0e-6)
    axis_confidence = (
        raw_span.detach().abs() / field_scale >= 1.0
    ).to(dtype)
    axis_span = torch.where(
        axis_confidence.bool(), raw_span.abs(), field_scale
    )
    oriented = orientation[:, None, None] * (
        prediction - axis_level[:, None, None]
    ) / axis_span[:, None, None]
    logits = (oriented / 0.055).flatten(1)
    logits = logits.masked_fill(~useful.flatten(1), -torch.inf)
    probability = torch.softmax(logits, dim=1).reshape_as(prediction)
    target_probability = axis_band / axis_band.flatten(1).sum(dim=1).clamp_min(1.0)[
        :, None, None
    ]
    predicted_r = (probability * rr[None]).sum(dim=(1, 2))
    predicted_z = (probability * zz[None]).sum(dim=(1, 2))
    target_r = (target_probability * rr[None]).sum(dim=(1, 2))
    target_z = (target_probability * zz[None]).sum(dim=(1, 2))
    per_frame_axis_error = (
        (predicted_r - target_r).square() + (predicted_z - target_z).square()
    )
    axis_coordinate_loss = (
        per_frame_axis_error * axis_confidence
    ).sum() / axis_confidence.sum().clamp_min(1.0)
    return surface_loss + 0.75 * axis_coordinate_loss


def dimensionless_flux_geometry_loss(
    prediction: torch.Tensor,
    target_psi_n: torch.Tensor,
) -> torch.Tensor:
    """Match axis and LCFS geometry on a machine-independent pixel canvas.

    The physical prediction is normalized from target-supplied axis/LCFS bands
    exactly as in :func:`normalized_flux_topology_loss`.  All subsequent
    features use coordinates in ``[-1, 1]``: plasma area, centroid, covariance,
    upper/lower boundary displacement, and a soft magnetic-axis coordinate.
    No machine dimensions or device identifier enter the objective.
    """

    finite = torch.isfinite(target_psi_n)
    target = torch.nan_to_num(target_psi_n).clamp(-0.5, 2.0)
    useful = finite & (target_psi_n >= -0.5) & (target_psi_n <= 2.0)
    useful_f = useful.to(prediction.dtype)
    axis_band = torch.exp(-0.5 * (target / 0.08).square()) * useful_f
    lcfs_band = torch.exp(-0.5 * ((target - 1.0) / 0.08).square()) * useful_f

    def weighted_level(field: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (field * weights).flatten(1).sum(dim=1) / weights.flatten(1).sum(
            dim=1
        ).clamp_min(1.0e-6)

    axis_level = weighted_level(prediction, axis_band)
    lcfs_level = weighted_level(prediction, lcfs_band)
    raw_span = axis_level - lcfs_level
    orientation = torch.where(
        raw_span.detach() >= 0.0,
        torch.ones_like(raw_span),
        -torch.ones_like(raw_span),
    )
    span = orientation * raw_span.abs().clamp_min(0.02)
    predicted_n = (axis_level[:, None, None] - prediction) / span[:, None, None]

    nz, nr = prediction.shape[-2:]
    r_coordinate = torch.linspace(
        -1.0, 1.0, nr, device=prediction.device, dtype=prediction.dtype
    )
    z_coordinate = torch.linspace(
        -1.0, 1.0, nz, device=prediction.device, dtype=prediction.dtype
    )
    rr, zz = torch.meshgrid(r_coordinate, z_coordinate, indexing="xy")

    def shape_features(coordinate: torch.Tensor) -> torch.Tensor:
        plasma = torch.sigmoid((1.0 - coordinate) / 0.07) * useful_f
        mass = plasma.sum(dim=(1, 2)).clamp_min(1.0e-6)
        useful_mass = useful_f.sum(dim=(1, 2)).clamp_min(1.0)
        area = mass / useful_mass
        center_r = (plasma * rr[None]).sum(dim=(1, 2)) / mass
        center_z = (plasma * zz[None]).sum(dim=(1, 2)) / mass
        delta_r = rr[None] - center_r[:, None, None]
        delta_z = zz[None] - center_z[:, None, None]
        moment_rr = (plasma * delta_r.square()).sum(dim=(1, 2)) / mass
        moment_zz = (plasma * delta_z.square()).sum(dim=(1, 2)) / mass
        moment_rz = (plasma * delta_r * delta_z).sum(dim=(1, 2)) / mass

        boundary = (
            torch.exp(-0.5 * ((coordinate - 1.0) / 0.065).square()) * useful_f
        )
        upper_gate = torch.sigmoid(
            (zz[None] - center_z[:, None, None]) / 0.06
        )
        upper = boundary * upper_gate
        lower = boundary * (1.0 - upper_gate)
        upper_r = (upper * rr[None]).sum(dim=(1, 2)) / upper.sum(
            dim=(1, 2)
        ).clamp_min(1.0e-6)
        lower_r = (lower * rr[None]).sum(dim=(1, 2)) / lower.sum(
            dim=(1, 2)
        ).clamp_min(1.0e-6)
        return torch.column_stack(
            (
                area,
                center_r,
                center_z,
                moment_rr,
                moment_zz,
                moment_rz,
                upper_r,
                lower_r,
            )
        )

    predicted_features = shape_features(predicted_n)
    target_features = shape_features(target)
    feature_weights = prediction.new_tensor((2.0, 4.0, 4.0, 2.0, 2.0, 1.0, 2.0, 2.0))
    geometry = torch.nn.functional.smooth_l1_loss(
        predicted_features,
        target_features,
        beta=0.03,
        reduction="none",
    )
    geometry_loss = (geometry * feature_weights[None]).sum() / (
        len(prediction) * feature_weights.sum()
    ).clamp_min(1.0)

    logits = (-predicted_n / 0.05).flatten(1).clamp(-60.0, 60.0)
    logits = logits.masked_fill(~useful.flatten(1), -torch.inf)
    axis_probability = torch.softmax(logits, dim=1).reshape_as(prediction)
    target_axis_probability = axis_band / axis_band.sum(dim=(1, 2)).clamp_min(1.0)[
        :, None, None
    ]
    predicted_axis_r = (axis_probability * rr[None]).sum(dim=(1, 2))
    predicted_axis_z = (axis_probability * zz[None]).sum(dim=(1, 2))
    target_axis_r = (target_axis_probability * rr[None]).sum(dim=(1, 2))
    target_axis_z = (target_axis_probability * zz[None]).sum(dim=(1, 2))
    axis_loss = (
        (predicted_axis_r - target_axis_r).square()
        + (predicted_axis_z - target_axis_z).square()
    ).mean()
    return geometry_loss + 1.5 * axis_loss


def _equilibrium_loss(
    model: NeuralGS,
    batch: dict,
    operators,
    gradient_weight: float = 0.12,
    topology_weight: float = 0.20,
    scalar_weight: float = 0.0,
    source_green_weight: float = 0.0,
    source_current_weight: float = 0.0,
    source_smooth_weight: float = 0.0,
    source_negative_weight: float = 0.0,
    source_coordinate_augmentation: float = 0.0,
    gauge_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    green, source_r, source_z, weights, grid_r, grid_z = operators
    output = model(batch, green, source_r, source_z, weights, grid_r, grid_z)
    target = batch["target_psi"]
    flux_scale = 0.5
    flux_loss = (output["psirz"] - target).square().mean() / flux_scale**2
    pred_dz, pred_dr = torch.gradient(output["psirz"], dim=(1, 2))
    true_dz, true_dr = torch.gradient(target, dim=(1, 2))
    gradient_loss = ((pred_dz - true_dz).square() + (pred_dr - true_dr).square()).mean() / 0.02**2
    topology_loss = normalized_flux_topology_loss(output["psirz"], batch["target_psi_n"])

    q_mask = torch.isfinite(batch["target_q95"])
    beta_mask = torch.isfinite(batch["target_betaN"])
    q_error = torch.where(
        q_mask, ((output["q95"] - torch.nan_to_num(batch["target_q95"])) / 5.0).square(), 0.0
    )
    beta_error = torch.where(
        beta_mask,
        ((output["betaN"] - torch.nan_to_num(batch["target_betaN"])) / 3.0).square(),
        0.0,
    )
    q_loss = q_error.sum() / q_mask.sum().clamp_min(1)
    beta_loss = beta_error.sum() / beta_mask.sum().clamp_min(1)
    loss = (
        flux_loss
        + gradient_weight * gradient_loss
        + topology_weight * topology_loss
        + scalar_weight * (q_loss + beta_loss)
    )
    parts = {
        "flux": flux_loss.detach(),
        "gradient": gradient_loss.detach(),
        "topology": topology_loss.detach(),
        "q95": q_loss.detach(),
        "betaN": beta_loss.detach(),
    }
    if model.source_head is not None and "oracle_current" in batch:
        source_parts = source_supervision_losses(
            output["current_density"],
            batch["oracle_current"],
            green,
            weights,
            grid_r,
            grid_z,
        )
        loss = loss + (
            source_green_weight * source_parts["source_green"]
            + source_current_weight * source_parts["source_current"]
            + source_smooth_weight * source_parts["source_smooth"]
            + source_negative_weight * source_parts["source_negative"]
        )
        parts.update({key: value.detach() for key, value in source_parts.items()})
    return loss, parts


def _run_epoch(
    model: NeuralGS,
    loader: DataLoader,
    operators,
    device: torch.device,
    stage: str,
    optimizer: torch.optim.Optimizer | None,
    channel_dropout: float,
    scaler=None,
    amp_enabled: bool = False,
    deadline: float | None = None,
    gradient_weight: float = 0.12,
    topology_weight: float = 0.20,
    scalar_weight: float = 0.30,
    source_green_weight: float = 0.0,
    source_current_weight: float = 0.0,
    source_smooth_weight: float = 0.0,
    source_negative_weight: float = 0.0,
    source_coordinate_augmentation: float = 0.0,
    gauge_weight: float = 0.0,
) -> tuple[dict[str, float], bool]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, torch.Tensor] = {}
    frames = 0
    completed = True
    for batch_index, raw_batch in enumerate(loader, start=1):
        if deadline is not None and time.monotonic() >= deadline:
            completed = False
            break
        batch = _move_batch(raw_batch, device, channel_dropout if training else 0.0)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss, parts = (
                    _profile_loss(
                        model,
                        batch,
                        operators,
                        source_green_weight=source_green_weight,
                        source_current_weight=source_current_weight,
                        source_smooth_weight=source_smooth_weight,
                        source_negative_weight=source_negative_weight,
                        source_coordinate_augmentation=source_coordinate_augmentation,
                        gauge_weight=gauge_weight,
                    )
                    if stage == "pretrain"
                    else _equilibrium_loss(
                        model,
                        batch,
                        operators,
                        gradient_weight=gradient_weight,
                        topology_weight=topology_weight,
                        scalar_weight=scalar_weight,
                        source_green_weight=source_green_weight,
                        source_current_weight=source_current_weight,
                        source_smooth_weight=source_smooth_weight,
                        source_negative_weight=source_negative_weight,
                    )
                )
        if not bool(torch.isfinite(loss).detach()):
            detail = {key: float(value) for key, value in parts.items()}
            raise FloatingPointError(
                f"non-finite {stage} loss at batch {batch_index}: {detail}"
            )
        if training:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                if not bool(torch.isfinite(grad_norm).detach()):
                    bad = [
                        name
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None
                        and not bool(torch.isfinite(parameter.grad).all().detach())
                    ]
                    raise FloatingPointError(
                        f"non-finite {stage} gradient at batch {batch_index}: {bad[:12]}"
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                if not bool(torch.isfinite(grad_norm).detach()):
                    bad = [
                        name
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None
                        and not bool(torch.isfinite(parameter.grad).all().detach())
                    ]
                    raise FloatingPointError(
                        f"non-finite {stage} gradient at batch {batch_index}: {bad[:12]}"
                    )
                optimizer.step()
        count = len(batch["ip_abs"])
        frames += count
        totals["loss"] = totals.get("loss", loss.detach().new_zeros(())) + loss.detach() * count
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value * count
    metrics = {
        key: float(value / max(frames, 1))
        for key, value in totals.items()
    }
    metrics["frames"] = float(frames)
    return metrics, completed


def _loaders(
    train_paths: list[Path],
    val_paths: list[Path],
    batch_size: int,
    frame_stride: int,
    num_workers: int,
    shot_batches: bool,
    seed: int,
    oracle_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader | None, int, int]:
    common = {
        "num_workers": max(int(num_workers), 0),
        "pin_memory": torch.cuda.is_available(),
    }
    if common["num_workers"]:
        common["persistent_workers"] = True
        common["prefetch_factor"] = 2
    generator = torch.Generator().manual_seed(seed)
    if shot_batches:
        train_data = PreparedShotDataset(
            train_paths, frame_stride=frame_stride, oracle_dir=oracle_dir
        )
        val_data = PreparedShotDataset(
            val_paths, frame_stride=frame_stride, oracle_dir=oracle_dir
        )
        train_shots_per_batch = max(1, int(batch_size) // train_data.max_frames)
        val_shots_per_batch = max(1, int(batch_size) // val_data.max_frames)
        train_loader = DataLoader(
            train_data,
            batch_size=train_shots_per_batch,
            shuffle=True,
            collate_fn=collate_prepared_shots,
            generator=generator,
            **common,
        )
        val_loader = (
            DataLoader(
                val_data,
                batch_size=val_shots_per_batch,
                shuffle=False,
                collate_fn=collate_prepared_shots,
                **common,
            )
            if val_paths
            else None
        )
        return train_loader, val_loader, train_data.n_frames, val_data.n_frames

    train_data = PreparedFrameDataset(
        train_paths, frame_stride=frame_stride, oracle_dir=oracle_dir
    )
    val_data = PreparedFrameDataset(
        val_paths, frame_stride=frame_stride, oracle_dir=oracle_dir
    )
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=generator, **common
    )
    val_loader = (
        DataLoader(val_data, batch_size=batch_size, shuffle=False, **common)
        if val_paths
        else None
    )
    return train_loader, val_loader, len(train_data), len(val_data)


def train_model(
    cache_dir: Path,
    artifact: Path,
    config: NeuralGSConfig,
    stage: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    n_val_shots: int,
    seed: int,
    device_name: str,
    resume: Path | None = None,
    channel_dropout: float = 0.10,
    frame_stride: int = 1,
    num_workers: int = 0,
    shot_batches: bool = False,
    amp: bool = False,
    deadline: float | None = None,
    gradient_weight: float = 0.12,
    topology_weight: float = 0.20,
    scalar_weight: float = 0.0,
    oracle_dir: Path | None = None,
    split_manifest: Path | None = None,
    source_transform: Path | None = None,
    freeze_source_probe: bool = False,
    freeze_raster_probe: bool = False,
    source_green_weight: float = 1.0,
    source_current_weight: float = 0.25,
    source_smooth_weight: float = 0.01,
    source_negative_weight: float = 0.002,
    source_coordinate_augmentation: float = 0.0,
    gauge_weight: float = 0.0,
) -> dict:
    """Train Stage A (inverse-GS sources) or Stage B (equilibrium fields)."""

    if stage not in {"pretrain", "finetune"}:
        raise ValueError("stage must be 'pretrain' or 'finetune'")
    if not 0.0 <= source_coordinate_augmentation <= 1.0:
        raise ValueError("source_coordinate_augmentation must be between 0 and 1")
    if gauge_weight < 0.0:
        raise ValueError("gauge_weight cannot be negative")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp_enabled = bool(amp and device.type == "cuda")
    if split_manifest is None:
        train_paths, val_paths = split_shards(cache_dir, n_val_shots, seed)
    else:
        train_paths, val_paths = split_shards_from_manifest(cache_dir, split_manifest)
        if oracle_dir is not None:
            split_payload = json.loads(Path(split_manifest).read_text())
            missing_targets = [
                Path(oracle_dir) / path.name
                for path in train_paths + val_paths
                if not (Path(oracle_dir) / path.name).exists()
            ]
            if missing_targets:
                raise FileNotFoundError(f"oracle target is missing: {missing_targets[0]}")
            leaked_final = [
                name
                for name in split_payload["final_holdout"]
                if (Path(oracle_dir) / name).exists()
            ]
            if leaked_final:
                raise RuntimeError(
                    f"final holdout oracle leakage detected: {leaked_final[0]}"
                )
    cache_manifest_path = Path(cache_dir) / "config.json"
    cache_manifest = (
        json.loads(cache_manifest_path.read_text())
        if cache_manifest_path.is_file()
        else {}
    )
    train_loader, val_loader, n_train_frames, n_val_frames = _loaders(
        train_paths,
        val_paths,
        batch_size,
        frame_stride,
        num_workers,
        shot_batches,
        seed,
        oracle_dir=oracle_dir,
    )
    operators = _grid_operators(train_paths[0], config, device)
    model = NeuralGS(config).to(device)
    if source_transform is not None:
        transform_artifact = torch.load(source_transform, map_location="cpu", weights_only=False)
        if isinstance(transform_artifact, dict):
            metadata = transform_artifact.get("metadata", {})
            if metadata.get("training_only") is not True:
                raise ValueError("source conditioning artifact is not marked training-only")
            if int(metadata.get("basis_size", -1)) != config.source_basis_size:
                raise ValueError("source conditioning basis size does not match the model")
            if int(metadata.get("source_stride", -1)) != config.source_stride:
                raise ValueError("source conditioning stride does not match the model")
        transform = (
            transform_artifact["transform"]
            if isinstance(transform_artifact, dict)
            else transform_artifact
        )
        if not bool(torch.isfinite(torch.as_tensor(transform)).all()):
            raise ValueError("source conditioning transform contains non-finite values")
        model.set_source_basis_transform(torch.as_tensor(transform))
    initialized_spatial_head = False
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        initialized_spatial_head = _load_resume_state(model, checkpoint["model_state"])
        if initialized_spatial_head:
            print(
                "initialized the opt-in spatial-current head as an identity "
                "(2 legacy checkpoint keys absent)",
                flush=True,
            )
    if freeze_source_probe:
        if freeze_raster_probe:
            raise ValueError("source and raster probe freezes are mutually exclusive")
        if model.source_head is None:
            raise ValueError("freeze_source_probe requires the conditioned signed source head")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.source_head.parameters():
            parameter.requires_grad_(True)
    elif freeze_raster_probe:
        if model.physics_raster is None:
            raise ValueError("freeze_raster_probe requires the physics raster encoder")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.physics_raster.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:  # PyTorch 2.0 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best = math.inf
    history = []
    started = time.monotonic()
    stopped_for_deadline = False
    saved = False
    artifact.parent.mkdir(parents=True, exist_ok=True)

    def save_artifact(epoch: int, validation_loss: float | None, complete: bool) -> None:
        nonlocal saved
        gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
        torch.save(
            {
                "artifact_version": 1,
                "config": config.to_dict(),
                "model_state": model.state_dict(),
                "metadata": {
                    "stage": stage,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "n_train_shots": len(train_paths),
                    "n_val_shots": len(val_paths),
                    "n_train_frames": n_train_frames,
                    "n_val_frames": n_val_frames,
                    "seed": seed,
                    "device": str(device),
                    "gpu_name": gpu_name,
                    "amp": amp_enabled,
                    "shot_batches": shot_batches,
                    "channel_dropout": float(channel_dropout),
                    "elapsed_seconds": time.monotonic() - started,
                    "complete": complete,
                    "resume": str(resume) if resume else None,
                    "resume_sha256": sha256_file(resume) if resume else None,
                    "data_policy": cache_manifest.get(
                        "data_policy", "legacy_unverified"
                    ),
                    "source_machines": cache_manifest.get("source_machines"),
                    "mast_data_used": cache_manifest.get("mast_data_used"),
                    "synthetic_data_used": cache_manifest.get(
                        "synthetic_data_used"
                    ),
                    "raw_input_dir": cache_manifest.get("raw_input_dir"),
                    "raw_input_filenames_sha256": cache_manifest.get(
                        "raw_input_filenames_sha256"
                    ),
                    "cache_manifest": str(cache_manifest_path.resolve())
                    if cache_manifest_path.is_file()
                    else None,
                    "cache_manifest_sha256": sha256_file(cache_manifest_path)
                    if cache_manifest_path.is_file()
                    else None,
                    "initialized_spatial_current_head": initialized_spatial_head,
                    "oracle_dir": str(oracle_dir) if oracle_dir else None,
                    "split_manifest": str(split_manifest) if split_manifest else None,
                    "source_transform": str(source_transform) if source_transform else None,
                    "freeze_source_probe": bool(freeze_source_probe),
                    "freeze_raster_probe": bool(freeze_raster_probe),
                    "loss_weights": {
                        "gradient": float(gradient_weight),
                        "topology": float(topology_weight),
                        "scalar": float(scalar_weight),
                        "source_green": float(source_green_weight),
                        "source_current": float(source_current_weight),
                        "source_smooth": float(source_smooth_weight),
                    "source_negative": float(source_negative_weight),
                    "source_coordinate_augmentation": float(
                        source_coordinate_augmentation
                    ),
                    "gauge": float(gauge_weight),
                    },
                },
            },
            artifact,
        )
        saved = True

    for epoch in range(1, epochs + 1):
        train_metrics, train_complete = _run_epoch(
            model,
            train_loader,
            operators,
            device,
            stage,
            optimizer,
            channel_dropout,
            scaler=scaler,
            amp_enabled=amp_enabled,
            deadline=deadline,
            gradient_weight=gradient_weight,
            topology_weight=topology_weight,
            scalar_weight=scalar_weight,
            source_green_weight=source_green_weight,
            source_current_weight=source_current_weight,
            source_smooth_weight=source_smooth_weight,
            source_negative_weight=source_negative_weight,
            source_coordinate_augmentation=source_coordinate_augmentation,
            gauge_weight=gauge_weight,
        )
        if not math.isfinite(train_metrics.get("loss", math.nan)):
            raise FloatingPointError(f"non-finite {stage} training loss at epoch {epoch}")
        if not train_complete:
            stopped_for_deadline = True
            if not saved:
                save_artifact(epoch, best if math.isfinite(best) else None, False)
            history.append({"epoch": epoch, "train": train_metrics, "partial": True})
            print(f"epoch {epoch:03d} stopped at the training deadline", flush=True)
            break
        if val_loader is None:
            scheduler.step()
            selection_loss = train_metrics["loss"]
            record = {"epoch": epoch, "train": train_metrics}
            history.append(record)
            print(
                f"epoch {epoch:03d} train={train_metrics['loss']:.6f} "
                "validation=disabled (full-data fit)",
                flush=True,
            )
            if selection_loss < best:
                best = selection_loss
                save_artifact(epoch, None, True)
            continue

        val_metrics, val_complete = _run_epoch(
            model,
            val_loader,
            operators,
            device,
            stage,
            None,
            0.0,
            amp_enabled=amp_enabled,
            deadline=deadline,
            gradient_weight=gradient_weight,
            topology_weight=topology_weight,
            scalar_weight=scalar_weight,
            source_green_weight=source_green_weight,
            source_current_weight=source_current_weight,
            source_smooth_weight=source_smooth_weight,
            source_negative_weight=source_negative_weight,
            source_coordinate_augmentation=source_coordinate_augmentation,
            gauge_weight=gauge_weight,
        )
        if not math.isfinite(val_metrics.get("loss", math.nan)):
            raise FloatingPointError(f"non-finite {stage} validation loss at epoch {epoch}")
        if not val_complete:
            stopped_for_deadline = True
            if not saved:
                save_artifact(epoch, best if math.isfinite(best) else None, False)
            history.append(
                {"epoch": epoch, "train": train_metrics, "validation": val_metrics, "partial": True}
            )
            print(f"epoch {epoch:03d} stopped during validation at the deadline", flush=True)
            break
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(record)
        print(
            f"epoch {epoch:03d} train={train_metrics['loss']:.6f} "
            f"validation={val_metrics['loss']:.6f}", flush=True,
        )
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            save_artifact(epoch, best, True)
    if not saved:
        save_artifact(0, best if math.isfinite(best) else None, not stopped_for_deadline)
    artifact.with_suffix(".json").write_text(json.dumps({"history": history}, indent=2) + "\n")
    return {
        "stage": stage,
        "elapsed_seconds": time.monotonic() - started,
        "stopped_for_deadline": stopped_for_deadline,
        "epochs_completed": sum(not record.get("partial", False) for record in history),
        "artifact": str(artifact),
    }
