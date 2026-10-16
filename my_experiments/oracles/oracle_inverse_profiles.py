#!/usr/bin/env python3
"""Nested forward-inverse oracle for the frozen NeuralGS physics layer.

This experiment deliberately does not train or alter NeuralGS.  It freezes a
checkpoint, initializes each held-out frame with its predicted source-profile
parameters, and optimizes increasingly expressive subsets of those parameters
through the existing differentiable free-boundary Grad--Shafranov solve.

The default sample is one deterministic central frame from each of the 200
held-out shots.  That gives equal shot weight and exactly 200 frames.  Every
completed stage is checkpointed before the (comparatively slow) exact challenge
scorer runs, making the experiment safe to launch in a persistent tmux/Slurm
session.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import local_score
from fusion_scoring.common import N_CONS, N_SCALARS, PSI_SIGNS, SCALARS
from fusion_scoring.metrics import Accum, finalize_machine
from neuralgs.config import NeuralGSConfig
from neuralgs.data import PER_FRAME_KEYS, TARGET_AXIS_SIGN, evenly_spaced_frame_indices
from neuralgs.geometry import plasma_green_matrix
from neuralgs.model import NeuralGS
from neuralgs.physics import (
    current_density_centroid,
    estimate_flux_coordinates,
    green_response,
    gs_current_from_flux,
    initial_current_density,
    modulate_spatial_current,
    profile_current_density,
    sample_grid,
)
from neuralgs.training import normalized_flux_topology_loss, split_shards
from my_experiments.neuralgs.neuralgs_direct_scalar import (
    build_direct_scalar_features,
    load_direct_scalar_artifact,
    predict_from_features,
    sha256_file,
)


@dataclass(frozen=True)
class Stage:
    name: str
    ff_count: int = 0
    temporal: bool = False


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stored_flux_from_canonical(values, machine: str = "DIII-D"):
    """Undo cache target canonicalization, independently of plasma-current sign."""

    return values / float(TARGET_AXIS_SIGN[machine])


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key not in {"grid_r", "grid_z", "canonical_current_sign"}
    }


def load_equal_shot_sample(
    cache_dir: Path,
    n_val_shots: int,
    seed: int,
    n_shots: int,
    frames_per_shot: int,
) -> tuple[dict[str, torch.Tensor], list[dict], list[str], np.ndarray]:
    """Load an equal number of deterministic frames from held-out shots."""

    _, validation = split_shards(cache_dir, n_val_shots=n_val_shots, seed=seed)
    selected_paths = validation[:n_shots]
    if not selected_paths:
        raise ValueError("no held-out shots selected")
    samples: list[dict[str, torch.Tensor]] = []
    groups: list[dict] = []
    names: list[str] = []
    frame_to_shot: list[int] = []
    shared: dict[str, torch.Tensor] | None = None
    offset = 0
    for shot_index, path in enumerate(selected_paths):
        with np.load(path, allow_pickle=False) as archive:
            n_frames = len(archive["ip_abs"])
            indices = evenly_spaced_frame_indices(n_frames, frames_per_shot)
            item: dict[str, torch.Tensor] = {}
            for key in archive.files:
                array = archive[key]
                if key in PER_FRAME_KEYS:
                    array = array[indices]
                item[key] = torch.from_numpy(np.array(array, copy=True))
        if shared is None:
            shared = {
                key: item[key]
                for key in ("grid_r", "grid_z", "canonical_current_sign")
                if key in item
            }
        samples.append(item)
        count = len(indices)
        groups.append({"name": path.stem, "start": offset, "stop": offset + count})
        names.append(path.stem)
        frame_to_shot.extend([shot_index] * count)
        offset += count
    assert shared is not None
    batch = {
        key: torch.cat([item[key] for item in samples], dim=0)
        for key in samples[0]
        if key in PER_FRAME_KEYS
    }
    batch.update(shared)
    return batch, groups, names, np.asarray(frame_to_shot, dtype=np.int64)


def load_frozen_model(path: Path, device: torch.device) -> tuple[NeuralGS, NeuralGSConfig, dict]:
    artifact = torch.load(path, map_location=device, weights_only=False)
    if artifact.get("artifact_version") != 1:
        raise ValueError(f"unsupported NeuralGS artifact version: {artifact.get('artifact_version')}")
    config = NeuralGSConfig.from_dict(artifact["config"])
    model = NeuralGS(config).to(device)
    model.load_state_dict(artifact["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config, artifact


def batched_model_outputs(
    model: NeuralGS,
    batch: dict[str, torch.Tensor],
    operators: tuple[torch.Tensor, ...],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    green, source_r, source_z, source_weights, grid_r, grid_z = operators
    outputs: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for start in range(0, len(batch["ip_abs"]), batch_size):
            part = {key: value[start : start + batch_size] for key, value in batch.items()}
            result = model(part, green, source_r, source_z, source_weights, grid_r, grid_z)
            for key, value in result.items():
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == len(part["ip_abs"]):
                    outputs.setdefault(key, []).append(value.detach())
    return {key: torch.cat(parts, dim=0) for key, parts in outputs.items()}


def apply_deployment_scalar_head(
    initial: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    operators: tuple[torch.Tensor, ...],
    base_artifact: Path,
    scalar_artifact: Path | None,
    model: NeuralGS,
    device: torch.device,
) -> dict:
    """Replace q95/betaN with the exact frozen head used by the v11 path."""

    initial["q95_base_head"] = initial["q95"].clone()
    initial["betaN_base_head"] = initial["betaN"].clone()
    if scalar_artifact is None:
        return {"kind": "base_model_scalar_head", "artifact": None}
    head, artifact = load_direct_scalar_artifact(scalar_artifact, device)
    expected_hash = artifact.get("base_artifact_sha256")
    actual_hash = sha256_file(base_artifact)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError(
            "direct scalar artifact was trained against a different NeuralGS base artifact"
        )
    _, source_r, source_z, source_weights, grid_r, grid_z = operators
    with torch.no_grad():
        features = build_direct_scalar_features(
            initial,
            batch,
            source_r,
            source_z,
            source_weights,
            grid_r,
            grid_z,
            model.config.lcfs_temperature,
        )
        prediction = predict_from_features(features, head, artifact)
    initial["q95"] = prediction[:, 0]
    initial["betaN"] = prediction[:, 1]
    return {
        "kind": "v11_frozen_direct_scalar_head",
        "artifact": str(scalar_artifact.resolve()),
        "base_artifact_sha256": actual_hash,
        "validation": artifact.get("training", {}).get("validation"),
    }


class OracleParameters(nn.Module):
    """Per-frame inverse variables initialized from a frozen NeuralGS prediction."""

    def __init__(self, initial: dict[str, torch.Tensor], target: torch.Tensor):
        super().__init__()
        n = len(target)
        self.log_flux_scale = nn.Parameter(target.new_zeros(n))
        self.gauge_shift = nn.Parameter(target.new_zeros(n))
        self.log_pressure_scale = nn.Parameter(initial["pressure_scale"].clamp_min(1.0e-3).log())
        fraction = initial["pressure_fraction"].clamp(1.0e-4, 1.0 - 1.0e-4)
        self.fraction_logit = nn.Parameter(torch.logit(fraction))
        self.pressure = nn.Parameter(initial["pressure_correction"].clone())
        self.ffprime = nn.Parameter(initial["ffprime_coefficients"].clone())
        scale = target.flatten(1).std(dim=1).clamp_min(1.0e-4)
        self.register_buffer("target_scale", scale)

    def transformed(self, index: slice | torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "flux_scale": self.log_flux_scale[index].clamp(-2.0, 2.0).exp(),
            "gauge_shift": self.gauge_shift[index] * self.target_scale[index],
            "pressure_scale": self.log_pressure_scale[index].clamp(-3.0, 3.0).exp(),
            "pressure_fraction": torch.sigmoid(self.fraction_logit[index]),
            "pressure_correction": self.pressure[index],
            "ffprime_coefficients": self.ffprime[index],
        }

    def profile_matrix(self) -> torch.Tensor:
        return torch.cat(
            [
                self.log_pressure_scale[:, None],
                self.fraction_logit[:, None],
                self.pressure,
                self.ffprime,
            ],
            dim=1,
        )


def _ff_order(n: int) -> list[int]:
    """Nested endpoint/midpoint ordering for spline knots."""

    if n <= 0:
        return []
    chosen: list[int] = []
    candidates: list[tuple[int, int]] = [(0, n - 1)]
    while candidates:
        left, right = candidates.pop(0)
        for value in (left, right, (left + right) // 2):
            if value not in chosen:
                chosen.append(value)
        if right - left > 2:
            middle = (left + right) // 2
            candidates.extend([(left + 1, middle - 1), (middle + 1, right - 1)])
    chosen.extend(index for index in range(n) if index not in chosen)
    return chosen[:n]


def set_active_stage(parameters: OracleParameters, stage: Stage) -> list[nn.Parameter]:
    for parameter in parameters.parameters():
        parameter.requires_grad_(False)
    active_names = ["log_flux_scale", "gauge_shift"]
    if stage.name != "affine":
        active_names += ["log_pressure_scale", "fraction_logit"]
    if stage.name not in {"affine", "balance"}:
        active_names.append("pressure")
    if stage.ff_count:
        # A gradient hook retains only the nested subset of spline knots.
        active_names.append("ffprime")
    active = []
    for name in active_names:
        parameter = getattr(parameters, name)
        parameter.requires_grad_(True)
        active.append(parameter)
    return active


def solve_oracle(
    model: NeuralGS,
    batch: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    base_gauge: torch.Tensor,
    spatial_coefficients: torch.Tensor | None,
    operators: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    """Mirror NeuralGS's frozen GS loop with explicit profile parameters."""

    green, source_r, source_z, source_weights, grid_r, grid_z = operators
    config = model.config
    current = initial_current_density(
        batch["ip_abs"], batch["token_coords"], batch["token_values"], batch["token_mask"],
        source_r, source_z, source_weights, config.initial_current_center,
    )
    full_shape = batch["vacuum_psi"].shape
    psi = model.vacuum_gain * batch["vacuum_psi"] + green_response(current, green).reshape(full_shape)
    pressure = None
    psi_n = None
    for _ in range(config.outer_iterations):
        if config.axis_estimator == "current_local":
            hint_r, hint_z = current_density_centroid(
                current.detach(), source_r, source_z, source_weights
            )
        else:
            hint_r = hint_z = None
        # Intentionally full-gradient: oracle observability includes flux-surface motion.
        psi_n, plasma_mask, _, _ = estimate_flux_coordinates(
            psi,
            grid_r,
            grid_z,
            temperature=config.lcfs_temperature,
            separatrix_span_scale=config.separatrix_span_scale,
            separatrix_estimator=config.separatrix_estimator,
            axis_estimator=config.axis_estimator,
            axis_hint_r=hint_r,
            axis_hint_z=hint_z,
        )
        token_s = sample_grid(psi_n, batch["token_coords"], grid_r, grid_z).clamp(-0.5, 1.5)
        stride = config.source_stride
        source_psi_n = psi_n[:, ::stride, ::stride][:, : len(source_z), : len(source_r)]
        source_mask = plasma_mask[:, ::stride, ::stride][:, : len(source_z), : len(source_r)]
        current, pressure = profile_current_density(
            source_psi_n,
            source_mask,
            token_s,
            batch["token_values"],
            batch["token_mask"],
            params["pressure_correction"],
            params["pressure_scale"],
            params["ffprime_coefficients"],
            params["pressure_fraction"],
            batch["ip_abs"],
            source_r,
            source_z,
            source_weights,
            config.pressure_bandwidth,
        )
        if spatial_coefficients is not None:
            current = modulate_spatial_current(
                current,
                spatial_coefficients,
                batch["ip_abs"],
                source_r,
                source_z,
                source_weights,
                config.spatial_current_amplitude,
            )
        solved = model.vacuum_gain * batch["vacuum_psi"] + green_response(current, green).reshape(full_shape)
        psi = config.picard_damping * solved + (1.0 - config.picard_damping) * psi
    psi = psi + base_gauge[:, None, None]
    psi = params["flux_scale"][:, None, None] * psi + params["gauge_shift"][:, None, None]
    return {"psirz": psi, "current_density": current, "pressure": pressure, "psi_n": psi_n}


def target_current(
    batch: dict[str, torch.Tensor],
    operators: tuple[torch.Tensor, ...],
    config: NeuralGSConfig,
) -> torch.Tensor:
    _, source_r, source_z, weights, grid_r, grid_z = operators
    current = gs_current_from_flux(batch["target_psi"], grid_r, grid_z)
    current = torch.nn.functional.avg_pool2d(current[:, None], 5, stride=1, padding=2)[:, 0]
    stride_r = config.source_stride
    current = torch.relu(current[:, ::stride_r, ::stride_r])[:, : len(source_z)]
    source_r_count = len(source_r)
    current = current[:, :, :source_r_count]
    target_mask = torch.sigmoid(
        (1.0 - batch["target_psi_n"][:, ::stride_r, ::stride_r][:, : len(source_z), :source_r_count])
        / config.lcfs_temperature
    )
    current = current * target_mask
    integral = (current.flatten(1) * weights[None]).sum(dim=1).clamp_min(1.0e-8)
    return current * (batch["ip_abs"] / integral)[:, None, None]


def oracle_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    target_j: torch.Tensor,
    profile: torch.Tensor,
    initial_profile: torch.Tensor,
    frame_to_shot: torch.Tensor,
    temporal_weight: float,
    gradient_weight: float,
    topology_weight: float,
    source_weight: float,
    smoothness_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred = prediction["psirz"]
    target = batch["target_psi"]
    scale = target.flatten(1).std(dim=1).clamp_min(1.0e-4)
    flux = ((pred - target) / scale[:, None, None]).square().mean()
    pred_dz, pred_dr = torch.gradient(pred, dim=(1, 2))
    true_dz, true_dr = torch.gradient(target, dim=(1, 2))
    grad_scale = torch.sqrt((true_dz.square() + true_dr.square()).flatten(1).mean(dim=1)).clamp_min(1.0e-5)
    gradient = (
        ((pred_dz - true_dz).square() + (pred_dr - true_dr).square())
        / grad_scale[:, None, None].square()
    ).mean()
    topology = normalized_flux_topology_loss(pred, batch["target_psi_n"])
    source = (
        torch.asinh(prediction["current_density"] / 2.0e6)
        - torch.asinh(target_j / 2.0e6)
    ).square().mean()
    pressure = profile[:, 2 : 2 + 6]
    ffprime = profile[:, 2 + 6 :]
    smoothness = pressure.diff(n=2, dim=1).square().mean() + ffprime.diff(n=2, dim=1).square().mean()
    temporal = pred.new_zeros(())
    if temporal_weight > 0.0 and len(profile) > 1:
        adjacent = frame_to_shot[1:] == frame_to_shot[:-1]
        if adjacent.any():
            delta = profile[1:] - profile[:-1]
            temporal = delta[adjacent].square().mean()
    loss = (
        flux
        + gradient_weight * gradient
        + topology_weight * topology
        + source_weight * source
        + smoothness_weight * smoothness
        + temporal_weight * temporal
    )
    return loss, {
        "loss": loss.detach(),
        "flux": flux.detach(),
        "gradient": gradient.detach(),
        "topology": topology.detach(),
        "source": source.detach(),
        "smoothness": smoothness.detach(),
        "temporal": temporal.detach(),
    }


def optimize_stage(
    stage: Stage,
    parameters: OracleParameters,
    model: NeuralGS,
    batch: dict[str, torch.Tensor],
    initial: dict[str, torch.Tensor],
    target_j: torch.Tensor,
    frame_to_shot: torch.Tensor,
    operators: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    active = set_active_stage(parameters, stage)
    optimizer = torch.optim.Adam(active, lr=args.learning_rate)
    n = len(batch["ip_abs"])
    ff_order = _ff_order(parameters.ffprime.shape[1])
    ff_mask = parameters.ffprime.new_zeros(parameters.ffprime.shape[1])
    ff_mask[ff_order[: stage.ff_count]] = 1.0
    initial_profile = torch.cat(
        [
            initial["pressure_scale"].log()[:, None],
            torch.logit(initial["pressure_fraction"].clamp(1.0e-4, 1.0 - 1.0e-4))[:, None],
            initial["pressure_correction"],
            initial["ffprime_coefficients"],
        ],
        dim=1,
    ).detach()
    last_metrics: dict[str, float] = {}
    for step in range(args.steps_per_stage):
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        for start in range(0, n, args.batch_size):
            stop = min(start + args.batch_size, n)
            index = slice(start, stop)
            part = {key: value[index] for key, value in batch.items()}
            transformed = parameters.transformed(index)
            # Inactive FF knots stay exactly at their network initialization.
            transformed["ffprime_coefficients"] = (
                transformed["ffprime_coefficients"] * ff_mask
                + initial["ffprime_coefficients"][index] * (1.0 - ff_mask)
            )
            prediction = solve_oracle(
                model,
                part,
                transformed,
                initial["gauge"][index],
                initial.get("spatial_current_coefficients", None)[index]
                if initial.get("spatial_current_coefficients") is not None else None,
                operators,
            )
            loss, metrics = oracle_loss(
                prediction,
                part,
                target_j[index],
                parameters.profile_matrix()[index],
                initial_profile[index],
                frame_to_shot[index],
                args.temporal_weight if stage.temporal else 0.0,
                args.gradient_weight,
                args.topology_weight,
                args.source_weight,
                args.smoothness_weight,
            )
            weight = (stop - start) / n
            (weight * loss).backward()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + weight * float(value)
        if parameters.ffprime.grad is not None:
            parameters.ffprime.grad.mul_(ff_mask[None])
        torch.nn.utils.clip_grad_norm_(active, args.clip_grad)
        optimizer.step()
        last_metrics = totals
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps_per_stage:
            detail = " ".join(f"{key}={value:.5g}" for key, value in totals.items())
            print(f"  {stage.name} step {step + 1:03d}/{args.steps_per_stage}: {detail}", flush=True)
    prediction_parts: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            stop = min(start + args.batch_size, n)
            index = slice(start, stop)
            part = {key: value[index] for key, value in batch.items()}
            transformed = parameters.transformed(index)
            transformed["ffprime_coefficients"] = (
                transformed["ffprime_coefficients"] * ff_mask
                + initial["ffprime_coefficients"][index] * (1.0 - ff_mask)
            )
            result = solve_oracle(
                model,
                part,
                transformed,
                initial["gauge"][index],
                initial.get("spatial_current_coefficients", None)[index]
                if initial.get("spatial_current_coefficients") is not None else None,
                operators,
            )
            for key, value in result.items():
                if value is not None:
                    prediction_parts.setdefault(key, []).append(value.detach())
    return {key: torch.cat(values) for key, values in prediction_parts.items()}, last_metrics


def build_score_context(batch_cpu: dict[str, torch.Tensor], groups: list[dict]) -> dict:
    mask = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    r, z = mask["grid_R"], mask["grid_Z"]
    mask_coarse = mask["mask_coarse"].astype(bool)
    mask_f = mask_coarse.astype(np.float64)
    # `canonical_current_sign` records Ip direction and must never be used as a
    # flux-storage conversion.  Cache targets use TARGET_AXIS_SIGN explicitly:
    # canonical_psi = target_flux_factor * stored_DIII-D_psi.
    target_flux_factor = float(TARGET_AXIS_SIGN[local_score.MACHINE])
    shots = []
    refs = []
    psi_sum = psi_sumsq = psi_n = 0.0
    scal_sum = np.zeros(N_SCALARS)
    scal_sumsq = np.zeros(N_SCALARS)
    scal_n = np.zeros(N_SCALARS)
    cons_sum = np.zeros(N_CONS)
    cons_sumsq = np.zeros(N_CONS)
    cons_n = np.zeros(N_CONS)
    target = stored_flux_from_canonical(batch_cpu["target_psi"].numpy(), local_score.MACHINE)
    q95 = batch_cpu["target_q95"].numpy().astype(np.float64)
    beta = batch_cpu["target_betaN"].numpy().astype(np.float64)
    for group in groups:
        section = slice(group["start"], group["stop"])
        shot = {"psi": target[section], "q95": q95[section], "betaN": beta[section]}
        shots.append(shot)
        ref = local_score.build_reference(shot["psi"], r, z, mask_coarse, mask_f)
        refs.append(ref)
        values = shot["psi"].astype(np.float64)
        finite = np.isfinite(values)
        psi_sum += float(values[finite].sum())
        psi_sumsq += float((values[finite] ** 2).sum())
        psi_n += int(finite.sum())
        for index, name in enumerate(SCALARS):
            values = shot[name][np.isfinite(shot[name])]
            scal_sum[index] += values.sum()
            scal_sumsq[index] += (values ** 2).sum()
            scal_n[index] += values.size
        cons_gt, cons_mask = ref[1], ref[2]
        for index in range(N_CONS):
            values = cons_gt[cons_mask[:, index], index]
            cons_sum[index] += values.sum()
            cons_sumsq[index] += (values ** 2).sum()
            cons_n[index] += values.size
    ref_stats = {
        "psi_sum": psi_sum, "psi_sumsq": psi_sumsq, "psi_n": psi_n,
        "scal_sum": scal_sum, "scal_sumsq": scal_sumsq, "scal_n": scal_n,
        "cons_sum": cons_sum, "cons_sumsq": cons_sumsq, "cons_n": cons_n,
    }
    means = (
        psi_sum / psi_n if psi_n else 0.0,
        np.divide(scal_sum, scal_n, out=np.zeros_like(scal_sum), where=scal_n > 0),
        np.divide(cons_sum, cons_n, out=np.zeros_like(cons_sum), where=cons_n > 0),
    )
    return {
        "r": r, "z": z, "mask_coarse": mask_coarse, "mask_f": mask_f,
        "target_flux_factor": target_flux_factor, "shots": shots, "refs": refs,
        "ref_stats": ref_stats, "means": means,
    }


def exact_score(
    psirz_canonical: np.ndarray,
    q95: np.ndarray,
    beta_n: np.ndarray,
    groups: list[dict],
    context: dict,
) -> dict:
    predictions = []
    for group in groups:
        section = slice(group["start"], group["stop"])
        predictions.append(
            {
                "psirz": stored_flux_from_canonical(
                    psirz_canonical[section], local_score.MACHINE
                ),
                "q95": q95[section],
                "betaN": beta_n[section],
            }
        )
    totals = {sign: 0.0 for sign in PSI_SIGNS}
    for shot, prediction in zip(context["shots"], predictions):
        residuals, _, _ = local_score.psi_residuals(
            shot["psi"], prediction["psirz"], context["means"][0]
        )
        for sign in PSI_SIGNS:
            totals[sign] += residuals[sign]
    psi_sign = min(PSI_SIGNS, key=lambda sign: totals[sign])
    accumulator = Accum()
    accumulator.psi_sign = psi_sign
    for shot, ref, prediction in zip(context["shots"], context["refs"], predictions):
        accumulator.add(
            local_score.score_shot(
                shot,
                ref,
                prediction,
                context["r"],
                context["z"],
                context["mask_coarse"],
                context["mask_f"],
                psi_sign,
                context["means"],
            )
        )
    result = finalize_machine(accumulator, context["ref_stats"])
    result["one_minus_dlcfs"] = 1.0 - min(1.0, result["dlcfs"])
    result["S_psi"] = round(
        0.55 * max(0.0, result["r2_psi"])
        + 0.10 * result["one_minus_dlcfs"]
        + 0.20 * result["consistency"],
        6,
    )
    return result


def score_both_ceilings(
    psirz: torch.Tensor,
    initial: dict[str, torch.Tensor],
    batch_cpu: dict[str, torch.Tensor],
    groups: list[dict],
    context: dict,
) -> dict:
    flux = psirz.detach().cpu().numpy().astype(np.float64)
    truth_q = batch_cpu["target_q95"].numpy().astype(np.float64)
    truth_b = batch_cpu["target_betaN"].numpy().astype(np.float64)
    deploy_q = initial["q95"].detach().cpu().numpy().astype(np.float64)
    deploy_b = initial["betaN"].detach().cpu().numpy().astype(np.float64)
    return {
        "physics": exact_score(flux, truth_q, truth_b, groups, context),
        "deploy": exact_score(flux, deploy_q, deploy_b, groups, context),
    }


def marginal(current: dict, previous: dict | None) -> dict:
    if previous is None:
        return {}
    keys = ("S", "S_psi", "r2_psi", "one_minus_dlcfs", "consistency", "r2_qb")
    return {key: round(current[key] - previous[key], 6) for key in keys}


def correction_covariance_spectrum(
    parameters: OracleParameters,
    initial: dict[str, torch.Tensor],
    count: int,
) -> dict:
    """Empirical spectrum of the fitted 18-D oracle correction vectors."""

    fitted = parameters.profile_matrix().detach().cpu().double().numpy()
    base = np.concatenate(
        [
            initial["pressure_scale"].detach().cpu().double().log().numpy()[:, None],
            torch.logit(initial["pressure_fraction"].clamp(1.0e-4, 1.0 - 1.0e-4)).detach().cpu().double().numpy()[:, None],
            initial["pressure_correction"].detach().cpu().double().numpy(),
            initial["ffprime_coefficients"].detach().cpu().double().numpy(),
        ],
        axis=1,
    )
    residual = fitted - base
    covariance = residual.T @ residual / max(len(residual), 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1].clip(min=0.0)
    total = eigenvalues.sum()
    probabilities = eigenvalues / total if total > 0 else eigenvalues
    entropy_rank = float(np.exp(-(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])).sum())) if total > 0 else 0.0
    threshold_rank = int(np.sum(eigenvalues > (eigenvalues[0] * 1.0e-3))) if len(eigenvalues) and eigenvalues[0] > 0 else 0
    return {
        "kind": "empirical_oracle_correction_covariance_not_measurement_gramian",
        "dimension": int(residual.shape[1]),
        "frames": int(len(residual)),
        "eigenvalues": eigenvalues[:count].tolist(),
        "entropy_effective_rank": entropy_rank,
        "relative_1e-3_rank": threshold_rank,
    }


def local_gs_fisher_spectrum(
    parameters: OracleParameters,
    model: NeuralGS,
    batch: dict[str, torch.Tensor],
    initial: dict[str, torch.Tensor],
    operators: tuple[torch.Tensor, ...],
    n_frames: int,
    epsilon: float,
    field_stride: int,
) -> dict:
    """Estimate local 18-D Gauss--Newton/Fisher spectra by central differences.

    The observations here are normalized oracle flux pixels, not Thomson
    measurements.  This diagnoses profile/solver identifiability at the oracle
    stage.  The later test-time-refinement experiment must separately construct
    the full and detached Thomson Gramians.
    """

    if n_frames <= 0:
        return {"kind": "local_gs_flux_gauss_newton", "frames": 0, "skipped": True}
    total_frames = len(batch["ip_abs"])
    selected = evenly_spaced_frame_indices(total_frames, min(n_frames, total_frames))
    profile = parameters.profile_matrix().detach()
    # Work in normalized coefficient coordinates so unlike parameter scales do
    # not manufacture large/small curvature eigenvalues.
    normalization = profile.std(dim=0, correction=0).clamp_min(0.05)
    gramians = []
    per_frame_eigenvalues = []
    with torch.no_grad():
        for frame in selected:
            index = slice(int(frame), int(frame) + 1)
            part = {key: value[index] for key, value in batch.items()}
            base = parameters.transformed(index)
            theta = profile[index].clone()
            columns = []
            for direction in range(theta.shape[1]):
                step = float(epsilon) * normalization[direction]
                fields = []
                for sign in (-1.0, 1.0):
                    perturbed = theta.clone()
                    perturbed[:, direction] += sign * step
                    trial = dict(base)
                    trial["pressure_scale"] = perturbed[:, 0].clamp(-3.0, 3.0).exp()
                    trial["pressure_fraction"] = torch.sigmoid(perturbed[:, 1])
                    trial["pressure_correction"] = perturbed[:, 2:8]
                    trial["ffprime_coefficients"] = perturbed[:, 8:18]
                    solved = solve_oracle(
                        model,
                        part,
                        trial,
                        initial["gauge"][index],
                        initial.get("spatial_current_coefficients", None)[index]
                        if initial.get("spatial_current_coefficients") is not None else None,
                        operators,
                    )["psirz"]
                    scale = part["target_psi"].flatten(1).std(dim=1).clamp_min(1.0e-4)
                    fields.append((solved[:, ::field_stride, ::field_stride] / scale[:, None, None]).flatten())
                # Derivative is with respect to the normalized coordinate eta,
                # where theta = theta_0 + normalization * eta.
                columns.append((fields[1] - fields[0]) / (2.0 * float(epsilon)))
            jacobian = torch.stack(columns, dim=1).double()
            gramian = jacobian.transpose(0, 1) @ jacobian / max(jacobian.shape[0], 1)
            gramian = 0.5 * (gramian + gramian.transpose(0, 1))
            gramians.append(gramian)
            per_frame_eigenvalues.append(
                torch.linalg.eigvalsh(gramian).flip(0).clamp_min(0.0).cpu().numpy()
            )
    global_gramian = torch.stack(gramians).mean(dim=0)
    eigenvalues = torch.linalg.eigvalsh(global_gramian).flip(0).clamp_min(0.0).cpu().numpy()
    total = float(eigenvalues.sum())
    probability = eigenvalues / total if total > 0 else eigenvalues
    positive = probability > 0
    entropy_rank = float(np.exp(-(probability[positive] * np.log(probability[positive])).sum())) if positive.any() else 0.0
    threshold_rank = int(np.sum(eigenvalues > eigenvalues[0] * 1.0e-3)) if len(eigenvalues) and eigenvalues[0] > 0 else 0
    return {
        "kind": "local_gs_flux_gauss_newton",
        "coordinate_system": "per_coefficient_standardized",
        "frames": int(len(selected)),
        "frame_indices": selected.tolist(),
        "central_difference_epsilon": float(epsilon),
        "field_stride": int(field_stride),
        "coefficient_normalization": normalization.cpu().tolist(),
        "mean_gramian_eigenvalues": eigenvalues.tolist(),
        "per_frame_eigenvalues": np.asarray(per_frame_eigenvalues).tolist(),
        "entropy_effective_rank": entropy_rank,
        "relative_1e-3_rank": threshold_rank,
    }


def parse_stages(ff_counts: str, temporal_weight: float) -> list[Stage]:
    counts = sorted({int(value) for value in ff_counts.split(",") if value.strip()})
    if any(value < 1 or value > 10 for value in counts):
        raise ValueError("--ff-counts values must be in [1, 10]")
    stages = [Stage("affine"), Stage("balance"), Stage("pressure")]
    stages.extend(Stage(f"ffprime_{count}", ff_count=count) for count in counts)
    if temporal_weight > 0.0:
        stages.append(Stage("temporal", ff_count=max(counts, default=10), temporal=True))
    return stages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ROOT / "my_experiments/artifacts/neuralgs_topology_holdout.pt")
    parser.add_argument(
        "--direct-scalar-artifact",
        type=Path,
        default=ROOT / "my_experiments/artifacts/neuralgs_direct_scalar_topology.pt",
        help="frozen deployment scalar head; pass a nonexistent/empty value only with --base-scalars",
    )
    parser.add_argument("--base-scalars", action="store_true", help="use the base model scalar head")
    parser.add_argument("--cache", type=Path, default=ROOT / "my_experiments/neuralgs_full_data_cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "my_experiments/oracle_inverse_profiles")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-val-shots", type=int, default=200)
    parser.add_argument("--n-shots", type=int, default=200)
    parser.add_argument("--frames-per-shot", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps-per-stage", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--ff-counts", default="1,2,4,6,8,10")
    parser.add_argument("--gradient-weight", type=float, default=0.12)
    parser.add_argument("--topology-weight", type=float, default=0.20)
    parser.add_argument("--source-weight", type=float, default=0.02)
    parser.add_argument("--smoothness-weight", type=float, default=1.0e-4)
    parser.add_argument("--temporal-weight", type=float, default=0.0)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--spectrum-values", type=int, default=18)
    parser.add_argument("--curvature-frames", type=int, default=4)
    parser.add_argument("--curvature-epsilon", type=float, default=0.01)
    parser.add_argument("--curvature-field-stride", type=int, default=8)
    parser.add_argument("--skip-exact-score", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    checkpoint_path = args.output_dir / "oracle_state.pt"
    started = time.monotonic()

    print(f"Loading frozen artifact {args.artifact}", flush=True)
    model, config, artifact = load_frozen_model(args.artifact, device)
    batch_cpu, groups, names, frame_to_shot_np = load_equal_shot_sample(
        args.cache, args.n_val_shots, args.seed, args.n_shots, args.frames_per_shot
    )
    batch = _move_batch(batch_cpu, device)
    frame_to_shot = torch.as_tensor(frame_to_shot_np, device=device)
    green_np, source_r_np, source_z_np, weights_np = plasma_green_matrix(
        batch_cpu["grid_r"].numpy(),
        batch_cpu["grid_z"].numpy(),
        config.source_stride,
        config.green_softening_cells,
    )
    operators = tuple(
        torch.as_tensor(value, device=device)
        for value in (
            green_np, source_r_np, source_z_np, weights_np,
            batch_cpu["grid_r"].numpy(), batch_cpu["grid_z"].numpy(),
        )
    )
    print(f"Frozen model inference on {len(batch['ip_abs'])} frames from {len(groups)} shots", flush=True)
    initial = batched_model_outputs(model, batch, operators, args.batch_size)
    scalar_provenance = apply_deployment_scalar_head(
        initial,
        batch,
        operators,
        args.artifact,
        None if args.base_scalars else args.direct_scalar_artifact,
        model,
        device,
    )
    parameters = OracleParameters(initial, batch["target_psi"]).to(device)
    target_j = target_current(batch, operators, config).detach()
    stages = parse_stages(args.ff_counts, args.temporal_weight)

    context = None if args.skip_exact_score else build_score_context(batch_cpu, groups)
    report = {
        "experiment": "nested_forward_inverse_profile_oracle",
        "architecture_frozen": True,
        "artifact": str(args.artifact.resolve()),
        "artifact_metadata": artifact.get("metadata", {}),
        "deployment_scalars": scalar_provenance,
        "cache": str(args.cache.resolve()),
        "config": config.to_dict(),
        "sampling": {
            "seed": args.seed,
            "validation_shots": args.n_val_shots,
            "selected_shots": len(groups),
            "frames_per_shot": args.frames_per_shot,
            "frames": len(batch["ip_abs"]),
            "shot_names": names,
        },
        "optimization": {
            key: getattr(args, key)
            for key in (
                "batch_size", "steps_per_stage", "learning_rate", "ff_counts",
                "gradient_weight", "topology_weight", "source_weight",
                "smoothness_weight", "temporal_weight", "clip_grad",
            )
        },
        "stages": [],
    }

    if context is not None:
        print("Building exact challenge reference and scoring frozen baseline", flush=True)
        baseline_scores = score_both_ceilings(
            initial["psirz"], initial, batch_cpu, groups, context
        )
        report["baseline"] = baseline_scores
        _write_json(report_path, report)

    previous_physics = report.get("baseline", {}).get("physics")
    for stage in stages:
        print(f"\nStarting nested stage: {stage.name}", flush=True)
        prediction, losses = optimize_stage(
            stage, parameters, model, batch, initial, target_j,
            frame_to_shot, operators, args,
        )
        torch.save(
            {
                "stage": stage.name,
                "parameters": parameters.state_dict(),
                "psirz": prediction["psirz"].detach().cpu(),
                "initial_q95": initial["q95"].detach().cpu(),
                "initial_betaN": initial["betaN"].detach().cpu(),
                "target_q95": batch_cpu["target_q95"],
                "target_betaN": batch_cpu["target_betaN"],
                "groups": groups,
                "config": config.to_dict(),
            },
            checkpoint_path,
        )
        stage_report = {
            "name": stage.name,
            "ffprime_count": stage.ff_count,
            "training_loss": losses,
        }
        if context is not None:
            print(f"Exact challenge scoring stage {stage.name}", flush=True)
            scores = score_both_ceilings(
                prediction["psirz"], initial, batch_cpu, groups, context
            )
            stage_report["scores"] = scores
            stage_report["marginal_physics"] = marginal(scores["physics"], previous_physics)
            previous_physics = scores["physics"]
            print(
                f"  physics S={scores['physics']['S']:.4f} "
                f"Spsi={scores['physics']['S_psi']:.4f}; "
                f"deploy S={scores['deploy']['S']:.4f}",
                flush=True,
            )
        report["stages"].append(stage_report)
        report["elapsed_seconds"] = time.monotonic() - started
        _write_json(report_path, report)

    report["oracle_correction_spectrum"] = correction_covariance_spectrum(
        parameters, initial, args.spectrum_values
    )
    print("Estimating local GS/profile Gauss-Newton spectrum", flush=True)
    report["local_gs_fisher"] = local_gs_fisher_spectrum(
        parameters,
        model,
        batch,
        initial,
        operators,
        args.curvature_frames,
        args.curvature_epsilon,
        args.curvature_field_stride,
    )
    report["elapsed_seconds"] = time.monotonic() - started
    if report.get("stages") and report["stages"][-1].get("scores"):
        ceiling = report["stages"][-1]["scores"]["physics"]["S"]
        if ceiling > 0.97:
            decision = "go: solver/profile representation clears 0.97; proceed to explicit profile supervision"
        elif ceiling >= 0.94:
            decision = "inspect: 0.94-0.97; isolate the limiting score term before ML changes"
        else:
            decision = "stop: below 0.94; fix GS/profile/boundary representation before encoder or MAST work"
        report["decision"] = decision
        print(f"\nDecision: {decision}", flush=True)
    _write_json(report_path, report)
    print(f"Wrote {report_path} and {checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
