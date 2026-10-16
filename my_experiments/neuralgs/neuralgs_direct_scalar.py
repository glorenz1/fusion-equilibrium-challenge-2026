"""Frozen-NeuralGS direct q95/betaN head and isolated inference helper.

The head deliberately consumes only machine-independent quantities already
available to NeuralGS: its Thomson/profile latent, dimensionless global inputs,
physical scalar estimates, normalized solved-field moments, current/pressure
moments, and masked Thomson summaries.  It does not receive a machine ID,
named coil channels, or any EFIT-derived value at inference time.

This module is intentionally kept under ``my_experiments``.  Nothing imports it
from the submission path unless an experiment explicitly opts in.
"""

from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from neuralgs.config import NeuralGSConfig
from neuralgs.data import preprocess_row
from neuralgs.geometry import plasma_green_matrix
from neuralgs.model import NeuralGS


ARTIFACT_VERSION = 1
FEATURE_VERSION = 1
SCALAR_NAMES = ("q95", "betaN")
SCALAR_BOUNDS = {"q95": (0.1, 30.0), "betaN": (0.0, 10.0)}
DEFAULT_DIRECT_SCALAR_ARTIFACT = (
    Path(__file__).resolve().parents[1] / "artifacts" / "neuralgs_direct_scalar.pt"
)

BASE_INPUT_KEYS = (
    "vacuum_psi",
    "ip_abs",
    "global_features",
    "token_coords",
    "token_values",
    "token_mask",
)


def sha256_file(path: Path | str) -> str:
    """Hash an artifact so a scalar head cannot silently use a different base."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weighted_moments(
    mass: torch.Tensor,
    r_normalized: torch.Tensor,
    z_normalized: torch.Tensor,
) -> torch.Tensor:
    """Seven bounded spatial moments for a non-negative ``[B,Z,R]`` mass."""

    mass = torch.relu(mass.float())
    denominator = mass.sum(dim=(1, 2)).clamp_min(1.0e-12)
    mean_r = (mass * r_normalized[None]).sum(dim=(1, 2)) / denominator
    mean_z = (mass * z_normalized[None]).sum(dim=(1, 2)) / denominator
    dr = r_normalized[None] - mean_r[:, None, None]
    dz = z_normalized[None] - mean_z[:, None, None]
    var_r = (mass * dr.square()).sum(dim=(1, 2)) / denominator
    var_z = (mass * dz.square()).sum(dim=(1, 2)) / denominator
    std_r = torch.sqrt(var_r.clamp_min(1.0e-10))
    std_z = torch.sqrt(var_z.clamp_min(1.0e-10))
    covariance = (mass * dr * dz).sum(dim=(1, 2)) / denominator
    correlation = covariance / (std_r * std_z).clamp_min(1.0e-6)
    skew_r = (mass * dr.pow(3)).sum(dim=(1, 2)) / (
        denominator * std_r.pow(3).clamp_min(1.0e-6)
    )
    skew_z = (mass * dz.pow(3)).sum(dim=(1, 2)) / (
        denominator * std_z.pow(3).clamp_min(1.0e-6)
    )
    return torch.stack(
        [mean_r, mean_z, std_r, std_z, correlation, skew_r, skew_z], dim=1
    ).clamp(-8.0, 8.0)


def _weighted_point_moments(
    mass: torch.Tensor,
    r_normalized: torch.Tensor,
    z_normalized: torch.Tensor,
) -> torch.Tensor:
    """Seven spatial moments for per-frame point clouds shaped ``[B,N]``."""

    mass = torch.relu(mass.float())
    denominator = mass.sum(dim=1).clamp_min(1.0e-12)
    mean_r = (mass * r_normalized).sum(dim=1) / denominator
    mean_z = (mass * z_normalized).sum(dim=1) / denominator
    dr = r_normalized - mean_r[:, None]
    dz = z_normalized - mean_z[:, None]
    var_r = (mass * dr.square()).sum(dim=1) / denominator
    var_z = (mass * dz.square()).sum(dim=1) / denominator
    std_r = torch.sqrt(var_r.clamp_min(1.0e-10))
    std_z = torch.sqrt(var_z.clamp_min(1.0e-10))
    covariance = (mass * dr * dz).sum(dim=1) / denominator
    correlation = covariance / (std_r * std_z).clamp_min(1.0e-6)
    skew_r = (mass * dr.pow(3)).sum(dim=1) / (
        denominator * std_r.pow(3).clamp_min(1.0e-6)
    )
    skew_z = (mass * dz.pow(3)).sum(dim=1) / (
        denominator * std_z.pow(3).clamp_min(1.0e-6)
    )
    return torch.stack(
        [mean_r, mean_z, std_r, std_z, correlation, skew_r, skew_z], dim=1
    ).clamp(-8.0, 8.0)


def _masked_mean_std(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask_f = mask.to(values.dtype)
    denominator = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * mask_f[..., None]).sum(dim=1) / denominator
    variance = (
        (values - mean[:, None]).square() * mask_f[..., None]
    ).sum(dim=1) / denominator
    return mean, torch.sqrt(variance.clamp_min(0.0))


def direct_scalar_feature_names(d_model: int) -> list[str]:
    names = [f"latent_{index:03d}" for index in range(int(d_model))]
    names += [f"global_{index}" for index in range(6)]
    names += ["log_q_physics", "log1p_beta_physics", "log_q_base", "log1p_beta_base"]
    names += [
        "psi_mean_asinh", "psi_std_log", "psi_span_log",
        "vacuum_mean_asinh", "vacuum_std_log", "vacuum_span_log",
        "plasma_area_fraction", "plasma_minor_fraction", "plasma_elongation_proxy",
    ]
    names += [f"plasma_{name}" for name in (
        "r_mean", "z_mean", "r_std", "z_std", "rz_corr", "r_skew", "z_skew"
    )]
    names += [f"current_{name}" for name in (
        "r_mean", "z_mean", "r_std", "z_std", "rz_corr", "r_skew", "z_skew"
    )]
    names += ["current_peak_log", "pressure_mean_log", "pressure_std_log", "pressure_peak_log"]
    names += ["token_valid_fraction", "token_core_fraction", "token_edge_fraction"]
    names += [f"token_mean_{name}" for name in ("log_te", "log_ne", "log_pe", "age")]
    names += [f"token_std_{name}" for name in ("log_te", "log_ne", "log_pe", "age")]
    names += [f"token_pressure_{name}" for name in (
        "r_mean", "z_mean", "r_std", "z_std", "rz_corr", "r_skew", "z_skew"
    )]
    return names


def build_direct_scalar_features(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    source_r: torch.Tensor,
    source_z: torch.Tensor,
    source_weights: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    lcfs_temperature: float,
) -> torch.Tensor:
    """Build one fixed-width, machine-independent feature vector per frame."""

    latent = output["latent"].float()
    global_features = batch["global_features"].float()
    scalar_features = torch.stack(
        [
            torch.log(output["q95_physics"].float().clamp_min(0.05)),
            torch.log1p(output["betaN_physics"].float().clamp_min(0.0)),
            torch.log(output["q95"].float().clamp_min(0.05)),
            torch.log1p(output["betaN"].float().clamp_min(0.0)),
        ],
        dim=1,
    )

    psi = output["psirz"].float()
    vacuum = batch["vacuum_psi"].float()

    def field_stats(field: torch.Tensor) -> torch.Tensor:
        flat = field.flatten(1)
        mean = flat.mean(dim=1)
        std = flat.std(dim=1, unbiased=False)
        span = flat.max(dim=1).values - flat.min(dim=1).values
        return torch.stack(
            [
                torch.asinh(mean / 0.5),
                torch.log1p(std / 0.05),
                torch.log1p(span / 0.05),
            ],
            dim=1,
        )

    stride = max((psi.shape[-1] - 1) // max(len(source_r) - 1, 1), 1)
    source_s = output["psi_n"].float()[:, ::stride, ::stride]
    source_s = source_s[:, : len(source_z), : len(source_r)]
    soft_mask = torch.sigmoid((1.0 - source_s) / max(float(lcfs_temperature), 1.0e-3))
    area_weights = source_weights.float().reshape(1, len(source_z), len(source_r))
    r_span = (grid_r[-1] - grid_r[0]).float().clamp_min(1.0e-6)
    z_span = (grid_z[-1] - grid_z[0]).float().clamp_min(1.0e-6)
    rr_physical, zz_physical = torch.meshgrid(source_r.float(), source_z.float(), indexing="xy")
    rr = 2.0 * (rr_physical - grid_r[0].float()) / r_span - 1.0
    zz = 2.0 * (zz_physical - grid_z[0].float()) / z_span - 1.0

    plasma_mass = soft_mask * area_weights
    area = plasma_mass.sum(dim=(1, 2))
    area_fraction = area / (r_span * z_span).clamp_min(1.0e-8)
    minor_fraction = torch.sqrt(area.clamp_min(1.0e-10) / math.pi) / torch.sqrt(
        (r_span * z_span).clamp_min(1.0e-8)
    )
    plasma_moments = _weighted_moments(plasma_mass, rr, zz)
    elongation = plasma_moments[:, 3] / plasma_moments[:, 2].clamp_min(1.0e-4)
    plasma_features = torch.column_stack(
        [area_fraction, minor_fraction, torch.log(elongation.clamp(0.1, 10.0)), plasma_moments]
    )

    current = torch.relu(output["current_density"].float())
    current_mass = current * area_weights
    current_moments = _weighted_moments(current_mass, rr, zz)
    current_peak = torch.log1p(current.flatten(1).max(dim=1).values / 1.0e5)

    pressure = torch.relu(output["pressure"].float())
    pressure_mass = plasma_mass * rr_physical.clamp_min(0.02)[None]
    pressure_denominator = pressure_mass.sum(dim=(1, 2)).clamp_min(1.0e-12)
    pressure_mean = (pressure * pressure_mass).sum(dim=(1, 2)) / pressure_denominator
    pressure_variance = (
        (pressure - pressure_mean[:, None, None]).square() * pressure_mass
    ).sum(dim=(1, 2)) / pressure_denominator
    pressure_features = torch.stack(
        [
            torch.log1p(pressure_mean / 1.0e3),
            torch.log1p(torch.sqrt(pressure_variance.clamp_min(0.0)) / 1.0e3),
            torch.log1p(pressure.flatten(1).max(dim=1).values / 1.0e3),
        ],
        dim=1,
    )

    token_values = batch["token_values"].float()
    token_mask = batch["token_mask"].bool()
    token_mean, token_std = _masked_mean_std(token_values[..., :4], token_mask)
    token_count = token_mask.sum(dim=1).float()
    token_valid = token_count / max(token_mask.shape[1], 1)
    system = token_values[..., 4]
    token_core = (token_mask & (system < 0.0)).sum(dim=1).float() / token_count.clamp_min(1.0)
    token_edge = (token_mask & (system > 0.0)).sum(dim=1).float() / token_count.clamp_min(1.0)
    token_coords = batch["token_coords"].float()
    token_r = 2.0 * (token_coords[..., 0] - grid_r[0].float()) / r_span - 1.0
    token_z = 2.0 * (token_coords[..., 1] - grid_z[0].float()) / z_span - 1.0
    token_pressure_mass = (
        torch.expm1(token_values[..., 2].clamp(0.0, 20.0)) * token_mask.float()
    )
    token_pressure_moments = _weighted_point_moments(
        token_pressure_mass, token_r, token_z
    )
    token_features = torch.column_stack(
        [
            token_valid,
            token_core,
            token_edge,
            token_mean,
            token_std,
            token_pressure_moments,
        ]
    )

    features = torch.column_stack(
        [
            latent,
            global_features,
            scalar_features,
            field_stats(psi),
            field_stats(vacuum),
            plasma_features,
            current_moments,
            current_peak,
            pressure_features,
            token_features,
        ]
    )
    expected = len(direct_scalar_feature_names(latent.shape[1]))
    if features.shape[1] != expected:
        raise RuntimeError(f"direct scalar feature width {features.shape[1]} != {expected}")
    return torch.nan_to_num(features, nan=0.0, posinf=12.0, neginf=-12.0)


class DirectScalarHead(nn.Module):
    """Small nonlinear head predicting standardized q95 and betaN directly."""

    def __init__(
        self,
        n_features: int,
        hidden_width: int = 128,
        bottleneck_width: int = 64,
        dropout: float = 0.05,
        zero_initialize: bool = True,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.hidden_width = int(hidden_width)
        self.bottleneck_width = int(bottleneck_width)
        self.dropout = float(dropout)
        self.network = nn.Sequential(
            nn.LayerNorm(self.n_features),
            nn.Linear(self.n_features, self.hidden_width),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_width, self.bottleneck_width),
            nn.SiLU(),
            nn.Linear(self.bottleneck_width, len(SCALAR_NAMES)),
        )
        if zero_initialize:
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, standardized_features: torch.Tensor) -> torch.Tensor:
        return self.network(standardized_features)

    def config_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "hidden_width": self.hidden_width,
            "bottleneck_width": self.bottleneck_width,
            "dropout": self.dropout,
        }


def _artifact_tensor(artifact: dict, key: str, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(artifact[key], device=device, dtype=torch.float32)


def load_direct_scalar_artifact(
    path: Path | str = DEFAULT_DIRECT_SCALAR_ARTIFACT,
    device: torch.device | str = "cpu",
) -> tuple[DirectScalarHead, dict]:
    device = torch.device(device)
    artifact = torch.load(Path(path), map_location=device, weights_only=False)
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(f"unsupported direct scalar artifact: {artifact.get('artifact_version')}")
    if artifact.get("feature_version") != FEATURE_VERSION:
        raise ValueError(f"unsupported direct scalar features: {artifact.get('feature_version')}")
    head = DirectScalarHead(**artifact["head_config"], zero_initialize=False).to(device)
    head.load_state_dict(artifact["head_state"])
    head.eval()
    return head, artifact


def predict_from_features(
    features: torch.Tensor,
    head: DirectScalarHead,
    artifact: dict,
) -> torch.Tensor:
    """Apply saved normalization and return physical ``[q95,betaN]`` values."""

    device = features.device
    active = torch.as_tensor(artifact["active_feature_indices"], device=device, dtype=torch.long)
    mean = _artifact_tensor(artifact, "feature_mean", device)
    scale = _artifact_tensor(artifact, "feature_scale", device)
    standardized = ((features[:, active] - mean) / scale).clamp(-12.0, 12.0)
    normalized_prediction = head(standardized)
    target_mean = _artifact_tensor(artifact, "target_mean", device)
    target_scale = _artifact_tensor(artifact, "target_scale", device)
    physical = normalized_prediction * target_scale + target_mean
    q_low, q_high = SCALAR_BOUNDS["q95"]
    b_low, b_high = SCALAR_BOUNDS["betaN"]
    return torch.column_stack(
        [physical[:, 0].clamp(q_low, q_high), physical[:, 1].clamp(b_low, b_high)]
    )


@lru_cache(maxsize=2)
def _load_inference_models(base_path: str, scalar_path: str, device_name: str):
    device = torch.device(device_name)
    base_checkpoint = torch.load(base_path, map_location=device, weights_only=False)
    if base_checkpoint.get("artifact_version") != 1:
        raise ValueError("unsupported NeuralGS base artifact")
    config = NeuralGSConfig.from_dict(base_checkpoint["config"])
    base = NeuralGS(config).to(device)
    base.load_state_dict(base_checkpoint["model_state"])
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    head, artifact = load_direct_scalar_artifact(scalar_path, device)
    expected_hash = artifact.get("base_artifact_sha256")
    if expected_hash and sha256_file(base_path) != expected_hash:
        raise ValueError("direct scalar head was trained against a different NeuralGS artifact")
    return base, config, head, artifact, device


@lru_cache(maxsize=4)
def _cached_green(r_bytes: bytes, z_bytes: bytes, stride: int, softening: float):
    r = np.frombuffer(r_bytes, dtype=np.float32).copy()
    z = np.frombuffer(z_bytes, dtype=np.float32).copy()
    return plasma_green_matrix(r, z, stride, softening)


def predict_row_scalars(
    row: dict,
    base_artifact: Path | str,
    scalar_artifact: Path | str = DEFAULT_DIRECT_SCALAR_ARTIFACT,
    device_name: str | None = None,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Predict only q95/betaN for a raw challenge row without reading targets."""

    if device_name is None:
        requested = os.environ.get("NEURALGS_DEVICE", "auto")
        device_name = (
            "cuda" if requested == "auto" and torch.cuda.is_available()
            else "cpu" if requested == "auto"
            else requested
        )
    base_path = str(Path(base_artifact).resolve())
    scalar_path = str(Path(scalar_artifact).resolve())
    base, config, head, artifact, device = _load_inference_models(
        base_path, scalar_path, device_name
    )
    prepared = preprocess_row(row, config, include_targets=False)
    r = np.asarray(prepared["grid_r"], dtype=np.float32)
    z = np.asarray(prepared["grid_z"], dtype=np.float32)
    green_np, sr_np, sz_np, weights_np = _cached_green(
        r.tobytes(), z.tobytes(), config.source_stride, config.green_softening_cells
    )
    operators = [
        torch.as_tensor(value, device=device)
        for value in (green_np, sr_np, sz_np, weights_np, r, z)
    ]
    green, source_r, source_z, source_weights, grid_r, grid_z = operators
    parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(prepared["ip_abs"]), batch_size):
            stop = min(start + batch_size, len(prepared["ip_abs"]))
            batch = {
                key: torch.as_tensor(prepared[key][start:stop], device=device)
                for key in BASE_INPUT_KEYS
            }
            output = base(
                batch, green, source_r, source_z, source_weights, grid_r, grid_z
            )
            features = build_direct_scalar_features(
                output,
                batch,
                source_r,
                source_z,
                source_weights,
                grid_r,
                grid_z,
                config.lcfs_temperature,
            )
            parts.append(predict_from_features(features, head, artifact).cpu())
    prediction = torch.cat(parts).numpy().astype(np.float32)
    return {"q95": prediction[:, 0], "betaN": prediction[:, 1]}
