"""Low-rank/convolutional correction for frozen NeuralGS flux.

The calibrator consumes only target-free NeuralGS inputs and solved-state
features.  It adds a smooth, bounded correction to the inferred plasma
response while preserving the known PF-coil vacuum field and learned gauge.
Legacy artifacts store the exact training coordinate vectors and take an
identity path on any other grid. Transfer-invariant artifacts operate on a
canonical pixel canvas and accept any finite grid of the trained shape.
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
from torch.nn import functional as F

from my_experiments.neuralgs.neuralgs_direct_scalar import (
    BASE_INPUT_KEYS,
    FEATURE_VERSION as DIRECT_FEATURE_VERSION,
    build_direct_scalar_features,
    load_direct_scalar_artifact,
    predict_from_features,
)
from neuralgs.config import NeuralGSConfig
from neuralgs.data import preprocess_row
from neuralgs.geometry import plasma_green_matrix
from neuralgs.model import NeuralGS
from neuralgs.scalar_ridge import (
    DEFAULT_SCALAR_ARTIFACT,
    artifact_matches_grid,
    cached_scalar_artifact,
    predict_prepared_scalars,
)


ARTIFACT_VERSION = 1
FEATURE_VERSION = 1
DEFAULT_FLUX_RESIDUAL_ARTIFACT = (
    Path(__file__).resolve().parents[1] / "artifacts" / "neuralgs_flux_residual.pt"
)
SPATIAL_CHANNEL_NAMES = (
    "plasma_response",
    "vacuum_field",
    "solved_psi_n",
    "plasma_mask",
    "current_density",
    "pressure",
    "normalized_r",
    "normalized_z",
)

# These solved-state features are dimensionless geometry or sampling
# fractions.  In particular, the transfer residual never sees the learned
# latent, device-scale global inputs, absolute field/current/pressure scales,
# or a machine identifier.
TRANSFER_FEATURE_NAMES = frozenset(
    {
        "plasma_area_fraction",
        "plasma_minor_fraction",
        "plasma_elongation_proxy",
        "token_valid_fraction",
        "token_core_fraction",
        "token_edge_fraction",
    }
    | {
        f"{prefix}_{suffix}"
        for prefix in ("plasma", "current", "token_pressure")
        for suffix in (
            "r_mean",
            "z_mean",
            "r_std",
            "z_std",
            "rz_corr",
            "r_skew",
            "z_skew",
        )
    }
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_sha256(grid_r: np.ndarray, grid_z: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(grid_r, dtype=np.float32).tobytes())
    digest.update(np.asarray(grid_z, dtype=np.float32).tobytes())
    return digest.hexdigest()


def grid_matches_artifact(
    grid_r: np.ndarray | torch.Tensor,
    grid_z: np.ndarray | torch.Tensor,
    artifact: dict,
    atol: float = 2.0e-6,
) -> bool:
    """Return whether an artifact's declared coordinate policy accepts a grid."""

    actual_r = np.asarray(
        grid_r.detach().cpu() if isinstance(grid_r, torch.Tensor) else grid_r,
        dtype=np.float32,
    )
    actual_z = np.asarray(
        grid_z.detach().cpu() if isinstance(grid_z, torch.Tensor) else grid_z,
        dtype=np.float32,
    )
    expected_r = np.asarray(artifact["training_grid_r"], dtype=np.float32)
    expected_z = np.asarray(artifact["training_grid_z"], dtype=np.float32)
    if artifact.get("grid_policy") == "canonical_same_shape":
        return (
            actual_r.shape == expected_r.shape
            and actual_z.shape == expected_z.shape
            and actual_r.ndim == 1
            and actual_z.ndim == 1
            and np.isfinite(actual_r).all()
            and np.isfinite(actual_z).all()
            and np.all(np.diff(actual_r) > 0.0)
            and np.all(np.diff(actual_z) > 0.0)
        )
    return (
        actual_r.shape == expected_r.shape
        and actual_z.shape == expected_z.shape
        and np.allclose(actual_r, expected_r, rtol=0.0, atol=atol)
        and np.allclose(actual_z, expected_z, rtol=0.0, atol=atol)
    )


def split_vacuum_and_plasma(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    gauge = output["gauge"].float()[:, None, None]
    vacuum_component = (
        output["vacuum_gain"].float() * batch["vacuum_psi"].float() + gauge
    )
    return vacuum_component, output["psirz"].float() - vacuum_component


def _normalize_field(field: torch.Tensor, center: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    flat = field.flatten(1)
    mean = flat.mean(dim=1) if center else torch.zeros(len(field), device=field.device)
    centered = field - mean[:, None, None]
    scale = centered.flatten(1).std(dim=1, unbiased=False).clamp_min(1.0e-4)
    return (centered / scale[:, None, None]).clamp(-8.0, 8.0), scale


def build_spatial_channels(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    lcfs_temperature: float,
    transfer_invariant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create bounded full-grid channels and the plasma-response scale."""

    vacuum_component, plasma_response = split_vacuum_and_plasma(output, batch)
    plasma_channel, plasma_scale = _normalize_field(plasma_response)
    vacuum_channel, _ = _normalize_field(batch["vacuum_psi"].float())
    psi_n = output["psi_n"].float().clamp(-0.5, 2.0)
    psi_channel = (psi_n - 0.75) / 1.25
    plasma_mask = torch.sigmoid(
        (1.0 - psi_n) / max(float(lcfs_temperature), 1.0e-3)
    )
    height, width = plasma_response.shape[-2:]
    current = F.interpolate(
        output["current_density"].float().unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    )[:, 0]
    pressure = F.interpolate(
        output["pressure"].float().unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    )[:, 0]
    if transfer_invariant:
        current_channel, _ = _normalize_field(current)
        pressure_channel, _ = _normalize_field(pressure)
    else:
        current_channel = torch.asinh(current / 5.0e5).clamp(-8.0, 8.0) / 4.0
        pressure_channel = torch.log1p(torch.relu(pressure) / 1.0e3).clamp(0.0, 8.0) / 4.0
    r_coordinate = torch.linspace(
        -1.0, 1.0, width, device=plasma_response.device, dtype=torch.float32
    ).reshape(1, 1, width).expand(len(plasma_response), height, width)
    z_coordinate = torch.linspace(
        -1.0, 1.0, height, device=plasma_response.device, dtype=torch.float32
    ).reshape(1, height, 1).expand(len(plasma_response), height, width)
    channels = torch.stack(
        [
            plasma_channel,
            vacuum_channel,
            psi_channel,
            plasma_mask,
            current_channel,
            pressure_channel,
            r_coordinate,
            z_coordinate,
        ],
        dim=1,
    )
    return channels, plasma_scale, vacuum_component, plasma_response


def _dct_modes(dct_order: int, grid_size: int) -> torch.Tensor:
    # Complete low-frequency square, excluding only the constant/gauge mode.
    frequencies = [
        (z_frequency, r_frequency)
        for z_frequency in range(dct_order)
        for r_frequency in range(dct_order)
        if (z_frequency, r_frequency) != (0, 0)
    ]
    r_index = torch.arange(grid_size, dtype=torch.float32) + 0.5
    z_index = torch.arange(grid_size, dtype=torch.float32) + 0.5
    modes = []
    for z_frequency, r_frequency in frequencies:
        r_mode = torch.cos(math.pi * r_frequency * r_index / grid_size)
        z_mode = torch.cos(math.pi * z_frequency * z_index / grid_size)
        mode = z_mode[:, None] * r_mode[None]
        mode = mode - mode.mean()
        modes.append(mode / mode.square().mean().sqrt().clamp_min(1.0e-6))
    return torch.stack(modes)


def _edge_window(grid_size: int) -> torch.Tensor:
    coordinate = torch.linspace(0.0, math.pi, grid_size)
    one_dimensional = torch.sin(coordinate).clamp_min(0.0).pow(0.5)
    return one_dimensional[:, None] * one_dimensional[None]


class FluxResidualCalibrator(nn.Module):
    """Feature-conditioned smooth convolution plus fixed low-rank DCT modes."""

    def __init__(
        self,
        n_features: int,
        grid_size: int = 65,
        dct_order: int = 6,
        hidden_channels: int = 24,
        dropout: float = 0.03,
        max_residual_fraction: float = 0.35,
        transfer_invariant: bool = False,
        zero_initialize: bool = True,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.grid_size = int(grid_size)
        self.dct_order = int(dct_order)
        self.n_modes = self.dct_order**2 - 1
        self.hidden_channels = int(hidden_channels)
        self.dropout = float(dropout)
        self.max_residual_fraction = float(max_residual_fraction)
        self.transfer_invariant = bool(transfer_invariant)
        self.input_convolution = nn.Conv2d(
            len(SPATIAL_CHANNEL_NAMES), self.hidden_channels, 5, padding=2,
            padding_mode="replicate",
        )
        groups = math.gcd(self.hidden_channels, 6)
        self.normalization_1 = nn.GroupNorm(groups, self.hidden_channels)
        self.convolution_2 = nn.Conv2d(
            self.hidden_channels, self.hidden_channels, 3, padding=1,
            padding_mode="replicate",
        )
        self.normalization_2 = nn.GroupNorm(groups, self.hidden_channels)
        self.convolution_3 = nn.Conv2d(
            self.hidden_channels, self.hidden_channels, 3, padding=1,
            padding_mode="replicate",
        )
        self.context = nn.Sequential(
            nn.LayerNorm(self.n_features),
            nn.Linear(self.n_features, self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )
        self.output_convolution = nn.Conv2d(self.hidden_channels, 1, 3, padding=1)
        self.coefficient_head = nn.Sequential(
            nn.LayerNorm(self.n_features),
            nn.Linear(self.n_features, 64),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.n_modes),
        )
        self.register_buffer("dct_modes", _dct_modes(self.dct_order, self.grid_size))
        self.register_buffer("edge_window", _edge_window(self.grid_size))
        if zero_initialize:
            nn.init.zeros_(self.output_convolution.weight)
            nn.init.zeros_(self.output_convolution.bias)
            nn.init.zeros_(self.coefficient_head[-1].weight)
            nn.init.zeros_(self.coefficient_head[-1].bias)

    def forward(
        self,
        standardized_features: torch.Tensor,
        spatial_channels: torch.Tensor,
        plasma_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if spatial_channels.shape[-2:] != (self.grid_size, self.grid_size):
            raise ValueError("flux residual calibrator received a non-training grid")
        hidden = self.input_convolution(spatial_channels)
        hidden = F.silu(self.normalization_1(hidden))
        hidden = hidden + self.context(standardized_features)[:, :, None, None]
        residual = F.silu(self.normalization_2(self.convolution_2(hidden)))
        hidden = hidden + F.dropout(residual, self.dropout, self.training)
        hidden = F.silu(self.convolution_3(hidden))
        convolutional = self.output_convolution(hidden)[:, 0]
        coefficients = self.coefficient_head(standardized_features)
        low_rank = torch.einsum("bk,khw->bhw", coefficients, self.dct_modes)
        # The topology capacity probe found that a 6x6 DCT block captures 98%
        # of the oracle residual. Keep it dominant and let the convolutional
        # branch supply only a modest local refinement.
        raw = low_rank + 0.25 * convolutional
        # A small mean filter suppresses pixel-scale shortcuts that score well
        # in MSE but destabilize LCFS extraction and derived functionals.
        smooth_raw = F.avg_pool2d(raw[:, None], 5, stride=1, padding=2)[:, 0]
        residual_fraction = self.max_residual_fraction * torch.tanh(smooth_raw)
        correction = (
            residual_fraction
            * plasma_scale[:, None, None]
            * self.edge_window[None]
        )
        diagnostics = {
            "correction": correction,
            "residual_fraction": residual_fraction,
            "coefficients": coefficients,
            "convolutional": convolutional,
            "low_rank": low_rank,
        }
        return correction, diagnostics

    def config_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "grid_size": self.grid_size,
            "dct_order": self.dct_order,
            "hidden_channels": self.hidden_channels,
            "dropout": self.dropout,
            "max_residual_fraction": self.max_residual_fraction,
            "transfer_invariant": self.transfer_invariant,
        }


def standardized_features(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    source_r: torch.Tensor,
    source_z: torch.Tensor,
    source_weights: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    lcfs_temperature: float,
    artifact: dict,
) -> torch.Tensor:
    features = build_direct_scalar_features(
        output,
        batch,
        source_r,
        source_z,
        source_weights,
        grid_r,
        grid_z,
        lcfs_temperature,
    )
    active = torch.as_tensor(
        artifact["active_feature_indices"], device=features.device, dtype=torch.long
    )
    mean = torch.as_tensor(
        artifact["feature_mean"], device=features.device, dtype=torch.float32
    )
    scale = torch.as_tensor(
        artifact["feature_scale"], device=features.device, dtype=torch.float32
    )
    return ((features[:, active] - mean) / scale).clamp(-12.0, 12.0)


def apply_flux_residual(
    calibrator: FluxResidualCalibrator,
    standardized_global_features: torch.Tensor,
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    lcfs_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    channels, plasma_scale, vacuum_component, plasma_response = build_spatial_channels(
        output,
        batch,
        grid_r,
        grid_z,
        lcfs_temperature,
        transfer_invariant=calibrator.transfer_invariant,
    )
    correction, diagnostics = calibrator(
        standardized_global_features, channels, plasma_scale
    )
    diagnostics["vacuum_component"] = vacuum_component
    diagnostics["plasma_response"] = plasma_response
    return vacuum_component + plasma_response + correction, diagnostics


def load_flux_residual_artifact(
    path: Path | str = DEFAULT_FLUX_RESIDUAL_ARTIFACT,
    device: torch.device | str = "cpu",
) -> tuple[FluxResidualCalibrator, dict]:
    device = torch.device(device)
    artifact = torch.load(Path(path), map_location=device, weights_only=False)
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(f"unsupported flux residual artifact: {artifact.get('artifact_version')}")
    if artifact.get("feature_version") != FEATURE_VERSION:
        raise ValueError("incompatible flux residual features")
    if artifact.get("direct_feature_version") != DIRECT_FEATURE_VERSION:
        raise ValueError("incompatible solved-state feature version")
    calibrator = FluxResidualCalibrator(
        **artifact["calibrator_config"], zero_initialize=False
    ).to(device)
    calibrator.load_state_dict(artifact["calibrator_state"])
    calibrator.eval()
    return calibrator, artifact


@lru_cache(maxsize=2)
def _load_inference_models(base_path: str, residual_path: str, device_name: str):
    device = torch.device(device_name)
    checkpoint = torch.load(base_path, map_location=device, weights_only=False)
    if checkpoint.get("artifact_version") != 1:
        raise ValueError("unsupported NeuralGS base artifact")
    config = NeuralGSConfig.from_dict(checkpoint["config"])
    base = NeuralGS(config).to(device)
    base.load_state_dict(checkpoint["model_state"])
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    calibrator, artifact = load_flux_residual_artifact(residual_path, device)
    expected_hash = artifact.get("base_artifact_sha256")
    actual_hash = sha256_file(base_path)
    derived_source_hash = checkpoint.get("metadata", {}).get(
        "physical_scalar_source_sha256"
    )
    if expected_hash and actual_hash != expected_hash and derived_source_hash != expected_hash:
        raise ValueError("flux residual was trained against a different NeuralGS artifact")
    return base, config, calibrator, artifact, device, derived_source_hash


@lru_cache(maxsize=4)
def _cached_green(r_bytes: bytes, z_bytes: bytes, stride: int, softening: float):
    r = np.frombuffer(r_bytes, dtype=np.float32).copy()
    z = np.frombuffer(z_bytes, dtype=np.float32).copy()
    return plasma_green_matrix(r, z, stride, softening)


@lru_cache(maxsize=4)
def _cached_direct_scalar_artifact(path: str, device_name: str):
    """Load the immutable direct-scalar inference head once per process/device."""

    return load_direct_scalar_artifact(path, torch.device(device_name))


def residual_inference_input_keys(prepared: dict) -> tuple[str, ...]:
    """Return base inputs plus optional fields required by newer artifacts."""

    return BASE_INPUT_KEYS + tuple(
        key
        for key in ("canonical_global_features", "tf_ampere", "tf_turns")
        if key in prepared and key not in BASE_INPUT_KEYS
    )


def diii_d_only_adapter_allowed(row: dict) -> bool:
    """Permit learned gate adapters only on the explicitly labeled DIII-D leg."""

    return str(row.get("source", "DIII-D")) == "DIII-D"


def predict_row_with_flux_residual(
    row: dict,
    base_artifact: Path | str,
    residual_artifact: Path | str = DEFAULT_FLUX_RESIDUAL_ARTIFACT,
    direct_scalar_artifact: Path | str | None = None,
    device_name: str | None = None,
    batch_size: int = 24,
    disable_residual: bool = False,
    residual_scale: float | None = None,
    return_direct_features: bool = False,
    frame_indices: np.ndarray | None = None,
    apply_topology: bool = True,
    prepared_inputs: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Predict a row, applying the residual only on its exact training grid."""

    if device_name is None:
        requested = os.environ.get("NEURALGS_DEVICE", "auto")
        device_name = (
            "cuda"
            if requested == "auto" and torch.cuda.is_available()
            else "cpu"
            if requested == "auto"
            else requested
        )
    base_path = str(Path(base_artifact).resolve())
    residual_path = str(Path(residual_artifact).resolve())
    base, config, calibrator, artifact, device, derived_source_hash = _load_inference_models(
        base_path, residual_path, device_name
    )
    iteration_override = int(os.environ.get("NEURALGS_OUTER_ITERATIONS", "0"))
    if iteration_override:
        if iteration_override < 1 or iteration_override > 64:
            raise ValueError("NEURALGS_OUTER_ITERATIONS must be between 1 and 64")
        config.outer_iterations = iteration_override
        base.config.outer_iterations = iteration_override
    stride_override = int(os.environ.get("NEURALGS_SOURCE_STRIDE", "0"))
    if stride_override:
        if stride_override not in (1, 2, 4, 8):
            raise ValueError("NEURALGS_SOURCE_STRIDE must be one of 1, 2, 4, 8")
        config.source_stride = stride_override
        base.config.source_stride = stride_override
    center_override = os.environ.get("NEURALGS_INITIAL_CURRENT_CENTER", "")
    if center_override:
        if center_override not in ("grid", "thomson"):
            raise ValueError("NEURALGS_INITIAL_CURRENT_CENTER must be grid or thomson")
        config.initial_current_center = center_override
        base.config.initial_current_center = center_override
    if prepared_inputs is None:
        prepared = preprocess_row(
            row, config, include_targets=False, frame_indices=frame_indices
        )
    else:
        if frame_indices is not None:
            raise ValueError("frame_indices cannot be combined with prepared_inputs")
        prepared = prepared_inputs
    if residual_scale is None:
        residual_scale = float(os.environ.get("NEURALGS_FLUX_RESIDUAL_SCALE", "1.0"))
    if not math.isfinite(residual_scale) or residual_scale < 0.0:
        raise ValueError("flux residual scale must be finite and non-negative")
    r = np.asarray(prepared["grid_r"], dtype=np.float32)
    z = np.asarray(prepared["grid_z"], dtype=np.float32)
    grid_allowed = (not disable_residual) and grid_matches_artifact(r, z, artifact)
    green_np, sr_np, sz_np, weights_np = _cached_green(
        r.tobytes(), z.tobytes(), config.source_stride, config.green_softening_cells
    )
    green, source_r, source_z, source_weights, grid_r, grid_z = [
        torch.as_tensor(value, device=device)
        for value in (green_np, sr_np, sz_np, weights_np, r, z)
    ]
    flux_parts: list[torch.Tensor] = []
    base_flux_parts: list[torch.Tensor] = []
    q_parts: list[torch.Tensor] = []
    beta_parts: list[torch.Tensor] = []
    psi_n_parts: list[torch.Tensor] = []
    fraction_parts: list[torch.Tensor] = []
    direct_parts: list[torch.Tensor] = []
    direct_feature_parts: list[torch.Tensor] = []
    direct_path = (
        str(Path(direct_scalar_artifact).resolve())
        if direct_scalar_artifact is not None
        else os.environ.get("NEURALGS_DIRECT_SCALAR_ARTIFACT")
    )
    topology_path = (
        os.environ.get("NEURALGS_D3D_TOPOLOGY_ARTIFACT")
        if apply_topology
        else None
    )
    direct_head = direct_artifact = None
    # The learned direct scalar head is allowed only on the labeled DIII-D leg.
    # A transfer-invariant residual may apply on both machines, so its grid
    # policy must never be reused as permission to run this DIII-D-only head.
    diii_d_scalar_allowed = diii_d_only_adapter_allowed(row)
    if direct_path and diii_d_scalar_allowed:
        direct_head, direct_artifact = _cached_direct_scalar_artifact(
            str(Path(direct_path).resolve()), str(device)
        )
        expected_hash = direct_artifact.get("base_artifact_sha256")
        actual_hash = sha256_file(base_path)
        if (
            expected_hash
            and actual_hash != expected_hash
            and derived_source_hash != expected_hash
        ):
            raise ValueError(
                "direct scalar head was trained against a different NeuralGS artifact"
            )
    n_frames = len(prepared["ip_abs"])
    inference_keys = residual_inference_input_keys(prepared)
    with torch.inference_mode():
        for start in range(0, n_frames, batch_size):
            stop = min(start + batch_size, n_frames)
            batch = {
                key: torch.as_tensor(prepared[key][start:stop], device=device)
                for key in inference_keys
            }
            output = base(
                batch, green, source_r, source_z, source_weights, grid_r, grid_z
            )
            if grid_allowed:
                features = standardized_features(
                    output,
                    batch,
                    source_r,
                    source_z,
                    source_weights,
                    grid_r,
                    grid_z,
                    config.lcfs_temperature,
                    artifact,
                )
                flux, diagnostics = apply_flux_residual(
                    calibrator,
                    features,
                    output,
                    batch,
                    grid_r,
                    grid_z,
                    config.lcfs_temperature,
                )
                if residual_scale != 1.0:
                    flux = output["psirz"].float() + residual_scale * diagnostics["correction"]
                fraction = diagnostics["residual_fraction"].square().mean(
                    dim=(1, 2)
                ).sqrt() * residual_scale
            else:
                flux = output["psirz"].float()
                fraction = torch.zeros(len(flux), device=device)
            flux_parts.append(flux.cpu())
            base_flux_parts.append(output["psirz"].float().cpu())
            q_parts.append(output["q95"].float().cpu())
            beta_parts.append(output["betaN"].float().cpu())
            if config.physical_scalar_postprocess:
                psi_n_parts.append(output["psi_n"].float().cpu())
            fraction_parts.append(fraction.cpu())
            need_direct_features = (
                (direct_head is not None and direct_artifact is not None)
                or return_direct_features
                or (topology_path is not None and diii_d_scalar_allowed)
            )
            if need_direct_features:
                direct_features = build_direct_scalar_features(
                    output,
                    batch,
                    source_r,
                    source_z,
                    source_weights,
                    grid_r,
                    grid_z,
                    config.lcfs_temperature,
                )
                if return_direct_features or (
                    topology_path is not None and diii_d_scalar_allowed
                ):
                    direct_feature_parts.append(direct_features.cpu())
                if direct_head is not None and direct_artifact is not None:
                    direct_parts.append(
                        predict_from_features(
                            direct_features, direct_head, direct_artifact
                        ).cpu()
                    )
    flux_all = torch.cat(flux_parts)
    base_flux_all = torch.cat(base_flux_parts)
    temporal_window = int(os.environ.get("NEURALGS_FLUX_RESIDUAL_TEMPORAL_WINDOW", "1"))
    if temporal_window < 1 or temporal_window % 2 == 0:
        raise ValueError("flux residual temporal window must be a positive odd integer")
    if grid_allowed and temporal_window > 1 and len(flux_all) > 1:
        correction = flux_all - base_flux_all
        time_length, height, width = correction.shape
        series = correction.permute(1, 2, 0).reshape(height * width, 1, time_length)
        radius = temporal_window // 2
        series = F.pad(series, (radius, radius), mode="replicate")
        smoothed = F.avg_pool1d(series, temporal_window, stride=1)
        correction = smoothed.reshape(height, width, time_length).permute(2, 0, 1)
        flux_all = base_flux_all + correction
    combined = {
        "psirz": flux_all.numpy().astype(np.float32),
        "q95": torch.cat(q_parts).numpy().astype(np.float32),
        "betaN": torch.cat(beta_parts).numpy().astype(np.float32),
        "residual_rms_fraction": torch.cat(fraction_parts).numpy().astype(np.float32),
        "residual_applied": np.asarray(grid_allowed),
    }
    scalar_path = os.environ.get("NEURALGS_SCALAR_ARTIFACT")
    calibrated_scalars_applied = False
    if direct_parts:
        direct = torch.cat(direct_parts).numpy().astype(np.float32)
        combined["q95"] = direct[:, 0]
        combined["betaN"] = direct[:, 1]
        calibrated_scalars_applied = True
    elif scalar_path is None and DEFAULT_SCALAR_ARTIFACT.exists():
        scalar_path = str(DEFAULT_SCALAR_ARTIFACT)
    if not direct_parts and scalar_path:
        scalar_artifact = cached_scalar_artifact(str(Path(scalar_path).resolve()))
        force_scalar = os.environ.get("NEURALGS_FORCE_SCALAR_RIDGE", "0") == "1"
        if force_scalar or artifact_matches_grid(prepared, scalar_artifact):
            combined.update(predict_prepared_scalars(prepared, scalar_artifact))
            calibrated_scalars_applied = True
    if config.physical_scalar_postprocess and not calibrated_scalars_applied:
        from my_experiments.neuralgs.mast_physical_scalars import physics_q95
        from my_experiments.neuralgs.validate_beta_physics import electron_beta_n_proxy

        q_physical = physics_q95(
            row,
            combined["psirz"],
            numerical_scale=float(config.physical_q_scale),
            axis_sign=1.0,
            use_machine_envelope=False,
        )
        combined["q95"] = np.clip(
            np.where(np.isfinite(q_physical), q_physical, combined["q95"]),
            0.1,
            30.0,
        ).astype(np.float32)
        psi_n = torch.cat(psi_n_parts).numpy().astype(np.float64)
        beta_physical = np.empty(len(psi_n), dtype=np.float64)
        machine = str(row.get("source", "DIII-D"))
        for frame, coordinate in enumerate(psi_n):
            beta_physical[frame] = electron_beta_n_proxy(
                coordinate,
                prepared["token_coords"][frame],
                prepared["token_values"][frame],
                prepared["token_mask"][frame],
                float(prepared["ip_abs"][frame]),
                float(prepared["tf_ampere"][frame]),
                r,
                z,
                machine=machine,
            )
        beta_physical *= float(config.physical_beta_scale)
        combined["betaN"] = np.clip(
            np.where(np.isfinite(beta_physical), beta_physical, combined["betaN"]),
            0.0,
            10.0,
        ).astype(np.float32)
    direct_features_all = (
        torch.cat(direct_feature_parts).numpy().astype(np.float32)
        if direct_feature_parts
        else None
    )
    if topology_path and diii_d_scalar_allowed:
        if direct_features_all is None:
            raise RuntimeError("DIII-D topology adapter did not receive solved-state features")
        from my_experiments.topology.d3d_label_topology import (
            load_artifact as load_topology_artifact,
            predict_topology_from_direct_features,
        )
        from my_experiments.topology.d3d_topology_postprocess import (
            apply_d3d_topology_calibration,
        )

        resolved_topology = str(Path(topology_path).resolve())
        topology_artifact = load_topology_artifact(resolved_topology)
        expected_hash = topology_artifact.get("metadata", {}).get(
            "base_artifact_sha256"
        )
        actual_hash = sha256_file(base_path)
        if (
            expected_hash
            and actual_hash != expected_hash
            and derived_source_hash != expected_hash
        ):
            raise ValueError("DIII-D topology head was trained against a different base")
        desired = predict_topology_from_direct_features(
            direct_features_all, resolved_topology
        )
        geometry_strength = float(
            os.environ.get("NEURALGS_D3D_TOPOLOGY_GEOMETRY_STRENGTH", "0.55")
        )
        profile_strength = float(
            os.environ.get("NEURALGS_D3D_TOPOLOGY_PROFILE_STRENGTH", "0.25")
        )
        iterations = int(
            os.environ.get("NEURALGS_D3D_TOPOLOGY_ITERATIONS", "2")
        )
        axis_strength = float(
            os.environ.get(
                "NEURALGS_D3D_TOPOLOGY_AXIS_STRENGTH", str(geometry_strength)
            )
        )
        shape_strength = float(
            os.environ.get(
                "NEURALGS_D3D_TOPOLOGY_SHAPE_STRENGTH", str(geometry_strength)
            )
        )
        ensure_axis_peak = (
            os.environ.get("NEURALGS_D3D_TOPOLOGY_ENSURE_AXIS_PEAK", "1") == "1"
        )
        axis_min_r = float(
            os.environ.get("NEURALGS_D3D_TOPOLOGY_AXIS_MIN_R", "0")
        )
        axis_min_z = float(
            os.environ.get("NEURALGS_D3D_TOPOLOGY_AXIS_MIN_Z", "0")
        )
        if not (
            math.isfinite(geometry_strength)
            and math.isfinite(profile_strength)
            and math.isfinite(axis_strength)
            and math.isfinite(shape_strength)
            and math.isfinite(axis_min_r)
            and math.isfinite(axis_min_z)
            and 0.0 <= geometry_strength <= 1.0
            and 0.0 <= profile_strength <= 1.0
            and 0.0 <= axis_strength <= 1.0
            and 0.0 <= shape_strength <= 1.0
            and 0.0 <= axis_min_r <= 0.25
            and 0.0 <= axis_min_z <= 0.35
            and 0 <= iterations <= 8
        ):
            raise ValueError("invalid DIII-D topology adapter strength")
        combined = apply_d3d_topology_calibration(
            row,
            combined,
            geometry_strength=geometry_strength,
            profile_strength=profile_strength,
            geometry_iterations=iterations,
            axis_strength=axis_strength,
            shape_strength=shape_strength,
            axis_min_r_disagreement=axis_min_r,
            axis_min_z_disagreement=axis_min_z,
            ensure_axis_peak=ensure_axis_peak,
            desired_override=desired,
        )
        combined["d3d_topology_adapter_applied"] = np.asarray(True)
    if return_direct_features:
        if direct_features_all is None:
            raise RuntimeError("direct topology features were requested but not produced")
        combined["direct_features"] = direct_features_all
    return combined


__all__ = [
    "ARTIFACT_VERSION",
    "FEATURE_VERSION",
    "DEFAULT_FLUX_RESIDUAL_ARTIFACT",
    "SPATIAL_CHANNEL_NAMES",
    "TRANSFER_FEATURE_NAMES",
    "FluxResidualCalibrator",
    "grid_sha256",
    "grid_matches_artifact",
    "residual_inference_input_keys",
    "diii_d_only_adapter_allowed",
    "split_vacuum_and_plasma",
    "build_spatial_channels",
    "standardized_features",
    "apply_flux_residual",
    "load_flux_residual_artifact",
    "predict_row_with_flux_residual",
]
