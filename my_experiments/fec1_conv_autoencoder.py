"""Compact convolutional representation for 65 x 65 FEC1 flux shapes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AutoencoderConfig:
    latent_width: int = 96
    base_channels: int = 16

    def validate(self) -> None:
        if self.latent_width < 8:
            raise ValueError("latent width must be at least 8")
        if self.base_channels < 4:
            raise ValueError("base channels must be at least 4")

    def to_dict(self) -> dict[str, int]:
        self.validate()
        return asdict(self)


def normalize_shape(field: torch.Tensor) -> torch.Tensor:
    """Remove the two affine flux coordinates carried outside the latent."""

    if field.shape[-2:] != (65, 65):
        raise ValueError("FEC1 flux rasters must have shape 65 x 65")
    mean = field.mean(dim=(-2, -1), keepdim=True)
    centered = field - mean
    rms = centered.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1.0e-7)
    return centered / rms


class ConvFluxAutoencoder(nn.Module):
    """Small residual basis with an exact 65 -> 33 -> 17 -> 9 -> 5 pyramid."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        c = config.base_channels

        def down(in_channels: int, out_channels: int, kernel: int = 3) -> nn.Sequential:
            padding = kernel // 2
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, stride=2, padding=padding),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(),
            )

        def up(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, 3, stride=2, padding=1,
                    output_padding=0,
                ),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(),
            )

        self.encoder_grid = nn.Sequential(
            down(1, c, kernel=5),
            down(c, 2 * c),
            down(2 * c, 4 * c),
            down(4 * c, 6 * c),
        )
        bottleneck = 6 * c * 5 * 5
        self.encoder_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck, 2 * config.latent_width),
            nn.SiLU(),
            nn.Linear(2 * config.latent_width, config.latent_width),
        )
        self.decoder_latent = nn.Sequential(
            nn.Linear(config.latent_width, 2 * config.latent_width),
            nn.SiLU(),
            nn.Linear(2 * config.latent_width, bottleneck),
            nn.SiLU(),
        )
        self.decoder_grid = nn.Sequential(
            up(6 * c, 4 * c),
            up(4 * c, 2 * c),
            up(2 * c, c),
            nn.ConvTranspose2d(c, 1, 3, stride=2, padding=1, output_padding=0),
        )

    def encode(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim == 3:
            field = field[:, None]
        if field.ndim != 4 or field.shape[1:] != (1, 65, 65):
            raise ValueError("autoencoder input must have shape (batch, 65, 65)")
        encoded = self.encoder_grid(field)
        if encoded.shape[1:] != (6 * self.config.base_channels, 5, 5):
            raise RuntimeError("unexpected convolutional encoder shape")
        return self.encoder_latent(encoded)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.config.latent_width:
            raise ValueError("latent batch has the wrong width")
        c = self.config.base_channels
        grid = self.decoder_latent(latent).reshape(len(latent), 6 * c, 5, 5)
        decoded = self.decoder_grid(grid)
        if decoded.shape[1:] != (1, 65, 65):
            raise RuntimeError("unexpected convolutional decoder shape")
        return decoded[:, 0]

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(field))


def pca_residual_reconstruction(
    model: ConvFluxAutoencoder,
    target_shape: torch.Tensor,
    pca_components: torch.Tensor,
    pca_mean: torch.Tensor,
    residual_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct a shape as PCA64 plus a compact nonlinear residual."""

    if residual_scale <= 0.0:
        raise ValueError("residual scale must be positive")
    # The residual is O(1e-3), so an autocast PCA round trip would be the same
    # scale as the signal being learned. Keep both cancellations in FP32.
    with torch.autocast(device_type=target_shape.device.type, enabled=False):
        target = normalize_shape(target_shape.float())
        components = pca_components.float()
        mean = pca_mean.float()
        flat = target.flatten(1)
        base_flat = (flat - mean) @ components.T @ components + mean
        base = base_flat.reshape_as(target)
        target_residual = (target - base) / residual_scale
    predicted_residual = model(target_residual)
    with torch.autocast(device_type=target_shape.device.type, enabled=False):
        reconstruction = normalize_shape(
            base + residual_scale * predicted_residual.float()
        )
    return reconstruction, target_residual, predicted_residual


def _masked_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def teacher_normalized_flux(
    reconstructed_shape: torch.Tensor,
    target_psi_n: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Put a reconstructed affine-free shape onto the teacher's psi_n scale."""

    target = torch.nan_to_num(target_psi_n).clamp(-0.5, 3.5)
    useful = torch.isfinite(target_psi_n) & (target_psi_n >= -0.5) & (target_psi_n <= 3.5)
    useful_f = useful.to(reconstructed_shape.dtype)
    axis_band = torch.exp(-0.5 * (target / 0.075).square()) * useful_f
    lcfs_band = torch.exp(-0.5 * ((target - 1.0) / 0.065).square()) * useful_f

    def level(weight: torch.Tensor) -> torch.Tensor:
        return (reconstructed_shape * weight).sum(dim=(-2, -1)) / weight.sum(
            dim=(-2, -1)
        ).clamp_min(1.0e-6)

    axis_level = level(axis_band)
    boundary_level = level(lcfs_band)
    span = axis_level - boundary_level
    span = torch.where(span >= 0.0, span.clamp_min(0.02), span.clamp_max(-0.02))
    psi_n = (
        axis_level[:, None, None] - reconstructed_shape
    ) / span[:, None, None]
    return psi_n, useful


def reconstruction_loss(
    reconstruction: torch.Tensor,
    target_shape: torch.Tensor,
    target_psi_n: torch.Tensor,
    *,
    pixel_weight: float = 1.0,
    boundary_weight: float = 0.5,
    gradient_weight: float = 0.25,
    curvature_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Geometry-aware reconstruction objective for the nonlinear basis."""

    reconstruction = normalize_shape(reconstruction)
    target_shape = normalize_shape(target_shape)
    target_n = torch.nan_to_num(target_psi_n).clamp(-0.5, 3.5)
    predicted_n, useful = teacher_normalized_flux(reconstruction, target_psi_n)
    # An untrained decoder can initially imply a nearly zero axis-to-boundary
    # span. Keep derivative objectives bounded during that warm-up without
    # clipping the submitted reconstruction itself.
    predicted_n_loss = predicted_n.clamp(-0.5, 3.5)
    useful_f = useful.to(reconstruction.dtype)

    pixel = (reconstruction - target_shape).square().mean()
    axis_band = torch.exp(-0.5 * (target_n / 0.08).square())
    lcfs_band = torch.exp(-0.5 * ((target_n - 1.0) / 0.055).square())
    surface_weight = useful_f * (0.15 + 5.0 * axis_band + 12.0 * lcfs_band)
    surface_error = F.smooth_l1_loss(
        predicted_n_loss, target_n, beta=0.03, reduction="none",
    )
    boundary = _masked_mean(surface_error, surface_weight)

    pred_dr = predicted_n_loss[:, :, 1:] - predicted_n_loss[:, :, :-1]
    true_dr = target_n[:, :, 1:] - target_n[:, :, :-1]
    pred_dz = predicted_n_loss[:, 1:, :] - predicted_n_loss[:, :-1, :]
    true_dz = target_n[:, 1:, :] - target_n[:, :-1, :]
    mask_dr = (useful[:, :, 1:] & useful[:, :, :-1]).to(reconstruction.dtype)
    mask_dz = (useful[:, 1:, :] & useful[:, :-1, :]).to(reconstruction.dtype)
    gradient = 0.5 * (
        _masked_mean((pred_dr - true_dr).square(), mask_dr)
        + _masked_mean((pred_dz - true_dz).square(), mask_dz)
    )

    pred_drr = pred_dr[:, :, 1:] - pred_dr[:, :, :-1]
    true_drr = true_dr[:, :, 1:] - true_dr[:, :, :-1]
    pred_dzz = pred_dz[:, 1:, :] - pred_dz[:, :-1, :]
    true_dzz = true_dz[:, 1:, :] - true_dz[:, :-1, :]
    mask_drr = (mask_dr[:, :, 1:] * mask_dr[:, :, :-1]).to(reconstruction.dtype)
    mask_dzz = (mask_dz[:, 1:, :] * mask_dz[:, :-1, :]).to(reconstruction.dtype)
    curvature = 0.5 * (
        _masked_mean((pred_drr - true_drr).square(), mask_drr)
        + _masked_mean((pred_dzz - true_dzz).square(), mask_dzz)
    )

    total = (
        pixel_weight * pixel
        + boundary_weight * boundary
        + gradient_weight * gradient
        + curvature_weight * curvature
    )
    return total, {
        "loss": total.detach(),
        "pixel": pixel.detach(),
        "boundary": boundary.detach(),
        "gradient": gradient.detach(),
        "curvature": curvature.detach(),
    }


def load_autoencoder_artifact(
    path: str, device: torch.device,
) -> tuple[ConvFluxAutoencoder, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    required = {
        "config", "model_state", "pca_components", "pca_mean", "residual_scale",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise RuntimeError("invalid FEC1 convolutional autoencoder artifact")
    raw_config = payload["config"]
    if not isinstance(raw_config, dict):
        raise RuntimeError("autoencoder configuration is not a mapping")
    config = AutoencoderConfig(
        latent_width=int(raw_config["latent_width"]),
        base_channels=int(raw_config["base_channels"]),
    )
    model = ConvFluxAutoencoder(config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload
