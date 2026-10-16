"""Supervised normalized-flux transform used by both geometry objectives."""
from __future__ import annotations
import torch


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
    psi_n = (axis_level[:, None, None] - reconstructed_shape) / span[:, None, None]
    return psi_n, useful
