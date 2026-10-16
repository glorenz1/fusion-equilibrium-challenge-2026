"""Amortized proposal conditioned on physical fields and Thomson sets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from neuralgs.geometry import MU0

from ..inputs.shot import TorchShot
from ..state.latent import StateLayout


@dataclass(frozen=True)
class ProposalDistribution:
    mean: torch.Tensor
    log_scale: torch.Tensor

    def sample(self, *, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape,
            device=self.mean.device,
            dtype=self.mean.dtype,
            generator=generator,
        )
        return self.mean + torch.exp(self.log_scale) * noise


class InitialStateProposal(nn.Module):
    """Predict a latent proposal without accepting a machine identifier.

    Active-coil currents enter only after deterministic field rendering.  The
    same adaptive spatial encoder and permutation-invariant Thomson encoder are
    used for every grid and diagnostic inventory.
    """

    def __init__(self, layout: StateLayout, hidden: int = 96) -> None:
        super().__init__()
        self.layout = layout
        self.spatial = nn.Sequential(
            nn.Conv2d(5, 24, 5, stride=2, padding=2),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.GroupNorm(6, 48),
            nn.GELU(),
            nn.Conv2d(48, hidden, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.thomson = nn.Sequential(
            nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.globals = nn.Sequential(
            nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(3 * hidden), nn.Linear(3 * hidden, hidden), nn.GELU()
        )
        self.temporal = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.mean_head = nn.Linear(2 * hidden, layout.dimension)
        self.scale_head = nn.Linear(2 * hidden, layout.dimension)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, -1.5)

    def _spatial_inputs(self, shot: TorchShot, length_scale_m: float) -> torch.Tensor:
        batch, nz, nr = shot.active_psi_Wb_per_rad.shape
        dtype, device = shot.active_psi_Wb_per_rad.dtype, shot.active_psi_Wb_per_rad.device
        safe_ip = shot.plasma_current_A.clamp_min(1.0e3)
        field_scale = (MU0 * safe_ip / float(length_scale_m)).clamp_min(1.0e-8)
        psi = torch.nan_to_num(
            shot.active_psi_Wb_per_rad / shot.flux_scale_Wb_per_rad[:, None, None]
        ).clamp(-6.0, 6.0)
        b_r = torch.nan_to_num(shot.active_b_r_T / field_scale[:, None, None]).clamp(-6.0, 6.0)
        b_z = torch.nan_to_num(shot.active_b_z_T / field_scale[:, None, None]).clamp(-6.0, 6.0)
        radial = torch.linspace(-1.0, 1.0, nr, dtype=dtype, device=device)
        vertical = torch.linspace(-1.0, 1.0, nz, dtype=dtype, device=device)
        radial = radial[None, None, :].expand(batch, nz, nr)
        vertical = vertical[None, :, None].expand(batch, nz, nr)
        return torch.stack([psi, b_r, b_z, radial, vertical], dim=1)

    def forward(self, shot: TorchShot, *, length_scale_m: float) -> ProposalDistribution:
        spatial = self.spatial(self._spatial_inputs(shot, length_scale_m)).flatten(1)
        token_hidden = self.thomson(shot.thomson_features)
        mask = shot.thomson_valid[..., None]
        token_sum = (token_hidden * mask).sum(dim=1)
        token_count = mask.sum(dim=1).clamp_min(1)
        token_pooled = token_sum / token_count
        global_hidden = self.globals(shot.global_features)
        fused = self.fusion(torch.cat([spatial, token_pooled, global_hidden], dim=-1))
        temporal, _ = self.temporal(fused[None])
        temporal = temporal[0]
        prior_center = self.layout.zeros(
            shot.frames, device=temporal.device, dtype=temporal.dtype
        )
        mean = prior_center + self.mean_head(temporal)
        log_scale = self.scale_head(temporal).clamp(-5.0, 1.0)
        return ProposalDistribution(mean=mean, log_scale=log_scale)
