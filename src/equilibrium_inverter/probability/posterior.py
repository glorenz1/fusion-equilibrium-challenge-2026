"""Bayesian posterior energy for target-free equilibrium inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import PosteriorConfig
from ..inputs.shot import TorchShot
from ..physics.forward import ForwardResult
from ..state.latent import StateLayout


@dataclass(frozen=True)
class PosteriorTerms:
    total: torch.Tensor
    thomson: torch.Tensor
    prior: torch.Tensor
    temporal: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "thomson": float(self.thomson.detach()),
            "prior": float(self.prior.detach()),
            "temporal": float(self.temporal.detach()),
        }


class PosteriorEnergy(nn.Module):
    """Negative log posterior with declared observation and process scales."""

    def __init__(self, layout: StateLayout, config: PosteriorConfig) -> None:
        super().__init__()
        self.layout = layout
        self.config = config

    def _prior_scales(self, reference: torch.Tensor) -> torch.Tensor:
        c = self.config
        pieces = [
            reference.new_full((self.layout.pressure_count,), c.profile_prior_sigma),
            reference.new_full((self.layout.ffprime_count,), c.profile_prior_sigma),
            reference.new_full((self.layout.harmonic_count,), c.harmonic_prior_sigma),
            reference.new_tensor(
                [
                    c.gauge_prior_sigma,
                    c.pressure_scale_prior_sigma,
                    1.25,
                    1.25,
                    c.scalar_log_prior_sigma,
                    c.scalar_log_prior_sigma,
                ]
            ),
            reference.new_full((self.layout.topology_count,), c.topology_logit_prior_sigma),
        ]
        result = torch.cat(pieces)
        if len(result) != self.layout.dimension:
            raise AssertionError("posterior prior scales do not match state layout")
        return result

    def forward(
        self,
        shot: TorchShot,
        latent_vector: torch.Tensor,
        prediction: ForwardResult,
        *,
        proposal_mean: torch.Tensor | None = None,
        proposal_log_scale: torch.Tensor | None = None,
    ) -> PosteriorTerms:
        observed = torch.log1p(shot.thomson_pressure_Pa.clamp_min(0.0) / 1.0e3)
        predicted = torch.log1p(
            prediction.predicted_thomson_pressure_Pa.clamp_min(0.0) / 1.0e3
        )
        valid = shot.thomson_valid & shot.frame_valid[:, None]
        residual = (predicted - observed) / self.config.thomson_log_sigma
        thomson = torch.where(valid, 0.5 * residual.square(), torch.zeros_like(residual))
        thomson = thomson.sum() / valid.sum().clamp_min(1)

        if proposal_mean is not None and proposal_log_scale is not None:
            scale = torch.exp(proposal_log_scale).clamp_min(1.0e-4)
            prior = 0.5 * (((latent_vector - proposal_mean) / scale).square() + 2.0 * proposal_log_scale)
            prior = prior.mean()
        else:
            center = self.layout.zeros(
                len(latent_vector), device=latent_vector.device, dtype=latent_vector.dtype
            )
            scale = self._prior_scales(latent_vector)
            prior = 0.5 * ((latent_vector - center) / scale[None]).square().mean()

        if len(latent_vector) > 1:
            dt_seconds = ((shot.times_ms[1:] - shot.times_ms[:-1]) / 1000.0).clamp_min(1.0e-4)
            process_scale = (
                self.config.temporal_sigma_per_sqrt_second * torch.sqrt(dt_seconds)
            )[:, None]
            temporal = 0.5 * ((latent_vector[1:] - latent_vector[:-1]) / process_scale).square().mean()
        else:
            temporal = latent_vector.new_zeros(())
        total = (
            self.config.thomson_weight * thomson
            + self.config.prior_weight * prior
            + self.config.temporal_weight * temporal
        )
        return PosteriorTerms(total=total, thomson=thomson, prior=prior, temporal=temporal)
