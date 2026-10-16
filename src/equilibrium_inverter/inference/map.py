"""Multi-start maximum-a-posteriori equilibrium inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import InferenceConfig
from ..initialization.proposal import ProposalDistribution
from ..inputs.shot import TorchShot
from ..physics.forward import EquilibriumForwardModel, ForwardResult
from ..probability.posterior import PosteriorEnergy


@dataclass(frozen=True)
class MAPResult:
    latent_vector: torch.Tensor
    forward: ForwardResult
    posterior_terms: dict[str, float]
    loss_history: tuple[float, ...]
    selected_start: int


class MAPInference:
    """Optimize per-shot latent variables while keeping shared weights frozen."""

    def __init__(
        self,
        forward_model: EquilibriumForwardModel,
        posterior: PosteriorEnergy,
        config: InferenceConfig,
    ) -> None:
        self.forward_model = forward_model
        self.posterior = posterior
        self.config = config

    def run(
        self,
        shot: TorchShot,
        proposal: ProposalDistribution,
    ) -> MAPResult:
        generator = torch.Generator(device=proposal.mean.device)
        generator.manual_seed(self.config.seed)
        best: MAPResult | None = None
        for start in range(self.config.starts):
            if start == 0:
                initial = proposal.mean.detach().clone()
            else:
                noise = torch.randn(
                    proposal.mean.shape,
                    generator=generator,
                    device=proposal.mean.device,
                    dtype=proposal.mean.dtype,
                )
                initial = proposal.mean.detach() + self.config.initialization_noise * noise
            latent = torch.nn.Parameter(initial)
            optimizer = torch.optim.Adam([latent], lr=self.config.learning_rate)
            history: list[float] = []
            for _ in range(self.config.steps):
                optimizer.zero_grad(set_to_none=True)
                prediction = self.forward_model(shot, latent)
                terms = self.posterior(
                    shot,
                    latent,
                    prediction,
                    proposal_mean=proposal.mean.detach(),
                    proposal_log_scale=proposal.log_scale.detach(),
                )
                if not torch.isfinite(terms.total):
                    break
                terms.total.backward()
                torch.nn.utils.clip_grad_norm_([latent], self.config.gradient_clip_norm)
                optimizer.step()
                history.append(float(terms.total.detach()))
            with torch.no_grad():
                prediction = self.forward_model(shot, latent)
                terms = self.posterior(
                    shot,
                    latent,
                    prediction,
                    proposal_mean=proposal.mean.detach(),
                    proposal_log_scale=proposal.log_scale.detach(),
                )
            if not torch.isfinite(terms.total):
                continue
            result = MAPResult(
                latent_vector=latent.detach().clone(),
                forward=prediction,
                posterior_terms=terms.detached(),
                loss_history=tuple(history),
                selected_start=start,
            )
            if best is None or result.posterior_terms["total"] < best.posterior_terms["total"]:
                best = result
        if best is None:
            raise FloatingPointError("all MAP starts produced a non-finite posterior")
        return best
