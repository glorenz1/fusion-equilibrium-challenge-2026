"""Flat latent-vector contract used by proposals and posterior optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import SolverConfig
from .topology import TOPOLOGY_ORDER


@dataclass(frozen=True)
class DecodedLatentState:
    """Constrained per-frame physical state decoded from an unconstrained vector."""

    pressure_shape: torch.Tensor
    ffprime_shape: torch.Tensor
    harmonic_coefficients: torch.Tensor
    gauge_dimensionless: torch.Tensor
    pressure_scale_multiplier: torch.Tensor
    electron_pressure_fraction: torch.Tensor
    pressure_current_fraction: torch.Tensor
    q95: torch.Tensor
    beta_n: torch.Tensor
    topology_logits: torch.Tensor

    @property
    def topology_probabilities(self) -> torch.Tensor:
        return torch.softmax(self.topology_logits, dim=-1)


@dataclass(frozen=True)
class StateLayout:
    """Stable slicing and physical transforms for the Bayesian latent."""

    pressure_count: int
    ffprime_count: int
    harmonic_count: int
    topology_count: int = len(TOPOLOGY_ORDER)

    @classmethod
    def from_solver_config(cls, config: SolverConfig) -> "StateLayout":
        return cls(
            pressure_count=config.pressure_coefficients,
            ffprime_count=config.ffprime_coefficients,
            harmonic_count=config.harmonic_modes,
        )

    @property
    def dimension(self) -> int:
        # Profiles + harmonics + gauge + pressure scale + electron fraction +
        # pressure-current fraction + q95 + betaN + topology logits.
        return (
            self.pressure_count
            + self.ffprime_count
            + self.harmonic_count
            + 6
            + self.topology_count
        )

    def zeros(
        self,
        frames: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if frames < 1:
            raise ValueError("frames must be positive")
        values = torch.zeros((frames, self.dimension), device=device, dtype=dtype)
        # q95=4 and betaN=2 are broad physical initial values, not machine IDs.
        q_index = self.pressure_count + self.ffprime_count + self.harmonic_count + 4
        values[:, q_index] = torch.log(torch.tensor(4.0, dtype=dtype, device=device))
        values[:, q_index + 1] = torch.log(torch.tensor(2.0, dtype=dtype, device=device))
        return values

    def decode(self, vector: torch.Tensor) -> DecodedLatentState:
        if vector.ndim != 2 or vector.shape[1] != self.dimension:
            raise ValueError(
                f"latent vector must have shape [frames, {self.dimension}]"
            )
        offset = 0

        def take(count: int) -> torch.Tensor:
            nonlocal offset
            result = vector[:, offset : offset + count]
            offset += count
            return result

        pressure = F.softplus(take(self.pressure_count)) + 1.0e-4
        ffprime = F.softplus(take(self.ffprime_count)) + 1.0e-4
        harmonic = take(self.harmonic_count)
        gauge = torch.tanh(take(1).squeeze(-1))
        pressure_scale = torch.exp(take(1).squeeze(-1).clamp(-4.0, 4.0))
        electron_fraction = 0.15 + 0.80 * torch.sigmoid(take(1).squeeze(-1))
        pressure_fraction = 0.02 + 0.88 * torch.sigmoid(take(1).squeeze(-1))
        q95 = torch.exp(take(1).squeeze(-1).clamp(-2.0, 4.0))
        beta_n = torch.exp(take(1).squeeze(-1).clamp(-3.0, 4.0))
        topology = take(self.topology_count)
        if offset != self.dimension:
            raise AssertionError("state layout did not consume its complete vector")
        return DecodedLatentState(
            pressure_shape=pressure,
            ffprime_shape=ffprime,
            harmonic_coefficients=harmonic,
            gauge_dimensionless=gauge,
            pressure_scale_multiplier=pressure_scale,
            electron_pressure_fraction=electron_fraction,
            pressure_current_fraction=pressure_fraction,
            q95=q95,
            beta_n=beta_n,
            topology_logits=topology,
        )
