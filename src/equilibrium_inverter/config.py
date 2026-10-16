"""Versioned configuration for the Equilibrium Inverter framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class SolverConfig:
    """Differentiable free-boundary profile solver settings."""

    source_stride: int = 4
    green_softening_cells: float = 0.45
    pressure_coefficients: int = 6
    ffprime_coefficients: int = 8
    harmonic_modes: int = 8
    picard_iterations: int = 4
    picard_damping: float = 0.65
    lcfs_temperature: float = 0.04
    current_window_margin: float = 0.055

    def __post_init__(self) -> None:
        if self.source_stride < 1:
            raise ValueError("source_stride must be positive")
        if self.pressure_coefficients < 2 or self.ffprime_coefficients < 2:
            raise ValueError("profile bases require at least two coefficients")
        if self.harmonic_modes < 0:
            raise ValueError("harmonic_modes cannot be negative")
        if self.picard_iterations < 1:
            raise ValueError("picard_iterations must be positive")
        if not 0.0 < self.picard_damping <= 1.0:
            raise ValueError("picard_damping must lie in (0, 1]")


@dataclass(frozen=True)
class PosteriorConfig:
    """Noise and prior scales in dimensionless latent coordinates."""

    thomson_log_sigma: float = 0.45
    profile_prior_sigma: float = 1.5
    harmonic_prior_sigma: float = 0.20
    gauge_prior_sigma: float = 0.35
    pressure_scale_prior_sigma: float = 1.25
    scalar_log_prior_sigma: float = 0.75
    topology_logit_prior_sigma: float = 1.5
    temporal_sigma_per_sqrt_second: float = 0.45
    thomson_weight: float = 1.0
    prior_weight: float = 1.0
    temporal_weight: float = 0.25

    def __post_init__(self) -> None:
        positive = (
            self.thomson_log_sigma,
            self.profile_prior_sigma,
            self.harmonic_prior_sigma,
            self.gauge_prior_sigma,
            self.pressure_scale_prior_sigma,
            self.scalar_log_prior_sigma,
            self.topology_logit_prior_sigma,
            self.temporal_sigma_per_sqrt_second,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("posterior scales must be positive")


@dataclass(frozen=True)
class InferenceConfig:
    """Multi-start maximum-a-posteriori optimizer settings."""

    starts: int = 4
    steps: int = 80
    learning_rate: float = 0.025
    gradient_clip_norm: float = 10.0
    initialization_noise: float = 0.20
    seed: int = 20260831

    def __post_init__(self) -> None:
        if self.starts < 1 or self.steps < 1:
            raise ValueError("MAP inference requires positive starts and steps")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True)
class ScoreTargets:
    """Promotion targets; these are requirements, not achieved results."""

    diii_d_composite: float = 0.90
    mast_composite: float = 0.90
    flux_r2: float = 0.985
    scalar_r2: float = 0.90
    lcfs_similarity: float = 0.96
    consistency: float = 0.65


@dataclass(frozen=True)
class EquilibriumInverterConfig:
    """Complete saved configuration for training and inference."""

    version: str = "0.1.0"
    solver: SolverConfig = field(default_factory=SolverConfig)
    posterior: PosteriorConfig = field(default_factory=PosteriorConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    score_targets: ScoreTargets = field(default_factory=ScoreTargets)
    strict_geometry: bool = False
    causal_measurements: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "EquilibriumInverterConfig":
        """Restore a saved configuration while retaining dataclass validation."""

        return cls(
            version=str(values.get("version", cls.version)),
            solver=SolverConfig(**dict(values.get("solver", {}))),
            posterior=PosteriorConfig(**dict(values.get("posterior", {}))),
            inference=InferenceConfig(**dict(values.get("inference", {}))),
            score_targets=ScoreTargets(**dict(values.get("score_targets", {}))),
            strict_geometry=bool(values.get("strict_geometry", False)),
            causal_measurements=bool(values.get("causal_measurements", False)),
        )


# Kept out of the documented API, but allows old imports to load saved runs.
BayesianEquilibriumConfig = EquilibriumInverterConfig
