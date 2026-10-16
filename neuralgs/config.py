from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class NeuralGSConfig:
    """Architecture and solver settings saved with every model artifact."""

    grid_size: int = 65
    source_stride: int = 2
    max_tokens: int = 192
    token_features: int = 5
    # Add machine-normalized physical (R, Z) coordinates to every Thomson
    # token.  Disabled by default so legacy artifacts retain their exact input
    # shape and predictions.
    coordinate_tokens: bool = False
    global_features: int = 6
    # ``canonical`` replaces device-scale current/time features in the learned
    # closure with size/current-normalized actuator statistics. Dedicated
    # physical tensors still carry Ip and TF current to the analytic decoder.
    global_feature_mode: str = "legacy"
    # Opt-in machine-state canvas built from analytic vacuum flux and physical
    # grid coordinates.  The final projection is zero-initialized so enabling
    # this on a legacy checkpoint is functionally neutral before training.
    physics_raster_encoder: bool = False
    raster_hidden_channels: int = 24
    # Preserve the first experimental canvas for old artifacts; new transfer
    # probes can keep physical R dependence exclusively in the GS decoder.
    raster_coordinate_mode: str = "cylindrical"
    d_model: int = 128
    n_heads: int = 4
    n_attention_blocks: int = 3
    ff_multiplier: int = 4
    dropout: float = 0.05
    n_pressure_coeffs: int = 6
    n_ffprime_coeffs: int = 10
    outer_iterations: int = 3
    detach_picard_coordinates: bool = False
    picard_damping: float = 0.65
    # Damping stabilizes coordinate iteration, but the submitted field should
    # be the exact Green rendering of the final inferred current source.
    exact_final_source_render: bool = False
    pressure_bandwidth: float = 0.12
    lcfs_temperature: float = 0.035
    # The differentiable saddle/perimeter estimator is deliberately cheaper
    # than the scorer's closed-contour bisection. This factor calibrates its
    # axis-to-separatrix span without introducing machine-specific geometry.
    separatrix_span_scale: float = 1.0
    # The perimeter level is robust during early Picard iterations.  The
    # legacy saddle selector is retained for backward-compatible checkpoints.
    separatrix_estimator: str = "saddle"
    # Use the prior Picard current centroid to choose the nearest genuine local
    # O-point.  This avoids coil-driven edge maxima without a machine-specific
    # axis prior; an empty candidate set still falls back to the global path.
    axis_estimator: str = "current_local"
    # Thomson channels are diagnostic chords rather than 2-D plasma samples.
    # A centered first current iterate is machine-independent and avoids
    # imprinting chord geometry on the equilibrium bootstrap.
    initial_current_center: str = "grid"
    initial_vacuum_gain: float = 0.47
    initial_pressure_fraction: float = 0.25
    use_scalar_calibration: bool = True
    # Use documented physical TF turn counts in the analytic q95/betaN path.
    # Disabled by default so legacy artifacts preserve exact predictions.
    physical_tf_turns: bool = False
    # Optional inference-only scalar equations using the submitted flux itself
    # plus measured Thomson/TF/Ip inputs. No learned image or machine embedding.
    physical_scalar_postprocess: bool = False
    physical_q_scale: float = 1.0
    physical_beta_scale: float = 1.0
    green_softening_cells: float = 0.45
    # Optional latent-conditioned 2-D correction to the otherwise flux-only
    # current profile.  Zero modes deliberately avoids constructing any new
    # parameters, so artifacts written before this option existed still load
    # strictly and produce bit-for-bit the legacy architecture.
    spatial_current_modes: int = 0
    spatial_current_amplitude: float = 0.75
    # Opt-in current source inferred independently of the Thomson pressure
    # branch.  ``legacy_pressure_mixture`` preserves all existing artifacts;
    # ``conditioned_signed_splines`` uses a support-projected exact-Ip basis.
    source_head_mode: str = "legacy_pressure_mixture"
    source_basis_size: int = 4

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "NeuralGSConfig":
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known})
