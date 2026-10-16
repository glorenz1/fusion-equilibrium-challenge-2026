"""Set encoder -> source profiles -> differentiable free-boundary GS solve."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import NeuralGSConfig
from .geometry import MU0
from .physics import (
    current_density_centroid,
    estimate_flux_coordinates,
    green_response,
    initial_current_density,
    modulate_spatial_current,
    plasma_window,
    profile_current_density,
    conditioned_signed_source_current,
    sample_grid,
)


class ThomsonSetEncoder(nn.Module):
    """Permutation-invariant Transformer with a learned pooling token."""

    def __init__(self, config: NeuralGSConfig):
        super().__init__()
        self.coordinate_tokens = config.coordinate_tokens
        self.input_projection = nn.Sequential(
            nn.Linear(config.token_features + 1 + 2 * int(config.coordinate_tokens), config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        self.pool_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.normal_(self.pool_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_multiplier * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.n_attention_blocks)
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        values: torch.Tensor,
        psi_n: torch.Tensor,
        mask: torch.Tensor,
        coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = [psi_n.unsqueeze(-1), values]
        if self.coordinate_tokens:
            if coordinates is None or coordinates.shape[-1] != 2:
                raise ValueError("coordinate-enabled Thomson encoder requires (..., 2) coordinates")
            features.append(coordinates)
        tokens = self.input_projection(torch.cat(features, dim=-1))
        batch = tokens.shape[0]
        pooled = self.pool_token.expand(batch, -1, -1)
        sequence = torch.cat([pooled, tokens], dim=1)
        padding = torch.cat(
            [torch.zeros((batch, 1), dtype=torch.bool, device=mask.device), ~mask.bool()], dim=1
        )
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        return self.output_norm(encoded[:, 0])


def build_physics_raster(
    vacuum_psi: torch.Tensor,
    ip_abs: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    coordinate_mode: str = "cylindrical",
) -> torch.Tensor:
    """Build a machine-normalized actuator canvas.

    The field channels use the analytic PF-coil vacuum flux already rendered
    on the machine's real grid.  Derivatives are taken with respect to the
    normalized canvas coordinates, which avoids teaching the CNN a DIII-D-only
    pixel scale. ``cylindrical`` retains the original six-channel experiment.
    ``canonical`` exposes only grid-relative R/Z and leaves actual cylindrical
    factors to the flux-function basis and Green decoder.
    """

    if vacuum_psi.ndim != 3:
        raise ValueError("vacuum_psi must have shape [B,Z,R]")
    if ip_abs.ndim != 1 or len(ip_abs) != len(vacuum_psi):
        raise ValueError("ip_abs must have shape [B]")
    if vacuum_psi.shape[-2:] != (len(grid_z), len(grid_r)):
        raise ValueError("vacuum canvas does not match the supplied physical grid")
    if len(grid_r) < 2 or len(grid_z) < 2:
        raise ValueError("physics raster requires at least a 2x2 grid")
    if coordinate_mode not in {"cylindrical", "canonical"}:
        raise ValueError(f"unknown raster coordinate mode: {coordinate_mode}")

    # mu0*Ip*R is the natural poloidal-flux scale under a uniform machine-size
    # dilation.  The legacy cylindrical canvas omitted R and is retained
    # exactly for old artifacts; the canonical transfer canvas removes it so a
    # geometrically similar smaller device does not look like a weaker field.
    dtype = vacuum_psi.dtype
    device = vacuum_psi.device
    r = grid_r.to(dtype=dtype, device=device).clamp_min(1.0e-4)
    z = grid_z.to(dtype=dtype, device=device)
    r_ref = torch.sqrt(r[0] * r[-1]).clamp_min(1.0e-4)
    length_scale = r_ref if coordinate_mode == "canonical" else r_ref.new_tensor(1.0)
    current_scale = (
        MU0 * ip_abs.to(dtype=dtype, device=device) * length_scale
    ).abs().clamp_min(1.0e-3)
    flux = (vacuum_psi / current_scale[:, None, None]).clamp(-4.0, 4.0)

    dr = flux.new_tensor(2.0 / max(len(grid_r) - 1, 1))
    dz = flux.new_tensor(2.0 / max(len(grid_z) - 1, 1))
    grad_r = torch.empty_like(flux)
    grad_z = torch.empty_like(flux)
    grad_r[..., 1:-1] = (flux[..., 2:] - flux[..., :-2]) / (2.0 * dr)
    grad_r[..., 0] = (flux[..., 1] - flux[..., 0]) / dr
    grad_r[..., -1] = (flux[..., -1] - flux[..., -2]) / dr
    grad_z[:, 1:-1] = (flux[:, 2:] - flux[:, :-2]) / (2.0 * dz)
    grad_z[:, 0] = (flux[:, 1] - flux[:, 0]) / dz
    grad_z[:, -1] = (flux[:, -1] - flux[:, -2]) / dz
    grad_r = grad_r.clamp(-8.0, 8.0)
    grad_z = grad_z.clamp(-8.0, 8.0)

    z_center = 0.5 * (z[0] + z[-1])
    z_half_span = (0.5 * (z[-1] - z[0]).abs()).clamp_min(1.0e-4)
    vertical = ((z - z_center) / z_half_span).clamp(-2.0, 2.0)
    shape = (len(vacuum_psi), len(grid_z), len(grid_r))
    vertical = vertical[None, :, None].expand(shape)
    if coordinate_mode == "canonical":
        radial = torch.linspace(-1.0, 1.0, len(r), device=device, dtype=dtype)
        radial = radial[None, None, :].expand(shape)
        return torch.stack([flux, grad_r, grad_z, radial, vertical], dim=1)
    radial = (r / r_ref).clamp(0.0, 8.0)
    inverse_radial = (r_ref / r).clamp(0.0, 8.0)
    radial = radial[None, None, :].expand(shape)
    inverse_radial = inverse_radial[None, None, :].expand(shape)
    return torch.stack([flux, grad_r, grad_z, radial, vertical, inverse_radial], dim=1)


class PhysicsRasterEncoder(nn.Module):
    """Encode the analytic actuator field without assuming a coil inventory."""

    def __init__(self, config: NeuralGSConfig):
        super().__init__()
        hidden = int(config.raster_hidden_channels)
        if hidden < 4:
            raise ValueError("raster_hidden_channels must be at least 4")
        if config.raster_coordinate_mode not in {"cylindrical", "canonical"}:
            raise ValueError(
                f"unknown raster coordinate mode: {config.raster_coordinate_mode}"
            )
        self.coordinate_mode = config.raster_coordinate_mode
        input_channels = 5 if self.coordinate_mode == "canonical" else 6
        self.convolutions = nn.Sequential(
            nn.Conv2d(input_channels, hidden, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 2 * hidden, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, 2 * hidden),
            nn.GELU(),
            nn.Conv2d(2 * hidden, 2 * hidden, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, 2 * hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output_projection = nn.Linear(2 * hidden, config.d_model)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        vacuum_psi: torch.Tensor,
        ip_abs: torch.Tensor,
        grid_r: torch.Tensor,
        grid_z: torch.Tensor,
    ) -> torch.Tensor:
        canvas = build_physics_raster(
            vacuum_psi,
            ip_abs,
            grid_r,
            grid_z,
            coordinate_mode=self.coordinate_mode,
        )
        hidden = self.convolutions(canvas).flatten(1)
        return self.output_projection(hidden)


class NeuralGS(nn.Module):
    """Learn only low-dimensional source profiles; render psi with GS physics."""

    def __init__(self, config: NeuralGSConfig | None = None):
        super().__init__()
        self.config = config or NeuralGSConfig()
        c = self.config
        if c.global_feature_mode not in {"legacy", "canonical"}:
            raise ValueError(f"unknown global feature mode: {c.global_feature_mode}")
        self.set_encoder = ThomsonSetEncoder(c)
        self.global_encoder = nn.Sequential(
            nn.Linear(c.global_features, c.d_model), nn.GELU(), nn.Linear(c.d_model, c.d_model)
        )
        self.physics_raster = (
            PhysicsRasterEncoder(c) if c.physics_raster_encoder else None
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(c.d_model), nn.Linear(c.d_model, c.d_model), nn.GELU()
        )
        n_profile = c.n_pressure_coeffs + c.n_ffprime_coeffs + 2
        self.profile_head = nn.Linear(c.d_model, n_profile)
        if c.source_head_mode not in {
            "legacy_pressure_mixture", "conditioned_signed_splines"
        }:
            raise ValueError(f"unknown source_head_mode: {c.source_head_mode}")
        if c.source_basis_size < 1:
            raise ValueError("source_basis_size must be positive")
        if c.source_head_mode == "conditioned_signed_splines":
            source_dimension = 2 * c.source_basis_size
            self.source_head: nn.Linear | None = nn.Linear(c.d_model, source_dimension)
            # Start at the exact-Ip base current when probing a frozen legacy
            # encoder.  Conditioning is loaded from a training-only artifact.
            nn.init.zeros_(self.source_head.weight)
            nn.init.zeros_(self.source_head.bias)
            self.register_buffer(
                "source_basis_transform", torch.eye(source_dimension), persistent=True
            )
        else:
            self.source_head = None
        self.gauge_head = nn.Sequential(nn.Linear(c.d_model, 32), nn.GELU(), nn.Linear(32, 1))
        # Two outputs per scalar: a bounded multiplicative and additive residual
        # around the physical estimate.  These are deliberately tiny heads.
        self.scalar_head = nn.Sequential(nn.Linear(c.d_model, 48), nn.GELU(), nn.Linear(48, 4))
        if c.spatial_current_modes < 0 or c.spatial_current_modes > 12:
            raise ValueError("spatial_current_modes must be between 0 and 12")
        if c.source_head_mode != "legacy_pressure_mixture" and c.spatial_current_modes:
            raise ValueError(
                "spatial_current_modes cannot be combined with the signed spline source head"
            )
        if c.spatial_current_modes:
            self.spatial_current_head: nn.Linear | None = nn.Linear(
                c.d_model, c.spatial_current_modes
            )
            # Identity at initialization is essential when adding the head to
            # an already trained Stage-A source model.
            nn.init.zeros_(self.spatial_current_head.weight)
            nn.init.zeros_(self.spatial_current_head.bias)
        else:
            self.spatial_current_head = None
        self.raw_vacuum_gain = nn.Parameter(torch.tensor(math.log(math.expm1(c.initial_vacuum_gain))))
        # F_boundary ~= N*mu0*I_tf/(2*pi); N is unavailable in the released
        # toroidal-coil metadata, so this one shared calibration is unavoidable.
        self.raw_tf_factor = nn.Parameter(torch.tensor(math.log(math.expm1(20.0))))

    @property
    def vacuum_gain(self) -> torch.Tensor:
        return F.softplus(self.raw_vacuum_gain)

    def set_source_basis_transform(self, transform: torch.Tensor) -> None:
        """Load a frozen training-derived source conditioning transform."""

        if self.source_head is None:
            raise RuntimeError("the configured model has no conditioned source head")
        expected = self.source_basis_transform.shape
        if transform.shape != expected:
            raise ValueError(f"source transform shape {transform.shape} != {expected}")
        with torch.no_grad():
            self.source_basis_transform.copy_(
                transform.to(
                    device=self.source_basis_transform.device,
                    dtype=self.source_basis_transform.dtype,
                )
            )

    def _profiles(
        self,
        token_values: torch.Tensor,
        token_s: torch.Tensor,
        token_mask: torch.Tensor,
        global_features: torch.Tensor,
        token_coordinates: torch.Tensor | None = None,
        raster_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent = self.set_encoder(token_values, token_s, token_mask, token_coordinates)
        latent = latent + self.global_encoder(global_features)
        if raster_context is not None:
            latent = latent + raster_context
        latent = self.fusion(latent)
        raw = self.profile_head(latent)
        kp = self.config.n_pressure_coeffs
        kf = self.config.n_ffprime_coeffs
        return {
            "latent": latent,
            "pressure_correction": raw[:, :kp],
            "ffprime_coefficients": raw[:, kp : kp + kf],
            "pressure_fraction": torch.sigmoid(raw[:, -2]),
            "pressure_scale": 1.0 + 4.0 * torch.sigmoid(raw[:, -1]),
        }

    def _current_from_coordinates(
        self,
        psi_n_full: torch.Tensor,
        plasma_mask_full: torch.Tensor,
        batch: dict[str, torch.Tensor],
        grid_r: torch.Tensor,
        grid_z: torch.Tensor,
        source_r: torch.Tensor,
        source_z: torch.Tensor,
        source_weights: torch.Tensor,
        token_psi_n_full: torch.Tensor | None = None,
        raster_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        token_field = psi_n_full if token_psi_n_full is None else token_psi_n_full
        token_s = sample_grid(token_field, batch["token_coords"], grid_r, grid_z).clamp(-0.5, 1.5)
        token_coordinates = None
        if self.config.coordinate_tokens:
            r_scale = (grid_r[-1] - grid_r[0]).clamp_min(1.0e-6)
            z_scale = (grid_z[-1] - grid_z[0]).clamp_min(1.0e-6)
            token_coordinates = torch.stack(
                [
                    2.0 * (batch["token_coords"][..., 0] - grid_r[0]) / r_scale - 1.0,
                    2.0 * (batch["token_coords"][..., 1] - grid_z[0]) / z_scale - 1.0,
                ],
                dim=-1,
            ).clamp(-2.0, 2.0)
        profiles = self._profiles(
            batch["token_values"],
            token_s,
            batch["token_mask"],
            batch[
                "canonical_global_features"
                if self.config.global_feature_mode == "canonical"
                else "global_features"
            ],
            token_coordinates,
            # The legacy profile source has no separate source-only branch, so
            # fuse the normalized actuator canvas into its low-dimensional
            # closure latent.  Signed-source experiments retain their original
            # source-only raster behavior below.
            raster_context if self.source_head is None else None,
        )
        stride = self.config.source_stride
        psi_n_source = psi_n_full[:, ::stride, ::stride]
        mask_source = plasma_mask_full[:, ::stride, ::stride]
        # A non-divisible future grid can append its last point in the Green
        # builder.  The official 65 grid is exactly represented by stride 2.
        psi_n_source = psi_n_source[:, : len(source_z), : len(source_r)]
        mask_source = mask_source[:, : len(source_z), : len(source_r)]
        legacy_current, pressure = profile_current_density(
            psi_n_source,
            mask_source,
            token_s,
            batch["token_values"],
            batch["token_mask"],
            profiles["pressure_correction"],
            profiles["pressure_scale"],
            profiles["ffprime_coefficients"],
            profiles["pressure_fraction"],
            batch["ip_abs"],
            source_r,
            source_z,
            source_weights,
            self.config.pressure_bandwidth,
        )
        if self.source_head is None:
            current = legacy_current
        else:
            # Keep the actuator canvas on the current-source branch.  Feeding
            # it into the shared latent silently shifts the frozen pressure,
            # gauge, and scalar heads when probing a legacy checkpoint.
            source_latent = profiles["latent"]
            if raster_context is not None:
                source_latent = source_latent + raster_context
                profiles["source_latent"] = source_latent
            source_coefficients = self.source_head(source_latent)
            current, source_base, source_basis = conditioned_signed_source_current(
                psi_n_source,
                mask_source,
                source_coefficients,
                batch["ip_abs"],
                source_r,
                source_z,
                source_weights,
                self.source_basis_transform,
            )
            profiles["source_coefficients"] = source_coefficients
            profiles["source_base"] = source_base
            profiles["source_basis"] = source_basis
        if self.spatial_current_head is not None:
            spatial_coefficients = self.spatial_current_head(profiles["latent"])
            current = modulate_spatial_current(
                current,
                spatial_coefficients,
                batch["ip_abs"],
                source_r,
                source_z,
                source_weights,
                self.config.spatial_current_amplitude,
            )
            profiles["spatial_current_coefficients"] = spatial_coefficients
        return current, pressure, profiles

    def _physics_scalars(
        self,
        psi_n: torch.Tensor,
        pressure: torch.Tensor,
        current: torch.Tensor,
        profiles: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        source_r: torch.Tensor,
        source_z: torch.Tensor,
        source_weights: torch.Tensor,
        axis_value: torch.Tensor,
        separatrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_s = psi_n[:, :: self.config.source_stride, :: self.config.source_stride]
        source_s = source_s[:, : len(source_z), : len(source_r)]
        mask = torch.sigmoid((1.0 - source_s) / self.config.lcfs_temperature)
        rr, _ = torch.meshgrid(source_r, source_z, indexing="xy")
        area = (mask.flatten(1) * source_weights[None]).sum(dim=1).clamp_min(1.0e-8)
        minor_radius = torch.sqrt(area / math.pi)
        volume_weight = (mask * rr[None]).flatten(1) * source_weights[None]
        pressure_average = (pressure.flatten(1) * volume_weight).sum(dim=1) / volume_weight.sum(dim=1).clamp_min(1.0e-8)

        tf_ampere = batch.get("tf_ampere")
        if tf_ampere is None:
            # Backward-compatible path for prepared caches written before TF
            # current was separated from the learned global feature vector.
            tf_ampere = 1.0e4 * torch.expm1(
                batch["global_features"][:, 1].clamp_min(0.0)
            )
        if self.config.physical_tf_turns:
            tf_turns = batch.get("tf_turns")
            if tf_turns is None:
                raise ValueError("physical TF scalar mode requires tf_turns")
            f_boundary = MU0 * tf_turns * tf_ampere / (2.0 * math.pi)
        else:
            f_boundary = F.softplus(self.raw_tf_factor) * MU0 * tf_ampere
        current_weight = current * source_weights.reshape(1, len(source_z), len(source_r))
        r_axis_proxy = (current_weight * rr[None]).sum(dim=(1, 2)) / current_weight.sum(dim=(1, 2)).clamp_min(1.0e-8)
        b_toroidal = f_boundary / r_axis_proxy.clamp_min(0.05)
        beta_t = 2.0 * MU0 * pressure_average / b_toroidal.square().clamp_min(1.0e-8)
        ip_ma = batch["ip_abs"] / 1.0e6
        beta_n = 100.0 * beta_t * minor_radius * b_toroidal / ip_ma.clamp_min(0.02)

        width = 0.035
        delta = torch.exp(-0.5 * ((source_s - 0.95) / width) ** 2) / (math.sqrt(2.0 * math.pi) * width)
        contour_integral = (
            (delta / rr[None].clamp_min(0.03)).flatten(1) * source_weights[None]
        ).sum(dim=1)
        flux_span = (axis_value - separatrix).clamp_min(1.0e-5)
        q95 = f_boundary * contour_integral / (2.0 * math.pi * flux_span)
        return q95.clamp(0.05, 50.0), beta_n.clamp(0.0, 20.0)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        green: torch.Tensor,
        source_r: torch.Tensor,
        source_z: torch.Tensor,
        source_weights: torch.Tensor,
        grid_r: torch.Tensor,
        grid_z: torch.Tensor,
        teacher_psi_n: torch.Tensor | None = None,
        teacher_token_psi_n: torch.Tensor | None = None,
        teacher_source_psi_n: torch.Tensor | None = None,
        coordinate_replay: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
        return_coordinate_trajectory: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Solve a batch sharing one physical grid and Green operator.

        ``teacher_psi_n`` activates Stage A: profiles are inferred on the true
        flux coordinates and a current source is returned without using target
        pixels as inputs to the iterative equilibrium prediction.
        """

        raster_context = (
            self.physics_raster(
                batch["vacuum_psi"], batch["ip_abs"], grid_r, grid_z
            )
            if self.physics_raster is not None
            else None
        )

        if teacher_psi_n is not None:
            source_coordinate = (
                teacher_psi_n if teacher_source_psi_n is None else teacher_source_psi_n
            )
            token_coordinate = (
                teacher_psi_n if teacher_token_psi_n is None else teacher_token_psi_n
            )
            source_coordinate = source_coordinate.to(
                device=teacher_psi_n.device, dtype=teacher_psi_n.dtype
            )
            token_coordinate = token_coordinate.to(
                device=teacher_psi_n.device, dtype=teacher_psi_n.dtype
            )
            mask = torch.sigmoid(
                (1.0 - source_coordinate) / self.config.lcfs_temperature
            )
            current, pressure, profiles = self._current_from_coordinates(
                source_coordinate,
                mask,
                batch,
                grid_r,
                grid_z,
                source_r,
                source_z,
                source_weights,
                token_psi_n_full=token_coordinate,
                raster_context=raster_context,
            )
            return {
                "current_density": current,
                "pressure": pressure,
                "psi_n": source_coordinate,
                **profiles,
            }

        initial_current = initial_current_density(
            batch["ip_abs"], batch["token_coords"], batch["token_values"], batch["token_mask"],
            source_r, source_z, source_weights, self.config.initial_current_center,
        )
        full_shape = batch["vacuum_psi"].shape
        plasma = green_response(initial_current, green).reshape(full_shape)
        psi = self.vacuum_gain * batch["vacuum_psi"] + plasma
        profiles = None
        pressure = None
        current = initial_current
        axis_value = separatrix = None
        psi_n = None
        damping = self.config.picard_damping
        coordinate_trajectory = []
        for iteration in range(self.config.outer_iterations):
            coordinate_psi = (
                psi.detach() if self.config.detach_picard_coordinates else psi
            )
            if self.config.axis_estimator == "current_local":
                axis_hint_r, axis_hint_z = current_density_centroid(
                    current.detach(), source_r, source_z, source_weights
                )
            else:
                axis_hint_r = axis_hint_z = None
            psi_n, plasma_mask, axis_value, separatrix = estimate_flux_coordinates(
                coordinate_psi,
                grid_r,
                grid_z,
                temperature=self.config.lcfs_temperature,
                separatrix_span_scale=self.config.separatrix_span_scale,
                separatrix_estimator=self.config.separatrix_estimator,
                axis_estimator=self.config.axis_estimator,
                axis_hint_r=axis_hint_r,
                axis_hint_z=axis_hint_z,
            )
            coordinate_trajectory.append(psi_n.detach())
            replay_psi_n = None
            if coordinate_replay is not None:
                if not coordinate_replay:
                    raise ValueError("coordinate_replay cannot be empty")
                replay_psi_n = coordinate_replay[min(iteration, len(coordinate_replay) - 1)]
                if replay_psi_n.shape != psi_n.shape:
                    raise ValueError("coordinate replay shape does not match the model batch")
                replay_psi_n = replay_psi_n.to(device=psi_n.device, dtype=psi_n.dtype).detach()
            source_coordinate = (
                teacher_source_psi_n
                if teacher_source_psi_n is not None
                else replay_psi_n if replay_psi_n is not None else psi_n
            )
            token_coordinate = (
                teacher_token_psi_n
                if teacher_token_psi_n is not None
                else replay_psi_n if replay_psi_n is not None else psi_n
            )
            source_coordinate = source_coordinate.to(device=psi_n.device, dtype=psi_n.dtype)
            token_coordinate = token_coordinate.to(device=psi_n.device, dtype=psi_n.dtype)
            source_mask = torch.sigmoid(
                (1.0 - source_coordinate) / self.config.lcfs_temperature
            )
            current, pressure, profiles = self._current_from_coordinates(
                source_coordinate,
                source_mask,
                batch,
                grid_r,
                grid_z,
                source_r,
                source_z,
                source_weights,
                token_psi_n_full=token_coordinate,
                raster_context=raster_context,
            )
            plasma_next = green_response(current, green).reshape(full_shape)
            solved = self.vacuum_gain * batch["vacuum_psi"] + plasma_next
            psi = damping * solved + (1.0 - damping) * psi

        assert profiles is not None and pressure is not None and psi_n is not None
        assert axis_value is not None and separatrix is not None
        if self.config.exact_final_source_render:
            psi = solved
        gauge = 0.65 * torch.tanh(self.gauge_head(profiles["latent"]).squeeze(-1)) * MU0 * batch["ip_abs"]
        psi = psi + gauge[:, None, None]
        q_physics, beta_physics = self._physics_scalars(
            psi_n, pressure, current, profiles, batch, source_r, source_z, source_weights,
            axis_value, separatrix,
        )
        if self.config.use_scalar_calibration:
            # Scalar supervision previously overwhelmed the shared equilibrium
            # encoder (q95 alone had ~13x the flux gradient norm).  Fit the
            # lightweight calibration head on frozen equilibrium features and
            # physical proxies; a separate direct scalar model is used at final
            # inference when available.
            residual = self.scalar_head(profiles["latent"].detach())
            q_base = q_physics.detach()
            beta_base = beta_physics.detach()
            q95 = q_base * torch.exp(torch.tanh(residual[:, 0])) + 2.0 * torch.tanh(residual[:, 1])
            beta_n = beta_base * torch.exp(0.7 * torch.tanh(residual[:, 2])) + torch.tanh(residual[:, 3])
        else:
            q95, beta_n = q_physics, beta_physics
        result = {
            "psirz": psi,
            "q95": q95.clamp(0.1, 30.0),
            "betaN": beta_n.clamp(0.0, 10.0),
            "q95_physics": q_physics,
            "betaN_physics": beta_physics,
            "current_density": current,
            "pressure": pressure,
            "psi_n": psi_n,
            "gauge": gauge,
            "vacuum_gain": self.vacuum_gain,
            **profiles,
        }
        if return_coordinate_trajectory:
            result["coordinate_trajectory"] = tuple(coordinate_trajectory)
        return result
