"""DIII-D-specialized encoder for Fusion Equilibrium Challenge 1.

The old cross-machine model stopped at a constrained physical decoder state.
That representation has a measured target-seeing ceiling below the Challenge-1
goal, so the active model also predicts a direct, normalized 65x65 flux
residual.  The physical state remains available as an auxiliary regularizer;
the direct residual is the submitted DIII-D field path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .common_inputs import (
    MODEL_SPATIAL_CHANNELS,
    SHARED_SCALAR_CHANNELS,
    THOMSON_TOKEN_FEATURES,
)


def _group_count(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups:
        groups -= 1
    return groups


@dataclass(frozen=True)
class CausalHybridEncoderConfig:
    spatial_channels: int = len(MODEL_SPATIAL_CHANNELS)
    scalar_features: int = len(SHARED_SCALAR_CHANNELS)
    thomson_features: int = len(THOMSON_TOKEN_FEATURES)
    spatial_widths: tuple[int, int, int, int] = (24, 48, 72, 96)
    spatial_output_width: int = 48
    thomson_width: int = 64
    thomson_latents: int = 8
    thomson_heads: int = 4
    thomson_self_blocks: int = 2
    thomson_ff_width: int = 128
    scalar_width: int = 64
    fusion_width: int = 128
    hidden_size: int = 128
    harmonic_count: int = 16
    current_grid_size: int = 33
    dropout: float = 0.05
    tau_min_ms: float = 0.5
    tau_max_ms: float = 100.0
    minimum_tau_ms: float = 1.0e-3
    film_bound: float = 0.25
    harmonic_bound: float = 5.0
    gauge_bound: float = 5.0
    flux_residual_bound: float = 8.0
    bidirectional_context: bool = True

    def __post_init__(self) -> None:
        positive_integers = {
            "spatial_channels": self.spatial_channels,
            "scalar_features": self.scalar_features,
            "thomson_features": self.thomson_features,
            "spatial_output_width": self.spatial_output_width,
            "thomson_width": self.thomson_width,
            "thomson_latents": self.thomson_latents,
            "thomson_heads": self.thomson_heads,
            "thomson_self_blocks": self.thomson_self_blocks,
            "thomson_ff_width": self.thomson_ff_width,
            "scalar_width": self.scalar_width,
            "fusion_width": self.fusion_width,
            "hidden_size": self.hidden_size,
            "harmonic_count": self.harmonic_count,
            "current_grid_size": self.current_grid_size,
        }
        if any(value < 1 for value in positive_integers.values()):
            raise ValueError(f"model dimensions must be positive: {positive_integers}")
        if len(self.spatial_widths) != 4 or any(value < 1 for value in self.spatial_widths):
            raise ValueError("spatial_widths must contain four positive values")
        if self.thomson_width % self.thomson_heads:
            raise ValueError("thomson_width must be divisible by thomson_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not 0.0 < self.tau_min_ms <= self.tau_max_ms:
            raise ValueError("tau bounds must be positive and ordered")
        if self.minimum_tau_ms <= 0.0 or self.minimum_tau_ms >= self.tau_min_ms:
            raise ValueError("minimum_tau_ms must be positive and below tau_min_ms")
        if (
            self.film_bound <= 0.0
            or self.harmonic_bound <= 0.0
            or self.gauge_bound <= 0.0
            or self.flux_residual_bound <= 0.0
        ):
            raise ValueError("output bounds must be positive")


@dataclass(frozen=True)
class CausalHybridEncoderOutput:
    """DIII-D field prediction and auxiliary decoder state for every frame."""

    normalized_flux_residual: torch.Tensor
    current_mass: torch.Tensor
    current_logits: torch.Tensor
    harmonics: torch.Tensor
    gauge: torch.Tensor
    q95: torch.Tensor
    betaN: torch.Tensor
    scalar_correction: torch.Tensor
    hidden_states: torch.Tensor
    final_hidden_state: torch.Tensor
    tau_ms: torch.Tensor

    def challenge1_flux(
        self,
        active_psi_Wb_per_rad: torch.Tensor,
        flux_scale_Wb_per_rad: torch.Tensor,
    ) -> torch.Tensor:
        """Render the direct DIII-D path in physical flux units."""

        expected = self.normalized_flux_residual.shape
        if active_psi_Wb_per_rad.shape != expected:
            raise ValueError("active flux shape differs from the direct DIII-D head")
        if flux_scale_Wb_per_rad.shape != expected[:2]:
            raise ValueError("flux scale must have shape (batch, time)")
        return active_psi_Wb_per_rad + (
            flux_scale_Wb_per_rad[..., None, None]
            * self.normalized_flux_residual
        )

    def decoder_state(self) -> dict[str, torch.Tensor]:
        """Return names used by the existing deterministic decoder boundary."""

        return {
            "current_mass": self.current_mass,
            "harmonic": self.harmonics,
            "gauge": self.gauge,
        }


class ResidualConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.convolution_1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding=1
        )
        self.normalization_1 = nn.GroupNorm(
            _group_count(output_channels), output_channels
        )
        self.convolution_2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1
        )
        self.normalization_2 = nn.GroupNorm(
            _group_count(output_channels), output_channels
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, kernel_size=1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        hidden = F.silu(self.normalization_1(self.convolution_1(values)))
        hidden = self.normalization_2(self.convolution_2(hidden))
        return F.silu(hidden + residual)


class CompactResidualUNet(nn.Module):
    """Encode 65x65 fields and decode only to the 33x33 current grid."""

    def __init__(self, config: CausalHybridEncoderConfig) -> None:
        super().__init__()
        width_65, width_33, width_17, width_9 = config.spatial_widths
        self.stem = nn.Conv2d(
            config.spatial_channels, width_65, kernel_size=3, padding=1
        )
        self.level_65 = ResidualConvBlock(width_65, width_65)
        self.down_33 = nn.Conv2d(
            width_65, width_33, kernel_size=3, stride=2, padding=1
        )
        self.level_33 = ResidualConvBlock(width_33, width_33)
        self.down_17 = nn.Conv2d(
            width_33, width_17, kernel_size=3, stride=2, padding=1
        )
        self.level_17 = ResidualConvBlock(width_17, width_17)
        self.down_9 = nn.Conv2d(
            width_17, width_9, kernel_size=3, stride=2, padding=1
        )
        self.level_9 = ResidualConvBlock(width_9, width_9)

        self.up_17 = nn.Conv2d(width_9, width_17, kernel_size=3, padding=1)
        self.decode_17 = ResidualConvBlock(2 * width_17, width_17)
        self.up_33 = nn.Conv2d(width_17, width_33, kernel_size=3, padding=1)
        self.decode_33 = ResidualConvBlock(2 * width_33, config.spatial_output_width)
        self.summary_norm = nn.LayerNorm(width_9)

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        level_65 = self.level_65(self.stem(values))
        level_33 = self.level_33(self.down_33(level_65))
        level_17 = self.level_17(self.down_17(level_33))
        level_9 = self.level_9(self.down_9(level_17))

        up_17 = F.interpolate(
            level_9, size=level_17.shape[-2:], mode="bilinear", align_corners=False
        )
        up_17 = self.up_17(up_17)
        decoded_17 = self.decode_17(torch.cat([up_17, level_17], dim=1))
        up_33 = F.interpolate(
            decoded_17, size=level_33.shape[-2:], mode="bilinear", align_corners=False
        )
        up_33 = self.up_33(up_33)
        decoded_33 = self.decode_33(torch.cat([up_33, level_33], dim=1))
        summary = self.summary_norm(F.adaptive_avg_pool2d(level_9, 1).flatten(1))
        return decoded_33, summary, level_65


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden_width: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, hidden_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, width),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class ThomsonCrossAttentionBlock(nn.Module):
    def __init__(self, config: CausalHybridEncoderConfig) -> None:
        super().__init__()
        width = config.thomson_width
        self.query_norm = nn.LayerNorm(width)
        self.token_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width,
            config.thomson_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(config.dropout)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = FeedForward(
            width, config.thomson_ff_width, config.dropout
        )

    def forward(
        self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(latents),
            self.token_norm(tokens),
            self.token_norm(tokens),
            key_padding_mask=~token_valid,
            need_weights=False,
        )
        latents = latents + self.output_dropout(attended)
        return latents + self.feedforward(self.feedforward_norm(latents))


class ThomsonLatentBlock(nn.Module):
    def __init__(self, config: CausalHybridEncoderConfig) -> None:
        super().__init__()
        width = config.thomson_width
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width,
            config.thomson_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(config.dropout)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = FeedForward(
            width, config.thomson_ff_width, config.dropout
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(latents)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        latents = latents + self.output_dropout(attended)
        return latents + self.feedforward(self.feedforward_norm(latents))


class CompactThomsonPerceiver(nn.Module):
    """Eight latent slots with one cross- and two latent-attention stages."""

    def __init__(self, config: CausalHybridEncoderConfig) -> None:
        super().__init__()
        width = config.thomson_width
        self.token_projection = nn.Sequential(
            nn.Linear(config.thomson_features, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.latents = nn.Parameter(torch.empty(config.thomson_latents, width))
        self.null_token = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.latents, std=0.02)
        nn.init.normal_(self.null_token, std=0.02)
        self.cross_attention = ThomsonCrossAttentionBlock(config)
        self.latent_blocks = nn.ModuleList(
            ThomsonLatentBlock(config)
            for _ in range(config.thomson_self_blocks)
        )
        self.output_norm = nn.LayerNorm(2 * width)

    def forward(
        self,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("Thomson values must have shape (batch, token, feature)")
        if attention_mask.shape != values.shape[:2]:
            raise ValueError("Thomson attention mask has the wrong shape")
        tokens = self.token_projection(values)
        batch = len(values)
        null_token = self.null_token.expand(batch, -1, -1)
        tokens = torch.cat([tokens, null_token], dim=1)
        valid = torch.cat(
            [
                attention_mask.bool(),
                torch.ones((batch, 1), dtype=torch.bool, device=values.device),
            ],
            dim=1,
        )
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        latents = self.cross_attention(latents, tokens, valid)
        for block in self.latent_blocks:
            latents = block(latents)
        pooled = torch.cat([latents.mean(dim=1), latents.amax(dim=1)], dim=-1)
        return self.output_norm(pooled)


class SharedScalarEncoder(nn.Module):
    def __init__(self, config: CausalHybridEncoderConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.scalar_features, config.scalar_width),
            nn.LayerNorm(config.scalar_width),
            nn.SiLU(),
            nn.Linear(config.scalar_width, config.scalar_width),
            nn.SiLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    """Stable inverse of softplus for strictly positive values."""

    return values + torch.log(-torch.expm1(-values))


class TimeDecayGRU(nn.Module):
    """GRU whose previous state decays using real frame spacing in ms."""

    def __init__(self, input_size: int, config: CausalHybridEncoderConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.minimum_tau_ms = float(config.minimum_tau_ms)
        tau = torch.logspace(
            math.log10(config.tau_min_ms),
            math.log10(config.tau_max_ms),
            config.hidden_size,
        )
        raw_tau = _inverse_softplus(tau - self.minimum_tau_ms)
        self.raw_tau_ms = nn.Parameter(raw_tau)
        self.cell = nn.GRUCell(input_size, config.hidden_size)

    @property
    def tau_ms(self) -> torch.Tensor:
        return F.softplus(self.raw_tau_ms) + self.minimum_tau_ms

    def forward(
        self,
        values: torch.Tensor,
        delta_time_ms: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        sequence_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3:
            raise ValueError("temporal input must have shape (batch, time, feature)")
        if delta_time_ms.shape != values.shape[:2]:
            raise ValueError("delta_time_ms must have shape (batch, time)")
        if not torch.isfinite(delta_time_ms).all() or torch.any(delta_time_ms < 0.0):
            raise ValueError("delta_time_ms must be finite and nonnegative")
        batch, time, _ = values.shape
        if initial_state is None:
            state = values.new_zeros((batch, self.hidden_size))
        else:
            if initial_state.shape != (batch, self.hidden_size):
                raise ValueError("initial temporal state has the wrong shape")
            state = initial_state.to(dtype=values.dtype, device=values.device)
        if sequence_mask is None:
            valid_sequence = torch.ones(
                (batch, time), dtype=torch.bool, device=values.device
            )
        else:
            if sequence_mask.shape != (batch, time):
                raise ValueError("sequence_mask must have shape (batch, time)")
            valid_sequence = sequence_mask.bool()

        tau = self.tau_ms.to(dtype=values.dtype, device=values.device)
        states = []
        for index in range(time):
            decay = torch.exp(-delta_time_ms[:, index, None].to(values) / tau[None])
            candidate = self.cell(values[:, index], state * decay)
            state = torch.where(valid_sequence[:, index, None], candidate, state)
            states.append(state)
        if states:
            sequence = torch.stack(states, dim=1)
        else:
            sequence = values.new_zeros((batch, 0, self.hidden_size))
        return sequence, state


class BidirectionalTimeDecayGRU(nn.Module):
    """Use past and future DIII-D observations around every EFIT frame."""

    def __init__(self, input_size: int, config: CausalHybridEncoderConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.forward_direction = TimeDecayGRU(input_size, config)
        self.backward_direction = TimeDecayGRU(input_size, config)
        self.projection = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
        )

    @property
    def tau_ms(self) -> torch.Tensor:
        return torch.stack(
            [
                self.forward_direction.tau_ms,
                self.backward_direction.tau_ms,
            ]
        )

    def forward(
        self,
        values: torch.Tensor,
        delta_time_ms: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        sequence_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if initial_state is not None:
            raise ValueError("bidirectional Challenge-1 context cannot be chunk-stateful")
        forward, _ = self.forward_direction(
            values, delta_time_ms, sequence_mask=sequence_mask
        )
        # On the reversed pass, elapsed time for original frame t is the gap
        # from t to t+1.  The last original frame starts with a zero gap.
        reverse_delta = torch.cat(
            [
                delta_time_ms[:, 1:],
                torch.zeros_like(delta_time_ms[:, :1]),
            ],
            dim=1,
        ).flip(1)
        reverse_mask = None if sequence_mask is None else sequence_mask.flip(1)
        backward_reversed, _ = self.backward_direction(
            values.flip(1), reverse_delta, sequence_mask=reverse_mask
        )
        backward = backward_reversed.flip(1)
        sequence = self.projection(torch.cat([forward, backward], dim=-1))
        if sequence_mask is None:
            final = sequence[:, -1]
        else:
            indices = sequence_mask.long().sum(dim=1).sub(1).clamp_min(0)
            # Sequence windows are left padded by the current data pipeline.
            indices = indices + (sequence.shape[1] - sequence_mask.long().sum(dim=1))
            final = sequence[
                torch.arange(len(sequence), device=sequence.device), indices
            ]
        return sequence, final


class CausalHybridEncoderV1(nn.Module):
    """DIII-D Challenge-1 encoder (legacy class name kept for artifacts)."""

    def __init__(self, config: CausalHybridEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausalHybridEncoderConfig()
        config = self.config
        self.spatial_encoder = CompactResidualUNet(config)
        self.thomson_encoder = CompactThomsonPerceiver(config)
        self.scalar_encoder = SharedScalarEncoder(config)
        fusion_input = (
            config.spatial_widths[-1]
            + 2 * config.thomson_width
            + config.scalar_width
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, config.fusion_width),
            nn.LayerNorm(config.fusion_width),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_width, config.fusion_width),
            nn.SiLU(),
        )
        self.temporal = (
            BidirectionalTimeDecayGRU(config.fusion_width, config)
            if config.bidirectional_context
            else TimeDecayGRU(config.fusion_width, config)
        )

        self.film = nn.Linear(config.hidden_size, 2 * config.spatial_output_width)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.current_head = nn.Sequential(
            ResidualConvBlock(
                config.spatial_output_width, config.spatial_output_width
            ),
            nn.Conv2d(config.spatial_output_width, 1, kernel_size=1),
        )
        width_65 = config.spatial_widths[0]
        self.flux_film = nn.Linear(config.hidden_size, 2 * width_65)
        nn.init.zeros_(self.flux_film.weight)
        nn.init.zeros_(self.flux_film.bias)
        self.flux_head = nn.Sequential(
            ResidualConvBlock(width_65, width_65),
            nn.Conv2d(width_65, 1, kernel_size=1),
        )
        nn.init.zeros_(self.flux_head[-1].weight)
        with torch.no_grad():
            self.flux_head[-1].bias.fill_(
                math.atanh(1.0 / config.flux_residual_bound)
            )
        self.harmonic_head = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.SiLU(),
            nn.Linear(64, config.harmonic_count),
        )
        self.gauge_head = nn.Sequential(
            nn.Linear(config.hidden_size, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.scalar_head = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(
        self,
        spatial_values: torch.Tensor,
        shared_scalars: torch.Tensor,
        thomson_values: torch.Tensor,
        thomson_attention_mask: torch.Tensor,
        delta_time_ms: torch.Tensor,
        *,
        initial_hidden_state: torch.Tensor | None = None,
        sequence_mask: torch.Tensor | None = None,
        current_support_mask: torch.Tensor | None = None,
    ) -> CausalHybridEncoderOutput:
        if spatial_values.ndim != 5:
            raise ValueError("spatial values must have shape (batch, time, channel, Z, R)")
        batch, time, channels, height, width = spatial_values.shape
        if time < 1:
            raise ValueError("encoder sequences must contain at least one frame")
        if channels != self.config.spatial_channels:
            raise ValueError("spatial input channel count differs from the model contract")
        if shared_scalars.shape != (batch, time, self.config.scalar_features):
            raise ValueError("shared scalar input has the wrong shape")
        if thomson_values.ndim != 4 or thomson_values.shape[:2] != (batch, time):
            raise ValueError("Thomson values must have shape (batch, time, token, feature)")
        if thomson_values.shape[-1] != self.config.thomson_features:
            raise ValueError("Thomson feature count differs from the model contract")
        if thomson_attention_mask.shape != thomson_values.shape[:3]:
            raise ValueError("Thomson attention mask has the wrong shape")

        flat_spatial = spatial_values.reshape(batch * time, channels, height, width)
        current_features, spatial_summary, flux_features = self.spatial_encoder(
            flat_spatial
        )
        if current_features.shape[-2:] != (
            self.config.current_grid_size,
            self.config.current_grid_size,
        ):
            current_features = F.interpolate(
                current_features,
                size=(self.config.current_grid_size, self.config.current_grid_size),
                mode="bilinear",
                align_corners=False,
            )
        token_count = thomson_values.shape[2]
        thomson_summary = self.thomson_encoder(
            thomson_values.reshape(
                batch * time, token_count, self.config.thomson_features
            ),
            thomson_attention_mask.reshape(batch * time, token_count),
        )
        scalar_summary = self.scalar_encoder(
            shared_scalars.reshape(batch * time, self.config.scalar_features)
        )
        fused = self.fusion(
            torch.cat([spatial_summary, thomson_summary, scalar_summary], dim=-1)
        ).reshape(batch, time, self.config.fusion_width)
        hidden_states, final_hidden = self.temporal(
            fused,
            delta_time_ms,
            initial_state=initial_hidden_state,
            sequence_mask=sequence_mask,
        )

        flat_hidden = hidden_states.reshape(batch * time, self.config.hidden_size)
        scale, shift = self.film(flat_hidden).chunk(2, dim=-1)
        bound = self.config.film_bound
        modulated = current_features * (
            1.0 + bound * torch.tanh(scale)[:, :, None, None]
        ) + bound * torch.tanh(shift)[:, :, None, None]
        logits = self.current_head(modulated).squeeze(1).reshape(
            batch,
            time,
            self.config.current_grid_size,
            self.config.current_grid_size,
        )
        if current_support_mask is not None:
            if current_support_mask.shape != logits.shape:
                raise ValueError("current support mask has the wrong shape")
            support = current_support_mask.bool()
            if not support.flatten(2).any(dim=-1).all():
                raise ValueError("every frame needs at least one current-support pixel")
            logits = logits.masked_fill(~support, -torch.inf)
        current_mass = torch.softmax(logits.flatten(2), dim=-1).reshape_as(logits)

        flux_scale, flux_shift = self.flux_film(flat_hidden).chunk(2, dim=-1)
        flux_modulated = flux_features * (
            1.0 + bound * torch.tanh(flux_scale)[:, :, None, None]
        ) + bound * torch.tanh(flux_shift)[:, :, None, None]
        normalized_flux_residual = self.config.flux_residual_bound * torch.tanh(
            self.flux_head(flux_modulated).squeeze(1)
        )
        normalized_flux_residual = normalized_flux_residual.reshape(
            batch, time, height, width
        )

        harmonics = self.config.harmonic_bound * torch.tanh(
            self.harmonic_head(flat_hidden)
        ).reshape(batch, time, self.config.harmonic_count)
        gauge = self.config.gauge_bound * torch.tanh(
            self.gauge_head(flat_hidden).squeeze(-1)
        ).reshape(batch, time)
        raw_scalars = self.scalar_head(flat_hidden)
        positive_scalars = F.softplus(raw_scalars) + 1.0e-4
        positive_scalars = positive_scalars.reshape(batch, time, 2)
        scalar_correction = torch.stack(
            (
                4.0 * torch.tanh(raw_scalars[:, 0]),
                2.0 * torch.tanh(raw_scalars[:, 1]),
            ),
            dim=-1,
        ).reshape(batch, time, 2)
        return CausalHybridEncoderOutput(
            normalized_flux_residual=normalized_flux_residual,
            current_mass=current_mass,
            current_logits=logits,
            harmonics=harmonics,
            gauge=gauge,
            q95=positive_scalars[..., 0],
            betaN=positive_scalars[..., 1],
            scalar_correction=scalar_correction,
            hidden_states=hidden_states,
            final_hidden_state=final_hidden,
            tau_ms=self.temporal.tau_ms,
        )


# Explicit names for new code.  The aliases keep old checkpoint tooling able to
# locate the class while making the DIII-D-only intent visible to callers.
D3DChallenge1EncoderConfig = CausalHybridEncoderConfig
D3DChallenge1Encoder = CausalHybridEncoderV1
