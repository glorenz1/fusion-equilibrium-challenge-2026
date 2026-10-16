"""Target-free DCT and convolutional correction decoders for FEC1."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from fec1.basis import dct_residual_basis


def within_shot_neighbors(names, frames, half_window=4):
    """Nine actual neighboring observations, padded without crossing shots."""
    names, frames = np.asarray(names), np.asarray(frames)
    offsets = np.arange(-half_window, half_window + 1)
    neighbors = np.empty((len(names), len(offsets)), dtype=np.int64)
    valid = np.empty_like(neighbors, dtype=bool)
    for shot in np.unique(names):
        rows = np.flatnonzero(names == shot)
        rows = rows[np.argsort(frames[rows], kind="stable")]
        if len(np.unique(frames[rows])) != len(rows):
            raise ValueError("duplicate within-shot frame identifiers")
        local = np.arange(len(rows))[:, None] + offsets[None]
        valid[rows] = (local >= 0) & (local < len(rows))
        neighbors[rows] = rows[local.clip(0, len(rows) - 1)]
    return neighbors, valid


def calibrated_prediction(base_field, base_scalar, field, scalar, strengths):
    """Apply already-frozen strengths using predicted quantities only."""
    from fec1.data.features import _coordinates

    base_mean, base_rms, base_shape = _coordinates(base_field)
    mean, rms, shape = _coordinates(field)
    coordinate = np.column_stack((base_mean, np.log(base_rms)))
    delta = np.column_stack((mean - base_mean, np.log(rms / base_rms)))
    coordinate += np.asarray(strengths["coordinate"], np.float32) * delta
    shape = base_shape + float(strengths["shape"]) * (shape - base_shape)
    shape = _coordinates(shape)[2]
    corrected = coordinate[:, 0, None, None] + np.exp(coordinate[:, 1, None, None]) * shape
    corrected_scalar = base_scalar + np.asarray(strengths["scalar"], np.float32) * (
        scalar - base_scalar
    )
    return corrected.astype(np.float32), corrected_scalar.astype(np.float32)


def calibrated_prediction_torch(base_field, base_scalar, field, scalar, strengths):
    """Differentiable counterpart of the frozen public export calibration."""
    base_coordinate, base_shape = coordinates(base_field)
    coordinate, shape = coordinates(field)
    coordinate_strength = field.new_tensor(strengths["coordinate"])
    scalar_strength = scalar.new_tensor(strengths["scalar"])
    coordinate = base_coordinate + coordinate_strength * (coordinate - base_coordinate)
    shape = base_shape + float(strengths["shape"]) * (shape - base_shape)
    _, shape = coordinates(shape)
    corrected = coordinate[:, 0, None, None] + coordinate[:, 1, None, None].exp() * shape
    corrected_scalar = base_scalar + scalar_strength * (scalar - base_scalar)
    return corrected, corrected_scalar


def coordinates(field):
    work = field.float()
    mean = work.mean((-2, -1))
    centered = work - mean[:, None, None]
    rms = centered.square().mean((-2, -1)).clamp_min(1e-14).sqrt()
    return torch.stack((mean, rms.log()), -1), centered / rms[:, None, None]


def mlp(input_width, width, output_width, zero=False):
    result = nn.Sequential(
        nn.Linear(input_width, width),
        nn.LayerNorm(width),
        nn.SiLU(),
        nn.Linear(width, width),
        nn.SiLU(),
        nn.Linear(width, output_width),
    )
    if zero:
        nn.init.zeros_(result[-1].weight)
        nn.init.zeros_(result[-1].bias)
    return result


class StructuredCorrection(nn.Module):
    """Inputs contain observations and base predictions; no targets are accepted."""

    def __init__(
        self,
        features,
        decoder="dct",
        order=24,
        width=384,
        channels=24,
        observation_encoder="pooled",
    ):
        super().__init__()
        if decoder not in ("dct", "conv") or not 2 <= order <= 65:
            raise ValueError("invalid decoder or DCT order")
        if observation_encoder not in ("pooled", "spatial_attention", "temporal_attention"):
            raise ValueError("invalid observation encoder")
        self.config = dict(
            features=features,
            decoder=decoder,
            order=order,
            width=width,
            channels=channels,
            observation_encoder=observation_encoder,
        )
        self.register_buffer("feature_mean", torch.zeros(features))
        self.register_buffer("feature_scale", torch.ones(features))
        self.register_buffer("token_mean", torch.zeros(7))
        self.register_buffer("token_scale", torch.ones(7))
        self.register_buffer("base_mean", torch.zeros(4))
        self.register_buffer("base_scale", torch.ones(4))
        self.register_buffer("coordinate_scale", torch.ones(2) * 0.01)
        self.register_buffer("scalar_scale", torch.ones(2) * 0.1)
        self.register_buffer("shape_scale", torch.tensor(0.02))
        self.register_buffer("coefficient_scale", torch.ones(order * order - 1) * 0.01)
        self.register_buffer(
            "basis", dct_residual_basis(order, 65, torch.device("cpu"), torch.float32).flatten(1)
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(7, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        # Spatial pooling uses only the predicted base shape. The same features
        # and token encoder are available to both decoder families.
        input_width = features + 81 + 4 + 128
        if observation_encoder == "spatial_attention":
            self.spatial_queries = nn.Linear(features + 81 + 4, 128)
            self.spatial_attention = nn.MultiheadAttention(64, 4, batch_first=True)
        if observation_encoder == "temporal_attention":
            if features % 3:
                raise ValueError("temporal encoder requires feature/difference/mean context")
            self.temporal_projection = nn.Linear(features // 3, 128)
            self.temporal_positions = nn.Parameter(torch.randn(1, 9, 128) * 0.01)
            self.temporal_query = nn.Linear(input_width, 128)
            self.temporal_attention = nn.MultiheadAttention(128, 4, batch_first=True)
            input_width += 128
        self.field_encoder = mlp(input_width, width, width)
        self.coordinate_head = mlp(input_width, width // 2, 2, zero=True)
        self.scalar_head = mlp(input_width, width // 2, 2, zero=True)
        if decoder == "dct":
            self.field_head = nn.Linear(width, order * order - 1)
            nn.init.zeros_(self.field_head.weight)
            nn.init.zeros_(self.field_head.bias)
        else:
            self.condition = nn.Linear(width, channels)
            self.spatial_in = nn.Conv2d(3, channels, 3, padding=1)
            self.down = nn.Sequential(
                nn.Conv2d(channels, 2 * channels, 3, 2, 1),
                nn.GroupNorm(8, 2 * channels),
                nn.SiLU(),
                nn.Conv2d(2 * channels, 2 * channels, 3, padding=1),
                nn.SiLU(),
            )
            self.up = nn.Sequential(
                nn.Conv2d(3 * channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.SiLU(),
            )
            self.field_head = nn.Conv2d(channels, 1, 3, padding=1)
            nn.init.zeros_(self.field_head.weight)
            nn.init.zeros_(self.field_head.bias)
            z, r = torch.meshgrid(
                torch.linspace(-1, 1, 65), torch.linspace(-1, 1, 65), indexing="ij"
            )
            self.register_buffer("grid", torch.stack((r, z))[None])

    def forward(
        self,
        features,
        base_field,
        base_scalar,
        token_values,
        token_coords,
        token_mask,
        sequence_features=None,
        sequence_mask=None,
        *,
        scalar_only=False,
    ):
        base_coordinate, base_shape = coordinates(base_field)
        token = torch.cat((token_values, token_coords), -1).float()
        token = torch.nan_to_num(token, nan=0.0, posinf=0.0, neginf=0.0)
        token = ((token - self.token_mean) / self.token_scale).clamp(-20, 20)
        encoded = self.token_encoder(token)
        mask = token_mask.bool()
        masked = encoded * mask[..., None]
        mean = masked.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        maximum = encoded.masked_fill(~mask[..., None], -1e4).max(1).values
        maximum = torch.where(mask.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        base_features = torch.cat((base_coordinate, base_scalar), -1)
        context = torch.cat(
            (
                ((features - self.feature_mean) / self.feature_scale).clamp(-20, 20),
                F.adaptive_avg_pool2d(base_shape[:, None], 9).flatten(1),
                ((base_features - self.base_mean) / self.base_scale).clamp(-20, 20),
            ),
            -1,
        )
        pooled = torch.cat((mean, maximum), -1)
        if self.config["observation_encoder"] == "spatial_attention":
            safe_mask = mask.clone()
            safe_mask[~mask.any(1), 0] = True
            values = encoded * mask[..., None]
            query = self.spatial_queries(context).reshape(len(features), 2, 64)
            pooled = self.spatial_attention(
                query, values, values, key_padding_mask=~safe_mask, need_weights=False
            )[0].flatten(1)
            pooled = pooled * mask.any(1, keepdim=True)
        x = torch.cat((context, pooled), -1)
        if self.config["observation_encoder"] == "temporal_attention":
            if (
                sequence_features is None
                or sequence_mask is None
                or sequence_features.shape[1] != 9
            ):
                raise ValueError("temporal encoder needs nine masked within-shot observations")
            count = features.shape[1] // 3
            sequence = (
                (sequence_features - self.feature_mean[:count]) / self.feature_scale[:count]
            ).clamp(-20, 20)
            sequence = self.temporal_projection(sequence) + self.temporal_positions
            if not bool(sequence_mask.any(1).all()):
                raise ValueError("every sequence must include its central observation")
            attended = self.temporal_attention(
                self.temporal_query(x)[:, None],
                sequence,
                sequence,
                key_padding_mask=~sequence_mask.bool(),
                need_weights=False,
            )[0][:, 0]
            x = torch.cat((x, attended), -1)
        scalar = base_scalar + self.scalar_head(x) * self.scalar_scale
        if scalar_only:
            return scalar
        encoded_field = self.field_encoder(x)
        if self.config["decoder"] == "dct":
            coefficient = self.field_head(encoded_field) * self.coefficient_scale
            residual = (coefficient @ self.basis).reshape_as(base_shape)
        else:
            grid = self.grid.expand(len(x), -1, -1, -1)
            local = self.spatial_in(torch.cat((base_shape[:, None], grid), 1))
            local = F.silu(local + self.condition(encoded_field)[:, :, None, None])
            coarse = F.interpolate(self.down(local), (65, 65), mode="bilinear", align_corners=True)
            residual = (
                self.field_head(self.up(torch.cat((local, coarse), 1)))[:, 0] * self.shape_scale
            )
            coefficient = residual.flatten(1) @ self.basis.T
        _, shape = coordinates(base_shape + residual)
        delta_coordinate = self.coordinate_head(x) * self.coordinate_scale
        delta_coordinate = torch.stack(
            (delta_coordinate[:, 0], delta_coordinate[:, 1].clamp(-1, 1)), -1
        )
        coordinate = base_coordinate + delta_coordinate
        field = coordinate[:, 0, None, None] + coordinate[:, 1, None, None].exp() * shape
        return {
            "field": field,
            "shape": shape,
            "coordinate": coordinate,
            "scalar": scalar,
            "coefficient": coefficient,
            "base_shape": base_shape,
            "base_coordinate": base_coordinate,
        }


class FrozenScalarPredictor(nn.Module):
    """Embed a fixed scalar path, including its observation encoder and strengths."""

    def __init__(self, checkpoint):
        super().__init__()
        if (
            not checkpoint.get("nested_shot_crossfit")
            or checkpoint.get("development_used")
            or checkpoint.get("frozen_audit_used")
        ):
            raise ValueError("scalar reference must come from nested training-shot crossfit")
        self.model = StructuredCorrection(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.register_buffer("strength", torch.tensor(checkpoint["correction_strengths"]["scalar"]))
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, *inputs):
        scalar = self.model(*inputs[:6], scalar_only=True)
        return inputs[2] + self.strength * (scalar - inputs[2])
