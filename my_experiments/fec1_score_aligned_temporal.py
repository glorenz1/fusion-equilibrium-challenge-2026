"""Temporal, score-aligned components for the train-only FEC1 residual stack.

The field model predicts a conventional PCA residual, but its magnetic-axis,
LCFS Fourier, exact-topology, and smooth-profile outputs are fed back through a
geometry adapter before the raster is decoded.  Geometry supervision therefore
cannot improve a detached diagnostic head without also affecting submitted
flux.  q95 and betaN use independent temporal models and checkpoints.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from my_experiments.fec1_direct_geometry import direct_raster_geometry_losses


CONSISTENCY_NAMES = (
    "R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li",
)


def _sequence_context(
    embedded: torch.Tensor, recurrent: nn.GRU, lengths: torch.Tensor,
) -> torch.Tensor:
    packed = pack_padded_sequence(
        embedded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False,
    )
    encoded, _ = recurrent(packed)
    encoded, _ = pad_packed_sequence(
        encoded, batch_first=True, total_length=embedded.shape[1],
    )
    return encoded


class ScoreAlignedTemporalResidual(nn.Module):
    """Whole-shot residual model with geometry-conditioned raster decoding.

    ``separate_decoder_branches`` keeps boundary and interior-profile
    corrections out of the ordinary field-regression path until the final
    PCA/DCT coefficients are assembled.  This makes each score-sensitive head
    causally responsible for changing the submitted raster while avoiding the
    old heuristic post-processing warp.
    """

    def __init__(
        self,
        feature_width: int,
        shape_width: int,
        hidden_width: int = 384,
        layers: int = 2,
        dropout: float = 0.05,
        fourier_modes: int = 8,
        profile_modes: int = 6,
        dct_order: int = 6,
        separate_decoder_branches: bool = False,
    ) -> None:
        super().__init__()
        if hidden_width < 32 or shape_width < 1 or fourier_modes < 1:
            raise ValueError("invalid temporal residual dimensions")
        self.feature_width = int(feature_width)
        self.shape_width = int(shape_width)
        self.hidden_width = int(hidden_width)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.fourier_modes = int(fourier_modes)
        self.profile_modes = int(profile_modes)
        self.dct_order = int(dct_order)
        self.separate_decoder_branches = bool(separate_decoder_branches)
        self.spatial_width = self.dct_order * self.dct_order - 1
        self.radius_width = 1 + 2 * self.fourier_modes

        self.input = nn.Sequential(
            nn.Linear(feature_width, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        recurrent_width = hidden_width // 2
        self.recurrent = nn.GRU(
            hidden_width,
            recurrent_width,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(2 * hidden_width),
            nn.Linear(2 * hidden_width, hidden_width),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.axis_head = nn.Linear(hidden_width, 2)
        self.radius_head = nn.Linear(hidden_width, self.radius_width)
        self.topology_head = nn.Linear(hidden_width, len(CONSISTENCY_NAMES))
        self.profile_head = nn.Linear(hidden_width, profile_modes)
        self.profile_li_head = nn.Linear(profile_modes, 1)
        self.direct_shape_head = nn.Linear(hidden_width, shape_width)
        self.direct_spatial_head = nn.Linear(hidden_width, self.spatial_width)
        self.coordinate_head = nn.Linear(hidden_width, 2)

        def adapter(input_width: int, output_width: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_width),
                nn.Linear(input_width, hidden_width // 2),
                nn.SiLU(),
                nn.Linear(hidden_width // 2, output_width, bias=False),
            )

        if self.separate_decoder_branches:
            # The first six consistency quantities describe the axis and LCFS
            # geometry. li and the smooth radial profile form a distinct
            # interior branch because they need different spatial modes.
            boundary_width = 2 + self.radius_width + 6
            profile_width = profile_modes + 2
            self.boundary_shape_adapter = adapter(boundary_width, shape_width)
            self.boundary_spatial_adapter = adapter(
                boundary_width, self.spatial_width
            )
            self.profile_shape_adapter = adapter(profile_width, shape_width)
            self.profile_spatial_adapter = adapter(
                profile_width, self.spatial_width
            )
            adapters = (
                self.boundary_shape_adapter,
                self.boundary_spatial_adapter,
                self.profile_shape_adapter,
                self.profile_spatial_adapter,
            )
        else:
            geometry_width = (
                2 + self.radius_width + len(CONSISTENCY_NAMES) + profile_modes
            )
            self.geometry_adapter = adapter(geometry_width, shape_width)
            self.spatial_adapter = adapter(geometry_width, self.spatial_width)
            adapters = (self.geometry_adapter, self.spatial_adapter)

        # Start close to the proven per-frame residual. Geometry influence is
        # learned from score-aligned losses instead of perturbing it at init.
        for branch_adapter in adapters:
            nn.init.zeros_(branch_adapter[-1].weight)
        nn.init.zeros_(self.direct_spatial_head.weight)
        nn.init.zeros_(self.direct_spatial_head.bias)

    def forward(
        self, values: torch.Tensor, lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embedded = self.input(values)
        encoded = _sequence_context(embedded, self.recurrent, lengths)
        hidden = self.trunk(torch.cat((embedded, encoded), dim=-1))
        axis = torch.sigmoid(self.axis_head(hidden))
        raw_radius = self.radius_head(hidden)
        radius = torch.cat(
            (
                0.02 + 0.78 * torch.sigmoid(raw_radius[..., :1]),
                0.30 * torch.tanh(raw_radius[..., 1:]),
            ),
            dim=-1,
        )
        topology = self.topology_head(hidden)
        profile = torch.tanh(self.profile_head(hidden))
        profile_li = self.profile_li_head(profile).squeeze(-1)
        shape = self.direct_shape_head(hidden)
        spatial_logits = self.direct_spatial_head(hidden)
        if self.separate_decoder_branches:
            boundary = torch.cat((axis, radius, topology[..., :6]), dim=-1)
            interior = torch.cat(
                (profile, topology[..., 6:], profile_li[..., None]), dim=-1,
            )
            shape = (
                shape
                + self.boundary_shape_adapter(boundary)
                + self.profile_shape_adapter(interior)
            )
            spatial_logits = (
                spatial_logits
                + self.boundary_spatial_adapter(boundary)
                + self.profile_spatial_adapter(interior)
            )
        else:
            geometry = torch.cat((axis, radius, topology, profile), dim=-1)
            shape = shape + self.geometry_adapter(geometry)
            spatial_logits = spatial_logits + self.spatial_adapter(geometry)
        spatial = 0.08 * torch.tanh(spatial_logits)
        return {
            "shape": shape,
            "spatial": spatial,
            "coordinate": self.coordinate_head(hidden),
            "axis": axis,
            "radius_fourier": radius,
            "topology": topology,
            "profile": profile,
            "profile_li": profile_li,
        }


class TemporalScalarResidual(nn.Module):
    """Independent whole-shot residual model for one submitted scalar."""

    def __init__(
        self,
        feature_width: int,
        hidden_width: int = 192,
        layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.feature_width = int(feature_width)
        self.hidden_width = int(hidden_width)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.input = nn.Sequential(
            nn.Linear(feature_width, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.recurrent = nn.GRU(
            hidden_width,
            hidden_width // 2,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(2 * hidden_width),
            nn.Linear(2 * hidden_width, hidden_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.input(values)
        encoded = _sequence_context(embedded, self.recurrent, lengths)
        return self.output(torch.cat((embedded, encoded), dim=-1)).squeeze(-1)


def valid_time_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    return torch.arange(width, device=lengths.device)[None] < lengths[:, None]


def decode_pca_field(
    coefficient: torch.Tensor,
    coordinate: torch.Tensor,
    components: torch.Tensor,
    pca_mean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode PCA shape and physical flux with differentiable normalization."""
    flat = coefficient @ components + pca_mean
    shape = flat.reshape(*coefficient.shape[:-1], 65, 65)
    centered = shape - shape.mean(dim=(-2, -1), keepdim=True)
    rms = centered.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1.0e-7)
    normalized = centered / rms
    field = (
        coordinate[..., 0, None, None]
        + coordinate[..., 1, None, None].exp() * normalized
    )
    return field, normalized


def dct_residual_basis(
    order: int, size: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Orthonormal low-frequency 2-D DCT basis without the constant mode."""
    coordinate = torch.arange(size, device=device, dtype=dtype) + 0.5
    one_d = []
    for frequency in range(order):
        value = torch.cos(math.pi * frequency * coordinate / size)
        value = value / value.square().sum().sqrt().clamp_min(1.0e-8)
        one_d.append(value)
    modes = []
    for kz in range(order):
        for kr in range(order):
            if kz == 0 and kr == 0:
                continue
            modes.append(one_d[kz][:, None] * one_d[kr][None, :])
    return torch.stack(modes)


def add_spatial_residual(
    field: torch.Tensor,
    coordinate: torch.Tensor,
    coefficient: torch.Tensor,
    order: int,
) -> torch.Tensor:
    """Add a smooth direct-raster residual in physical flux units."""
    basis = dct_residual_basis(order, 65, field.device, field.dtype)
    residual = (coefficient @ basis.flatten(1)).reshape_as(field)
    return field + coordinate[..., 1, None, None].exp() * residual


def normalize_prediction_from_teacher(
    prediction: torch.Tensor, target_psi_n: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Affine-normalize predicted flux using teacher axis/LCFS bands."""
    finite = torch.isfinite(target_psi_n)
    target = torch.nan_to_num(target_psi_n).clamp(-0.5, 2.0)
    useful = finite & (target_psi_n >= -0.5) & (target_psi_n <= 2.0)
    useful_f = useful.to(prediction.dtype)
    axis_band = torch.exp(-0.5 * (target / 0.075).square()) * useful_f
    lcfs_band = torch.exp(-0.5 * ((target - 1.0) / 0.065).square()) * useful_f

    def level(weights: torch.Tensor) -> torch.Tensor:
        return (prediction * weights).sum(dim=(-2, -1)) / weights.sum(
            dim=(-2, -1)
        ).clamp_min(1.0e-6)

    axis_level = level(axis_band)
    boundary_level = level(lcfs_band)
    raw_span = axis_level - boundary_level
    safe_span = torch.where(
        raw_span >= 0,
        raw_span.clamp_min(0.02),
        raw_span.clamp_max(-0.02),
    )
    predicted_n = (
        axis_level[..., None, None] - prediction
    ) / safe_span[..., None, None]
    return predicted_n, useful


def soft_axis_coordinates(
    psi_n: torch.Tensor,
    useful: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    temperature: float = 0.045,
) -> torch.Tensor:
    """Differentiable magnetic-axis position in physical R/Z coordinates."""
    logits = (-psi_n / temperature).flatten(1).clamp(-80.0, 80.0)
    logits = logits.masked_fill(~useful.flatten(1), -torch.inf)
    probability = torch.softmax(logits, dim=1).reshape_as(psi_n)
    rr, zz = torch.meshgrid(grid_r, grid_z, indexing="xy")
    return torch.column_stack(
        (
            (probability * rr[None]).sum(dim=(-2, -1)),
            (probability * zz[None]).sum(dim=(-2, -1)),
        )
    )


def fourier_design(
    angles: torch.Tensor, modes: int,
) -> torch.Tensor:
    pieces = [torch.ones_like(angles)]
    for mode in range(1, modes + 1):
        pieces.extend((torch.cos(mode * angles), torch.sin(mode * angles)))
    return torch.stack(pieces, dim=-1)


def fourier_radius(
    coefficient: torch.Tensor, angles: torch.Tensor,
) -> torch.Tensor:
    modes = (coefficient.shape[-1] - 1) // 2
    return (coefficient @ fourier_design(angles, modes).T).clamp(0.015, 1.5)


def radial_geometry_surrogates(
    radius: torch.Tensor,
    axis_unit: torch.Tensor,
    angles: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    temperature: float = 0.025,
) -> torch.Tensor:
    """Differentiable kappa/triangularity/volume from a radial LCFS."""
    x = axis_unit[:, :1] + radius * torch.cos(angles)[None]
    y = axis_unit[:, 1:] + radius * torch.sin(angles)[None]

    def extremum(value: torch.Tensor, sign: float) -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch.softmax(sign * value / temperature, dim=1)
        return (weight * x).sum(dim=1), (weight * y).sum(dim=1)

    right_x, _ = extremum(x, 1.0)
    left_x, _ = extremum(x, -1.0)
    top_x, top_y = extremum(y, 1.0)
    bottom_x, bottom_y = extremum(y, -1.0)
    r_span = grid_r[-1] - grid_r[0]
    z_span = grid_z[-1] - grid_z[0]
    major_radius = (
        grid_r[0] + 0.5 * (right_x + left_x) * r_span
    ).clamp_min(1.0e-4)
    minor_radius = (0.5 * (right_x - left_x) * r_span).clamp_min(1.0e-4)
    half_height = (0.5 * (top_y - bottom_y) * z_span).clamp_min(1.0e-4)
    kappa = half_height / minor_radius
    center_x = 0.5 * (right_x + left_x)
    triangularity_top = (center_x - top_x) * r_span / minor_radius
    triangularity_bottom = (center_x - bottom_x) * r_span / minor_radius
    volume = 2.0 * math.pi ** 2 * major_radius * minor_radius * half_height
    return torch.column_stack(
        (kappa, triangularity_top, triangularity_bottom, volume)
    )


def soft_radial_signature(
    target_psi_n: torch.Tensor,
    axis_unit: torch.Tensor,
    angles: int = 32,
    radii: int = 64,
    temperature: float = 0.035,
) -> torch.Tensor:
    """Extract a differentiable first LCFS crossing on a unit-square canvas."""
    if target_psi_n.ndim != 3 or axis_unit.shape != (len(target_psi_n), 2):
        raise ValueError("radial signature inputs have incompatible shapes")
    device, dtype = target_psi_n.device, target_psi_n.dtype
    theta = torch.linspace(0.0, 2.0 * math.pi, angles + 1, device=device, dtype=dtype)[:-1]
    radius = torch.linspace(0.01, 1.45, radii, device=device, dtype=dtype)
    direction = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    point = (
        axis_unit[:, None, None, :]
        + radius[None, None, :, None] * direction[None, :, None, :]
    )
    valid = (point >= 0.0).all(dim=-1) & (point <= 1.0).all(dim=-1)
    grid = 2.0 * point - 1.0
    sampled = torch.nn.functional.grid_sample(
        target_psi_n[:, None], grid, mode="bilinear",
        padding_mode="border", align_corners=True,
    )[:, 0]
    score = -torch.abs(sampled - 1.0) / temperature - 0.015 * radius[None, None]
    score = score.masked_fill(~valid, -80.0)
    weight = torch.softmax(score, dim=-1)
    return (weight * radius[None, None]).sum(dim=-1)


def gradient_curvature_losses(
    prediction_n: torch.Tensor,
    target_n: torch.Tensor,
    useful: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First- and second-difference losses on normalized flux."""
    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if bool(mask.any()):
            return value[mask].mean()
        return value.sum() * 0.0

    target = torch.nan_to_num(target_n).clamp(-0.5, 2.0)
    pred_dr = prediction_n[..., :, 1:] - prediction_n[..., :, :-1]
    true_dr = target[..., :, 1:] - target[..., :, :-1]
    pred_dz = prediction_n[..., 1:, :] - prediction_n[..., :-1, :]
    true_dz = target[..., 1:, :] - target[..., :-1, :]
    mask_dr = useful[..., :, 1:] & useful[..., :, :-1]
    mask_dz = useful[..., 1:, :] & useful[..., :-1, :]
    gradient = (
        masked_mean((pred_dr - true_dr).square(), mask_dr)
        + masked_mean((pred_dz - true_dz).square(), mask_dz)
    )
    pred_drr = pred_dr[..., :, 1:] - pred_dr[..., :, :-1]
    true_drr = true_dr[..., :, 1:] - true_dr[..., :, :-1]
    pred_dzz = pred_dz[..., 1:, :] - pred_dz[..., :-1, :]
    true_dzz = true_dz[..., 1:, :] - true_dz[..., :-1, :]
    mask_drr = mask_dr[..., :, 1:] & mask_dr[..., :, :-1]
    mask_dzz = mask_dz[..., 1:, :] & mask_dz[..., :-1, :]
    curvature = (
        masked_mean((pred_drr - true_drr).square(), mask_drr)
        + masked_mean((pred_dzz - true_dzz).square(), mask_dzz)
    )
    return gradient, curvature


def interior_profile_loss(
    prediction_n: torch.Tensor,
    target_n: torch.Tensor,
    useful: torch.Tensor,
    li_target: torch.Tensor,
    li_reference: torch.Tensor,
) -> torch.Tensor:
    """Match li-sensitive normalized-flux gradient energy inside the LCFS."""
    target = torch.nan_to_num(target_n).clamp(-0.5, 2.0)
    pred_dr = prediction_n[..., :, 1:] - prediction_n[..., :, :-1]
    true_dr = target[..., :, 1:] - target[..., :, :-1]
    pred_dz = prediction_n[..., 1:, :] - prediction_n[..., :-1, :]
    true_dz = target[..., 1:, :] - target[..., :-1, :]
    mask_dr = (
        useful[..., :, 1:] & useful[..., :, :-1]
        & (target[..., :, 1:] > 0.04) & (target[..., :, 1:] < 0.96)
        & (target[..., :, :-1] > 0.04) & (target[..., :, :-1] < 0.96)
    )
    mask_dz = (
        useful[..., 1:, :] & useful[..., :-1, :]
        & (target[..., 1:, :] > 0.04) & (target[..., 1:, :] < 0.96)
        & (target[..., :-1, :] > 0.04) & (target[..., :-1, :] < 0.96)
    )
    pred_energy = (
        (pred_dr.square() * mask_dr).sum(dim=(-2, -1))
        + (pred_dz.square() * mask_dz).sum(dim=(-2, -1))
    )
    true_energy = (
        (true_dr.square() * mask_dr).sum(dim=(-2, -1))
        + (true_dz.square() * mask_dz).sum(dim=(-2, -1))
    )
    count = (
        mask_dr.sum(dim=(-2, -1)) + mask_dz.sum(dim=(-2, -1))
    ).clamp_min(1)
    pred_energy = (pred_energy / count).clamp_min(1.0e-8)
    true_energy = (true_energy / count).clamp_min(1.0e-8)
    error = torch.nn.functional.smooth_l1_loss(
        pred_energy.log(), true_energy.log(), beta=0.15, reduction="none"
    )
    safe_li = torch.where(torch.isfinite(li_target), li_target, li_reference)
    li_weight = (safe_li / li_reference.clamp_min(1.0e-4)).clamp(0.5, 2.0)
    valid = (count > 1) & torch.isfinite(error)
    if bool(valid.any()):
        return (error[valid] * li_weight[valid]).mean()
    return error.sum() * 0.0


def score_aligned_field_loss(
    prediction: Mapping[str, torch.Tensor],
    predicted_field: torch.Tensor,
    target_field: torch.Tensor,
    target_psi_n: torch.Tensor,
    target_topology: torch.Tensor,
    topology_mean: torch.Tensor,
    topology_scale: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    target_shape_residual_normalized: torch.Tensor | None = None,
    target_coordinate_residual_normalized: torch.Tensor | None = None,
    exact_target_radius: torch.Tensor | None = None,
    contour_angles: torch.Tensor | None = None,
    exact_target_contour_offset: torch.Tensor | None = None,
    objective_weights: Mapping[str, float] | None = None,
    consistency_channel_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Metric-aligned differentiable proxy used for training and selection."""
    predicted_n, useful = normalize_prediction_from_teacher(
        predicted_field, target_psi_n
    )
    target_n = torch.nan_to_num(target_psi_n).clamp(-0.5, 2.0)
    useful_f = useful.to(predicted_field.dtype)
    axis_band = torch.exp(-0.5 * (target_n / 0.075).square()) * useful_f
    lcfs_band = torch.exp(-0.5 * ((target_n - 1.0) / 0.065).square()) * useful_f
    narrow_weight = useful_f * (0.20 + 18.0 * axis_band + 8.0 * lcfs_band)
    surface_error = torch.nn.functional.smooth_l1_loss(
        predicted_n, target_n, beta=0.04, reduction="none"
    )
    surface = (surface_error * narrow_weight).sum() / narrow_weight.sum().clamp_min(1.0)

    field_scale = target_field.flatten(1).std(dim=1).clamp_min(1.0e-6)
    flux = (
        (predicted_field - target_field).square().mean(dim=(-2, -1))
        / field_scale.square()
    ).mean()
    gradient, curvature = gradient_curvature_losses(
        predicted_n, target_n, useful
    )
    axis_physical = soft_axis_coordinates(
        predicted_n, useful, grid_r, grid_z
    )
    topology_valid = torch.isfinite(target_topology)
    safe_topology = torch.where(
        topology_valid, target_topology, topology_mean[None]
    )
    axis_target = safe_topology[:, :2]
    axis_scale = torch.stack((grid_r[-1] - grid_r[0], grid_z[-1] - grid_z[0]))
    axis = ((axis_physical - axis_target) / axis_scale[None]).square().mean()

    predicted_topology = topology_mean + topology_scale * prediction["topology"]
    topology_error = ((predicted_topology - safe_topology) / topology_scale).square()
    if consistency_channel_weights is None:
        channel_weights = torch.ones(
            topology_error.shape[1], device=topology_error.device,
            dtype=topology_error.dtype,
        )
    else:
        channel_weights = consistency_channel_weights.to(topology_error).flatten()
        if channel_weights.shape != (len(CONSISTENCY_NAMES),):
            raise ValueError("consistency channel weights must have seven entries")
        if not bool(torch.isfinite(channel_weights).all()) or bool((channel_weights < 0).any()):
            raise ValueError("consistency channel weights must be finite and nonnegative")
    topology_weights = topology_valid.to(topology_error) * channel_weights[None]
    topology = (
        (topology_error * topology_weights).sum()
        / topology_weights.sum().clamp_min(1.0)
    )
    li_valid = topology_valid[:, -1]
    profile_li_physical = (
        topology_mean[-1] + topology_scale[-1] * prediction["profile_li"]
    )
    profile_li_error = (
        (profile_li_physical - safe_topology[:, -1]) / topology_scale[-1]
    ).square()
    profile_li = (
        profile_li_error[li_valid].mean()
        if bool(li_valid.any())
        else profile_li_error.sum() * 0.0
    )
    interior_profile = interior_profile_loss(
        predicted_n, target_n, useful, target_topology[:, -1], topology_mean[-1]
    )
    # Exact contour tracing and li integration are the expensive terms.  A
    # rotating shot shuffle makes a bounded, evenly spaced frame subset an
    # unbiased-enough training estimate while keeping GPU memory predictable.
    direct_step = max(1, math.ceil(len(predicted_n) / 96))
    direct_rows = torch.arange(
        0, len(predicted_n), direct_step, device=predicted_n.device,
    )[:96]
    direct_geometry = direct_raster_geometry_losses(
        predicted_n[direct_rows],
        target_n[direct_rows],
        useful[direct_rows],
        target_topology[direct_rows],
        grid_r,
        grid_z,
        exact_target_radius=(
            exact_target_radius[direct_rows]
            if exact_target_radius is not None else None
        ),
        contour_angles=contour_angles,
        exact_target_contour_offset=(
            exact_target_contour_offset[direct_rows]
            if exact_target_contour_offset is not None else None
        ),
    )

    axis_unit_target = torch.column_stack(
        (
            (axis_target[:, 0] - grid_r[0]) / (grid_r[-1] - grid_r[0]),
            (axis_target[:, 1] - grid_z[0]) / (grid_z[-1] - grid_z[0]),
        )
    ).clamp(0.0, 1.0)
    target_radius = soft_radial_signature(target_n, axis_unit_target)
    angles = torch.linspace(
        0.0, 2.0 * math.pi, target_radius.shape[1] + 1,
        device=target_radius.device, dtype=target_radius.dtype,
    )[:-1]
    predicted_radius = fourier_radius(prediction["radius_fourier"], angles)
    radius = ((predicted_radius - target_radius) / 0.20).square().mean()
    axis_head = ((prediction["axis"] - axis_unit_target) / 0.15).square().mean()
    raster_radius = soft_radial_signature(predicted_n, axis_unit_target)
    raster_radius_loss = ((raster_radius - target_radius) / 0.20).square().mean()
    radius_coupling = ((predicted_radius - raster_radius) / 0.20).square().mean()
    target_geometry = radial_geometry_surrogates(
        target_radius, axis_unit_target, angles, grid_r, grid_z
    )
    raster_geometry = radial_geometry_surrogates(
        raster_radius, axis_unit_target, angles, grid_r, grid_z
    )
    geometry_scale = torch.column_stack((
        torch.full_like(target_geometry[:, 0], 0.20),
        torch.full_like(target_geometry[:, 1], 0.15),
        torch.full_like(target_geometry[:, 2], 0.15),
        (0.15 * target_geometry[:, 3].abs()).clamp_min(0.25),
    ))
    raster_geometry_loss = torch.nn.functional.smooth_l1_loss(
        raster_geometry / geometry_scale,
        target_geometry / geometry_scale,
        beta=0.15,
    )

    coefficient = predicted_field.new_zeros(())
    if target_shape_residual_normalized is not None:
        coefficient = coefficient + torch.nn.functional.smooth_l1_loss(
            prediction["shape"], target_shape_residual_normalized, beta=0.20
        )
    if target_coordinate_residual_normalized is not None:
        coefficient = coefficient + 0.5 * torch.nn.functional.smooth_l1_loss(
            prediction["coordinate"],
            target_coordinate_residual_normalized,
            beta=0.20,
        )

    field_objective = (
        0.30 * flux
        + 0.20 * gradient
        + 0.04 * curvature
        + 0.20 * coefficient
    )
    boundary_objective = (
        1.00 * surface
        + 0.30 * radius
        + 0.35 * raster_radius_loss
        + 0.10 * radius_coupling
        + 0.35 * direct_geometry["direct_lcfs"]
        + 0.40 * direct_geometry["direct_exact_lcfs"]
    )
    consistency_objective = (
        0.75 * axis
        + 0.10 * axis_head
        + 0.25 * raster_geometry_loss
        + 0.35 * topology
        + 0.75 * direct_geometry["direct_axis"]
        + 0.30 * direct_geometry["direct_geometry"]
    )
    profile_objective = (
        0.20 * profile_li
        + 0.20 * interior_profile
        + 0.35 * direct_geometry["direct_li"]
    )
    weights = {
        "field": 1.0,
        "boundary": 1.0,
        "consistency": 1.0,
        "profile": 1.0,
    }
    if objective_weights is not None:
        unknown = set(objective_weights) - set(weights)
        if unknown:
            raise ValueError(f"unknown field objective weights: {sorted(unknown)}")
        weights.update({key: float(value) for key, value in objective_weights.items()})
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("field objective weights must be finite and nonnegative")

    parts = {
        "surface": surface,
        "flux": flux,
        "gradient": gradient,
        "curvature": curvature,
        "axis": axis,
        "axis_head": axis_head,
        "radius": radius,
        "raster_radius": raster_radius_loss,
        "radius_coupling": radius_coupling,
        "raster_geometry": raster_geometry_loss,
        "topology": topology,
        "profile_li": profile_li,
        "interior_profile": interior_profile,
        "coefficient": coefficient,
        "field_objective": field_objective,
        "boundary_objective": boundary_objective,
        "consistency_objective": consistency_objective,
        "profile_objective": profile_objective,
        **direct_geometry,
    }
    total = (
        weights["field"] * field_objective
        + weights["boundary"] * boundary_objective
        + weights["consistency"] * consistency_objective
        + weights["profile"] * profile_objective
    )
    return total, parts


def scalar_temporal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    gradient_weight: float = 0.08,
) -> torch.Tensor:
    point_error = (prediction - target).square()
    point = point_error[valid].mean() if bool(valid.any()) else point_error.sum() * 0.0
    pair_valid = valid[:, 1:] & valid[:, :-1]
    predicted_difference = prediction[:, 1:] - prediction[:, :-1]
    target_difference = target[:, 1:] - target[:, :-1]
    gradient_error = (predicted_difference - target_difference).square()
    gradient = (
        gradient_error[pair_valid].mean()
        if bool(pair_valid.any())
        else gradient_error.sum() * 0.0
    )
    return point + gradient_weight * gradient


def pooled_r2(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(target) & torch.isfinite(prediction)
    truth = target[valid]
    estimate = prediction[valid]
    total = (truth - truth.mean()).square().sum().clamp_min(1.0e-20)
    return 1.0 - (truth - estimate).square().sum() / total
