"""Differentiable profile-based free-boundary equilibrium forward model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralgs.geometry import MU0

from ..config import SolverConfig
from ..geometry.compiler import TorchMachineOperator
from ..inputs.shot import TorchShot
from ..state.latent import DecodedLatentState, StateLayout


@dataclass(frozen=True)
class ForwardResult:
    """Equilibrium and synthetic diagnostics decoded from one latent trajectory."""

    psi_Wb_per_rad: torch.Tensor
    psi_n: torch.Tensor
    plasma_mask: torch.Tensor
    current_density_A_per_m2: torch.Tensor
    net_plasma_current_A: torch.Tensor
    predicted_thomson_pressure_Pa: torch.Tensor
    q95: torch.Tensor
    beta_n: torch.Tensor
    q95_physics: torch.Tensor
    beta_n_physics: torch.Tensor
    scalar_physics_valid: torch.Tensor
    topology_probabilities: torch.Tensor
    axis_flux_Wb_per_rad: torch.Tensor
    boundary_flux_Wb_per_rad: torch.Tensor


def _linear_spline_basis(values: torch.Tensor, count: int) -> torch.Tensor:
    x = values.clamp(0.0, 1.0)
    centers = torch.linspace(0.0, 1.0, count, dtype=x.dtype, device=x.device)
    spacing = 1.0 / max(count - 1, 1)
    basis = torch.relu(1.0 - torch.abs(x[..., None] - centers) / spacing)
    return basis / basis.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def _evaluate_profile(values: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    basis = _linear_spline_basis(values, coefficients.shape[-1])
    return torch.einsum("b...k,bk->b...", basis, coefficients)


def _perimeter(field: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [field[:, 0], field[:, 1:-1, -1], field[:, -1].flip(1), field[:, 1:-1, 0].flip(1)],
        dim=1,
    )


def _flux_coordinates(
    psi: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Piecewise-differentiable axis/perimeter normalized flux coordinates."""

    batch, nz, nr = psi.shape
    r0, r1 = max(1, int(0.07 * nr)), min(nr - 1, int(0.93 * nr))
    z0, z1 = max(1, int(0.07 * nz)), min(nz - 1, int(0.93 * nz))
    central = psi[:, z0:z1, r0:r1]
    flat = central.flatten(1).argmax(dim=1)
    axis_z = torch.div(flat, central.shape[2], rounding_mode="floor") + z0
    axis_r = flat % central.shape[2] + r0
    batch_index = torch.arange(batch, device=psi.device)
    axis = psi[batch_index, axis_z, axis_r]
    boundary = torch.quantile(_perimeter(psi), 0.90, dim=1)
    scale = psi.flatten(1).std(dim=1).clamp_min(1.0e-7)
    span = (axis - boundary).clamp_min(0.10 * scale)
    boundary = axis - span
    psi_n = (axis[:, None, None] - psi) / span[:, None, None]
    mask = torch.sigmoid((1.0 - psi_n) / max(float(temperature), 1.0e-3))
    return psi_n, mask, axis, boundary


def _plasma_window(
    source_r: torch.Tensor,
    source_z: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    rn = (source_r - source_r[0]) / (source_r[-1] - source_r[0]).clamp_min(1.0e-8)
    zn = (source_z - source_z[0]) / (source_z[-1] - source_z[0]).clamp_min(1.0e-8)
    rr, zz = torch.meshgrid(rn, zn, indexing="xy")
    sharpness = 45.0
    return (
        torch.sigmoid(sharpness * (rr - margin))
        * torch.sigmoid(sharpness * (1.0 - margin - rr))
        * torch.sigmoid(sharpness * (zz - margin))
        * torch.sigmoid(sharpness * (1.0 - margin - zz))
    )


def _normalized_source(
    shape: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    integral = (shape.flatten(1) * weights[None]).sum(dim=1).clamp_min(1.0e-12)
    return shape / integral[:, None, None]


def _trapezoid_weights(values: torch.Tensor) -> torch.Tensor:
    weights = torch.empty_like(values)
    weights[0] = 0.5 * (values[1] - values[0])
    weights[-1] = 0.5 * (values[-1] - values[-2])
    weights[1:-1] = 0.5 * (values[2:] - values[:-2])
    return weights


def _sample_field(
    field: torch.Tensor,
    coordinates_m: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
) -> torch.Tensor:
    r = 2.0 * (coordinates_m[..., 0] - grid_r[0]) / (grid_r[-1] - grid_r[0]).clamp_min(1.0e-8) - 1.0
    z = 2.0 * (coordinates_m[..., 1] - grid_z[0]) / (grid_z[-1] - grid_z[0]).clamp_min(1.0e-8) - 1.0
    query = torch.stack([r, z], dim=-1).unsqueeze(2)
    sampled = F.grid_sample(
        field.unsqueeze(1), query, mode="bilinear", padding_mode="border", align_corners=True
    )
    return sampled[:, 0, :, 0]


class EquilibriumForwardModel(nn.Module):
    """Render equilibria from universal profile latents and a supplied operator.

    The learned state never contains a coil name or machine embedding.  Every
    Picard source is a Grad--Shafranov flux function and is normalized to the
    measured plasma current exactly under the source quadrature.
    """

    def __init__(
        self,
        operator: TorchMachineOperator,
        layout: StateLayout,
        config: SolverConfig,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.config = config
        self.register_buffer("grid_r_m", operator.grid_r_m)
        self.register_buffer("grid_z_m", operator.grid_z_m)
        self.register_buffer("source_r_m", operator.source_r_m)
        self.register_buffer("source_z_m", operator.source_z_m)
        self.register_buffer("source_weights_m2", operator.source_weights_m2)
        self.register_buffer("green_matrix", operator.green_matrix)
        self.register_buffer("harmonic_fields", operator.harmonic_fields)
        self.register_buffer("source_r_indices", operator.source_r_indices)
        self.register_buffer("source_z_indices", operator.source_z_indices)
        full_weights = torch.outer(
            _trapezoid_weights(operator.grid_z_m),
            _trapezoid_weights(operator.grid_r_m),
        )
        self.register_buffer("full_grid_weights_m2", full_weights)
        self.length_scale_m = float(operator.length_scale_m)

    def _initial_current(self, ip: torch.Tensor) -> torch.Tensor:
        rr, zz = torch.meshgrid(self.source_r_m, self.source_z_m, indexing="xy")
        center_r = 0.5 * (self.source_r_m[0] + self.source_r_m[-1])
        center_z = 0.5 * (self.source_z_m[0] + self.source_z_m[-1])
        width_r = 0.18 * (self.source_r_m[-1] - self.source_r_m[0])
        width_z = 0.20 * (self.source_z_m[-1] - self.source_z_m[0])
        radius2 = ((rr - center_r) / width_r.clamp_min(1.0e-6)).square()
        radius2 += ((zz - center_z) / width_z.clamp_min(1.0e-6)).square()
        shape = torch.exp(-0.5 * radius2) * _plasma_window(
            self.source_r_m,
            self.source_z_m,
            self.config.current_window_margin,
        )
        normalized = _normalized_source(shape[None].expand(len(ip), -1, -1), self.source_weights_m2)
        return ip[:, None, None] * normalized

    def _green_response(self, current_density: torch.Tensor) -> torch.Tensor:
        flat = current_density.float().flatten(1) @ self.green_matrix.float().transpose(0, 1)
        return flat.reshape(len(current_density), len(self.grid_z_m), len(self.grid_r_m))

    def _profile_source(
        self,
        psi_n: torch.Tensor,
        mask: torch.Tensor,
        state: DecodedLatentState,
        ip: torch.Tensor,
    ) -> torch.Tensor:
        radial = self.source_r_m[None, None, :]
        window = _plasma_window(
            self.source_r_m,
            self.source_z_m,
            self.config.current_window_margin,
        )
        support = mask * window[None]
        pressure_shape = radial * _evaluate_profile(psi_n, state.pressure_shape) * support
        ffprime_shape = (
            _evaluate_profile(psi_n, state.ffprime_shape)
            / radial.clamp_min(1.0e-4)
            * support
        )
        pressure = _normalized_source(pressure_shape, self.source_weights_m2)
        ffprime = _normalized_source(ffprime_shape, self.source_weights_m2)
        fraction = state.pressure_current_fraction[:, None, None]
        return ip[:, None, None] * (fraction * pressure + (1.0 - fraction) * ffprime)

    def forward(self, shot: TorchShot, latent_vector: torch.Tensor) -> ForwardResult:
        if shot.frames != len(latent_vector):
            raise ValueError("shot and latent trajectory have different frame counts")
        state = self.layout.decode(latent_vector)
        active = torch.nan_to_num(shot.active_psi_Wb_per_rad)
        external = active.clone()
        if self.harmonic_fields.shape[0]:
            external = external + torch.einsum(
                "bh,hzr->bzr",
                state.harmonic_coefficients * shot.flux_scale_Wb_per_rad[:, None],
                self.harmonic_fields,
            )
        external = external + (
            state.gauge_dimensionless * shot.flux_scale_Wb_per_rad
        )[:, None, None]

        current = self._initial_current(shot.plasma_current_A)
        psi = external + self._green_response(current)
        for _ in range(self.config.picard_iterations):
            source_psi = psi[:, self.source_z_indices][:, :, self.source_r_indices]
            source_n, source_mask, _, _ = _flux_coordinates(
                source_psi, self.config.lcfs_temperature
            )
            new_current = self._profile_source(
                source_n, source_mask, state, shot.plasma_current_A
            )
            new_psi = external + self._green_response(new_current)
            damping = float(self.config.picard_damping)
            psi = (1.0 - damping) * psi + damping * new_psi
            current = (1.0 - damping) * current + damping * new_current

        # The final source rendering, unlike a damped iterate, exactly matches
        # the state and measured-current constraint.
        source_psi = psi[:, self.source_z_indices][:, :, self.source_r_indices]
        source_n, source_mask, _, _ = _flux_coordinates(
            source_psi, self.config.lcfs_temperature
        )
        current = self._profile_source(source_n, source_mask, state, shot.plasma_current_A)
        psi = external + self._green_response(current)
        psi_n, plasma_mask, axis, boundary = _flux_coordinates(
            psi, self.config.lcfs_temperature
        )

        token_s = _sample_field(
            psi_n,
            shot.thomson_coordinates_m,
            self.grid_r_m,
            self.grid_z_m,
        ).clamp(0.0, 1.25)
        pressure_shape = _evaluate_profile(token_s.clamp(0.0, 1.0), state.pressure_shape)
        core_shape = _evaluate_profile(torch.zeros_like(token_s), state.pressure_shape)
        normalized_pressure = (
            (1.0 - token_s.clamp(0.0, 1.0))
            * pressure_shape
            / core_shape.clamp_min(1.0e-6)
        )
        total_pressure = (
            shot.pressure_scale_Pa[:, None]
            * state.pressure_scale_multiplier[:, None]
            * normalized_pressure
        )
        electron_pressure = total_pressure * state.electron_pressure_fraction[:, None]
        grid_pressure_shape = _evaluate_profile(
            psi_n.clamp(0.0, 1.0), state.pressure_shape
        )
        grid_core_shape = _evaluate_profile(
            torch.zeros_like(psi_n), state.pressure_shape
        )
        total_pressure_grid = (
            shot.pressure_scale_Pa[:, None, None]
            * state.pressure_scale_multiplier[:, None, None]
            * (1.0 - psi_n.clamp(0.0, 1.0))
            * grid_pressure_shape
            / grid_core_shape.clamp_min(1.0e-6)
        )

        integration_weight = plasma_mask * self.full_grid_weights_m2[None]
        plasma_area = integration_weight.sum(dim=(1, 2)).clamp_min(1.0e-8)
        radial_grid = self.grid_r_m[None, None, :]
        vertical_grid = self.grid_z_m[None, :, None]
        center_r = (integration_weight * radial_grid).sum(dim=(1, 2)) / plasma_area
        center_z = (integration_weight * vertical_grid).sum(dim=(1, 2)) / plasma_area
        radial_variance = (
            integration_weight * (radial_grid - center_r[:, None, None]).square()
        ).sum(dim=(1, 2)) / plasma_area
        vertical_variance = (
            integration_weight * (vertical_grid - center_z[:, None, None]).square()
        ).sum(dim=(1, 2)) / plasma_area
        minor_radius = 2.0 * torch.sqrt(radial_variance.clamp_min(1.0e-8))
        elongation = torch.sqrt(
            vertical_variance.clamp_min(1.0e-8)
            / radial_variance.clamp_min(1.0e-8)
        ).clamp(0.4, 4.0)
        toroidal_field = shot.toroidal_F_Tm / center_r.clamp_min(0.05)
        safe_toroidal_field = toroidal_field.clamp_min(1.0e-6)
        q95_physics = (
            2.0
            * torch.pi
            * minor_radius.square()
            * safe_toroidal_field
            / (MU0 * center_r.clamp_min(0.05) * shot.plasma_current_A.clamp_min(1.0))
            * 0.5
            * (1.0 + elongation.square())
        )
        average_pressure = (
            total_pressure_grid * integration_weight
        ).sum(dim=(1, 2)) / plasma_area
        beta_t_percent = 100.0 * 2.0 * MU0 * average_pressure / safe_toroidal_field.square()
        beta_n_physics = (
            beta_t_percent
            * minor_radius
            * safe_toroidal_field
            / (shot.plasma_current_A.clamp_min(1.0) / 1.0e6)
        )
        scalar_valid = shot.toroidal_response_valid & (shot.toroidal_F_Tm > 0.0)
        q95 = torch.where(
            scalar_valid,
            q95_physics.clamp(0.05, 30.0) * (state.q95 / 4.0),
            state.q95,
        )
        beta_n = torch.where(
            scalar_valid,
            beta_n_physics.clamp(0.01, 10.0) * (state.beta_n / 2.0),
            state.beta_n,
        )
        net_current = (current.flatten(1) * self.source_weights_m2[None]).sum(dim=1)
        return ForwardResult(
            psi_Wb_per_rad=psi,
            psi_n=psi_n,
            plasma_mask=plasma_mask,
            current_density_A_per_m2=current,
            net_plasma_current_A=net_current,
            predicted_thomson_pressure_Pa=electron_pressure,
            q95=q95,
            beta_n=beta_n,
            q95_physics=q95_physics,
            beta_n_physics=beta_n_physics,
            scalar_physics_valid=scalar_valid,
            topology_probabilities=state.topology_probabilities,
            axis_flux_Wb_per_rad=axis,
            boundary_flux_Wb_per_rad=boundary,
        )
