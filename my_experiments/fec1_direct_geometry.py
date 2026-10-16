"""Differentiable approximations to geometry derived by the FEC1 scorer.

Unlike auxiliary topology heads, every quantity in this module is computed
from the rendered flux raster.  The LCFS is represented by the first outward
``psi_n == 1`` crossing from the raster-derived magnetic axis, and internal
inductance follows the scorer's volume/boundary-gradient ratio.
"""

from __future__ import annotations

import math

import torch


def _angles(count: int, reference: torch.Tensor) -> torch.Tensor:
    return torch.linspace(
        0.0, 2.0 * math.pi, count + 1,
        device=reference.device, dtype=reference.dtype,
    )[:-1]


def soft_local_axis(
    psi_n: torch.Tensor,
    useful: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    temperature: float = 0.035,
) -> torch.Tensor:
    """Return an interior, local-basin-weighted soft magnetic axis."""
    if psi_n.ndim != 3 or useful.shape != psi_n.shape:
        raise ValueError("axis inputs must have shape (batch, Z, R)")
    work = psi_n.float()
    valid = useful.bool().clone()
    valid[:, :2] = False
    valid[:, -2:] = False
    valid[:, :, :2] = False
    valid[:, :, -2:] = False
    neighbour_mean = torch.nn.functional.avg_pool2d(
        work[:, None], kernel_size=3, stride=1, padding=1,
    )[:, 0]
    basin = torch.relu(neighbour_mean - work)
    logits = (-work + 2.0 * basin) / temperature
    logits = logits.flatten(1).clamp(-80.0, 80.0)
    logits = logits.masked_fill(~valid.flatten(1), -80.0)
    probability = torch.softmax(logits, dim=1).reshape_as(work)
    rr, zz = torch.meshgrid(grid_r.float(), grid_z.float(), indexing="xy")
    return torch.column_stack((
        (probability * rr[None]).sum(dim=(-2, -1)),
        (probability * zz[None]).sum(dim=(-2, -1)),
    ))


def soft_first_lcfs_radius(
    psi_n: torch.Tensor,
    axis: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    angles: torch.Tensor | None = None,
    angle_count: int = 64,
    radial_samples: int = 112,
    temperature: float = 0.025,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trace a soft first outward ``psi_n == 1`` crossing in physical metres."""
    if psi_n.ndim != 3 or axis.shape != (len(psi_n), 2):
        raise ValueError("LCFS inputs have incompatible shapes")
    if radial_samples < 8 or angle_count < 8:
        raise ValueError("LCFS sampling is too sparse")
    work = torch.nan_to_num(psi_n.float(), nan=2.0, posinf=2.0, neginf=-0.5)
    axis = axis.float()
    grid_r = grid_r.float()
    grid_z = grid_z.float()
    theta = _angles(angle_count, work) if angles is None else angles.to(work).flatten()
    direction = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    diagonal = torch.sqrt(
        (grid_r[-1] - grid_r[0]).square()
        + (grid_z[-1] - grid_z[0]).square()
    )
    radial = torch.linspace(
        0.0, 1.0, radial_samples, device=work.device, dtype=work.dtype,
    ) * diagonal
    point = (
        axis[:, None, None, :]
        + radial[None, None, :, None] * direction[None, :, None, :]
    )
    unit_r = (point[..., 0] - grid_r[0]) / (grid_r[-1] - grid_r[0])
    unit_z = (point[..., 1] - grid_z[0]) / (grid_z[-1] - grid_z[0])
    valid = (unit_r >= 0.0) & (unit_r <= 1.0) & (unit_z >= 0.0) & (unit_z <= 1.0)
    sample_grid = torch.stack((2.0 * unit_r - 1.0, 2.0 * unit_z - 1.0), dim=-1)
    sampled = torch.nn.functional.grid_sample(
        work[:, None], sample_grid, mode="bilinear",
        padding_mode="border", align_corners=True,
    )[:, 0]

    # Treat the outward transition as a discrete hazard.  Multiplication by
    # survival makes later coil-driven psi==1 crossings irrelevant.
    hazard = torch.sigmoid((sampled - 1.0) / temperature)
    hazard = torch.where(valid, hazard, torch.ones_like(hazard))
    hazard[..., 0] = 0.0
    hazard[..., -1] = 1.0
    safe = hazard.clamp(1.0e-7, 1.0 - 1.0e-7)
    log_not = torch.log1p(-safe)
    log_survival = torch.cumsum(log_not, dim=-1) - log_not
    probability = torch.exp(log_survival) * safe
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1.0e-7)
    radius = (probability * radial[None, None]).sum(dim=-1)
    return radius, theta


def contour_points(
    axis: torch.Tensor, radius: torch.Tensor, angles: torch.Tensor,
) -> torch.Tensor:
    direction = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    return axis[:, None, :] + radius[..., None] * direction[None]


def contour_geometry(
    points: torch.Tensor, temperature: float = 0.012,
) -> torch.Tensor:
    """Approximate kappa, triangularities and toroidal volume of a contour."""
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("contour points must have shape (batch, points, 2)")
    r, z = points[..., 0], points[..., 1]

    def soft_at(value: torch.Tensor, sign: float) -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch.softmax(sign * value / temperature, dim=1)
        return (weight * r).sum(dim=1), (weight * z).sum(dim=1)

    r_at_max, _ = soft_at(r, 1.0)
    r_at_min, _ = soft_at(r, -1.0)
    r_at_top, z_max = soft_at(z, 1.0)
    r_at_bottom, z_min = soft_at(z, -1.0)
    width = (r_at_max - r_at_min).clamp_min(1.0e-4)
    minor = 0.5 * width
    r_geo = 0.5 * (r_at_max + r_at_min)
    kappa = (z_max - z_min) / width
    tri_top = (r_geo - r_at_top) / minor
    tri_bottom = (r_geo - r_at_bottom) / minor

    following = torch.roll(points, shifts=-1, dims=1)
    cross = r * following[..., 1] - following[..., 0] * z
    first_moment = ((r + following[..., 0]) * cross).sum(dim=1) / 6.0
    volume = 2.0 * math.pi * torch.sqrt(first_moment.square() + 1.0e-10)
    return torch.column_stack((kappa, tri_top, tri_bottom, volume))


def _physical_gradient(
    field: torch.Tensor, grid_r: torch.Tensor, grid_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dr_middle = (
        field[..., 2:] - field[..., :-2]
    ) / (grid_r[2:] - grid_r[:-2])[None, None]
    dr_first = (field[..., 1:2] - field[..., :1]) / (grid_r[1] - grid_r[0])
    dr_last = (field[..., -1:] - field[..., -2:-1]) / (grid_r[-1] - grid_r[-2])
    derivative_r = torch.cat((dr_first, dr_middle, dr_last), dim=-1)

    dz_middle = (
        field[..., 2:, :] - field[..., :-2, :]
    ) / (grid_z[2:] - grid_z[:-2])[None, :, None]
    dz_first = (field[..., 1:2, :] - field[..., :1, :]) / (grid_z[1] - grid_z[0])
    dz_last = (field[..., -1:, :] - field[..., -2:-1, :]) / (grid_z[-1] - grid_z[-2])
    derivative_z = torch.cat((dz_first, dz_middle, dz_last), dim=-2)
    return derivative_r, derivative_z


def direct_internal_inductance(
    psi_n: torch.Tensor,
    axis: torch.Tensor,
    radius: torch.Tensor,
    angles: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    mask_temperature: float = 0.018,
) -> torch.Tensor:
    """Differentiable analogue of the scorer's ITER li(2) functional."""
    work = torch.nan_to_num(psi_n.float(), nan=2.0, posinf=2.0, neginf=-0.5)
    grid_r = grid_r.float()
    grid_z = grid_z.float()
    derivative_r, derivative_z = _physical_gradient(work, grid_r, grid_z)
    rr, zz = torch.meshgrid(grid_r, grid_z, indexing="xy")
    bp = torch.sqrt(derivative_r.square() + derivative_z.square() + 1.0e-10) / rr[None]

    delta_r = rr[None] - axis[:, 0, None, None]
    delta_z = zz[None] - axis[:, 1, None, None]
    pixel_radius = torch.sqrt(delta_r.square() + delta_z.square() + 1.0e-10)
    pixel_angle = torch.remainder(torch.atan2(delta_z, delta_r), 2.0 * math.pi)
    angle_position = pixel_angle * (radius.shape[1] / (2.0 * math.pi))
    lower = torch.floor(angle_position).long().remainder(radius.shape[1])
    upper = (lower + 1).remainder(radius.shape[1])
    fraction = angle_position - torch.floor(angle_position)
    lower_radius = torch.gather(radius, 1, lower.flatten(1)).reshape_as(pixel_radius)
    upper_radius = torch.gather(radius, 1, upper.flatten(1)).reshape_as(pixel_radius)
    boundary_radius = lower_radius + fraction * (upper_radius - lower_radius)
    inside = torch.sigmoid((boundary_radius - pixel_radius) / mask_temperature)
    volume_weight = inside * (2.0 * math.pi * rr[None])
    bp2_average = (
        (bp.square() * volume_weight).sum(dim=(-2, -1))
        / volume_weight.sum(dim=(-2, -1)).clamp_min(1.0e-8)
    )

    points = contour_points(axis, radius, angles)
    unit_r = (points[..., 0] - grid_r[0]) / (grid_r[-1] - grid_r[0])
    unit_z = (points[..., 1] - grid_z[0]) / (grid_z[-1] - grid_z[0])
    sample_grid = torch.stack((2.0 * unit_r - 1.0, 2.0 * unit_z - 1.0), dim=-1)
    bp_on_contour = torch.nn.functional.grid_sample(
        bp[:, None], sample_grid[:, :, None, :], mode="bilinear",
        padding_mode="border", align_corners=True,
    )[:, 0, :, 0]
    following = torch.roll(points, shifts=-1, dims=1)
    segment = torch.sqrt((following - points).square().sum(dim=-1) + 1.0e-10)
    bp_midpoint = 0.5 * (bp_on_contour + torch.roll(bp_on_contour, shifts=-1, dims=1))
    boundary_average = (
        (bp_midpoint * segment).sum(dim=1) / segment.sum(dim=1).clamp_min(1.0e-8)
    )
    return bp2_average / boundary_average.square().clamp_min(1.0e-8)


def exact_contour_levelset_loss(
    prediction_n: torch.Tensor,
    axis_target: torch.Tensor,
    contour_offset: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
) -> torch.Tensor:
    """Approximate normal displacement at exact scorer-contour points."""
    work = torch.nan_to_num(prediction_n.float(), nan=2.0, posinf=2.0, neginf=-0.5)
    offset = contour_offset.to(work).float()
    valid = torch.isfinite(offset).all(dim=-1)
    point = axis_target[:, None] + torch.nan_to_num(offset)
    unit_r = (point[..., 0] - grid_r[0]) / (grid_r[-1] - grid_r[0])
    unit_z = (point[..., 1] - grid_z[0]) / (grid_z[-1] - grid_z[0])
    valid &= (unit_r >= 0.0) & (unit_r <= 1.0) & (unit_z >= 0.0) & (unit_z <= 1.0)
    sample_grid = torch.stack((2.0 * unit_r - 1.0, 2.0 * unit_z - 1.0), dim=-1)
    sampled = torch.nn.functional.grid_sample(
        work[:, None], sample_grid[:, :, None, :], mode="bilinear",
        padding_mode="border", align_corners=True,
    )[:, 0, :, 0]
    derivative_r, derivative_z = _physical_gradient(
        work, grid_r.float(), grid_z.float(),
    )
    gradient = torch.sqrt(derivative_r.square() + derivative_z.square() + 1.0e-10)
    sampled_gradient = torch.nn.functional.grid_sample(
        gradient[:, None], sample_grid[:, :, None, :], mode="bilinear",
        padding_mode="border", align_corners=True,
    )[:, 0, :, 0]
    normal_displacement = torch.abs(sampled - 1.0) / sampled_gradient.clamp_min(0.05)
    error = torch.nn.functional.smooth_l1_loss(
        normal_displacement / 0.005,
        torch.zeros_like(normal_displacement),
        beta=0.5,
        reduction="none",
    )
    if bool(valid.any()):
        return error[valid].mean()
    return error.sum() * 0.0


def direct_raster_geometry_losses(
    prediction_n: torch.Tensor,
    target_n: torch.Tensor,
    useful: torch.Tensor,
    target_topology: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    exact_target_radius: torch.Tensor | None = None,
    contour_angles: torch.Tensor | None = None,
    exact_target_contour_offset: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Losses for axis, exact-LCFS displacement, shape and internal inductance."""
    prediction_n = prediction_n.float()
    target_n = torch.nan_to_num(target_n.float(), nan=2.0, posinf=2.0, neginf=-0.5)
    axis_target = target_topology[:, :2].float()
    axis_fallback = soft_local_axis(target_n, useful, grid_r, grid_z)
    axis_valid = torch.isfinite(axis_target).all(dim=1)
    axis_target = torch.where(axis_valid[:, None], axis_target, axis_fallback)
    axis_prediction = soft_local_axis(prediction_n, useful, grid_r, grid_z)
    r_span = (grid_r[-1] - grid_r[0]).float()
    z_span = (grid_z[-1] - grid_z[0]).float()
    axis_loss = (
        ((axis_prediction[:, 0] - axis_target[:, 0]) / r_span).square()
        + ((axis_prediction[:, 1] - axis_target[:, 1]) / z_span).square()
    ).mean()

    if contour_angles is None:
        angle_count = 64 if exact_target_radius is None else exact_target_radius.shape[1]
        theta = _angles(angle_count, prediction_n)
    else:
        theta = contour_angles.to(prediction_n).flatten()
        angle_count = len(theta)
    predicted_radius, theta = soft_first_lcfs_radius(
        prediction_n, axis_prediction, grid_r, grid_z,
        angles=theta, angle_count=angle_count,
    )
    derived_target_radius, _ = soft_first_lcfs_radius(
        target_n, axis_target, grid_r, grid_z,
        angles=theta, angle_count=angle_count,
    )
    if exact_target_radius is None:
        target_radius = derived_target_radius
    else:
        exact = exact_target_radius.to(prediction_n).float()
        exact_valid = torch.isfinite(exact).all(dim=1) & (exact > 0.0).all(dim=1)
        target_radius = torch.where(
            exact_valid[:, None], exact, derived_target_radius,
        )

    predicted_points = contour_points(axis_prediction, predicted_radius, theta)
    target_points = contour_points(axis_target, target_radius, theta)
    displacement = torch.sqrt(
        (predicted_points - target_points).square().sum(dim=-1) + 1.0e-10
    )
    lcfs = torch.nn.functional.smooth_l1_loss(
        displacement / 0.01,
        torch.zeros_like(displacement),
        beta=0.5,
    )

    predicted_geometry = contour_geometry(predicted_points)
    target_geometry_proxy = contour_geometry(target_points)
    exact_geometry = target_topology[:, 2:6].to(prediction_n).float()
    exact_valid = torch.isfinite(exact_geometry)
    geometry_target = torch.where(exact_valid, exact_geometry, target_geometry_proxy)
    scale = torch.column_stack((
        torch.full_like(geometry_target[:, 0], 0.10),
        torch.full_like(geometry_target[:, 1], 0.08),
        torch.full_like(geometry_target[:, 2], 0.08),
        (0.05 * geometry_target[:, 3].abs()).clamp_min(0.20),
    ))
    geometry = torch.nn.functional.smooth_l1_loss(
        predicted_geometry / scale,
        geometry_target / scale,
        beta=0.20,
    )

    predicted_li = direct_internal_inductance(
        prediction_n, axis_prediction, predicted_radius, theta, grid_r, grid_z,
    ).clamp(0.05, 8.0)
    target_li_proxy = direct_internal_inductance(
        target_n, axis_target, target_radius, theta, grid_r, grid_z,
    ).detach().clamp(0.05, 8.0)
    li = torch.nn.functional.smooth_l1_loss(
        predicted_li.log(), target_li_proxy.log(), beta=0.08,
    )
    exact_lcfs = prediction_n.sum() * 0.0
    if exact_target_contour_offset is not None:
        exact_lcfs = exact_contour_levelset_loss(
            prediction_n,
            axis_target,
            exact_target_contour_offset,
            grid_r,
            grid_z,
        )
    return {
        "direct_axis": axis_loss,
        "direct_lcfs": lcfs,
        "direct_geometry": geometry,
        "direct_li": li,
        "direct_exact_lcfs": exact_lcfs,
    }
