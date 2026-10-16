"""Training-only geometry objectives with identical processing on both sides.

Targets define supervised normalization bands, never inference features. Both
fields pass through those bands and the same soft geometry extractor. Exact
topology labels and polar contour radii do not enter this objective.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from fec1.losses.normalization import teacher_normalized_flux
from fec1.losses.geometry import (
    soft_local_axis,
    soft_first_lcfs_radius,
    contour_points,
    contour_geometry,
    direct_internal_inductance,
)


def extract_geometry(
    shape, target_psi_n, grid_r, grid_z, *, angles=None, angle_count=64, radial_samples=112
):
    """Apply the same differentiable functional to a prediction or a target."""
    normalized, useful = teacher_normalized_flux(shape.float(), target_psi_n)
    axis = soft_local_axis(normalized, useful, grid_r, grid_z)
    radius, theta = soft_first_lcfs_radius(
        normalized,
        axis,
        grid_r,
        grid_z,
        angles=angles,
        angle_count=angle_count,
        radial_samples=radial_samples,
    )
    points = contour_points(axis, radius, theta)
    geometry = contour_geometry(points)
    li = direct_internal_inductance(
        normalized,
        axis,
        radius,
        theta,
        grid_r,
        grid_z,
    ).clamp(0.05, 8.0)
    return {
        "normalized": normalized,
        "axis": axis,
        "radius": radius,
        "points": points,
        "geometry": geometry,
        "li": li,
        "scalars": torch.cat((axis, geometry, li[:, None]), dim=1),
    }


def symmetric_geometry_losses(
    prediction_shape,
    target_shape,
    target_psi_n,
    grid_r,
    grid_z,
    *,
    angles=None,
    angle_count=64,
    radial_samples=112,
):
    """Remove extractor/normalization bias while retaining legacy loss scales.

    This deliberately keeps the axis, displacement, shape and log-li units and
    relative weights of the existing controlled experiment. Variance weighting
    or denser radial tracing would be separate experimental changes."""
    prediction = extract_geometry(
        prediction_shape,
        target_psi_n,
        grid_r,
        grid_z,
        angles=angles,
        angle_count=angle_count,
        radial_samples=radial_samples,
    )
    with torch.no_grad():
        target = extract_geometry(
            target_shape.detach(),
            target_psi_n,
            grid_r,
            grid_z,
            angles=angles,
            angle_count=angle_count,
            radial_samples=radial_samples,
        )
    span = torch.stack((grid_r[-1] - grid_r[0], grid_z[-1] - grid_z[0]))
    axis = ((prediction["axis"] - target["axis"]) / span).square().sum(-1).mean()
    # vector_norm has a defined zero subgradient; no positive epsilon loss floor.
    displacement = torch.linalg.vector_norm(prediction["points"] - target["points"], dim=-1)
    lcfs = F.smooth_l1_loss(displacement / 0.01, torch.zeros_like(displacement), beta=0.5)
    target_geometry = target["geometry"]
    scale = torch.stack(
        (
            torch.full_like(target_geometry[:, 0], 0.10),
            torch.full_like(target_geometry[:, 1], 0.08),
            torch.full_like(target_geometry[:, 2], 0.08),
            (0.05 * target_geometry[:, 3].abs()).clamp_min(0.20),
        ),
        dim=1,
    )
    geometry = F.smooth_l1_loss(prediction["geometry"] / scale, target_geometry / scale, beta=0.20)
    li = F.smooth_l1_loss(prediction["li"].log(), target["li"].log(), beta=0.08)
    return {"direct_axis": axis, "direct_lcfs": lcfs, "direct_geometry": geometry, "direct_li": li}


def geometry_objective(parts, components="both"):
    """Boundary and profile contributions can be ablated independently."""
    if components not in ("both", "boundary", "profile"):
        raise ValueError("geometry components must be both, boundary, or profile")
    boundary = 20 * parts["direct_axis"] + parts["direct_lcfs"] + 3 * parts["direct_geometry"]
    profile = 3 * parts["direct_li"]
    if components == "boundary":
        return boundary
    if components == "profile":
        return profile
    return boundary + profile
