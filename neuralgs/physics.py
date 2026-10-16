"""Differentiable free-boundary Grad--Shafranov building blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import MU0


def linear_spline_basis(s: torch.Tensor, n_coefficients: int) -> torch.Tensor:
    """Clamped, partition-of-unity degree-1 B-spline basis on [0, 1]."""

    s = s.clamp(0.0, 1.0)
    centers = torch.linspace(0.0, 1.0, n_coefficients, device=s.device, dtype=s.dtype)
    spacing = 1.0 / max(n_coefficients - 1, 1)
    basis = torch.relu(1.0 - torch.abs(s.unsqueeze(-1) - centers) / spacing)
    return basis / basis.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def clamped_bspline_basis(
    s: torch.Tensor,
    n_coefficients: int,
    degree: int = 3,
) -> torch.Tensor:
    """Clamped uniform B-splines matching the inverse-oracle basis.

    The Cox--de Boor recursion stays in torch so gradients remain available
    through source-coordinate updates.  The oracle also reduces the degree
    when fewer than four coefficients are requested.
    """

    if n_coefficients < 1:
        raise ValueError("n_coefficients must be positive")
    actual_degree = min(int(degree), n_coefficients - 1)
    x = s.clamp(0.0, 1.0)
    n_interior = n_coefficients - actual_degree - 1
    pieces = [torch.zeros(actual_degree + 1, device=x.device, dtype=x.dtype)]
    if n_interior > 0:
        pieces.append(
            torch.linspace(
                0.0,
                1.0,
                n_interior + 2,
                device=x.device,
                dtype=x.dtype,
            )[1:-1]
        )
    pieces.append(torch.ones(actual_degree + 1, device=x.device, dtype=x.dtype))
    knots = torch.cat(pieces)

    basis = ((x[..., None] >= knots[:-1]) & (x[..., None] < knots[1:])).to(x.dtype)
    for order in range(1, actual_degree + 1):
        count = basis.shape[-1] - 1
        left_denominator = knots[order : order + count] - knots[:count]
        right_denominator = (
            knots[order + 1 : order + count + 1] - knots[1 : count + 1]
        )
        left_term = torch.where(
            left_denominator > 0,
            (x[..., None] - knots[:count])
            / left_denominator.clamp_min(torch.finfo(x.dtype).tiny)
            * basis[..., :count],
            torch.zeros_like(basis[..., :count]),
        )
        right_term = torch.where(
            right_denominator > 0,
            (knots[order + 1 : order + count + 1] - x[..., None])
            / right_denominator.clamp_min(torch.finfo(x.dtype).tiny)
            * basis[..., 1 : count + 1],
            torch.zeros_like(basis[..., :count]),
        )
        basis = left_term + right_term

    endpoint = torch.zeros_like(basis)
    endpoint[..., -1] = 1.0
    return torch.where((x >= 1.0)[..., None], endpoint, basis)


def evaluate_spline(s: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    basis = linear_spline_basis(s, coefficients.shape[-1])
    return torch.einsum("b...k,bk->b...", basis, coefficients)


def projected_flux_source_basis(
    psi_n: torch.Tensor,
    plasma_mask: torch.Tensor,
    source_r: torch.Tensor,
    source_z: torch.Tensor,
    source_weights: torch.Tensor,
    n_coefficients: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a unit-current base and integral-null signed 4+4-style basis.

    The two raw source families span ``(R/R0) B_k(psi_N)`` and
    ``(R0/R) B_k(psi_N)``.  Dividing by the supported area gives every column
    units of inverse area.  Projection is recomputed after the frame-specific
    support is applied, so every correction column has exactly zero discrete
    ``dR*dZ`` integral under ``source_weights``.
    """

    if psi_n.ndim != 3 or plasma_mask.shape != psi_n.shape:
        raise ValueError("psi_n and plasma_mask must both have shape [B,Z,R]")
    if n_coefficients < 1:
        raise ValueError("n_coefficients must be positive")
    expected = (len(source_z), len(source_r))
    if tuple(psi_n.shape[-2:]) != expected:
        raise ValueError(f"source coordinate shape {tuple(psi_n.shape[-2:])} != {expected}")
    dtype = psi_n.dtype
    device = psi_n.device
    weights = source_weights.to(device=device, dtype=dtype).reshape(1, *expected)
    window = plasma_window(source_r, source_z).to(device=device, dtype=dtype)
    support = plasma_mask.to(dtype) * window[None]
    area = (support * weights).sum(dim=(1, 2)).clamp_min(
        torch.finfo(dtype).tiny
    )
    s = psi_n.clamp(0.0, 1.0)
    spline = clamped_bspline_basis(s, n_coefficients)
    rr, _ = torch.meshgrid(
        source_r.to(device=device, dtype=dtype),
        source_z.to(device=device, dtype=dtype),
        indexing="xy",
    )
    r0 = torch.sqrt(
        source_r[0].to(dtype=dtype, device=device).clamp_min(1.0e-4)
        * source_r[-1].to(dtype=dtype, device=device).clamp_min(1.0e-4)
    )
    radial = torch.cat(
        [
            spline * (rr / r0)[None, ..., None],
            spline * (r0 / rr.clamp_min(1.0e-4))[None, ..., None],
        ],
        dim=-1,
    )
    raw = radial * support[..., None] / area[:, None, None, None]

    # A broad, positive, mildly core-weighted flux-function prior carries Ip.
    base_shape = support * torch.sqrt((1.0 - s).clamp_min(0.02))
    base_integral = (base_shape * weights).sum(dim=(1, 2)).clamp_min(
        torch.finfo(dtype).tiny
    )
    base = base_shape / base_integral[:, None, None]
    raw_integral = (raw * weights[..., None]).sum(dim=(1, 2))
    projected = raw - base[..., None] * raw_integral[:, None, None, :]
    return base, projected


def conditioned_signed_source_current(
    psi_n: torch.Tensor,
    plasma_mask: torch.Tensor,
    coefficients: torch.Tensor,
    ip_abs: torch.Tensor,
    source_r: torch.Tensor,
    source_z: torch.Tensor,
    source_weights: torch.Tensor,
    basis_transform: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct ``Ip * (j_base + B_tilde T z)`` with exact discrete Ip."""

    if coefficients.ndim != 2 or coefficients.shape[1] % 2:
        raise ValueError("source coefficients must have shape [B, 2*K]")
    if basis_transform.shape != (coefficients.shape[1], coefficients.shape[1]):
        raise ValueError("basis transform does not match source coefficient dimension")
    base, projected = projected_flux_source_basis(
        psi_n,
        plasma_mask,
        source_r,
        source_z,
        source_weights,
        coefficients.shape[1] // 2,
    )
    transform = basis_transform.to(device=projected.device, dtype=projected.dtype)
    conditioned = torch.einsum("bzrm,mn->bzrn", projected, transform)
    correction = torch.einsum(
        "bzrm,bm->bzr", conditioned, coefficients.to(projected.dtype)
    )
    current = ip_abs.to(projected.dtype)[:, None, None] * (base + correction)
    return current, base, conditioned


def sample_grid(field: torch.Tensor, coords: torch.Tensor, grid_r: torch.Tensor, grid_z: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``field[B,Z,R]`` at physical ``coords[B,N,(R,Z)]``."""

    r_norm = 2.0 * (coords[..., 0] - grid_r[0]) / (grid_r[-1] - grid_r[0]).clamp_min(1.0e-8) - 1.0
    z_norm = 2.0 * (coords[..., 1] - grid_z[0]) / (grid_z[-1] - grid_z[0]).clamp_min(1.0e-8) - 1.0
    sampling_grid = torch.stack([r_norm, z_norm], dim=-1).unsqueeze(2)
    sampled = F.grid_sample(
        field.unsqueeze(1), sampling_grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return sampled[:, 0, :, 0]


def plasma_window(source_r: torch.Tensor, source_z: torch.Tensor) -> torch.Tensor:
    """Machine-agnostic computational guard that excludes coils at grid edges."""

    rn = (source_r - source_r[0]) / (source_r[-1] - source_r[0]).clamp_min(1.0e-8)
    zn = (source_z - source_z[0]) / (source_z[-1] - source_z[0]).clamp_min(1.0e-8)
    rr, zz = torch.meshgrid(rn, zn, indexing="xy")
    sharpness = 45.0
    return (
        torch.sigmoid(sharpness * (rr - 0.055))
        * torch.sigmoid(sharpness * (0.945 - rr))
        * torch.sigmoid(sharpness * (zz - 0.055))
        * torch.sigmoid(sharpness * (0.945 - zz))
    )


def spatial_current_basis(
    source_r: torch.Tensor,
    source_z: torch.Tensor,
    n_modes: int,
) -> torch.Tensor:
    """Return smooth machine-normalized 2-D current-shaping modes.

    The bounded Legendre-product bank spans horizontal/vertical shifts,
    elongation, tilt, and up/down-dependent radial skew (a low-order proxy for
    triangularity).  Coordinates use only each supplied computational grid's
    extent, so the same learned coefficients transfer between machines.
    """

    if not 0 <= int(n_modes) <= 12:
        raise ValueError("spatial current modes must be between 0 and 12")
    if n_modes == 0:
        return source_r.new_empty((0, len(source_z), len(source_r)))
    x = 2.0 * (source_r - source_r[0]) / (
        source_r[-1] - source_r[0]
    ).clamp_min(1.0e-8) - 1.0
    y = 2.0 * (source_z - source_z[0]) / (
        source_z[-1] - source_z[0]
    ).clamp_min(1.0e-8) - 1.0
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    p2x = 0.5 * (3.0 * xx.square() - 1.0)
    p2y = 0.5 * (3.0 * yy.square() - 1.0)
    p3x = 0.5 * (5.0 * xx.pow(3) - 3.0 * xx)
    p3y = 0.5 * (5.0 * yy.pow(3) - 3.0 * yy)
    bank = torch.stack(
        (
            xx,
            yy,
            xx * yy,
            p2x,
            p2y,
            xx * p2y,
            p2x * yy,
            p3x,
            p3y,
            p2x * p2y,
            p3x * yy,
            xx * p3y,
        )
    )
    return bank[:n_modes]


def modulate_spatial_current(
    current_density: torch.Tensor,
    coefficients: torch.Tensor,
    ip_abs: torch.Tensor,
    source_r: torch.Tensor,
    source_z: torch.Tensor,
    source_weights: torch.Tensor,
    amplitude: float,
) -> torch.Tensor:
    """Apply a positive smooth 2-D modulation and exactly restore ``|Ip|``.

    Multiplicative exponential modulation retains the non-negative canonical
    current convention.  Removing the per-frame largest log multiplier is a
    scale-only stabilization because the subsequent quadrature normalization
    cancels any constant factor.
    """

    if current_density.ndim != 3 or coefficients.ndim != 2:
        raise ValueError("expected current [B,Z,R] and coefficients [B,M]")
    if current_density.shape[0] != coefficients.shape[0]:
        raise ValueError("current and coefficient batches do not match")
    n_modes = coefficients.shape[1]
    if n_modes == 0:
        return current_density
    basis = spatial_current_basis(source_r, source_z, n_modes).to(
        device=current_density.device, dtype=current_density.dtype
    )
    bounded = torch.tanh(coefficients.to(current_density.dtype))
    log_multiplier = float(amplitude) * torch.einsum(
        "bm,mzr->bzr", bounded, basis
    ) / float(n_modes) ** 0.5
    log_multiplier = log_multiplier - log_multiplier.amax(dim=(1, 2), keepdim=True)
    multiplier = torch.exp(log_multiplier.clamp_min(-30.0))
    shaped = current_density.clamp_min(0.0) * multiplier
    weights = source_weights.reshape(1, *current_density.shape[-2:]).to(
        device=current_density.device, dtype=current_density.dtype
    )
    integral = (shaped * weights).sum(dim=(1, 2))
    scale = ip_abs.to(current_density.dtype) / integral.clamp_min(
        torch.finfo(current_density.dtype).tiny
    )
    return shaped * scale[:, None, None]


def initial_current_density(
    ip_abs: torch.Tensor,
    token_coords: torch.Tensor,
    token_values: torch.Tensor,
    token_mask: torch.Tensor,
    source_r: torch.Tensor,
    source_z: torch.Tensor,
    source_weights: torch.Tensor,
    center_mode: str = "grid",
) -> torch.Tensor:
    """Generic smooth current prior, exactly normalized to measured Ip."""

    pressure = torch.expm1(token_values[..., 2].clamp(0.0, 20.0)) * 1.0e3
    weights = pressure * token_mask.to(pressure.dtype)
    weight_sum = weights.sum(dim=1).clamp_min(1.0e-6)
    default_r = 0.5 * (source_r[0] + source_r[-1])
    default_z = 0.5 * (source_z[0] + source_z[-1])
    r_span = source_r[-1] - source_r[0]
    z_span = source_z[-1] - source_z[0]
    if center_mode == "grid":
        center_r = default_r.expand_as(ip_abs)
        center_z = default_z.expand_as(ip_abs)
        width_r = (0.12 * r_span).expand_as(ip_abs)
        width_z = (0.14 * z_span).expand_as(ip_abs)
    elif center_mode == "thomson":
        center_r = (weights * token_coords[..., 0]).sum(dim=1) / weight_sum
        center_z = (weights * token_coords[..., 1]).sum(dim=1) / weight_sum
        has_data = token_mask.any(dim=1)
        center_r = torch.where(has_data, center_r, default_r)
        center_z = torch.where(has_data, center_z, default_z)
        var_r = (
            weights * (token_coords[..., 0] - center_r[:, None]) ** 2
        ).sum(dim=1) / weight_sum
        var_z = (
            weights * (token_coords[..., 1] - center_z[:, None]) ** 2
        ).sum(dim=1) / weight_sum
        width_r = torch.sqrt(var_r.clamp_min(0.0)).clamp(0.12 * r_span, 0.28 * r_span)
        width_z = torch.sqrt(var_z.clamp_min(0.0)).clamp(0.14 * z_span, 0.30 * z_span)
    else:
        raise ValueError(f"unknown initial-current center mode: {center_mode}")
    rr, zz = torch.meshgrid(source_r, source_z, indexing="xy")
    radius2 = (
        ((rr[None] - center_r[:, None, None]) / width_r[:, None, None]) ** 2
        + ((zz[None] - center_z[:, None, None]) / width_z[:, None, None]) ** 2
    )
    shape = torch.exp(-0.5 * radius2) * plasma_window(source_r, source_z)[None]
    integral = (shape.flatten(1) * source_weights[None]).sum(dim=1).clamp_min(1.0e-12)
    return shape * (ip_abs / integral)[:, None, None]


def green_response(current_density: torch.Tensor, green: torch.Tensor) -> torch.Tensor:
    """Apply the free-space axisymmetric Green integral."""

    # J_phi is routinely O(1e6) A/m^2 and cannot be represented by float16.
    # Keep this physical integral in float32 even when the learned encoders use
    # CUDA autocast; otherwise Stage B overflows before the first optimizer step.
    with torch.autocast(device_type=current_density.device.type, enabled=False):
        return current_density.float().flatten(1) @ green.float().transpose(0, 1)


def trapezoid_grid_weights(grid_r: torch.Tensor, grid_z: torch.Tensor) -> torch.Tensor:
    """Tensor-product trapezoid weights for a possibly nonuniform R,Z grid."""

    def one_dimensional(values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 1 or len(values) < 2:
            raise ValueError("grid coordinates must be one-dimensional with at least two points")
        result = torch.empty_like(values)
        result[0] = 0.5 * (values[1] - values[0])
        result[-1] = 0.5 * (values[-1] - values[-2])
        if len(values) > 2:
            result[1:-1] = 0.5 * (values[2:] - values[:-2])
        return result

    return torch.outer(one_dimensional(grid_z), one_dimensional(grid_r))


def weighted_gauge_project(field: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Remove the additive gauge minimizing the supplied weighted norm."""

    if field.ndim < 2:
        raise ValueError("field must include a batch and one or more spatial dimensions")
    flattened = field.flatten(1)
    vector = weights.to(device=field.device, dtype=field.dtype).reshape(-1)
    if flattened.shape[1] != len(vector):
        raise ValueError("field and gauge weights have different spatial sizes")
    mean = (flattened * vector[None]).sum(dim=1) / vector.sum().clamp_min(
        torch.finfo(field.dtype).tiny
    )
    return field - mean.reshape((-1,) + (1,) * (field.ndim - 1))


def _perimeter(field: torch.Tensor) -> torch.Tensor:
    return torch.cat([field[:, 0], field[:, -1], field[:, 1:-1, 0], field[:, 1:-1, -1]], dim=1)


def current_density_centroid(
    current_density: torch.Tensor,
    source_r: torch.Tensor,
    source_z: torch.Tensor,
    source_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the physical (R, Z) centroid of canonical toroidal current.

    NeuralGS constructs non-negative canonical ``J_phi`` and enforces its area
    integral to ``|Ip|``.  Its centroid is consequently a machine-independent
    hint for the plasma O-point.  The hint is used only for discrete candidate
    selection, so callers should detach it before passing it to the flux
    coordinate estimator.
    """

    nz, nr = current_density.shape[-2:]
    if nz != len(source_z) or nr != len(source_r):
        raise ValueError("current-density shape does not match the source grid")
    quadrature = source_weights.reshape(1, nz, nr).to(
        device=current_density.device, dtype=current_density.dtype
    )
    weighted = current_density.clamp_min(0.0) * quadrature
    total = weighted.sum(dim=(1, 2))
    rr, zz = torch.meshgrid(source_r, source_z, indexing="xy")
    rr = rr.to(device=current_density.device, dtype=current_density.dtype)
    zz = zz.to(device=current_density.device, dtype=current_density.dtype)
    safe_total = total.clamp_min(torch.finfo(current_density.dtype).tiny)
    centroid_r = (weighted * rr[None]).sum(dim=(1, 2)) / safe_total
    centroid_z = (weighted * zz[None]).sum(dim=(1, 2)) / safe_total
    fallback_r = 0.5 * (source_r[0] + source_r[-1])
    fallback_z = 0.5 * (source_z[0] + source_z[-1])
    valid = torch.isfinite(total) & (total > 0.0)
    centroid_r = torch.where(valid, centroid_r, fallback_r.to(centroid_r))
    centroid_z = torch.where(valid, centroid_z, fallback_z.to(centroid_z))
    return centroid_r, centroid_z


def _axis_indices(
    psi: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    central: torch.Tensor,
    estimator: str,
    hint_r: torch.Tensor | None = None,
    hint_z: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select piecewise-constant O-point indices for each frame.

    ``current_local`` enumerates genuine 8-neighbour local maxima and chooses
    the one nearest the prior-current centroid in normalized physical distance.
    The gathered flux value remains differentiable; only candidate identity is
    discrete.  Empty/invalid candidate sets safely fall back to the legacy
    central global maximum.
    """

    batch, nz, nr = psi.shape
    finite_psi = torch.nan_to_num(psi, nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf)
    global_flat = finite_psi.masked_fill(~central[None], -torch.inf).flatten(1).argmax(dim=1)
    global_z = torch.div(global_flat, nr, rounding_mode="floor")
    global_r = global_flat % nr
    if estimator == "global":
        return global_z, global_r
    if estimator != "current_local":
        raise ValueError(f"unknown axis estimator: {estimator}")
    if hint_r is None or hint_z is None:
        return global_z, global_r

    center = finite_psi[:, 1:-1, 1:-1]
    neighbours = torch.stack(
        [
            finite_psi[:, 1 + dz : nz - 1 + dz, 1 + dr : nr - 1 + dr]
            for dz in (-1, 0, 1)
            for dr in (-1, 0, 1)
            if dz != 0 or dr != 0
        ],
        dim=0,
    )
    # Permit a discretized/flat-topped O-point but reject a completely flat
    # field: at least one neighbour must be strictly lower.
    local_inner = (center >= neighbours.max(dim=0).values) & (
        center > neighbours.min(dim=0).values
    )
    local = torch.zeros_like(psi, dtype=torch.bool)
    local[:, 1:-1, 1:-1] = local_inner
    local &= central[None] & torch.isfinite(psi)

    r_span = (grid_r[-1] - grid_r[0]).abs().clamp_min(1.0e-8)
    z_span = (grid_z[-1] - grid_z[0]).abs().clamp_min(1.0e-8)
    rr, zz = torch.meshgrid(grid_r, grid_z, indexing="xy")
    distance2 = (
        ((rr[None] - hint_r.detach()[:, None, None]) / r_span).square()
        + ((zz[None] - hint_z.detach()[:, None, None]) / z_span).square()
    )
    distance2 = distance2.masked_fill(~local, torch.inf)
    local_flat = distance2.flatten(1).argmin(dim=1)
    local_z = torch.div(local_flat, nr, rounding_mode="floor")
    local_r = local_flat % nr
    has_local = local.flatten(1).any(dim=1)
    valid_hint = torch.isfinite(hint_r) & torch.isfinite(hint_z)
    use_local = has_local & valid_hint
    return torch.where(use_local, local_z, global_z), torch.where(use_local, local_r, global_r)


def estimate_flux_coordinates(
    psi: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    temperature: float = 0.035,
    separatrix_span_scale: float = 1.0,
    separatrix_estimator: str = "saddle",
    axis_estimator: str = "global",
    axis_hint_r: torch.Tensor | None = None,
    axis_hint_z: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate axis, separatrix, psi_N, and a smooth plasma mask.

    The legacy ``saddle`` mode selects the highest stationary point with
    negative Hessian determinant outside the magnetic-axis neighborhood.  The
    ``perimeter`` mode uses the robust perimeter level directly; this is more
    stable for early Picard iterates and avoids selecting interior false saddles.
    Selection is piecewise differentiable: gradients flow through the selected
    flux values, while the discrete candidate identity is treated as fixed.
    """

    batch, nz, nr = psi.shape
    rn = (grid_r - grid_r[0]) / (grid_r[-1] - grid_r[0]).clamp_min(1.0e-8)
    zn = (grid_z - grid_z[0]) / (grid_z[-1] - grid_z[0]).clamp_min(1.0e-8)
    rr, zz = torch.meshgrid(rn, zn, indexing="xy")
    central = (rr > 0.08) & (rr < 0.92) & (zz > 0.07) & (zz < 0.93)
    axis_z, axis_r = _axis_indices(
        psi,
        grid_r,
        grid_z,
        central,
        axis_estimator,
        axis_hint_r,
        axis_hint_z,
    )
    batch_index = torch.arange(batch, device=psi.device)
    axis_value = psi[batch_index, axis_z, axis_r]

    perimeter = _perimeter(psi)
    boundary_level = torch.quantile(perimeter, 0.90, dim=1)
    field_std = psi.flatten(1).std(dim=1).clamp_min(1.0e-5)
    span = (axis_value - boundary_level).clamp_min(0.10 * field_std)

    if separatrix_estimator == "perimeter":
        # Use the already protected axis-to-boundary span.  Taking the raw
        # perimeter quantile directly can put the boundary above the selected
        # axis on coil-dominated MAST iterates and collapse the denominator.
        separatrix = axis_value - span
    elif separatrix_estimator == "saddle":
        # Passing coordinate tensors avoids a device synchronization for every
        # Picard iteration and also handles a future nonuniform grid correctly.
        dpsi_dz, dpsi_dr = torch.gradient(
            psi, spacing=(grid_z, grid_r), dim=(1, 2), edge_order=1
        )
        d2z = torch.gradient(
            dpsi_dz, spacing=(grid_z,), dim=(1,), edge_order=1
        )[0]
        d2r = torch.gradient(
            dpsi_dr, spacing=(grid_r,), dim=(2,), edge_order=1
        )[0]
        drz = torch.gradient(
            dpsi_dr, spacing=(grid_z,), dim=(1,), edge_order=1
        )[0]
        determinant = d2r * d2z - drz.square()
        gradient2 = dpsi_dr.square() + dpsi_dz.square()

        distance2 = (
            (rr[None] - rn[axis_r][:, None, None]) ** 2
            + (zz[None] - zn[axis_z][:, None, None]) ** 2
        )
        candidate = central[None] & (distance2 > 0.035) & (determinant < 0.0)
        candidate &= psi < (axis_value - 0.025 * span)[:, None, None]
        candidate &= psi > (boundary_level - 0.25 * span)[:, None, None]
        grad_scale = gradient2.flatten(1).median(dim=1).values.clamp_min(1.0e-12)
        score = psi / span[:, None, None] - 0.08 * gradient2 / grad_scale[:, None, None]
        score = score.masked_fill(~candidate, -torch.inf)
        saddle_flat = score.flatten(1).argmax(dim=1)
        saddle_z = torch.div(saddle_flat, nr, rounding_mode="floor")
        saddle_r = saddle_flat % nr
        saddle_value = psi[batch_index, saddle_z, saddle_r]
        has_saddle = candidate.flatten(1).any(dim=1)
        separatrix = torch.where(has_saddle, saddle_value, boundary_level)
        lower = axis_value - 0.96 * span
        upper = axis_value - 0.07 * span
        separatrix = torch.maximum(lower, torch.minimum(separatrix, upper))
    else:
        raise ValueError(f"unknown separatrix estimator: {separatrix_estimator}")
    denominator = (axis_value - separatrix).clamp_min(1.0e-6)
    denominator = denominator * max(float(separatrix_span_scale), 1.0e-3)
    # Return the effective level as well so scalar derivations use the same
    # calibrated flux span as the current-profile coordinates.
    separatrix = axis_value - denominator
    psi_n = (axis_value[:, None, None] - psi) / denominator[:, None, None]
    mask = torch.sigmoid((1.0 - psi_n) / max(temperature, 1.0e-3))
    return psi_n, mask, axis_value, separatrix


def smooth_pressure(
    query_s: torch.Tensor,
    token_s: torch.Tensor,
    token_pressure: torch.Tensor,
    token_mask: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    """Nadaraya--Watson pressure profile in normalized-flux coordinates."""

    distance = (query_s.flatten(1).unsqueeze(-1) - token_s.unsqueeze(1)) / bandwidth
    weights = torch.exp(-0.5 * distance.square()) * token_mask[:, None].to(query_s.dtype)
    numerator = (weights * token_pressure[:, None]).sum(dim=-1)
    denominator = weights.sum(dim=-1)
    fallback = (
        (token_pressure * token_mask.to(query_s.dtype)).sum(dim=1)
        / token_mask.sum(dim=1).clamp_min(1).to(query_s.dtype)
    )
    values = torch.where(denominator > 1.0e-7, numerator / denominator.clamp_min(1.0e-7), fallback[:, None])
    return values.reshape_as(query_s)


def profile_current_density(
    psi_n: torch.Tensor,
    plasma_mask: torch.Tensor,
    token_s: torch.Tensor,
    token_values: torch.Tensor,
    token_mask: torch.Tensor,
    pressure_correction: torch.Tensor,
    pressure_scale: torch.Tensor,
    ffprime_coefficients: torch.Tensor,
    pressure_fraction: torch.Tensor,
    ip_abs: torch.Tensor,
    source_r: torch.Tensor,
    source_z: torch.Tensor,
    source_weights: torch.Tensor,
    pressure_bandwidth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct p'- and FF'-like current terms and enforce total measured Ip."""

    token_pressure = torch.expm1(token_values[..., 2].clamp(0.0, 20.0)) * 1.0e3
    s = psi_n.clamp(-0.05, 1.15)
    epsilon = 0.015
    edge_pressure = smooth_pressure(
        torch.ones_like(s[:, :1, :1]), token_s, token_pressure, token_mask, pressure_bandwidth
    )

    def total_pressure(at_s: torch.Tensor) -> torch.Tensor:
        base = smooth_pressure(at_s, token_s, token_pressure, token_mask, pressure_bandwidth)
        anchored = torch.relu(base - edge_pressure)
        multiplier = torch.exp(0.45 * torch.tanh(evaluate_spline(at_s.clamp(0, 1), pressure_correction)))
        return (
            pressure_scale[:, None, None]
            * anchored * multiplier * (1.0 - at_s.clamp(0, 1)).sqrt()
        )

    pressure = total_pressure(s)
    dp_ds = (total_pressure(s + epsilon) - total_pressure(s - epsilon)) / (2.0 * epsilon)
    rr = source_r[None, None, :]
    pressure_shape = torch.relu(-dp_ds) * rr
    ff_coeff = F.softplus(ffprime_coefficients) + 1.0e-4
    ff_shape = evaluate_spline(s, ff_coeff) / rr.clamp_min(1.0e-4)
    window = plasma_window(source_r, source_z)
    support = plasma_mask * window[None]
    pressure_shape *= support
    ff_shape *= support

    pressure_integral = (pressure_shape.flatten(1) * source_weights[None]).sum(dim=1)
    pressure_shape = torch.where(
        (pressure_integral > 1.0e-12)[:, None, None], pressure_shape, ff_shape
    )

    def normalized(shape: torch.Tensor) -> torch.Tensor:
        integral = (shape.flatten(1) * source_weights[None]).sum(dim=1).clamp_min(1.0e-12)
        return shape / integral[:, None, None]

    fraction = pressure_fraction.clamp(0.02, 0.90)
    current = ip_abs[:, None, None] * (
        fraction[:, None, None] * normalized(pressure_shape)
        + (1.0 - fraction)[:, None, None] * normalized(ff_shape)
    )
    return current, pressure


def gs_current_from_flux(psi: torch.Tensor, grid_r: torch.Tensor, grid_z: torch.Tensor) -> torch.Tensor:
    """Inverse-GS current-density estimate used for Stage-A supervision."""

    dpsi_dz, dpsi_dr = torch.gradient(
        psi, spacing=(grid_z, grid_r), dim=(1, 2), edge_order=1
    )
    d2z = torch.gradient(
        dpsi_dz, spacing=(grid_z,), dim=(1,), edge_order=1
    )[0]
    d2r = torch.gradient(
        dpsi_dr, spacing=(grid_r,), dim=(2,), edge_order=1
    )[0]
    rr = grid_r[None, None, :]
    delta_star = d2r - dpsi_dr / rr.clamp_min(1.0e-4) + d2z
    return -delta_star / (MU0 * rr.clamp_min(1.0e-4))
