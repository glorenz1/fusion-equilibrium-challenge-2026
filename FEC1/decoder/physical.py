"""Constrained physical state decoding on a supplied machine grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuralgs.geometry import plasma_green_matrix

from .basis import GreenModeBasis, HarmonicBasis, build_harmonic_basis


@dataclass(frozen=True)
class SpatialCurrentDecoderBasis:
    """Geometry-native basis for the encoder's positive current-mass output.

    ``green_matrix`` includes source-cell quadrature and therefore maps
    current density in A/m^2 to flux in Wb/rad. ``source_weights_m2`` converts
    the encoder's unit-sum cell masses back to current density.
    """

    grid_r_m: np.ndarray
    grid_z_m: np.ndarray
    source_r_m: np.ndarray
    source_z_m: np.ndarray
    source_weights_m2: np.ndarray
    green_matrix_Wb_per_rad_per_A_per_m2: np.ndarray
    harmonics: HarmonicBasis
    source_stride: int
    softening_cells: float

    def __post_init__(self) -> None:
        grid_r = np.asarray(self.grid_r_m, dtype=np.float64).reshape(-1)
        grid_z = np.asarray(self.grid_z_m, dtype=np.float64).reshape(-1)
        source_r = np.asarray(self.source_r_m, dtype=np.float64).reshape(-1)
        source_z = np.asarray(self.source_z_m, dtype=np.float64).reshape(-1)
        source_weights = np.asarray(self.source_weights_m2, dtype=np.float64).reshape(-1)
        green = np.asarray(
            self.green_matrix_Wb_per_rad_per_A_per_m2, dtype=np.float64
        )
        expected_green_shape = (
            len(grid_r) * len(grid_z),
            len(source_r) * len(source_z),
        )
        if green.shape != expected_green_shape:
            raise ValueError(f"green_matrix must have shape {expected_green_shape}")
        if source_weights.shape != (expected_green_shape[1],):
            raise ValueError("source weights do not match the source grid")
        if not np.isfinite(green).all() or not np.isfinite(source_weights).all():
            raise ValueError("Green operator and source weights must be finite")
        if np.any(source_weights <= 0.0):
            raise ValueError("source weights must be strictly positive")
        if not np.array_equal(
            grid_r, np.asarray(self.harmonics.grid_r_m)
        ) or not np.array_equal(grid_z, np.asarray(self.harmonics.grid_z_m)):
            raise ValueError("harmonic and Green-operator grids differ")
        object.__setattr__(self, "grid_r_m", grid_r)
        object.__setattr__(self, "grid_z_m", grid_z)
        object.__setattr__(self, "source_r_m", source_r)
        object.__setattr__(self, "source_z_m", source_z)
        object.__setattr__(self, "source_weights_m2", source_weights)
        object.__setattr__(
            self, "green_matrix_Wb_per_rad_per_A_per_m2", green
        )

    def manifest(self) -> dict[str, object]:
        return {
            "construction": "positive spatial current mass plus free-space Green response",
            "output_shape": [len(self.grid_z_m), len(self.grid_r_m)],
            "source_shape": [len(self.source_z_m), len(self.source_r_m)],
            "source_stride": int(self.source_stride),
            "softening_cells": float(self.softening_cells),
            "harmonics": self.harmonics.manifest(),
            "current_constraint": "nonnegative unit-sum mass scaled by measured |Ip|",
        }


def build_spatial_current_decoder_basis(
    grid_r_m: np.ndarray,
    grid_z_m: np.ndarray,
    *,
    harmonic_count: int = 16,
    source_stride: int = 2,
    softening_cells: float = 0.45,
    harmonic_fields: np.ndarray | None = None,
) -> SpatialCurrentDecoderBasis:
    """Build the exact deployment basis for ``CausalHybridEncoderV1``."""

    grid_r = np.asarray(grid_r_m, dtype=np.float64).reshape(-1)
    grid_z = np.asarray(grid_z_m, dtype=np.float64).reshape(-1)
    green, source_r, source_z, source_weights = plasma_green_matrix(
        grid_r,
        grid_z,
        source_stride=source_stride,
        softening_cells=softening_cells,
    )
    if harmonic_fields is None:
        harmonics = build_harmonic_basis(
            grid_r, grid_z, mode_count=harmonic_count
        )
    else:
        fields = np.asarray(harmonic_fields, dtype=np.float64)
        if fields.shape != (harmonic_count, len(grid_z), len(grid_r)):
            raise ValueError(
                "harmonic_fields must match harmonic_count and the output grid"
            )
        harmonics = HarmonicBasis(grid_r, grid_z, fields)
    return SpatialCurrentDecoderBasis(
        grid_r_m=grid_r,
        grid_z_m=grid_z,
        source_r_m=source_r,
        source_z_m=source_z,
        source_weights_m2=source_weights,
        green_matrix_Wb_per_rad_per_A_per_m2=green,
        harmonics=harmonics,
        source_stride=int(source_stride),
        softening_cells=float(softening_cells),
    )


@dataclass(frozen=True)
class PhysicalDecoderBasis:
    """Fields needed to decode the state predicted by the encoder."""

    core_source_atoms_per_m2: np.ndarray
    core_flux_fields_Wb_per_rad_per_A: np.ndarray
    green_modes: GreenModeBasis
    harmonics: HarmonicBasis

    def __post_init__(self) -> None:
        atoms = np.asarray(self.core_source_atoms_per_m2)
        fields = np.asarray(self.core_flux_fields_Wb_per_rad_per_A)
        if atoms.ndim != 3 or fields.ndim != 3 or len(atoms) != len(fields):
            raise ValueError("core source atoms and flux fields must be aligned 3-D banks")
        if atoms.shape[1:] != self.green_modes.source_modes_per_m2.shape[1:]:
            raise ValueError("core and refinement source grids differ")
        if fields.shape[1:] != self.green_modes.flux_fields_Wb_per_rad_per_A.shape[1:]:
            raise ValueError("core and refinement output grids differ")
        if fields.shape[1:] != self.harmonics.flux_fields_Wb_per_rad_per_coefficient.shape[1:]:
            raise ValueError("plasma and harmonic output grids differ")
        quadrature = self.green_modes.source_weights_m2.reshape(atoms.shape[1:])
        masses = np.sum(atoms * quadrature[None], axis=(1, 2))
        if not np.isfinite(masses).all() or np.any(np.abs(masses) < 1.0e-20):
            raise ValueError("core source atoms must have finite nonzero current")
        if not np.allclose(masses, 1.0, rtol=0.0, atol=2.0e-5):
            raise ValueError("core source atoms are not approximately unit-current")
        # Green/source artifacts may be stored as float32. Normalize atoms and
        # their matching fields together so exact Ip does not depend on their
        # serialization precision.
        object.__setattr__(
            self,
            "core_source_atoms_per_m2",
            atoms / masses[:, None, None],
        )
        object.__setattr__(
            self,
            "core_flux_fields_Wb_per_rad_per_A",
            fields / masses[:, None, None],
        )

    def manifest(self) -> dict[str, object]:
        return {
            "core_atom_count": len(self.core_source_atoms_per_m2),
            "green_modes": self.green_modes.manifest(),
            "harmonics": self.harmonics.manifest(),
            "current_constraint": "simplex core plus zero-net refinement",
        }


@dataclass(frozen=True)
class DecoderState:
    """One or more encoder-predicted physical latent states."""

    simplex_weights: np.ndarray
    green_coefficients: np.ndarray
    harmonic_coefficients: np.ndarray
    gauge_Wb_per_rad: np.ndarray


@dataclass(frozen=True)
class SpatialDecoderState:
    """State emitted by the existing measurement encoder."""

    current_mass: np.ndarray
    harmonic_coefficients: np.ndarray
    gauge_Wb_per_rad: np.ndarray


@dataclass(frozen=True)
class DecodedEquilibrium:
    psi_Wb_per_rad: np.ndarray
    plasma_current_density_A_per_m2: np.ndarray
    net_plasma_current_A: np.ndarray
    current_redistribution_l1_fraction: np.ndarray
    current_mode_scale: np.ndarray


def decode_spatial_current(
    active_psi_Wb_per_rad: np.ndarray,
    plasma_current_magnitude_A: np.ndarray,
    state: SpatialDecoderState,
    basis: SpatialCurrentDecoderBasis,
) -> DecodedEquilibrium:
    """Render the encoder's positive spatial current state on a machine grid."""

    current_mass = np.asarray(state.current_mass, dtype=np.float64)
    if current_mass.ndim == 2:
        current_mass = current_mass[None]
    source_shape = (len(basis.source_z_m), len(basis.source_r_m))
    if current_mass.ndim != 3 or current_mass.shape[1:] != source_shape:
        raise ValueError(
            f"current_mass must have shape (batch, {source_shape[0]}, {source_shape[1]})"
        )
    batch = len(current_mass)
    harmonic_fields = np.asarray(
        basis.harmonics.flux_fields_Wb_per_rad_per_coefficient,
        dtype=np.float64,
    )
    harmonic = _matrix(
        "harmonic_coefficients",
        state.harmonic_coefficients,
        len(harmonic_fields),
    )
    gauge = np.asarray(state.gauge_Wb_per_rad, dtype=np.float64).reshape(-1)
    active = np.asarray(active_psi_Wb_per_rad, dtype=np.float64)
    if active.ndim == 2:
        active = active[None]
    output_shape = (len(basis.grid_z_m), len(basis.grid_r_m))
    ip = np.asarray(plasma_current_magnitude_A, dtype=np.float64).reshape(-1)
    if active.shape != (batch, *output_shape):
        raise ValueError(f"active psi must have shape {(batch, *output_shape)}")
    if len(harmonic) != batch or len(gauge) != batch or len(ip) != batch:
        raise ValueError("all spatial decoder blocks must share one batch dimension")
    if (
        not np.isfinite(current_mass).all()
        or not np.isfinite(active).all()
        or not np.isfinite(ip).all()
        or np.any(ip < 0.0)
        or np.min(current_mass) < -1.0e-10
    ):
        raise ValueError("spatial decoder values must be finite and current mass nonnegative")
    masses = np.sum(current_mass, axis=(1, 2))
    if not np.allclose(masses, 1.0, rtol=0.0, atol=2.0e-6):
        raise ValueError("current_mass must sum to one in every frame")
    current_mass = np.clip(current_mass, 0.0, None)
    current_mass /= np.sum(current_mass, axis=(1, 2), keepdims=True)
    source_weights = basis.source_weights_m2.reshape(source_shape)
    current_density = ip[:, None, None] * current_mass / source_weights[None]
    green = basis.green_matrix_Wb_per_rad_per_A_per_m2
    plasma = (
        current_density.reshape(batch, -1) @ green.T
    ).reshape(batch, *output_shape)
    psi = active + plasma
    if len(harmonic_fields):
        psi += np.einsum(
            "bk,kzr->bzr", harmonic, harmonic_fields, optimize=True
        )
    psi += gauge[:, None, None]
    net_current = np.sum(current_density * source_weights[None], axis=(1, 2))
    if not np.allclose(net_current, ip, rtol=2.0e-8, atol=1.0e-6):
        raise FloatingPointError("decoded current does not equal measured Ip")
    return DecodedEquilibrium(
        psi_Wb_per_rad=psi,
        plasma_current_density_A_per_m2=current_density,
        net_plasma_current_A=net_current,
        current_redistribution_l1_fraction=np.zeros(batch, dtype=np.float64),
        current_mode_scale=np.ones(batch, dtype=np.float64),
    )


def _matrix(name: str, values: np.ndarray, columns: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[None]
    if array.ndim != 2 or array.shape[1] != columns or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape (batch, {columns})")
    return array


def decode_equilibrium(
    active_psi_Wb_per_rad: np.ndarray,
    plasma_current_magnitude_A: np.ndarray,
    state: DecoderState,
    basis: PhysicalDecoderBasis,
    *,
    maximum_current_redistribution: float | None = None,
) -> DecodedEquilibrium:
    """Decode canonicalized flux while preserving measured total plasma current.

    ``maximum_current_redistribution`` is the L1 norm of the zero-net source
    correction divided by ``|Ip|``. Passing ``None`` exposes the mathematical
    representation ceiling; deployment should use a validated encoder prior or
    explicit cap and report it in provenance.
    """

    core_atoms = np.asarray(basis.core_source_atoms_per_m2, dtype=np.float64)
    core_fields = np.asarray(basis.core_flux_fields_Wb_per_rad_per_A, dtype=np.float64)
    green_source = np.asarray(basis.green_modes.source_modes_per_m2, dtype=np.float64)
    green_fields = np.asarray(
        basis.green_modes.flux_fields_Wb_per_rad_per_A, dtype=np.float64
    )
    harmonic_fields = np.asarray(
        basis.harmonics.flux_fields_Wb_per_rad_per_coefficient, dtype=np.float64
    )
    weights = _matrix("simplex_weights", state.simplex_weights, len(core_atoms))
    green = _matrix("green_coefficients", state.green_coefficients, len(green_source))
    harmonic = _matrix(
        "harmonic_coefficients", state.harmonic_coefficients, len(harmonic_fields)
    )
    gauge = np.asarray(state.gauge_Wb_per_rad, dtype=np.float64).reshape(-1)
    active = np.asarray(active_psi_Wb_per_rad, dtype=np.float64)
    if active.ndim == 2:
        active = active[None]
    ip = np.asarray(plasma_current_magnitude_A, dtype=np.float64).reshape(-1)
    batch = len(weights)
    expected_output_shape = core_fields.shape[1:]
    if active.shape != (batch, *expected_output_shape):
        raise ValueError(f"active psi must have shape {(batch, *expected_output_shape)}")
    if len(green) != batch or len(harmonic) != batch or len(gauge) != batch or len(ip) != batch:
        raise ValueError("all decoder state blocks must share one batch dimension")
    if not np.isfinite(active).all() or not np.isfinite(ip).all() or np.any(ip < 0.0):
        raise ValueError("active psi and nonnegative plasma-current magnitudes must be finite")
    if np.min(weights) < -1.0e-10 or not np.allclose(
        np.sum(weights, axis=1), 1.0, rtol=0.0, atol=1.0e-8
    ):
        raise ValueError("simplex weights must be nonnegative and sum to one")
    weights = np.clip(weights, 0.0, None)
    weights /= np.sum(weights, axis=1, keepdims=True)
    correction = np.einsum("bk,kzr->bzr", green, green_source, optimize=True)
    quadrature = basis.green_modes.source_weights_m2.reshape(green_source.shape[1:])
    redistribution = np.sum(
        np.abs(correction) * quadrature[None], axis=(1, 2)
    )
    scale = np.ones(batch, dtype=np.float64)
    if maximum_current_redistribution is not None:
        if maximum_current_redistribution <= 0.0:
            raise ValueError("maximum_current_redistribution must be positive")
        scale = np.minimum(
            1.0,
            float(maximum_current_redistribution) / np.maximum(redistribution, 1.0e-30),
        )
        green = green * scale[:, None]
        correction = correction * scale[:, None, None]
        redistribution = redistribution * scale
    normalized_current = (
        np.einsum("bk,kzr->bzr", weights, core_atoms, optimize=True) + correction
    )
    current_density = ip[:, None, None] * normalized_current
    net_current = np.sum(current_density * quadrature[None], axis=(1, 2))
    psi = active + ip[:, None, None] * (
        np.einsum("bk,kzr->bzr", weights, core_fields, optimize=True)
        + np.einsum("bk,kzr->bzr", green, green_fields, optimize=True)
    )
    if len(harmonic_fields):
        psi += np.einsum("bk,kzr->bzr", harmonic, harmonic_fields, optimize=True)
    psi += gauge[:, None, None]
    if not np.allclose(net_current, ip, rtol=2.0e-8, atol=1.0e-6):
        raise FloatingPointError("decoded current does not equal measured Ip")
    return DecodedEquilibrium(
        psi_Wb_per_rad=psi,
        plasma_current_density_A_per_m2=current_density,
        net_plasma_current_A=net_current,
        current_redistribution_l1_fraction=redistribution,
        current_mode_scale=scale,
    )
