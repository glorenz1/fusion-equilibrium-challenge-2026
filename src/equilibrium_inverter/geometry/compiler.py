"""Compile geometry into Green, quadrature, and harmonic operators."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from neuralgs.geometry import loop_psi_per_ampere

from ..config import EquilibriumInverterConfig
from ..inputs.shot import PreparedShot
from ..physics.harmonics import build_harmonic_basis


@dataclass(frozen=True)
class TorchMachineOperator:
    grid_r_m: torch.Tensor
    grid_z_m: torch.Tensor
    source_r_m: torch.Tensor
    source_z_m: torch.Tensor
    source_weights_m2: torch.Tensor
    green_matrix: torch.Tensor
    harmonic_fields: torch.Tensor
    source_r_indices: torch.Tensor
    source_z_indices: torch.Tensor
    length_scale_m: float
    reference_radius_m: float


@dataclass(frozen=True)
class MachineOperator:
    """Static machine operator; no learned machine embedding is present."""

    grid_r_m: np.ndarray
    grid_z_m: np.ndarray
    source_r_m: np.ndarray
    source_z_m: np.ndarray
    source_weights_m2: np.ndarray
    green_matrix_Wb_per_rad_per_A_per_m2: np.ndarray
    harmonic_fields_dimensionless: np.ndarray
    source_r_indices: np.ndarray
    source_z_indices: np.ndarray
    length_scale_m: float
    reference_radius_m: float
    geometry_sha256: str
    toroidal_field_response_Tm_per_A: float
    approximations: tuple[str, ...]

    def to_torch(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> TorchMachineOperator:
        def tensor(values: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(values, device=device, dtype=dtype)

        return TorchMachineOperator(
            grid_r_m=tensor(self.grid_r_m),
            grid_z_m=tensor(self.grid_z_m),
            source_r_m=tensor(self.source_r_m),
            source_z_m=tensor(self.source_z_m),
            source_weights_m2=tensor(self.source_weights_m2),
            green_matrix=tensor(self.green_matrix_Wb_per_rad_per_A_per_m2),
            harmonic_fields=tensor(self.harmonic_fields_dimensionless),
            source_r_indices=torch.as_tensor(
                self.source_r_indices, device=device, dtype=torch.long
            ),
            source_z_indices=torch.as_tensor(
                self.source_z_indices, device=device, dtype=torch.long
            ),
            length_scale_m=float(self.length_scale_m),
            reference_radius_m=float(self.reference_radius_m),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "geometry_sha256": self.geometry_sha256,
            "grid_shape": [len(self.grid_z_m), len(self.grid_r_m)],
            "source_shape": [len(self.source_z_m), len(self.source_r_m)],
            "harmonic_modes": len(self.harmonic_fields_dimensionless),
            "length_scale_m": self.length_scale_m,
            "reference_radius_m": self.reference_radius_m,
            "toroidal_field_response_Tm_per_A": (
                self.toroidal_field_response_Tm_per_A
                if np.isfinite(self.toroidal_field_response_Tm_per_A)
                else None
            ),
            "approximations": list(self.approximations),
            "machine_identifier_is_model_input": False,
        }


def _geometry_hash(
    prepared: PreparedShot,
    source_r_indices: np.ndarray,
    source_z_indices: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    fields = prepared.active_fields
    for array in (
        fields.grid_r_m,
        fields.grid_z_m,
        fields.psi_per_si,
        fields.b_r_per_si,
        fields.b_z_per_si,
        source_r_indices,
        source_z_indices,
    ):
        digest.update(np.asarray(array).tobytes())
    digest.update("|".join(fields.current_units).encode("utf-8"))
    digest.update(
        np.asarray(prepared.toroidal_field_response_Tm_per_A, dtype=np.float64).tobytes()
    )
    return digest.hexdigest()


def _source_indices(count: int, stride: int) -> np.ndarray:
    result = np.arange(0, count, stride, dtype=np.int64)
    if result[-1] != count - 1:
        result = np.append(result, count - 1)
    return result


def _trapezoid_weights(coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    result = np.empty_like(values)
    result[0] = 0.5 * (values[1] - values[0])
    result[-1] = 0.5 * (values[-1] - values[-2])
    result[1:-1] = 0.5 * (values[2:] - values[:-2])
    return result


def _rectangular_green_operator(
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    *,
    source_stride: int,
    softening_cells: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a Green operator with independent radial and vertical sources."""

    r_indices = _source_indices(len(grid_r), source_stride)
    z_indices = _source_indices(len(grid_z), source_stride)
    source_r, source_z = grid_r[r_indices], grid_z[z_indices]
    target_r, target_z = np.meshgrid(grid_r, grid_z)
    source_rr, source_zz = np.meshgrid(source_r, source_z)
    dr = float(np.mean(np.diff(source_r)))
    dz = float(np.mean(np.diff(source_z)))
    softening = softening_cells * np.sqrt(dr * dr + dz * dz)
    green = loop_psi_per_ampere(
        target_r.reshape(-1, 1),
        target_z.reshape(-1, 1),
        source_rr.reshape(1, -1),
        source_zz.reshape(1, -1),
        softening=softening,
    )
    weights = np.outer(
        _trapezoid_weights(source_z), _trapezoid_weights(source_r)
    ).reshape(-1)
    green *= weights[None]
    return green, source_r, source_z, weights, r_indices, z_indices


def compile_machine_operator(
    prepared: PreparedShot,
    config: EquilibriumInverterConfig | None = None,
) -> MachineOperator:
    """Compile one supplied geometry without reading equilibrium targets."""

    cfg = config or EquilibriumInverterConfig()
    grid_r = np.asarray(prepared.measurements.grid_r_m, dtype=np.float64)
    grid_z = np.asarray(prepared.measurements.grid_z_m, dtype=np.float64)
    if len(grid_r) < 3 or len(grid_z) < 3:
        raise ValueError("equilibrium operators require at least three R and Z points")
    (
        green,
        source_r,
        source_z,
        source_weights,
        source_r_indices,
        source_z_indices,
    ) = _rectangular_green_operator(
        grid_r,
        grid_z,
        source_stride=cfg.solver.source_stride,
        softening_cells=cfg.solver.green_softening_cells,
    )
    harmonics = build_harmonic_basis(grid_r, grid_z, cfg.solver.harmonic_modes)
    if len(harmonics):
        rms = np.sqrt(np.mean(np.square(harmonics), axis=(1, 2))).clip(1.0e-12)
        harmonics = harmonics / rms[:, None, None]
    span_r = float(grid_r[-1] - grid_r[0])
    span_z = float(grid_z[-1] - grid_z[0])
    length_scale = float(np.sqrt(max(span_r * span_z, 1.0e-12)))
    reference_radius = float(np.sqrt(grid_r[0] * grid_r[-1]))
    return MachineOperator(
        grid_r_m=grid_r,
        grid_z_m=grid_z,
        source_r_m=np.asarray(source_r, dtype=np.float64),
        source_z_m=np.asarray(source_z, dtype=np.float64),
        source_weights_m2=np.asarray(source_weights, dtype=np.float64),
        green_matrix_Wb_per_rad_per_A_per_m2=np.asarray(green, dtype=np.float64),
        harmonic_fields_dimensionless=np.asarray(harmonics, dtype=np.float64),
        source_r_indices=source_r_indices,
        source_z_indices=source_z_indices,
        length_scale_m=length_scale,
        reference_radius_m=reference_radius,
        geometry_sha256=_geometry_hash(
            prepared, source_r_indices, source_z_indices
        ),
        toroidal_field_response_Tm_per_A=float(
            prepared.toroidal_field_response_Tm_per_A
        ),
        approximations=tuple(prepared.active_fields.approximations),
    )
