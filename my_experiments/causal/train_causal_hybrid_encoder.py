#!/usr/bin/env python3
"""Train ``FEC1.encoder.CausalHybridEncoderV1`` on causal DIII-D histories.

The supervised state is the frozen positive 33x33 spatial-current bridge.  The
loss combines current-distribution, harmonic, gauge, q95, betaN, and
differentiable physical-decoder reconstruction terms.  Model selection uses the
predeclared whole-shot development split only; the frozen 200-shot decoder audit
is neither trained on nor used for checkpoint selection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuralgs.geometry import MU0, plasma_green_matrix
from neuralgs.training import (
    dimensionless_flux_geometry_loss,
    normalized_flux_topology_loss,
)
from FEC1.encoder.model import CausalHybridEncoderConfig, CausalHybridEncoderV1


DEFAULT_INPUTS = ROOT / "my_experiments/causal_hybrid_encoder_inputs/inputs.npz"
DEFAULT_INPUT_REPORT = ROOT / "my_experiments/causal_hybrid_encoder_inputs/report.json"
DEFAULT_BRIDGE = ROOT / "my_experiments/d3d_spatial_current_bridge_wide/bridge.npz"
DEFAULT_BRIDGE_REPORT = ROOT / "my_experiments/d3d_spatial_current_bridge_wide/report.json"
DEFAULT_OUTPUT = ROOT / "my_experiments/causal_hybrid_encoder_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        # NPZ members are materialized as independent ndarrays.  Copying the
        # 76+ GiB float16 DIII-D spatial member would double peak RAM without
        # extending its lifetime or safety after the archive closes.
        return {key: archive[key] for key in archive.files}


@dataclass(frozen=True)
class InputNormalizer:
    spatial_mean: np.ndarray
    spatial_std: np.ndarray
    scalar_mean: np.ndarray
    scalar_std: np.ndarray
    thomson_mean: np.ndarray
    thomson_std: np.ndarray

    def to_json(self) -> dict[str, list[float]]:
        return {
            key: np.asarray(getattr(self, key)).tolist()
            for key in (
                "spatial_mean",
                "spatial_std",
                "scalar_mean",
                "scalar_std",
                "thomson_mean",
                "thomson_std",
            )
        }


@dataclass(frozen=True)
class TargetScales:
    harmonic_mean: np.ndarray
    harmonic_std: np.ndarray
    gauge_mean: float
    gauge_std: float
    log_q95_mean: float
    log_q95_std: float
    log_betaN_mean: float
    log_betaN_std: float
    q95_mean: float = 0.0
    q95_std: float = 1.0
    betaN_mean: float = 0.0
    betaN_std: float = 1.0

    def to_json(self) -> dict[str, float | list[float]]:
        return {
            key: (
                np.asarray(getattr(self, key)).tolist()
                if isinstance(getattr(self, key), np.ndarray)
                else float(getattr(self, key))
            )
            for key in self.__dataclass_fields__
        }


def _safe_std(values: np.ndarray, axis: tuple[int, ...] | int = 0) -> np.ndarray:
    std = np.asarray(np.std(values, axis=axis), dtype=np.float64)
    return np.where(std >= 1.0e-5, std, 1.0)


def fit_input_normalizer(
    inputs: dict[str, np.ndarray], positions: np.ndarray
) -> InputNormalizer:
    positions = np.asarray(positions, dtype=np.int64)
    spatial_channels = inputs["spatial_values"].shape[2]
    scalar_features = inputs["shared_scalars"].shape[-1]
    thomson_features = inputs["thomson_values"].shape[-1]
    sums = {
        "spatial": np.zeros(spatial_channels, dtype=np.float64),
        "scalar": np.zeros(scalar_features, dtype=np.float64),
        "thomson": np.zeros(thomson_features, dtype=np.float64),
    }
    sumsquares = {key: np.zeros_like(value) for key, value in sums.items()}
    counts = {key: 0 for key in sums}
    # The full DIII-D cache is intentionally larger than RAM in float64.  Fit
    # exact training-only moments in bounded chunks instead of materializing a
    # second copy of every spatial window.
    for start in range(0, len(positions), 64):
        selected = positions[start : start + 64]
        sequence_mask = inputs["sequence_mask"][selected].astype(bool)
        spatial = inputs["spatial_values"][selected].astype(np.float64)
        spatial_values = spatial[sequence_mask]
        sums["spatial"] += spatial_values.sum(axis=(0, 2, 3))
        sumsquares["spatial"] += np.square(spatial_values).sum(axis=(0, 2, 3))
        counts["spatial"] += spatial_values.shape[0] * spatial_values.shape[-2] * spatial_values.shape[-1]

        scalar_values = inputs["shared_scalars"][selected].astype(np.float64)[
            sequence_mask
        ]
        sums["scalar"] += scalar_values.sum(axis=0)
        sumsquares["scalar"] += np.square(scalar_values).sum(axis=0)
        counts["scalar"] += len(scalar_values)

        token_mask = (
            inputs["thomson_attention_mask"][selected].astype(bool)
            & sequence_mask[:, :, None]
        )
        token_values = inputs["thomson_values"][selected].astype(np.float64)[
            token_mask
        ]
        if len(token_values):
            sums["thomson"] += token_values.sum(axis=0)
            sumsquares["thomson"] += np.square(token_values).sum(axis=0)
            counts["thomson"] += len(token_values)

    def moments(name: str) -> tuple[np.ndarray, np.ndarray]:
        count = counts[name]
        if not count:
            return np.zeros_like(sums[name]), np.ones_like(sums[name])
        mean = sums[name] / count
        variance = np.maximum(sumsquares[name] / count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        return mean, np.where(std >= 1.0e-5, std, 1.0)

    spatial_mean, spatial_std = moments("spatial")
    scalar_mean, scalar_std = moments("scalar")
    thomson_mean, thomson_std = moments("thomson")
    return InputNormalizer(
        spatial_mean=spatial_mean,
        spatial_std=spatial_std,
        scalar_mean=scalar_mean,
        scalar_std=scalar_std,
        thomson_mean=thomson_mean,
        thomson_std=thomson_std,
    )


def fit_target_scales(
    bridge: dict[str, np.ndarray], source_indices: np.ndarray
) -> TargetScales:
    harmonic = bridge["harmonic"][source_indices].astype(np.float64)
    gauge = bridge["gauge"][source_indices].astype(np.float64)
    q95 = bridge["target_q95"][source_indices].astype(np.float64)
    beta = bridge["target_betaN"][source_indices].astype(np.float64)
    valid_q95 = np.isfinite(q95) & (q95 > 0.0)
    valid_beta = np.isfinite(beta) & (beta > 0.0)
    if not valid_q95.any() or not valid_beta.any():
        raise RuntimeError("positive q95 and betaN labels are required")
    log_q95 = np.log(q95[valid_q95])
    log_beta = np.log(beta[valid_beta])
    return TargetScales(
        harmonic_mean=harmonic.mean(axis=0),
        harmonic_std=_safe_std(harmonic),
        gauge_mean=float(gauge.mean()),
        gauge_std=float(_safe_std(gauge)),
        log_q95_mean=float(log_q95.mean()),
        log_q95_std=float(_safe_std(log_q95)),
        log_betaN_mean=float(log_beta.mean()),
        log_betaN_std=float(_safe_std(log_beta)),
        q95_mean=float(q95[valid_q95].mean()),
        q95_std=float(_safe_std(q95[valid_q95])),
        betaN_mean=float(beta[valid_beta].mean()),
        betaN_std=float(_safe_std(beta[valid_beta])),
    )


class CausalBridgeDataset(Dataset):
    def __init__(
        self,
        inputs: dict[str, np.ndarray],
        bridge: dict[str, np.ndarray],
        positions: np.ndarray,
        normalizer: InputNormalizer,
    ) -> None:
        self.inputs = inputs
        self.bridge = bridge
        self.positions = np.asarray(positions, dtype=np.int64)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        position = int(self.positions[item])
        source = int(self.inputs["source_index"][position])
        sequence_mask = self.inputs["sequence_mask"][position].astype(bool)
        token_mask = self.inputs["thomson_attention_mask"][position].astype(bool)
        spatial = (
            self.inputs["spatial_values"][position].astype(np.float32)
            - self.normalizer.spatial_mean[None, :, None, None]
        ) / self.normalizer.spatial_std[None, :, None, None]
        scalars = (
            self.inputs["shared_scalars"][position].astype(np.float32)
            - self.normalizer.scalar_mean[None]
        ) / self.normalizer.scalar_std[None]
        thomson = (
            self.inputs["thomson_values"][position].astype(np.float32)
            - self.normalizer.thomson_mean[None, None]
        ) / self.normalizer.thomson_std[None, None]
        spatial[~sequence_mask] = 0.0
        scalars[~sequence_mask] = 0.0
        thomson[~token_mask] = 0.0
        mass = np.clip(
            self.bridge["current_mass"][source].astype(np.float32), 0.0, None
        )
        mass /= max(float(mass.sum()), 1.0e-30)
        result = {
            "spatial_values": torch.from_numpy(spatial.astype(np.float32)),
            "shared_scalars": torch.from_numpy(scalars.astype(np.float32)),
            "thomson_values": torch.from_numpy(thomson.astype(np.float32)),
            "thomson_attention_mask": torch.from_numpy(token_mask),
            "delta_time_ms": torch.from_numpy(
                self.inputs["delta_time_ms"][position].astype(np.float32)
            ),
            "sequence_mask": torch.from_numpy(sequence_mask),
            "current_mass": torch.from_numpy(mass),
            "harmonic": torch.from_numpy(
                self.bridge["harmonic"][source].astype(np.float32)
            ),
            "gauge": torch.tensor(
                self.bridge["gauge"][source], dtype=torch.float32
            ),
            "q95": torch.tensor(
                self.bridge["target_q95"][source], dtype=torch.float32
            ),
            "betaN": torch.tensor(
                self.bridge["target_betaN"][source], dtype=torch.float32
            ),
            "active_psi": torch.from_numpy(
                self.inputs["active_psi_Wb_per_rad"][position].astype(np.float32)
            ),
            "ip_abs": torch.tensor(
                self.inputs["plasma_current_magnitude_A"][position],
                dtype=torch.float32,
            ),
            "target_psi": torch.from_numpy(
                self.bridge["target_psi"][source].astype(np.float32)
            ),
            "source_index": torch.tensor(source, dtype=torch.int64),
        }
        if "target_psi_n" in self.bridge:
            result["target_psi_n"] = torch.from_numpy(
                self.bridge["target_psi_n"][source].astype(np.float32)
            )
        if "base_q95" in self.bridge and "base_betaN" in self.bridge:
            result["base_q95"] = torch.tensor(
                self.bridge["base_q95"][source], dtype=torch.float32
            )
            result["base_betaN"] = torch.tensor(
                self.bridge["base_betaN"][source], dtype=torch.float32
            )
        if "target_sequence_index" in self.inputs:
            target_sequence_index = int(
                self.inputs["target_sequence_index"][position]
            )
        else:
            target_sequence_index = int(np.flatnonzero(sequence_mask)[-1])
        result["target_sequence_index"] = torch.tensor(
            target_sequence_index, dtype=torch.int64
        )
        if "regime_flags" in self.inputs:
            result["regime_flags"] = torch.from_numpy(
                self.inputs["regime_flags"][position].astype(bool)
            )
        if "sample_weight" in self.inputs:
            result["sample_weight"] = torch.tensor(
                self.inputs["sample_weight"][position], dtype=torch.float32
            )
        if "shot_index" in self.inputs:
            result["shot_index"] = torch.tensor(
                self.inputs["shot_index"][position], dtype=torch.int64
            )
        return result


class FrozenSpatialDecoder(nn.Module):
    """DIII-D field renderer and auxiliary physical decoder."""

    # A 10 kA denominator makes otherwise ordinary pre-ramp DIII-D flux
    # offsets require direct-head values above 50. The head is deliberately
    # bounded, so use a DIII-D-specific scale floor that keeps every audited
    # target comfortably representable without changing the decoded units.
    DIRECT_FLUX_CURRENT_FLOOR_A = 2.5e5

    def __init__(self, bridge: dict[str, np.ndarray]) -> None:
        super().__init__()
        green, source_r, source_z, source_weights = plasma_green_matrix(
            bridge["grid_r"].astype(np.float64),
            bridge["grid_z"].astype(np.float64),
            source_stride=2,
            softening_cells=0.45,
        )
        if (len(source_z), len(source_r)) != (33, 33):
            raise RuntimeError("frozen decoder source grid is not 33x33")
        np.testing.assert_allclose(
            source_weights,
            bridge["source_weights"].astype(np.float64),
            rtol=0.0,
            atol=1.0e-12,
        )
        self.register_buffer("green", torch.from_numpy(green.astype(np.float32)))
        self.register_buffer(
            "inverse_source_weights",
            torch.from_numpy((1.0 / source_weights).astype(np.float32)),
        )
        self.register_buffer(
            "harmonic_fields",
            torch.from_numpy(bridge["harmonics"].astype(np.float32)),
        )
        self.output_shape = (len(bridge["grid_z"]), len(bridge["grid_r"]))
        self.length_scale_m = 0.5 * float(
            bridge["grid_r"][-1] - bridge["grid_r"][0]
        )
        self.current_floor_A = self.DIRECT_FLUX_CURRENT_FLOOR_A

    def flux_scale(self, ip_abs: torch.Tensor) -> torch.Tensor:
        """Match the fixed normalization used by the DIII-D input raster."""

        return (
            MU0
            * ip_abs.clamp_min(self.current_floor_A)
            * self.length_scale_m
        )

    def direct_flux(
        self,
        active_psi: torch.Tensor,
        ip_abs: torch.Tensor,
        normalized_flux_residual: torch.Tensor,
    ) -> torch.Tensor:
        if normalized_flux_residual.shape != active_psi.shape:
            raise ValueError("direct residual and active flux shapes differ")
        return active_psi + self.flux_scale(ip_abs)[:, None, None] * normalized_flux_residual

    def forward(
        self,
        active_psi: torch.Tensor,
        ip_abs: torch.Tensor,
        current_mass: torch.Tensor,
        harmonic: torch.Tensor,
        gauge: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(active_psi)
        density = (
            ip_abs[:, None]
            * current_mass.reshape(batch, -1)
            * self.inverse_source_weights[None]
        )
        plasma = (density @ self.green.T).reshape(batch, *self.output_shape)
        result = active_psi + plasma
        result = result + torch.einsum(
            "bk,kzr->bzr", harmonic, self.harmonic_fields
        )
        return result + gauge[:, None, None]


def current_moments(mass: torch.Tensor) -> torch.Tensor:
    size_z, size_r = mass.shape[-2:]
    radial = torch.linspace(-1.0, 1.0, size_r, device=mass.device, dtype=mass.dtype)
    vertical = torch.linspace(-1.0, 1.0, size_z, device=mass.device, dtype=mass.dtype)
    rr, zz = torch.meshgrid(radial, vertical, indexing="xy")
    features = torch.stack([rr, zz, rr**2, zz**2, rr * zz], dim=0)
    return torch.einsum("bzr,kzr->bk", mass, features)


def normalized_field_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = (prediction - target).flatten(1).square().sum(dim=1)
    centered = target - target.mean(dim=(-2, -1), keepdim=True)
    scale = centered.flatten(1).square().sum(dim=1).clamp_min(1.0e-8)
    return (residual / scale).mean()


def normalized_gradient_mse(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    losses = []
    for dimension in (-2, -1):
        predicted_gradient = torch.diff(prediction, dim=dimension)
        target_gradient = torch.diff(target, dim=dimension)
        numerator = (predicted_gradient - target_gradient).flatten(1).square().sum(1)
        denominator = target_gradient.flatten(1).square().sum(1).clamp_min(1.0e-8)
        losses.append(numerator / denominator)
    return torch.stack(losses).mean()


def zero_safe_scalar_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_mean: float,
    log_std: float,
) -> torch.Tensor:
    """Use normalized log error for positive labels and a finite zero target."""

    finite = torch.isfinite(target) & (target >= 0.0)
    if not bool(finite.any()):
        return prediction.sum() * 0.0
    per_sample = torch.zeros_like(prediction)
    positive = finite & (target > 0.0)
    if bool(positive.any()):
        per_sample[positive] = (
            (torch.log(prediction[positive]) - log_mean) / log_std
            - (torch.log(target[positive]) - log_mean) / log_std
        ).square()
    zero = finite & (target == 0.0)
    if bool(zero.any()):
        physical_scale = math.exp(log_mean)
        per_sample[zero] = torch.log1p(prediction[zero] / physical_scale).square()
    return per_sample[finite].mean()


def normalized_scalar_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mean: float,
    std: float,
) -> torch.Tensor:
    """Optimize the same raw-space squared error used by official scalar R2."""

    finite = torch.isfinite(target) & (target >= 0.0)
    if not bool(finite.any()):
        return prediction.sum() * 0.0
    return F.mse_loss(
        (prediction[finite] - mean) / std,
        (target[finite] - mean) / std,
    )


def select_target_frame(values: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Select each window's supervised frame (not necessarily its last frame)."""

    indices = batch["target_sequence_index"].long()
    if values.ndim < 2 or len(values) != len(indices):
        raise ValueError("sequence output and target indices are incompatible")
    if torch.any(indices < 0) or torch.any(indices >= values.shape[1]):
        raise IndexError("target sequence index is outside the model window")
    return values[torch.arange(len(values), device=values.device), indices]


def predicted_scalars(
    output,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode either absolute scalars or residuals around the frozen V4 parent."""

    if "base_q95" in batch and "base_betaN" in batch:
        correction = select_target_frame(output.scalar_correction, batch)
        q95 = (batch["base_q95"] + correction[:, 0]).clamp_min(1.0e-4)
        beta = (batch["base_betaN"] + correction[:, 1]).clamp_min(1.0e-4)
        return q95, beta
    return (
        select_target_frame(output.q95, batch),
        select_target_frame(output.betaN, batch),
    )


def encoder_loss(
    output,
    batch: dict[str, torch.Tensor],
    decoder: FrozenSpatialDecoder,
    target_scales: TargetScales,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    predicted_mass = select_target_frame(output.current_mass, batch)
    predicted_harmonic = select_target_frame(output.harmonics, batch)
    predicted_gauge = select_target_frame(output.gauge, batch)
    predicted_q95, predicted_beta = predicted_scalars(output, batch)
    predicted_flux_residual = select_target_frame(
        output.normalized_flux_residual, batch
    )
    target_mass = batch["current_mass"].clamp_min(1.0e-12)
    target_mass = target_mass / target_mass.sum(dim=(-2, -1), keepdim=True)
    log_prediction = torch.log(predicted_mass.clamp_min(1.0e-12))
    current_kl = (
        target_mass * (torch.log(target_mass) - log_prediction)
    ).sum(dim=(-2, -1)).mean()
    moments = F.mse_loss(
        current_moments(predicted_mass), current_moments(target_mass)
    )

    harmonic_mean = predicted_harmonic.new_tensor(target_scales.harmonic_mean)
    harmonic_std = predicted_harmonic.new_tensor(target_scales.harmonic_std)
    harmonic = F.mse_loss(
        (predicted_harmonic - harmonic_mean) / harmonic_std,
        (batch["harmonic"] - harmonic_mean) / harmonic_std,
    )
    gauge = F.mse_loss(
        (predicted_gauge - target_scales.gauge_mean) / target_scales.gauge_std,
        (batch["gauge"] - target_scales.gauge_mean) / target_scales.gauge_std,
    )
    q95 = normalized_scalar_mse(
        predicted_q95,
        batch["q95"],
        target_scales.q95_mean,
        target_scales.q95_std,
    )
    beta = normalized_scalar_mse(
        predicted_beta,
        batch["betaN"],
        target_scales.betaN_mean,
        target_scales.betaN_std,
    )
    decoded = decoder.direct_flux(
        batch["active_psi"],
        batch["ip_abs"],
        predicted_flux_residual,
    )
    target_flux_residual = (
        batch["target_psi"] - batch["active_psi"]
    ) / decoder.flux_scale(batch["ip_abs"])[:, None, None]
    direct_residual = F.mse_loss(
        predicted_flux_residual, target_flux_residual
    )
    flux = normalized_field_mse(decoded, batch["target_psi"])
    gradient = normalized_gradient_mse(decoded, batch["target_psi"])
    if "target_psi_n" in batch:
        topology = normalized_flux_topology_loss(decoded, batch["target_psi_n"])
        geometry = dimensionless_flux_geometry_loss(decoded, batch["target_psi_n"])
    else:
        topology = decoded.sum() * 0.0
        geometry = decoded.sum() * 0.0
    if weights.get("physical_flux", 0.0) > 0.0:
        auxiliary_physical = decoder(
            batch["active_psi"],
            batch["ip_abs"],
            predicted_mass,
            predicted_harmonic,
            predicted_gauge,
        )
        physical_flux = normalized_field_mse(
            auxiliary_physical, batch["target_psi"]
        )
    else:
        physical_flux = decoded.sum() * 0.0
    terms = {
        "current_kl": current_kl,
        "moments": moments,
        "harmonic": harmonic,
        "gauge": gauge,
        "q95": q95,
        "betaN": beta,
        "flux": flux,
        "direct_residual": direct_residual,
        "gradient": gradient,
        "topology": topology,
        "geometry": geometry,
        "physical_flux": physical_flux,
    }
    total = sum(weights.get(name, 0.0) * value for name, value in terms.items())
    return total, {"total": total, **terms}, decoded


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def run_epoch(
    model: CausalHybridEncoderV1,
    loader: DataLoader,
    decoder: FrozenSpatialDecoder,
    target_scales: TargetScales,
    weights: dict[str, float],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    samples = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                output = model(
                    batch["spatial_values"],
                    batch["shared_scalars"],
                    batch["thomson_values"],
                    batch["thomson_attention_mask"],
                    batch["delta_time_ms"],
                    sequence_mask=batch["sequence_mask"],
                )
                loss, terms, _ = encoder_loss(
                    output, batch, decoder, target_scales, weights
                )
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            count = len(batch["spatial_values"])
            samples += count
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * count
    return {key: value / max(samples, 1) for key, value in totals.items()}


def coefficient_of_determination(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.sum((reference - np.mean(reference)) ** 2))
    return 1.0 - float(np.sum((prediction - reference) ** 2)) / max(
        denominator, 1.0e-20
    )


@torch.no_grad()
def evaluate_predictions(
    model: CausalHybridEncoderV1,
    loader: DataLoader,
    decoder: FrozenSpatialDecoder,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "mass",
            "target_mass",
            "harmonic",
            "target_harmonic",
            "gauge",
            "target_gauge",
            "q95",
            "target_q95",
            "betaN",
            "target_betaN",
            "field",
            "target_field",
        )
    }
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model(
            batch["spatial_values"],
            batch["shared_scalars"],
            batch["thomson_values"],
            batch["thomson_attention_mask"],
            batch["delta_time_ms"],
            sequence_mask=batch["sequence_mask"],
        )
        mass = select_target_frame(output.current_mass, batch)
        harmonic = select_target_frame(output.harmonics, batch)
        gauge = select_target_frame(output.gauge, batch)
        field = decoder.direct_flux(
            batch["active_psi"],
            batch["ip_abs"],
            select_target_frame(output.normalized_flux_residual, batch),
        )
        pairs = {
            "mass": mass,
            "target_mass": batch["current_mass"],
            "harmonic": harmonic,
            "target_harmonic": batch["harmonic"],
            "gauge": gauge,
            "target_gauge": batch["gauge"],
            "q95": predicted_scalars(output, batch)[0],
            "target_q95": batch["q95"],
            "betaN": predicted_scalars(output, batch)[1],
            "target_betaN": batch["betaN"],
            "field": field,
            "target_field": batch["target_psi"],
        }
        for key, value in pairs.items():
            collected[key].append(value.detach().cpu().numpy())
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    residual = arrays["field"] - arrays["target_field"]
    centered = arrays["target_field"] - arrays["target_field"].mean(
        axis=(1, 2), keepdims=True
    )
    per_frame_r2 = 1.0 - np.sum(residual**2, axis=(1, 2)) / np.maximum(
        np.sum(centered**2, axis=(1, 2)), 1.0e-20
    )
    current_similarity = 1.0 - 0.5 * np.sum(
        np.abs(arrays["mass"] - arrays["target_mass"]), axis=(1, 2)
    )
    valid_q95 = np.isfinite(arrays["target_q95"]) & (arrays["target_q95"] > 0.0)
    valid_beta = np.isfinite(arrays["target_betaN"]) & (arrays["target_betaN"] > 0.0)
    return {
        "frames": len(per_frame_r2),
        "flux_r2": {
            "aggregate": coefficient_of_determination(
                arrays["target_field"], arrays["field"]
            ),
            "mean_per_frame": float(np.mean(per_frame_r2)),
            "p10_per_frame": float(np.percentile(per_frame_r2, 10.0)),
            "minimum_per_frame": float(np.min(per_frame_r2)),
        },
        "current_similarity_1_minus_tv": {
            "mean": float(np.mean(current_similarity)),
            "p10": float(np.percentile(current_similarity, 10.0)),
            "minimum": float(np.min(current_similarity)),
        },
        "harmonic_rmse": float(
            np.sqrt(np.mean((arrays["harmonic"] - arrays["target_harmonic"]) ** 2))
        ),
        "gauge_rmse_Wb_per_rad": float(
            np.sqrt(np.mean((arrays["gauge"] - arrays["target_gauge"]) ** 2))
        ),
        "q95_r2": coefficient_of_determination(
            arrays["target_q95"][valid_q95], arrays["q95"][valid_q95]
        ),
        "betaN_r2": coefficient_of_determination(
            arrays["target_betaN"][valid_beta], arrays["betaN"][valid_beta]
        ),
        "finite": bool(all(np.isfinite(value).all() for value in arrays.values())),
    }


def initialize_physical_heads(model: CausalHybridEncoderV1, target_scales: TargetScales) -> None:
    nn.init.zeros_(model.current_head[-1].weight)
    nn.init.zeros_(model.current_head[-1].bias)
    for head in (model.harmonic_head, model.gauge_head):
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
    nn.init.zeros_(model.scalar_head[-1].weight)
    q95 = target_scales.q95_mean
    beta = target_scales.betaN_mean
    desired = torch.tensor([q95, beta], dtype=model.scalar_head[-1].bias.dtype)
    raw = desired + torch.log(-torch.expm1(-desired))
    with torch.no_grad():
        model.scalar_head[-1].bias.copy_(raw)
    nn.init.zeros_(model.flux_head[-1].weight)
    nn.init.constant_(
        model.flux_head[-1].bias,
        math.atanh(1.0 / model.config.flux_residual_bound),
    )


def compact_config() -> CausalHybridEncoderConfig:
    return CausalHybridEncoderConfig(
        spatial_widths=(8, 12, 16, 20),
        spatial_output_width=12,
        thomson_width=16,
        thomson_heads=4,
        thomson_ff_width=32,
        scalar_width=8,
        fusion_width=24,
        hidden_size=20,
        dropout=0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--input-report", type=Path, default=DEFAULT_INPUT_REPORT)
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--bridge-report", type=Path, default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-development", type=int)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--current-kl-weight", type=float, default=0.0)
    parser.add_argument("--moment-weight", type=float, default=0.0)
    parser.add_argument("--harmonic-weight", type=float, default=0.0)
    parser.add_argument("--gauge-weight", type=float, default=0.0)
    parser.add_argument("--q95-weight", type=float, default=0.3)
    parser.add_argument("--betaN-weight", type=float, default=0.3)
    parser.add_argument("--flux-weight", type=float, default=1.0)
    parser.add_argument("--direct-residual-weight", type=float, default=0.5)
    parser.add_argument("--gradient-weight", type=float, default=0.15)
    parser.add_argument("--physical-flux-weight", type=float, default=0.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch size, and patience must be positive")
    input_report = json.loads(args.input_report.read_text())
    bridge_report = json.loads(args.bridge_report.read_text())
    if input_report.get("target_columns_loaded") is not False:
        raise RuntimeError("input cache does not prove its target firewall")
    if input_report.get("strictly_causal_interpolation") is not True:
        raise RuntimeError("input cache is not strictly causal")
    if bridge_report.get("mast_targets_used") is not False:
        raise RuntimeError("MAST targets are forbidden")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "model.pt"
    report_path = args.output_dir / "report.json"
    if report_path.exists() and not args.overwrite:
        print(f"completed training report exists at {report_path}; exiting", flush=True)
        return 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    inputs = load_npz(args.inputs)
    bridge = load_npz(args.bridge)
    source_indices = inputs["source_index"].astype(np.int64)
    if not np.array_equal(
        inputs["shot_name"].astype(str), bridge["shot_name"][source_indices].astype(str)
    ):
        raise RuntimeError("input cache and bridge shot names are misaligned")
    split = bridge["split"][source_indices].astype(str)
    train_positions = np.flatnonzero(split == "train")
    development_positions = np.flatnonzero(split == "development")
    if args.limit_train is not None:
        train_positions = train_positions[: args.limit_train]
    if args.limit_development is not None:
        development_positions = development_positions[: args.limit_development]
    if not len(train_positions) or not len(development_positions):
        raise RuntimeError("both training and development samples are required")
    normalizer = fit_input_normalizer(inputs, train_positions)
    target_scales = fit_target_scales(
        bridge, source_indices[train_positions]
    )
    train_dataset = CausalBridgeDataset(
        inputs, bridge, train_positions, normalizer
    )
    development_dataset = CausalBridgeDataset(
        inputs, bridge, development_positions, normalizer
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.loader_workers,
        pin_memory=device.type == "cuda",
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=args.batch_size,
        num_workers=args.loader_workers,
        pin_memory=device.type == "cuda",
    )
    config = compact_config() if args.compact else CausalHybridEncoderConfig()
    model = CausalHybridEncoderV1(config).to(device)
    initialize_physical_heads(model, target_scales)
    decoder = FrozenSpatialDecoder(bridge).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.03
    )
    weights = {
        "current_kl": args.current_kl_weight,
        "moments": args.moment_weight,
        "harmonic": args.harmonic_weight,
        "gauge": args.gauge_weight,
        "q95": args.q95_weight,
        "betaN": args.betaN_weight,
        "flux": args.flux_weight,
        "direct_residual": args.direct_residual_weight,
        "gradient": args.gradient_weight,
        "physical_flux": args.physical_flux_weight,
    }
    history = []
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            decoder,
            target_scales,
            weights,
            device,
            optimizer,
        )
        development_metrics = run_epoch(
            model,
            development_loader,
            decoder,
            target_scales,
            weights,
            device,
            None,
        )
        scheduler.step()
        selection = development_metrics["total"]
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "development": development_metrics,
            }
        )
        if selection < best_loss - 1.0e-6:
            best_loss = selection
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch:03d} train={train_metrics['total']:.6f} "
            f"development={selection:.6f} flux={development_metrics['flux']:.6f} "
            f"best={best_loss:.6f}@{best_epoch}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"early stopping after epoch {epoch}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("training produced no finite checkpoint")
    model.load_state_dict(best_state)
    development_evaluation = evaluate_predictions(
        model, development_loader, decoder, device
    )
    artifact = {
        "model_state": best_state,
        "model_config": asdict(config),
        "model_class": "FEC1.encoder.model.CausalHybridEncoderV1",
        "input_normalizer": normalizer.to_json(),
        "target_scales": target_scales.to_json(),
        "harmonic_fields": torch.from_numpy(bridge["harmonics"].copy()),
        "grid_r_m": torch.from_numpy(bridge["grid_r"].copy()),
        "grid_z_m": torch.from_numpy(bridge["grid_z"].copy()),
        "source_weights_m2": torch.from_numpy(bridge["source_weights"].copy()),
        "best_epoch": best_epoch,
        "loss_weights": weights,
        "input_schema": {
            "spatial_channels": config.spatial_channels,
            "scalar_features": config.scalar_features,
            "thomson_features": config.thomson_features,
            "history_frames": int(inputs["spatial_values"].shape[1]),
        },
        "provenance": {
            "inputs": str(args.inputs.resolve()),
            "inputs_sha256": sha256_file(args.inputs),
            "bridge": str(args.bridge.resolve()),
            "bridge_sha256": sha256_file(args.bridge),
            "strictly_causal": True,
            "mast_inputs_used": False,
            "mast_targets_used": False,
            "frozen_audit_used_for_selection": False,
        },
    }
    torch.save(artifact, checkpoint_path)
    report = {
        "complete": True,
        "experiment": "causal_hybrid_encoder_v1_training",
        "model_class": artifact["model_class"],
        "model_parameters": model.trainable_parameter_count(),
        "device": str(device),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_development_loss": best_loss,
        "epochs_completed": len(history),
        "split_counts": {
            "train_shots": len(train_positions),
            "development_shots": len(development_positions),
        },
        "loss_weights": weights,
        "development_evaluation": development_evaluation,
        "history": history,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "inputs": str(args.inputs.resolve()),
        "inputs_sha256": sha256_file(args.inputs),
        "bridge": str(args.bridge.resolve()),
        "bridge_sha256": sha256_file(args.bridge),
        "strictly_causal": True,
        "target_columns_in_encoder_inputs": False,
        "mast_inputs_used": False,
        "mast_targets_used": False,
        "frozen_d3d_audit_used_for_training_or_selection": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "development_evaluation": development_evaluation,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
