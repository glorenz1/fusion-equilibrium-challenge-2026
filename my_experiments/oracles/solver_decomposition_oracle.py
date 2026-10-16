#!/usr/bin/env python3
"""Frozen forward-solver decomposition audit for NeuralGS.

No model parameter is trained here.  The script reuses the exact 200-shot
oracle manifest and the vendored challenge scorer, then asks progressively
whether EFIT-derived or freely solved toroidal-current sources can reproduce
the target equilibrium through NeuralGS's integral Green operator.

The first implemented phase is deliberately deterministic and inexpensive:

* perfect-target scorer closure under both global flux orientations;
* inverse/forward ``J_phi`` closure at source strides 1, 2, and 4;
* nested smoothing, support, sign, and exact-Ip transformations;
* exact challenge scoring and current-conditioning diagnostics at every step.

Later phases consume the saved manifest and phase-one artifacts rather than
resampling the data, so every score remains directly comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from my_experiments.oracles.oracle_inverse_profiles import (
    build_score_context,
    exact_score,
    load_equal_shot_sample,
    stored_flux_from_canonical,
)
from neuralgs.config import NeuralGSConfig
from neuralgs.data import TARGET_AXIS_SIGN, evenly_spaced_frame_indices
from neuralgs.geometry import plasma_green_matrix
from neuralgs.physics import gs_current_from_flux, plasma_window
from neuralgs.training import split_shards


DEFAULT_CACHE = ROOT / "my_experiments/neuralgs_full_data_cache"
DEFAULT_ARTIFACT = ROOT / "my_experiments/artifacts/neuralgs_topology_holdout.pt"
DEFAULT_OUTPUT = ROOT / "my_experiments/solver_decomposition_oracle"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> tuple[NeuralGSConfig, float, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = NeuralGSConfig.from_dict(checkpoint["config"])
    raw_gain = checkpoint["model_state"]["raw_vacuum_gain"].float()
    gain = float(F.softplus(raw_gain))
    return config, gain, checkpoint.get("metadata", {})


def build_manifest(
    cache: Path,
    n_val_shots: int,
    seed: int,
    n_shots: int,
    frames_per_shot: int,
) -> dict:
    _, validation = split_shards(cache, n_val_shots=n_val_shots, seed=seed)
    records = []
    for path in validation[:n_shots]:
        with np.load(path, allow_pickle=False) as archive:
            n_frames = len(archive["ip_abs"])
            indices = evenly_spaced_frame_indices(n_frames, frames_per_shot)
            current_sign = float(np.asarray(archive["canonical_current_sign"]).reshape(-1)[0])
        records.append(
            {
                "shot": path.stem,
                "cache_file": path.name,
                "source_frame_count": n_frames,
                "selected_frame_indices": indices.tolist(),
                "canonical_current_sign": current_sign,
            }
        )
    return {
        "version": 1,
        "cache": str(cache.resolve()),
        "n_val_shots": int(n_val_shots),
        "seed": int(seed),
        "n_shots": len(records),
        "frames_per_shot": int(frames_per_shot),
        "frames": int(sum(len(record["selected_frame_indices"]) for record in records)),
        "target_axis_sign": float(TARGET_AXIS_SIGN["DIII-D"]),
        "records": records,
    }


def validate_loaded_manifest(manifest: dict, groups: list[dict], names: list[str]) -> None:
    expected_names = [record["shot"] for record in manifest["records"]]
    if expected_names != names:
        raise RuntimeError("loaded held-out shots do not match the frozen audit manifest")
    expected_counts = [len(record["selected_frame_indices"]) for record in manifest["records"]]
    actual_counts = [group["stop"] - group["start"] for group in groups]
    if expected_counts != actual_counts:
        raise RuntimeError("loaded frame counts do not match the frozen audit manifest")


@dataclass
class Operator:
    stride: int
    green: torch.Tensor
    source_r: torch.Tensor
    source_z: torch.Tensor
    weights: torch.Tensor

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.source_z), len(self.source_r)


def build_operator(
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    stride: int,
    softening_cells: float,
) -> Operator:
    values = plasma_green_matrix(
        grid_r.numpy(), grid_z.numpy(), stride, softening_cells
    )
    green, source_r, source_z, weights = [torch.from_numpy(value).float() for value in values]
    return Operator(stride, green, source_r, source_z, weights)


def analytic_gauge(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Optimal per-frame unweighted additive flux gauge."""

    return (target - prediction).flatten(1).mean(dim=1)


def forward_current(
    current: torch.Tensor,
    operator: Operator,
    vacuum: torch.Tensor,
    target: torch.Tensor,
    vacuum_gain: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    plasma = current.flatten(1) @ operator.green.transpose(0, 1)
    prediction = plasma.reshape_as(vacuum) + float(vacuum_gain) * vacuum
    gauge = analytic_gauge(prediction, target)
    return prediction + gauge[:, None, None], gauge


def current_diagnostics(current: torch.Tensor, operator: Operator, ip_abs: torch.Tensor) -> dict:
    weights = operator.weights.reshape(1, *operator.shape)
    positive = (torch.relu(current) * weights).sum(dim=(1, 2))
    negative = (torch.relu(-current) * weights).sum(dim=(1, 2))
    net = ((current * weights).sum(dim=(1, 2)))
    weighted_l2 = torch.sqrt((current.square() * weights).sum(dim=(1, 2)).clamp_min(0.0))
    if current.shape[-1] > 1 and current.shape[-2] > 1:
        dr = float(torch.mean(torch.diff(operator.source_r)))
        dz = float(torch.mean(torch.diff(operator.source_z)))
        roughness = torch.sqrt(
            (torch.diff(current, dim=2).square().mean(dim=(1, 2)) / max(dr * dr, 1.0e-12))
            + (torch.diff(current, dim=1).square().mean(dim=(1, 2)) / max(dz * dz, 1.0e-12))
        )
    else:
        roughness = torch.zeros_like(net)

    def summary(values: torch.Tensor) -> dict:
        array = values.detach().cpu().double().numpy()
        return {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95.0)),
            "max": float(np.max(array)),
        }

    scale = ip_abs.clamp_min(1.0)
    return {
        "net_current_A": summary(net),
        "net_over_Ip": summary(net / scale),
        "positive_over_Ip": summary(positive / scale),
        "negative_magnitude_over_Ip": summary(negative / scale),
        "weighted_current_l2": summary(weighted_l2),
        "physical_gradient_rms": summary(roughness),
    }


def normalize_ip(current: torch.Tensor, operator: Operator, ip_abs: torch.Tensor) -> torch.Tensor:
    weights = operator.weights.reshape(1, *operator.shape)
    integral = (current * weights).sum(dim=(1, 2))
    safe = torch.where(
        integral.abs() > 1.0e-8,
        integral,
        torch.ones_like(integral),
    )
    return current * (ip_abs / safe)[:, None, None]


def source_support(
    target_psi_n: torch.Tensor,
    operator: Operator,
    temperature: float,
) -> torch.Tensor:
    sampled = target_psi_n[:, :: operator.stride, :: operator.stride]
    sampled = sampled[:, : operator.shape[0], : operator.shape[1]]
    soft_lcfs = torch.sigmoid((1.0 - sampled) / max(float(temperature), 1.0e-3))
    window = plasma_window(operator.source_r, operator.source_z)
    return soft_lcfs * window[None]


def derived_current_variants(
    target_psi: torch.Tensor,
    target_psi_n: torch.Tensor,
    ip_abs: torch.Tensor,
    grid_r: torch.Tensor,
    grid_z: torch.Tensor,
    operator: Operator,
    lcfs_temperature: float,
) -> list[tuple[str, torch.Tensor]]:
    full_raw = gs_current_from_flux(target_psi, grid_r, grid_z)
    full_smooth = F.avg_pool2d(full_raw[:, None], 5, stride=1, padding=2)[:, 0]

    def sample(field: torch.Tensor) -> torch.Tensor:
        return field[:, :: operator.stride, :: operator.stride][
            :, : operator.shape[0], : operator.shape[1]
        ]

    raw = sample(full_raw)
    smooth = sample(full_smooth)
    support = source_support(target_psi_n, operator, lcfs_temperature)
    supported = smooth * support
    sign_constrained = torch.relu(supported)
    ip_constrained = normalize_ip(sign_constrained, operator, ip_abs)
    return [
        ("raw_signed", raw),
        ("smoothed_signed", smooth),
        ("supported_signed", supported),
        ("supported_nonnegative", sign_constrained),
        ("supported_nonnegative_exact_ip", ip_constrained),
    ]


def score_physics(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
    groups: list[dict],
    score_context: dict,
) -> dict:
    return exact_score(
        prediction.detach().cpu().numpy().astype(np.float64),
        batch["target_q95"].numpy().astype(np.float64),
        batch["target_betaN"].numpy().astype(np.float64),
        groups,
        score_context,
    )


def score_summary(score: dict) -> dict:
    keys = (
        "S", "S_psi", "r2_psi", "r2_qb", "one_minus_dlcfs",
        "dlcfs", "consistency", "lcfs_fail_frac", "cons_fail_frac",
        "r2_cons_each",
    )
    return {key: score[key] for key in keys}


def perfect_closure(
    batch: dict[str, torch.Tensor],
    groups: list[dict],
    score_context: dict,
    tolerance: float,
) -> dict:
    target = batch["target_psi"]
    direct = score_physics(target, batch, groups, score_context)
    flipped = score_physics(-target, batch, groups, score_context)
    required = {
        "S": 1.0,
        "r2_psi": 1.0,
        "r2_qb": 1.0,
        "one_minus_dlcfs": 1.0,
        "consistency": 1.0,
    }
    failures = []
    for orientation, result in (("stored", direct), ("globally_flipped", flipped)):
        for key, wanted in required.items():
            if abs(float(result[key]) - wanted) > tolerance:
                failures.append(f"{orientation}.{key}={result[key]} (wanted {wanted})")
    if failures:
        raise RuntimeError("perfect-psi scorer closure failed: " + "; ".join(failures))
    return {
        "stored_orientation": score_summary(direct),
        "globally_flipped_orientation": score_summary(flipped),
        "tolerance": float(tolerance),
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-val-shots", type=int, default=200)
    parser.add_argument("--n-shots", type=int, default=200)
    parser.add_argument("--frames-per-shot", type=int, default=1)
    parser.add_argument("--strides", default="1,2,4")
    parser.add_argument("--perfect-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--skip-perfect", action="store_true")
    parser.add_argument("--skip-derived", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    report_path = args.output_dir / "closure_report.json"
    predictions_path = args.output_dir / "closure_predictions.pt"
    config, vacuum_gain, artifact_metadata = load_config(args.artifact)
    manifest = build_manifest(
        args.cache,
        args.n_val_shots,
        args.seed,
        args.n_shots,
        args.frames_per_shot,
    )
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise RuntimeError("refusing to change the frozen audit manifest")
    else:
        write_json(manifest_path, manifest)

    batch, groups, names, _ = load_equal_shot_sample(
        args.cache,
        args.n_val_shots,
        args.seed,
        args.n_shots,
        args.frames_per_shot,
    )
    validate_loaded_manifest(manifest, groups, names)
    score_context = build_score_context(batch, groups)
    report = {
        "experiment": "solver_decomposition_closure",
        "architecture_frozen": True,
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": sha256_file(args.artifact),
        "artifact_metadata": artifact_metadata,
        "cache": str(args.cache.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "scorer_version": getattr(__import__("local_score"), "SCORING_VERSION"),
        "vacuum_gain_frozen": vacuum_gain,
        "config": config.to_dict(),
        "weighting": {
            "flux_objective": "uniform full 65x65 grid",
            "current_norm": "trapezoidal dR*dZ quadrature",
            "challenge_scorer": "vendored exact metric",
        },
        "perfect_closure": None,
        "derived_current": [],
    }
    prediction_artifact: dict[str, torch.Tensor | dict] = {
        "manifest": manifest,
        "target_psi": batch["target_psi"],
    }

    if not args.skip_perfect:
        print("Running perfect-psi scorer closure under both global flux signs", flush=True)
        report["perfect_closure"] = perfect_closure(
            batch, groups, score_context, args.perfect_tolerance
        )
        write_json(report_path, report)

    if not args.skip_derived:
        grid_r = batch["grid_r"].float()
        grid_z = batch["grid_z"].float()
        strides = [int(value) for value in args.strides.split(",") if value.strip()]
        for stride in strides:
            print(f"Building stride-{stride} Green operator", flush=True)
            operator = build_operator(
                grid_r, grid_z, stride, config.green_softening_cells
            )
            variants = derived_current_variants(
                batch["target_psi"].float(),
                batch["target_psi_n"].float(),
                batch["ip_abs"].float(),
                grid_r,
                grid_z,
                operator,
                config.lcfs_temperature,
            )
            for variant_name, current in variants:
                print(f"  scoring stride={stride} variant={variant_name}", flush=True)
                prediction, gauge = forward_current(
                    current,
                    operator,
                    batch["vacuum_psi"].float(),
                    batch["target_psi"].float(),
                    vacuum_gain,
                )
                score = score_physics(prediction, batch, groups, score_context)
                entry = {
                    "stride": stride,
                    "variant": variant_name,
                    "source_shape": list(operator.shape),
                    "source_dofs": int(current[0].numel()),
                    "vacuum_gain": vacuum_gain,
                    "gauge": {
                        "mean": float(gauge.mean()),
                        "std": float(gauge.std(correction=0)),
                    },
                    "current": current_diagnostics(
                        current, operator, batch["ip_abs"].float()
                    ),
                    "score": score_summary(score),
                }
                report["derived_current"].append(entry)
                key = f"derived_stride{stride}_{variant_name}"
                prediction_artifact[f"{key}_psirz"] = prediction.half()
                prediction_artifact[f"{key}_current"] = current.half()
                write_json(report_path, report)
                torch.save(prediction_artifact, predictions_path)
                print(
                    f"    S={score['S']:.4f} R2psi={score['r2_psi']:.4f} "
                    f"1-D={score['one_minus_dlcfs']:.4f} C={score['consistency']:.4f}",
                    flush=True,
                )

    report["elapsed_seconds"] = time.monotonic() - started
    write_json(report_path, report)
    torch.save(prediction_artifact, predictions_path)
    print(f"Wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
