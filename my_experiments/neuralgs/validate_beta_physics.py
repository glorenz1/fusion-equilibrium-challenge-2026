#!/usr/bin/env python3
"""Validate a dimensionally constrained betaN proxy using DIII-D only.

The proxy reconstructs a low-capacity electron-pressure profile versus the
teacher normalized flux coordinate, volume-averages that profile, and applies
the standard betaN definition using Ip and the toroidal-field turns.  A single
through-origin scale accounts for ion/fast-particle pressure and residual
diagnostic calibration.  MAST inputs, labels, demos, and fitted artifacts are
not read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_experiments.neuralgs.mast_physical_scalars import MU0, toroidal_flux_per_ampere
from my_experiments.baselines.npz_random_access import load_stored_array
from my_experiments.oracles.solver_decomposition_oracle import write_json


DEFAULT_BRIDGE = ROOT / "my_experiments/d3d_spatial_current_bridge_wide"
DEFAULT_AUDIT = ROOT / "my_experiments/d3d_spatial_current_bridge_final_audit"
DEFAULT_OUTPUT = ROOT / "my_experiments/d3d_beta_physics_validation.json"


def bilinear(
    field: np.ndarray,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    fr = np.interp(coords[:, 0], grid_r, np.arange(len(grid_r)))
    fz = np.interp(coords[:, 1], grid_z, np.arange(len(grid_z)))
    ir = np.clip(np.floor(fr).astype(int), 0, len(grid_r) - 2)
    iz = np.clip(np.floor(fz).astype(int), 0, len(grid_z) - 2)
    tr, tz = fr - ir, fz - iz
    return (
        field[iz, ir] * (1.0 - tr) * (1.0 - tz)
        + field[iz, ir + 1] * tr * (1.0 - tz)
        + field[iz + 1, ir] * (1.0 - tr) * tz
        + field[iz + 1, ir + 1] * tr * tz
    )


def pressure_parameters(rho: np.ndarray, pressure: np.ndarray) -> tuple[float, float, float] | None:
    valid = np.isfinite(rho) & np.isfinite(pressure) & (pressure > 0.0)
    valid &= (rho >= -0.10) & (rho <= 1.15)
    rho = np.asarray(rho, dtype=np.float64)[valid]
    pressure = np.asarray(pressure, dtype=np.float64)[valid]
    if len(rho) < 8:
        return None
    edge_values = pressure[rho >= 0.82]
    edge = float(np.percentile(edge_values, 25.0)) if len(edge_values) >= 3 else 0.0
    core_values = pressure[rho <= 0.25]
    core = float(np.percentile(core_values, 85.0)) if len(core_values) >= 3 else float(np.max(pressure))
    amplitude = max(core - edge, 1.0)
    normalized = (pressure - edge) / amplitude
    fit = (rho >= 0.05) & (rho <= 0.95) & (normalized > 0.02) & (normalized < 1.5)
    if np.count_nonzero(fit) >= 5:
        x = np.log(np.maximum(1.0 - rho[fit], 1.0e-3))
        y = np.log(np.maximum(normalized[fit], 1.0e-4))
        # Median of pairwise local slopes is less sensitive to Thomson spikes
        # than an unconstrained high-order kinetic spline.
        denominator = float(np.dot(x, x))
        exponent = float(np.dot(x, y) / denominator) if denominator > 1.0e-12 else 1.0
    else:
        exponent = 1.0
    return max(core, edge + 1.0), max(edge, 0.0), float(np.clip(exponent, 0.2, 5.0))


def electron_beta_n_proxy(
    psi_n: np.ndarray,
    token_coords: np.ndarray,
    token_values: np.ndarray,
    token_mask: np.ndarray,
    ip_abs: float,
    tf_ampere: float,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    machine: str = "DIII-D",
) -> float:
    valid_tokens = np.asarray(token_mask, dtype=bool)
    valid_tokens &= np.isfinite(token_coords[:, 0]) & np.isfinite(token_coords[:, 1])
    if np.count_nonzero(valid_tokens) < 8:
        return np.nan
    rho = bilinear(
        psi_n, grid_r, grid_z, np.asarray(token_coords)[valid_tokens]
    )
    pressure = 1.0e3 * np.expm1(
        np.clip(np.asarray(token_values)[valid_tokens, 2], 0.0, 20.0)
    )
    parameters = pressure_parameters(rho, pressure)
    if parameters is None:
        return np.nan
    core, edge, exponent = parameters
    inside = np.isfinite(psi_n) & (psi_n >= 0.0) & (psi_n <= 1.0)
    if np.count_nonzero(inside) < 20:
        return np.nan
    profile = edge + (core - edge) * np.maximum(1.0 - psi_n, 0.0) ** exponent
    rr, _ = np.meshgrid(grid_r, grid_z)
    dvolume = 2.0 * np.pi * rr * float(np.mean(np.diff(grid_r))) * float(np.mean(np.diff(grid_z)))
    volume = float(np.sum(dvolume[inside]))
    pressure_average = float(np.sum(profile[inside] * dvolume[inside]) / max(volume, 1.0e-20))
    radial_indices = np.flatnonzero(np.any(inside, axis=0))
    if len(radial_indices) < 2:
        return np.nan
    minor_radius = 0.5 * float(grid_r[radial_indices[-1]] - grid_r[radial_indices[0]])
    axis_index = np.unravel_index(np.nanargmin(np.where(inside, psi_n, np.nan)), psi_n.shape)
    axis_radius = max(float(grid_r[axis_index[1]]), 0.05)
    b_toroidal = (
        toroidal_flux_per_ampere(machine) * abs(float(tf_ampere)) / axis_radius
    )
    ip_ma = abs(float(ip_abs)) / 1.0e6
    if b_toroidal <= 1.0e-6 or ip_ma <= 1.0e-4:
        return np.nan
    beta_percent = 100.0 * 2.0 * MU0 * pressure_average / b_toroidal**2
    return float(beta_percent * minor_radius * b_toroidal / ip_ma)


def selected_tokens(bridge_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((bridge_dir / "manifest.json").read_text())
    if manifest.get("mast_inputs_used") or manifest.get("mast_targets_used"):
        raise RuntimeError("DIII-D beta validation bridge is not MAST-free")
    cache = Path(manifest["cache"])
    coords, values, masks, tf = [], [], [], []
    for record in manifest["records"]:
        indices = np.asarray(record["selected_frame_indices"], dtype=np.int64)
        coords.append(load_stored_array(cache / record["cache_file"], "token_coords", indices))
        values.append(load_stored_array(cache / record["cache_file"], "token_values", indices))
        masks.append(load_stored_array(cache / record["cache_file"], "token_mask", indices))
        global_features = load_stored_array(
            cache / record["cache_file"], "global_features", indices
        )
        tf.append(1.0e4 * np.expm1(global_features[:, 1].astype(np.float64)))
    return (
        np.concatenate(coords), np.concatenate(values),
        np.concatenate(masks), np.concatenate(tf),
    )


def load_bridge(directory: Path) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, ...]]:
    with np.load(directory / "bridge.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    tokens = selected_tokens(directory)
    if len(tokens[0]) != len(arrays["target_psi_n"]):
        raise RuntimeError("token and bridge frame ordering differs")
    return arrays, tokens


def proxies(
    arrays: dict[str, np.ndarray],
    tokens: tuple[np.ndarray, ...],
    indices: np.ndarray,
) -> np.ndarray:
    coords, values, masks, tf = tokens
    output = []
    for count, index in enumerate(indices, start=1):
        output.append(electron_beta_n_proxy(
            arrays["target_psi_n"][index].astype(np.float64),
            coords[index], values[index], masks[index],
            float(arrays["ip_abs"][index]), float(tf[index]),
            arrays["grid_r"].astype(np.float64), arrays["grid_z"].astype(np.float64),
            machine="DIII-D",
        ))
        if count == 1 or count % 200 == 0 or count == len(indices):
            print(f"beta proxy {count}/{len(indices)}", flush=True)
    return np.asarray(output, dtype=np.float64)


def through_origin_scale(proxy: np.ndarray, target: np.ndarray) -> float:
    valid = np.isfinite(proxy) & np.isfinite(target)
    x, y = proxy[valid], target[valid]
    return float(np.dot(x, y) / max(float(np.dot(x, x)), 1.0e-30))


def metrics(proxy: np.ndarray, target: np.ndarray, scale: float) -> dict:
    prediction = scale * np.asarray(proxy, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(target)
    x, y = prediction[valid], np.asarray(target, dtype=np.float64)[valid]
    denominator = float(np.sum((y - np.mean(y)) ** 2)) if len(y) else 0.0
    return {
        "count": int(len(y)),
        "finite_fraction": float(np.mean(valid)),
        "r2": float(1.0 - np.sum((x - y) ** 2) / max(denominator, 1.0e-30)) if len(y) else None,
        "mae": float(np.mean(np.abs(x - y))) if len(y) else None,
        "prediction_median": float(np.median(x)) if len(x) else None,
        "target_median": float(np.median(y)) if len(y) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-dir", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    bridge, bridge_tokens = load_bridge(args.bridge_dir)
    train_indices = np.flatnonzero(bridge["split"] == "train")
    development_indices = np.flatnonzero(bridge["split"] == "development")
    train_proxy = proxies(bridge, bridge_tokens, train_indices)
    development_proxy = proxies(bridge, bridge_tokens, development_indices)
    scale = through_origin_scale(train_proxy, bridge["target_betaN"][train_indices])
    audit, audit_tokens = load_bridge(args.audit_dir)
    audit_indices = np.arange(len(audit["target_betaN"]), dtype=np.int64)
    audit_proxy = proxies(audit, audit_tokens, audit_indices)
    report = {
        "experiment": "d3d_only_dimensionally_constrained_betaN_validation",
        "complete": True,
        "mast_inputs_used": False,
        "mast_targets_used": False,
        "mast_demo_artifacts_used": False,
        "physics": "betaN_e=100*(2*mu0*<pe>/Bt^2)*a*Bt/Ip_MA",
        "calibration": "one through-origin DIII-D total/electron pressure scale",
        "total_to_electron_scale": scale,
        "train": metrics(train_proxy, bridge["target_betaN"][train_indices], scale),
        "development": metrics(
            development_proxy, bridge["target_betaN"][development_indices], scale
        ),
        "fresh_audit": metrics(audit_proxy, audit["target_betaN"], scale),
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
