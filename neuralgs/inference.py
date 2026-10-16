"""Artifact loading and one-shot challenge inference."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .config import NeuralGSConfig
from .data import preprocess_row
from .geometry import plasma_green_matrix
from .model import NeuralGS
from .scalar_ridge import (
    DEFAULT_SCALAR_ARTIFACT,
    artifact_matches_grid,
    cached_scalar_artifact,
    predict_prepared_scalars,
)

DEFAULT_ARTIFACT = Path(__file__).resolve().parent.parent / "my_experiments" / "artifacts" / "neuralgs.pt"


@lru_cache(maxsize=3)
def _load_model(path: str, device_name: str) -> tuple[NeuralGS, NeuralGSConfig, torch.device]:
    device = torch.device(device_name)
    artifact = torch.load(path, map_location=device, weights_only=False)
    if artifact.get("artifact_version") != 1:
        raise ValueError(f"unsupported NeuralGS artifact version: {artifact.get('artifact_version')}")
    config = NeuralGSConfig.from_dict(artifact["config"])
    model = NeuralGS(config).to(device)
    model.load_state_dict(artifact["model_state"])
    model.eval()
    return model, config, device


@lru_cache(maxsize=4)
def _cached_green(r_bytes: bytes, z_bytes: bytes, stride: int, softening: float):
    r = np.frombuffer(r_bytes, dtype=np.float32).copy()
    z = np.frombuffer(z_bytes, dtype=np.float32).copy()
    return plasma_green_matrix(r, z, stride, softening)


def _device_name() -> str:
    requested = os.environ.get("NEURALGS_DEVICE", "auto")
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def _load_direct_scalar(path: str, device_name: str):
    from my_experiments.neuralgs.neuralgs_direct_scalar import load_direct_scalar_artifact

    return load_direct_scalar_artifact(path, torch.device(device_name))


def predict_row(
    row: dict,
    artifact_path: Path | str = DEFAULT_ARTIFACT,
    batch_size: int = 16,
    use_auxiliary_scalars: bool = True,
    return_internal_coordinates: bool = False,
) -> dict[str, np.ndarray]:
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(
            f"NeuralGS artifact not found at {path}. See docs/NEURALGS.md for the prepare, "
            "pretrain, and finetune commands."
        )
    model, config, device = _load_model(str(path.resolve()), _device_name())
    prepared = preprocess_row(row, config, include_targets=False)
    r = np.asarray(prepared["grid_r"], dtype=np.float32)
    z = np.asarray(prepared["grid_z"], dtype=np.float32)
    green_np, sr_np, sz_np, weights_np = _cached_green(
        r.tobytes(), z.tobytes(), config.source_stride, config.green_softening_cells
    )
    green, source_r, source_z, source_weights, grid_r, grid_z = [
        torch.as_tensor(value, device=device)
        for value in (green_np, sr_np, sz_np, weights_np, r, z)
    ]
    frame_keys = [
        "vacuum_psi", "ip_abs", "global_features", "canonical_global_features",
        "tf_ampere", "tf_turns", "token_coords", "token_values", "token_mask"
    ]
    outputs = {"psirz": [], "q95": [], "betaN": []}
    physical_psi_n: list[np.ndarray] = []
    direct_scalar_path = (
        os.environ.get("NEURALGS_DIRECT_SCALAR_ARTIFACT")
        if use_auxiliary_scalars
        else None
    )
    direct_head = direct_artifact = None
    direct_parts: list[np.ndarray] = []
    if direct_scalar_path:
        from my_experiments.neuralgs.neuralgs_direct_scalar import sha256_file

        resolved_direct = str(Path(direct_scalar_path).resolve())
        direct_head, direct_artifact = _load_direct_scalar(
            resolved_direct, str(device)
        )
        expected_hash = direct_artifact.get("base_artifact_sha256")
        if expected_hash and sha256_file(path) != expected_hash:
            raise ValueError(
                "direct scalar head was trained against a different NeuralGS artifact"
            )
    n_frames = len(prepared["ip_abs"])
    with torch.inference_mode():
        for start in range(0, n_frames, batch_size):
            stop = min(start + batch_size, n_frames)
            batch = {
                key: torch.as_tensor(prepared[key][start:stop], device=device)
                for key in frame_keys
            }
            result = model(batch, green, source_r, source_z, source_weights, grid_r, grid_z)
            for key in outputs:
                outputs[key].append(result[key].detach().cpu().numpy())
            if config.physical_scalar_postprocess or return_internal_coordinates:
                physical_psi_n.append(result["psi_n"].detach().cpu().numpy())
            if direct_head is not None and direct_artifact is not None:
                from my_experiments.neuralgs.neuralgs_direct_scalar import (
                    build_direct_scalar_features,
                    predict_from_features,
                )

                features = build_direct_scalar_features(
                    result,
                    batch,
                    source_r,
                    source_z,
                    source_weights,
                    grid_r,
                    grid_z,
                    config.lcfs_temperature,
                )
                direct_parts.append(
                    predict_from_features(features, direct_head, direct_artifact)
                    .detach()
                    .cpu()
                    .numpy()
                )
    combined = {
        key: np.concatenate(parts, axis=0).astype(np.float32)
        for key, parts in outputs.items()
    }
    scalar_path = (
        os.environ.get("NEURALGS_SCALAR_ARTIFACT")
        if use_auxiliary_scalars
        else None
    )
    if (
        use_auxiliary_scalars
        and scalar_path is None
        and DEFAULT_SCALAR_ARTIFACT.exists()
    ):
        scalar_path = str(DEFAULT_SCALAR_ARTIFACT)
    if direct_parts:
        direct = np.concatenate(direct_parts, axis=0).astype(np.float32)
        combined["q95"] = direct[:, 0]
        combined["betaN"] = direct[:, 1]
    elif scalar_path:
        scalar_artifact_path = Path(scalar_path)
        if not scalar_artifact_path.exists():
            raise FileNotFoundError(
                f"NeuralGS scalar artifact not found at {scalar_artifact_path}"
            )
        scalar_artifact = cached_scalar_artifact(str(scalar_artifact_path.resolve()))
        force_scalar = os.environ.get("NEURALGS_FORCE_SCALAR_RIDGE", "0") == "1"
        if force_scalar or artifact_matches_grid(prepared, scalar_artifact):
            combined.update(predict_prepared_scalars(prepared, scalar_artifact))
    if config.physical_scalar_postprocess:
        from my_experiments.neuralgs.mast_physical_scalars import physics_q95
        from my_experiments.neuralgs.validate_beta_physics import electron_beta_n_proxy

        # NeuralGS canonicalizes both machines to an axis-maximum field. The
        # scorer later handles each machine's storage sign independently.
        q_physical = physics_q95(
            row,
            combined["psirz"],
            numerical_scale=float(config.physical_q_scale),
            axis_sign=1.0,
            use_machine_envelope=False,
        )
        q_fallback = np.asarray(combined["q95"], dtype=np.float64)
        combined["q95"] = np.clip(
            np.where(np.isfinite(q_physical), q_physical, q_fallback),
            0.1,
            30.0,
        ).astype(np.float32)

        psi_n = np.concatenate(physical_psi_n, axis=0).astype(np.float64)
        beta_physical = np.empty(len(psi_n), dtype=np.float64)
        machine = str(row.get("source", "DIII-D"))
        for frame in range(len(psi_n)):
            beta_physical[frame] = electron_beta_n_proxy(
                psi_n[frame],
                prepared["token_coords"][frame],
                prepared["token_values"][frame],
                prepared["token_mask"][frame],
                float(prepared["ip_abs"][frame]),
                float(prepared["tf_ampere"][frame]),
                r,
                z,
                machine=machine,
            )
        beta_physical *= float(config.physical_beta_scale)
        beta_fallback = np.asarray(combined["betaN"], dtype=np.float64)
        combined["betaN"] = np.clip(
            np.where(np.isfinite(beta_physical), beta_physical, beta_fallback),
            0.0,
            10.0,
        ).astype(np.float32)
    if return_internal_coordinates:
        combined["_psi_n"] = np.concatenate(physical_psi_n, axis=0).astype(
            np.float32
        )
    return combined


def predict_with_saved_model(row: dict, artifact_path: Path | str = DEFAULT_ARTIFACT) -> dict[str, np.ndarray]:
    return predict_row(row, artifact_path)
