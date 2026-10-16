"""Large-corpus DIII-D topology head using frozen NeuralGS solved-state features."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn


NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
DEFAULT_ARTIFACT = (
    Path(__file__).resolve().parents[1] / "artifacts" / "d3d_label_topology_head.pt"
)


class D3DTopologyHead(nn.Module):
    def __init__(self, n_features: int, hidden_width: int = 192,
                 bottleneck_width: int = 96, dropout: float = 0.04):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, hidden_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, bottleneck_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_width, len(NAMES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@lru_cache(maxsize=2)
def load_artifact(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def predict_topology_from_direct_features(
    direct_features: torch.Tensor | np.ndarray,
    artifact_path: Path | str = DEFAULT_ARTIFACT,
    batch_size: int = 8192,
    apply_calibration: bool = True,
) -> np.ndarray:
    artifact = load_artifact(str(Path(artifact_path).resolve()))
    config = artifact["head_config"]
    head = D3DTopologyHead(**config)
    head.load_state_dict(artifact["head_state"])
    head.eval()
    features = torch.as_tensor(direct_features, dtype=torch.float32)
    active = torch.as_tensor(artifact["active_feature_indices"], dtype=torch.long)
    mean = torch.as_tensor(artifact["feature_mean"], dtype=torch.float32)
    scale = torch.as_tensor(artifact["feature_scale"], dtype=torch.float32)
    target_mean = torch.as_tensor(artifact["target_mean"], dtype=torch.float32)
    target_scale = torch.as_tensor(artifact["target_scale"], dtype=torch.float32)
    parts = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            x = ((features[start : start + batch_size, active] - mean) / scale).clamp(-12, 12)
            parts.append(head(x) * target_scale + target_mean)
    prediction = torch.cat(parts).numpy().astype(np.float64)
    if apply_calibration:
        matrix = np.asarray(artifact["calibration_matrix"], dtype=np.float64)
        intercept = np.asarray(artifact["calibration_intercept"], dtype=np.float64)
        prediction = prediction @ matrix.T + intercept
    return np.clip(
        prediction,
        np.asarray(artifact["target_low"], dtype=np.float64),
        np.asarray(artifact["target_high"], dtype=np.float64),
    )


__all__ = [
    "D3DTopologyHead",
    "DEFAULT_ARTIFACT",
    "NAMES",
    "predict_topology_from_direct_features",
]
