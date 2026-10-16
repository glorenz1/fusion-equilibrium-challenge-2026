"""Exact scorer topology for training labels."""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from fec1.paths import ROOT
from fec1.scoring.vendor.common import CONS_SCALARS
from fec1.scoring.vendor.derive import derive_frame
from fec1.scoring.vendor.lcfs import extract_lcfs


@lru_cache(maxsize=1)
def scorer_mask() -> tuple[np.ndarray, np.ndarray]:
    with np.load(ROOT / "fec1/scoring" / "masks" / "d3d_envelope.npz") as saved:
        mask = saved["mask_coarse"].astype(bool)
    return mask, mask.astype(np.float64)


def derive_flux_topology(
    flux: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    axis_sign: float,
    n_iter: int = 22,
    n_points: int = 512,
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    mask, mask_f = scorer_mask()
    values = np.full((len(flux), len(CONS_SCALARS)), np.nan, dtype=np.float64)
    contours: list[np.ndarray | None] = []
    for index, frame in enumerate(np.asarray(flux)):
        contour = extract_lcfs(
            frame,
            r,
            z,
            "DIII-D",
            mask,
            mask_f,
            n_iter=n_iter,
            n_points=n_points,
            axis_sign=axis_sign,
        )
        derived = derive_frame(
            frame,
            r,
            z,
            "DIII-D",
            mask,
            mask_f,
            contour=contour,
            with_li=True,
            axis_sign=axis_sign,
        )
        values[index] = [derived[name] for name in CONS_SCALARS]
        contours.append(contour)
    return values, contours
