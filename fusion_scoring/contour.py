"""Contour tracing, resampling, and symmetric Hausdorff distance.

`find_contours` traces iso-level curves on a 2-D array and returns each as an (N, 2) array of
fractional (row, col) = (Z-index, R-index) grid coordinates, with closed curves returned with
identical first/last points. We use scikit-image's marching-squares (pinned in the scoring docker
image and required); if it is somehow missing, we fail loudly rather than silently mis-scoring.
"""
from __future__ import annotations

import numpy as np

try:
    from skimage.measure import find_contours as _sk_find_contours
    _HAVE_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAVE_SKIMAGE = False


def find_contours(arr: np.ndarray, level: float) -> list[np.ndarray]:
    """Iso-level contours as a list of (N, 2) fractional (row, col) arrays."""
    if not _HAVE_SKIMAGE:
        raise ImportError(
            "scikit-image is required for LCFS contour extraction but is not installed. "
            "The scoring docker image pins scikit-image==0.25.2; install it before scoring.")
    return _sk_find_contours(np.asarray(arr, dtype=np.float64), level)


# ---- geometry helpers ----------------------------------------------------------------------
def grid_to_physical(contour_rc: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Map fractional (row=Z-index, col=R-index) contour coords to physical (R, Z) metres."""
    rows = contour_rc[:, 0]
    cols = contour_rc[:, 1]
    idxR = np.arange(len(R))
    idxZ = np.arange(len(Z))
    Rp = np.interp(cols, idxR, R)
    Zp = np.interp(rows, idxZ, Z)
    return np.column_stack([Rp, Zp])


def is_closed(contour_rc: np.ndarray, tol: float = 1e-6) -> bool:
    return np.allclose(contour_rc[0], contour_rc[-1], atol=tol)


def encloses_point(contour_rc: np.ndarray, row: float, col: float) -> bool:
    """Even-odd ray-cast: is (row, col) inside the polygon (in grid-index space)?"""
    y, x = row, col
    poly = contour_rc
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = poly[i, 0], poly[i, 1]
        yj, xj = poly[j, 0], poly[j, 1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def resample_closed(pts_xy: np.ndarray, n: int = 512) -> np.ndarray:
    """Resample a closed polygon to `n` points evenly spaced by arc length."""
    p = np.asarray(pts_xy, dtype=np.float64)
    if len(p) < 2:
        return p
    if not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[0]])
    seg = np.sqrt(((np.diff(p, axis=0)) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return p[:1]
    targets = np.linspace(0.0, total, n, endpoint=False)
    Rp = np.interp(targets, s, p[:, 0])
    Zp = np.interp(targets, s, p[:, 1])
    return np.column_stack([Rp, Zp])


def symmetric_hausdorff(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    """max of the two directed Hausdorff distances between point sets a and b (metres)."""
    from scipy.spatial.distance import directed_hausdorff
    d1 = directed_hausdorff(a_xy, b_xy)[0]
    d2 = directed_hausdorff(b_xy, a_xy)[0]
    return float(max(d1, d2))
