"""Release-input alignment repairs retained by the original preprocessor."""
from __future__ import annotations
import numpy as np


_BREAKDOWN_WINDOW_MS = (-500.0, 1000.0)


def fix_d3d_ip_times(row) -> np.ndarray:
    """Return the corrected time axis for ``magnetics_plasma_current`` (DIII-D).

    Accepts a Hugging Face dataset row (dict) or a pandas row from a demo
    parquet. It needs only input columns, so it works identically on the
    train and the inputs-only test splits.

    Per shot, this keeps the shipped ``magnetics_plasma_current_times`` if it
    already places plasma breakdown (first |Ip| crossing of 10% of the shot's
    peak) near t = 0, and otherwise re-origins the axis to start at
    ``magnetics_time[0]``, which is where the affected shots' Ip acquisition
    actually began. Validated on all 9,113 DIII-D source shots: the decision
    is unambiguous on every one, correcting 6,322 shots (4,930 of 7,041 in
    ``diii_d_train``, 586 of 874 in ``diii_d_public_test``) and leaving the
    already-correct 2,791 untouched.

    Do NOT blanket-shift every shot instead -- ~31% of shots ship with the
    correct axis, and shifting those would corrupt them.
    """
    ipt = np.asarray(row["magnetics_plasma_current_times"], dtype=np.float64)
    ip = np.abs(np.asarray(row["magnetics_plasma_current"], dtype=np.float64))
    threshold = 0.10 * np.nanpercentile(ip, 99.5)
    breakdown = float(ipt[int(np.argmax(ip > threshold))])
    lo, hi = _BREAKDOWN_WINDOW_MS
    if lo <= breakdown <= hi:
        return ipt
    return float(row["magnetics_time"][0]) + (ipt - ipt[0])


def align_mast_thomson_core(row) -> tuple:
    """Return ``(R, Te, ne)`` for a MAST row with the ghost channel dropped.

    MAST rows ship 130 coordinates in ``thomson_core_R`` but 131 channels in
    ``thomson_core_Te`` / ``thomson_core_ne``; the extra data channel is
    channel 0, which is NaN in every shot. This drops it, so column ``j`` of
    the returned ``Te`` / ``ne`` (shape ``(T, 130)``) sits at radius ``R[j]``.
    Rows whose counts already match (all DIII-D shots) pass through unchanged.
    """
    R = np.asarray(row["thomson_core_R"], dtype=np.float64)
    Te = np.stack([np.asarray(p) for p in row["thomson_core_Te"]])
    ne = np.stack([np.asarray(p) for p in row["thomson_core_ne"]])
    if Te.shape[1] == R.shape[0]:
        return R, Te, ne
    if Te.shape[1] != R.shape[0] + 1:
        raise ValueError(
            f"unexpected Thomson core layout: {R.shape[0]} coordinates vs "
            f"{Te.shape[1]} data channels"
        )
    return R, Te[:, 1:], ne[:, 1:]
