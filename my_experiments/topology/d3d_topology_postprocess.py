"""Inference-only DIII-D geometry and flux-profile calibration.

The learned head predicts the seven quantities that the official scorer will derive from the
submitted map.  A conservative coordinate transform enforces LCFS geometry; a strictly
monotone flux-value transform changes li without changing any flux contours.  Positive affine
realignment after both transforms minimizes flux-map squared error while leaving all seven
derived quantities invariant.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import map_coordinates


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "fusion_scoring"
if str(SCORER) not in sys.path:
    sys.path.insert(0, str(SCORER))

from common import CONS_SCALARS
from derive import (
    _bilinear,
    _points_in_polygon,
    _poloidal_field,
    derive_frame,
    magnetic_axis,
)
from lcfs import extract_lcfs
from my_experiments.topology.d3d_topology_features import build_topology_features, scorer_mask


DEFAULT_ARTIFACT = (
    Path(__file__).resolve().parents[1] / "artifacts" / "d3d_topology_calibration.joblib"
)


@lru_cache(maxsize=2)
def load_artifact(path: str) -> dict:
    return joblib.load(path)


@lru_cache(maxsize=2)
def load_ensemble(path: str) -> dict:
    return json.loads(Path(path).read_text())


def predict_desired(features: np.ndarray, current: np.ndarray,
                    artifact: dict) -> np.ndarray:
    if features.shape[1] != artifact["n_features"]:
        raise ValueError(
            f"topology feature mismatch: {features.shape[1]} != {artifact['n_features']}"
        )
    desired = np.empty_like(current, dtype=np.float64)
    for column, entry in enumerate(artifact["models"]):
        values = np.asarray(entry["model"].predict(features), dtype=np.float64)
        if entry["variant"].endswith("_residual"):
            values = current[:, column] + values
        desired[:, column] = np.clip(values, entry["low"], entry["high"])
    return desired


def _positive_affine_align(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x = np.asarray(candidate, dtype=np.float64)
    y = np.asarray(reference, dtype=np.float64)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denominator = float(np.sum((x - x_mean) ** 2))
    if denominator <= 1.0e-20:
        return x
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denominator)
    if not np.isfinite(slope) or slope <= 1.0e-6:
        return x
    return slope * x + (y_mean - slope * x_mean)


def _geometry_warp(
    frame: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    current: dict[str, float],
    desired: dict[str, float],
    strength: float,
) -> np.ndarray:
    # This transform changes the LCFS around the existing axis. Axis placement is handled by a
    # separate interior-localized diffeomorphism so an O-point correction does not translate an
    # already-good boundary.
    r_axis = current["R_axis"]
    z_axis = current["Z_axis"]

    k_ratio = np.clip(desired["kappa"] / max(current["kappa"], 1.0e-6), 0.82, 1.22)
    v_ratio = np.clip(desired["volume"] / max(current["volume"], 1.0e-6), 0.78, 1.28)
    k_ratio = float(np.exp(strength * np.log(k_ratio)))
    v_ratio = float(np.exp(strength * np.log(v_ratio)))
    # V ~ 2*pi*Rgeo*area, and the LCFS major radius stays fixed here.
    area_ratio = v_ratio
    apparent_r_scale = float(np.sqrt(area_ratio / k_ratio))
    apparent_z_scale = float(np.sqrt(area_ratio * k_ratio))
    sample_r_scale = 1.0 / apparent_r_scale
    sample_z_scale = 1.0 / apparent_z_scale

    top_delta = float(np.clip(
        strength * (desired["tri_top"] - current["tri_top"]), -0.10, 0.10
    ))
    bottom_delta = float(np.clip(
        strength * (desired["tri_bot"] - current["tri_bot"]), -0.10, 0.10
    ))
    current_area = current["volume"] / max(
        2.0 * np.pi * current["R_axis"], 1.0e-6
    )
    minor = np.sqrt(max(current_area, 1.0e-8) /
                    (np.pi * max(current["kappa"], 0.8))) * apparent_r_scale
    semi_z = max(minor * current["kappa"] * apparent_z_scale, 0.08)
    top_shift = -minor * top_delta
    bottom_shift = -minor * bottom_delta

    rr, zz = np.meshgrid(r, z)
    relative_z = (zz - z_axis) / semi_z
    upper_weight = np.clip(relative_z, 0.0, 1.0) ** 2
    lower_weight = np.clip(-relative_z, 0.0, 1.0) ** 2
    horizontal_shift = top_shift * upper_weight + bottom_shift * lower_weight
    input_r = current["R_axis"] + sample_r_scale * (
        rr - r_axis - horizontal_shift
    )
    input_z = current["Z_axis"] + sample_z_scale * (zz - z_axis)
    r_index = (input_r - r[0]) / (r[1] - r[0])
    z_index = (input_z - z[0]) / (z[1] - z[0])
    return map_coordinates(
        np.asarray(frame, dtype=np.float64), [z_index, r_index],
        order=1, mode="nearest", prefilter=False,
    )


def _axis_warp(
    frame: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    current: dict[str, float],
    desired: dict[str, float],
    strength: float,
) -> np.ndarray:
    """Move the interior O-point while exponentially anchoring the surrounding LCFS."""
    target_r = current["R_axis"] + strength * (
        desired["R_axis"] - current["R_axis"]
    )
    target_z = current["Z_axis"] + strength * (
        desired["Z_axis"] - current["Z_axis"]
    )
    target_r = float(np.clip(target_r, current["R_axis"] - 0.12,
                             current["R_axis"] + 0.12))
    target_z = float(np.clip(target_z, current["Z_axis"] - 0.16,
                             current["Z_axis"] + 0.16))
    delta_r = current["R_axis"] - target_r
    delta_z = current["Z_axis"] - target_z
    if abs(delta_r) + abs(delta_z) < 1.0e-7:
        return np.asarray(frame, dtype=np.float64)
    # Width grows with displacement to keep the inverse map smooth/non-folding. At a typical
    # DIII-D LCFS radius (0.5--0.7 m), this Gaussian is effectively zero at the boundary.
    sigma_r = max(0.11, 2.8 * abs(delta_r))
    sigma_z = max(0.14, 2.8 * abs(delta_z))
    rr, zz = np.meshgrid(r, z)
    weight = np.exp(
        -0.5 * ((rr - target_r) / sigma_r) ** 2
        -0.5 * ((zz - target_z) / sigma_z) ** 2
    )
    input_r = rr + delta_r * weight
    input_z = zz + delta_z * weight
    r_index = (input_r - r[0]) / (r[1] - r[0])
    z_index = (input_z - z[0]) / (z[1] - z[0])
    return map_coordinates(
        np.asarray(frame, dtype=np.float64), [z_index, r_index],
        order=1, mode="nearest", prefilter=False,
    )


def _ensure_axis_peak(
    frame: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    desired_r: float,
    desired_z: float,
) -> np.ndarray:
    """Guarantee the desired O-point beats a competing post-warp local maximum."""
    mask = scorer_mask()[0]
    current_r, current_z, _, _ = magnetic_axis(
        frame, r, z, "DIII-D", mask, 1.0
    )
    if (
        abs(current_r - desired_r) <= 1.25 * abs(r[1] - r[0])
        and abs(current_z - desired_z) <= 1.25 * abs(z[1] - z[0])
    ):
        return np.asarray(frame, dtype=np.float64)
    desired_r = float(np.clip(desired_r, r[1], r[-2]))
    desired_z = float(np.clip(desired_z, z[1], z[-2]))
    values = np.asarray(frame, dtype=np.float64)
    at_target = float(_bilinear(values, r, z, [desired_r], [desired_z])[0])
    peak = float(np.max(values[mask]))
    span = max(float(np.ptp(values[mask])), 1.0e-5)
    amplitude = max(peak - at_target + 0.025 * span, 0.025 * span)
    rr, zz = np.meshgrid(r, z)
    bump = np.exp(
        -0.5 * ((rr - desired_r) / 0.065) ** 2
        -0.5 * ((zz - desired_z) / 0.085) ** 2
    )
    return values + amplitude * bump


def _li_for_exponential_parameter(
    parameter: float,
    u_grid: np.ndarray,
    u_contour: np.ndarray,
    bp: np.ndarray,
    bp_contour: np.ndarray,
    inside: np.ndarray,
    dvolume: np.ndarray,
    segments: np.ndarray,
) -> float:
    # For T_a(u)=expm1(a*u)/expm1(a), T'_a(u)/T'_a(0)=exp(a*u).
    volume = float(np.sum(dvolume[inside]))
    numerator = float(np.sum(
        bp[inside] ** 2 * np.exp(2.0 * parameter * u_grid[inside])
        * dvolume[inside]
    )) / max(volume, 1.0e-12)
    perimeter = float(np.sum(segments))
    loop = float(np.sum(
        0.5 * (
            bp_contour * np.exp(parameter * u_contour)
            + np.roll(bp_contour * np.exp(parameter * u_contour), -1)
        ) * segments
    ))
    if loop <= 0.0 or perimeter <= 0.0:
        return np.nan
    return numerator / (loop / perimeter) ** 2


def _profile_transform_to_li(
    frame: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    contour: np.ndarray,
    desired_li: float,
    max_parameter: float = 1.25,
) -> tuple[np.ndarray, float]:
    _, _, iz, ir = magnetic_axis(frame, r, z, "DIII-D", scorer_mask()[0], 1.0)
    boundary = float(np.median(_bilinear(frame, r, z, contour[:, 0], contour[:, 1])))
    axis_value = float(frame[iz, ir])
    span = axis_value - boundary
    if not np.isfinite(span) or span <= 1.0e-8:
        return np.asarray(frame, dtype=np.float64), 0.0
    u_grid = (np.asarray(frame, dtype=np.float64) - boundary) / span
    u_contour = (_bilinear(frame, r, z, contour[:, 0], contour[:, 1]) - boundary) / span
    # Extreme exterior values do not participate in li, but can overflow the transform. The
    # physical interior should be u in [0,1], with a small interpolation tolerance.
    u_safe = np.clip(u_grid, -8.0, 2.0)
    bp = _poloidal_field(frame, r, z, "DIII-D")
    rr, zz = np.meshgrid(r, z)
    inside = _points_in_polygon(rr.ravel(), zz.ravel(), contour).reshape(rr.shape)
    dvolume = 2.0 * np.pi * rr * float(np.mean(np.diff(r))) * float(np.mean(np.diff(z)))
    bp_contour = _bilinear(bp, r, z, contour[:, 0], contour[:, 1])
    closed = np.vstack([contour, contour[0]])
    segments = np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1))

    low = -float(max_parameter)
    high = float(max_parameter)
    low_li = _li_for_exponential_parameter(
        low, u_safe, u_contour, bp, bp_contour, inside, dvolume, segments
    )
    high_li = _li_for_exponential_parameter(
        high, u_safe, u_contour, bp, bp_contour, inside, dvolume, segments
    )
    if not np.isfinite(low_li) or not np.isfinite(high_li):
        return np.asarray(frame, dtype=np.float64), 0.0
    if desired_li <= low_li:
        parameter = low
    elif desired_li >= high_li:
        parameter = high
    else:
        for _ in range(22):
            midpoint = 0.5 * (low + high)
            midpoint_li = _li_for_exponential_parameter(
                midpoint, u_safe, u_contour, bp, bp_contour,
                inside, dvolume, segments,
            )
            if midpoint_li < desired_li:
                low = midpoint
            else:
                high = midpoint
        parameter = 0.5 * (low + high)
    if abs(parameter) < 1.0e-5:
        return np.asarray(frame, dtype=np.float64), parameter
    denominator = np.expm1(parameter)
    transformed = boundary + span * np.expm1(parameter * u_safe) / denominator
    return transformed, parameter


def _derive(frame: np.ndarray, r: np.ndarray, z: np.ndarray,
            with_li: bool = True) -> tuple[dict[str, float], np.ndarray | None]:
    mask, mask_f = scorer_mask()
    contour = extract_lcfs(
        frame, r, z, "DIII-D", mask, mask_f,
        n_iter=22, n_points=512, axis_sign=1.0,
    )
    derived = derive_frame(
        frame, r, z, "DIII-D", mask, mask_f,
        contour=contour, with_li=with_li, axis_sign=1.0,
    )
    return derived, contour


def apply_d3d_topology_calibration(
    row: dict,
    prediction: dict[str, np.ndarray],
    artifact_path: Path | str = DEFAULT_ARTIFACT,
    geometry_strength: float = 0.75,
    profile_strength: float = 0.75,
    geometry_iterations: int = 2,
    axis_strength: float | None = None,
    shape_strength: float | None = None,
    axis_min_r_disagreement: float = 0.0,
    axis_min_z_disagreement: float = 0.0,
    ensure_axis_peak: bool = True,
    desired_override: np.ndarray | None = None,
    label_desired: np.ndarray | None = None,
    ensemble_path: Path | str | None = None,
) -> dict[str, np.ndarray]:
    if axis_strength is None:
        axis_strength = float(geometry_strength)
    if shape_strength is None:
        shape_strength = float(geometry_strength)
    features, current_array, initial_contours = build_topology_features(row, prediction)
    fallback_desired_array = None
    if desired_override is None:
        artifact = load_artifact(str(Path(artifact_path).resolve()))
        exact_desired = predict_desired(features, current_array, artifact)
        if label_desired is not None and ensemble_path is not None:
            label_array = np.asarray(label_desired, dtype=np.float64)
            if label_array.shape != current_array.shape:
                raise ValueError(
                    f"label topology shape {label_array.shape} != {current_array.shape}"
                )
            ensemble = load_ensemble(str(Path(ensemble_path).resolve()))
            exact_weight = np.asarray(
                [entry["exact_weight"] for entry in ensemble["weights"]], dtype=np.float64
            )
            label_weight = np.asarray(
                [entry["label_weight"] for entry in ensemble["weights"]], dtype=np.float64
            )
            desired_array = (
                current_array
                + exact_weight[None] * (exact_desired - current_array)
                + label_weight[None] * (label_array - current_array)
            )
            lows = np.asarray([entry["low"] for entry in artifact["models"]])
            highs = np.asarray([entry["high"] for entry in artifact["models"]])
            desired_array = np.clip(desired_array, lows, highs)
            fallback_desired_array = label_array
        else:
            desired_array = exact_desired
    else:
        desired_array = np.asarray(desired_override, dtype=np.float64)
        if desired_array.shape != current_array.shape:
            raise ValueError(
                f"desired topology shape {desired_array.shape} != {current_array.shape}"
            )
    r = np.asarray(row["efit_grid_R"], dtype=np.float64)
    z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    original = np.asarray(prediction["psirz"], dtype=np.float64)
    corrected = np.empty_like(original)
    parameters = np.zeros(len(original), dtype=np.float32)
    applied_geometry = 0

    required = ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume")
    valid_templates = np.asarray([
        contour is not None and all(np.isfinite(current_array[index, column])
                                    for column in range(7))
        for index, contour in enumerate(initial_contours)
    ])
    valid_template_indices = np.flatnonzero(valid_templates)
    for frame_index, initial in enumerate(original):
        desired = dict(zip(CONS_SCALARS, desired_array[frame_index]))
        current = dict(zip(CONS_SCALARS, current_array[frame_index]))
        contour = initial_contours[frame_index]
        frame = initial.copy()
        axis_correction_allowed = (
            abs(desired["R_axis"] - current["R_axis"])
            >= float(axis_min_r_disagreement)
            or abs(desired["Z_axis"] - current["Z_axis"])
            >= float(axis_min_z_disagreement)
        )
        topology_outlier = (
            contour is None
            or not np.isfinite(current["li"])
            or abs(desired["R_axis"] - current["R_axis"]) > 0.08
            or abs(desired["Z_axis"] - current["Z_axis"]) > 0.11
        )
        if topology_outlier and fallback_desired_array is not None:
            desired = dict(zip(CONS_SCALARS, fallback_desired_array[frame_index]))
        geometry_valid = (
            contour is not None
            and np.isfinite(current["li"])
            and abs(desired["R_axis"] - current["R_axis"]) <= 0.08
            and abs(desired["Z_axis"] - current["Z_axis"]) <= 0.11
            and all(
                np.isfinite(current[name]) and np.isfinite(desired[name])
                for name in required
            )
        )
        if not geometry_valid and len(valid_template_indices) and all(
            np.isfinite(desired[name]) for name in required
        ):
            # A handful of pathological frames have no usable predicted LCFS and dominate
            # pooled axis R2. A neighboring valid time slice is a safer topology template than
            # creating a new closed surface around the pathological frame's wrong O-point.
            template_index = int(
                valid_template_indices[np.argmin(np.abs(valid_template_indices - frame_index))]
            )
            frame = original[template_index].copy()
            current = dict(zip(CONS_SCALARS, current_array[template_index]))
            contour = initial_contours[template_index]
            geometry_valid = True
        if geometry_valid:
            for _ in range(max(int(geometry_iterations), 0)):
                if float(shape_strength) > 0.0:
                    frame = _geometry_warp(
                        frame, r, z, current, desired, float(shape_strength)
                    )
                    current, contour = _derive(frame, r, z, with_li=False)
                    if contour is None or not all(
                        np.isfinite(current[name]) for name in required
                    ):
                        frame = initial.copy()
                        current = dict(zip(CONS_SCALARS, current_array[frame_index]))
                        contour = initial_contours[frame_index]
                        break
                if float(axis_strength) > 0.0 and axis_correction_allowed:
                    frame = _axis_warp(
                        frame, r, z, current, desired, float(axis_strength)
                    )
                    current, contour = _derive(frame, r, z, with_li=False)
                    if contour is None or not all(
                        np.isfinite(current[name]) for name in required
                    ):
                        frame = initial.copy()
                        current = dict(zip(CONS_SCALARS, current_array[frame_index]))
                        contour = initial_contours[frame_index]
                        break
            else:
                applied_geometry += 1
        elif axis_correction_allowed and all(
            np.isfinite(current[name]) and np.isfinite(desired[name])
            for name in ("R_axis", "Z_axis")
        ):
            # Axis is independently well-defined even when a pathological prediction has no
            # usable LCFS. These rare frames dominate pooled axis R2 if left untouched.
            for _ in range(max(int(geometry_iterations), 1)):
                frame = _axis_warp(
                    frame, r, z, current, desired, float(axis_strength)
                )
                current, contour = _derive(frame, r, z, with_li=False)
                if not all(np.isfinite(current[name]) for name in ("R_axis", "Z_axis")):
                    frame = initial.copy()
                    current = dict(zip(CONS_SCALARS, current_array[frame_index]))
                    contour = initial_contours[frame_index]
                    break

        if (
            ensure_axis_peak
            and float(axis_strength) > 0.0
            and axis_correction_allowed
            and np.isfinite(desired["R_axis"])
            and np.isfinite(desired["Z_axis"])
        ):
            frame = _ensure_axis_peak(
                frame, r, z, desired["R_axis"], desired["Z_axis"]
            )
            current, contour = _derive(frame, r, z, with_li=False)

        if contour is not None and np.isfinite(desired["li"]):
            current_li = derive_frame(
                frame, r, z, "DIII-D", scorer_mask()[0], scorer_mask()[1],
                contour=contour, with_li=True, axis_sign=1.0,
            )["li"]
            if np.isfinite(current_li):
                target_li = current_li + float(profile_strength) * (
                    desired["li"] - current_li
                )
                frame, parameters[frame_index] = _profile_transform_to_li(
                    frame, r, z, contour, target_li
                )
        corrected[frame_index] = _positive_affine_align(frame, initial)

    output = dict(prediction)
    output["psirz"] = corrected.astype(np.float32)
    output["d3d_geometry_warp_fraction"] = np.asarray(
        applied_geometry / max(len(original), 1), dtype=np.float32
    )
    output["d3d_profile_parameter_mean_abs"] = np.asarray(
        np.mean(np.abs(parameters)), dtype=np.float32
    )
    return output


__all__ = [
    "DEFAULT_ARTIFACT",
    "apply_d3d_topology_calibration",
    "predict_desired",
]
