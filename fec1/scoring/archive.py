#!/usr/bin/env python3
"""Score an aligned FEC1 prediction archive against development cache targets."""

from __future__ import annotations

from fec1.paths import CACHE

from fec1.paths import ROOT

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import pickle

import numpy as np


from fec1.data.features import _read


from fec1.scoring.vendor.common import N_CONS, N_SCALARS, PSI_SIGNS, SCALARS  # noqa: E402
from fec1.scoring.reference import build_reference, psi_residuals, score_shot  # noqa: E402
from fec1.scoring.vendor.metrics import Accum, finalize_machine  # noqa: E402


def _reference_task(task):
    selected, psi, q95, beta_n, grid_r, grid_z, mask_coarse, mask_f = task
    gt = {"psi": psi, "q95": q95, "betaN": beta_n}
    return selected, gt, build_reference(psi, grid_r, grid_z, mask_coarse, mask_f)


def _score_task(task):
    gt, reference, prediction, grid_r, grid_z, mask_coarse, mask_f, sign, means = task
    return score_shot(gt, reference, prediction, grid_r, grid_z, mask_coarse, mask_f, sign, means)


def _reference_cache_key(values: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    key = []
    names = values["shot_name"].astype(str)
    frames = values["source_index"].astype(np.int64)
    for shot_index in np.unique(values["shot_index"]):
        selected = np.flatnonzero(values["shot_index"] == shot_index)
        unique_names = np.unique(names[selected])
        if len(unique_names) != 1:
            raise RuntimeError("a numeric shot group contains multiple shot names")
        key.append((str(unique_names[0]), frames[selected].copy()))
    return key


def _cache_matches(
    cached: object,
    key: list[tuple[str, np.ndarray]],
    source: str,
) -> bool:
    if not isinstance(cached, dict):
        return False
    if cached.get("version") != 1 or cached.get("reference_source") != source:
        return False
    saved_key = cached.get("key")
    references = cached.get("references")
    if not isinstance(saved_key, list) or not isinstance(references, list):
        return False
    if len(saved_key) != len(key) or len(references) != len(key):
        return False
    return all(
        saved_name == name and np.array_equal(np.asarray(saved_frames, dtype=np.int64), frames)
        for (saved_name, saved_frames), (name, frames) in zip(saved_key, key)
    )


def _cached_population_size(cached: object, source: str) -> int:
    """Return cached shot count, or zero for an invalid/different cache."""
    if not isinstance(cached, dict):
        return 0
    if cached.get("version") != 1 or cached.get("reference_source") != source:
        return 0
    saved_key = cached.get("key")
    references = cached.get("references")
    if not isinstance(saved_key, list) or not isinstance(references, list):
        return 0
    if len(saved_key) != len(references):
        return 0
    return len(saved_key)


def _reference_groups(
    tasks: list[tuple],
    values: dict[str, np.ndarray],
    workers: int,
    cache_path: Path | None,
) -> list[tuple]:
    if cache_path is None:
        if workers == 1:
            return [_reference_task(task) for task in tasks]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_reference_task, tasks, chunksize=1))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_name(cache_path.name + ".lock")
    key = _reference_cache_key(values)
    source = str(values["reference_source"])
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        cached = None
        if cache_path.is_file():
            try:
                with cache_path.open("rb") as stream:
                    cached = pickle.load(stream)
            except (EOFError, OSError, pickle.UnpicklingError):
                cached = None
        if _cache_matches(cached, key, source):
            print(f"loaded official reference cache {cache_path}", flush=True)
            references = cached["references"]
        else:
            if workers == 1:
                groups = [_reference_task(task) for task in tasks]
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    groups = list(executor.map(_reference_task, tasks, chunksize=1))
            references = [reference for _, _, reference in groups]
            cached_size = _cached_population_size(cached, source)
            if cached_size <= len(key):
                payload = {
                    "version": 1,
                    "reference_source": source,
                    "key": key,
                    "references": references,
                }
                temporary = cache_path.with_name(f"{cache_path.name}.tmp-{os.getpid()}")
                try:
                    with temporary.open("wb") as stream:
                        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, cache_path)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                print(f"wrote official reference cache {cache_path}", flush=True)
            else:
                print(
                    f"preserved larger {cached_size}-shot reference cache "
                    f"while scoring {len(key)} shots",
                    flush=True,
                )
    groups = []
    for task, reference in zip(tasks, references):
        selected, psi, q95, beta_n, *_ = task
        gt = {"psi": psi, "q95": q95, "betaN": beta_n}
        groups.append((selected, gt, reference))
    return groups


def _parallel_official_score(
    values: dict[str, np.ndarray],
    workers: int,
    reference_cache: Path | None = None,
    details_path: Path | None = None,
) -> dict[str, object]:
    mask = np.load(ROOT / "fec1/scoring/masks/d3d_envelope.npz")
    grid_r = mask["grid_R"]
    grid_z = mask["grid_Z"]
    mask_coarse = mask["mask_coarse"].astype(bool)
    mask_f = mask_coarse.astype(np.float64)
    tasks = []
    for shot_index in np.unique(values["shot_index"]):
        selected = np.flatnonzero(values["shot_index"] == shot_index)
        tasks.append(
            (
                selected,
                -values["target_field"][selected].astype(np.float64),
                values["target_q95"][selected].astype(np.float64),
                values["target_betaN"][selected].astype(np.float64),
                grid_r,
                grid_z,
                mask_coarse,
                mask_f,
            )
        )
    groups = _reference_groups(tasks, values, workers, reference_cache)

    psi_sum = psi_sumsq = 0.0
    psi_count = 0
    scalar_sum = np.zeros(N_SCALARS)
    scalar_sumsq = np.zeros(N_SCALARS)
    scalar_count = np.zeros(N_SCALARS)
    consistency_sum = np.zeros(N_CONS)
    consistency_sumsq = np.zeros(N_CONS)
    consistency_count = np.zeros(N_CONS)
    for _, gt, reference in groups:
        psi = gt["psi"]
        finite = np.isfinite(psi)
        psi_sum += float(psi[finite].sum())
        psi_sumsq += float(np.square(psi[finite]).sum())
        psi_count += int(finite.sum())
        for index, name in enumerate(SCALARS):
            scalar = gt[name][np.isfinite(gt[name])]
            scalar_sum[index] += scalar.sum()
            scalar_sumsq[index] += np.square(scalar).sum()
            scalar_count[index] += len(scalar)
        consistency, valid = reference[1], reference[2]
        for index in range(N_CONS):
            scalar = consistency[valid[:, index], index]
            consistency_sum[index] += scalar.sum()
            consistency_sumsq[index] += np.square(scalar).sum()
            consistency_count[index] += len(scalar)
    reference_stats = {
        "psi_sum": psi_sum,
        "psi_sumsq": psi_sumsq,
        "psi_n": psi_count,
        "scal_sum": scalar_sum,
        "scal_sumsq": scalar_sumsq,
        "scal_n": scalar_count,
        "cons_sum": consistency_sum,
        "cons_sumsq": consistency_sumsq,
        "cons_n": consistency_count,
    }
    means = (
        psi_sum / max(psi_count, 1),
        np.divide(scalar_sum, np.maximum(scalar_count, 1)),
        np.divide(consistency_sum, np.maximum(consistency_count, 1)),
    )
    sign_totals = {sign: 0.0 for sign in PSI_SIGNS}
    for selected, gt, _ in groups:
        prediction = {
            "psirz": -values["field"][selected].astype(np.float64),
            "q95": values["q95"][selected].astype(np.float64),
            "betaN": values["betaN"][selected].astype(np.float64),
        }
        residuals, _, _ = psi_residuals(gt["psi"], prediction["psirz"], means[0])
        for sign in PSI_SIGNS:
            sign_totals[sign] += residuals[sign]
    psi_sign = min(PSI_SIGNS, key=lambda sign: sign_totals[sign])
    accumulator = Accum()
    accumulator.psi_sign = psi_sign

    def score_tasks():
        for selected, gt, reference in groups:
            prediction = {
                "psirz": -values["field"][selected].astype(np.float64),
                "q95": values["q95"][selected].astype(np.float64),
                "betaN": values["betaN"][selected].astype(np.float64),
            }
            yield (
                gt,
                reference,
                prediction,
                grid_r,
                grid_z,
                mask_coarse,
                mask_f,
                psi_sign,
                means,
            )

    details = []
    if workers == 1:
        for task in score_tasks():
            part = _score_task(task)
            accumulator.add(part)
            if details_path is not None:
                details.append(part)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for part in executor.map(_score_task, score_tasks(), chunksize=1):
                accumulator.add(part)
                if details_path is not None:
                    details.append(part)
    result = finalize_machine(accumulator, reference_stats)
    if details_path is not None:
        details_path = Path(details_path)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            details_path,
            shot_name=np.asarray([values["shot_name"][selected[0]] for selected, _, _ in groups]),
            frame_count=np.asarray([len(selected) for selected, _, _ in groups]),
            ss_res_psi=np.asarray([part["ss_res_psi"][psi_sign] for part in details]),
            ss_res_scal=np.asarray([part["ss_res_scal"] for part in details]),
            ss_res_cons=np.asarray([part["ss_res_cons"] for part in details]),
            shot_dlcfs=np.asarray(
                [np.nan if part["shot_dlcfs"] is None else part["shot_dlcfs"] for part in details]
            ),
            cons_fails=np.asarray([part["cons_fails"] for part in details]),
            psi_sign=np.asarray(psi_sign),
            **{f"ref_{key}": np.asarray(value) for key, value in reference_stats.items()},
        )
    return {
        key: (
            value.item()
            if isinstance(value, np.generic)
            else (
                {
                    subkey: (subvalue.item() if isinstance(subvalue, np.generic) else subvalue)
                    for subkey, subvalue in value.items()
                }
                if isinstance(value, dict)
                else value
            )
        )
        for key, value in result.items()
    }


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    total = np.square(y - y.mean()).sum(dtype=np.float64)
    return float(1.0 - np.square(y - p).sum(dtype=np.float64) / max(total, 1.0e-30))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--max-shots", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--reference-cache",
        type=Path,
        help="optional reusable cache of aligned target-side official references",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.predictions, allow_pickle=False) as saved:
        prediction = {
            key: np.asarray(saved[key])
            for key in ("sample_index", "shot_name", "frame_index", "psirz", "q95", "betaN")
        }
        if "frozen_audit_used" in saved and saved["frozen_audit_used"].item():
            raise RuntimeError("prediction archive used frozen audit")
    order = np.argsort(prediction["sample_index"].astype(np.int64))
    prediction = {key: value[order] for key, value in prediction.items()}
    names = prediction["shot_name"].astype(str)
    shots = np.unique(names)
    if args.max_shots > 0:
        shots = shots[: args.max_shots]
        keep = np.isin(names, shots)
        prediction = {key: value[keep] for key, value in prediction.items()}
        names = prediction["shot_name"].astype(str)

    targets = []
    target_q = []
    target_beta = []
    shot_index = np.empty(len(names), dtype=np.int64)
    source_index = prediction["frame_index"].astype(np.int64)
    for index, shot in enumerate(shots):
        positions = np.flatnonzero(names == shot)
        frames = source_index[positions]
        data = _read(args.cache_dir / f"{shot}.npz", include_targets=True)
        targets.append(data["target_psi"][frames].astype(np.float32))
        target_q.append(data["target_q95"][frames].astype(np.float32))
        target_beta.append(data["target_betaN"][frames].astype(np.float32))
        shot_index[positions] = index
    # The arrays above are concatenated in sorted-shot order; prediction rows
    # may instead follow manifest order, so place targets back by row position.
    target_field = np.empty_like(prediction["psirz"], dtype=np.float32)
    target_q95 = np.empty(len(names), dtype=np.float32)
    target_beta_n = np.empty(len(names), dtype=np.float32)
    for index, shot in enumerate(shots):
        positions = np.flatnonzero(names == shot)
        target_field[positions] = targets[index]
        target_q95[positions] = target_q[index]
        target_beta_n[positions] = target_beta[index]

    values = {
        "field": prediction["psirz"].astype(np.float32),
        "target_field": target_field,
        "q95": prediction["q95"].astype(np.float32),
        "target_q95": target_q95,
        "betaN": prediction["betaN"].astype(np.float32),
        "target_betaN": target_beta_n,
        "shot_index": shot_index,
        "source_index": source_index,
        "shot_name": prediction["shot_name"],
        "reference_source": str(args.cache_dir.resolve()),
    }
    print(f"building official context shots={len(shots)} frames={len(names)}", flush=True)
    score = (
        _parallel_official_score(values, args.workers, args.reference_cache)
        if args.workers >= 1
        else None
    )
    report = {
        "predictions": str(args.predictions.resolve()),
        "shots": int(len(shots)),
        "frames": int(len(names)),
        "flux_r2": _r2(target_field, values["field"]),
        "q95_r2": _r2(target_q95, values["q95"]),
        "betaN_r2": _r2(target_beta_n, values["betaN"]),
        "score": score,
        "data_policy": "real_diii_d_only",
        "frozen_audit_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
