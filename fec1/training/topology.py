"""Extract exact scorer topology for immutable FEC1 training shots.

The immutable shot manifest and requested frame stride determine the rows.
Target flux is read from per-shot caches only after the shot is confirmed to
belong to the train split.  Development and frozen-audit shot files are therefore never
opened by this program.
"""
from __future__ import annotations
from fec1.paths import DATA_POLICY
from fec1.paths import CACHE, MANIFEST
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from fec1.scoring.vendor.common import CONS_SCALARS
from fec1.scoring.topology import derive_flux_topology


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _one(task: tuple[str, np.ndarray, str]) -> tuple[str, np.ndarray, np.ndarray]:
    shot, frame_index, cache_dir = task
    path = Path(cache_dir) / f"{shot}.npz"
    with np.load(path, allow_pickle=False) as saved:
        fields = np.asarray(saved["target_psi"])[frame_index].astype(np.float64)
        grid_r = np.asarray(saved["grid_r"], dtype=np.float64)
        grid_z = np.asarray(saved["grid_z"], dtype=np.float64)
    target, _ = derive_flux_topology(fields, grid_r, grid_z, axis_sign=1.0, n_iter=22, n_points=512)
    return (shot, frame_index, target.astype(np.float32))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--all-train-stride",
        type=int,
        default=4,
        help="extract every Nth frame from every immutable train shot instead of selected rows",
    )
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--max-shots", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.all_train_stride <= 0:
        parser.error("--all-train-stride must be positive")
    output_split = "train"
    requested_stride = args.all_train_stride
    metadata_source = args.manifest
    manifest = json.loads(args.manifest.read_text())
    records = [record for record in manifest["records"] if record["split"] == "train"]
    shot_parts, frame_parts, sample_parts = ([], [], [])
    cursor = 0
    for record in records:
        frames = np.arange(0, int(record["source_frame_count"]), requested_stride, dtype=np.int32)
        shot_parts.append(np.full(len(frames), record["shot"], dtype="U19"))
        frame_parts.append(frames)
        sample_parts.append(np.arange(cursor, cursor + len(frames), dtype=np.int64))
        cursor += len(frames)
    sample_index = np.concatenate(sample_parts)
    shot_name = np.concatenate(shot_parts)
    frame_index = np.concatenate(frame_parts)
    shots = np.unique(shot_name)
    if args.max_shots > 0:
        shots = shots[: args.max_shots]
        keep = np.isin(shot_name, shots)
        sample_index = sample_index[keep]
        shot_name = shot_name[keep]
        frame_index = frame_index[keep]
    tasks = []
    for shot in shots:
        rows = np.flatnonzero(shot_name == shot)
        path = args.cache_dir / f"{shot}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        tasks.append((shot, frame_index[rows], str(args.cache_dir.resolve())))
    print(
        f"exact {output_split} topology shots={len(tasks)} frames={len(sample_index)}; frozen-audit shot files remain unopened",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        results = []
        for count, result in enumerate(executor.map(_one, tasks, chunksize=1), start=1):
            results.append(result)
            if count == 1 or count % 100 == 0 or count == len(tasks):
                print(f"exact {output_split} topology {count}/{len(tasks)} shots", flush=True)
    lookup = {
        (shot, int(frame)): target
        for shot, frames, targets in results
        for frame, target in zip(frames, targets)
    }
    target = np.asarray(
        [lookup[shot, int(frame)] for shot, frame in zip(shot_name, frame_index)], dtype=np.float32
    )
    if len(lookup) != len(target):
        raise RuntimeError("exact train topology rows are incomplete")
    finite_fraction = np.isfinite(target).mean(axis=0)
    if np.any(finite_fraction < 0.995):
        failed = {
            name: float(fraction)
            for name, fraction in zip(CONS_SCALARS, finite_fraction)
            if fraction < 0.995
        }
        raise RuntimeError(f"too many non-finite exact train topology targets: {failed}")
    order = np.argsort(sample_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_index=sample_index[order],
        shot_name=shot_name[order].astype("U19"),
        frame_index=frame_index[order],
        split=np.full(len(target), output_split, dtype="U12"),
        target=target[order],
        names=np.asarray(CONS_SCALARS),
        data_policy=np.asarray(DATA_POLICY),
        mast_data_used=np.asarray(False),
        development_used=np.asarray(output_split == "development"),
        frozen_audit_used=np.asarray(False),
        label_source=np.asarray(f"exact scorer derivation from {output_split} target flux"),
        axis_sign=np.asarray(1.0),
        lcfs_iterations=np.asarray(22, dtype=np.int64),
        lcfs_points=np.asarray(512, dtype=np.int64),
        metadata_sha256=np.asarray(_sha256(metadata_source)),
        frame_stride=np.asarray(requested_stride if requested_stride > 0 else 0, dtype=np.int64),
    )
    report = {
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "shots": int(len(shots)),
        "frames": int(len(target)),
        "finite_fraction": dict(zip(CONS_SCALARS, finite_fraction.tolist())),
        "fully_finite_frames": int(np.isfinite(target).all(axis=1).sum()),
        "split": output_split,
        "frame_stride": requested_stride if requested_stride > 0 else 0,
        "metadata_source": str(metadata_source.resolve()),
        "metadata_sha256": _sha256(metadata_source),
        "data_policy": DATA_POLICY,
        "development_used": output_split == "development",
        "frozen_audit_used": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
