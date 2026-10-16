"""Prepare real DIII-D shot caches without loading any neural solver."""
from __future__ import annotations
import argparse, hashlib, json, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
from fec1.data.config import PreprocessingConfig
from fec1.data.records import evenly_spaced_frame_indices, save_prepared_shot


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_names(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _prepare_one_shot(task: tuple[str, str, dict, int, bool]) -> tuple[str, int]:
    path_text, destination_text, config_values, frames_per_shot, compressed = task
    path = Path(path_text)
    destination = Path(destination_text)
    config = PreprocessingConfig.from_dict(config_values)
    row = pd.read_parquet(path).iloc[0].to_dict()
    if str(row.get("source", "DIII-D")) != "DIII-D":
        raise ValueError(
            f"refusing non-DIII-D training row {path}: FEC1 training must remain DIII-D-only"
        )
    selected = evenly_spaced_frame_indices(len(row["efit_times"]), frames_per_shot)
    save_prepared_shot(
        row,
        destination,
        config,
        frame_indices=selected,
        compressed=compressed,
    )
    return destination.name, len(selected)


def prepare_local_shards(
    input_dir: Path,
    output_dir: Path,
    config: PreprocessingConfig,
    skip: int = 0,
    n_shots: int = 0,
    frames_per_shot: int = 0,
    workers: int = 1,
    compressed: bool = True,
) -> list[Path]:
    """Convert raw one-row parquet shots to FEC1 shot caches."""

    files = sorted(Path(input_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet shots found under {input_dir}")
    selected = files[skip:] if n_shots <= 0 else files[skip : skip + n_shots]
    output_dir.mkdir(parents=True, exist_ok=True)
    written = [output_dir / f"{path.stem}.npz" for path in selected]
    tasks = [
        (
            str(path),
            str(destination),
            config.to_dict(),
            int(frames_per_shot),
            bool(compressed),
        )
        for path, destination in zip(selected, written)
    ]
    workers = max(1, min(int(workers), len(tasks) or 1))
    started = time.monotonic()
    if workers == 1:
        results = map(_prepare_one_shot, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_prepare_one_shot, tasks, chunksize=1)
    frame_count = 0
    try:
        for index, (name, n_frames) in enumerate(results, start=1):
            frame_count += n_frames
            if index == 1 or index % 100 == 0 or index == len(selected):
                print(
                    f"  prepared {index}/{len(selected)}: {name} "
                    f"({time.monotonic() - started:.1f}s)",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    manifest = {
        "data_policy": "real_diii_d_only",
        "source_machines": ["DIII-D"],
        "mast_data_used": False,
        "synthetic_data_used": False,
        "raw_input_dir": str(Path(input_dir).resolve()),
        "raw_input_filenames_sha256": sha256_names(selected),
        "config": config.to_dict(),
        "preparation": {
            "shots": len(written),
            "frames": frame_count,
            "frames_per_shot": int(frames_per_shot),
            "workers": workers,
            "compressed": bool(compressed),
            "elapsed_seconds": time.monotonic() - started,
        },
        "files": [path.name for path in written],
    }
    (output_dir / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/diii_d_train"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--frames-per-shot", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args()
    prepare_local_shards(
        args.input_dir,
        args.output_dir,
        PreprocessingConfig(),
        frames_per_shot=args.frames_per_shot,
        workers=args.workers,
        compressed=not args.uncompressed,
    )


if __name__ == "__main__":
    main()
