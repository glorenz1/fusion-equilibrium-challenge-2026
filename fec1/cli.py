#!/usr/bin/env python3
"""Download data, train the FEC1 ensemble from scratch, evaluate and predict."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import re
import subprocess
from pathlib import Path

from fec1.paths import ROOT


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest():
    return json.loads((ROOT / "repro/artifacts.json").read_text())


def verify_files(root, records):
    """Validate all bytes before deserializing trusted model artifacts."""
    for item in records:
        path = Path(root) / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {path}")


def import_artifacts(source, destination, *, copy=False):
    records = manifest()["artifacts"]
    # Complete the checks before creating any destination files.
    verify_files(source, [dict(r, path=r["source"]) for r in records])
    destinations = [Path(destination) / r["path"] for r in records]
    if any(p.exists() or p.is_symlink() for p in destinations):
        raise FileExistsError("use a fresh artifact directory")
    for item, target in zip(records, destinations):
        target.parent.mkdir(parents=True, exist_ok=True)
        original = (Path(source) / item["source"]).resolve()
        if copy:
            shutil.copyfile(original, target)
        else:
            target.symlink_to(original)


def verify_source():
    record = json.loads((ROOT / "repro/source_manifest.json").read_text())
    for name, expected in record["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    return len(record["files"])


def predict(args):
    verify_source()
    verify_files(args.models, manifest()["artifacts"])
    from fec1 import inference as exporter

    # The frozen exporter owns all preprocessing, context, calibration and averaging.
    sys.argv = [
        "fec1.inference",
        "--inputs",
        str(args.inputs.resolve()),
        "--output",
        str(args.output.resolve()),
        "--selected-root",
        str(args.models.resolve()),
        "--base-root",
        str(args.models.resolve()),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--batch-size",
        str(args.batch_size),
        "--max-shots",
        str(args.max_shots),
    ]
    return exporter.main()


def compare_archives(actual, expected, *, atol=2e-6, rtol=2e-5):
    """Compare numerical arrays, since ZIP metadata/compression need not match."""
    import numpy as np

    largest = 0.0
    with np.load(actual, allow_pickle=False) as a, np.load(expected, allow_pickle=False) as b:
        if not a.files or not set(a.files).issubset(b.files):
            raise ValueError("empty or incompatible prediction archive")
        shots = set()
        for key in a.files:
            match = re.fullmatch(r"shot_(\d{4})_(psirz|q95|betaN)", key)
            if match is None:
                raise ValueError(f"unexpected prediction key: {key}")
            shots.add(int(match[1]))
        required = {
            f"shot_{shot:04d}_{name}"
            for shot in range(len(shots))
            for name in ("psirz", "q95", "betaN")
        }
        if set(a.files) != required:
            raise ValueError(
                "archive must contain complete psirz/q95/betaN arrays for the first N shots"
            )
        for key in a.files:
            x, y = a[key], b[key]
            if x.shape != y.shape or x.dtype != np.float32:
                raise ValueError(f"invalid output schema: {key}")
            count = len(a[key.rsplit("_", 1)[0] + "_q95"])
            required_shape = (count, 65, 65) if key.endswith("_psirz") else (count,)
            if x.shape != required_shape or count == 0:
                raise ValueError(f"invalid output schema: {key}")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"nonfinite output: {key}")
            np.testing.assert_allclose(x, y, atol=atol, rtol=rtol, err_msg=key)
            largest = max(largest, float(np.max(np.abs(x - y))))
    return {
        "passed": True,
        "arrays": len(a.files),
        "max_absolute_error": largest,
        "atol": atol,
        "rtol": rtol,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="download verified official raw parquet data")
    fetch.add_argument("--output", type=Path, default=ROOT / "data")
    fetch.add_argument("--split", action="append", choices=("diii_d_train", "diii_d_public_test"))
    fetch.add_argument("--revision")
    fetch.add_argument("--workers", type=int, default=4)
    prep = commands.add_parser("prepare", help="build all training inputs from raw labelled shots")
    prep.add_argument("--data", type=Path, required=True)
    prep.add_argument("--run", type=Path, required=True)
    prep.add_argument("--workers", type=int, default=4)
    prep.add_argument("--resume", action="store_true")
    for name in ("train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--resume", action="store_true")
    score = commands.add_parser("score", help="score complete local development predictions")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--predictions", type=Path)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--workers", type=int, default=4)
    smoke = commands.add_parser("smoke", help="exercise the complete workflow on synthetic data")
    smoke.add_argument("--run", type=Path, required=True)
    smoke.add_argument("--workers", type=int, default=1)
    smoke.add_argument("--resume", action="store_true")
    commands.add_parser("verify-source")
    verify = commands.add_parser("verify-artifacts")
    verify.add_argument("--models", type=Path, default=ROOT / "artifacts/incumbent")
    load = commands.add_parser("import-artifacts")
    load.add_argument(
        "--source",
        type=Path,
        required=True,
        help="retained starter checkout or exported artifact store",
    )
    load.add_argument("--models", type=Path, default=ROOT / "artifacts/incumbent")
    load.add_argument(
        "--copy", action="store_true", help="make independent copies instead of local symlinks"
    )
    infer = commands.add_parser("predict")
    infer.add_argument("--run", type=Path, help="run containing a model trained from scratch")
    infer.add_argument("--inputs", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--models", type=Path, default=ROOT / "artifacts/incumbent")
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--workers", type=int, default=4)
    infer.add_argument("--batch-size", type=int, default=128)
    infer.add_argument("--max-shots", type=int, default=0)
    compare = commands.add_parser("compare")
    compare.add_argument("actual", type=Path)
    compare.add_argument("expected", type=Path)
    compare.add_argument("--report", type=Path)
    args = parser.parse_args()
    if hasattr(args, "workers") and args.workers < 1:
        parser.error("workers must be positive")
    if args.command == "download":
        from fec1.download import download

        download(args.output, args.split or ["diii_d_train"], args.workers, args.revision)
    elif args.command in ("prepare", "train", "evaluate"):
        from fec1 import pipeline

        if args.command == "prepare":
            pipeline.prepare(args.run, args.data, args.workers, args.resume)
        else:
            getattr(pipeline, args.command)(args.run, args.device, args.workers, args.resume)
    elif args.command == "smoke":
        from fec1.smoke import smoke

        smoke(args.run, args.workers, args.resume)
    elif args.command == "score":
        from fec1 import pipeline
        from fec1.provenance import verify_records

        pipeline.load_run(args.run)
        if args.output.exists():
            raise FileExistsError("use a new score output path")
        manifest = json.loads((args.run / "splits.json").read_text())
        development = {r["shot"] for r in manifest["records"] if r["split"] == "development"}
        cache = args.run / "cache"
        contents = json.loads((cache / "config.json").read_text())["content"]
        verify_records(cache, [r for r in contents if Path(r["path"]).stem in development])
        predictions = args.predictions or args.run / "evaluation/development_predictions.npz"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "fec1.scoring.archive",
                "--predictions",
                str(predictions.resolve()),
                "--output",
                str(args.output.resolve()),
                "--workers",
                str(args.workers),
            ],
            cwd=ROOT,
            env=pipeline.subprocess_environment(args.run, args.workers),
            check=True,
        )
    elif args.command == "verify-source":
        print(json.dumps({"passed": True, "files": verify_source()}))
    elif args.command == "verify-artifacts":
        verify_files(args.models, manifest()["artifacts"])
        print(json.dumps({"passed": True, "artifacts": 10}))
    elif args.command == "import-artifacts":
        import_artifacts(args.source, args.models, copy=args.copy)
    elif args.command == "predict":
        if args.workers < 1 or args.batch_size < 1 or args.max_shots < 0:
            parser.error("workers/batch-size must be positive and max-shots nonnegative")
        if args.run:
            from fec1 import pipeline

            return pipeline.predict(
                args.run,
                args.inputs,
                args.output,
                args.device,
                args.workers,
                args.batch_size,
                args.max_shots,
            )
        return predict(args)
    elif args.command == "compare":
        result = compare_archives(args.actual, args.expected)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
