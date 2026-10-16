#!/usr/bin/env python3
"""Portable commands for the frozen FEC1 DIII-D 0.9931 ensemble."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


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
    from my_experiments import export_fec1_dct24_public as exporter
    exporter.PUBLIC = args.inputs.resolve()
    # The frozen exporter owns all preprocessing, context, calibration and averaging.
    sys.argv = ["export_fec1_dct24_public.py", "--output", str(args.output.resolve()),
                "--selected-root", str(args.models.resolve()),
                "--base-root", str(args.models.resolve()), "--device", args.device,
                "--workers", str(args.workers), "--batch-size", str(args.batch_size),
                "--max-shots", str(args.max_shots)]
    return exporter.main()


def compare_archives(actual, expected, *, atol=2e-6, rtol=2e-5):
    """Compare numerical arrays, since ZIP metadata/compression need not match."""
    import numpy as np
    largest = 0.
    with np.load(actual, allow_pickle=False) as a, np.load(expected, allow_pickle=False) as b:
        if not a.files or not set(a.files).issubset(b.files):
            raise ValueError("empty or incompatible prediction archive")
        for key in a.files:
            x, y = a[key], b[key]
            if x.shape != y.shape or x.dtype != np.float32:
                raise ValueError(f"invalid output schema: {key}")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"nonfinite output: {key}")
            np.testing.assert_allclose(x, y, atol=atol, rtol=rtol, err_msg=key)
            largest = max(largest, float(np.max(np.abs(x-y))))
    return {"passed": True, "arrays": len(a.files), "max_absolute_error": largest,
            "atol": atol, "rtol": rtol}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-source")
    verify = commands.add_parser("verify-artifacts")
    verify.add_argument("--models", type=Path, default=ROOT / "artifacts/incumbent")
    load = commands.add_parser("import-artifacts")
    load.add_argument("--source", type=Path, required=True, help="retained starter checkout or exported artifact store")
    load.add_argument("--models", type=Path, default=ROOT / "artifacts/incumbent")
    load.add_argument("--copy", action="store_true", help="make independent copies instead of local symlinks")
    infer = commands.add_parser("predict")
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
    if args.command == "verify-source":
        print(json.dumps({"passed": True, "files": verify_source()}))
    elif args.command == "verify-artifacts":
        verify_files(args.models, manifest()["artifacts"])
        print(json.dumps({"passed": True, "artifacts": 10}))
    elif args.command == "import-artifacts":
        import_artifacts(args.source, args.models, copy=args.copy)
    elif args.command == "predict":
        if args.workers < 1 or args.batch_size < 1 or args.max_shots < 0:
            parser.error("workers/batch-size must be positive and max-shots nonnegative")
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
