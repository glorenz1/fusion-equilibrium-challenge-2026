"""Run the complete raw-data-to-model recipe in an isolated local directory."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys

from fec1.paths import ROOT
from fec1.provenance import file_records, sha256, source_identity, verify_records, write_json

FULL = dict(
    folds=5,
    inner_folds=3,
    components=64,
    pca_frames=60000,
    estimators=96,
    epochs=36,
    batch_size=128,
    width=384,
    channels=24,
    selection_shots=64,
    train_stride=4,
    exact_every=6,
    geometry_warmup=4,
    angles=64,
    contour_points=128,
)
SMOKE = {
    **FULL,
    "components": 4,
    "pca_frames": 128,
    "estimators": 4,
    "epochs": 1,
    "width": 32,
    "channels": 8,
    "selection_shots": 1,
    "exact_every": 1,
    "geometry_warmup": 0,
    "angles": 16,
    "contour_points": 16,
}


def environment():
    return dict(
        python=sys.version,
        platform=platform.platform(),
        packages={
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "torch",
                "scipy",
                "pandas",
                "pyarrow",
                "scikit-learn",
                "scikit-image",
            )
        },
    )


@contextmanager
def run_lock(run):
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process is using {run}") from error
        yield run


def load_run(run):
    run = Path(run).resolve()
    record = json.loads((run / "run.json").read_text())
    if record["source_sha256"] != source_identity(ROOT):
        raise ValueError(
            "implementation/config changed since this run began; use a new run directory"
        )
    if record["splits_sha256"] != sha256(run / "splits.json"):
        raise ValueError("run split identities changed")
    return record


def initialize(run, data, *, resume=False, smoke_manifest=None):
    run, data = Path(run).resolve(), Path(data).resolve()
    if (run / "run.json").exists():
        if not resume:
            raise FileExistsError("run already exists; use --resume or a new --run directory")
        record = load_run(run)
        if record["raw_data"] != str(data):
            raise ValueError("raw data directory differs from the original run")
        verify_records(data, record["raw_files"])
        return record
    existing = [p for p in run.iterdir() if p.name != ".lock"]
    if existing:
        raise FileExistsError(f"new run directory must be empty: {run}")
    manifest = smoke_manifest or json.loads((ROOT / "repro/splits/shots.json").read_text())
    files = sorted(data.glob("*.parquet"))
    expected = {r["shot"] for r in manifest["records"]}
    if {p.stem for p in files} != expected:
        raise ValueError(
            f"raw files must match the {len(expected)} committed shot identities exactly"
        )
    records = file_records(data, files)
    download_path = data / "download.json"
    download = json.loads(download_path.read_text()) if download_path.exists() else None
    if download and {r["path"]: r for r in records} != {r["path"]: r for r in download["files"]}:
        raise ValueError("raw data differs from its download manifest")
    write_json(run / "splits.json", manifest)
    record = dict(
        schema_version=1,
        profile="synthetic_smoke" if smoke_manifest else "full",
        parameters=SMOKE if smoke_manifest else FULL,
        raw_data=str(data),
        raw_files=records,
        dataset=download and {k: download[k] for k in ("repo_id", "revision", "split")},
        source_sha256=source_identity(ROOT),
        splits_sha256=sha256(run / "splits.json"),
        environment=environment(),
        stages={},
    )
    write_json(run / "run.json", record)
    return record


def subprocess_environment(run, workers):
    env = dict(
        os.environ,
        FEC1_RUN_DIR=str(Path(run).resolve()),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS=str(workers),
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS=str(workers),
    )
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def stage(run, name, module, arguments, outputs, workers, resume):
    run = Path(run)
    record = load_run(run)
    prior = record["stages"].get(name)
    outputs = [run / p for p in outputs]
    if prior and not resume:
        raise FileExistsError(f"stage {name} already attempted; use --resume")
    if prior and prior["status"] == "complete":
        if not all(p.exists() for p in outputs):
            raise ValueError(f"completed stage {name} is missing outputs")
        print(f"{name}: already complete", flush=True)
        return
    attempt = (prior or {}).get("attempt", 0) + 1
    # Preserve partial attempts before restarting a failed stage from scratch.
    for output in outputs:
        if output.exists():
            if not prior:
                raise FileExistsError(f"unexpected pre-existing stage output: {output}")
            destination = run / "failed" / f"{name}-{attempt - 1}" / output.relative_to(run)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(output), str(destination))
    command = [sys.executable, "-m", module, *map(str, arguments)]
    log = run / "logs" / f"{name}.{attempt:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    item = dict(
        status="running",
        attempt=attempt,
        command=command,
        log=str(log.relative_to(run)),
        started=datetime.now(timezone.utc).isoformat(),
        environment=environment(),
    )
    record["stages"][name] = item
    write_json(run / "run.json", record)
    print(f"{name}: running (log: {log})", flush=True)
    with log.open("w") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=subprocess_environment(run, workers),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    item.update(
        status="complete"
        if result.returncode == 0 and all(p.exists() for p in outputs)
        else "failed",
        exit_code=result.returncode,
        finished=datetime.now(timezone.utc).isoformat(),
    )
    write_json(run / "run.json", record)
    if item["status"] != "complete":
        raise RuntimeError(
            f"{name} failed; inspect {log}; --resume preserves this attempt and retries"
        )


def prepare(run, data, workers=4, resume=False, smoke_manifest=None):
    data = Path(data).resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    with run_lock(run) as run:
        record = initialize(run, data, resume=resume, smoke_manifest=smoke_manifest)
        p = record["parameters"]
        a = run / "artifacts"
        calls = [
            (
                "cache",
                "fec1.data.cache",
                [
                    "--input-dir",
                    data,
                    "--output-dir",
                    run / "cache",
                    "--workers",
                    workers,
                    "--uncompressed",
                ],
                ["cache"],
            ),
            (
                "pca",
                "fec1.training.pca",
                [
                    "--output",
                    a / "source_pca.joblib",
                    "--workers",
                    workers,
                    "--components",
                    p["components"],
                    "--frames",
                    p["pca_frames"],
                ],
                ["artifacts/source_pca.joblib"],
            ),
            (
                "source",
                "fec1.training.source_rows",
                [
                    "--output",
                    a / "source_rows.npz",
                    "--workers",
                    workers,
                    "--estimators",
                    p["estimators"],
                    "--folds",
                    p["folds"],
                    "--train-stride",
                    p["train_stride"],
                ],
                ["artifacts/source_rows.npz", "artifacts/source_rows.json"],
            ),
            (
                "inputs",
                "fec1.training.inputs",
                [
                    "--oof",
                    a / "source_rows.npz",
                    "--cache-dir",
                    run / "cache",
                    "--output-dir",
                    a / "inputs",
                ],
                ["artifacts/inputs"],
            ),
            (
                "targets",
                "fec1.training.targets",
                [
                    "--oof",
                    a / "source_rows.npz",
                    "--base-artifact",
                    a / "source_pca.joblib",
                    "--output",
                    a / "target_normalized.npy",
                    "--pca-output",
                    a / "pca_basis.npz",
                ],
                [
                    "artifacts/target_normalized.npy",
                    "artifacts/target_normalized.json",
                    "artifacts/pca_basis.npz",
                ],
            ),
            (
                "topology",
                "fec1.training.topology",
                [
                    "--all-train-stride",
                    p["train_stride"],
                    "--workers",
                    workers,
                    "--output",
                    a / "topology.npz",
                ],
                ["artifacts/topology.npz", "artifacts/topology.json"],
            ),
            (
                "contours",
                "fec1.training.contours",
                [
                    "--reference",
                    a / "topology.npz",
                    "--angles",
                    p["angles"],
                    "--contour-points",
                    p["contour_points"],
                    "--workers",
                    workers,
                    "--output",
                    a / "contours.npz",
                ],
                ["artifacts/contours.npz", "artifacts/contours.json"],
            ),
        ]
        for name, module, arguments, outputs in calls:
            stage(run, name, module, arguments, outputs, workers, resume)


def train(run, device="cuda", workers=4, resume=False):
    if workers < 1:
        raise ValueError("workers must be positive")
    with run_lock(run) as run:
        record = load_run(run)
        if record["stages"].get("contours", {}).get("status") != "complete":
            raise ValueError("run prepare successfully before train")
        p = record["parameters"]
        a = run / "artifacts"
        for fold in range(p["folds"]):
            base, scalar, expanded, final = [
                a / f"{prefix}_fold{fold}" for prefix in ("base", "scalar", "data", "dct")
            ]
            stage(
                run,
                f"carrier-{fold}",
                "fec1.training.carriers",
                [
                    "--fold",
                    fold,
                    "--output",
                    base,
                    "--components",
                    p["components"],
                    "--estimators",
                    p["estimators"],
                    "--workers",
                    workers,
                ],
                [str(base.relative_to(run))],
                workers,
                resume,
            )
            common = [
                "--decoder",
                "dct",
                "--epochs",
                p["epochs"],
                "--batch-size",
                p["batch_size"],
                "--width",
                p["width"],
                "--channels",
                p["channels"],
                "--device",
                device,
                "--score-workers",
                workers,
                "--selection-shots",
                p["selection_shots"],
                "--exact-every",
                p["exact_every"],
                "--geometry-warmup",
                p["geometry_warmup"],
            ]
            stage(
                run,
                f"scalar-{fold}",
                "fec1.training.fit",
                ["--data", base, "--output", scalar, *common],
                [str(scalar.relative_to(run))],
                workers,
                resume,
            )
            stage(
                run,
                f"expand-{fold}",
                "fec1.training.expanded",
                ["--fold", fold, "--output", expanded, "--selection-shots", p["selection_shots"]],
                [str(expanded.relative_to(run))],
                workers,
                resume,
            )
            stage(
                run,
                f"field-{fold}",
                "fec1.training.fit",
                [
                    "--data",
                    expanded,
                    "--output",
                    final,
                    *common,
                    "--loss-version",
                    "symmetric",
                    "--observation-encoder",
                    "spatial_attention",
                    "--scalar-reference",
                    expanded / "scalar_reference.pt",
                ],
                [str(final.relative_to(run))],
                workers,
                resume,
            )
        paths = [
            a / f"{prefix}_fold{fold}" / filename
            for fold in range(p["folds"])
            for prefix, filename in (("base", "base.joblib"), ("dct", "checkpoint.pt"))
        ]
        write_json(
            run / "model.json",
            dict(
                schema_version=1,
                profile=record["profile"],
                source_sha256=record["source_sha256"],
                ensemble_weights=[0.2] * 5,
                artifacts=file_records(a, paths),
                environment=environment(),
            ),
        )
        print(f"Model ready: {run / 'model.json'}", flush=True)


def evaluate(run, device="cuda", workers=4, resume=False):
    with run_lock(run) as run:
        record = load_run(run)
        verify_records(run / "artifacts", json.loads((run / "model.json").read_text())["artifacts"])
        stage(
            run,
            "evaluate",
            "fec1.evaluation",
            [
                "--root",
                run / "artifacts",
                "--output",
                run / "evaluation",
                "--device",
                device,
                "--score-workers",
                workers,
            ],
            ["evaluation"],
            workers,
            resume,
        )
        result = json.loads((run / "evaluation/result.json").read_text())
        result.update(
            profile=record["profile"],
            public_score_recomputed=False,
            interpretation="synthetic software check only"
            if record["profile"] == "synthetic_smoke"
            else "local held-shot metrics; public hidden targets are unavailable",
        )
        write_json(run / "evaluation/result.json", result)
        print(json.dumps(result), flush=True)


def predict(run, inputs, output, device="cpu", workers=4, batch_size=128, max_shots=0):
    run = Path(run).resolve()
    load_run(run)
    model = json.loads((run / "model.json").read_text())
    verify_records(run / "artifacts", model["artifacts"])
    if Path(output).exists():
        raise FileExistsError("prediction output must be a new directory")
    command = [
        sys.executable,
        "-m",
        "fec1.inference",
        "--inputs",
        str(Path(inputs).resolve()),
        "--output",
        str(Path(output).resolve()),
        "--selected-root",
        str(run / "artifacts"),
        "--base-root",
        str(run / "artifacts"),
        "--device",
        device,
        "--workers",
        str(workers),
        "--batch-size",
        str(batch_size),
        "--max-shots",
        str(max_shots),
    ]
    subprocess.run(command, cwd=ROOT, env=subprocess_environment(run, workers), check=True)
