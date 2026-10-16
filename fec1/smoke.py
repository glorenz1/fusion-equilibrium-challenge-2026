"""Generate a tiny synthetic dataset and exercise the full training workflow."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

from fec1.paths import ROOT
from fec1.data.projection import safe_input_columns


def make_data(directory):
    import numpy as np
    import pandas as pd

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with np.load(ROOT / "fec1/scoring/masks/d3d_envelope.npz") as saved:
        r, z = saved["grid_R"], saved["grid_Z"]
    rr, zz = np.meshgrid(r, z, indexing="xy")
    times = np.arange(8, dtype=float) * 20 + 1000
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    records = []
    for shot in range(23):
        name = f"d3d_shot_{shot:010x}"
        axis_r = 1.7 + 0.001 * shot + 0.002 * np.arange(8)
        axis_z = 0.05 + 0.001 * shot + 0.001 * np.arange(8)
        minor = 0.43 + 0.0005 * shot
        vertical = 0.68 + 0.0007 * shot
        normalized = ((rr[None] - axis_r[:, None, None]) / minor) ** 2
        normalized += ((zz[None] - axis_z[:, None, None]) / vertical) ** 2
        # The released DIII-D convention has an axis minimum; preprocessing flips it.
        flux = -0.15 * (1 - normalized)
        row = dict(
            source="DIII-D",
            coil_input_column=[],
            coil_R=[],
            coil_Z=[],
            efit_times=times.tolist(),
            efit_grid_R=r.tolist(),
            efit_grid_Z=z.tolist(),
            efit_psirz=flux.tolist(),
            efit_q95=(2 + 0.02 * shot + 0.005 * np.arange(8)).tolist(),
            efit_beta_n=(0.7 + 0.01 * shot + 0.002 * np.arange(8)).tolist(),
            efit_r_axis=axis_r.tolist(),
            efit_z_axis=axis_z.tolist(),
            efit_lcfs_n=[64] * 8,
            efit_lcfs_r=(axis_r[:, None] + minor * np.cos(angles)).tolist(),
            efit_lcfs_z=(axis_z[:, None] + vertical * np.sin(angles)).tolist(),
            magnetics_time=times.tolist(),
            magnetics_plasma_current_times=times.tolist(),
            magnetics_plasma_current=(700 + 3 * shot + np.arange(8)).tolist(),
            magnetics_bcoil=(100 + 0.1 * shot + 0.02 * np.arange(8)).tolist(),
            magnetics_ECOILA=(10 + 0.1 * shot + 0.02 * np.arange(8)).tolist(),
        )
        safe_input_columns(tuple(row))
        pd.DataFrame([row]).to_parquet(directory / f"{name}.parquet", index=False)
        records.append(
            dict(shot=name, split="train" if shot < 20 else "development", source_frame_count=8)
        )
    return {"records": records, "synthetic": True}


def smoke(run, workers=1, resume=False):
    from fec1 import pipeline

    run = Path(run).resolve()
    raw = run.parent / (run.name + "-synthetic-data")
    if resume and (run / "run.json").exists():
        manifest = json.loads((run / "splits.json").read_text())
    else:
        manifest = make_data(raw)
    pipeline.prepare(run, raw, workers, resume, smoke_manifest=manifest)
    pipeline.train(run, "cpu", workers, resume)
    pipeline.evaluate(run, "cpu", workers, resume)
    output = run / "public-smoke"
    complete = (output / "diii_d_public_test.npz").is_file() and (
        output / "predictions.json"
    ).is_file()
    if output.exists() and not complete:
        failed = run / "failed"
        failed.mkdir(exist_ok=True)
        destination = failed / f"public-smoke-{len(list(failed.glob('public-smoke-*')))}"
        shutil.move(str(output), str(destination))
    if not complete:
        pipeline.predict(run, raw, output, "cpu", workers, max_shots=1)
    score = run / "rescored.json"
    if not score.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "fec1",
                "score",
                "--run",
                str(run),
                "--predictions",
                str(run / "evaluation/development_predictions.npz"),
                "--output",
                str(score),
                "--workers",
                str(workers),
            ],
            cwd=ROOT,
            env=pipeline.subprocess_environment(run, workers),
            check=True,
        )
    print(
        f"Synthetic smoke completed: {run}. These scores measure software execution only.",
        flush=True,
    )
