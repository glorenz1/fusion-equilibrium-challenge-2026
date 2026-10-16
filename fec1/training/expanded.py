#!/usr/bin/env python3
"""Move unused selection shots into DCT24 training without base leakage."""

from __future__ import annotations

from fec1.paths import ARTIFACTS, SYNTHETIC

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

SOURCE_ROOT = ARTIFACTS


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output: Path, fold: int, seed: int, selection_shots: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if fold not in range(5):
        raise ValueError("fold must be 0 through 4")
    source = SOURCE_ROOT / f"base_fold{fold}"
    scalar_path = SOURCE_ROOT / f"scalar_fold{fold}/checkpoint.pt"
    original = json.loads((source / "ready.json").read_text())
    if (
        original["fold"] != fold
        or (original["smoke_only"] and not SYNTHETIC)
        or not original["nested_shot_crossfit"]
        or original["development_used"]
        or original["frozen_audit_used"]
    ):
        raise RuntimeError("source is not a frozen nested base fold")
    with np.load(source / "index.npz", allow_pickle=False) as index:
        arrays = {key: index[key] for key in index.files}
    names = arrays["shot_name"].astype(str)
    previous = arrays["role"].copy()
    validation = np.unique(names[previous == 1])
    chosen = np.random.default_rng(seed + fold).permutation(validation)[:selection_shots]
    moving = (previous == 1) & ~np.isin(names, chosen)
    if (
        len(chosen) != selection_shots
        or len(np.unique(names[moving])) != len(validation) - selection_shots
        or np.any(previous[moving] != 1)
    ):
        raise RuntimeError("selection partition changed")
    arrays["role"][moving] = 0
    if not np.array_equal(previous == 2, arrays["role"] == 2):
        raise RuntimeError("outer evaluation role changed")

    output.mkdir(parents=True)
    for name in ("features.npy", "base_field.npy", "base_scalar.npy", "base.joblib"):
        os.symlink((source / name).resolve(), output / name)
    np.savez_compressed(output / "index.npz", **arrays)
    new_report = dict(original)
    new_report["roles"] = {
        str(role): np.unique(names[arrays["role"] == role]).tolist() for role in (0, 1, 2)
    }
    new_report["augmentation"] = {
        "kind": "former_validation_shots_with_out_of_sample_base_predictions",
        "source_ready_sha256": sha(source / "ready.json"),
        "source_index_sha256": sha(source / "index.npz"),
        "original_base_fits_unchanged": True,
        "moved_shots": len(np.unique(names[moving])),
        "selection_shots_kept": selection_shots,
        "selection_seed": seed + fold,
        "outer_shots_unchanged": True,
    }
    (output / "ready.json").write_text(json.dumps(new_report, indent=2) + "\n")

    scalar = torch.load(scalar_path, map_location="cpu", weights_only=False)
    if (
        scalar["fold"] != fold
        or Path(scalar["base_directory"]).resolve() != source.resolve()
        or not scalar["nested_shot_crossfit"]
        or scalar["development_used"]
        or scalar["frozen_audit_used"]
    ):
        raise RuntimeError("scalar reference provenance changed")
    scalar["base_directory"] = str(output.resolve())
    scalar["expanded_training_provenance"] = {
        "source_checkpoint": str(scalar_path),
        "source_sha256": sha(scalar_path),
        "weights_unchanged": True,
        "outer_shots_excluded_from_original_fit": True,
    }
    torch.save(scalar, output / "scalar_reference.pt")
    result = {
        "source_shots": len(np.unique(names)),
        "original_training_shots": len(np.unique(names[previous == 0])),
        "expanded_training_shots": len(new_report["roles"]["0"]),
        "selection_shots": len(new_report["roles"]["1"]),
        "outer_shots": len(new_report["roles"]["2"]),
        "source_scalar_sha256": sha(scalar_path),
        "prepared_scalar_sha256": sha(output / "scalar_reference.pt"),
    }
    (output / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260905)
    parser.add_argument("--selection-shots", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.fold, args.seed, args.selection_shots)), flush=True)


if __name__ == "__main__":
    main()
