#!/usr/bin/env python3
"""Prepare data and train the Learned-Profile Grad--Shafranov network."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from neuralgs.config import NeuralGSConfig
from neuralgs.inference import DEFAULT_ARTIFACT
from neuralgs.training import prepare_local_shards, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="convert raw DIII-D parquet shots to shards")
    prepare.add_argument("--input-dir", type=Path, default=Path("data/diii_d_train"))
    prepare.add_argument("--output-dir", type=Path, default=Path("my_experiments/neuralgs_cache"))
    prepare.add_argument("--skip", type=int, default=0)
    prepare.add_argument("--n-shots", type=int, default=0, help="0 means every remaining shot")
    prepare.add_argument("--frames-per-shot", type=int, default=0, help="0 keeps every frame")
    prepare.add_argument("--workers", type=int, default=1)
    prepare.add_argument(
        "--uncompressed", action="store_true", help="trade cache space for faster training reads"
    )

    for name in ("pretrain", "finetune"):
        train = subparsers.add_parser(name, help=f"run NeuralGS {name}")
        train.add_argument("--cache-dir", type=Path, default=Path("my_experiments/neuralgs_cache"))
        train.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
        train.add_argument("--resume", type=Path)
        train.add_argument("--epochs", type=int, default=20 if name == "pretrain" else 40)
        train.add_argument("--batch-size", type=int, default=16)
        train.add_argument("--learning-rate", type=float, default=2e-4 if name == "pretrain" else 5e-5)
        train.add_argument("--n-val-shots", type=int, default=20)
        train.add_argument("--seed", type=int, default=42)
        train.add_argument("--device", default="auto")
        train.add_argument("--channel-dropout", type=float, default=0.10)
        train.add_argument(
            "--gradient-weight",
            type=float,
            default=0.12,
            help="weight of the spatial flux-gradient loss during finetuning",
        )
        train.add_argument(
            "--topology-weight",
            type=float,
            default=0.20,
            help="weight of the gauge/amplitude-invariant normalized-flux loss",
        )
        train.add_argument(
            "--scalar-weight",
            type=float,
            default=0.0,
            help="scalar-head-only loss weight; direct scalar artifacts are preferred",
        )
        train.add_argument("--outer-iterations", type=int, default=3)
        train.add_argument(
            "--separatrix-span-scale",
            type=float,
            default=1.0,
            help="calibrate the differentiable axis-to-separatrix flux span",
        )
        train.add_argument(
            "--separatrix-estimator",
            choices=("saddle", "perimeter"),
            default="saddle",
            help="choose the Picard flux-coordinate separatrix estimator",
        )
        train.add_argument(
            "--axis-estimator",
            choices=("global", "current_local"),
            default="current_local",
            help="choose a global maximum or current-centroid-guided local O-point",
        )
        train.add_argument(
            "--initial-current-center",
            choices=("grid", "thomson"),
            default="grid",
            help="center the first Picard current prior on the grid or Thomson chord",
        )
        train.add_argument(
            "--spatial-current-modes",
            type=int,
            choices=range(13),
            default=0,
            metavar="0..12",
            help="enable this many machine-normalized smooth 2-D current residual modes",
        )
        train.add_argument(
            "--spatial-current-amplitude",
            type=float,
            default=0.75,
            help="maximum scale of the bounded spatial-current log modulation",
        )
        train.add_argument(
            "--source-head-mode",
            choices=("legacy_pressure_mixture", "conditioned_signed_splines"),
            default="legacy_pressure_mixture",
        )
        train.add_argument("--source-basis-size", type=int, default=4)
        train.add_argument(
            "--physics-raster-encoder",
            action="store_true",
            help="fuse an analytic vacuum-field canvas into the source latent",
        )
        train.add_argument("--raster-hidden-channels", type=int, default=24)
        train.add_argument(
            "--raster-coordinate-mode",
            choices=("cylindrical", "canonical"),
            default="cylindrical",
        )
        train.add_argument("--oracle-dir", type=Path)
        train.add_argument("--split-manifest", type=Path)
        train.add_argument("--source-transform", type=Path)
        train.add_argument(
            "--freeze-source-probe",
            action="store_true",
            help="train only the new source head for clean teacher-coordinate Test A",
        )
        train.add_argument(
            "--freeze-raster-probe",
            action="store_true",
            help="train only the new raster encoder on a frozen deterministic model",
        )
        train.add_argument("--source-green-weight", type=float, default=1.0)
        train.add_argument("--source-current-weight", type=float, default=0.25)
        train.add_argument("--source-smooth-weight", type=float, default=0.01)
        train.add_argument("--source-negative-weight", type=float, default=0.002)
        train.add_argument(
            "--source-coordinate-augmentation",
            type=float,
            default=0.0,
            help="blend teacher source coordinates toward a detached deployment replay",
        )
        train.add_argument(
            "--gauge-weight",
            type=float,
            default=0.0,
            help="fit the gauge head while detaching its shared encoder latent",
        )
        train.add_argument("--d-model", type=int, default=128)
        train.add_argument("--n-heads", type=int, default=4)
        train.add_argument("--attention-blocks", type=int, default=3)
        train.add_argument("--source-stride", type=int, default=2)
        train.add_argument("--max-tokens", type=int, default=192)
        train.add_argument(
            "--global-feature-mode",
            choices=("legacy", "canonical"),
            default="legacy",
            help="select legacy device-scale or canonical dimensionless closure inputs",
        )
        train.add_argument(
            "--physical-tf-turns",
            action="store_true",
            help="use documented machine TF turns in analytic q95/betaN",
        )
        train.add_argument(
            "--disable-scalar-calibration",
            action="store_true",
            help="return analytic q95/betaN without the learned residual head",
        )
        train.add_argument(
            "--coordinate-tokens",
            action="store_true",
            help="append machine-normalized physical R,Z to each Thomson token",
        )
        train.add_argument("--detach-picard-coordinates", action="store_true")
        train.add_argument(
            "--exact-final-source-render",
            action="store_true",
            help="remove finite-iteration damping from the final rendered field",
        )
        train.add_argument("--frame-stride", type=int, default=1,
                           help="use every Nth frame (useful for smoke tests; final training should use 1)")
        train.add_argument("--num-workers", type=int, default=0)
        train.add_argument(
            "--shot-batches", action="store_true",
            help="shuffle shots and concatenate their frames to avoid archive-cache thrashing",
        )
        train.add_argument("--amp", action="store_true", help="use CUDA mixed precision")
        train.add_argument(
            "--max-minutes", type=float, default=0.0,
            help="stop safely after this stage duration; 0 disables the deadline",
        )

    light = subparsers.add_parser(
        "compute-light",
        help="prepare all shots and train one device inside a badge-oriented wall-clock budget",
    )
    light.add_argument("--input-dir", type=Path, default=Path("data/diii_d_train"))
    light.add_argument(
        "--cache-dir", type=Path, default=Path("my_experiments/neuralgs_compute_light_cache")
    )
    light.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    light.add_argument(
        "--stage-a-artifact",
        type=Path,
        default=Path("my_experiments/artifacts/neuralgs_compute_light_stage_a.pt"),
    )
    light.add_argument("--frames-per-shot", type=int, default=8)
    light.add_argument("--n-shots", type=int, default=0, help="0 means all 7,041 shots")
    light.add_argument("--prepare-workers", type=int, default=min(16, os.cpu_count() or 1))
    light.add_argument("--loader-workers", type=int, default=4)
    light.add_argument("--budget-minutes", type=float, default=110.0)
    light.add_argument("--pretrain-minutes", type=float, default=20.0)
    light.add_argument("--pretrain-epochs", type=int, default=3)
    light.add_argument("--finetune-epochs", type=int, default=6)
    light.add_argument("--batch-size", type=int, default=128)
    light.add_argument("--n-val-shots", type=int, default=200)
    light.add_argument("--learning-rate-pretrain", type=float, default=2e-4)
    light.add_argument("--learning-rate-finetune", type=float, default=5e-5)
    light.add_argument("--seed", type=int, default=42)
    light.add_argument("--device", default="auto")
    light.add_argument("--d-model", type=int, default=64)
    light.add_argument("--n-heads", type=int, default=4)
    light.add_argument("--attention-blocks", type=int, default=1)
    light.add_argument("--source-stride", type=int, default=4)
    light.add_argument("--outer-iterations", type=int, default=2)
    light.add_argument("--max-tokens", type=int, default=96)
    light.add_argument("--coordinate-tokens", action="store_true")
    light.add_argument(
        "--detach-picard-coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="truncate backward through discrete flux-coordinate selection",
    )

    return parser


def run_compute_light(args: argparse.Namespace) -> int:
    """Run the raw-data-to-artifact path with one shared wall-clock deadline."""

    started = time.monotonic()
    deadline = started + 60.0 * args.budget_minutes
    config = NeuralGSConfig(
        outer_iterations=args.outer_iterations,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_attention_blocks=args.attention_blocks,
        source_stride=args.source_stride,
        max_tokens=args.max_tokens,
        coordinate_tokens=args.coordinate_tokens,
        detach_picard_coordinates=args.detach_picard_coordinates,
    )
    print(
        f"compute-light budget={args.budget_minutes:.1f} minutes "
        f"frames_per_shot={args.frames_per_shot}",
        flush=True,
    )
    prepared_paths = prepare_local_shards(
        args.input_dir,
        args.cache_dir,
        config,
        n_shots=args.n_shots,
        frames_per_shot=args.frames_per_shot,
        workers=args.prepare_workers,
        compressed=False,
    )
    remaining = deadline - time.monotonic()
    if remaining <= 15.0 * 60.0:
        raise RuntimeError(
            f"preparation left only {remaining / 60.0:.1f} minutes; refusing an indefensible "
            "compute-light training claim"
        )
    pretrain_deadline = min(
        time.monotonic() + 60.0 * args.pretrain_minutes,
        deadline - 10.0 * 60.0,
    )
    pretrain_result = train_model(
        cache_dir=args.cache_dir,
        artifact=args.stage_a_artifact,
        config=config,
        stage="pretrain",
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate_pretrain,
        n_val_shots=args.n_val_shots,
        seed=args.seed,
        device_name=args.device,
        channel_dropout=0.10,
        num_workers=args.loader_workers,
        shot_batches=True,
        # Rare full-dataset Thomson payloads can overflow encoder gradients on
        # V100 FP16 even when sampled training is stable. Keep Stage A in FP32.
        amp=False,
        deadline=pretrain_deadline,
    )
    if pretrain_result["epochs_completed"] < 1:
        raise RuntimeError("Stage A did not complete one all-shot epoch inside its budget")
    finetune_result = train_model(
        cache_dir=args.cache_dir,
        artifact=args.artifact,
        config=config,
        stage="finetune",
        epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate_finetune,
        n_val_shots=args.n_val_shots,
        seed=args.seed,
        device_name=args.device,
        resume=args.stage_a_artifact,
        channel_dropout=0.10,
        num_workers=args.loader_workers,
        shot_batches=True,
        # The learned encoder is safe in FP16 during teacher-forced Stage A,
        # but end-to-end GS gradients overflow float16 on V100 hardware.
        amp=False,
        deadline=deadline,
    )
    if finetune_result["epochs_completed"] < 1:
        raise RuntimeError("Stage B did not complete one all-shot epoch inside the total budget")
    elapsed = time.monotonic() - started
    checkpoint = torch.load(args.artifact, map_location="cpu", weights_only=False)
    manifest = json.loads((args.cache_dir / "config.json").read_text())
    prepared_frames = int(manifest["preparation"]["frames"])
    checkpoint["metadata"]["compute_light"] = {
        "end_to_end_elapsed_seconds": elapsed,
        "budget_minutes": args.budget_minutes,
        "internal_under_two_hours": elapsed < 2.0 * 60.0 * 60.0,
        "frames_per_shot": args.frames_per_shot,
        "raw_shots": len(prepared_paths),
        "prepared_frames": prepared_frames,
        "full_data_fit": args.frames_per_shot == 0 and args.n_shots == 0 and args.n_val_shots == 0,
        "pretrain": pretrain_result,
        "finetune": finetune_result,
    }
    torch.save(checkpoint, args.artifact)
    report = args.artifact.with_name(f"{args.artifact.stem}_compute_light.json")
    report.write_text(json.dumps(checkpoint["metadata"]["compute_light"], indent=2) + "\n")
    print(f"compute-light end-to-end elapsed={elapsed / 60.0:.2f} minutes", flush=True)
    print(f"artifact={args.artifact}", flush=True)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "compute-light":
        return run_compute_light(args)
    config = NeuralGSConfig(
        outer_iterations=getattr(args, "outer_iterations", 3),
        d_model=getattr(args, "d_model", 128),
        n_heads=getattr(args, "n_heads", 4),
        n_attention_blocks=getattr(args, "attention_blocks", 3),
        source_stride=getattr(args, "source_stride", 2),
        max_tokens=getattr(args, "max_tokens", 192),
        coordinate_tokens=getattr(args, "coordinate_tokens", False),
        global_feature_mode=getattr(args, "global_feature_mode", "legacy"),
        physical_tf_turns=getattr(args, "physical_tf_turns", False),
        use_scalar_calibration=not getattr(args, "disable_scalar_calibration", False),
        detach_picard_coordinates=getattr(args, "detach_picard_coordinates", False),
        exact_final_source_render=getattr(args, "exact_final_source_render", False),
        separatrix_span_scale=getattr(args, "separatrix_span_scale", 1.0),
        separatrix_estimator=getattr(args, "separatrix_estimator", "saddle"),
        axis_estimator=getattr(args, "axis_estimator", "current_local"),
        initial_current_center=getattr(args, "initial_current_center", "grid"),
        spatial_current_modes=getattr(args, "spatial_current_modes", 0),
        spatial_current_amplitude=getattr(args, "spatial_current_amplitude", 0.75),
        source_head_mode=getattr(args, "source_head_mode", "legacy_pressure_mixture"),
        source_basis_size=getattr(args, "source_basis_size", 4),
        physics_raster_encoder=getattr(args, "physics_raster_encoder", False),
        raster_hidden_channels=getattr(args, "raster_hidden_channels", 24),
        raster_coordinate_mode=getattr(args, "raster_coordinate_mode", "cylindrical"),
    )
    if args.command == "prepare":
        prepare_local_shards(
            args.input_dir,
            args.output_dir,
            config,
            args.skip,
            args.n_shots,
            frames_per_shot=args.frames_per_shot,
            workers=args.workers,
            compressed=not args.uncompressed,
        )
        return 0
    deadline = time.monotonic() + 60.0 * args.max_minutes if args.max_minutes > 0 else None
    train_model(
        cache_dir=args.cache_dir,
        artifact=args.artifact,
        config=config,
        stage=args.command,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_val_shots=args.n_val_shots,
        seed=args.seed,
        device_name=args.device,
        resume=args.resume,
        channel_dropout=args.channel_dropout,
        frame_stride=args.frame_stride,
        num_workers=args.num_workers,
        shot_batches=args.shot_batches,
        amp=args.amp,
        deadline=deadline,
        gradient_weight=args.gradient_weight,
        topology_weight=args.topology_weight,
        scalar_weight=args.scalar_weight,
        oracle_dir=args.oracle_dir,
        split_manifest=args.split_manifest,
        source_transform=args.source_transform,
        freeze_source_probe=args.freeze_source_probe,
        freeze_raster_probe=args.freeze_raster_probe,
        source_green_weight=args.source_green_weight,
        source_current_weight=args.source_current_weight,
        source_smooth_weight=args.source_smooth_weight,
        source_negative_weight=args.source_negative_weight,
        source_coordinate_augmentation=args.source_coordinate_augmentation,
        gauge_weight=args.gauge_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
