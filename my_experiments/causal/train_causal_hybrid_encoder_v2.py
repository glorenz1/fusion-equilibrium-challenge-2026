#!/usr/bin/env python3
"""Fine-tune V2 on exact-decoder multi-frame labels with regime-aware selection."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_score import (  # noqa: E402
    CONS_SCALARS,
    N_CONS,
    N_SCALARS,
    PSI_SIGNS,
    SCALARS,
    Accum,
    build_reference,
    finalize_machine,
    psi_residuals,
    score_shot,
)
from my_experiments.causal.train_causal_hybrid_encoder import (  # noqa: E402
    CausalBridgeDataset,
    FrozenSpatialDecoder,
    evaluate_predictions,
    fit_input_normalizer,
    fit_target_scales,
    initialize_physical_heads,
    load_npz,
    move_batch,
    predicted_scalars,
    run_epoch,
    select_target_frame,
    sha256_file,
)
from FEC1.encoder.model import CausalHybridEncoderConfig, CausalHybridEncoderV1  # noqa: E402


DEFAULT_DATASET = ROOT / "my_experiments/causal_hybrid_encoder_v2_dataset"
DEFAULT_INITIAL = ROOT / "my_experiments/causal_hybrid_encoder_v1/model.pt"
DEFAULT_OUTPUT = ROOT / "my_experiments/causal_hybrid_encoder_v2"
REGIME_NAMES = (
    "short_history",
    "low_current",
    "high_q95",
    "low_betaN",
    "low_flux_energy",
    "reverse_current",
)
CHALLENGE1_MINIMUM_S = 0.999


def _model_config(payload: dict[str, Any]) -> CausalHybridEncoderConfig:
    valid = {item.name for item in fields(CausalHybridEncoderConfig)}
    values = {key: value for key, value in payload.items() if key in valid}
    if "spatial_widths" in values:
        values["spatial_widths"] = tuple(values["spatial_widths"])
    return CausalHybridEncoderConfig(**values)


@torch.no_grad()
def collect_predictions(
    model: CausalHybridEncoderV1,
    loader: DataLoader,
    decoder: FrozenSpatialDecoder,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    keys = (
        "field",
        "target_field",
        "q95",
        "target_q95",
        "betaN",
        "target_betaN",
        "mass",
        "target_mass",
        "regime_flags",
        "shot_index",
        "source_index",
    )
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model(
            batch["spatial_values"],
            batch["shared_scalars"],
            batch["thomson_values"],
            batch["thomson_attention_mask"],
            batch["delta_time_ms"],
            sequence_mask=batch["sequence_mask"],
        )
        mass = select_target_frame(output.current_mass, batch)
        q95, beta = predicted_scalars(output, batch)
        field = decoder.direct_flux(
            batch["active_psi"],
            batch["ip_abs"],
            select_target_frame(output.normalized_flux_residual, batch),
        )
        values = {
            "field": field,
            "target_field": batch["target_psi"],
            "q95": q95,
            "target_q95": batch["q95"],
            "betaN": beta,
            "target_betaN": batch["betaN"],
            "mass": mass,
            "target_mass": batch["current_mass"],
            "regime_flags": batch["regime_flags"],
            "shot_index": batch["shot_index"],
            "source_index": batch["source_index"],
        }
        for key, value in values.items():
            pieces[key].append(value.detach().cpu().numpy())
    return {key: np.concatenate(value) for key, value in pieces.items()}


def _r2(reference: np.ndarray, prediction: np.ndarray) -> float | None:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    valid = np.isfinite(reference) & np.isfinite(prediction)
    if valid.sum() < 2:
        return None
    target = reference[valid]
    denominator = np.square(target - target.mean()).sum()
    if denominator <= 1.0e-20:
        return None
    return float(
        1.0 - np.square(prediction[valid] - target).sum() / denominator
    )


def regime_metrics(values: dict[str, np.ndarray]) -> dict[str, Any]:
    target = values["target_field"].astype(np.float64)
    prediction = values["field"].astype(np.float64)
    residual = np.square(prediction - target).sum(axis=(1, 2))
    centered = np.square(target - target.mean(axis=(1, 2), keepdims=True)).sum(
        axis=(1, 2)
    )
    per_frame = 1.0 - residual / np.maximum(centered, 1.0e-20)

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        count = int(mask.sum())
        if not count:
            return {"frames": 0}
        selected_target = target[mask]
        denominator = np.square(selected_target - selected_target.mean()).sum()
        current = 1.0 - 0.5 * np.abs(
            values["mass"][mask] - values["target_mass"][mask]
        ).sum(axis=(1, 2))
        return {
            "frames": count,
            "aggregate_flux_r2": float(
                1.0 - residual[mask].sum() / max(float(denominator), 1.0e-20)
            ),
            "mean_per_frame_flux_r2": float(per_frame[mask].mean()),
            "p10_per_frame_flux_r2": float(np.percentile(per_frame[mask], 10.0)),
            "minimum_per_frame_flux_r2": float(per_frame[mask].min()),
            "mean_current_similarity_1_minus_tv": float(current.mean()),
            "q95_r2": _r2(values["target_q95"][mask], values["q95"][mask]),
            "betaN_r2": _r2(values["target_betaN"][mask], values["betaN"][mask]),
        }

    flags = values["regime_flags"].astype(bool)
    masks = {name: flags[:, index] for index, name in enumerate(REGIME_NAMES)}
    masks["rare_union"] = flags.any(axis=1)
    masks["nominal_complement"] = ~flags.any(axis=1)
    masks["all"] = np.ones(len(flags), dtype=bool)
    slices = {name: summarize(mask) for name, mask in masks.items()}
    rare_values = [
        slices[name]["aggregate_flux_r2"]
        for name in REGIME_NAMES
        if slices[name]["frames"] >= 2
    ]
    return {
        "slices": slices,
        "rare_regime_flux_floor": float(min(rare_values)),
    }


def build_official_context(values: dict[str, np.ndarray]) -> dict[str, Any]:
    mask = np.load(ROOT / "fusion_scoring/masks/d3d_envelope.npz")
    grid_r = mask["grid_R"]
    grid_z = mask["grid_Z"]
    mask_coarse = mask["mask_coarse"].astype(bool)
    mask_f = mask_coarse.astype(np.float64)
    groups = []
    psi_sum = 0.0
    psi_sumsq = 0.0
    psi_count = 0
    scalar_sum = np.zeros(N_SCALARS)
    scalar_sumsq = np.zeros(N_SCALARS)
    scalar_count = np.zeros(N_SCALARS)
    consistency_sum = np.zeros(N_CONS)
    consistency_sumsq = np.zeros(N_CONS)
    consistency_count = np.zeros(N_CONS)
    for shot_index in np.unique(values["shot_index"]):
        selected = np.flatnonzero(values["shot_index"] == shot_index)
        # V2 targets are in the decoder's canonical convention. The official
        # DIII-D reference convention is the opposite global sign.
        psi = -values["target_field"][selected].astype(np.float64)
        gt = {
            "psi": psi,
            "q95": values["target_q95"][selected].astype(np.float64),
            "betaN": values["target_betaN"][selected].astype(np.float64),
        }
        reference = build_reference(
            psi, grid_r, grid_z, mask_coarse, mask_f
        )
        groups.append((selected, gt, reference))
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
    return {
        "grid_r": grid_r,
        "grid_z": grid_z,
        "mask_coarse": mask_coarse,
        "mask_f": mask_f,
        "groups": groups,
        "reference_stats": reference_stats,
        "means": means,
    }


def official_score(
    values: dict[str, np.ndarray], context: dict[str, Any]
) -> dict[str, Any]:
    sign_totals = {sign: 0.0 for sign in PSI_SIGNS}
    predictions = []
    for selected, gt, _ in context["groups"]:
        prediction = {
            "psirz": -values["field"][selected].astype(np.float64),
            "q95": values["q95"][selected].astype(np.float64),
            "betaN": values["betaN"][selected].astype(np.float64),
        }
        predictions.append(prediction)
        residuals, _, _ = psi_residuals(
            gt["psi"], prediction["psirz"], context["means"][0]
        )
        for sign in PSI_SIGNS:
            sign_totals[sign] += residuals[sign]
    psi_sign = min(PSI_SIGNS, key=lambda sign: sign_totals[sign])
    accumulator = Accum()
    accumulator.psi_sign = psi_sign
    for (_, gt, reference), prediction in zip(context["groups"], predictions):
        accumulator.add(
            score_shot(
                gt,
                reference,
                prediction,
                context["grid_r"],
                context["grid_z"],
                context["mask_coarse"],
                context["mask_f"],
                psi_sign,
                context["means"],
            )
        )
    result = finalize_machine(accumulator, context["reference_stats"])
    return {
        key: (
            value.item()
            if isinstance(value, np.generic)
            else (
                {subkey: (subvalue.item() if isinstance(subvalue, np.generic) else subvalue)
                 for subkey, subvalue in value.items()}
                if isinstance(value, dict)
                else value
            )
        )
        for key, value in result.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_DATASET / "inputs.npz")
    parser.add_argument("--bridge", type=Path, default=DEFAULT_DATASET / "bridge.npz")
    parser.add_argument("--dataset-report", type=Path, default=DEFAULT_DATASET / "report.json")
    parser.add_argument("--initialize", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument(
        "--v4-predictions",
        type=Path,
        help=(
            "use preserved target-free V4 predictions as the direct-flux base "
            "and as two broadcast scalar features"
        ),
    )
    parser.add_argument(
        "--target-psi-n",
        type=Path,
        help="offline exact normalized-flux targets for topology-aware training",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--candidate-count", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--current-kl-weight", type=float, default=0.0)
    parser.add_argument("--moment-weight", type=float, default=0.0)
    parser.add_argument("--harmonic-weight", type=float, default=0.0)
    parser.add_argument("--gauge-weight", type=float, default=0.0)
    parser.add_argument("--q95-weight", type=float, default=0.3)
    parser.add_argument("--betaN-weight", type=float, default=0.3)
    parser.add_argument("--flux-weight", type=float, default=1.0)
    parser.add_argument("--direct-residual-weight", type=float, default=0.5)
    parser.add_argument("--gradient-weight", type=float, default=0.15)
    parser.add_argument("--topology-weight", type=float, default=0.0)
    parser.add_argument("--geometry-weight", type=float, default=0.0)
    parser.add_argument("--physical-flux-weight", type=float, default=0.0)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.candidate_count) < 1:
        raise ValueError("training counts must be positive")
    dataset_report = json.loads(args.dataset_report.read_text())
    required_truths = {
        "complete": True,
        "target_columns_in_encoder_inputs": False,
        "challenge1_noncausal_context": True,
        "direct_flux_target_path": True,
        "frozen_audit_used": False,
    }
    for key, expected in required_truths.items():
        if dataset_report.get(key) is not expected:
            raise RuntimeError(f"V2 dataset provenance check failed: {key}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    checkpoint_path = args.output_dir / "model.pt"
    if report_path.exists() and not args.overwrite:
        print(f"completed V2 report exists at {report_path}; exiting", flush=True)
        return 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    inputs = load_npz(args.inputs)
    bridge = load_npz(args.bridge)
    source_indices = inputs["source_index"].astype(np.int64)
    if not np.array_equal(source_indices, np.arange(len(source_indices))):
        raise RuntimeError("V2 source indices are not dense and aligned")
    if not np.array_equal(inputs["shot_name"].astype(str), bridge["shot_name"].astype(str)):
        raise RuntimeError("V2 input and label shot names differ")
    v4_provenance = None
    if args.v4_predictions is not None:
        with np.load(args.v4_predictions, allow_pickle=False) as saved:
            for key, expected in (
                ("data_policy", "real_diii_d_only"),
                ("mast_data_used", False),
                ("target_columns_used", False),
            ):
                if saved[key].item() != expected:
                    raise RuntimeError(f"V4 prediction provenance rejected: {key}")
            order = np.argsort(np.asarray(saved["sample_index"], dtype=np.int64))
            sample_index = np.asarray(saved["sample_index"], dtype=np.int64)[order]
            expected_index = np.arange(len(inputs["source_index"]), dtype=np.int64)
            if not np.array_equal(sample_index, expected_index):
                raise RuntimeError("V4 predictions do not cover the dense V2 sample index")
            if not np.array_equal(
                np.asarray(saved["shot_name"])[order].astype(str),
                inputs["shot_name"].astype(str),
            ) or not np.array_equal(
                np.asarray(saved["frame_index"], dtype=np.int64)[order],
                inputs["frame_index"].astype(np.int64),
            ):
                raise RuntimeError("V4 predictions are not aligned to the V2 dataset")
            v4_flux = np.asarray(saved["psirz"], dtype=np.float32)[order]
            v4_q95 = np.asarray(saved["q95"], dtype=np.float32)[order]
            v4_beta = np.asarray(saved["betaN"], dtype=np.float32)[order]
            v4_models = {}
            for name in ("base", "residual", "direct"):
                path_key = f"{name}_artifact"
                hash_key = f"{name}_artifact_sha256"
                if path_key not in saved or hash_key not in saved:
                    raise RuntimeError(f"V4 prediction archive lacks {name} model binding")
                model_path = Path(str(saved[path_key].item()))
                expected_hash = str(saved[hash_key].item())
                if not model_path.is_file() or sha256_file(model_path) != expected_hash:
                    raise RuntimeError(f"V4 {name} model binding fails its checksum")
                v4_models[name] = {
                    "artifact": str(model_path.resolve()),
                    "sha256": expected_hash,
                }
        if v4_flux.shape != inputs["active_psi_Wb_per_rad"].shape:
            raise RuntimeError("V4 direct-flux base shape differs from the V2 dataset")
        if not all(np.isfinite(value).all() for value in (v4_flux, v4_q95, v4_beta)):
            raise RuntimeError("V4 direct predictions contain non-finite values")
        inputs["active_psi_Wb_per_rad"] = v4_flux
        bridge["base_q95"] = v4_q95
        bridge["base_betaN"] = v4_beta
        v4_provenance = {
            "artifact": str(args.v4_predictions.resolve()),
            "sha256": sha256_file(args.v4_predictions),
            "flux_base": True,
            "scalar_residual_base": True,
            "broadcast_scalar_features": [],
            "models": v4_models,
        }
    target_psi_n_provenance = None
    if args.target_psi_n is not None:
        with np.load(args.target_psi_n, allow_pickle=False) as saved:
            if (
                saved["data_policy"].item() != "real_diii_d_only"
                or saved["mast_data_used"].item() is not False
                or saved["inference_input"].item() is not False
            ):
                raise RuntimeError("normalized-flux target provenance rejected")
            order = np.argsort(np.asarray(saved["sample_index"], dtype=np.int64))
            sample_index = np.asarray(saved["sample_index"], dtype=np.int64)[order]
            if not np.array_equal(
                sample_index, np.arange(len(inputs["source_index"]), dtype=np.int64)
            ):
                raise RuntimeError("normalized-flux targets do not cover V2 samples")
            if not np.array_equal(
                np.asarray(saved["shot_name"])[order].astype(str),
                inputs["shot_name"].astype(str),
            ) or not np.array_equal(
                np.asarray(saved["frame_index"], dtype=np.int64)[order],
                inputs["frame_index"].astype(np.int64),
            ):
                raise RuntimeError("normalized-flux targets are not aligned")
            bridge["target_psi_n"] = np.asarray(
                saved["target_psi_n"], dtype=np.float32
            )[order]
        target_psi_n_provenance = {
            "artifact": str(args.target_psi_n.resolve()),
            "sha256": sha256_file(args.target_psi_n),
            "offline_target_only": True,
        }
    split = bridge["split"].astype(str)
    train_positions = np.flatnonzero(split == "train")
    development_positions = np.flatnonzero(split == "development")
    frozen_audit_positions = np.flatnonzero(split == "frozen_audit")
    if not len(frozen_audit_positions):
        raise RuntimeError("a sealed DIII-D frozen-audit split is required")
    normalizer = fit_input_normalizer(inputs, train_positions)
    target_scales = fit_target_scales(bridge, train_positions)
    train_dataset = CausalBridgeDataset(inputs, bridge, train_positions, normalizer)
    development_dataset = CausalBridgeDataset(
        inputs, bridge, development_positions, normalizer
    )
    frozen_audit_dataset = CausalBridgeDataset(
        inputs, bridge, frozen_audit_positions, normalizer
    )
    sample_weights = torch.from_numpy(
        inputs["sample_weight"][train_positions].astype(np.float64)
    )
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_positions),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.loader_workers,
        pin_memory=device.type == "cuda",
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=args.batch_size,
        num_workers=args.loader_workers,
        pin_memory=device.type == "cuda",
    )
    frozen_audit_loader = DataLoader(
        frozen_audit_dataset,
        batch_size=args.batch_size,
        num_workers=args.loader_workers,
        pin_memory=device.type == "cuda",
    )
    initial = None
    if args.initialize.is_file():
        initial = torch.load(args.initialize, map_location="cpu", weights_only=False)
        initial_config = _model_config(initial["model_config"])
        config_values = asdict(initial_config)
    else:
        config_values = asdict(CausalHybridEncoderConfig())
    config_values.update(
        spatial_channels=int(inputs["spatial_values"].shape[2]),
        scalar_features=int(inputs["shared_scalars"].shape[2]),
        thomson_features=int(inputs["thomson_values"].shape[3]),
        bidirectional_context=True,
    )
    config = CausalHybridEncoderConfig(**config_values)
    model = CausalHybridEncoderV1(config).to(device)
    compatible = {}
    if initial is not None:
        compatible = {
            key: value
            for key, value in initial["model_state"].items()
            if key in model.state_dict() and model.state_dict()[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
    else:
        initialize_physical_heads(model, target_scales)
    if args.v4_predictions is not None:
        # V4 is already the decoded base field/scalar pair. These two residual
        # heads must begin at exact identity even if a compatible trunk was
        # initialized from an earlier encoder checkpoint.
        torch.nn.init.zeros_(model.flux_head[-1].weight)
        torch.nn.init.zeros_(model.flux_head[-1].bias)
        torch.nn.init.zeros_(model.scalar_head[-1].weight)
        torch.nn.init.zeros_(model.scalar_head[-1].bias)
    initialized_from = str(args.initialize.resolve()) if initial is not None else None
    initialized_from_sha256 = sha256_file(args.initialize) if initial is not None else None
    decoder = FrozenSpatialDecoder(bridge).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.03
    )
    weights = {
        "current_kl": args.current_kl_weight,
        "moments": args.moment_weight,
        "harmonic": args.harmonic_weight,
        "gauge": args.gauge_weight,
        "q95": args.q95_weight,
        "betaN": args.betaN_weight,
        "flux": args.flux_weight,
        "direct_residual": args.direct_residual_weight,
        "gradient": args.gradient_weight,
        "topology": args.topology_weight,
        "geometry": args.geometry_weight,
        "physical_flux": args.physical_flux_weight,
    }
    initial_development = run_epoch(
        model,
        development_loader,
        decoder,
        target_scales,
        weights,
        device,
        None,
        use_amp=use_amp,
    )
    history = [{"epoch": 0, "development": initial_development}]
    baseline_candidate = {
        "epoch": 0,
        "development_loss": initial_development["total"],
        "model_state": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
    }
    candidates: list[dict[str, Any]] = [baseline_candidate]
    best_development = float(initial_development["total"])
    stale = 0
    started = time.monotonic()
    print(
        f"epoch=000 development={initial_development['total']:.6f} "
        f"flux={initial_development['flux']:.6f} (protected baseline)",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(
            model,
            train_loader,
            decoder,
            target_scales,
            weights,
            device,
            optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        development = run_epoch(
            model,
            development_loader,
            decoder,
            target_scales,
            weights,
            device,
            None,
            use_amp=use_amp,
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": training,
                "development": development,
            }
        )
        candidates.append(
            {
                "epoch": epoch,
                "development_loss": development["total"],
                "model_state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
        )
        trained_candidates = sorted(
            (item for item in candidates if item["epoch"] != 0),
            key=lambda item: item["development_loss"],
        )[: args.candidate_count]
        candidates = [baseline_candidate, *trained_candidates]
        if development["total"] < best_development - 1.0e-6:
            best_development = development["total"]
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch:03d} train={training['total']:.6f} "
            f"development={development['total']:.6f} flux={development['flux']:.6f} "
            f"best={best_development:.6f}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"early stopping after epoch {epoch}", flush=True)
            break
    official_context = None
    candidate_reports = []
    for candidate in candidates:
        model.load_state_dict(candidate["model_state"], strict=True)
        values = collect_predictions(model, development_loader, decoder, device)
        if official_context is None:
            official_context = build_official_context(values)
        official = official_score(values, official_context)
        regimes = regime_metrics(values)
        regime_floor = float(np.clip(regimes["rare_regime_flux_floor"], -1.0, 1.0))
        selection = float(official["S"])
        candidate_reports.append(
            {
                "epoch": candidate["epoch"],
                "development_loss": candidate["development_loss"],
                "official_composite": official,
                "regime_evaluation": regimes,
                "selection_score": selection,
                "model_state": candidate["model_state"],
            }
        )
        print(
            f"candidate epoch={candidate['epoch']} S={official['S']:.6f} "
            f"regime_floor={regime_floor:.6f} challenge1_S={selection:.6f}",
            flush=True,
        )
    selected = max(candidate_reports, key=lambda item: item["selection_score"])
    best_state = selected.pop("model_state")
    for item in candidate_reports:
        item.pop("model_state", None)
    model.load_state_dict(best_state, strict=True)
    development_evaluation = evaluate_predictions(
        model, development_loader, decoder, device
    )
    frozen_audit_values = collect_predictions(
        model, frozen_audit_loader, decoder, device
    )
    frozen_audit_official = official_score(
        frozen_audit_values,
        build_official_context(frozen_audit_values),
    )
    frozen_audit_regimes = regime_metrics(frozen_audit_values)
    challenge1_acceptance_passed = (
        float(frozen_audit_official["S"]) > CHALLENGE1_MINIMUM_S
    )
    artifact = {
        "model_state": best_state,
        "model_config": asdict(config),
        "model_class": "FEC1.encoder.model.D3DChallenge1Encoder",
        "input_normalizer": normalizer.to_json(),
        "target_scales": target_scales.to_json(),
        "harmonic_fields": torch.from_numpy(bridge["harmonics"].copy()),
        "grid_r_m": torch.from_numpy(bridge["grid_r"].copy()),
        "grid_z_m": torch.from_numpy(bridge["grid_z"].copy()),
        "source_weights_m2": torch.from_numpy(bridge["source_weights"].copy()),
        "best_epoch": selected["epoch"],
        "frozen_audit_official": frozen_audit_official,
        "challenge1_minimum_S_exclusive": CHALLENGE1_MINIMUM_S,
        "challenge1_acceptance_passed": challenge1_acceptance_passed,
        "loss_weights": weights,
        "input_schema": {
            "spatial_channels": config.spatial_channels,
            "scalar_features": config.scalar_features,
            "thomson_features": config.thomson_features,
            "history_frames": int(inputs["spatial_values"].shape[1]),
        },
        "provenance": {
            "version": 2,
            "inputs": str(args.inputs.resolve()),
            "inputs_sha256": sha256_file(args.inputs),
            "bridge": str(args.bridge.resolve()),
            "bridge_sha256": sha256_file(args.bridge),
            "initialized_from": initialized_from,
            "initialized_from_sha256": initialized_from_sha256,
            "initialized_parameter_tensors": len(compatible),
            "v4_parent": v4_provenance,
            "target_psi_n": target_psi_n_provenance,
            "strictly_causal": False,
            "challenge1_noncausal_context": True,
            "supported_machine": "DIII-D",
            "direct_flux_head": True,
            "direct_flux_current_floor_A": decoder.current_floor_A,
            "direct_flux_target_path": True,
            "physical_decoder_auxiliary_only": True,
            "sample_weighted": True,
            "mast_inputs_used": False,
            "mast_targets_used": False,
            "frozen_audit_used_for_selection": False,
        },
    }
    torch.save(artifact, checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    (args.output_dir / "baseline_manifest.json").write_text(
        json.dumps(
            {
                "baseline": f"CausalHybridEncoderV2 epoch {selected['epoch']}",
                "checkpoint": "model.pt",
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "checkpoint_sha256": checkpoint_hash,
                "development_selected_epoch": selected["epoch"],
                "immutable": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    report = {
        "complete": True,
        "experiment": "causal_hybrid_encoder_v2_training",
        "device": str(device),
        "mixed_precision_training": use_amp,
        "seed": args.seed,
        "epochs_completed": len(history),
        "selected_epoch": selected["epoch"],
        "selection_definition": "highest development official DIII-D Challenge-1 composite S",
        "selected_candidate": selected,
        "candidate_evaluations": candidate_reports,
        "development_evaluation": development_evaluation,
        "frozen_audit_official": frozen_audit_official,
        "frozen_audit_regime_evaluation": frozen_audit_regimes,
        "challenge1_minimum_S_exclusive": CHALLENGE1_MINIMUM_S,
        "challenge1_acceptance_passed": challenge1_acceptance_passed,
        "history": history,
        "split_counts": {
            "train_samples": len(train_positions),
            "development_samples": len(development_positions),
            "train_shots": int(len(np.unique(inputs["shot_index"][train_positions]))),
            "development_shots": int(
                len(np.unique(inputs["shot_index"][development_positions]))
            ),
            "frozen_audit_samples": len(frozen_audit_positions),
            "frozen_audit_shots": int(
                len(np.unique(inputs["shot_index"][frozen_audit_positions]))
            ),
        },
        "sample_weighted": True,
        "loss_weights": weights,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "initialized_from": initialized_from,
        "initialized_from_sha256": initialized_from_sha256,
        "initialized_parameter_tensors": len(compatible),
        "v4_parent": v4_provenance,
        "target_psi_n": target_psi_n_provenance,
        "inputs": str(args.inputs.resolve()),
        "inputs_sha256": sha256_file(args.inputs),
        "bridge": str(args.bridge.resolve()),
        "bridge_sha256": sha256_file(args.bridge),
        "strictly_causal": False,
        "challenge1_noncausal_context": True,
        "supported_machine": "DIII-D",
        "direct_flux_head": True,
        "direct_flux_current_floor_A": decoder.current_floor_A,
        "target_columns_in_encoder_inputs": False,
        "direct_flux_target_path": True,
        "physical_decoder_auxiliary_only": True,
        "frozen_d3d_audit_used_for_training_or_selection": False,
        "mast_inputs_used": False,
        "mast_targets_used": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "selected_epoch": selected["epoch"],
                "development_S": selected["official_composite"]["S"],
                "development_regime_floor": selected["regime_evaluation"][
                    "rare_regime_flux_floor"
                ],
                "frozen_audit_S": frozen_audit_official["S"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not challenge1_acceptance_passed:
        raise RuntimeError(
            "sealed DIII-D Challenge-1 acceptance failed: "
            f"S={frozen_audit_official['S']} must exceed {CHALLENGE1_MINIMUM_S}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
