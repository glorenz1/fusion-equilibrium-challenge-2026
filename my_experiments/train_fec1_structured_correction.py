#!/usr/bin/env python3
"""Train a correction on inner OOF errors; score unseen outer shots once."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from my_experiments.fec1_structured_correction import StructuredCorrection, coordinates, calibrated_prediction, calibrated_prediction_torch, FrozenScalarPredictor, within_shot_neighbors
from my_experiments.fec1_symmetric_geometry import symmetric_geometry_losses, geometry_objective
from my_experiments.prepare_fec1_structured_crossfit import ARTIFACTS, SPATIAL, row_digest
from my_experiments.fec1_conv_autoencoder import teacher_normalized_flux
from my_experiments.fec1_direct_geometry import direct_raster_geometry_losses
from my_experiments.train_fec1_score_aligned_temporal import _align_topology, _align_contours
from my_experiments.score_fec1_prediction_archive import _parallel_official_score
from my_experiments.fit_fec1_full_frame_pca import CACHE, _coordinates
from src.equilibrium_inverter.diiid_hybrid.scorer_geometry import ScorerGeometryLoss, smooth_backward


def load_training_array(path, preload=True):
    """Read sequentially into RAM to avoid random Lustre faults every batch."""
    started = time.monotonic()
    print(json.dumps({"stage": "load_array_start", "path": str(path),
                      "preloaded": preload}), flush=True)
    array = np.load(path, mmap_mode=None if preload else "r")
    print(json.dumps({"stage": "load_array", "path": str(path),
                      "preloaded": preload, "bytes": array.nbytes,
                      "elapsed_seconds": time.monotonic() - started}), flush=True)
    return array


def save_checkpoint_atomic(checkpoint, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


class FoldData:
    def __init__(self, path, preload=True):
        self.path = Path(path)
        self.report = json.loads((self.path / "ready.json").read_text())
        if (self.report["status"] != "complete" or not self.report["nested_shot_crossfit"]
                or self.report["development_used"] or self.report["frozen_audit_used"]):
            raise RuntimeError("unsafe nested base provenance")
        with np.load(self.path / "index.npz", allow_pickle=False) as index:
            self.names = index["shot_name"].astype(str)
            self.frames = index["frame_index"]
            self.source = index["source_row"]
            self.role = index["role"]
            self.scalar = index["target_scalar"]
        if row_digest(self.names, self.frames) != self.report["row_digest"]:
            raise RuntimeError("row identity changed")
        for left, right in ((0, 1), (0, 2), (1, 2)):
            if set(self.names[self.role == left]) & set(self.names[self.role == right]):
                raise RuntimeError("shot roles overlap")
        self.features = load_training_array(self.path / "features.npy", preload)
        self.base = load_training_array(self.path / "base_field.npy", preload)
        self.base_scalar = load_training_array(self.path / "base_scalar.npy", preload)
        self.target = load_training_array(SPATIAL / "target_field.npy", preload)
        self.token_values = load_training_array(SPATIAL / "token_values.npy", preload)
        self.token_coords = load_training_array(SPATIAL / "token_coords.npy", preload)
        self.token_mask = load_training_array(SPATIAL / "token_mask.npy", preload)

    def inputs(self, rows, device):
        source = self.source[rows]
        arrays = (self.features[rows], self.base[rows], self.base_scalar[rows],
                  self.token_values[source], self.token_coords[source], self.token_mask[source])
        if hasattr(self, "sequence_neighbors"):
            arrays += (self.features[self.sequence_neighbors[rows], :self.features.shape[1]//3],
                       self.sequence_mask[rows])
        return tuple(torch.as_tensor(np.array(a), device=device) for a in arrays)


@torch.no_grad()
def initialize_statistics(model, data, rows, device, seed):
    if np.any(data.role[rows] != 0):
        raise RuntimeError("statistics must use correction-training rows only")
    rng = np.random.default_rng(seed)
    selected = rng.choice(rows, min(12000, len(rows)), replace=False)
    feature = np.asarray(data.features[selected])
    model.feature_mean.copy_(torch.from_numpy(feature.mean(0)).to(device))
    model.feature_scale.copy_(torch.from_numpy(feature.std(0).clip(1e-5)).to(device))
    base_parts, delta_parts, coefficients = [], [], []
    token_sum = torch.zeros(7, device=device, dtype=torch.float64)
    token_square = torch.zeros_like(token_sum)
    token_count = 0
    shape_square, shape_count = 0., 0
    for start in range(0, len(selected), 256):
        batch = selected[start:start + 256]
        _, base, scalar, values, coords, mask = data.inputs(batch, device)[:6]
        target = torch.as_tensor(np.array(data.target[data.source[batch]]), device=device)
        base_coord, base_shape = coordinates(base)
        target_coord, target_shape = coordinates(target)
        target_scalar = torch.as_tensor(data.scalar[batch], device=device)
        delta_shape = target_shape - base_shape
        base_parts.append(torch.cat((base_coord, scalar), -1))
        delta_parts.append(torch.cat((target_coord - base_coord, target_scalar - scalar), -1))
        coefficients.append(delta_shape.flatten(1) @ model.basis.T)
        shape_square += float(delta_shape.square().sum())
        shape_count += delta_shape.numel()
        tokens = torch.cat((values, coords), -1)[mask.bool()].double()
        token_sum += tokens.sum(0)
        token_square += tokens.square().sum(0)
        token_count += len(tokens)
    base = torch.cat(base_parts)
    delta = torch.cat(delta_parts)
    coefficient = torch.cat(coefficients)
    model.base_mean.copy_(base.mean(0))
    model.base_scale.copy_(base.std(0, unbiased=False).clamp_min(1e-5))
    model.coordinate_scale.copy_(delta[:, :2].square().mean(0).sqrt().clamp_min(1e-5))
    model.scalar_scale.copy_(delta[:, 2:].square().mean(0).sqrt().clamp_min(1e-4))
    model.coefficient_scale.copy_(coefficient.square().mean(0).sqrt().clamp_min(1e-5))
    model.shape_scale.fill_(max((shape_square / shape_count) ** 0.5, 1e-5))
    mean = token_sum / max(token_count, 1)
    std = (token_square / max(token_count, 1) - mean.square()).clamp_min(1e-10).sqrt()
    model.token_mean.copy_(mean.float())
    model.token_scale.copy_(std.float())


def objective(model, output, target, scalar, geometry=None, geometry_weight=0.,
              loss_version="legacy", geometry_components="both", fixed_scalars=False,
              base_field=None, frozen_strengths=None, scorer_context=None):
    target_coord, target_shape = coordinates(target)
    shape_error = (output["shape"] - target_shape) / model.shape_scale
    shape_loss = shape_error.square().mean()
    gradient = (shape_error[..., 1:] - shape_error[..., :-1]).square().mean()
    gradient += (shape_error[..., 1:, :] - shape_error[..., :-1, :]).square().mean()
    coordinate_loss = ((output["coordinate"] - target_coord) / model.coordinate_scale).square().mean()
    scalar_loss = ((output["scalar"] - scalar) / model.scalar_scale).square().mean()
    target_coefficient = (target_shape - output["base_shape"]).flatten(1) @ model.basis.T
    coefficient_loss = ((output["coefficient"] - target_coefficient)
                        / model.coefficient_scale).square().mean()
    loss = shape_loss + 0.1 * coefficient_loss + 0.25 * gradient
    loss = loss + 0.5 * coordinate_loss
    if not fixed_scalars:
        loss = loss + 0.5 * scalar_loss
    terms = dict(shape=shape_loss, coefficient=coefficient_loss, gradient=gradient,
                 coordinate=coordinate_loss, scalar=scalar_loss)
    if scorer_context is not None and geometry_weight > 0:
        if loss_version != "scorer_implicit" or geometry_components != "both":
            raise ValueError("scorer context requires the complete scorer_implicit objective")
        if base_field is None or frozen_strengths is None:
            raise ValueError("scorer contour loss requires the frozen calibrated field")
        r, z, envelope, statistics, names, workers, coefficients, backward_sigma, contour_backward, contour_temperature = scorer_context
        count = len(names)
        calibrated_field, _ = calibrated_prediction_torch(
            base_field[:count], torch.zeros_like(output["scalar"][:count]),
            output["field"][:count], output["scalar"][:count], frozen_strengths)
        exact = ScorerGeometryLoss(target[:count], r, z, envelope,
                                   shot_names=names, workers=workers,
                                   statistics=statistics,
                                   contour_backward=contour_backward,
                                   contour_temperature=contour_temperature)
        consistency, contour, _ = exact(smooth_backward(calibrated_field, backward_sigma))
        geometry_loss = coefficients[0] * consistency + coefficients[1] * contour
        loss = loss + geometry_weight * geometry_loss
        terms.update(geometry=geometry_loss, exact_consistency_error=consistency,
                     exact_lcfs=contour)
    elif geometry is not None and geometry_weight > 0:
        target_n, topology, radius, angles, grid_r, grid_z = geometry
        count = len(target_n)
        geometry_shape = output["shape"][:count]
        if frozen_strengths is not None:
            if base_field is None:
                raise ValueError("calibrated geometry requires the same batch's base field")
            calibrated_field, _ = calibrated_prediction_torch(
                base_field[:count], torch.zeros_like(output["scalar"][:count]),
                output["field"][:count], output["scalar"][:count], frozen_strengths)
            geometry_shape = coordinates(calibrated_field)[1]
        if loss_version == "legacy":
            predicted_n, useful = teacher_normalized_flux(geometry_shape, target_n)
            parts = direct_raster_geometry_losses(predicted_n, target_n, useful, topology,
                                                  grid_r, grid_z, radius, angles)
        elif loss_version == "symmetric":
            parts = symmetric_geometry_losses(geometry_shape, target_shape[:count],
                                               target_n, grid_r, grid_z, angles=angles)
        else:
            raise ValueError("unknown geometry loss version")
        geometry_loss = geometry_objective(parts, geometry_components)
        loss = loss + geometry_weight * geometry_loss
        terms["geometry"] = geometry_loss
        terms.update({k: v for k, v in parts.items() if k != "direct_exact_lcfs"})
    terms["loss"] = loss
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(f"non-finite correction loss: { {k: float(v) for k, v in terms.items()} }")
    return loss, {k: float(v.detach()) for k, v in terms.items()}


@torch.no_grad()
def predict(model, data, rows, device, batch_size):
    model.eval()
    field = np.empty((len(rows), 65, 65), np.float32)
    scalar = np.empty((len(rows), 2), np.float32)
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        result = model(*data.inputs(batch, device))
        field[start:start + len(batch)] = result["field"].cpu().numpy()
        scalar[start:start + len(batch)] = (data.fixed_scalar[batch] if getattr(data, "fixed_scalar", None) is not None
                                           else result["scalar"].cpu().numpy())
    if not np.isfinite(field).all() or not np.isfinite(scalar).all():
        raise RuntimeError("non-finite target-free prediction")
    return field, scalar


def score(data, rows, field, scalar, cache, details_path=None):
    names = data.names[rows]
    values = {"field": field, "q95": scalar[:, 0], "betaN": scalar[:, 1],
              "target_field": np.asarray(data.target[data.source[rows]]),
              "target_q95": data.scalar[rows, 0], "target_betaN": data.scalar[rows, 1],
              "shot_index": np.unique(names, return_inverse=True)[1],
              "shot_name": names, "source_index": data.frames[rows],
              "reference_source": str(CACHE.resolve())}
    return _parallel_official_score(values, getattr(data, "score_workers", 1), cache, details_path)


def select_strengths(data, rows, field, scalar, cache):
    if np.any(data.role[rows] != 1):
        raise RuntimeError("calibration must use inner selection shots only")
    base = np.array(data.base[rows])
    base_scalar = np.array(data.base_scalar[rows])
    true_mean, true_rms, _ = _coordinates(data.target[data.source[rows]])
    base_mean, base_rms, _ = _coordinates(base)
    mean, rms, _ = _coordinates(field)
    predicted_delta = np.column_stack((mean - base_mean, np.log(rms / base_rms), scalar - base_scalar))
    target_delta = np.column_stack((true_mean - base_mean, np.log(true_rms / base_rms),
                                   data.scalar[rows] - base_scalar))
    p, y = predicted_delta.astype(np.float64), target_delta.astype(np.float64)
    alpha = np.clip((p * y).sum(0) / np.maximum((p * p).sum(0), 1e-30), 0., 1.5)
    strengths = {"coordinate": alpha[:2].tolist(), "scalar": alpha[2:].tolist()}
    if getattr(data, "fixed_scalar", None) is not None:
        strengths["scalar"] = [1., 1.]
    trials = []
    for shape_strength in (0., 0.25, 0.5, 0.75, 1.):
        candidate_strengths = {**strengths, "shape": shape_strength}
        corrected, corrected_scalar = calibrated_prediction(base, base_scalar, field, scalar, candidate_strengths)
        result = score(data, rows, corrected, corrected_scalar, cache)
        trials.append({"strengths": candidate_strengths, "score": result})
    return max(trials, key=lambda trial: trial["score"]["S"]), trials


def frozen_strengths_for_fold(path, fold):
    """Read the selected public DCT recipe without refitting on held shots."""
    recipe = json.loads(Path(path).read_text())
    if recipe.get("selected_family") != "dct" or recipe.get("frozen_audit_used"):
        raise ValueError("frozen calibration must be the selected DCT recipe")
    folds = recipe["reports"]["dct"]["folds"]
    matches = [entry for entry in folds if entry["fold"] == fold]
    if len(matches) != 1 or not matches[0].get("nested_shot_crossfit"):
        raise ValueError("missing or unsafe same-fold frozen calibration")
    strengths = matches[0]["correction_strengths"]
    if (len(strengths["coordinate"]) != 2 or len(strengths["scalar"]) != 2
            or not np.isfinite([strengths["shape"], *strengths["coordinate"],
                                *strengths["scalar"]]).all()):
        raise ValueError("invalid frozen calibration strengths")
    return strengths


def score_at_frozen_strengths(data, rows, field, scalar, cache, strengths):
    if np.any(data.role[rows] != 1):
        raise RuntimeError("checkpoint selection must use inner selection shots only")
    field, scalar = calibrated_prediction(np.array(data.base[rows]),
                                          np.array(data.base_scalar[rows]),
                                          field, scalar, strengths)
    result = score(data, rows, field, scalar, cache)
    return {"strengths": strengths, "score": result}, [{"strengths": strengths, "score": result}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoder", choices=("dct", "conv"), required=True)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--geometry-weight", type=float, default=0.003)
    parser.add_argument("--loss-version", choices=("legacy", "symmetric", "scorer_implicit"), default="legacy")
    parser.add_argument("--geometry-components", choices=("both", "boundary", "profile"), default="both")
    parser.add_argument("--frozen-calibration", type=Path,
                        help="selected DCT frozen_selection.json; fixes strengths for training and exact selection")
    parser.add_argument("--geometry-on-calibrated-field", action="store_true",
                        help="compute geometry loss on the field after frozen export calibration")
    parser.add_argument("--contour-step-audit", type=Path,
                        help="passing held-shot local-direction gate for scorer_implicit")
    parser.add_argument("--scalar-reference", type=Path, help="freeze the entire scalar predictor from this same-fold checkpoint")
    parser.add_argument("--initial-checkpoint", type=Path,
                        help="continue from a same-fold structured-correction checkpoint")
    parser.add_argument("--reference-cache-directory", type=Path, help="reuse same-fold scorer reference caches")
    parser.add_argument("--score-workers", type=int, default=1)
    parser.add_argument("--observation-encoder", choices=("pooled", "spatial_attention", "temporal_attention"), default="pooled")
    parser.add_argument("--geometry-batch", type=int, default=16)
    parser.add_argument("--geometry-warmup", type=int, default=4)
    parser.add_argument("--exact-every", type=int, default=6)
    parser.add_argument("--selection-shots", type=int, default=64)
    parser.add_argument("--selection-shot-list", type=Path,
                        help="JSON shot_names list for a predeclared role-1 selection panel")
    parser.add_argument("--selection-dlcfs-gain", type=float, default=1e-4,
                        help="minimum exact LCFS error decrease required for scorer_implicit selection")
    parser.add_argument("--selection-consistency-gain", type=float, default=1e-4,
                        help="minimum exact consistency gain required for scorer_implicit selection")
    parser.add_argument("--selection-score-tolerance", type=float, default=1e-4,
                        help="maximum exact S decrease allowed for scorer_implicit selection")
    parser.add_argument("--patience-checks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=260905)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-map", action="store_true", help="diagnostic only: disable RAM preloading")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.exact_every, args.selection_shots) < 1:
        parser.error("training counts must be positive")
    if (args.learning_rate <= 0 or args.selection_dlcfs_gain < 0
            or args.selection_consistency_gain < 0 or args.selection_score_tolerance < 0):
        parser.error("learning rate and selection thresholds must be nonnegative, with positive learning rate")
    if args.geometry_on_calibrated_field and args.frozen_calibration is None:
        parser.error("--geometry-on-calibrated-field requires --frozen-calibration")
    contour_audit_sha256 = None
    scorer_coefficients = None
    scorer_backward_sigma = None
    scorer_contour_backward = None
    scorer_contour_temperature = None
    if args.loss_version == "scorer_implicit":
        if not args.geometry_on_calibrated_field or args.contour_step_audit is None:
            parser.error("scorer_implicit requires calibrated geometry and --contour-step-audit")
        if args.geometry_components != "both":
            parser.error("scorer_implicit currently requires --geometry-components both")
        audit = json.loads(args.contour_step_audit.read_text())
        scorer_hash = hashlib.sha256((ROOT / "src/equilibrium_inverter/diiid_hybrid/scorer_geometry.py").read_bytes()).hexdigest()
        if (audit.get("complete") is not True or audit.get("robust_gate_passed") is not True
                or audit.get("role") != 1 or audit.get("scorer_geometry_sha256") != scorer_hash
                or audit.get("frozen_audit_used") is not False
                or audit.get("statistics_role0") is not True
                or len(audit.get("gate_shots", [])) < 32):
            parser.error("scorer_implicit requires a robust role-1 LCFS and consistency gate")
        if audit.get("confirmation_of"):
            source_path = Path(audit["confirmation_of"])
            if (not source_path.is_file() or hashlib.sha256(source_path.read_bytes()).hexdigest()
                    != audit.get("confirmation_source_sha256")):
                parser.error("the contour step source gate has changed or is missing")
        for source in audit.get("source_reports", []):
            source_path = Path(source["path"])
            if (not source_path.is_file() or hashlib.sha256(source_path.read_bytes()).hexdigest()
                    != source["sha256"]):
                parser.error("a source report for the pooled contour gate has changed or is missing")
        saved_coefficients = audit.get("gradient_coefficients", {})
        scorer_coefficients = (float(saved_coefficients.get("consistency", -1)),
                               float(saved_coefficients.get("contour", -1)))
        if (not all(np.isfinite(scorer_coefficients)) or min(scorer_coefficients) < 0
                or sum(scorer_coefficients) <= 0):
            parser.error("passing audit must identify its tested scorer-gradient coefficients")
        scorer_backward_sigma = float(audit.get("backward_smoothing_sigma", -1))
        if not np.isfinite(scorer_backward_sigma) or scorer_backward_sigma < 0:
            parser.error("passing audit must identify its tested backward smoothing")
        scorer_contour_backward = audit.get("contour_backward")
        scorer_contour_temperature = float(audit.get("contour_temperature", -1))
        if (scorer_contour_backward not in ("implicit", "predicted_level_residual",
                                            "symmetric_level_residual")
                or not np.isfinite(scorer_contour_temperature) or scorer_contour_temperature <= 0):
            parser.error("passing audit must identify its tested contour backward")
        contour_audit_sha256 = hashlib.sha256(args.contour_step_audit.read_bytes()).hexdigest()
        args.selection_consistency_gain = max(
            args.selection_consistency_gain,
            float(audit.get("minimum_consistency_gain", args.selection_consistency_gain)),
        )
    if args.output.exists():
        raise FileExistsError(f"use a fresh output directory: {args.output}")
    data = FoldData(args.data, preload=not args.memory_map)
    if args.loss_version == "scorer_implicit" and (
            Path(audit.get("base_directory", "")).resolve() != args.data.resolve()
            or audit.get("fold") != data.report["fold"]):
        parser.error("contour step audit must match this exact nested base fold")
    frozen_strengths = (frozen_strengths_for_fold(args.frozen_calibration, data.report["fold"])
                        if args.frozen_calibration else None)
    data.score_workers = args.score_workers
    if args.observation_encoder == "temporal_attention":
        data.sequence_neighbors, data.sequence_mask = within_shot_neighbors(data.names, data.frames)
    train_rows = np.flatnonzero(data.role == 0)
    outer_rows = np.flatnonzero(data.role == 2)
    seed = args.seed + data.report["fold"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    role1_shots = np.unique(data.names[data.role == 1])
    if args.selection_shot_list is None:
        selection_shots = rng.permutation(role1_shots)[:args.selection_shots]
    else:
        selection_spec = json.loads(args.selection_shot_list.read_text())
        selection_shots = np.asarray(
            selection_spec.get("shot_names", selection_spec)
            if isinstance(selection_spec, dict) else selection_spec,
            dtype=str,
        )
        if (len(selection_shots) != args.selection_shots
                or len(np.unique(selection_shots)) != len(selection_shots)
                or not set(selection_shots).issubset(set(role1_shots))):
            parser.error("selection shot list must contain exactly --selection-shots unique role-1 shots")
    selection_rows = np.flatnonzero((data.role == 1) & np.isin(data.names, selection_shots))
    device = torch.device(args.device)
    model = StructuredCorrection(data.features.shape[1], args.decoder, width=args.width,
                                  channels=args.channels, observation_encoder=args.observation_encoder).to(device)
    initialize_statistics(model, data, train_rows, device, seed)
    initial_checkpoint_sha256 = None
    if args.initial_checkpoint is not None:
        initial_checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        if (initial_checkpoint.get("fold") != data.report["fold"]
                or Path(initial_checkpoint.get("base_directory", "")).resolve() != args.data.resolve()
                or initial_checkpoint.get("model_config") != model.config):
            raise RuntimeError("initial checkpoint must match this fold, data and model configuration")
        expected_calibration = (Path(args.frozen_calibration).resolve()
                                if args.frozen_calibration is not None else None)
        saved_calibration = initial_checkpoint.get("frozen_calibration")
        saved_calibration = Path(saved_calibration).resolve() if saved_calibration else None
        if expected_calibration != saved_calibration:
            raise RuntimeError("initial checkpoint uses a different frozen calibration")
        if args.scalar_reference is not None:
            expected_scalar_hash = hashlib.sha256(args.scalar_reference.read_bytes()).hexdigest()
            if initial_checkpoint.get("scalar_reference_sha256") != expected_scalar_hash:
                raise RuntimeError("initial checkpoint uses a different frozen scalar predictor")
        if (frozen_strengths is not None
                and initial_checkpoint.get("correction_strengths") != frozen_strengths):
            raise RuntimeError("initial checkpoint calibration strengths differ from the frozen recipe")
        model.load_state_dict(initial_checkpoint["state_dict"], strict=True)
        initial_checkpoint_sha256 = hashlib.sha256(args.initial_checkpoint.read_bytes()).hexdigest()
        print(json.dumps({"stage": "initialized_from_checkpoint",
                          "path": str(args.initial_checkpoint.resolve()),
                          "sha256": initial_checkpoint_sha256,
                          "selected_epoch": initial_checkpoint.get("selected_epoch")}), flush=True)
    fixed_scalar_checkpoint = None
    scalar_reference_sha256 = None
    if args.scalar_reference:
        fixed_scalar_checkpoint = torch.load(args.scalar_reference, map_location="cpu", weights_only=False)
        if (fixed_scalar_checkpoint["fold"] != data.report["fold"]
                or Path(fixed_scalar_checkpoint["base_directory"]).resolve() != args.data.resolve()):
            raise RuntimeError("scalar reference must use this exact nested base fold")
        scalar_reference_sha256 = hashlib.sha256(args.scalar_reference.read_bytes()).hexdigest()
        frozen_scalar = FrozenScalarPredictor(fixed_scalar_checkpoint).to(device)
        data.fixed_scalar = np.empty_like(data.scalar)
        for start in range(0, len(data.names), args.batch_size):
            rows = np.arange(start, min(start+args.batch_size, len(data.names)))
            data.fixed_scalar[rows] = frozen_scalar(*data.inputs(rows, device)).cpu().numpy()
        model.scalar_head.requires_grad_(False)
        del frozen_scalar
        print(json.dumps({"stage": "frozen_scalars_ready", "reference_sha256": scalar_reference_sha256}), flush=True)
    args.output.mkdir(parents=True)
    cache_directory = args.reference_cache_directory or args.output
    selection_cache = cache_directory / "selection_reference.pkl"
    initial_field, initial_scalar = predict(model, data, selection_rows, device, args.batch_size)
    if frozen_strengths is None:
        initial = score(data, selection_rows, initial_field, initial_scalar, selection_cache)
    else:
        initial = score_at_frozen_strengths(data, selection_rows, initial_field,
                                            initial_scalar, selection_cache, frozen_strengths)[0]["score"]
    del initial_field, initial_scalar
    print(json.dumps({"epoch": 0, "selection_score": initial}), flush=True)
    best_score = float(initial["S"])
    best_epoch = 0
    best_strengths = copy.deepcopy(frozen_strengths) if frozen_strengths is not None else {
        "shape": 0., "coordinate": [0., 0.], "scalar": [0., 0.]}
    if fixed_scalar_checkpoint is not None and frozen_strengths is None:
        best_strengths["scalar"] = [1., 1.]
    best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    history = [{"epoch": 0, "selection_score": initial}]
    if args.loss_version == "scorer_implicit":
        target_n = topology = radius = angles = None
    else:
        target_n = load_training_array(ARTIFACTS / "fec1_train_stride4_target_psi_n.npy",
                                       preload=not args.memory_map)
        topology = _align_topology(ARTIFACTS / "fec1_train_all_frame_stride4_exact_topology.npz",
                                   data.names, data.frames)
        radius, angles, _ = _align_contours(ARTIFACTS / "fec1_train_stride4_exact_contour_radius.npz",
                                           data.names, data.frames)
    with np.load(ROOT / "fusion_scoring/masks/d3d_envelope.npz") as mask:
        grid_r_np, grid_z_np = mask["grid_R"].copy(), mask["grid_Z"].copy()
        envelope = mask["mask_coarse"].astype(bool)
    grid_r = torch.as_tensor(grid_r_np, dtype=torch.float32, device=device)
    grid_z = torch.as_tensor(grid_z_np, dtype=torch.float32, device=device)
    if angles is not None:
        angles = torch.as_tensor(angles, device=device)
    scorer_statistics = None
    if args.loss_version == "scorer_implicit":
        statistics_rows = np.random.default_rng(seed + 491).choice(
            train_rows, min(256, len(train_rows)), replace=False)
        reference_targets = torch.as_tensor(np.array(data.target[data.source[statistics_rows]]))
        reference = ScorerGeometryLoss(reference_targets, grid_r_np, grid_z_np, envelope,
                                       shot_names=data.names[statistics_rows], workers=args.score_workers)
        scorer_statistics = (reference.mean.copy(), reference.std.copy())
        recorded_statistics = audit.get("scorer_statistics", {})
        if (not np.allclose(reference.mean, recorded_statistics.get("mean", []), rtol=0, atol=1e-12)
                or not np.allclose(reference.std, recorded_statistics.get("std", []), rtol=0, atol=1e-12)):
            raise RuntimeError("role-0 scorer statistics differ from the passing contour audit")
        print(json.dumps({"stage": "scorer_geometry_statistics", "role": 0,
                          "rows": len(statistics_rows),
                          "mean": reference.mean.tolist(), "std": reference.std.tolist()}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=min(1e-5, args.learning_rate * 0.05))
    started = time.monotonic()
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals, count = {}, 0
        ordered = rng.permutation(train_rows)
        geometry_weight = args.geometry_weight * min(1., max(0., (epoch - args.geometry_warmup) / 4))
        for start in range(0, len(ordered), args.batch_size):
            batch = ordered[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            inputs = data.inputs(batch, device)
            output = model(*inputs)
            if fixed_scalar_checkpoint is not None:
                output["scalar"] = torch.as_tensor(data.fixed_scalar[batch], device=device)
            target = torch.as_tensor(np.array(data.target[data.source[batch]]), device=device)
            scalar = torch.as_tensor(data.scalar[batch], device=device)
            geo_rows = batch[:args.geometry_batch]
            geometry = None
            if geometry_weight > 0 and args.loss_version != "scorer_implicit":
                geometry = (
                    torch.as_tensor(np.array(target_n[data.source[geo_rows]], dtype=np.float32), device=device),
                    torch.as_tensor(topology[geo_rows], device=device),
                    torch.as_tensor(radius[geo_rows], device=device), angles, grid_r, grid_z,
                )
            scorer_context = None
            if geometry_weight > 0 and args.loss_version == "scorer_implicit":
                scorer_context = (grid_r_np, grid_z_np, envelope, scorer_statistics,
                                  data.names[geo_rows], args.score_workers, scorer_coefficients,
                                  scorer_backward_sigma, scorer_contour_backward,
                                  scorer_contour_temperature)
            loss, terms = objective(model, output, target, scalar, geometry, geometry_weight,
                                    args.loss_version, args.geometry_components,
                                    fixed_scalar_checkpoint is not None,
                                    base_field=inputs[1] if args.geometry_on_calibrated_field else None,
                                    frozen_strengths=frozen_strengths if args.geometry_on_calibrated_field else None,
                                    scorer_context=scorer_context)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.) + value * len(batch)
            count += len(batch)
            if start == 0 or count % (100 * args.batch_size) == 0:
                print(json.dumps({"stage": "training_batch", "epoch": epoch,
                                  "rows_done": count, "rows_total": len(ordered),
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
        scheduler.step()
        entry = {"epoch": epoch, "training": {k: v / count for k, v in totals.items()},
                 "geometry_weight": geometry_weight, "elapsed_seconds": time.monotonic() - started}
        if epoch % args.exact_every == 0 or epoch == args.epochs:
            field, scalar = predict(model, data, selection_rows, device, args.batch_size)
            if frozen_strengths is None:
                calibrated, trials = select_strengths(data, selection_rows, field, scalar, selection_cache)
            else:
                calibrated, trials = score_at_frozen_strengths(
                    data, selection_rows, field, scalar, selection_cache, frozen_strengths)
            selected_score = calibrated["score"]
            entry["selection_score"] = selected_score
            entry["calibration_trials"] = trials
            del field, scalar
            if args.loss_version == "scorer_implicit":
                geometry_gate = (
                    float(initial["dlcfs"]) - float(selected_score["dlcfs"])
                    >= args.selection_dlcfs_gain
                    and float(selected_score["consistency"])
                    >= float(initial["consistency"]) + args.selection_consistency_gain
                    and float(selected_score["S"])
                    >= float(initial["S"]) - args.selection_score_tolerance
                )
                entry["selection_geometry_gate"] = {
                    "passed": geometry_gate,
                    "baseline_dlcfs": float(initial["dlcfs"]),
                    "candidate_dlcfs": float(selected_score["dlcfs"]),
                    "baseline_consistency": float(initial["consistency"]),
                    "candidate_consistency": float(selected_score["consistency"]),
                    "minimum_consistency_gain": args.selection_consistency_gain,
                    "minimum_dlcfs_gain": args.selection_dlcfs_gain,
                    "baseline_S": float(initial["S"]),
                    "candidate_S": float(selected_score["S"]),
                    "maximum_S_decrease": args.selection_score_tolerance,
                }
                if geometry_gate and (best_epoch == 0 or float(selected_score["S"]) > best_score):
                    best_score, best_epoch, stale = float(selected_score["S"]), epoch, 0
                    best_strengths = calibrated["strengths"]
                    best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                else:
                    stale += 1
            elif float(selected_score["S"]) > best_score:
                best_score, best_epoch, stale = float(selected_score["S"]), epoch, 0
                best_strengths = calibrated["strengths"]
                best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            else:
                stale += 1
        history.append(entry)
        print(json.dumps(entry), flush=True)
        checkpoint = {"model_config": model.config, "state_dict": best_state,
                      "base_directory": str(args.data.resolve()), "fold": data.report["fold"],
                      "folds": data.report["folds"], "selected_epoch": best_epoch,
                      "selection_score": best_score, "nested_shot_crossfit": True,
                      "selection_shot_names": selection_shots.tolist(),
                      "selection_geometry_gate_passed": bool(best_epoch > 0) if args.loss_version == "scorer_implicit" else None,
                      "correction_strengths": best_strengths,
                      "frozen_calibration": str(args.frozen_calibration) if args.frozen_calibration else None,
                      "geometry_on_calibrated_field": args.geometry_on_calibrated_field,
                      "loss_version": args.loss_version, "geometry_components": args.geometry_components,
                      "contour_audit_sha256": contour_audit_sha256,
                      "initial_checkpoint_sha256": initial_checkpoint_sha256,
                      "scorer_coefficients": scorer_coefficients,
                      "scorer_backward_sigma": scorer_backward_sigma,
                      "scorer_contour_backward": scorer_contour_backward,
                      "scorer_contour_temperature": scorer_contour_temperature,
                      "fixed_scalar_checkpoint": fixed_scalar_checkpoint,
                      "scalar_reference_sha256": scalar_reference_sha256,
                      "development_used": False, "frozen_audit_used": False,
                      "data_policy": "real_diii_d_only", "smoke_only": data.report["smoke_only"]}
        save_checkpoint_atomic(checkpoint, args.output / "checkpoint.pt")
        # Keep the latest learned state even before the next expensive scorer
        # checkpoint, so an interrupted run does not discard its training.
        save_checkpoint_atomic({**checkpoint, "state_dict": model.state_dict(), "epoch": epoch,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                    "best_checkpoint": checkpoint, "history": history, "stale_checks": stale,
                    "training_arguments": {k: str(v) if isinstance(v, Path) else v
                                           for k, v in vars(args).items()},
                    "row_digest": data.report["row_digest"]}, args.output / "latest.pt")
        (args.output / "training.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale >= args.patience_checks:
            break
    model.load_state_dict(best_state)
    if frozen_strengths is not None:
        # These inner shots are the only shots used to select ensemble weights.
        selection_field, selection_scalar = predict(model, data, selection_rows, device, args.batch_size)
        selection_field, selection_scalar = calibrated_prediction(
            np.array(data.base[selection_rows]), np.array(data.base_scalar[selection_rows]),
            selection_field, selection_scalar, best_strengths)
        np.savez_compressed(args.output / "selection_predictions.npz",
                            shot_name=data.names[selection_rows], frame_index=data.frames[selection_rows],
                            source_row=data.source[selection_rows], psirz=selection_field,
                            q95=selection_scalar[:, 0], betaN=selection_scalar[:, 1],
                            fold=np.asarray(data.report["fold"]),
                            nested_shot_crossfit=np.asarray(True), frozen_audit_used=np.asarray(False))
        del selection_field, selection_scalar
    # Outer targets enter only the following scoring calls, after selection.
    field, scalar = predict(model, data, outer_rows, device, args.batch_size)
    field, scalar = calibrated_prediction(np.array(data.base[outer_rows]),
                                          np.array(data.base_scalar[outer_rows]),
                                          field, scalar, best_strengths)
    np.savez_compressed(args.output / "outer_predictions.npz", shot_name=data.names[outer_rows],
                        frame_index=data.frames[outer_rows], source_row=data.source[outer_rows],
                        psirz=field, q95=scalar[:, 0], betaN=scalar[:, 1],
                        fold=np.asarray(data.report["fold"]),
                        nested_shot_crossfit=np.asarray(True), frozen_audit_used=np.asarray(False))
    outer_cache = cache_directory / "outer_reference.pkl"
    result = score(data, outer_rows, field, scalar, outer_cache, args.output / "outer_contributions.npz")
    del field, scalar
    base_result = score(data, outer_rows, np.array(data.base[outer_rows]),
                        np.array(data.fixed_scalar[outer_rows] if fixed_scalar_checkpoint is not None
                                 else data.base_scalar[outer_rows]), outer_cache)
    report = {"decoder": args.decoder, "fold": data.report["fold"],
              "selected_epoch": best_epoch, "selection_score": best_score,
              "selection_geometry_gate_passed": bool(best_epoch > 0) if args.loss_version == "scorer_implicit" else None,
              "selection_shot_names": selection_shots.tolist(),
              "correction_strengths": best_strengths,
              "frozen_calibration": str(args.frozen_calibration) if args.frozen_calibration else None,
              "geometry_on_calibrated_field": args.geometry_on_calibrated_field,
              "loss_version": args.loss_version, "geometry_components": args.geometry_components,
              "contour_audit_sha256": contour_audit_sha256,
              "initial_checkpoint_sha256": initial_checkpoint_sha256,
              "scorer_coefficients": scorer_coefficients,
              "scorer_backward_sigma": scorer_backward_sigma,
              "scorer_contour_backward": scorer_contour_backward,
              "scorer_contour_temperature": scorer_contour_temperature,
              "scalar_reference_sha256": scalar_reference_sha256,
              "base_score_uses_fixed_scalars": fixed_scalar_checkpoint is not None,
              "training_arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "outer_score": result, "outer_base_score": base_result,
              "outer_gain": result["S"] - base_result["S"],
              "minimum_score": 0.996, "target_score": 0.999,
              "meets_minimum_on_this_fold": result["S"] >= 0.996,
              "promotion_passed": False, "promotion_reason": "requires complete pooled OOF and full development evaluation",
              "nested_shot_crossfit": True, "development_used": False,
              "frozen_audit_used": False, "smoke_only": data.report["smoke_only"],
              "elapsed_seconds": time.monotonic() - started}
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
