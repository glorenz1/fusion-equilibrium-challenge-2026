import numpy as np
import pytest
import torch

from my_experiments.fec1_structured_correction import StructuredCorrection, coordinates, calibrated_prediction, FrozenScalarPredictor
from my_experiments.prepare_fec1_structured_crossfit import partitions, fit_base, predict_base
from my_experiments.train_fec1_structured_correction import objective, load_training_array
from my_experiments.evaluate_fec1_structured_crossfit import promotion


def test_ram_preloading_preserves_values_and_does_not_return_memmap(tmp_path):
    expected = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    path = tmp_path / "training.npy"
    np.save(path, expected)
    resident = load_training_array(path)
    mapped = load_training_array(path, preload=False)
    assert not isinstance(resident, np.memmap)
    assert isinstance(mapped, np.memmap)
    np.testing.assert_array_equal(resident, expected)
    np.testing.assert_array_equal(resident, mapped)


def test_outer_folds_cover_each_shot_once_and_keep_roles_disjoint():
    names = np.repeat([f"shot_{i}" for i in range(30)], [i % 4 + 1 for i in range(30)])
    coverage = np.zeros(len(names), int)
    for fold in range(5):
        role = partitions(names, fold)
        coverage += role == 2
        for shot in np.unique(names):
            assert len(np.unique(role[names == shot])) == 1
        assert set(role) == {0, 1, 2}
    np.testing.assert_array_equal(coverage, 1)


def test_held_labels_cannot_change_base_representation_or_predictions():
    rng = np.random.default_rng(42)
    names = np.repeat([f"shot_{i}" for i in range(12)], 2)
    features = rng.normal(size=(24, 4)).astype(np.float32)
    target = rng.normal(size=(24, 65, 65)).astype(np.float32)
    scalar = rng.normal(size=(24, 2)).astype(np.float32)
    fit = np.arange(18)
    held = np.arange(18, 24)
    kwargs = dict(components=2, estimators=3, workers=1, seed=42)
    first = fit_base(features, target, np.arange(24), scalar, names, fit, **kwargs)
    field_a, scalar_a = predict_base(first, features[held])
    target[held] = rng.normal(size=target[held].shape) * 1000
    scalar[held] = -1e6
    second = fit_base(features, target, np.arange(24), scalar, names, fit, **kwargs)
    field_b, scalar_b = predict_base(second, features[held])
    np.testing.assert_array_equal(field_a, field_b)
    np.testing.assert_array_equal(scalar_a, scalar_b)
    assert not set(first["fit_shots"]) & set(names[held])


@pytest.mark.parametrize("decoder", ["dct", "conv"])
def test_zero_initialization_preserves_field_and_scalars_with_missing_tokens(decoder):
    torch.manual_seed(3)
    model = StructuredCorrection(6, decoder=decoder, order=4, width=32, channels=8)
    base = torch.randn(3, 65, 65) * 0.2 + 0.7
    scalar = torch.rand(3, 2) + 1
    inputs = (torch.randn(3, 6), base, scalar, torch.randn(3, 5, 5),
              torch.randn(3, 5, 2), torch.zeros(3, 5, dtype=torch.bool))
    result = model(*inputs)
    torch.testing.assert_close(result["field"], base, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(result["scalar"], scalar)
    for value in result.values():
        assert torch.isfinite(value).all()


@pytest.mark.parametrize("decoder", ["dct", "conv"])
def test_training_loss_reaches_field_coordinate_and_scalar_heads(decoder):
    torch.manual_seed(5)
    model = StructuredCorrection(6, decoder=decoder, order=4, width=32, channels=8)
    base = torch.randn(2, 65, 65)
    scalar = torch.rand(2, 2) + 1
    inputs = (torch.randn(2, 6), base, scalar, torch.randn(2, 7, 5),
              torch.randn(2, 7, 2), torch.ones(2, 7, dtype=torch.bool))
    result = model(*inputs)
    ramp = torch.linspace(-0.03, 0.03, 65)[None, None, :]
    loss, _ = objective(model, result, base * 1.01 + 0.01 + ramp, scalar + 0.1)
    loss.backward()
    for head in (model.field_head, model.coordinate_head[-1], model.scalar_head[-1]):
        assert head.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_degenerate_field_normalization_has_finite_backward():
    field = torch.zeros(2, 65, 65, requires_grad=True)
    coord, shape = coordinates(field)
    (coord.sum() + shape.square().sum()).backward()
    assert torch.isfinite(field.grad).all()


def test_promotion_enforces_minimum_and_geometry_guards():
    baseline = dict(S=0.992516, r2_psi=0.9993, dlcfs=0.012439, consistency=0.973985)
    candidate = dict(S=0.997, r2_psi=0.9995, dlcfs=0.006, consistency=0.992)
    assert promotion(candidate, baseline, {"S": 0.9965})["passed"]
    assert not promotion(candidate, baseline, {"S": 0.9959})["passed"]
    assert not promotion({**candidate, "S": 0.9959}, baseline, {"S": 0.997})["passed"]
    assert not promotion({**candidate, "r2_psi": 0.999}, baseline, {"S": 0.997})["passed"]


def test_calibration_can_independently_freeze_field_and_scalar_corrections():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(3, 65, 65)).astype(np.float32)
    base_scalar = rng.normal(size=(3, 2)).astype(np.float32)
    field, scalar = calibrated_prediction(base, base_scalar, base + 0.3, base_scalar + 1,
                                          {"shape": 0., "coordinate": [0., 0.], "scalar": [0., 1.]})
    np.testing.assert_allclose(field, base, atol=1e-6)
    np.testing.assert_allclose(scalar[:, 0], base_scalar[:, 0])
    np.testing.assert_allclose(scalar[:, 1], base_scalar[:, 1] + 1)


def test_frozen_scalar_predictor_preserves_full_path_and_calibration():
    model = StructuredCorrection(6, order=4, width=32, channels=8)
    with torch.no_grad():
        model.scalar_head[-1].bias.fill_(.5)
    checkpoint = {"model_config": model.config, "state_dict": model.state_dict(),
                  "correction_strengths": {"scalar": [.6, .8]},
                  "nested_shot_crossfit": True, "development_used": False, "frozen_audit_used": False}
    frozen = FrozenScalarPredictor(checkpoint)
    inputs = (torch.randn(2, 6), torch.randn(2, 65, 65), torch.ones(2, 2),
              torch.randn(2, 7, 5), torch.randn(2, 7, 2), torch.ones(2, 7, dtype=torch.bool))
    expected = inputs[2]+torch.tensor([.6, .8])*(model(*inputs)["scalar"]-inputs[2])
    torch.testing.assert_close(frozen(*inputs), expected)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.)
    torch.testing.assert_close(frozen(*inputs), expected)
    assert all(not p.requires_grad for p in frozen.parameters())


def test_fixed_scalar_objective_does_not_train_scalar_head():
    model = StructuredCorrection(6, order=4, width=32, channels=8)
    inputs = (torch.randn(2, 6), torch.randn(2, 65, 65), torch.ones(2, 2),
              torch.randn(2, 7, 5), torch.randn(2, 7, 2), torch.ones(2, 7, dtype=torch.bool))
    output = model(*inputs)
    loss, _ = objective(model, output, inputs[1]+.01, inputs[2]+1, fixed_scalars=True)
    loss.backward()
    assert all(p.grad is None for p in model.scalar_head.parameters())
