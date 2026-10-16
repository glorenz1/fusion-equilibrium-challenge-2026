import pytest
import torch

from fec1.model import coordinates
from fec1.losses.symmetric import (
    symmetric_geometry_losses,
    geometry_objective,
)


def ellipse():
    r = torch.linspace(1.0, 2.4, 65)
    z = torch.linspace(-1.4, 1.4, 65)
    rr, zz = torch.meshgrid(r, z, indexing="xy")
    n = (((rr - 1.7) / 0.36).square() + ((zz - 0.10) / 0.58).square())[None]
    # FEC1 source flux has its axis at a maximum.
    shape = coordinates(-n)[1]
    return r, z, rr, zz, n, shape


@pytest.mark.parametrize("components", ["boundary", "profile", "both"])
def test_correct_field_has_zero_loss_and_zero_finite_gradient(components):
    r, z, _, _, n, target = ellipse()
    prediction = target.detach().clone().requires_grad_(True)
    parts = symmetric_geometry_losses(prediction, target, n, r, z)
    loss = geometry_objective(parts, components)
    loss.backward()
    assert float(loss) < 1e-10
    assert torch.isfinite(prediction.grad).all()
    assert float(prediction.grad.abs().max()) < 1e-8


@pytest.mark.parametrize("kind", ["shift", "elongation", "boundary", "profile"])
def test_geometric_distortions_have_finite_descent_direction_toward_truth(kind):
    r, z, rr, zz, n, target = ellipse()
    if kind == "shift":
        altered = (((rr - 1.74) / 0.36).square() + ((zz - 0.10) / 0.58).square())[None]
    elif kind == "elongation":
        altered = (((rr - 1.7) / 0.36).square() + ((zz - 0.10) / 0.63).square())[None]
    elif kind == "boundary":
        altered = n + 0.08 * torch.exp(-((n - 1) / 0.3).square()) * ((rr - 1.7) / 0.36)[None]
    else:
        altered = n + 0.15 * n * (1 - n) * torch.exp(-n.square())
    bad = coordinates(-altered)[1]
    alpha = torch.tensor(1.0, requires_grad=True)
    prediction = target + alpha * (bad - target)
    parts = symmetric_geometry_losses(prediction, target, n, r, z)
    loss = geometry_objective(parts, "profile" if kind == "profile" else "boundary")
    loss.backward()
    assert float(loss) > 1e-7
    assert torch.isfinite(alpha.grad)
    assert float(alpha.grad) > 0, "reducing distortion should reduce the objective"


def test_target_branch_does_not_receive_gradients():
    r, z, _, _, n, target = ellipse()
    target.requires_grad_(True)
    pred = (target.detach() + 0.001 * torch.randn_like(target)).requires_grad_(True)
    loss = geometry_objective(symmetric_geometry_losses(pred, target, n, r, z))
    loss.backward()
    assert target.grad is None
    assert torch.isfinite(pred.grad).all()


def test_component_ablation_is_additive():
    parts = {
        name: torch.tensor(value)
        for name, value in zip(
            ("direct_axis", "direct_lcfs", "direct_geometry", "direct_li"), (1.0, 2.0, 3.0, 4.0)
        )
    }
    assert geometry_objective(parts) == geometry_objective(parts, "boundary") + geometry_objective(
        parts, "profile"
    )
    with pytest.raises(ValueError):
        geometry_objective(parts, "invalid")
