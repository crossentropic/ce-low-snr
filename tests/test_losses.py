import pytest
import torch
from src.losses import (
    CrossEntropyLoss,
    WeightDecayedCrossEntropyLoss,
    SquareLoss,
    LabelSmoothingLoss,
)
from src.models import LinearProbe


def test_finite_difference_gradients():
    """Verify analytical gradients against finite differences for each loss."""
    torch.manual_seed(42)
    N, D = 20, 10
    X = torch.randn(N, D, dtype=torch.float64)
    y = torch.randint(0, 2, (N,), dtype=torch.float64) * 2.0 - 1.0

    losses = [
        CrossEntropyLoss(),
        WeightDecayedCrossEntropyLoss(weight_decay=1e-2),
        SquareLoss(),
        LabelSmoothingLoss(epsilon=0.1),
    ]

    for loss_fn in losses:
        model = LinearProbe(in_features=D, init_zeros=False)
        w = model.w

        # Compute analytical grad
        logits = model(X)
        loss = loss_fn(logits, y, model)
        loss.backward()
        analytical_grad = w.grad.clone()

        # Compute numerical gradient via central differences
        eps = 1e-6
        numerical_grad = torch.zeros_like(w)
        for i in range(D):
            w_orig = w.data[i].item()

            w.data[i] = w_orig + eps
            l_plus = loss_fn(model(X), y, model).item()

            w.data[i] = w_orig - eps
            l_minus = loss_fn(model(X), y, model).item()

            numerical_grad[i] = (l_plus - l_minus) / (2.0 * eps)
            w.data[i] = w_orig

        rel_error = torch.norm(analytical_grad - numerical_grad) / (torch.norm(numerical_grad) + 1e-10)
        assert rel_error < 1e-5, f"Gradient check failed for {loss_fn.name}: rel error {rel_error}"


def test_lipschitz_constants():
    ce = CrossEntropyLoss()
    wd = WeightDecayedCrossEntropyLoss()
    sq = SquareLoss()
    ls = LabelSmoothingLoss()

    assert ce.lipschitz_beta == 0.25
    assert wd.lipschitz_beta == 0.25
    assert sq.lipschitz_beta == 1.0
    assert ls.lipschitz_beta == 0.25
