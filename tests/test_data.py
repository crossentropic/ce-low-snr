import pytest
import torch
from src.data import make_low_snr_dataset, apply_invariant_shift


def test_dataset_generation_invariants():
    N, k, d = 100, 2, 50
    snr = 0.1
    gamma = 1.0
    ds = make_low_snr_dataset(N=N, k=k, d=d, snr=snr, gamma=gamma, seed=42)

    assert ds.X.shape == (N, k + d)
    assert ds.y.shape == (N,)
    assert ds.w_star.shape == (k + d,)
    assert ds.X.dtype == torch.float64
    assert ds.y.dtype == torch.float64
    assert ds.w_star.dtype == torch.float64

    # Labels must be in {-1.0, +1.0}
    unique_labels = torch.unique(ds.y).tolist()
    assert set(unique_labels) == {-1.0, 1.0}

    # Oracle must have zero weights on noise coordinates
    assert torch.all(ds.w_star[k:] == 0.0)

    # Oracle must have unit norm on signal coordinates
    assert abs(torch.norm(ds.w_star[:k]).item() - 1.0) < 1e-10

    # Separability guarantee on signal: y_i * (s_i @ w_s*) >= gamma
    functional_margins = ds.y * (ds.S @ ds.w_star[:k])
    assert torch.all(functional_margins >= gamma - 1e-10)

    # Oracle must have 100% accuracy on full data X
    preds = torch.where(ds.X @ ds.w_star >= 0.0, 1.0, -1.0)
    assert torch.all(preds == ds.y)


def test_invariant_shift_guarantees():
    N, k, d = 80, 2, 40
    ds = make_low_snr_dataset(N=N, k=k, d=d, snr=0.2, seed=123)

    for shift_type in ["variance_scaling", "covariance_rotation", "anisotropic"]:
        shifted = apply_invariant_shift(ds, shift_type=shift_type, severity=3.0, seed=456)

        # Signal and labels must be 100% identical
        assert torch.allclose(ds.S, shifted.S, atol=1e-12)
        assert torch.all(ds.y == shifted.y)

        # Oracle w* accuracy must remain 100% invariant
        preds_orig = torch.where(ds.X @ ds.w_star >= 0.0, 1.0, -1.0)
        preds_shift = torch.where(shifted.X @ shifted.w_star >= 0.0, 1.0, -1.0)
        assert torch.all(preds_orig == ds.y)
        assert torch.all(preds_shift == ds.y)

        # Noise submatrix must have actually changed
        assert not torch.allclose(ds.Xi, shifted.Xi)


def test_dataset_w_star_sharing():
    # Verify train and test can share the exact same w_star
    train = make_low_snr_dataset(N=50, k=2, d=20, snr=1.0, seed=1)
    test = make_low_snr_dataset(N=100, k=2, d=20, snr=1.0, w_star=train.w_star, seed=2)

    assert torch.allclose(train.w_star, test.w_star)
    # Oracle must have 100% accuracy on both
    train_preds = torch.where(train.X @ train.w_star >= 0, 1.0, -1.0)
    test_preds = torch.where(test.X @ test.w_star >= 0, 1.0, -1.0)
    assert torch.all(train_preds == train.y)
    assert torch.all(test_preds == test.y)

