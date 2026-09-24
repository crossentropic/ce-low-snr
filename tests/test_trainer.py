import pytest
import torch
from src.data import make_low_snr_dataset
from src.models import LinearProbe
from src.losses import CrossEntropyLoss, SquareLoss
from src.trainer import Trainer, compute_theoretical_lr


def test_theoretical_lr_scaling():
    ds = make_low_snr_dataset(N=50, k=2, d=20, snr=1.0, seed=42)
    ce_loss = CrossEntropyLoss()
    sq_loss = SquareLoss()

    lr_ce = compute_theoretical_lr(ds.X, ce_loss)
    lr_sq = compute_theoretical_lr(ds.X, sq_loss)

    # Square Loss beta=1.0 vs CE beta=0.25 => lr_sq must be exactly 4x smaller
    assert abs(lr_ce / lr_sq - 4.0) < 1e-6


def test_trainer_monotonic_convergence():
    ds = make_low_snr_dataset(N=60, k=2, d=30, snr=1.0, seed=42)

    # Test Cross-Entropy
    model_ce = LinearProbe(in_features=ds.D)
    trainer_ce = Trainer(model_ce, CrossEntropyLoss())
    history_ce = trainer_ce.fit(ds, max_steps=500)

    # Loss must decrease monotonically in early steps
    assert history_ce.loss[-1] < history_ce.loss[0]
    assert history_ce.train_error[-1] <= history_ce.train_error[0]

    # Test Square Loss
    model_sq = LinearProbe(in_features=ds.D)
    trainer_sq = Trainer(model_sq, SquareLoss())
    history_sq = trainer_sq.fit(ds, max_steps=500)

    # Square loss must converge cleanly without NaN
    assert not any(torch.isnan(torch.tensor(history_sq.loss)))
    assert history_sq.loss[-1] < history_sq.loss[0]
