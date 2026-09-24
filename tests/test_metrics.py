import pytest
import torch
from src.metrics import (
    cosine_similarity,
    noise_energy_fraction,
    normalized_margin,
    expected_calibration_error,
)


def test_cosine_similarity():
    # Identical vectors
    v1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert abs(cosine_similarity(v1, v1) - 1.0) < 1e-10

    # Orthogonal vectors
    v2 = torch.tensor([-2.0, 1.0, 0.0], dtype=torch.float64)
    assert abs(cosine_similarity(v1, v2)) < 1e-10

    # Zero vector
    v_zero = torch.zeros(3, dtype=torch.float64)
    assert cosine_similarity(v1, v_zero) == 0.0


def test_noise_energy_fraction():
    k = 2
    # w with norm 1 in signal, norm 1 in noise => energy = 1^2 / (1^2 + 1^2) = 0.5
    w = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    assert abs(noise_energy_fraction(w, k=k) - 0.5) < 1e-10


def test_expected_calibration_error():
    # Saturated overconfident predictions on 50/50 split -> ECE should approach 0.50
    targets = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64)
    # Logits: +10, -10, +10, -10 (50% accurate, ~100% confident)
    logits = torch.tensor([10.0, -10.0, 10.0, -10.0], dtype=torch.float64)
    ece = expected_calibration_error(logits, targets, num_bins=10)
    # Predicted accuracy is 0.5, confidence is ~1.0, gap is ~0.5
    assert abs(ece - 0.5) < 1e-3

    # Empty bins guard: make sure it runs without NaN
    logits_extreme = torch.tensor([100.0, 100.0], dtype=torch.float64)
    targets_extreme = torch.tensor([1.0, 1.0], dtype=torch.float64)
    ece_extreme = expected_calibration_error(logits_extreme, targets_extreme, num_bins=15)
    assert not torch.isnan(torch.tensor(ece_extreme))
    assert abs(ece_extreme - 0.0) < 1e-3
