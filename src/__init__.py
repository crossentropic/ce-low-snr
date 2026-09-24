"""
ce-low-snr: Investigation of Cross-Entropy's implicit max-margin bias under low-SNR
and invariant covariate shifts.
"""

from .data import SyntheticDataset, make_low_snr_dataset, apply_invariant_shift
from .models import LinearProbe
from .losses import (
    BaseLoss,
    CrossEntropyLoss,
    WeightDecayedCrossEntropyLoss,
    SquareLoss,
    LabelSmoothingLoss,
)
from .metrics import (
    cosine_similarity,
    noise_energy_fraction,
    normalized_margin,
    expected_calibration_error,
    evaluate_model,
)
from .trainer import Trainer, TrainingHistory, compute_theoretical_lr

__all__ = [
    "SyntheticDataset",
    "make_low_snr_dataset",
    "apply_invariant_shift",
    "LinearProbe",
    "BaseLoss",
    "CrossEntropyLoss",
    "WeightDecayedCrossEntropyLoss",
    "SquareLoss",
    "LabelSmoothingLoss",
    "cosine_similarity",
    "noise_energy_fraction",
    "normalized_margin",
    "expected_calibration_error",
    "evaluate_model",
    "Trainer",
    "TrainingHistory",
    "compute_theoretical_lr",
]
