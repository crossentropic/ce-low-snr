"""
Loss functions representing competing optimization geometries.
All losses return a scalar loss and define their scalar Lipschitz smoothness beta.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(ABC, nn.Module):
    """Abstract base class for classification losses with defined gradient Lipschitz constant."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def lipschitz_beta(self) -> float:
        """Maximum second derivative beta of scalar loss l(z; y) w.r.t logit z."""
        pass

    @abstractmethod
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, model: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Computes the empirical mean loss.
        Args:
            logits: (N,) unnormalized predictions z_i = w^T x_i
            targets: (N,) binary labels y_i in {-1.0, +1.0}
            model: Optional model reference (needed for explicit regularizers like weight decay)
        """
        pass


class CrossEntropyLoss(BaseLoss):
    """
    Unregularized binary logistic/cross-entropy loss:
        L(w) = (1/N) * sum_i log(1 + exp(-y_i * z_i))
    
    Asymptotically pushes ||w||_2 -> infinity along the L2 max-margin separator (Soudry et al. 2018).
    Smoothness beta = 0.25 (since d^2/dz^2 softplus(z) = sigma(z)(1-sigma(z)) <= 1/4).
    """

    @property
    def name(self) -> str:
        return "Cross-Entropy"

    @property
    def lipschitz_beta(self) -> float:
        return 0.25

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, model: Optional[nn.Module] = None) -> torch.Tensor:
        # softplus(-y * z) = log(1 + exp(-y * z)) in numerically stable form
        return F.softplus(-targets * logits).mean()


class WeightDecayedCrossEntropyLoss(BaseLoss):
    """
    Cross-Entropy with L2 Weight Decay (isotropic ridge penalty):
        L(w) = L_CE(w) + (lambda / 2) * ||w||_2^2
        
    Controls weight norm, but penalizes signal and noise coordinates isotropically,
    failing to halt directional tilt into ambient noise coordinates.
    """

    def __init__(self, weight_decay: float = 1e-3):
        super().__init__()
        self.weight_decay = weight_decay

    @property
    def name(self) -> str:
        return f"CE + L2 (λ={self.weight_decay:.1e})"

    @property
    def lipschitz_beta(self) -> float:
        return 0.25

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, model: Optional[nn.Module] = None) -> torch.Tensor:
        ce_loss = F.softplus(-targets * logits).mean()
        if model is not None and self.weight_decay > 0.0:
            reg = 0.5 * self.weight_decay * torch.sum(model.w ** 2)
            return ce_loss + reg
        return ce_loss


class SquareLoss(BaseLoss):
    """
    Linear Least Squares (Square Loss) on {-1, +1} labels:
        L(w) = (1 / 2N) * sum_i (z_i - y_i)^2
        
    Converges to the minimum L2-norm interpolator (pseudo-inverse).
    Gradients vanish as residuals vanish, eliminating the asymptotic margin-maximizing drive.
    Smoothness beta = 1.0 (since d^2/dz^2 [0.5 * (z - y)^2] = 1.0).
    """

    @property
    def name(self) -> str:
        return "Square Loss"

    @property
    def lipschitz_beta(self) -> float:
        return 1.0

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, model: Optional[nn.Module] = None) -> torch.Tensor:
        return 0.5 * torch.mean((logits - targets) ** 2)


class LabelSmoothingLoss(BaseLoss):
    """
    Cross-Entropy with Label Smoothing:
        Replaces hard labels y in {-1, +1} with smoothed probability targets:
        q(+1) = (1 - eps/2), q(-1) = eps/2.
        
    Creates a finite optimal logit z* = log((1 - eps/2) / (eps/2)), halting norm growth
    and suppressing overconfident logit saturation.
    """

    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    @property
    def name(self) -> str:
        return f"Label Smoothing (ε={self.epsilon})"

    @property
    def lipschitz_beta(self) -> float:
        return 0.25

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, model: Optional[nn.Module] = None) -> torch.Tensor:
        # Map {-1, +1} -> {0, 1}
        targets_01 = 0.5 * (targets + 1.0)
        # Smoothed target
        q_target = targets_01 * (1.0 - self.epsilon) + 0.5 * self.epsilon
        return F.binary_cross_entropy_with_logits(logits, q_target)
