"""
Full-batch Gradient Descent engine with theoretical step size bounds
and logarithmic checkpoint logging.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import torch

from .models import LinearProbe
from .losses import BaseLoss
from .data import SyntheticDataset
from .metrics import evaluate_model, cosine_similarity, noise_energy_fraction, normalized_margin


def compute_spectral_norm_sq(X: torch.Tensor) -> float:
    """Computes sigma_max^2(X) = ||X||_2^2 using SVD or power iteration."""
    # X is (N, D). ||X||_2 is largest singular value.
    s = torch.linalg.svdvals(X)
    return float((s[0] ** 2).item())


def compute_theoretical_lr(
    X: torch.Tensor,
    loss_fn: BaseLoss,
    safety_factor: float = 0.25,
) -> float:
    """
    Computes exact theoretical learning rate based on Soudry et al. (2018) Lemma 1:
        eta = rho * (2 * N) / (beta_loss * sigma_max^2(X))
    
    Guarantees monotonic energy descent and avoids divergence across varying loss Lipschitz constants.
    """
    N = X.shape[0]
    sigma_max_sq = compute_spectral_norm_sq(X)
    beta = loss_fn.lipschitz_beta
    eta = safety_factor * (2.0 * N) / (beta * sigma_max_sq)
    return float(eta)


def generate_logarithmic_schedule(max_steps: int, num_log_points: int = 100) -> List[int]:
    """
    Generates checkpoint steps: dense linear steps {1..50} + geometrically spaced points to max_steps.
    Matches the O(1 / log^2 t) asymptotic convergence rate of cross-entropy.
    """
    early_steps = list(range(1, min(51, max_steps + 1)))
    if max_steps > 50:
        log_points = np.geomspace(50, max_steps, num=num_log_points, endpoint=True)
        log_steps = [int(round(x)) for x in log_points]
        combined = sorted(list(set(early_steps + log_steps)))
    else:
        combined = early_steps
    return combined


@dataclass
class TrainingHistory:
    steps: List[int] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    train_error: List[float] = field(default_factory=list)
    train_margin: List[float] = field(default_factory=list)
    weight_norm: List[float] = field(default_factory=list)
    cos_sim: List[float] = field(default_factory=list)
    noise_energy: List[float] = field(default_factory=list)
    
    # Test metrics (optional)
    test_error: List[float] = field(default_factory=list)
    test_ece: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    
    # Shifted test metrics (optional)
    shifted_error: List[float] = field(default_factory=list)
    shifted_ece: List[float] = field(default_factory=list)
    shifted_loss: List[float] = field(default_factory=list)

    # Diagnostic milestones
    t_zero: Optional[int] = None
    learning_rate: float = 0.0


class Trainer:
    """
    Deterministic full-batch Gradient Descent optimizer.
    """

    def __init__(
        self,
        model: LinearProbe,
        loss_fn: BaseLoss,
        learning_rate: Optional[float] = None,
        safety_factor: float = 0.25,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.safety_factor = safety_factor

    def fit(
        self,
        train_data: SyntheticDataset,
        max_steps: int = 100_000,
        test_data: Optional[SyntheticDataset] = None,
        shifted_data: Optional[SyntheticDataset] = None,
        verbose: bool = False,
    ) -> TrainingHistory:
        """
        Executes full-batch gradient descent for max_steps.
        """
        X = train_data.X
        y = train_data.y

        # Determine step size if not explicitly provided
        if self.learning_rate is None:
            lr = compute_theoretical_lr(X, self.loss_fn, self.safety_factor)
        else:
            lr = self.learning_rate

        log_schedule = set(generate_logarithmic_schedule(max_steps))
        history = TrainingHistory(learning_rate=lr)

        w = self.model.w
        if w.grad is not None:
            w.grad.zero_()

        for step in range(1, max_steps + 1):
            # Forward pass
            logits = self.model(X)
            loss = self.loss_fn(logits, y, self.model)

            # Backward pass
            loss.backward()

            # Gradient step (exact full-batch GD)
            with torch.no_grad():
                w.data.sub_(lr * w.grad.data)
                w.grad.zero_()

            # Milestone check for t_zero (zero training error)
            if history.t_zero is None:
                with torch.no_grad():
                    preds = torch.where(logits >= 0.0, 1.0, -1.0)
                    train_err = float(torch.mean((preds != y).to(torch.float64)).item())
                    if train_err == 0.0:
                        history.t_zero = step

            # Logging hook
            if step in log_schedule:
                with torch.no_grad():
                    train_eval = evaluate_model(self.model, train_data)
                    
                    history.steps.append(step)
                    history.loss.append(train_eval["loss"])
                    history.train_error.append(train_eval["error_rate"])
                    history.train_margin.append(train_eval["normalized_margin"])
                    history.weight_norm.append(train_eval["weight_norm"])
                    history.cos_sim.append(train_eval["cos_sim"])
                    history.noise_energy.append(train_eval["noise_energy"])

                    if test_data is not None:
                        test_eval = evaluate_model(self.model, test_data)
                        history.test_error.append(test_eval["error_rate"])
                        history.test_ece.append(test_eval["ece"])
                        history.test_loss.append(test_eval["loss"])

                    if shifted_data is not None:
                        shift_eval = evaluate_model(self.model, shifted_data)
                        history.shifted_error.append(shift_eval["error_rate"])
                        history.shifted_ece.append(shift_eval["ece"])
                        history.shifted_loss.append(shift_eval["loss"])

                if verbose and (step == 1 or step % 10_000 == 0 or step == max_steps):
                    print(
                        f"Step {step:7d} | Loss: {history.loss[-1]:.6f} | "
                        f"TrainErr: {history.train_error[-1]:.3f} | "
                        f"||w||: {history.weight_norm[-1]:.3f} | "
                        f"cos(w, w*): {history.cos_sim[-1]:.4f}"
                    )

        return history
