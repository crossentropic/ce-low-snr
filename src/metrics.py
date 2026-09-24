"""
Diagnostic metrics for geometric alignment, margin clearance, and calibration.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F


def cosine_similarity(w: torch.Tensor, w_star: torch.Tensor) -> float:
    """Computes cos(w, w*) = (w^T w*) / (||w|| * ||w*||)."""
    norm_w = torch.norm(w)
    norm_star = torch.norm(w_star)
    if norm_w < 1e-12 or norm_star < 1e-12:
        return 0.0
    cos = torch.dot(w, w_star) / (norm_w * norm_star)
    # Clip to [-1.0, 1.0] for floating point precision
    return float(torch.clamp(cos, -1.0, 1.0).item())


def noise_energy_fraction(w: torch.Tensor, k: int) -> float:
    """Computes ||w_xi||^2 / ||w||^2, measuring the fraction of parameter norm in noise."""
    norm_sq = torch.sum(w ** 2).item()
    if norm_sq < 1e-12:
        return 0.0
    w_xi = w[k:]
    noise_norm_sq = torch.sum(w_xi ** 2).item()
    return float(noise_norm_sq / norm_sq)


def normalized_margin(w: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> float:
    """Computes gamma(w) = min_i [ y_i * (w^T x_i) / ||w||_2 ]."""
    norm_w = torch.norm(w).item()
    if norm_w < 1e-12:
        return 0.0
    functional_margins = y * (X @ w)
    min_margin = torch.min(functional_margins).item()
    return float(min_margin / norm_w)


def expected_calibration_error(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 15,
) -> float:
    """
    Computes Expected Calibration Error (ECE) across confidence bins in [0.5, 1.0].
    
    Guards defensively against division-by-zero on empty bins when logits saturate.
    """
    probs = torch.sigmoid(logits)
    # Confidence in predicted binary class
    confs = torch.maximum(probs, 1.0 - probs)
    preds = torch.where(logits >= 0.0, 1.0, -1.0)
    accuracies = (preds == targets).to(torch.float64)

    bin_boundaries = torch.linspace(0.5, 1.0, num_bins + 1, dtype=torch.float64, device=logits.device)
    ece = 0.0
    total_samples = float(logits.shape[0])

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            # Include upper edge in last bin
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs >= bin_lower) & (confs < bin_upper)

        bin_count = float(torch.sum(in_bin).item())
        if bin_count == 0.0:
            continue

        bin_acc = float(torch.mean(accuracies[in_bin]).item())
        bin_conf = float(torch.mean(confs[in_bin]).item())
        gap = abs(bin_acc - bin_conf)
        ece += (bin_count / total_samples) * gap

    return float(ece)


def evaluate_model(
    model: torch.nn.Module,
    dataset: Any,
) -> Dict[str, float]:
    """
    Evaluates a model against a SyntheticDataset and returns a dictionary of diagnostic metrics.
    """
    with torch.no_grad():
        w = model.w.data
        logits = model(dataset.X)
        targets = dataset.y

        # Binary predictions and error rate
        preds = torch.where(logits >= 0.0, 1.0, -1.0)
        acc = float(torch.mean((preds == targets).to(torch.float64)).item())
        error_rate = 1.0 - acc

        # Cross-Entropy / NLL
        ce_loss = float(F.softplus(-targets * logits).mean().item())

        # Calibration
        ece = expected_calibration_error(logits, targets)

        # Geometric metrics
        w_norm = float(torch.norm(w).item())
        cos_sim = cosine_similarity(w, dataset.w_star)
        noise_energy = noise_energy_fraction(w, dataset.k)
        margin = normalized_margin(w, dataset.X, targets)

    return {
        "loss": ce_loss,
        "accuracy": acc,
        "error_rate": error_rate,
        "ece": ece,
        "weight_norm": w_norm,
        "cos_sim": cos_sim,
        "noise_energy": noise_energy,
        "normalized_margin": margin,
    }
