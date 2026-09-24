"""
Homogeneous model architectures strictly obeying scale-sensitivity assumptions.
No normalization layers (BatchNorm / LayerNorm) and no bias terms.
"""

from typing import Optional
import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """
    Homogeneous linear model: f(x) = x @ w.
    
    Zero bias term ensures strict scale-homogeneity f(c * x) = c * f(x)
    and f(w; c * x) = c * f(w; x), exactly matching the theoretical framework of
    Soudry et al. (2018).
    """

    def __init__(
        self,
        in_features: int,
        init_zeros: bool = True,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.in_features = in_features
        self.w = nn.Parameter(torch.empty(in_features, dtype=dtype, device=device))
        self.reset_parameters(init_zeros)

    def reset_parameters(self, init_zeros: bool = True):
        if init_zeros:
            nn.init.zeros_(self.w)
        else:
            nn.init.normal_(self.w, mean=0.0, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes logits: z = x @ w.
        Args:
            x: (N, in_features)
        Returns:
            logits: (N,)
        """
        return x @ self.w

    @property
    def weight(self) -> torch.Tensor:
        return self.w.data

    def get_signal_weights(self, k: int) -> torch.Tensor:
        return self.w[:k]

    def get_noise_weights(self, k: int) -> torch.Tensor:
        return self.w[k:]

    def weight_norm(self) -> float:
        return float(torch.norm(self.w).item())
