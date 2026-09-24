"""
Synthetic low-SNR data generation and invariant-signal covariate shift logic.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class SyntheticDataset:
    X: torch.Tensor          # Shape: (N, D), dtype: torch.float64
    y: torch.Tensor          # Shape: (N,), values in {-1.0, +1.0}, dtype: torch.float64
    w_star: torch.Tensor     # Shape: (D,), ground truth oracle [w_s*, 0_d]
    k: int                   # Signal dimension
    d: int                   # Ambient noise dimension
    gamma_true: float        # True geometric margin of signal
    snr: float               # Signal-to-noise ratio

    @property
    def N(self) -> int:
        return self.X.shape[0]

    @property
    def D(self) -> int:
        return self.X.shape[1]

    @property
    def S(self) -> torch.Tensor:
        """Extract the signal submatrix (N, k)."""
        return self.X[:, :self.k]

    @property
    def Xi(self) -> torch.Tensor:
        """Extract the noise submatrix (N, d)."""
        return self.X[:, self.k:]


def generate_signal(
    N: int,
    k: int = 2,
    gamma: float = 1.0,
    signal_noise_std: float = 0.2,
    margin_spread: float = 0.4,
    w_s_star: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates strictly linearly separable signal points S in R^k and labels y in {-1, +1}.
    
    Returns:
        S: (N, k) tensor
        y: (N,) tensor in {-1.0, +1.0}
        w_s_star: (k,) unit oracle vector
    """
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    # 1. Oracle direction w_s^* (normalized unit vector)
    if w_s_star is None:
        w_s_star = torch.randn(k, dtype=torch.float64, generator=generator, device=device)
        w_s_star = w_s_star / torch.norm(w_s_star)
    else:
        w_s_star = (w_s_star[:k] / torch.norm(w_s_star[:k])).to(dtype=torch.float64, device=device)

    # 2. Balanced binary labels
    y = torch.randint(0, 2, (N,), generator=generator, device=device, dtype=torch.float64)
    y = 2.0 * y - 1.0  # map {0, 1} -> {-1.0, +1.0}

    # 3. Continuous margin distribution along w_s^*:
    # m_i = gamma + |delta_i|, where delta_i ~ N(0, margin_spread^2).
    # Guarantees y_i * (s_i @ w_s^*) >= gamma > 0 strictly, while creating a realistic continuous point cloud.
    delta = torch.randn(N, dtype=torch.float64, generator=generator, device=device).abs() * margin_spread
    margins = gamma + delta

    # 4. Orthogonal basis complement for dispersion transverse to w_s^*
    z_raw = torch.randn(N, k, dtype=torch.float64, generator=generator, device=device)
    # Project out component along w_s^*: z_perp = z_raw - (z_raw @ w_s^*) w_s^*
    proj = (z_raw @ w_s_star).unsqueeze(1) * w_s_star.unsqueeze(0)
    z_perp = signal_noise_std * (z_raw - proj)

    # Combine: S_i = y_i * m_i * w_s^* + z_perp
    # Exact check: y_i * (s_i @ w_s^*) = m_i >= gamma > 0 strictly!
    S = (y * margins).unsqueeze(1) * w_s_star.unsqueeze(0) + z_perp

    return S, y, w_s_star


def make_low_snr_dataset(
    N: int = 200,
    k: int = 2,
    d: int = 498,
    snr: float = 0.1,
    gamma: float = 1.0,
    margin_spread: float = 0.4,
    w_star: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    device: torch.device = torch.device("cpu"),
) -> SyntheticDataset:
    """
    Constructs a synthetic dataset X = [S, Xi] in R^{N x (k+d)}.
    
    Data is strictly separable by the true signal alone with margin gamma_true.
    Ambient noise is drawn from N(0, sigma_xi^2 I_d) where sigma_xi = gamma / sqrt(snr).
    """
    w_s_star_in = w_star[:k] if w_star is not None else None
    S, y, w_s_star = generate_signal(
        N=N, k=k, gamma=gamma, margin_spread=margin_spread, w_s_star=w_s_star_in, seed=seed, device=device
    )

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed + 1000)

    # SNR = gamma^2 / sigma_xi^2 => sigma_xi = gamma / sqrt(SNR)
    sigma_xi = gamma / (snr ** 0.5)

    Xi = torch.randn(N, d, dtype=torch.float64, generator=generator, device=device) * sigma_xi

    X = torch.cat([S, Xi], dim=1)

    # Oracle weight vector: w* = [w_s*, 0_d]
    full_w_star = torch.cat([w_s_star, torch.zeros(d, dtype=torch.float64, device=device)])

    return SyntheticDataset(
        X=X,
        y=y,
        w_star=full_w_star,
        k=k,
        d=d,
        gamma_true=gamma,
        snr=snr,
    )


def apply_invariant_shift(
    dataset: SyntheticDataset,
    shift_type: str = "variance_scaling",
    severity: float = 3.0,
    seed: Optional[int] = 999,
) -> SyntheticDataset:
    """
    Applies an invariant-signal covariate shift.
    
    Leaves P(Y | S) strictly unchanged, but alters the ambient noise manifold Xi.
    Because w* = [w_s*, 0_d], the oracle classifier has 0% performance degradation.
    
    Shift types:
        - "variance_scaling": inflates noise standard deviation by severity factor (e.g. 3x)
        - "covariance_rotation": applies random orthogonal rotation to noise dimensions
        - "anisotropic": stretches first half of noise dimensions by severity, compresses rest
    """
    generator = torch.Generator(device=dataset.X.device)
    if seed is not None:
        generator.manual_seed(seed)

    S = dataset.S.clone()
    Xi = dataset.Xi.clone()
    d = dataset.d

    if shift_type == "variance_scaling":
        # Multiplies active noise coordinates by severity
        Xi_shifted = Xi * severity

    elif shift_type == "covariance_rotation":
        # Random orthogonal rotation matrix Q in R^{d x d}
        H = torch.randn(d, d, dtype=torch.float64, generator=generator, device=dataset.X.device)
        Q, _ = torch.linalg.qr(H)
        Xi_shifted = Xi @ Q

    elif shift_type == "anisotropic":
        # Skew noise variances along coordinates
        scales = torch.linspace(severity, 1.0 / severity, d, dtype=torch.float64, device=dataset.X.device)
        Xi_shifted = Xi * scales.unsqueeze(0)

    else:
        raise ValueError(f"Unknown shift type: {shift_type}")

    X_shifted = torch.cat([S, Xi_shifted], dim=1)

    return SyntheticDataset(
        X=X_shifted,
        y=dataset.y.clone(),
        w_star=dataset.w_star.clone(),
        k=dataset.k,
        d=dataset.d,
        gamma_true=dataset.gamma_true,
        snr=dataset.snr,
    )
