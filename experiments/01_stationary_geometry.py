"""
Experiment 01: Stationary Geometry & Asymptotic Directional Tilt.

Proves:
1. Under Cross-Entropy, ||w(t)||_2 grows logarithmically to infinity.
2. Training loss reaches zero early (t_zero), but the decision boundary continues to rotate.
3. In low SNR, the normalized margin is maximized by tilting into the ambient noise subspace,
   causing cos(w(t), w*) to decay and noise energy fraction ||w_xi||^2 / ||w||^2 to approach 1.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import make_low_snr_dataset
from src.models import LinearProbe
from src.losses import CrossEntropyLoss
from src.trainer import Trainer


def run_stationary_geometry_experiment(
    snr_levels: List[float] = [10.0, 1.0, 0.2, 0.05],
    max_steps: int = 100_000,
    seed: int = 42,
    output_path: str = "results/01_stationary_geometry.json",
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = {
        "snr_levels": snr_levels,
        "max_steps": max_steps,
        "seed": seed,
        "runs": {},
    }

    print(f"=== Starting Experiment 01: Stationary Geometry (Steps: {max_steps}) ===")

    for snr in snr_levels:
        t0 = time.time()
        print(f"\n>> Running SNR = {snr:.2f}...")
        ds = make_low_snr_dataset(N=200, k=2, d=498, snr=snr, gamma=1.0, seed=seed)
        model = LinearProbe(in_features=ds.D)
        loss_fn = CrossEntropyLoss()
        trainer = Trainer(model, loss_fn)

        history = trainer.fit(ds, max_steps=max_steps, verbose=False)
        elapsed = time.time() - t0

        final_w = model.weight
        w_s = model.get_signal_weights(ds.k)
        w_xi = model.get_noise_weights(ds.k)
        w_s_star = ds.w_star[:ds.k]

        # 2D projection coordinates: s_parallel and xi_parallel
        s_parallel = (ds.S @ w_s_star).tolist()
        norm_w_xi = float(torch.norm(w_xi).item())
        if norm_w_xi > 1e-12:
            xi_parallel = ((ds.Xi @ w_xi) / norm_w_xi).tolist()
        else:
            xi_parallel = torch.zeros(ds.N).tolist()

        run_data = {
            "snr": snr,
            "t_zero": history.t_zero,
            "final_weight_norm": history.weight_norm[-1],
            "final_cos_sim": history.cos_sim[-1],
            "final_noise_energy": history.noise_energy[-1],
            "final_train_margin": history.train_margin[-1],
            "history": {
                "steps": history.steps,
                "loss": history.loss,
                "train_error": history.train_error,
                "train_margin": history.train_margin,
                "weight_norm": history.weight_norm,
                "cos_sim": history.cos_sim,
                "noise_energy": history.noise_energy,
            },
            "projection_data": {
                "s_parallel": s_parallel,
                "xi_parallel": xi_parallel,
                "labels": ds.y.tolist(),
                "signal_component": float(torch.dot(w_s, w_s_star).item()),
                "noise_norm": norm_w_xi,
            },
            "elapsed_sec": elapsed,
        }

        print(
            f"   Done in {elapsed:.2f}s | t_zero: {history.t_zero} | "
            f"Final ||w||: {history.weight_norm[-1]:.3f} | "
            f"cos(w, w*): {history.cos_sim[-1]:.4f} | "
            f"Noise Energy: {history.noise_energy[-1]*100:.1f}%"
        )
        results["runs"][str(snr)] = run_data

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[✓] Results successfully saved to {output_path}")
    return results


if __name__ == "__main__":
    run_stationary_geometry_experiment()
