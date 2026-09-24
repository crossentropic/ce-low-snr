"""
Experiment 03: Loss Geometry Tournament & The Isotropic Fallacy.

Proves:
1. Under low SNR, Cross-Entropy suffers higher directional tilt and worse calibration
   under covariate shift compared to bounded loss geometries (Square Loss / Label Smoothing).
2. The Isotropic Fallacy: L2 Weight Decay reduces weight norm, but isotropic penalty
   penalizes signal and noise coordinates equally, failing to restore directional alignment
   cos(w, w*).
3. Square Loss (converging to min-norm interpolator without asymptotic margin drive) preserves
   superior directional alignment at low SNR.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import make_low_snr_dataset, apply_invariant_shift
from src.models import LinearProbe
from src.losses import (
    CrossEntropyLoss,
    WeightDecayedCrossEntropyLoss,
    SquareLoss,
    LabelSmoothingLoss,
)
from src.trainer import Trainer
from src.metrics import evaluate_model


def run_loss_comparison_experiment(
    snr_levels: List[float] = [0.02, 0.1, 0.5, 2.0, 10.0],
    seeds: List[int] = [42, 123, 456],
    max_steps: int = 50_000,
    output_path: str = "results/03_loss_comparison.json",
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"=== Starting Experiment 03: Loss Comparison Tournament ===")

    loss_factories = {
        "Cross-Entropy": lambda: CrossEntropyLoss(),
        "CE + Weight Decay (1e-3)": lambda: WeightDecayedCrossEntropyLoss(weight_decay=1e-3),
        "Square Loss": lambda: SquareLoss(),
        "Label Smoothing (0.1)": lambda: LabelSmoothingLoss(epsilon=0.1),
    }

    tournament_results: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        loss_name: {str(snr): [] for snr in snr_levels} for loss_name in loss_factories
    }

    total_runs = len(loss_factories) * len(snr_levels) * len(seeds)
    run_idx = 0
    t_start = time.time()

    for snr in snr_levels:
        for seed in seeds:
            # Training and test datasets
            train_ds = make_low_snr_dataset(N=200, k=2, d=498, snr=snr, gamma=1.0, seed=seed)
            test_stat = make_low_snr_dataset(
                N=1000, k=2, d=498, snr=snr, gamma=1.0, w_star=train_ds.w_star, seed=seed + 1000
            )
            test_shift = apply_invariant_shift(
                test_stat, shift_type="variance_scaling", severity=3.0, seed=seed + 2000
            )

            for loss_name, factory in loss_factories.items():
                run_idx += 1
                loss_fn = factory()
                model = LinearProbe(in_features=train_ds.D)
                trainer = Trainer(model, loss_fn)

                trainer.fit(
                    train_ds,
                    max_steps=max_steps,
                    verbose=False,
                )

                stat_eval = evaluate_model(model, test_stat)
                shift_eval = evaluate_model(model, test_shift)

                entry = {
                    "seed": seed,
                    "cos_sim": stat_eval["cos_sim"],
                    "weight_norm": stat_eval["weight_norm"],
                    "noise_energy": stat_eval["noise_energy"],
                    "stat_error": stat_eval["error_rate"],
                    "stat_ece": stat_eval["ece"],
                    "stat_loss": stat_eval["loss"],
                    "shift_error": shift_eval["error_rate"],
                    "shift_ece": shift_eval["ece"],
                    "shift_loss": shift_eval["loss"],
                }
                tournament_results[loss_name][str(snr)].append(entry)

        print(
            f">> Completed SNR {snr:5.2f} ({run_idx}/{total_runs} runs, "
            f"elapsed: {time.time() - t_start:.1f}s)"
        )

    # 2. Isotropic Fallacy Ablation: Sweep weight decay lambda at fixed low SNR = 0.1
    print("\n=== Running Isotropic Fallacy Ablation (Weight Decay Sweep at SNR = 0.1) ===")
    lambdas = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    wd_ablation_results = []

    ds_wd_train = make_low_snr_dataset(N=200, k=2, d=498, snr=0.1, gamma=1.0, seed=42)
    ds_wd_stat = make_low_snr_dataset(
        N=1000, k=2, d=498, snr=0.1, gamma=1.0, w_star=ds_wd_train.w_star, seed=1042
    )
    ds_wd_shift = apply_invariant_shift(
        ds_wd_stat, shift_type="variance_scaling", severity=3.0, seed=2042
    )

    for lam in lambdas:
        loss_fn = WeightDecayedCrossEntropyLoss(weight_decay=lam)
        model = LinearProbe(in_features=ds_wd_train.D)
        trainer = Trainer(model, loss_fn)
        trainer.fit(ds_wd_train, max_steps=max_steps, verbose=False)

        stat_eval = evaluate_model(model, ds_wd_stat)
        shift_eval = evaluate_model(model, ds_wd_shift)

        res = {
            "lambda": lam,
            "weight_norm": stat_eval["weight_norm"],
            "cos_sim": stat_eval["cos_sim"],
            "noise_energy": stat_eval["noise_energy"],
            "stat_error": stat_eval["error_rate"],
            "shift_error": shift_eval["error_rate"],
            "shift_ece": shift_eval["ece"],
            "shift_loss": shift_eval["loss"],
        }
        wd_ablation_results.append(res)
        print(
            f"   λ = {lam:7.1e} | ||w||: {res['weight_norm']:.4f} | "
            f"cos(w, w*): {res['cos_sim']:.4f} | Shift Err: {res['shift_error']:.3f}"
        )

    full_results = {
        "snr_levels": snr_levels,
        "seeds": seeds,
        "max_steps": max_steps,
        "tournament": tournament_results,
        "isotropic_ablation": wd_ablation_results,
    }

    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\n[✓] Results successfully saved to {output_path}")
    return full_results


if __name__ == "__main__":
    run_loss_comparison_experiment()
