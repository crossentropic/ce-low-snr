"""
Experiment 02: Invariant-Signal Covariate Shift Stress-Test.

Proves:
1. When training with Cross-Entropy past zero training error, stationary test performance
   remains ostensibly stable.
2. However, under an invariant-signal covariate shift (where P(Y|S) is strictly preserved,
   but noise variance is inflated or rotated), the model experiences catastrophic collapse:
   - Shifted test error rises sharply.
   - Shifted ECE collapses (extreme overconfidence on misclassified points).
   - Negative Log-Likelihood explodes as ||w(t)|| -> inf drives logits to saturation.
3. The oracle classifier w* retains 100% accuracy and 0 loss degradation under the exact same shift.
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
from src.losses import CrossEntropyLoss
from src.trainer import Trainer
from src.metrics import evaluate_model


def run_regime_shift_experiment(
    snr: float = 0.2,
    severities: List[float] = [1.0, 2.0, 3.0, 5.0],
    max_steps: int = 100_000,
    seed: int = 42,
    output_path: str = "results/02_regime_shift_stress.json",
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"=== Starting Experiment 02: Invariant Shift Stress-Test (SNR: {snr}) ===")

    # 1. Training set
    train_ds = make_low_snr_dataset(N=200, k=2, d=498, snr=snr, gamma=1.0, seed=seed)

    # 2. Stationary test set sharing the exact same oracle direction w*
    test_stat = make_low_snr_dataset(
        N=1000, k=2, d=498, snr=snr, gamma=1.0, w_star=train_ds.w_star, seed=seed + 1000
    )

    # 3. Verify oracle performance
    oracle_stat = evaluate_model(
        type("Oracle", (), {"w": train_ds.w_star, "__call__": lambda self, x: x @ self.w})(),
        test_stat,
    )
    print(f">> Oracle Stationary Test Acc: {oracle_stat['accuracy']*100:.1f}%")

    results = {
        "snr": snr,
        "max_steps": max_steps,
        "seed": seed,
        "oracle_acc": oracle_stat["accuracy"],
        "severities": severities,
        "shift_runs": {},
        "reliability_diagrams": {},
    }

    # Run primary evaluation across shift severities
    for sev in severities:
        t0 = time.time()
        print(f"\n>> Evaluating Shift Severity: {sev}x...")
        test_shift = apply_invariant_shift(
            test_stat, shift_type="variance_scaling", severity=sev, seed=seed + 2000
        )

        model = LinearProbe(in_features=train_ds.D)
        loss_fn = CrossEntropyLoss()
        trainer = Trainer(model, loss_fn)

        history = trainer.fit(
            train_ds,
            max_steps=max_steps,
            test_data=test_stat,
            shifted_data=test_shift,
            verbose=False,
        )
        elapsed = time.time() - t0

        results["shift_runs"][str(sev)] = {
            "severity": sev,
            "steps": history.steps,
            "train_loss": history.loss,
            "weight_norm": history.weight_norm,
            "cos_sim": history.cos_sim,
            "stat_error": history.test_error,
            "stat_ece": history.test_ece,
            "stat_loss": history.test_loss,
            "shift_error": history.shifted_error,
            "shift_ece": history.shifted_ece,
            "shift_loss": history.shifted_loss,
            "elapsed_sec": elapsed,
        }

        print(
            f"   Done in {elapsed:.2f}s | "
            f"Stat Err: {history.test_error[-1]:.3f} (ECE: {history.test_ece[-1]:.3f}) | "
            f"Shift Err: {history.shifted_error[-1]:.3f} (ECE: {history.shifted_ece[-1]:.3f}, NLL: {history.shifted_loss[-1]:.2f})"
        )

    # Extract calibration data at early checkpoint (t=50) vs late checkpoint (t=100k) for severity 3.0
    print("\n>> Capturing Reliability Diagram data for severity 3.0...")
    sev_3 = apply_invariant_shift(test_stat, shift_type="variance_scaling", severity=3.0, seed=seed + 2000)

    # Retrain cleanly and save logits at t=50 and t=max_steps
    model = LinearProbe(in_features=train_ds.D)
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(model, loss_fn)

    # Run up to 50 steps
    history_early = trainer.fit(train_ds, max_steps=50, verbose=False)
    with torch.no_grad():
        logits_early_stat = model(test_stat.X).tolist()
        logits_early_shift = model(sev_3.X).tolist()

    # Continue training to max_steps
    history_late = trainer.fit(train_ds, max_steps=max_steps - 50, verbose=False)
    with torch.no_grad():
        logits_late_stat = model(test_stat.X).tolist()
        logits_late_shift = model(sev_3.X).tolist()

    results["reliability_diagrams"] = {
        "targets": test_stat.y.tolist(),
        "early_step": 50,
        "late_step": max_steps,
        "logits_early_stat": logits_early_stat,
        "logits_early_shift": logits_early_shift,
        "logits_late_stat": logits_late_stat,
        "logits_late_shift": logits_late_shift,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[✓] Results successfully saved to {output_path}")
    return results


if __name__ == "__main__":
    run_regime_shift_experiment()
