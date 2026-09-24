"""
Single reproduction entrypoint for ce-low-snr research repository.
Executes the test suite, all three experimental phases, and generates figures.
"""

import sys
import time
import subprocess
from pathlib import Path


def run_command(desc: str, cmd: list[str]) -> None:
    print(f"\n{'='*70}")
    print(f">> {desc}")
    print(f"{'='*70}")
    t0 = time.time()
    res = subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"[✓] {desc} finished in {elapsed:.1f}s")


def main() -> None:
    t_start = time.time()
    python_bin = sys.executable

    print("\n" + "#"*70)
    print("  Cross-Entropy Implicit Bias in Low-SNR Regimes: Full Reproduction")
    print("#"*70)

    # 1. Run unit invariant tests
    run_command("Step 1/5: Unit Invariant Verification", [python_bin, "-m", "pytest", "tests/"])

    # 2. Phase 1: Stationary geometry experiment
    run_command("Step 2/5: Phase 1 - Stationary Geometry Experiment", [python_bin, "experiments/01_stationary_geometry.py"])

    # 3. Phase 2: Invariant shift stress test
    run_command("Step 3/5: Phase 2 - Invariant Shift Stress-Test", [python_bin, "experiments/02_regime_shift_stress.py"])

    # 4. Phase 3: Loss comparison tournament
    run_command("Step 4/5: Phase 3 - Loss Geometry Tournament & Isotropic Ablation", [python_bin, "experiments/03_loss_comparison.py"])

    # 5. Plot figures
    run_command("Step 5/5: Generating Publication-Grade Figures", [python_bin, "scripts/plot_figures.py"])

    total_time = time.time() - t_start
    print("\n" + "#"*70)
    print(f"[✓] Complete reproduction successful in {total_time:.1f}s ({total_time/60:.2f} min)")
    print("    Figures available at: figures/ (fig1 through fig4)")
    print("    Raw JSON data at:    results/ (01 through 03)")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
