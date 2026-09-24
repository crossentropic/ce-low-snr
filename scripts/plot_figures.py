"""
Publication-grade figure generation for technical report.
Produces clean vector-compatible figures in figures/.
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Styling for academic publication feel
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")


def plot_fig1_stationary_tilt():
    """Figure 1: Decision boundary tilt and asymptotic dynamics."""
    with open(RESULTS_DIR / "01_stationary_geometry.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel A: 2D Canonical Projection Plane
    ax_a = axes[0]
    run_01 = data["runs"]["0.2"]
    proj = run_01["projection_data"]
    s_par = np.array(proj["s_parallel"])
    xi_par = np.array(proj["xi_parallel"])
    labels = np.array(proj["labels"])

    pos_mask = labels > 0
    neg_mask = labels < 0

    ax_a.scatter(
        s_par[pos_mask], xi_par[pos_mask],
        c="#1f77b4", marker="o", s=35, alpha=0.75, edgecolors="none", label="Class +1"
    )
    ax_a.scatter(
        s_par[neg_mask], xi_par[neg_mask],
        c="#d62728", marker="x", s=40, alpha=0.75, linewidths=1.5, label="Class -1"
    )

    # Ground Truth Oracle Separator: purely vertical line s_parallel = 0
    ax_a.axvline(x=0.0, color="#2ca02c", linestyle="-", linewidth=2.5, label="Oracle Boundary ($w^*$)")

    # Learned Separator: w_s_star^T w_s * s_parallel + ||w_xi|| * xi_parallel = 0
    # xi_parallel = - (w_s_star^T w_s / ||w_xi||) * s_parallel
    slope = - proj["signal_component"] / max(proj["noise_norm"], 1e-6)
    s_grid = np.linspace(-3.0, 3.0, 100)
    xi_boundary = slope * s_grid
    ax_a.plot(s_grid, xi_boundary, color="#9467bd", linestyle="--", linewidth=2.5, label="CE Boundary ($t=10^5$)")

    ax_a.set_xlim(-3.0, 3.0)
    ax_a.set_ylim(-6.0, 6.0)
    ax_a.set_xlabel(r"Signal Projection: $s_{\parallel} = s^\top w_s^*$")
    ax_a.set_ylabel(r"Noise Projection: $\xi_{\parallel} = \xi^\top w_\xi / \|w_\xi\|_2$")
    ax_a.set_title("(a) Decision Boundary Tilt into Noise Subspace")
    ax_a.legend(loc="upper left", framealpha=0.95)
    ax_a.grid(True)

    # Panel B: Asymptotic Dynamics over Logarithmic Time
    ax_b = axes[1]
    colors = {"10.0": "#1f77b4", "1.0": "#ff7f0e", "0.2": "#2ca02c", "0.05": "#d62728"}
    labels_snr = {
        "10.0": "SNR = 10.0 (High)",
        "1.0": "SNR = 1.0 (Unit)",
        "0.2": "SNR = 0.2 (Low)",
        "0.05": "SNR = 0.05 (Extreme Low)",
    }

    for snr_str, run in data["runs"].items():
        steps = run["history"]["steps"]
        cos_sim = run["history"]["cos_sim"]
        c = colors.get(snr_str, "#333333")
        lbl = labels_snr.get(snr_str, f"SNR = {snr_str}")
        ax_b.plot(steps, cos_sim, label=lbl, color=c)

    ax_b.set_xscale("log")
    ax_b.set_ylim(0.0, 1.30)  # Generous headroom for non-overlapping legend
    ax_b.set_xlabel(r"Gradient Descent Steps ($t$, log scale)")
    ax_b.set_ylabel(r"Directional Alignment: $\cos(w(t), w^*)$")
    ax_b.set_title(r"(b) Directional Alignment Dynamics ($\mathcal{O}(1/\log^2 t)$)")
    ax_b.legend(loc="upper left", ncol=2, framealpha=0.95)
    ax_b.grid(True)

    plt.tight_layout()
    output_png = FIGURES_DIR / "fig1_stationary_tilt.png"
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved {output_png}")


def plot_fig2_invariant_shift_collapse():
    """Figure 2: Invariant covariate shift collapse and reliability diagrams."""
    with open(RESULTS_DIR / "02_regime_shift_stress.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel A: Test Error across Optimization Steps for different shift severities
    ax_a = axes[0]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Stationary baseline
    stat_run = data["shift_runs"]["1.0"]
    ax_a.plot(
        stat_run["steps"], stat_run["stat_error"],
        color="#333333", linestyle="-", linewidth=2.5, label="Stationary Test Error"
    )

    for i, (sev_str, run) in enumerate(data["shift_runs"].items()):
        if sev_str == "1.0":
            continue
        ax_a.plot(
            run["steps"], run["shift_error"],
            color=palette[i % len(palette)], linestyle="--",
            label=f"Shifted ({sev_str}× Noise Variance)"
        )

    ax_a.set_xscale("log")
    ax_a.set_ylim(0.25, 0.55)  # Headroom in [0.25, 0.40] for non-overlapping legend
    ax_a.set_xlabel(r"Gradient Descent Steps ($t$, log scale)")
    ax_a.set_ylabel("Classification Error Rate")
    ax_a.set_title("(a) Stationary vs. Shifted Error During Training")
    ax_a.legend(loc="lower left", framealpha=0.95)
    ax_a.grid(True)

    # Panel B: Reliability Diagrams (Confidence vs. Accuracy)
    ax_b = axes[1]
    rel = data["reliability_diagrams"]
    targets = np.array(rel["targets"])
    logits_early = np.array(rel["logits_early_shift"])
    logits_late = np.array(rel["logits_late_shift"])

    def compute_diagram(logits, targets, num_bins=8):
        probs = 1.0 / (1.0 + np.exp(-logits))
        confs = np.maximum(probs, 1.0 - probs)
        preds = np.where(logits >= 0.0, 1.0, -1.0)
        accs = (preds == targets).astype(float)

        bins = np.linspace(0.5, 1.0, num_bins + 1)
        bin_accs, bin_confs = [], []
        for b_low, b_high in zip(bins[:-1], bins[1:]):
            mask = (confs >= b_low) & (confs <= b_high)
            if np.sum(mask) > 0:
                bin_accs.append(np.mean(accs[mask]))
                bin_confs.append(np.mean(confs[mask]))
        return bin_confs, bin_accs

    early_conf, early_acc = compute_diagram(logits_early, targets)
    late_conf, late_acc = compute_diagram(logits_late, targets)

    ax_b.plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.6, label="Perfect Calibration")
    ax_b.plot(early_conf, early_acc, "o-", color="#1f77b4", label="Early Checkpoint ($t=50$)")
    ax_b.plot(late_conf, late_acc, "s-", color="#d62728", label=r"Late Checkpoint ($t=10^5$)")

    ax_b.set_xlim(0.48, 1.02)
    ax_b.set_ylim(0.35, 1.15)  # Headroom in [0.95, 1.15] for non-overlapping legend
    ax_b.set_xlabel(r"Confidence $\max(p, 1-p)$")
    ax_b.set_ylabel("Empirical Accuracy")
    ax_b.set_title("(b) Reliability Diagram Under Noise Covariate Shift")
    ax_b.legend(loc="upper left", framealpha=0.95)
    ax_b.grid(True)

    plt.tight_layout()
    output_png = FIGURES_DIR / "fig2_invariant_shift_collapse.png"
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved {output_png}")


def plot_fig3_loss_tournament():
    """Figure 3: Multi-loss geometry tournament across SNR."""
    with open(RESULTS_DIR / "03_loss_comparison.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    snr_levels = data["snr_levels"]
    loss_styles = {
        "Cross-Entropy": {"color": "#d62728", "marker": "o", "label": "Cross-Entropy (Unreg)"},
        "CE + Weight Decay (1e-3)": {"color": "#ff7f0e", "marker": "^", "label": "CE + L2 WD ($10^{-3}$)"},
        "Square Loss": {"color": "#2ca02c", "marker": "s", "label": "Square Loss (Min-Norm)"},
        "Label Smoothing (0.1)": {"color": "#1f77b4", "marker": "D", "label": r"Label Smoothing ($\epsilon=0.1$)"},
    }

    # Panel 1: Cosine Similarity vs SNR
    ax_1 = axes[0]
    for loss_name, style in loss_styles.items():
        means, stds = [], []
        for snr in snr_levels:
            runs = data["tournament"][loss_name][str(snr)]
            vals = [r["cos_sim"] for r in runs]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax_1.errorbar(snr_levels, means, yerr=stds, fmt=style["marker"]+"-", color=style["color"],
                      label=style["label"], capsize=3)

    ax_1.set_xscale("log")
    ax_1.set_xlabel("Signal-to-Noise Ratio (SNR, log scale)")
    ax_1.set_ylabel(r"Directional Alignment: $\cos(w, w^*)$")
    ax_1.set_title("(a) True Signal Directional Alignment")
    ax_1.grid(True)

    # Panel 2: Shifted Error vs SNR
    ax_2 = axes[1]
    for loss_name, style in loss_styles.items():
        means, stds = [], []
        for snr in snr_levels:
            runs = data["tournament"][loss_name][str(snr)]
            vals = [r["shift_error"] for r in runs]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax_2.errorbar(snr_levels, means, yerr=stds, fmt=style["marker"]+"-", color=style["color"],
                      label=style["label"], capsize=3)

    ax_2.set_xscale("log")
    ax_2.set_xlabel("Signal-to-Noise Ratio (SNR, log scale)")
    ax_2.set_ylabel("Shifted Test Error Rate")
    ax_2.set_title("(b) Robustness Under Invariant Shift")
    ax_2.grid(True)

    # Panel 3: Shifted ECE vs SNR
    ax_3 = axes[2]
    for loss_name, style in loss_styles.items():
        means, stds = [], []
        for snr in snr_levels:
            runs = data["tournament"][loss_name][str(snr)]
            vals = [r["shift_ece"] for r in runs]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax_3.errorbar(snr_levels, means, yerr=stds, fmt=style["marker"]+"-", color=style["color"],
                      label=style["label"], capsize=3)

    ax_3.set_xscale("log")
    ax_3.set_xlabel("Signal-to-Noise Ratio (SNR, log scale)")
    ax_3.set_ylabel("Expected Calibration Error (ECE)")
    ax_3.set_title("(c) Expected Calibration Error Under Shift")
    ax_3.grid(True)

    # Place a single shared, horizontal legend above all 3 subplots
    handles, labels = ax_1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, framealpha=0.95)

    plt.tight_layout()
    output_png = FIGURES_DIR / "fig3_loss_tournament.png"
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved {output_png}")


def plot_fig4_isotropic_fallacy():
    """Figure 4: The Isotropic Fallacy (Weight Decay ablation)."""
    with open(RESULTS_DIR / "03_loss_comparison.json") as f:
        data = json.load(f)

    ablation = data["isotropic_ablation"]
    lambdas = [row["lambda"] for row in ablation]
    norms = [row["weight_norm"] for row in ablation]
    cos_sims = [row["cos_sim"] for row in ablation]

    fig, ax1 = plt.subplots(figsize=(8, 4.8))

    # Weight norm on primary y-axis
    color_norm = "#1f77b4"
    ax1.set_xlabel(r"Weight Decay Regularization $\lambda$ (log scale)")
    ax1.set_ylabel(r"Weight Norm $\|w\|_2$", color=color_norm)
    # Replace lambda=0 with 1e-6 for log plotting
    lambdas_plot = [max(l, 1e-6) for l in lambdas]
    line1 = ax1.plot(lambdas_plot, norms, "o-", color=color_norm, linewidth=2.5, label=r"Weight Norm $\|w\|_2$")
    ax1.tick_params(axis="y", labelcolor=color_norm)
    ax1.set_xscale("log")
    ax1.grid(True)

    # Cosine similarity on secondary y-axis
    ax2 = ax1.twinx()
    color_cos = "#d62728"
    ax2.set_ylabel(r"Directional Alignment $\cos(w, w^*)$", color=color_cos)
    line2 = ax2.plot(lambdas_plot, cos_sims, "s--", color=color_cos, linewidth=2.5, label=r"Alignment $\cos(w, w^*)$")
    ax2.tick_params(axis="y", labelcolor=color_cos)
    ax2.set_ylim(0.0, 0.5)

    # Combined legend placed in the open top-right area
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", framealpha=0.95)

    plt.title("The Isotropic Fallacy: Norm Shrinkage Without Angular Recovery (SNR = 0.1)")
    plt.tight_layout()
    output_png = FIGURES_DIR / "fig4_isotropic_fallacy.png"
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved {output_png}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("=== Generating Figures ===")
    plot_fig1_stationary_tilt()
    plot_fig2_invariant_shift_collapse()
    plot_fig3_loss_tournament()
    plot_fig4_isotropic_fallacy()
    print("\n[✓] All figures generated successfully in figures/")


if __name__ == "__main__":
    main()
