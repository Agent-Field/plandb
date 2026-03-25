#!/usr/bin/env python3
"""Plot RG Scaling Laws experiment results.

Reads actual results from rg_scaling_results.csv and theoretical predictions
from rg_scaling_theory.csv, generates 8 plots comparing predicted vs actual.
"""

import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    """Read CSV file into list of dicts with numeric conversion."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for k, v in row.items():
                try:
                    converted[k] = float(v)
                except ValueError:
                    converted[k] = v
            rows.append(converted)
    return rows


def compute_r2(actual, predicted):
    """Compute R^2 coefficient of determination."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    results_path = os.path.join(base, 'rg_scaling_results.csv')
    theory_path = os.path.join(base, 'rg_scaling_theory.csv')

    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found. Run --exp-rg-scaling first.")
        sys.exit(1)

    actual = read_csv(results_path)
    theory = read_csv(theory_path) if os.path.exists(theory_path) else None

    layers = [r['layers'] for r in actual]
    std_params = [r['std_params'] for r in actual]
    rg_params = [r['rg_params'] for r in actual]
    savings = [r['param_savings_pct'] for r in actual]
    std_composite = [r['std_composite'] for r in actual]
    rg_composite = [r['rg_composite'] for r in actual]
    alpha_std = [r['alpha_std'] for r in actual]
    ff_w1_mean = [r['ff_w1_alpha_mean'] for r in actual]
    std_loss = [r['std_final_loss'] for r in actual]
    rg_loss = [r['rg_final_loss'] for r in actual]

    # Theoretical curves (dense)
    L_dense = np.linspace(1.5, 10, 100)
    P_std_theory = 12480 + L_dense * 27888
    P_rg_theory = 40128 + L_dense * 440
    savings_theory = (1.0 - P_rg_theory / P_std_theory) * 100.0

    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'std': '#e74c3c', 'rg': '#2ecc71', 'theory': '#3498db'}

    # ── Plot 1: Parameter count vs depth ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(layers, std_params, 'o-', color=colors['std'], linewidth=2,
                markersize=8, label='Standard GPT (actual)')
    ax.semilogy(layers, rg_params, 's-', color=colors['rg'], linewidth=2,
                markersize=8, label='RG-GPT (actual)')
    ax.semilogy(L_dense, P_std_theory, '--', color=colors['std'], alpha=0.5,
                label='Std predicted: 12480+L*27888')
    ax.semilogy(L_dense, P_rg_theory, '--', color=colors['rg'], alpha=0.5,
                label='RG predicted: 40128+L*440')
    ax.set_xlabel('Number of Layers (L)', fontsize=12)
    ax.set_ylabel('Parameter Count (log scale)', fontsize=12)
    ax.set_title('Parameter Scaling: Standard GPT vs RG-GPT', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xticks([int(l) for l in layers])
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'param_scaling.png'), dpi=150)
    plt.close(fig)
    print("  Saved param_scaling.png")

    # ── Plot 2: Parameter savings % vs depth ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, savings, 'o-', color=colors['rg'], linewidth=2,
            markersize=8, label='Actual savings')
    ax.plot(L_dense, savings_theory, '--', color=colors['theory'], alpha=0.7,
            label='Predicted (hyperbolic)')
    ax.axhline(y=98.4, color='gray', linestyle=':', alpha=0.5,
               label='Asymptote: 98.4%')
    ax.set_xlabel('Number of Layers (L)', fontsize=12)
    ax.set_ylabel('Parameter Savings (%)', fontsize=12)
    ax.set_title('Parameter Savings vs Depth', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xticks([int(l) for l in layers])
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'param_savings.png'), dpi=150)
    plt.close(fig)
    print("  Saved param_savings.png")

    # ── Plot 3: Composite score vs depth ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, std_composite, 'o-', color=colors['std'], linewidth=2,
            markersize=8, label='Standard GPT')
    ax.plot(layers, rg_composite, 's-', color=colors['rg'], linewidth=2,
            markersize=8, label='RG-GPT')
    ax.set_xlabel('Number of Layers (L)', fontsize=12)
    ax.set_ylabel('Composite Score', fontsize=12)
    ax.set_title('Task Performance vs Depth', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xticks([int(l) for l in layers])
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'composite_vs_depth.png'), dpi=150)
    plt.close(fig)
    print("  Saved composite_vs_depth.png")

    # ── Plot 4: Parameter efficiency ──
    std_eff = [c / (p / 1000.0) for c, p in zip(std_composite, std_params)]
    rg_eff = [c / (p / 1000.0) for c, p in zip(rg_composite, rg_params)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, std_eff, 'o-', color=colors['std'], linewidth=2,
            markersize=8, label='Standard GPT')
    ax.plot(layers, rg_eff, 's-', color=colors['rg'], linewidth=2,
            markersize=8, label='RG-GPT')
    ax.set_xlabel('Number of Layers (L)', fontsize=12)
    ax.set_ylabel('Composite / K-params', fontsize=12)
    ax.set_title('Parameter Efficiency vs Depth', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xticks([int(l) for l in layers])
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'param_efficiency.png'), dpi=150)
    plt.close(fig)
    print("  Saved param_efficiency.png")

    # ── Plot 5: Alpha divergence vs depth ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, alpha_std, 'o-', color=colors['rg'], linewidth=2,
            markersize=8, label='Actual alpha std')
    ax.axhline(y=0.046, color=colors['theory'], linestyle='--', alpha=0.7,
               label='Predicted: 0.046')
    ax.set_xlabel('Number of Layers (L)', fontsize=12)
    ax.set_ylabel('Std Dev of Alpha Values', fontsize=12)
    ax.set_title('Alpha Divergence vs Depth', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xticks([int(l) for l in layers])
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'alpha_divergence.png'), dpi=150)
    plt.close(fig)
    print("  Saved alpha_divergence.png")

    # ── Plot 6: Generalization gap proxy (train loss - composite) ──
    # Using (1 - composite) as a proxy for generalization gap
    std_gap = [1.0 - c for c in std_composite]
    rg_gap = [1.0 - c for c in rg_composite]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, std_gap, 'o-', color=colors['std'], linewidth=2,
            markersize=8, label='Standard GPT (1 - composite)')
    ax.plot(layers, rg_gap, 's-', color=colors['rg'], linewidth=2,
            markersize=8, label='RG-GPT (1 - composite)')
    ax.set_xlabel('Number of Layers (L)', fontsize=12)
    ax.set_ylabel('Performance Gap (1 - composite)', fontsize=12)
    ax.set_title('Generalization Gap Proxy vs Depth', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xticks([int(l) for l in layers])
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'gen_gap_vs_depth.png'), dpi=150)
    plt.close(fig)
    print("  Saved gen_gap_vs_depth.png")

    # ── Plot 7: ff_w1 alpha mean vs depth ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, ff_w1_mean, 'o-', color=colors['rg'], linewidth=2,
            markersize=8, label='Actual ff_w1 alpha mean')
    ax.axhline(y=1.5, color=colors['theory'], linestyle='--', alpha=0.7,
               label='Predicted fixed point: 1.5')
    ax.set_xlabel('Number of Layers (L)', fontsize=12)
    ax.set_ylabel('ff_w1 Alpha Mean', fontsize=12)
    ax.set_title('ff_w1 Fixed Point Operator vs Depth', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xticks([int(l) for l in layers])
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'ff_w1_fixed_point.png'), dpi=150)
    plt.close(fig)
    print("  Saved ff_w1_fixed_point.png")

    # ── Plot 8: Convergence speed (final loss vs depth) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, std_loss, 'o-', color=colors['std'], linewidth=2,
            markersize=8, label='Standard GPT')
    ax.plot(layers, rg_loss, 's-', color=colors['rg'], linewidth=2,
            markersize=8, label='RG-GPT')
    ax.set_xlabel('Number of Layers (L)', fontsize=12)
    ax.set_ylabel('Final Training Loss', fontsize=12)
    ax.set_title('Training Loss at Final Step vs Depth', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xticks([int(l) for l in layers])
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'convergence_speed.png'), dpi=150)
    plt.close(fig)
    print("  Saved convergence_speed.png")

    # ── Print summary ──
    print("\n  === Scaling Validation Summary ===")
    r2_std_p = compute_r2(std_params, [12480 + l * 27888 for l in layers])
    r2_rg_p = compute_r2(rg_params, [40128 + l * 440 for l in layers])
    r2_sav = compute_r2(savings,
        [(1.0 - (40128 + l * 440) / (12480 + l * 27888)) * 100.0 for l in layers])
    r2_alpha = compute_r2(alpha_std, [0.046] * len(layers))
    r2_ff = compute_r2(ff_w1_mean, [1.5] * len(layers))

    for name, r2 in [('Std param scaling', r2_std_p),
                     ('RG param scaling', r2_rg_p),
                     ('Savings %', r2_sav),
                     ('Alpha std ~ 0.046', r2_alpha),
                     ('ff_w1 ~ 1.5', r2_ff)]:
        status = 'VALIDATED' if r2 > 0.8 else ('PARTIAL' if r2 > 0.5 else 'FALSIFIED')
        print(f"    {name:30s}  R²={r2:.4f}  {status}")

    print(f"\n  All plots saved to {plots_dir}/")


if __name__ == '__main__':
    main()
