"""Visualization and Plotting Utilities for Phase 2A Reports.

Generates:
1. reports/phase2a/plots/overall_metrics_comparison.png
2. reports/phase2a/plots/position_controlled_performance.png
3. reports/phase2a/plots/positional_degradation_sensitivity.png
4. reports/phase2a/plots/psd_by_layer.png
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_overall_metrics_comparison(
    metrics_json_path: str = "reports/phase2a/metrics_comparison.json",
    output_png: str = "reports/phase2a/plots/overall_metrics_comparison.png",
) -> None:
    """Plots bar chart comparing Baseline vs PSD overall metrics."""
    if not os.path.exists(metrics_json_path):
        print(f"[Plotting] Warning: {metrics_json_path} not found. Skipping plot.")
        return

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    overall = data.get("overall", {})
    baseline_scores = overall.get("baseline", {})
    psd_scores = overall.get("psd", {})

    metrics = [k for k in ["bleu", "rouge1", "rouge2", "rougeL", "triple_coverage"] if k in baseline_scores]
    if not metrics:
        return

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    rects1 = ax.bar(x - width/2, [baseline_scores[m] for m in metrics], width, label="Baseline (LM Only)", color="#3498db")
    rects2 = ax.bar(x + width/2, [psd_scores[m] for m in metrics], width, label="PSD (LM + PSD)", color="#2ecc71")

    ax.set_ylabel("Score / Percentage")
    ax.set_title("Phase 2A: Overall Metric Comparison (Baseline vs PSD)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper().replace("_", " ") for m in metrics])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Label bars
    ax.bar_label(rects1, padding=3, fmt="%.1f", fontsize=8)
    ax.bar_label(rects2, padding=3, fmt="%.1f", fontsize=8)

    fig.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"[Plotting] Saved {output_png}")


def plot_position_controlled_performance(
    pos_json_path: str = "reports/phase2a/position_controlled_results.json",
    output_png: str = "reports/phase2a/plots/position_controlled_performance.png",
) -> None:
    """Plots generation and coverage performance across Beginning, Middle, End positions."""
    if not os.path.exists(pos_json_path):
        print(f"[Plotting] Warning: {pos_json_path} not found. Skipping plot.")
        return

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    with open(pos_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    positions = data.get("positions", ["beginning", "middle", "end"])
    b_pos = data.get("baseline_by_position", {})
    p_pos = data.get("psd_by_position", {})

    x = np.arange(len(positions))
    width = 0.35

    # Plot ROUGE-L and Coverage
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 1. ROUGE-L by position
    b_rl = [b_pos.get(p, {}).get("rougeL", 0.0) for p in positions]
    p_rl = [p_pos.get(p, {}).get("rougeL", 0.0) for p in positions]

    axes[0].bar(x - width/2, b_rl, width, label="Baseline", color="#e74c3c")
    axes[0].bar(x + width/2, p_rl, width, label="PSD", color="#27ae60")
    axes[0].set_ylabel("ROUGE-L F1")
    axes[0].set_title("ROUGE-L by Structured Position")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([p.capitalize() for p in positions])
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Triple/Cell Coverage by position
    b_cov = [b_pos.get(p, {}).get("triple_coverage", 0.0) for p in positions]
    p_cov = [p_pos.get(p, {}).get("triple_coverage", 0.0) for p in positions]

    axes[1].bar(x - width/2, b_cov, width, label="Baseline", color="#e74c3c")
    axes[1].bar(x + width/2, p_cov, width, label="PSD", color="#27ae60")
    axes[1].set_ylabel("Structured Coverage (%)")
    axes[1].set_title("Structured Coverage by Position")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([p.capitalize() for p in positions])
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"[Plotting] Saved {output_png}")


def plot_positional_degradation_sensitivity(
    pos_json_path: str = "reports/phase2a/position_controlled_results.json",
    output_png: str = "reports/phase2a/plots/positional_degradation_sensitivity.png",
) -> None:
    """Plots line chart demonstrating sensitivity/degradation curves across positions."""
    if not os.path.exists(pos_json_path):
        return

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    with open(pos_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    positions = data.get("positions", ["beginning", "middle", "end"])
    b_pos = data.get("baseline_by_position", {})
    p_pos = data.get("psd_by_position", {})

    b_scores = [b_pos.get(p, {}).get("rougeL", 0.0) for p in positions]
    p_scores = [p_pos.get(p, {}).get("rougeL", 0.0) for p in positions]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(positions, b_scores, marker="o", linewidth=2.5, label="Baseline (LM Only)", color="#e74c3c")
    ax.plot(positions, p_scores, marker="s", linewidth=2.5, label="PSD (LM + PSD)", color="#2ecc71")

    ax.set_xlabel("Target Information Placement in Structured Input")
    ax.set_ylabel("ROUGE-L Score")
    ax.set_title("Positional Degradation Curve (Lost-in-the-Middle Assessment)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"[Plotting] Saved {output_png}")


def plot_psd_by_layer(
    psd_training_json: str = "reports/phase2a/psd_training.json",
    output_png: str = "reports/phase2a/plots/psd_by_layer.png",
) -> None:
    """Plots layer-wise PSD divergence across training steps for layers 12, 24, 35."""
    if not os.path.exists(psd_training_json):
        return

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    with open(psd_training_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    step_logs = data.get("step_logs", [])
    if not step_logs:
        return

    steps = [l["step"] for l in step_logs]
    l12 = [l.get("psd_layer_12", 0.0) for l in step_logs]
    l24 = [l.get("psd_layer_24", 0.0) for l in step_logs]
    l35 = [l.get("psd_layer_35", 0.0) for l in step_logs]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(steps, l12, marker="o", label="Layer 12 (Early/Mid)", color="#f39c12")
    ax.plot(steps, l24, marker="^", label="Layer 24 (Mid/Deep)", color="#3498db")
    ax.plot(steps, l35, marker="s", label="Layer 35 (Deep)", color="#9b59b6")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("PSD Loss (KL Divergence)")
    ax.set_title("Layer-wise PSD Loss Trajectory Across Training Steps")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"[Plotting] Saved {output_png}")


def generate_all_plots() -> None:
    """Generates all 4 Phase 2A figures."""
    plot_overall_metrics_comparison()
    plot_position_controlled_performance()
    plot_positional_degradation_sensitivity()
    plot_psd_by_layer()


if __name__ == "__main__":
    generate_all_plots()
