"""Phase 4 Long-Context Publication Plotting Module.

Generates 6 publication-quality figures:
1. phase4_bleu_scaling.png: BLEU-4 scaling curve across L in [256, 512, 1024, 2048].
2. phase4_meteor_scaling.png: METEOR scaling curve.
3. phase4_rouge_l_scaling.png: ROUGE-L scaling curve.
4. phase4_bertscore_scaling.png: BERTScore F1 scaling curve.
5. phase4_vram_and_throughput.png: Peak VRAM and throughput scaling from preflight.
6. phase4_bucket_breakdown.png: Performance breakdown by length buckets.
"""

import os
import sys
import json
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PHASE4_DIR = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase4_long_context")
PLOTS_DIR = os.path.join(PHASE4_DIR, "plots")


def apply_publication_style():
    """Applies clean publication aesthetic."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
    })


def plot_metric_scaling(
    x_lens: List[int],
    base_scores: List[float],
    psd_scores: List[float],
    metric_name: str,
    output_filename: str,
    color_base: str = "#4A5568",
    color_psd: str = "#2B6CB0",
):
    """Plots scaling comparison between Baseline and Full PSD for a given metric."""
    apply_publication_style()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.plot(x_lens, base_scores, marker="o", color=color_base, label="LM Baseline", linestyle="--")
    ax.plot(x_lens, psd_scores, marker="s", color=color_psd, label="Full PSD (Joint Loss)", linestyle="-")

    for x, y_b, y_p in zip(x_lens, base_scores, psd_scores):
        ax.annotate(f"{y_b:.1f}", (x, y_b), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color=color_base)
        ax.annotate(f"{y_p:.1f}", (x, y_p), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=color_psd, weight="bold")

    ax.set_xlabel("Context Length (Tokens)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Cross-Context Scaling: {metric_name} (Qwen3-4B)")
    ax.set_xticks(x_lens)
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, output_filename)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def plot_preflight_vram_and_throughput(preflight_results: List[Dict[str, Any]]):
    """Plots Peak VRAM and Throughput from preflight profiling."""
    apply_publication_style()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if not preflight_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # 1. Peak VRAM
    clens = [r["context_length"] for r in preflight_results if r["mode"] == "baseline"]
    vram_base = [r["peak_reserved_mb"] for r in preflight_results if r["mode"] == "baseline"]
    vram_psd = [r["peak_reserved_mb"] for r in preflight_results if r["mode"] == "psd"]

    if clens and vram_base and vram_psd:
        ax1.plot(clens, vram_base, marker="o", color="#4A5568", label="Baseline", linestyle="--")
        ax1.plot(clens, vram_psd, marker="s", color="#D69E2E", label="Full PSD", linestyle="-")
        ax1.axhline(8192, color="#E53E3E", linestyle=":", label="GPU VRAM Limit (8 GB)")
        ax1.set_xlabel("Context Length (Tokens)")
        ax1.set_ylabel("Peak Reserved VRAM (MB)")
        ax1.set_title("Memory Utilization Scaling")
        ax1.legend(frameon=True)
        ax1.set_xticks(clens)

    # 2. Throughput
    tok_base = [r["tokens_per_sec"] for r in preflight_results if r["mode"] == "baseline"]
    tok_psd = [r["tokens_per_sec"] for r in preflight_results if r["mode"] == "psd"]

    if clens and tok_base and tok_psd:
        ax2.plot(clens, tok_base, marker="o", color="#4A5568", label="Baseline", linestyle="--")
        ax2.plot(clens, tok_psd, marker="s", color="#319795", label="Full PSD", linestyle="-")
        ax2.set_xlabel("Context Length (Tokens)")
        ax2.set_ylabel("Throughput (Tokens / Sec)")
        ax2.set_title("Training Throughput")
        ax2.legend(frameon=True)
        ax2.set_xticks(clens)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "phase4_vram_and_throughput.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def generate_all_plots():
    """Generates all publication plots if summary data exists."""
    preflight_path = os.path.join(PHASE4_DIR, "preflight_results.json")
    if os.path.exists(preflight_path):
        with open(preflight_path, "r", encoding="utf-8") as f:
            pf_data = json.load(f)
        plot_preflight_vram_and_throughput(pf_data)


if __name__ == "__main__":
    generate_all_plots()
