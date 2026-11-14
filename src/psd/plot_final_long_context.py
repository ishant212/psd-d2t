"""Final Long-Context Evaluation Publication Plotting Module.

Generates 7 publication-quality figures:
1. context_vs_bleu.png: BLEU-4 scaling curve across L in [256, 512, 1024, 2048].
2. context_vs_meteor.png: METEOR scaling curve across L in [256, 512, 1024, 2048].
3. context_vs_bertscore.png: BERTScore F1 scaling curve across L in [256, 512, 1024, 2048].
4. context_vs_hallucination.png: Hallucination rate scaling curve across L in [256, 512, 1024, 2048].
5. context_vs_factual_precision.png: Factual precision scaling curve across L in [256, 512, 1024, 2048].
6. truncation_vs_performance.png: Truncation impact on performance across models and lengths.
7. context_bucket_comparison.png: Grouped comparison across original source length buckets.
"""

import os
import sys
import json
import csv
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "experiments", "final_long_context")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


def apply_publication_style():
    """Applies clean, high-contrast publication aesthetic."""
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


def load_evaluation_data(output_dir: str = OUTPUT_DIR) -> Dict[str, Any]:
    """Loads evaluation results, length summaries, bucket summaries, and instance results."""
    eval_results_path = os.path.join(output_dir, "evaluation_results.json")
    len_summary_path = os.path.join(output_dir, "context_length_summary.csv")
    bucket_summary_path = os.path.join(output_dir, "context_bucket_summary.csv")
    jsonl_path = os.path.join(output_dir, "per_example_results.jsonl")

    data = {
        "evaluation_results": [],
        "length_summary": [],
        "bucket_summary": [],
        "instance_results": [],
    }

    if os.path.exists(eval_results_path):
        with open(eval_results_path, "r", encoding="utf-8") as f:
            data["evaluation_results"] = json.load(f)

    if os.path.exists(len_summary_path):
        with open(len_summary_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data["length_summary"] = [dict(r) for r in reader]

    if os.path.exists(bucket_summary_path):
        with open(bucket_summary_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data["bucket_summary"] = [dict(r) for r in reader]

    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            data["instance_results"] = [json.loads(line) for line in f if line.strip()]

    return data


def plot_scaling_metric(
    eval_results: List[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    output_filename: str,
    plots_dir: str = PLOTS_DIR,
    invert_direction: bool = False,
):
    """Plots a single scaling metric comparing Baseline vs Full PSD across context lengths."""
    apply_publication_style()
    os.makedirs(plots_dir, exist_ok=True)

    clens = [256, 512, 1024, 2048]
    base_scores = []
    psd_scores = []

    for clen in clens:
        b_res = next((r for r in eval_results if r["condition"] == "Baseline" and r["context_length"] == clen), None)
        p_res = next((r for r in eval_results if r["condition"] == "Full PSD" and r["context_length"] == clen), None)
        base_scores.append(float(b_res[metric_key]) if b_res and metric_key in b_res else 0.0)
        psd_scores.append(float(p_res[metric_key]) if p_res and metric_key in p_res else 0.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    color_base = "#4A5568"
    color_psd = "#2B6CB0"

    ax.plot(clens, base_scores, marker="o", color=color_base, label="Baseline (Standard CE)", linestyle="--", linewidth=2.2)
    ax.plot(clens, psd_scores, marker="s", color=color_psd, label="Full PSD (Joint Loss)", linestyle="-", linewidth=2.4)

    for x, y_b, y_p in zip(clens, base_scores, psd_scores):
        ax.annotate(f"{y_b:.2f}", (x, y_b), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color=color_base)
        ax.annotate(f"{y_p:.2f}", (x, y_p), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=color_psd, weight="bold")

    ax.set_xlabel("Context Budget L (Tokens)")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Context Length Scaling: {metric_label} (Qwen3-4B)")
    ax.set_xticks(clens)
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()

    out_path = os.path.join(plots_dir, output_filename)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_truncation_vs_performance(
    instance_results: List[Dict[str, Any]],
    plots_dir: str = PLOTS_DIR,
):
    """Plots truncation impact on METEOR and Factual Precision across models."""
    apply_publication_style()
    os.makedirs(plots_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))

    color_base = "#4A5568"
    color_psd = "#2B6CB0"

    # Scatter of Truncation % vs METEOR
    b_insts = [it for it in instance_results if it.get("condition") == "Baseline"]
    p_insts = [it for it in instance_results if it.get("condition") == "Full PSD"]

    if b_insts and p_insts:
        # Panel 1: Truncation % vs METEOR
        b_trunc = [it.get("truncation_pct", 0.0) for it in b_insts]
        b_meteor = [it.get("meteor", 0.0) for it in b_insts]
        p_trunc = [it.get("truncation_pct", 0.0) for it in p_insts]
        p_meteor = [it.get("meteor", 0.0) for it in p_insts]

        ax1.scatter(b_trunc, b_meteor, alpha=0.45, color=color_base, label="Baseline", marker="o", s=28)
        ax1.scatter(p_trunc, p_meteor, alpha=0.55, color=color_psd, label="Full PSD", marker="s", s=30)

        # Trendlines
        if len(b_trunc) > 1 and np.std(b_trunc) > 0:
            z_b = np.polyfit(b_trunc, b_meteor, 1)
            p_b = np.poly1d(z_b)
            x_line = np.linspace(0, max(b_trunc), 50)
            ax1.plot(x_line, p_b(x_line), color=color_base, linestyle="--", alpha=0.8)

        if len(p_trunc) > 1 and np.std(p_trunc) > 0:
            z_p = np.polyfit(p_trunc, p_meteor, 1)
            p_p = np.poly1d(z_p)
            x_line = np.linspace(0, max(p_trunc), 50)
            ax1.plot(x_line, p_p(x_line), color=color_psd, linestyle="-", alpha=0.9)

        ax1.set_xlabel("Source Truncation Percentage (%)")
        ax1.set_ylabel("Instance METEOR Score")
        ax1.set_title("Impact of Input Truncation on METEOR")
        ax1.legend(frameon=True)

        # Panel 2: Truncation % vs Factual Precision
        b_fp = [it.get("factual_precision", 0.0) for it in b_insts]
        p_fp = [it.get("factual_precision", 0.0) for it in p_insts]

        ax2.scatter(b_trunc, b_fp, alpha=0.45, color=color_base, label="Baseline", marker="o", s=28)
        ax2.scatter(p_trunc, p_fp, alpha=0.55, color=color_psd, label="Full PSD", marker="s", s=30)

        if len(b_trunc) > 1 and np.std(b_trunc) > 0:
            z_b = np.polyfit(b_trunc, b_fp, 1)
            p_b = np.poly1d(z_b)
            x_line = np.linspace(0, max(b_trunc), 50)
            ax2.plot(x_line, p_b(x_line), color=color_base, linestyle="--", alpha=0.8)

        if len(p_trunc) > 1 and np.std(p_trunc) > 0:
            z_p = np.polyfit(p_trunc, p_fp, 1)
            p_p = np.poly1d(z_p)
            x_line = np.linspace(0, max(p_trunc), 50)
            ax2.plot(x_line, p_p(x_line), color=color_psd, linestyle="-", alpha=0.9)

        ax2.set_xlabel("Source Truncation Percentage (%)")
        ax2.set_ylabel("Factual Precision (%)")
        ax2.set_title("Impact of Input Truncation on Factuality")
        ax2.legend(frameon=True)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, "truncation_vs_performance.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_context_bucket_comparison(
    bucket_summary: List[Dict[str, Any]],
    plots_dir: str = PLOTS_DIR,
):
    """Plots grouped bar comparison across source length buckets at max context L=2048."""
    apply_publication_style()
    os.makedirs(plots_dir, exist_ok=True)

    # Filter for L=2048 (where inputs are least constrained)
    b_rows_2048 = [r for r in bucket_summary if str(r.get("context_length")) == "2048"]
    if not b_rows_2048:
        b_rows_2048 = bucket_summary

    buckets = ["<=256", "257-512", "513-1024", "1025-2048", ">2048"]
    available_buckets = [b for b in buckets if any(r.get("context_bucket") == b for r in b_rows_2048)]

    base_meteor = []
    psd_meteor = []
    base_fp = []
    psd_fp = []

    for b in available_buckets:
        row = next((r for r in b_rows_2048 if r.get("context_bucket") == b), None)
        if row:
            base_meteor.append(float(row.get("baseline_meteor", 0.0)))
            psd_meteor.append(float(row.get("psd_meteor", 0.0)))
            base_fp.append(float(row.get("baseline_factual_prec", 0.0)))
            psd_fp.append(float(row.get("psd_factual_prec", 0.0)))
        else:
            base_meteor.append(0.0)
            psd_meteor.append(0.0)
            base_fp.append(0.0)
            psd_fp.append(0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))
    x = np.arange(len(available_buckets))
    width = 0.35

    color_base = "#4A5568"
    color_psd = "#2B6CB0"

    # Panel 1: METEOR by Bucket
    rects1 = ax1.bar(x - width / 2, base_meteor, width, label="Baseline", color=color_base, alpha=0.85)
    rects2 = ax1.bar(x + width / 2, psd_meteor, width, label="Full PSD", color=color_psd, alpha=0.9)
    ax1.set_xlabel("Source Context Bucket (Tokens)")
    ax1.set_ylabel("METEOR Score")
    ax1.set_title("Performance across Source Length Buckets (L=2048)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(available_buckets, rotation=15)
    ax1.legend(frameon=True)

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}", (r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}", (r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, weight="bold")

    # Panel 2: Factual Precision by Bucket
    rects3 = ax2.bar(x - width / 2, base_fp, width, label="Baseline", color=color_base, alpha=0.85)
    rects4 = ax2.bar(x + width / 2, psd_fp, width, label="Full PSD", color=color_psd, alpha=0.9)
    ax2.set_xlabel("Source Context Bucket (Tokens)")
    ax2.set_ylabel("Factual Precision (%)")
    ax2.set_title("Factual Precision across Source Length Buckets (L=2048)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(available_buckets, rotation=15)
    ax2.legend(frameon=True)

    for r in rects3:
        h = r.get_height()
        ax2.annotate(f"{h:.1f}", (r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for r in rects4:
        h = r.get_height()
        ax2.annotate(f"{h:.1f}", (r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, weight="bold")

    plt.tight_layout()
    out_path = os.path.join(plots_dir, "context_bucket_comparison.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def generate_all_plots(output_dir: str = OUTPUT_DIR):
    """Generates all 7 required publication plots."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    data = load_evaluation_data(output_dir)

    eval_results = data["evaluation_results"]
    bucket_summary = data["bucket_summary"]
    instance_results = data["instance_results"]

    if not eval_results:
        print(f"[Plotting] No evaluation_results found in {output_dir}. Skipping plot generation.")
        return

    print("\n[Plotting] Generating 7 publication figures in:", plots_dir)

    # 1. context_vs_bleu.png
    plot_scaling_metric(eval_results, "bleu_4", "BLEU-4 Score", "context_vs_bleu.png", plots_dir)

    # 2. context_vs_meteor.png
    plot_scaling_metric(eval_results, "meteor", "METEOR Score", "context_vs_meteor.png", plots_dir)

    # 3. context_vs_bertscore.png
    plot_scaling_metric(eval_results, "bertscore_f1", "BERTScore F1", "context_vs_bertscore.png", plots_dir)

    # 4. context_vs_hallucination.png
    plot_scaling_metric(eval_results, "hallucination_rate", "Hallucination Rate (%)", "context_vs_hallucination.png", plots_dir, invert_direction=True)

    # 5. context_vs_factual_precision.png
    plot_scaling_metric(eval_results, "factual_precision", "Factual Precision (%)", "context_vs_factual_precision.png", plots_dir)

    # 6. truncation_vs_performance.png
    if instance_results:
        plot_truncation_vs_performance(instance_results, plots_dir)

    # 7. context_bucket_comparison.png
    if bucket_summary:
        plot_context_bucket_comparison(bucket_summary, plots_dir)

    print("[Plotting] All 7 publication figures generated successfully.\n")


if __name__ == "__main__":
    generate_all_plots()
