"""Plotting script for Final ToTTo Long-Context Evaluation.

Generates:
1. context_vs_meteor.png
2. context_vs_hallucination.png
3. context_vs_factual_precision.png
4. truncation_vs_performance.png
"""

import os
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "experiments", "final_totto_long_context")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")


def plot_all():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    bucket_csv = os.path.join(ANALYSIS_DIR, "context_bucket_summary.csv")
    length_csv = os.path.join(ANALYSIS_DIR, "context_length_summary.csv")
    trunc_csv = os.path.join(ANALYSIS_DIR, "truncation_analysis.csv")

    if not os.path.exists(bucket_csv) or not os.path.exists(length_csv):
        print("Data files not found yet. Run evaluation first.")
        return

    with open(bucket_csv, "r", encoding="utf-8") as f:
        bucket_rows = list(csv.DictReader(f))
    with open(length_csv, "r", encoding="utf-8") as f:
        length_rows = list(csv.DictReader(f))

    # Set styles
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    models = ["Qwen3-4B", "Llama-3.2-3B", "Gemma-3-4B"]
    buckets = ["257-512", "513-1024", "1025-2048", ">2048"]
    context_lengths = [256, 512, 1024, 2048]

    # ---------------------------------------------------------
    # Figure 1: METEOR vs Source-Length Bucket across Models (at L=1024 / overall)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("ToTTo Generation Quality (METEOR) vs. Source-Length Bucket", fontweight="bold", y=1.02)

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        b_scores = []
        p_scores = []
        for b in buckets:
            # Average across context lengths for this bucket
            rows_b = [r for r in bucket_rows if r["model"] == model_name and r["source_bucket"] == b]
            if rows_b:
                b_scores.append(np.mean([float(r["baseline_meteor"]) for r in rows_b]))
                p_scores.append(np.mean([float(r["psd_meteor"]) for r in rows_b]))
            else:
                b_scores.append(0)
                p_scores.append(0)

        x = np.arange(len(buckets))
        width = 0.35
        rects1 = ax.bar(x - width/2, b_scores, width, label="Baseline", color="#4A90E2", alpha=0.9, edgecolor="black", linewidth=0.5)
        rects2 = ax.bar(x + width/2, p_scores, width, label="Full PSD", color="#E94E77", alpha=0.9, edgecolor="black", linewidth=0.5)

        ax.set_title(model_name, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=15)
        ax.set_xlabel("Source Length Bucket (tokens)")
        if idx == 0:
            ax.set_ylabel("METEOR Score (%)")
            ax.legend(frameon=True, facecolor="white", framealpha=0.9)

        # Add values on bars
        for r1, r2 in zip(rects1, rects2):
            h1 = r1.get_height()
            h2 = r2.get_height()
            ax.text(r1.get_x() + r1.get_width()/2., h1 + 0.8, f"{h1:.1f}", ha="center", va="bottom", fontsize=8)
            ax.text(r2.get_x() + r2.get_width()/2., h2 + 0.8, f"{h2:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_ylim(0, max(max(b_scores + p_scores) + 8, 50))

    plt.tight_layout()
    fig1_path = os.path.join(PLOTS_DIR, "context_vs_meteor.png")
    fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "context_vs_meteor.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig1_path}")

    # ---------------------------------------------------------
    # Figure 2: Hallucination Rate vs Source-Length Bucket
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("ToTTo Hallucination Rate (%) vs. Source-Length Bucket (Lower is Better)", fontweight="bold", y=1.02)

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        b_scores = []
        p_scores = []
        for b in buckets:
            rows_b = [r for r in bucket_rows if r["model"] == model_name and r["source_bucket"] == b]
            if rows_b:
                b_scores.append(np.mean([float(r["baseline_halluc"]) for r in rows_b]))
                p_scores.append(np.mean([float(r["psd_halluc"]) for r in rows_b]))
            else:
                b_scores.append(0)
                p_scores.append(0)

        x = np.arange(len(buckets))
        width = 0.35
        rects1 = ax.bar(x - width/2, b_scores, width, label="Baseline", color="#E67E22", alpha=0.9, edgecolor="black", linewidth=0.5)
        rects2 = ax.bar(x + width/2, p_scores, width, label="Full PSD", color="#27AE60", alpha=0.9, edgecolor="black", linewidth=0.5)

        ax.set_title(model_name, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=15)
        ax.set_xlabel("Source Length Bucket (tokens)")
        if idx == 0:
            ax.set_ylabel("Hallucination Rate (%)")
            ax.legend(frameon=True, facecolor="white", framealpha=0.9)

        for r1, r2 in zip(rects1, rects2):
            h1 = r1.get_height()
            h2 = r2.get_height()
            ax.text(r1.get_x() + r1.get_width()/2., h1 + 0.8, f"{h1:.1f}", ha="center", va="bottom", fontsize=8)
            ax.text(r2.get_x() + r2.get_width()/2., h2 + 0.8, f"{h2:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_ylim(0, max(max(b_scores + p_scores) + 8, 40))

    plt.tight_layout()
    fig2_path = os.path.join(PLOTS_DIR, "context_vs_hallucination.png")
    fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "context_vs_hallucination.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig2_path}")

    # ---------------------------------------------------------
    # Figure 3: Factual Precision vs Source-Length Bucket
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("ToTTo Factual Precision (%) vs. Source-Length Bucket (Higher is Better)", fontweight="bold", y=1.02)

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        b_scores = []
        p_scores = []
        for b in buckets:
            rows_b = [r for r in bucket_rows if r["model"] == model_name and r["source_bucket"] == b]
            if rows_b:
                b_scores.append(np.mean([float(r["baseline_fact_prec"]) for r in rows_b]))
                p_scores.append(np.mean([float(r["psd_fact_prec"]) for r in rows_b]))
            else:
                b_scores.append(0)
                p_scores.append(0)

        x = np.arange(len(buckets))
        width = 0.35
        rects1 = ax.bar(x - width/2, b_scores, width, label="Baseline", color="#7F8C8D", alpha=0.9, edgecolor="black", linewidth=0.5)
        rects2 = ax.bar(x + width/2, p_scores, width, label="Full PSD", color="#2980B9", alpha=0.9, edgecolor="black", linewidth=0.5)

        ax.set_title(model_name, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=15)
        ax.set_xlabel("Source Length Bucket (tokens)")
        if idx == 0:
            ax.set_ylabel("Factual Precision (%)")
            ax.legend(frameon=True, facecolor="white", framealpha=0.9)

        for r1, r2 in zip(rects1, rects2):
            h1 = r1.get_height()
            h2 = r2.get_height()
            ax.text(r1.get_x() + r1.get_width()/2., h1 + 0.8, f"{h1:.1f}", ha="center", va="bottom", fontsize=8)
            ax.text(r2.get_x() + r2.get_width()/2., h2 + 0.8, f"{h2:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_ylim(0, 105)

    plt.tight_layout()
    fig3_path = os.path.join(PLOTS_DIR, "context_vs_factual_precision.png")
    fig.savefig(fig3_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "context_vs_factual_precision.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig3_path}")

    # ---------------------------------------------------------
    # Figure 4: Effect of Context Budget / Truncation on METEOR
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Effect of Context Budget L on Generation Quality (METEOR)", fontweight="bold", y=1.02)

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        b_scores = []
        p_scores = []
        for clen in context_lengths:
            row_b = next((r for r in length_rows if r["model"] == model_name and r["condition"] == "Baseline" and int(r["context_length"]) == clen), None)
            row_p = next((r for r in length_rows if r["model"] == model_name and r["condition"] == "Full PSD" and int(r["context_length"]) == clen), None)
            b_scores.append(float(row_b["meteor"]) if row_b else 0.0)
            p_scores.append(float(row_p["meteor"]) if row_p else 0.0)

        ax.plot(context_lengths, b_scores, marker="o", linewidth=2.5, color="#4A90E2", label="Baseline")
        ax.plot(context_lengths, p_scores, marker="s", linewidth=2.5, color="#E94E77", label="Full PSD")

        ax.set_title(model_name, fontweight="bold")
        ax.set_xticks(context_lengths)
        ax.set_xlabel("Context Length Budget L (tokens)")
        if idx == 0:
            ax.set_ylabel("Macro METEOR Score (%)")
            ax.legend(frameon=True, facecolor="white", framealpha=0.9)

        for clen, bv, pv in zip(context_lengths, b_scores, p_scores):
            ax.annotate(f"{bv:.1f}", (clen, bv), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8)
            ax.annotate(f"{pv:.1f}", (clen, pv), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig4_path = os.path.join(PLOTS_DIR, "truncation_vs_performance.png")
    fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "truncation_vs_performance.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig4_path}")


if __name__ == "__main__":
    plot_all()
