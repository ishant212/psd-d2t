"""Publication-Quality Plotting Engine for Phase 2B.

Generates 8 standardized plots:
1. baseline_vs_psd_train_lm_loss.png
2. baseline_vs_psd_validation_lm_loss.png
3. psd_loss_by_layer.png
4. baseline_vs_psd_bleu.png
5. baseline_vs_psd_rouge.png
6. baseline_vs_psd_bertscore.png
7. baseline_vs_psd_length_buckets.png
8. gpu_memory_comparison.png
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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_csv_logs(csv_path: str) -> List[Dict[str, Any]]:
    """Loads CSV log rows into dictionary list."""
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return rows


def generate_all_phase2b_plots(
    phase2b_dir: str = "reports/experiments/phase2b",
    plots_dir: str = "reports/experiments/phase2b/plots",
) -> None:
    """Generates all 8 publication-quality plots for Phase 2B."""
    os.makedirs(plots_dir, exist_ok=True)

    b_train_csv = os.path.join(phase2b_dir, "baseline", "training_log.csv")
    p_train_csv = os.path.join(phase2b_dir, "psd", "training_log.csv")
    b_val_csv = os.path.join(phase2b_dir, "baseline", "validation_log.csv")
    p_val_csv = os.path.join(phase2b_dir, "psd", "validation_log.csv")
    b_gen_json = os.path.join(phase2b_dir, "baseline", "generation_metrics.json")
    p_gen_json = os.path.join(phase2b_dir, "psd", "generation_metrics.json")
    length_json = os.path.join(phase2b_dir, "length_bucket_analysis.json")

    b_train_rows = load_csv_logs(b_train_csv)
    p_train_rows = load_csv_logs(p_train_csv)
    b_val_rows = load_csv_logs(b_val_csv)
    p_val_rows = load_csv_logs(p_val_csv)

    # Style configuration
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "lines.linewidth": 1.75,
    })

    # 1. Train LM Loss
    if b_train_rows and p_train_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        b_steps = [int(r["step"]) for r in b_train_rows]
        b_lm = [float(r["train_lm_loss"]) for r in b_train_rows]
        p_steps = [int(r["step"]) for r in p_train_rows]
        p_lm = [float(r["train_lm_loss"]) for r in p_train_rows]

        ax.plot(b_steps, b_lm, label="Baseline (LM Loss Only)", color="#2b5c8f", alpha=0.9)
        ax.plot(p_steps, p_lm, label="PSD (Composite LM + 0.1 * PSD)", color="#27ae60", alpha=0.9)
        ax.set_xlabel("Optimizer Step (Grad Accum = 4)")
        ax.set_ylabel("Training Language Modeling Loss")
        ax.set_title("Phase 2B: Training LM Loss Trajectory (Baseline vs PSD)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        p1 = os.path.join(plots_dir, "baseline_vs_psd_train_lm_loss.png")
        plt.savefig(p1)
        plt.close()
        print(f"[Plotting] Saved {p1}")

    # 2. Validation LM Loss
    if b_val_rows and p_val_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        b_steps = [int(r["step"]) for r in b_val_rows]
        b_vlm = [float(r["val_lm_loss"]) for r in b_val_rows]
        p_steps = [int(r["step"]) for r in p_val_rows]
        p_vlm = [float(r["val_lm_loss"]) for r in p_val_rows]

        ax.plot(b_steps, b_vlm, label="Baseline Validation LM Loss", color="#c0392b", marker="o", markersize=3)
        ax.plot(p_steps, p_vlm, label="PSD Validation LM Loss", color="#16a085", marker="s", markersize=3)
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Validation LM Loss (Target-only)")
        ax.set_title("Phase 2B: Validation LM Loss Progression")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        p2 = os.path.join(plots_dir, "baseline_vs_psd_validation_lm_loss.png")
        plt.savefig(p2)
        plt.close()
        print(f"[Plotting] Saved {p2}")

    # 3. PSD Loss by Layer
    if p_train_rows and "psd_layer_12" in p_train_rows[0]:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        p_steps = [int(r["step"]) for r in p_train_rows]
        l12 = [float(r["psd_layer_12"]) if r["psd_layer_12"] != "NA" else 0.0 for r in p_train_rows]
        l24 = [float(r["psd_layer_24"]) if r["psd_layer_24"] != "NA" else 0.0 for r in p_train_rows]
        l35 = [float(r["psd_layer_35"]) if r["psd_layer_35"] != "NA" else 0.0 for r in p_train_rows]

        ax.plot(p_steps, l12, label="Layer 12 (Internal [11])", color="#e67e22")
        ax.plot(p_steps, l24, label="Layer 24 (Internal [23])", color="#9b59b6")
        ax.plot(p_steps, l35, label="Layer 35 (Internal [34])", color="#2980b9")
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Layer-wise PSD KL Divergence")
        ax.set_title("Phase 2B: Layer-wise PSD Loss Evolution")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        p3 = os.path.join(plots_dir, "psd_loss_by_layer.png")
        plt.savefig(p3)
        plt.close()
        print(f"[Plotting] Saved {p3}")

    # 4, 5, 6. BLEU, ROUGE, BERTScore across evaluation checkpoints
    if os.path.exists(b_gen_json) and os.path.exists(p_gen_json):
        with open(b_gen_json, "r", encoding="utf-8") as f:
            b_gen = json.load(f)
        with open(p_gen_json, "r", encoding="utf-8") as f:
            p_gen = json.load(f)

        eval_steps = [entry["step"] for entry in b_gen]
        b_bleu = [entry["metrics"].get("bleu", 0.0) for entry in b_gen]
        p_bleu = [entry["metrics"].get("bleu", 0.0) for entry in p_gen]
        b_rl = [entry["metrics"].get("rougeL", 0.0) for entry in b_gen]
        p_rl = [entry["metrics"].get("rougeL", 0.0) for entry in p_gen]
        b_bs = [entry["metrics"].get("bertscore_f1", 0.0) for entry in b_gen]
        p_bs = [entry["metrics"].get("bertscore_f1", 0.0) for entry in p_gen]

        # 4. BLEU Plot
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax.plot(eval_steps, b_bleu, label="Baseline BLEU", marker="o", color="#34495e")
        ax.plot(eval_steps, p_bleu, label="PSD BLEU", marker="^", color="#27ae60")
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Corpus BLEU Score")
        ax.set_title("Phase 2B: BLEU Score Progression on Fixed Validation Set")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        p4 = os.path.join(plots_dir, "baseline_vs_psd_bleu.png")
        plt.savefig(p4)
        plt.close()
        print(f"[Plotting] Saved {p4}")

        # 5. ROUGE-L Plot
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax.plot(eval_steps, b_rl, label="Baseline ROUGE-L", marker="o", color="#d35400")
        ax.plot(eval_steps, p_rl, label="PSD ROUGE-L", marker="^", color="#2980b9")
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("ROUGE-L F1 Score")
        ax.set_title("Phase 2B: ROUGE-L Progression on Fixed Validation Set")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        p5 = os.path.join(plots_dir, "baseline_vs_psd_rouge.png")
        plt.savefig(p5)
        plt.close()
        print(f"[Plotting] Saved {p5}")

        # 6. BERTScore Plot
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax.plot(eval_steps, b_bs, label="Baseline BERTScore F1", marker="o", color="#8e44ad")
        ax.plot(eval_steps, p_bs, label="PSD BERTScore F1", marker="^", color="#16a085")
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("BERTScore F1 (%)")
        ax.set_title("Phase 2B: BERTScore F1 Progression on Fixed Validation Set")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        p6 = os.path.join(plots_dir, "baseline_vs_psd_bertscore.png")
        plt.savefig(p6)
        plt.close()
        print(f"[Plotting] Saved {p6}")

    # 7. Length Buckets Plot
    if os.path.exists(length_json):
        with open(length_json, "r", encoding="utf-8") as f:
            l_data = json.load(f)
        buckets = l_data.get("buckets", {})
        b_names = [b for b, v in buckets.items() if v.get("count", 0) > 0]
        if b_names:
            b_sc = [buckets[b]["baseline"].get("structured_coverage", 0.0) for b in b_names]
            p_sc = [buckets[b]["psd"].get("structured_coverage", 0.0) for b in b_names]

            x = np.arange(len(b_names))
            width = 0.35
            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
            r1 = ax.bar(x - width/2, b_sc, width, label="Baseline", color="#7f8c8d")
            r2 = ax.bar(x + width/2, p_sc, width, label="PSD", color="#27ae60")
            ax.set_xlabel("Original Serialized Input Length Bucket (Tokens/Words)")
            ax.set_ylabel("Structured Content Coverage (%)")
            ax.set_title("Phase 2B: Structured Content Coverage across Length Buckets")
            ax.set_xticks(x)
            ax.set_xticklabels(b_names)
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.bar_label(r1, padding=2, fmt="%.1f", fontsize=8)
            ax.bar_label(r2, padding=2, fmt="%.1f", fontsize=8)
            fig.tight_layout()
            p7 = os.path.join(plots_dir, "baseline_vs_psd_length_buckets.png")
            plt.savefig(p7)
            plt.close()
            print(f"[Plotting] Saved {p7}")

    # 8. GPU Memory Comparison
    if b_train_rows and p_train_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        b_steps = [int(r["step"]) for r in b_train_rows]
        b_alloc = [float(r["gpu_allocated_mb"]) for r in b_train_rows]
        b_res = [float(r["gpu_reserved_mb"]) for r in b_train_rows]
        p_alloc = [float(r["gpu_allocated_mb"]) for r in p_train_rows]
        p_res = [float(r["gpu_reserved_mb"]) for r in p_train_rows]

        ax.plot(b_steps, b_alloc, label="Baseline VRAM Allocated", color="#2980b9", linestyle="-")
        ax.plot(p_steps, p_alloc, label="PSD VRAM Allocated", color="#27ae60", linestyle="-")
        ax.plot(b_steps, b_res, label="Baseline VRAM Reserved", color="#2980b9", linestyle=":", alpha=0.6)
        ax.plot(p_steps, p_res, label="PSD VRAM Reserved", color="#27ae60", linestyle=":", alpha=0.6)
        ax.axhline(y=8192, color="r", linestyle="--", alpha=0.7, label="Hardware VRAM Capacity (8 GB)")
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("GPU Memory (MB)")
        ax.set_title("Phase 2B: GPU VRAM Allocated & Reserved Comparison")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="center right")
        fig.tight_layout()
        p8 = os.path.join(plots_dir, "gpu_memory_comparison.png")
        plt.savefig(p8)
        plt.close()
        print(f"[Plotting] Saved {p8}")


if __name__ == "__main__":
    generate_all_phase2b_plots()
