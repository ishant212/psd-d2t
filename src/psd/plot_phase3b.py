"""Publication-Quality Plotting Module for Phase 3B Confirmation Experiment.

Generates:
1. plots/bleu_vs_steps.png
2. plots/rouge_vs_steps.png
3. plots/bertscore_vs_steps.png
4. plots/train_lm_loss_vs_steps.png
5. plots/val_lm_loss_vs_steps.png
6. plots/psd_loss_vs_steps.png
7. plots/gpu_memory_comparison.png
8. plots/final_metric_comparison_step500.png
"""

import os
import sys
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PHASE3B_DIR = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase3b")
PLOTS_DIR = os.path.join(PHASE3B_DIR, "plots")

# Consistent color styling
COLOR_MAP = {
    "baseline": "#7f7f7f",       # Neutral grey
    "position_only": "#1f77b4",  # Blue (geometric)
    "full_psd": "#2ca02c",       # Green (full system)
}

LABEL_MAP = {
    "baseline": "Baseline (LM Only)",
    "position_only": "Position Only (-β·dP)",
    "full_psd": "Full PSD (α·S - β·dP - γ·dG)",
}


def load_all_variant_data():
    """Loads CSV training logs and generation JSON logs for all 3 variants."""
    variants = ["baseline", "position_only", "full_psd"]
    data = {}
    for v in variants:
        v_dir = os.path.join(PHASE3B_DIR, v)
        gen_path = os.path.join(v_dir, "generation_metrics.json")
        train_path = os.path.join(v_dir, "training_log.csv")
        val_path = os.path.join(v_dir, "validation_log.csv")
        metrics_path = os.path.join(v_dir, "metrics.json")

        v_data = {"gen": [], "train": [], "val": [], "metrics": {}}
        if os.path.exists(gen_path):
            with open(gen_path, "r", encoding="utf-8") as f:
                v_data["gen"] = json.load(f)
        if os.path.exists(train_path):
            with open(train_path, "r", encoding="utf-8") as f:
                v_data["train"] = list(csv.DictReader(f))
        if os.path.exists(val_path):
            with open(val_path, "r", encoding="utf-8") as f:
                v_data["val"] = list(csv.DictReader(f))
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                v_data["metrics"] = json.load(f)

        data[v] = v_data
    return data


def generate_all_phase3b_plots():
    """Generates all 8 publication figures."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    all_data = load_all_variant_data()

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    })

    # 1. BLEU vs Optimizer Steps
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for v_key in ["baseline", "position_only", "full_psd"]:
        gen_list = all_data[v_key]["gen"]
        steps = [entry["step"] for entry in gen_list if "step" in entry]
        bleus = [entry["metrics"].get("bleu", entry["metrics"].get("bleu_score", 0.0)) for entry in gen_list if "metrics" in entry]
        if steps and bleus:
            ax.plot(steps, bleus, marker="o", linewidth=2.2, label=LABEL_MAP[v_key], color=COLOR_MAP[v_key])
            for s, b in zip(steps, bleus):
                if s in [150, 300, 500]:
                    ax.annotate(f"{b:.1f}", (s, b), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)

    ax.set_title("Phase 3B: BLEU Trajectory Across 500 Optimizer Steps")
    ax.set_xlabel("Optimizer Steps (Batch=1, GradAccum=4, Seed=42)")
    ax.set_ylabel("BLEU-4 Score")
    ax.set_xticks([1, 50, 100, 150, 200, 300, 400, 500])
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "bleu_vs_steps.png"))
    plt.close(fig)

    # 2. ROUGE-L vs Optimizer Steps
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for v_key in ["baseline", "position_only", "full_psd"]:
        gen_list = all_data[v_key]["gen"]
        steps = [entry["step"] for entry in gen_list if "step" in entry]
        rouges = [entry["metrics"].get("rougeL", entry["metrics"].get("rouge_l", 0.0)) for entry in gen_list if "metrics" in entry]
        if steps and rouges:
            ax.plot(steps, rouges, marker="s", linewidth=2.2, label=LABEL_MAP[v_key], color=COLOR_MAP[v_key])
            for s, r in zip(steps, rouges):
                if s in [150, 300, 500]:
                    ax.annotate(f"{r:.1f}", (s, r), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)

    ax.set_title("Phase 3B: ROUGE-L Trajectory Across 500 Optimizer Steps")
    ax.set_xlabel("Optimizer Steps")
    ax.set_ylabel("ROUGE-L Score")
    ax.set_xticks([1, 50, 100, 150, 200, 300, 400, 500])
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "rouge_vs_steps.png"))
    plt.close(fig)

    # 3. BERTScore vs Optimizer Steps
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for v_key in ["baseline", "position_only", "full_psd"]:
        gen_list = all_data[v_key]["gen"]
        steps = [entry["step"] for entry in gen_list if "step" in entry]
        berts = [entry["metrics"].get("bertscore_f1", 0.0) for entry in gen_list if "metrics" in entry]
        if steps and berts:
            ax.plot(steps, berts, marker="^", linewidth=2.2, label=LABEL_MAP[v_key], color=COLOR_MAP[v_key])
            for s, b in zip(steps, berts):
                if s in [150, 300, 500]:
                    ax.annotate(f"{b:.1f}", (s, b), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)

    ax.set_title("Phase 3B: BERTScore F1 Trajectory Across 500 Optimizer Steps")
    ax.set_xlabel("Optimizer Steps")
    ax.set_ylabel("BERTScore F1 (%)")
    ax.set_xticks([1, 50, 100, 150, 200, 300, 400, 500])
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "bertscore_vs_steps.png"))
    plt.close(fig)

    # 4. Training LM Loss vs Steps
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for v_key in ["baseline", "position_only", "full_psd"]:
        train_list = all_data[v_key]["train"]
        if train_list:
            steps = [int(r["step"]) for r in train_list]
            lm_losses = [float(r["train_lm_loss"]) for r in train_list]
            ax.plot(steps, lm_losses, alpha=0.25, color=COLOR_MAP[v_key])
            # Smoothed curve
            window = 15
            if len(lm_losses) >= window:
                smoothed = np.convolve(lm_losses, np.ones(window)/window, mode="valid")
                ax.plot(steps[window-1:], smoothed, linewidth=2.2, label=LABEL_MAP[v_key], color=COLOR_MAP[v_key])
            else:
                ax.plot(steps, lm_losses, linewidth=2.2, label=LABEL_MAP[v_key], color=COLOR_MAP[v_key])

    ax.set_title("Phase 3B: Training LM Loss Convergence (Moving Avg w=15)")
    ax.set_xlabel("Optimizer Steps")
    ax.set_ylabel("Cross-Entropy Loss (Target Tokens Only)")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "train_lm_loss_vs_steps.png"))
    plt.close(fig)

    # 5. Validation LM Loss vs Steps
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for v_key in ["baseline", "position_only", "full_psd"]:
        val_list = all_data[v_key]["val"]
        if val_list:
            steps = [int(r["step"]) for r in val_list]
            val_losses = [float(r["val_lm_loss"]) for r in val_list]
            window = 15
            if len(val_losses) >= window:
                smoothed = np.convolve(val_losses, np.ones(window)/window, mode="valid")
                ax.plot(steps[window-1:], smoothed, linewidth=2.2, label=LABEL_MAP[v_key], color=COLOR_MAP[v_key])
            else:
                ax.plot(steps, val_losses, linewidth=2.2, label=LABEL_MAP[v_key], color=COLOR_MAP[v_key])

    ax.set_title("Phase 3B: Validation LM Loss Convergence")
    ax.set_xlabel("Optimizer Steps")
    ax.set_ylabel("Validation Loss")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "val_lm_loss_vs_steps.png"))
    plt.close(fig)

    # 6. PSD Loss vs Steps (Position-Only vs Full PSD)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for v_key in ["position_only", "full_psd"]:
        train_list = all_data[v_key]["train"]
        if train_list:
            steps = [int(r["step"]) for r in train_list if r.get("train_psd_loss") not in ["NA", None, ""]]
            psd_losses = [float(r["train_psd_loss"]) for r in train_list if r.get("train_psd_loss") not in ["NA", None, ""]]
            if steps and psd_losses:
                window = 15
                if len(psd_losses) >= window:
                    smoothed = np.convolve(psd_losses, np.ones(window)/window, mode="valid")
                    ax.plot(steps[window-1:], smoothed, linewidth=2.2, label=LABEL_MAP[v_key], color=COLOR_MAP[v_key])
                else:
                    ax.plot(steps, psd_losses, linewidth=2.2, label=LABEL_MAP[v_key], color=COLOR_MAP[v_key])

    ax.set_title("Phase 3B: PSD KL Divergence Loss Progression (λ=0.1)")
    ax.set_xlabel("Optimizer Steps")
    ax.set_ylabel("PSD Loss (KL(A_l || P_target))")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "psd_loss_vs_steps.png"))
    plt.close(fig)

    # 7. GPU Memory Comparison
    fig, ax = plt.subplots(figsize=(7.5, 5))
    variants = ["baseline", "position_only", "full_psd"]
    labels = [LABEL_MAP[v] for v in variants]
    allocated = [all_data[v]["metrics"].get("peak_gpu_allocated_mb", 0.0) for v in variants]
    reserved = [all_data[v]["metrics"].get("peak_gpu_reserved_mb", 0.0) for v in variants]

    x = np.arange(len(variants))
    width = 0.35

    ax.bar(x - width/2, allocated, width, label="Allocated VRAM (MB)", color="#3498db")
    ax.bar(x + width/2, reserved, width, label="Reserved VRAM (MB)", color="#9b59b6")

    for i in x:
        ax.annotate(f"{allocated[i]:.1f} MB", (i - width/2, allocated[i]), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
        ax.annotate(f"{reserved[i]:.1f} MB", (i + width/2, reserved[i]), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)

    ax.set_title("Phase 3B: Peak GPU Memory Utilization (RTX 4060 8GB)")
    ax.set_ylabel("Memory (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylim(0, max(reserved + [3500]) * 1.15)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "gpu_memory_comparison.png"))
    plt.close(fig)

    # 8. Final Metric Comparison at Step 500
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bleu_vals = [all_data[v]["metrics"].get("final_generation_metrics", {}).get("bleu", all_data[v]["metrics"].get("final_generation_metrics", {}).get("bleu_score", 0.0)) for v in variants]
    rouge_vals = [all_data[v]["metrics"].get("final_generation_metrics", {}).get("rougeL", all_data[v]["metrics"].get("final_generation_metrics", {}).get("rouge_l", 0.0)) for v in variants]
    bert_vals = [all_data[v]["metrics"].get("final_generation_metrics", {}).get("bertscore_f1", 0.0) for v in variants]

    x = np.arange(len(variants))
    w = 0.25

    b1 = ax.bar(x - w, bleu_vals, w, label="BLEU-4", color="#e74c3c")
    b2 = ax.bar(x, rouge_vals, w, label="ROUGE-L", color="#3498db")
    b3 = ax.bar(x + w, bert_vals, w, label="BERTScore F1", color="#2ecc71")

    for rects in [b1, b2, b3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width()/2, h), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5)

    ax.set_title("Phase 3B: Final Metric Comparison at 500 Optimizer Steps")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "final_metric_comparison_step500.png"))
    plt.close(fig)


if __name__ == "__main__":
    generate_all_phase3b_plots()
    print("[plot_phase3b] All 8 confirmation plots generated successfully.")
