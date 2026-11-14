"""Publication-Quality Plotting Engine for Phase 3A Ablation Screening.

Generates 7 publication plots:
1. ablation_final_bleu.png
2. ablation_rouge.png
3. ablation_bertscore.png
4. ablation_lm_loss.png
5. ablation_psd_loss.png
6. ablation_component_comparison.png
7. ablation_runtime_vram.png
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

from src.psd.analyze_phase3a import VARIANTS


def load_csv(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return rows


def generate_all_phase3a_plots(
    phase3a_dir: str = "reports/experiments/phase3a",
    plots_dir: str = "reports/experiments/phase3a/plots",
) -> None:
    """Generates all 7 publication plots for Phase 3A."""
    os.makedirs(plots_dir, exist_ok=True)
    summary_json_path = os.path.join(phase3a_dir, "ablation_summary.json")

    if not os.path.exists(summary_json_path):
        from src.psd.analyze_phase3a import compute_ablation_summary
        compute_ablation_summary(phase3a_dir=phase3a_dir)

    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    rows = summary_data.get("summary_rows", [])
    if not rows:
        print("[Plotting] No rows found in ablation_summary.json. Skipping.")
        return

    labels = [r["label"] for r in rows]
    bleu_150 = [float(r["bleu_step150"]) for r in rows]
    rougeL_150 = [float(r["rougeL_step150"]) for r in rows]
    bertscore_150 = [float(r["bertscore_f1_step150"]) for r in rows]
    vram = [float(r["peak_vram_mb"]) for r in rows]
    runtimes = [float(r["runtime_s"]) for r in rows]

    # Colors
    colors = ["#7f8c8d", "#e67e22", "#3498db", "#9b59b6", "#1abc9c", "#e74c3c", "#f39c12", "#27ae60"]

    # 1. Final BLEU Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, bleu_150, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("BLEU-4 Score (150 Optimizer Steps)")
    ax.set_title("Phase 3A Ablation Screening: Final BLEU Score across 8 Variants")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.bar_label(bars, padding=4, fmt="%.2f", fontsize=9)
    fig.tight_layout()
    p1 = os.path.join(plots_dir, "ablation_final_bleu.png")
    plt.savefig(p1)
    plt.close()
    print(f"[Plotting] Saved {p1}")

    # 2. ROUGE-L Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars = ax.barh(y_pos, rougeL_150, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("ROUGE-L F1 Score (%)")
    ax.set_title("Phase 3A Ablation Screening: ROUGE-L across 8 Variants")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.bar_label(bars, padding=4, fmt="%.2f", fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(plots_dir, "ablation_rouge.png")
    plt.savefig(p2)
    plt.close()
    print(f"[Plotting] Saved {p2}")

    # 3. BERTScore Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars = ax.barh(y_pos, bertscore_150, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("BERTScore F1 (%)")
    ax.set_title("Phase 3A Ablation Screening: BERTScore F1 across 8 Variants")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.bar_label(bars, padding=4, fmt="%.2f", fontsize=9)
    fig.tight_layout()
    p3 = os.path.join(plots_dir, "ablation_bertscore.png")
    plt.savefig(p3)
    plt.close()
    print(f"[Plotting] Saved {p3}")

    # 4. Training LM Loss Trajectories
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    for idx, (v_key, v_label, _, _, _) in enumerate(VARIANTS):
        v_csv = os.path.join(phase3a_dir, v_key, "training_log.csv")
        v_rows = load_csv(v_csv)
        if v_rows:
            steps = [int(r["step"]) for r in v_rows]
            losses = [float(r["train_lm_loss"]) for r in v_rows]
            ax.plot(steps, losses, label=v_label, color=colors[idx % len(colors)], alpha=0.85, linewidth=1.5)
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Training LM Loss")
    ax.set_title("Phase 3A: Training LM Loss Trajectories across Variants")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p4 = os.path.join(plots_dir, "ablation_lm_loss.png")
    plt.savefig(p4)
    plt.close()
    print(f"[Plotting] Saved {p4}")

    # 5. PSD Loss Trajectories
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    for idx, (v_key, v_label, _, _, _) in enumerate(VARIANTS):
        if v_key == "baseline":
            continue
        v_csv = os.path.join(phase3a_dir, v_key, "training_log.csv")
        v_rows = load_csv(v_csv)
        if v_rows:
            steps = [int(r["step"]) for r in v_rows if r.get("train_psd_loss") != "NA"]
            psd_losses = [float(r["train_psd_loss"]) for r in v_rows if r.get("train_psd_loss") != "NA"]
            if steps:
                ax.plot(steps, psd_losses, label=v_label, color=colors[idx % len(colors)], alpha=0.85, linewidth=1.5)
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("PSD Loss (KL Divergence)")
    ax.set_title("Phase 3A: PSD Loss Trajectories across Ablation Variants")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p5 = os.path.join(plots_dir, "ablation_psd_loss.png")
    plt.savefig(p5)
    plt.close()
    print(f"[Plotting] Saved {p5}")

    # 6. Component Comparison (Individual vs Pairwise vs Full)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    categories = [
        ("Single Components", ["semantic_only", "position_only", "graph_only"]),
        ("Pairwise Combinations", ["semantic_position", "semantic_graph", "position_graph"]),
        ("Full vs Baseline", ["baseline", "full_psd"]),
    ]
    
    cat_names = []
    cat_bleus = []
    cat_colors = []
    row_dict = {r["variant"]: r for r in rows}

    for cat_title, var_keys in categories:
        for vk in var_keys:
            if vk in row_dict:
                cat_names.append(row_dict[vk]["label"])
                cat_bleus.append(float(row_dict[vk]["bleu_step150"]))
                cat_colors.append("#2ecc71" if vk == "full_psd" else ("#95a5a6" if vk == "baseline" else "#3498db"))

    x = np.arange(len(cat_names))
    bars = ax.bar(x, cat_bleus, color=cat_colors, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("BLEU-4 Score")
    ax.set_title("Phase 3A: Structural Component Attribution (Single vs Pairwise vs Full)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.bar_label(bars, padding=3, fmt="%.2f", fontsize=8)
    fig.tight_layout()
    p6 = os.path.join(plots_dir, "ablation_component_comparison.png")
    plt.savefig(p6)
    plt.close()
    print(f"[Plotting] Saved {p6}")

    # 7. Runtime & VRAM Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    bars1 = ax1.barh(y_pos, runtimes, color="#34495e", height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Runtime (Seconds for 150 Steps)")
    ax1.set_title("Execution Runtime")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.bar_label(bars1, padding=4, fmt="%.1f", fontsize=8)

    bars2 = ax2.barh(y_pos, vram, color="#16a085", height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Peak VRAM Allocated (MB)")
    ax2.set_title("Peak GPU Memory")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    ax2.bar_label(bars2, padding=4, fmt="%.1f", fontsize=8)

    fig.suptitle("Phase 3A: Computational & Memory Footprint Comparison", fontsize=12)
    fig.tight_layout()
    p7 = os.path.join(plots_dir, "ablation_runtime_vram.png")
    plt.savefig(p7)
    plt.close()
    print(f"[Plotting] Saved {p7}")


if __name__ == "__main__":
    generate_all_phase3a_plots()
