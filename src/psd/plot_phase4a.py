"""Publication-Quality Plotting Module for Phase 4A Long-Context Scaling Evaluation.

Generates 7 publication figures:
1. plots/bleu_vs_context_length.png
2. plots/rougeL_vs_context_length.png
3. plots/bertscore_vs_context_length.png
4. plots/content_coverage_vs_context_length.png
5. plots/baseline_vs_psd_degradation.png
6. plots/structural_retention_vs_context_length.png
7. plots/psd_vs_context_length.png
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
PHASE4A_DIR = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase4a")
PLOTS_DIR = os.path.join(PHASE4A_DIR, "plots")
CONTEXT_LENGTHS = [256, 512, 1024, 2048]

COLOR_MAP = {
    "baseline": "#7f7f7f",  # Neutral grey
    "full_psd": "#2ca02c",  # Green (Full PSD)
}

LABEL_MAP = {
    "baseline": "Baseline (LM Only)",
    "full_psd": "Full PSD (α·S - β·dP - γ·dG)",
}


def load_all_phase4a_data():
    """Loads results and degradation datasets."""
    res_path = os.path.join(PHASE4A_DIR, "context_length_results.json")
    deg_path = os.path.join(PHASE4A_DIR, "degradation_analysis.csv")
    ret_path = os.path.join(PHASE4A_DIR, "structural_retention.csv")

    data = {"results": {}, "degradation": [], "retention": []}
    if os.path.exists(res_path):
        with open(res_path, "r", encoding="utf-8") as f:
            data["results"] = json.load(f)
    if os.path.exists(deg_path):
        with open(deg_path, "r", encoding="utf-8") as f:
            data["degradation"] = list(csv.DictReader(f))
    if os.path.exists(ret_path):
        with open(ret_path, "r", encoding="utf-8") as f:
            data["retention"] = list(csv.DictReader(f))

    return data


def generate_all_phase4a_plots():
    """Generates all 7 publication-ready figures."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    all_data = load_all_phase4a_data()
    res = all_data["results"]

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

    lengths = [256, 512, 1024, 2048]
    x_positions = np.arange(len(lengths))
    str_lengths = [str(l) for l in lengths]

    # 1. BLEU vs Context Length
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for m in ["baseline", "full_psd"]:
        m_dict = res.get(m, {})
        bleus = [m_dict.get(str(L), {}).get("metrics", {}).get("bleu", 0.0) for L in lengths]
        ax.plot(x_positions, bleus, marker="o", linewidth=2.4, label=LABEL_MAP[m], color=COLOR_MAP[m])
        for i, b in enumerate(bleus):
            ax.annotate(f"{b:.2f}", (x_positions[i], b), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9.5)

    ax.set_title("Phase 4A: BLEU-4 Score vs. Serialized Context Length")
    ax.set_xlabel("Context Length (Tokens)")
    ax.set_ylabel("BLEU-4 Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(str_lengths)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "bleu_vs_context_length.png"))
    plt.close(fig)

    # 2. ROUGE-L vs Context Length
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for m in ["baseline", "full_psd"]:
        m_dict = res.get(m, {})
        rouges = [m_dict.get(str(L), {}).get("metrics", {}).get("rougeL", 0.0) for L in lengths]
        ax.plot(x_positions, rouges, marker="s", linewidth=2.4, label=LABEL_MAP[m], color=COLOR_MAP[m])
        for i, r in enumerate(rouges):
            ax.annotate(f"{r:.2f}", (x_positions[i], r), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9.5)

    ax.set_title("Phase 4A: ROUGE-L vs. Serialized Context Length")
    ax.set_xlabel("Context Length (Tokens)")
    ax.set_ylabel("ROUGE-L Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(str_lengths)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "rougeL_vs_context_length.png"))
    plt.close(fig)

    # 3. BERTScore vs Context Length
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for m in ["baseline", "full_psd"]:
        m_dict = res.get(m, {})
        berts = [m_dict.get(str(L), {}).get("metrics", {}).get("bertscore_f1", 0.0) for L in lengths]
        ax.plot(x_positions, berts, marker="^", linewidth=2.4, label=LABEL_MAP[m], color=COLOR_MAP[m])
        for i, b in enumerate(berts):
            ax.annotate(f"{b:.2f}%", (x_positions[i], b), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9.5)

    ax.set_title("Phase 4A: BERTScore F1 vs. Serialized Context Length")
    ax.set_xlabel("Context Length (Tokens)")
    ax.set_ylabel("BERTScore F1 (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(str_lengths)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "bertscore_vs_context_length.png"))
    plt.close(fig)

    # 4. Content Coverage vs Context Length
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for m in ["baseline", "full_psd"]:
        m_dict = res.get(m, {})
        covs = [m_dict.get(str(L), {}).get("metrics", {}).get("structured_coverage", 0.0) for L in lengths]
        ax.plot(x_positions, covs, marker="d", linewidth=2.4, label=LABEL_MAP[m], color=COLOR_MAP[m])
        for i, c in enumerate(covs):
            ax.annotate(f"{c:.1f}%", (x_positions[i], c), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9.5)

    ax.set_title("Phase 4A: Structured Content Coverage vs. Context Length")
    ax.set_xlabel("Context Length (Tokens)")
    ax.set_ylabel("Structured Content Coverage (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(str_lengths)
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "content_coverage_vs_context_length.png"))
    plt.close(fig)

    # 5. Baseline vs PSD Degradation (Metric(256) - Metric(L))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    deg_data = all_data["degradation"]
    if deg_data:
        b_deg = [float(row["baseline_bleu_degradation"]) for row in deg_data]
        p_deg = [float(row["psd_bleu_degradation"]) for row in deg_data]

        w = 0.35
        ax.bar(x_positions - w/2, b_deg, w, label="Baseline BLEU Degradation", color="#e74c3c")
        ax.bar(x_positions + w/2, p_deg, w, label="Full PSD BLEU Degradation", color="#2ecc71")

        for i in range(len(lengths)):
            ax.annotate(f"{b_deg[i]:+.2f}", (x_positions[i] - w/2, b_deg[i]), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
            ax.annotate(f"{p_deg[i]:+.2f}", (x_positions[i] + w/2, p_deg[i]), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)

    ax.set_title("Phase 4A: BLEU Degradation Relative to 256-Token Context")
    ax.set_xlabel("Context Length (Tokens)")
    ax.set_ylabel("Degradation: Metric(256) - Metric(L)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(str_lengths)
    ax.axhline(0, color="grey", linestyle="--", alpha=0.7)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "baseline_vs_psd_degradation.png"))
    plt.close(fig)

    # 6. Structural Retention vs Context Length
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ret_data = all_data["retention"]
    if ret_data:
        # Filter for 'all' dataset
        all_ret = [r for r in ret_data if r.get("dataset") == "all"]
        node_ret = [float(r["mean_node_retention_pct"]) for r in all_ret]
        edge_ret = [float(r["mean_edge_retention_pct"]) for r in all_ret]
        hl_ret = [float(r["mean_highlighted_retention_pct"]) for r in all_ret]

        ax.plot(x_positions, node_ret, marker="o", linewidth=2.2, label="Node Retention (%)", color="#3498db")
        ax.plot(x_positions, edge_ret, marker="s", linewidth=2.2, label="Edge Retention (%)", color="#9b59b6")
        ax.plot(x_positions, hl_ret, marker="^", linewidth=2.2, label="Highlighted Cell Retention (%)", color="#e67e22")

        for i in range(len(lengths)):
            ax.annotate(f"{node_ret[i]:.1f}%", (x_positions[i], node_ret[i]), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8.5)

    ax.set_title("Phase 4A: Structural Element Retention Across Context Windows")
    ax.set_xlabel("Context Length (Tokens)")
    ax.set_ylabel("Retention Percentage (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(str_lengths)
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "structural_retention_vs_context_length.png"))
    plt.close(fig)

    # 7. PSD KL Loss vs Context Length
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    psd_dict = res.get("full_psd", {})
    psd_losses = []
    for L in lengths:
        p_val = psd_dict.get(str(L), {}).get("psd_divergence", {}).get("mean_psd_loss")
        if p_val not in ["NA", None, ""]:
            try:
                psd_losses.append(float(p_val))
            except ValueError:
                psd_losses.append(0.0)
        else:
            psd_losses.append(0.0)

    if any(p > 0 for p in psd_losses):
        ax.plot(x_positions, psd_losses, marker="o", linewidth=2.4, label="Mean PSD KL Loss", color="#2ca02c")
        for i, p in enumerate(psd_losses):
            if p > 0:
                ax.annotate(f"{p:.4f}", (x_positions[i], p), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9.5)

        ax.set_title("Phase 4A: Forward-Pass PSD KL Divergence Across Context Lengths")
        ax.set_xlabel("Context Length (Tokens)")
        ax.set_ylabel("PSD KL Divergence Loss")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(str_lengths)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "psd_vs_context_length.png"))
    plt.close(fig)


if __name__ == "__main__":
    generate_all_phase4a_plots()
    print("[plot_phase4a] All 7 Phase 4A plots generated successfully.")
