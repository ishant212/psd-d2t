import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "reports", "experiments", "plots")
ARTIFACT_DIR = r"C:\Users\laksh\.gemini\antigravity\brain\d0d10d4b-0a53-48af-9173-30adee9add8e"
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
})

def plot_ablation_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Subplot 1: Lambda Loss Weight Sensitivity ---
    # Lambdas: 0.02, 0.05, 0.10 (default), 0.20
    lambdas = [0.02, 0.05, 0.10, 0.20]
    
    # BLEU scores
    gemma_bleu = [6.10, 7.89, 7.09, 6.93]
    llama_bleu = [5.87, 5.51, 7.58, 5.79]
    qwen_bleu =  [5.77, 4.89, 4.37, 5.56]

    ax1.plot(lambdas, gemma_bleu, "s-", color="#1976D2", linewidth=2.5, markersize=8, label="Gemma 3 (4B)")
    ax1.plot(lambdas, llama_bleu, "d-", color="#388E3C", linewidth=2.5, markersize=8, label="Llama 3.2 (3B)")
    ax1.plot(lambdas, qwen_bleu, "*-", color="#E65100", linewidth=2.5, markersize=9, label="Qwen 2.5 (3B)")

    ax1.set_xlabel("PSD Loss Weight ($\lambda_{\mathrm{PSD}}$)")
    ax1.set_ylabel("BLEU-4 Score")
    ax1.set_title("(a) Loss Weight ($\lambda_{\mathrm{PSD}}$) Sensitivity Curve", fontweight="bold")
    ax1.set_xticks(lambdas)
    ax1.set_xticklabels(["0.02\n(Low)", "0.05\n(Moderate)", "0.10\n(Optimal)", "0.20\n(High)"])
    ax1.set_ylim(3.5, 9.0)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Highlight optimal region
    ax1.axvspan(0.04, 0.11, color="green", alpha=0.08, label="Optimal Grounding Regime")

    # --- Subplot 2: Layer-Selective Supervision (Early vs Middle vs Deep) ---
    bands = ["Early Layers\n(1-33%)", "Middle Layers\n(34-66%)", "Deep Layers\n(67-100%)"]
    x = np.arange(len(bands))
    width = 0.25

    gemma_layer_fact = [77.63, 84.18, 83.18]
    llama_layer_fact = [80.44, 76.49, 89.05]
    qwen_layer_fact =  [69.52, 67.06, 75.14]

    r1 = ax2.bar(x - width, gemma_layer_fact, width, label="Gemma 3 (4B)", color="#1976D2", edgecolor="black")
    r2 = ax2.bar(x, llama_layer_fact, width, label="Llama 3.2 (3B)", color="#388E3C", edgecolor="black")
    r3 = ax2.bar(x + width, qwen_layer_fact, width, label="Qwen 2.5 (3B)", color="#E65100", edgecolor="black")

    ax2.set_ylabel("Factual Precision (%)")
    ax2.set_title("(b) Layer-Selective Placement Factual Precision", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bands, fontweight="semibold")
    ax2.set_ylim(60, 96)
    ax2.legend(loc="upper left", frameon=True)
    ax2.bar_label(r1, padding=2, fmt="%.1f%%")
    ax2.bar_label(r2, padding=2, fmt="%.1f%%")
    ax2.bar_label(r3, padding=2, fmt="%.1f%%")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    fig.suptitle("Cross-Model Ablation Studies: Hyperparameter and Architectural Sensitivity", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "fig5_ablation_analysis.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")
    if os.path.exists(ARTIFACT_DIR):
        shutil.copyfile(out_path, os.path.join(ARTIFACT_DIR, "fig5_ablation_analysis.png"))

if __name__ == "__main__":
    plot_ablation_analysis()
