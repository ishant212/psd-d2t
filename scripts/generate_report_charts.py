import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Output directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "reports", "experiments", "plots")
ARTIFACT_DIR = r"C:\Users\laksh\.gemini\antigravity\brain\d0d10d4b-0a53-48af-9173-30adee9add8e"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set high-quality style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300,
})

colors_baseline = "#5C6BC0"   # Indigo
colors_psd = "#00897B"        # Teal

def save_and_copy(fig, filename):
    local_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(local_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {local_path}")
    if os.path.exists(ARTIFACT_DIR):
        artifact_path = os.path.join(ARTIFACT_DIR, filename)
        shutil.copyfile(local_path, artifact_path)
        print(f"Copied to artifact dir: {artifact_path}")

# ==============================================================================
# Figure 1: Long-Context (L=1024 and L=2048) Performance Comparison
# ==============================================================================
def plot_long_context_performance():
    models = ["Gemma-4B\n(L=1024)", "Gemma-4B\n(L=2048)", "Qwen-3B\n(L=1024)", "Qwen-3B\n(L=2048)", "Llama-3B\n(L=1024)", "Llama-3B\n(L=2048)"]
    x = np.arange(len(models))
    width = 0.35

    # Data
    rouge_baseline = [57.22, 58.38, 51.03, 57.33, 53.68, 57.10]
    rouge_psd =      [60.74, 65.06, 57.21, 59.45, 53.26, 55.22]

    bleu_baseline =  [19.13, 22.70, 12.31, 19.76, 16.60, 22.59]
    bleu_psd =       [19.73, 27.47, 12.67, 18.82, 15.44, 22.03]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: ROUGE-L
    rects1 = ax1.bar(x - width/2, rouge_baseline, width, label="Baseline LoRA", color=colors_baseline, alpha=0.9, edgecolor="black")
    rects2 = ax1.bar(x + width/2, rouge_psd, width, label="PSD (Ours)", color=colors_psd, alpha=0.9, edgecolor="black")
    ax1.set_ylabel("ROUGE-L Score")
    ax1.set_title("(a) Text Overlap & Recall (ROUGE-L)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="semibold")
    ax1.set_ylim(45, 72)
    ax1.legend(loc="upper left", frameon=True)
    ax1.bar_label(rects1, padding=3, fmt="%.1f")
    ax1.bar_label(rects2, padding=3, fmt="%.1f")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Highlight Gemma L2048
    ax1.annotate("+6.68", xy=(1 + width/2, 65.06), xytext=(1, 69),
                 arrowprops=dict(facecolor="green", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, fontweight="bold", color="green", ha="center")
    # Highlight Qwen L1024
    ax1.annotate("+6.18", xy=(2 + width/2, 57.21), xytext=(2, 62),
                 arrowprops=dict(facecolor="green", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, fontweight="bold", color="green", ha="center")

    # Subplot 2: BLEU-4
    rects3 = ax2.bar(x - width/2, bleu_baseline, width, label="Baseline LoRA", color=colors_baseline, alpha=0.9, edgecolor="black")
    rects4 = ax2.bar(x + width/2, bleu_psd, width, label="PSD (Ours)", color=colors_psd, alpha=0.9, edgecolor="black")
    ax2.set_ylabel("BLEU-4 Score")
    ax2.set_title("(b) Precision & Generation Fluency (BLEU-4)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="semibold")
    ax2.set_ylim(8, 32)
    ax2.legend(loc="upper left", frameon=True)
    ax2.bar_label(rects3, padding=3, fmt="%.1f")
    ax2.bar_label(rects4, padding=3, fmt="%.1f")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    # Highlight Gemma L2048 BLEU
    ax2.annotate("+4.77", xy=(1 + width/2, 27.47), xytext=(1, 29.5),
                 arrowprops=dict(facecolor="green", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, fontweight="bold", color="green", ha="center")

    fig.suptitle("Long-Context Benchmark Performance ($L=1024$ and $L=2048$ Tokens)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_and_copy(fig, "fig1_long_context_performance.png")

# ==============================================================================
# Figure 2: Hallucination Rate Reduction across Architectures
# ==============================================================================
def plot_hallucination_reduction():
    models = ["Qwen-3B\n(L=2048)", "Qwen-3B\n(L=1024)", "Gemma-4B\n(L=2048)", "Gemma-4B\n(L=256)", "Llama-8B\n(L=256)", "Mistral-7B\n(L=256)"]
    baseline_halluc = [11.52, 20.00, 12.50, 23.02, 41.48, 23.75]
    psd_halluc =      [ 9.69, 12.16, 11.82, 16.07, 17.44, 20.29]

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5))
    r1 = ax.bar(x - width/2, baseline_halluc, width, label="Baseline Hallucination Rate (%)", color="#E57373", edgecolor="black")
    r2 = ax.bar(x + width/2, psd_halluc, width, label="PSD Hallucination Rate (%)", color="#2E7D32", edgecolor="black")

    ax.set_ylabel("Hallucination Rate (%) [Lower is Better]")
    ax.set_title("Hallucination Rate Reduction Across Sequence Lengths & Architectures", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="semibold")
    ax.set_ylim(0, 50)
    ax.legend(loc="upper right", frameon=True)
    ax.bar_label(r1, padding=3, fmt="%.1f%%")
    ax.bar_label(r2, padding=3, fmt="%.1f%%")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Highlight massive drops
    ax.annotate("▼ -24.0%", xy=(4 + width/2, 17.44), xytext=(4, 32),
                arrowprops=dict(facecolor="#1B5E20", shrink=0.08, width=1.5, headwidth=6),
                fontsize=11, fontweight="bold", color="#1B5E20", ha="center")
    ax.annotate("Sub-10%!\n(9.69%)", xy=(0 + width/2, 9.69), xytext=(0, 22),
                arrowprops=dict(facecolor="#1B5E20", shrink=0.08, width=1.5, headwidth=6),
                fontsize=10, fontweight="bold", color="#1B5E20", ha="center")

    plt.tight_layout()
    save_and_copy(fig, "fig2_hallucination_reduction.png")

# ==============================================================================
# Figure 3: Factual Precision Scaling from 256 -> 1024 -> 2048 Tokens
# ==============================================================================
def plot_context_scaling():
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    context_lens = [256, 1024, 2048]

    # Gemma 4B
    ax.plot(context_lens, [76.98, 84.43, 87.50], "o--", color="#BDBDBD", label="Gemma-4B Baseline", linewidth=2)
    ax.plot(context_lens, [83.93, 83.16, 88.18], "s-", color="#1976D2", label="Gemma-4B PSD (Ours)", linewidth=2.5, markersize=8)

    # Llama 3B
    ax.plot(context_lens, [78.91, 84.01, 87.38], "^--", color="#9E9E9E", label="Llama-3B Baseline", linewidth=2)
    ax.plot(context_lens, [77.54, 86.32, 87.82], "d-", color="#388E3C", label="Llama-3B PSD (Ours)", linewidth=2.5, markersize=8)

    # Qwen 3B
    ax.plot(context_lens, [67.02, 80.00, 88.48], "x--", color="#757575", label="Qwen-3B Baseline", linewidth=2)
    ax.plot(context_lens, [67.38, 87.84, 90.31], "*-", color="#E65100", label="Qwen-3B PSD (Ours)", linewidth=2.5, markersize=9)

    ax.set_xlabel("Context Length (Tokens)")
    ax.set_ylabel("Factual Claim Precision (%)")
    ax.set_title("Factual Precision Scaling Curve ($L = 256 \\rightarrow 1024 \\rightarrow 2048$)", fontweight="bold")
    ax.set_xticks([256, 1024, 2048])
    ax.set_xticklabels(["256 Tokens", "1024 Tokens", "2048 Tokens"])
    ax.set_ylim(60, 95)
    ax.legend(loc="lower right", frameon=True, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Annotation for Qwen peaking over 90%
    ax.annotate("Qwen PSD hits 90.31%\nPrecision at L=2048", xy=(2048, 90.31), xytext=(1700, 92.5),
                arrowprops=dict(facecolor="#E65100", shrink=0.08, width=1.5, headwidth=6),
                fontsize=10, fontweight="bold", color="#E65100", ha="right")

    plt.tight_layout()
    save_and_copy(fig, "fig3_factual_precision_scaling.png")

# ==============================================================================
# Figure 4: Relative Metric Gains of PSD over Baseline (% improvement)
# ==============================================================================
def plot_relative_gains():
    models = ["Llama 3.1\n(8B)", "Mistral\n(7B)", "Gemma 3\n(4B)", "Qwen 2.5\n(3B L1024)", "Qwen 2.5\n(7B)", "Llama 3.2\n(3B)"]
    bleu_gain = [
        ((14.84 - 5.97) / 5.97) * 100,      # Llama 8B: +148.6%
        ((14.07 - 9.60) / 9.60) * 100,      # Mistral 7B: +46.6%
        ((7.09 - 4.68) / 4.68) * 100,        # Gemma 4B: +51.5%
        ((12.67 - 12.31) / 12.31) * 100,    # Qwen 3B L1024: +2.9%
        ((7.01 - 6.79) / 6.79) * 100,        # Qwen 7B: +3.2%
        ((7.58 - 5.37) / 5.37) * 100,        # Llama 3B: +41.2%
    ]
    rouge_gain = [
        ((37.99 - 36.32) / 36.32) * 100,
        ((50.58 - 49.04) / 49.04) * 100,
        ((47.42 - 30.73) / 30.73) * 100,    # Gemma 4B: +54.3%
        ((57.21 - 51.03) / 51.03) * 100,    # Qwen 3B L1024: +12.1%
        ((46.19 - 46.25) / 46.25) * 100,
        ((41.16 - 40.07) / 40.07) * 100,
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    r1 = ax.bar(x - width/2, bleu_gain, width, label="Relative BLEU-4 Gain (%)", color="#3949AB", edgecolor="black")
    r2 = ax.bar(x + width/2, rouge_gain, width, label="Relative ROUGE-L Gain (%)", color="#00897B", edgecolor="black")

    ax.set_ylabel("Relative Improvement over Baseline (%)")
    ax.set_title("Relative Percentage Gains from Topology-Aware PSD Supervision", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="semibold")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(-5, 165)
    ax.bar_label(r1, padding=3, fmt="+%.1f%%")
    ax.bar_label(r2, padding=3, fmt="+%.1f%%")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_and_copy(fig, "fig4_relative_metric_gains.png")

if __name__ == "__main__":
    plot_long_context_performance()
    plot_hallucination_reduction()
    plot_context_scaling()
    plot_relative_gains()
    print("\nAll 4 figures generated and saved successfully!")
