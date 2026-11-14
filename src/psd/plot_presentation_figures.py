"""Generate clean, presentation-ready figures highlighting PSD's long-context advantages,
truncation robustness, and architectural analysis for slides.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("reports/experiments/final_totto_long_context/presentation_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set clean aesthetic for slides
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
})

# -------------------------------------------------------------
# Figure 1: The Long-Context Scaling Crossover (Delta vs Length)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

buckets = ["257-512\n(Control)", "513-1024\n(Medium)", "1025-2048\n(Long)", ">2048\n(Extreme)"]
x = np.arange(len(buckets))

# Qwen METEOR Delta at L=1024 and L=2048
# From context_bucket_summary.csv:
# Qwen L=1024: -0.46, -4.11, +1.82, +2.48
# Qwen L=2048: -0.46, -4.11, -0.70, +2.73
qwen_l1024_meteor = [-0.46, -4.11, 1.82, 2.48]
qwen_l2048_meteor = [-0.46, -4.11, -0.70, 2.73]

ax1.axhline(0, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
ax1.plot(x, qwen_l1024_meteor, marker="o", color="#1f77b4", label="Qwen3-4B (Context 1024)")
ax1.plot(x, qwen_l2048_meteor, marker="s", color="#004c6d", linestyle="-.", label="Qwen3-4B (Context 2048)")
ax1.fill_between(x, 0, [max(0, v) for v in qwen_l2048_meteor], color="#2ca02c", alpha=0.15, label="PSD Outperforms")

ax1.set_xticks(x)
ax1.set_xticklabels(buckets)
ax1.set_ylabel("METEOR Advantage (\u0394 = PSD - Baseline)")
ax1.set_title("Qwen3-4B: METEOR Advantage by Table Length")
ax1.legend(loc="upper left", frameon=True)
ax1.set_ylim(-5, 4)

# Gemma BERTScore F1 Delta across buckets
# Gemma L=1024 BERTScore Delta:
# 257-512: +0.27, 513-1024: +0.65, 1025-2048: +17.61, >2048: +8.85
gemma_bertscore_delta = [0.27, 0.65, 17.61, 8.85]

ax2.axhline(0, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
bars = ax2.bar(x, gemma_bertscore_delta, color="#2ca02c", width=0.55, alpha=0.85, edgecolor="black")
ax2.set_xticks(x)
ax2.set_xticklabels(buckets)
ax2.set_ylabel("BERTScore F1 Advantage (\u0394 points)")
ax2.set_title("Gemma-3-4B: Semantic Fidelity Advantage (Context 1024)")
ax2.set_ylim(-2, 20)

for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f"+{yval:.1f}", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "presentation_fig1_long_context_crossover.png", dpi=300)
plt.close(fig)

# -------------------------------------------------------------
# Figure 2: The Hardest Regime & Truncation Robustness
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: Extreme Long Context (>2048 tokens) Head-to-Head
metrics = ["METEOR", "BLEU-4", "ROUGE-L"]
# Qwen at L=2048 on >2048 bucket:
# Baseline: METEOR 35.51, BLEU 12.01, ROUGE-L 37.89
# PSD: METEOR 38.24, BLEU 13.41, ROUGE-L 40.00
# Delta: +2.73, +1.40, +2.11
base_vals = [35.51, 12.01, 37.89]
psd_vals = [38.24, 13.41, 40.00]

bar_width = 0.32
idx = np.arange(len(metrics))
b1 = ax1.bar(idx - bar_width/2, base_vals, width=bar_width, color="#7f7f7f", label="Baseline", edgecolor="black")
b2 = ax1.bar(idx + bar_width/2, psd_vals, width=bar_width, color="#1f77b4", label="Full PSD", edgecolor="black")

ax1.set_xticks(idx)
ax1.set_xticklabels(metrics)
ax1.set_ylabel("Score")
ax1.set_title("Qwen3-4B on Extreme Tables (>2048 tokens, Budget 2048)")
ax1.legend(loc="upper left", frameon=True)
ax1.set_ylim(0, 48)

for i in range(len(metrics)):
    diff = psd_vals[i] - base_vals[i]
    ax1.text(idx[i] + bar_width/2, psd_vals[i] + 1.0, f"+{diff:.2f}", ha="center", va="bottom", color="#004c6d", fontweight="bold")

# Right: Truncation Robustness
# When inputs are truncated by context limit:
# Qwen L=1024: Base 36.15 vs PSD 38.27 (+2.12)
# Qwen L=2048: Base 35.04 vs PSD 38.80 (+3.76)
# Gemma L=1024: Base 6.64 vs PSD 9.57 (+2.93)
cases = ["Qwen (L=1024)", "Qwen (L=2048)", "Gemma (L=1024)"]
trunc_base = [36.15, 35.04, 6.64]
trunc_psd = [38.27, 38.80, 9.57]

idx2 = np.arange(len(cases))
ax2.bar(idx2 - bar_width/2, trunc_base, width=bar_width, color="#7f7f7f", label="Baseline", edgecolor="black")
ax2.bar(idx2 + bar_width/2, trunc_psd, width=bar_width, color="#2ca02c", label="Full PSD", edgecolor="black")

ax2.set_xticks(idx2)
ax2.set_xticklabels(cases)
ax2.set_ylabel("METEOR Score on Truncated Inputs")
ax2.set_title("Robustness Under Context Truncation")
ax2.legend(loc="upper left", frameon=True)
ax2.set_ylim(0, 48)

for i in range(len(cases)):
    diff = trunc_psd[i] - trunc_base[i]
    ax2.text(idx2[i] + bar_width/2, trunc_psd[i] + 1.0, f"+{diff:.2f}", ha="center", va="bottom", color="#1b661b", fontweight="bold")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "presentation_fig2_extreme_and_truncation.png", dpi=300)
plt.close(fig)

print("Presentation figures successfully generated in:", OUTPUT_DIR)
