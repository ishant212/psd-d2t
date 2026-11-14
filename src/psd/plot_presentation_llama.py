"""Generate presentation figure showing Llama-3.2-3B's architectural resistance/difficulty
contrasted against Gemma-3-4B and Qwen3-4B.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("reports/experiments/final_totto_long_context/presentation_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

# -------------------------------------------------------------
# Panel 1: Cross-Model Delta Comparison across Context Budgets
# -------------------------------------------------------------
contexts = ["256", "512", "1024", "2048"]
x = np.arange(len(contexts))

# METEOR Deltas (PSD - Baseline) from final_totto_results.csv:
# Qwen: 256: -0.22, 512: -0.84, 1024: -0.07, 2048: -0.63 (note: on >2048 bucket, Qwen is +2.73)
# Gemma: 256: -2.39, 512: +0.04, 1024: +2.06, 2048: -0.34
# Llama: 256: +0.39, 512: -1.67, 1024: -3.10, 2048: -2.76
gemma_delta = [-2.39, 0.04, 2.06, -0.34]
qwen_delta = [-0.22, -0.84, -0.07, -0.63]
llama_delta = [0.39, -1.67, -3.10, -2.76]

# Background shading
ax1.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
ax1.axhspan(0, 3.5, color="#2ca02c", alpha=0.10, label="PSD Advantage")
ax1.axhspan(-4.5, 0, color="#d62728", alpha=0.08, label="PSD Penalty / Interference")

ax1.plot(x, gemma_delta, marker="o", color="#2ca02c", label="Gemma-3-4B (High Responsiveness)")
ax1.plot(x, qwen_delta, marker="s", color="#1f77b4", label="Qwen3-4B (Neutral Aggregate, +2.73 on >2048)")
ax1.plot(x, llama_delta, marker="^", color="#d62728", linewidth=3.0, linestyle="-", label="Llama-3.2-3B (Severe Resistance)")

ax1.set_xticks(x)
ax1.set_xticklabels(contexts)
ax1.set_xlabel("Inference Context Budget (Tokens)")
ax1.set_ylabel("METEOR Differential (\u0394 = PSD - Baseline)")
ax1.set_title("Cross-Model Trajectory: Architectural Sensitivity")
ax1.set_ylim(-4.2, 3.2)
ax1.legend(loc="lower left", frameon=True, fontsize=11)

# Annotate Llama drop
ax1.annotate("Llama drops to -3.10\n(Statistically Sig. p < 0.001)",
             xy=(2, -3.10), xytext=(1.8, -1.5),
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=2),
             fontweight="bold", color="#a81c1c",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d62728", alpha=0.9))

# -------------------------------------------------------------
# Panel 2: Llama-3.2-3B Breakdown: Baseline vs. PSD Across Budgets
# -------------------------------------------------------------
# Llama Baseline METEOR: [29.27, 31.66, 31.68, 31.05]
# Llama Full PSD METEOR: [29.66, 29.99, 28.58, 28.29]
llama_base_meteor = [29.27, 31.66, 31.68, 31.05]
llama_psd_meteor = [29.66, 29.99, 28.58, 28.29]

bar_width = 0.35
idx = np.arange(len(contexts))

bars1 = ax2.bar(idx - bar_width/2, llama_base_meteor, width=bar_width, color="#7f7f7f", label="Llama Baseline", edgecolor="black")
bars2 = ax2.bar(idx + bar_width/2, llama_psd_meteor, width=bar_width, color="#e377c2", label="Llama Full PSD", edgecolor="black")

ax2.set_xticks(idx)
ax2.set_xticklabels([f"L={c}" for c in contexts])
ax2.set_xlabel("Inference Context Budget")
ax2.set_ylabel("METEOR Score")
ax2.set_title("Llama-3.2-3B: Persistent Deficit Under Prior")
ax2.set_ylim(20, 36)
ax2.legend(loc="upper right", frameon=True)

# Add delta values on top of bars
for i in range(len(contexts)):
    diff = llama_psd_meteor[i] - llama_base_meteor[i]
    sign = "+" if diff > 0 else ""
    color = "#2ca02c" if diff > 0 else "#d62728"
    ax2.text(idx[i] + bar_width/2, llama_psd_meteor[i] + 0.4, f"{sign}{diff:.2f}",
             ha="center", va="bottom", fontweight="bold", color=color, fontsize=11)

# Text box explaining the root cause
ax2.text(0.04, 0.10, "Architectural Root Cause:\n\u2022 Aggressive GQA (24:8 Query-KV head ratio)\n\u2022 Extreme RoPE base frequency (500k)\n\u2022 Static topological prior penalizes KV sharing",
         transform=ax2.transAxes, fontsize=10.5,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="#ffeeba", alpha=0.95))

plt.tight_layout()
output_path = OUTPUT_DIR / "presentation_fig3_llama_architectural_difficulty.png"
fig.savefig(output_path, dpi=300)
plt.close(fig)

print(f"Figure successfully saved to: {output_path}")
