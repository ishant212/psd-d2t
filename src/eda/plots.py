"""Publication-Quality Visualizations for Phase 1 EDA.

Generates clean, styled figures for:
1. Input length distributions
2. Output length distributions
3. Input vs Output length correlation
4. Structural elements complexity distribution
5. Long-context percentage threshold breakdown
6. Dataset size comparison
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

PALETTE = {
    "totto": "#1f77b4", "dart": "#ff7f0e", "webnlg": "#2ca02c",
    "TOTTO": "#1f77b4", "DART": "#ff7f0e", "WEBNLG": "#2ca02c"
}


class EDAPlotter:
    """Generates and exports Phase 1 EDA figures."""

    def __init__(self, output_dir: str = "reports/phase1/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_input_length_distribution(self, length_results: Dict[str, Dict[str, Any]]) -> str:
        """Plots KDE and histogram of input token lengths across datasets."""
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        
        for ds_name, stats in length_results.items():
            lengths = stats.get("raw_input_lengths", [])
            if lengths:
                sns.kdeplot(
                    lengths,
                    ax=ax,
                    label=f"{ds_name.upper()} (Median={stats.get('input_median', 0):.0f}, P95={stats.get('input_p95', 0):.0f})",
                    color=PALETTE.get(ds_name.lower(), "#333333"),
                    linewidth=2.2,
                    fill=True,
                    alpha=0.15
                )

        ax.set_title("Input Sequence Length Distribution across Datasets", fontweight="bold", pad=12)
        ax.set_xlabel("Serialized Input Length (Tokens / Words)")
        ax.set_ylabel("Density")
        ax.set_xlim(left=0)
        ax.legend(title="Dataset", frameon=True)
        
        path = os.path.join(self.output_dir, "01_input_length_distribution.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_output_length_distribution(self, length_results: Dict[str, Dict[str, Any]]) -> str:
        """Plots KDE and histogram of target reference lengths."""
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        
        for ds_name, stats in length_results.items():
            lengths = stats.get("raw_target_lengths", [])
            if lengths:
                sns.kdeplot(
                    lengths,
                    ax=ax,
                    label=f"{ds_name.upper()} (Median={stats.get('target_median', 0):.0f}, P95={stats.get('target_p95', 0):.0f})",
                    color=PALETTE.get(ds_name.lower(), "#333333"),
                    linewidth=2.2,
                    fill=True,
                    alpha=0.15
                )

        ax.set_title("Target Reference Length Distribution across Datasets", fontweight="bold", pad=12)
        ax.set_xlabel("Target Text Length (Tokens / Words)")
        ax.set_ylabel("Density")
        ax.set_xlim(left=0)
        ax.legend(title="Dataset", frameon=True)
        
        path = os.path.join(self.output_dir, "02_output_length_distribution.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_input_vs_output_length(self, length_results: Dict[str, Dict[str, Any]]) -> str:
        """Plots scatter / joint correlation between input length and target length."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300, sharey=True)
        
        for idx, (ds_name, stats) in enumerate(length_results.items()):
            if idx >= 3:
                break
            ax = axes[idx]
            in_len = stats.get("raw_input_lengths", [])
            out_len = stats.get("raw_target_lengths", [])
            color = PALETTE.get(ds_name.lower(), "#1f77b4")
            
            if in_len and out_len:
                # Subsample if too large for clean scatter
                if len(in_len) > 2000:
                    sample_indices = np.random.choice(len(in_len), 2000, replace=False)
                    in_len_s = [in_len[i] for i in sample_indices]
                    out_len_s = [out_len[i] for i in sample_indices]
                else:
                    in_len_s, out_len_s = in_len, out_len

                ax.scatter(in_len_s, out_len_s, alpha=0.35, color=color, edgecolors="none", s=25)
                
                # Fit trend line
                if len(in_len_s) > 1:
                    m, b = np.polyfit(in_len_s, out_len_s, 1)
                    xs = np.linspace(min(in_len_s), max(in_len_s), 100)
                    ax.plot(xs, m * xs + b, color="black", linestyle="--", linewidth=1.5, label=f"Fit (slope={m:.2f})")
                    ax.legend(loc="upper right")

            ax.set_title(f"{ds_name.upper()}", fontweight="bold")
            ax.set_xlabel("Input Length (Tokens)")
            if idx == 0:
                ax.set_ylabel("Target Length (Tokens)")

        fig.suptitle("Input vs. Target Reference Length Correlation", fontweight="bold", y=1.02)
        path = os.path.join(self.output_dir, "03_input_vs_output_length.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_structural_elements(self, topology_results: Dict[str, Dict[str, Any]]) -> str:
        """Bar plot showing average nodes, edges, or cells per dataset."""
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        
        categories = []
        element_counts = []
        dataset_labels = []

        for ds_name, stats in topology_results.items():
            if stats.get("type") == "table":
                categories.extend(["Total Cells", "Highlighted Cells"])
                element_counts.extend([stats.get("avg_cells", 0), stats.get("avg_highlighted_cells", 0)])
                dataset_labels.extend([ds_name.upper(), ds_name.upper()])
            else:
                categories.extend(["Nodes (Entities)", "Edges (Triples)"])
                element_counts.extend([stats.get("avg_nodes", 0), stats.get("avg_edges", 0)])
                dataset_labels.extend([ds_name.upper(), ds_name.upper()])

        df = pd.DataFrame({
            "Dataset": dataset_labels,
            "Structural Element": categories,
            "Average Count": element_counts
        })

        sns.barplot(data=df, x="Structural Element", y="Average Count", hue="Dataset", ax=ax, palette=PALETTE)
        ax.set_title("Structural Complexity across Datasets (Average per Example)", fontweight="bold", pad=12)
        ax.set_ylabel("Average Count per Example")
        ax.legend(title="Dataset", frameon=True)

        path = os.path.join(self.output_dir, "04_structural_elements_complexity.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_long_context_percentage(self, length_results: Dict[str, Dict[str, Any]]) -> str:
        """Bar chart of % examples exceeding candidate token thresholds."""
        thresholds = [512, 1024, 2048, 4096, 8192]
        records = []

        for ds_name, stats in length_results.items():
            for t in thresholds:
                pct = stats.get(f"pct_gt_{t}", 0.0)
                records.append({
                    "Dataset": ds_name.upper(),
                    "Threshold": f">{t} tokens",
                    "Percentage": pct
                })

        df = pd.DataFrame(records)
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        sns.barplot(data=df, x="Threshold", y="Percentage", hue="Dataset", ax=ax, palette=PALETTE)
        
        ax.set_title("Long-Context Prevalence across Context Thresholds", fontweight="bold", pad=12)
        ax.set_ylabel("Percentage of Examples (%)")
        ax.set_xlabel("Context Length Threshold")
        ax.legend(title="Dataset", frameon=True)

        path = os.path.join(self.output_dir, "05_long_context_percentage.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_dataset_sizes(self, dataset_stats: Dict[str, Dict[str, Any]]) -> str:
        """Grouped bar plot showing train / validation / test splits count per dataset."""
        records = []
        for ds_name, stats in dataset_stats.items():
            records.append({"Dataset": ds_name.upper(), "Split": "Train", "Count": stats.get("train_count", 0)})
            records.append({"Dataset": ds_name.upper(), "Split": "Validation", "Count": stats.get("validation_count", 0)})
            records.append({"Dataset": ds_name.upper(), "Split": "Test", "Count": stats.get("test_count", 0)})

        df = pd.DataFrame(records)
        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
        sns.barplot(data=df, x="Dataset", y="Count", hue="Split", ax=ax, palette="Blues")

        ax.set_title("Official Split Size Comparison across Datasets", fontweight="bold", pad=12)
        ax.set_ylabel("Number of Records")
        ax.legend(title="Split", frameon=True)

        path = os.path.join(self.output_dir, "06_dataset_size_comparison.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path
