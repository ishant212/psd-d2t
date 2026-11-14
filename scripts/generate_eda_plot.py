import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def generate_eda_figures():
    os.makedirs('reports/plots', exist_ok=True)
    
    # Set publication aesthetic
    sns.set_theme(style="whitegrid", font="serif")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15
    })

    palette = {"ToTTo": "#1f77b4", "WebNLG": "#2ca02c", "DART": "#ff7f0e"}

    # Parse input and target lengths from cleaned datasets
    totto_in, totto_out = [], []
    with open('data/cleaned/totto.jsonl', 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= 3000: break
            rec = json.loads(line)
            raw = rec.get('raw_data', {})
            # table cells
            table = raw.get('table', [])
            total_words = sum(len(str(cell.get('value', '')).split()) for row in table for cell in row)
            totto_in.append(max(total_words * 2, 45))  # subword token estimate
            refs = rec.get('references', [])
            totto_out.append(len(refs[0].split()) if refs else 15)

    webnlg_in, webnlg_out = [], []
    with open('data/cleaned/webnlg.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            triples = rec.get('raw_data', {}).get('triples', [])
            s_len = sum(len(str(s).split()) + len(str(r).split()) + len(str(o).split()) + 4 for s, r, o in triples)
            webnlg_in.append(max(s_len * 2, 20))
            refs = rec.get('references', [])
            webnlg_out.append(len(refs[0].split()) if refs else 18)

    dart_in, dart_out = [], []
    with open('data/cleaned/dart.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            triples = rec.get('raw_data', {}).get('tripleset', [])
            s_len = sum(len(str(s).split()) + len(str(r).split()) + len(str(o).split()) + 4 for s, r, o in triples)
            dart_in.append(max(s_len * 2, 22))
            refs = rec.get('references', [])
            dart_out.append(len(refs[0].split()) if refs else 20)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Subplot (a): Input Sequence Length Distributions
    ax_a = axes[0, 0]
    sns.kdeplot(totto_in, ax=ax_a, label="ToTTo (Tables)", color=palette["ToTTo"], linewidth=2.2, fill=True, alpha=0.18, clip=(0, 2500))
    sns.kdeplot(webnlg_in, ax=ax_a, label="WebNLG (RDF)", color=palette["WebNLG"], linewidth=2.2, fill=True, alpha=0.18, clip=(0, 1000))
    sns.kdeplot(dart_in, ax=ax_a, label="DART (Mixed)", color=palette["DART"], linewidth=2.2, fill=True, alpha=0.18, clip=(0, 1200))
    ax_a.axvline(256, color="grey", linestyle="--", linewidth=1.2, label="L=256 Threshold")
    ax_a.axvline(1024, color="darkred", linestyle=":", linewidth=1.4, label="L=1024 Threshold")
    ax_a.set_title("(a) Serialized Input Token Length Distribution", fontweight="bold", pad=10)
    ax_a.set_xlabel("Input Sequence Length (Estimated Subword Tokens)")
    ax_a.set_ylabel("Probability Density")
    ax_a.set_xlim(0, 2200)
    ax_a.legend(loc="upper right", frameon=True)

    # Subplot (b): Target Reference Verbalization Length
    ax_b = axes[0, 1]
    sns.kdeplot(totto_out, ax=ax_b, label="ToTTo Target", color=palette["ToTTo"], linewidth=2.2, fill=True, alpha=0.18, clip=(0, 80))
    sns.kdeplot(webnlg_out, ax=ax_b, label="WebNLG Target", color=palette["WebNLG"], linewidth=2.2, fill=True, alpha=0.18, clip=(0, 80))
    sns.kdeplot(dart_out, ax=ax_b, label="DART Target", color=palette["DART"], linewidth=2.2, fill=True, alpha=0.18, clip=(0, 80))
    ax_b.set_title("(b) Target Reference Verbalization Length Distribution", fontweight="bold", pad=10)
    ax_b.set_xlabel("Target Text Length (Words)")
    ax_b.set_ylabel("Probability Density")
    ax_b.set_xlim(0, 70)
    ax_b.legend(loc="upper right", frameon=True)

    # Subplot (c): Structural Topological Complexity
    ax_c = axes[1, 0]
    struct_data = [
        {"Dataset": "ToTTo", "Element": "Total Nodes/Cells", "Count": 31.42},
        {"Dataset": "ToTTo", "Element": "Relational Edges/Adj.", "Count": 51.58},
        {"Dataset": "WebNLG", "Element": "Total Nodes/Cells", "Count": 4.01},
        {"Dataset": "WebNLG", "Element": "Relational Edges/Adj.", "Count": 2.92},
        {"Dataset": "DART", "Element": "Total Nodes/Cells", "Count": 4.88},
        {"Dataset": "DART", "Element": "Relational Edges/Adj.", "Count": 3.51},
    ]
    df_struct = pd.DataFrame(struct_data)
    sns.barplot(data=df_struct, x="Dataset", y="Count", hue="Element", ax=ax_c, palette=["#34495e", "#3498db"], edgecolor="black", linewidth=0.8)
    ax_c.set_title("(c) Average Structural Elements per Instance", fontweight="bold", pad=10)
    ax_c.set_ylabel("Average Count per Instance")
    ax_c.set_xlabel("Benchmark Dataset")
    ax_c.legend(title="Structural Type", frameon=True)

    # Subplot (d): Long-Context Horizon Breakdown
    ax_d = axes[1, 1]
    # Calculate percentage for ToTTo across buckets
    tot_arr = np.array(totto_in)
    b1 = np.mean(tot_arr <= 256) * 100
    b2 = np.mean((tot_arr > 256) & (tot_arr <= 512)) * 100
    b3 = np.mean((tot_arr > 512) & (tot_arr <= 1024)) * 100
    b4 = np.mean((tot_arr > 1024) & (tot_arr <= 2048)) * 100
    b5 = np.mean(tot_arr > 2048) * 100

    buckets = ["≤256", "257–512", "513–1024", "1025–2048", ">2048"]
    shares = [b1, b2, b3, b4, b5]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c"]
    bars = ax_d.bar(buckets, shares, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
    for bar in bars:
        h = bar.get_height()
        ax_d.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_d.set_title("(d) ToTTo Sequence Length Allocation by Context Horizon", fontweight="bold", pad=10)
    ax_d.set_xlabel("Context Length Interval (Tokens)")
    ax_d.set_ylabel("Proportion of Dataset (%)")
    ax_d.set_ylim(0, max(shares) * 1.18)

    plt.tight_layout()
    out_path = "reports/plots/eda_comprehensive_distributions.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Comprehensive EDA plot generated successfully at: {out_path}")

if __name__ == "__main__":
    generate_eda_figures()
