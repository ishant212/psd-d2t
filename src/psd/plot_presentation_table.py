"""Generate two publication-grade academic table images matching Table 2 style:
1. By Source-Length Buckets (257-512, 513-1024, 1025-2048, >2048) at Budget 1024
2. Multi-Metric Comprehensive Long-Context Table (BLEU-4, ROUGE-L, METEOR, BERTScore, Fact. Prec.)
"""

import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("reports/experiments/final_totto_long_context/presentation_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Table 1: Source-Length Buckets (Table Scaling) at Context 1024
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13.5, 5.2), dpi=300)
ax.axis('off')

headers_buckets = [
    "Family", 
    "Model Variant", 
    "Size\n(Billion)", 
    "257–512\n(Control)", 
    "513–1024\n(Medium)", 
    "1025–2048\n(Long)", 
    ">2048\n(Extreme)", 
    "Overall\n(160 tables)"
]

# METEOR scores at context 1024 from context_bucket_summary.csv and final_totto_results.csv
rows_buckets = [
    # Qwen Family
    ["Qwen", "Qwen3-4B (Baseline)", "4.0", "53.47", "54.93", "44.66", "26.86", "44.98"],
    ["", "Qwen3-4B (Full PSD)", "4.0", "53.00", "50.82", "46.48*", "29.34*", "44.91"],
    # Gemma Family
    ["Gemma", "Gemma-3-4B (Baseline)", "4.0", "38.04", "39.93", "9.53", "3.47", "22.74"],
    ["", "Gemma-3-4B (Full PSD)", "4.0", "37.81", "42.17*", "11.89*", "7.30*", "24.80*"],
    # Llama Family
    ["Llama", "Llama-3.2-3B (Baseline)", "3.2", "37.46", "36.30", "26.12", "26.86", "31.68"],
    ["", "Llama-3.2-3B (Full PSD)", "3.2", "32.78", "33.81", "24.91", "22.83", "28.58"],
]

t1 = ax.table(cellText=rows_buckets, colLabels=headers_buckets, loc='center', cellLoc='center')
t1.auto_set_font_size(False)
t1.set_fontsize(11.5)
t1.scale(1.0, 2.0)

for (r, c), cell in t1.get_celld().items():
    cell.set_edgecolor('white')
    cell.set_linewidth(0)
    if r == 0:
        cell.set_text_props(weight='bold', color='black')
        cell.visible_edges = 'TB'
        cell.set_edgecolor('black')
        cell.set_linewidth(1.8)
    else:
        text = cell.get_text().get_text()
        if "*" in text:
            cell.set_text_props(weight='bold', color='#004c6d')
        elif r in [1, 3, 5]:
            base_txt = rows_buckets[r-1][c].replace("*","")
            psd_txt = rows_buckets[r][c].replace("*","")
            try:
                if float(base_txt) > float(psd_txt):
                    # bold baseline
                    t1.get_celld()[(r, c)].set_text_props()
            except:
                pass
        if r in [3, 5]:
            cell.visible_edges = 'T'
            cell.set_edgecolor('#d0d0d0')
            cell.set_linewidth(0.8)
        if r == len(rows_buckets):
            cell.visible_edges = 'B'
            cell.set_edgecolor('black')
            cell.set_linewidth(1.8)

# Clean asterisk and bold winning baseline where needed
for r in [1, 3, 5]:
    for c in range(3, len(headers_buckets)):
        base_val = float(rows_buckets[r-1][c].replace("*",""))
        psd_val = float(rows_buckets[r][c].replace("*",""))
        cell_base = t1.get_celld()[(r, c)]
        cell_psd = t1.get_celld()[(r+1, c)]
        if base_val >= psd_val:
            cell_base.set_text_props(weight='bold')
        else:
            cell_psd.set_text_props(weight='bold', color='#004c6d')

for (r, c), cell in t1.get_celld().items():
    txt = cell.get_text().get_text()
    if "*" in txt:
        cell.get_text().set_text(txt.replace("*",""))

plt.title("Table 1. Long-Context METEOR Evaluation Across Table Length Buckets (Context = 1024)", 
          fontsize=14, weight='bold', pad=25, loc='center')
plt.tight_layout()
out1 = OUTPUT_DIR / "presentation_table_source_buckets.png"
plt.savefig(out1, bbox_inches='tight', dpi=300)
plt.close(fig)

# -------------------------------------------------------------
# Table 2: Multi-Metric Long-Context Evaluation (Budget 1024)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13.5, 5.2), dpi=300)
ax.axis('off')

headers_multimetric = [
    "Family", 
    "Model Variant", 
    "Size\n(Billion)", 
    "BLEU-4", 
    "ROUGE-L", 
    "METEOR", 
    "BERTScore\nF1", 
    "Factual\nPrecision (%)"
]

rows_multi = [
    # Qwen
    ["Qwen", "Qwen3-4B (Baseline)", "4.0", "21.30", "41.79", "44.98", "92.47", "88.30"],
    ["", "Qwen3-4B (Full PSD)", "4.0", "20.72", "41.18", "44.91", "92.49*", "88.34*"],
    # Gemma
    ["Gemma", "Gemma-3-4B (Baseline)", "4.0", "9.14", "21.43", "22.74", "57.17", "92.13"],
    ["", "Gemma-3-4B (Full PSD)", "4.0", "10.45*", "23.64*", "24.80*", "64.01*", "90.93"],
    # Llama
    ["Llama", "Llama-3.2-3B (Baseline)", "3.2", "11.33", "30.13", "31.68", "90.01", "88.22"],
    ["", "Llama-3.2-3B (Full PSD)", "3.2", "9.42", "28.28", "28.58", "89.70", "86.71"],
]

t2 = ax.table(cellText=rows_multi, colLabels=headers_multimetric, loc='center', cellLoc='center')
t2.auto_set_font_size(False)
t2.set_fontsize(11.5)
t2.scale(1.0, 2.0)

for (r, c), cell in t2.get_celld().items():
    cell.set_edgecolor('white')
    cell.set_linewidth(0)
    if r == 0:
        cell.set_text_props(weight='bold', color='black')
        cell.visible_edges = 'TB'
        cell.set_edgecolor('black')
        cell.set_linewidth(1.8)
    else:
        if r in [3, 5]:
            cell.visible_edges = 'T'
            cell.set_edgecolor('#d0d0d0')
            cell.set_linewidth(0.8)
        if r == len(rows_multi):
            cell.visible_edges = 'B'
            cell.set_edgecolor('black')
            cell.set_linewidth(1.8)

for r in [1, 3, 5]:
    for c in range(3, len(headers_multimetric)):
        base_val = float(rows_multi[r-1][c].replace("*",""))
        psd_val = float(rows_multi[r][c].replace("*",""))
        cell_base = t2.get_celld()[(r, c)]
        cell_psd = t2.get_celld()[(r+1, c)]
        if base_val >= psd_val:
            cell_base.set_text_props(weight='bold')
        else:
            cell_psd.set_text_props(weight='bold', color='#004c6d')

for (r, c), cell in t2.get_celld().items():
    txt = cell.get_text().get_text()
    if "*" in txt:
        cell.get_text().set_text(txt.replace("*",""))

plt.title("Table 2. Multi-Metric Long-Context Evaluation on ToTTo (Context Budget = 1024)", 
          fontsize=14, weight='bold', pad=25, loc='center')
plt.tight_layout()
out2 = OUTPUT_DIR / "presentation_table_multimetric_results.png"
plt.savefig(out2, bbox_inches='tight', dpi=300)
plt.close(fig)

print("Both table images generated successfully!")
