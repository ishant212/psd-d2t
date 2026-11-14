"""
Generate Figure 6: Multi-Modal Topological Distance Spaces & Token-Level Projection
Matching the style of Fig. 4 from the reference paper (hierarchical breakout, clean pill boxes, dashed projection lines).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

OUT_DIR = r"C:\ME\SEM-7\PJT1\paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'

def generate_figure_6():
    fig = plt.figure(figsize=(15, 6.8), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Color Palette matching reference style
    c_graph = "#2563EB"       # Blue
    c_graph_bg = "#EFF6FF"
    c_table = "#0D9488"       # Teal
    c_table_bg = "#F0FDFA"
    c_align = "#7C3AED"       # Purple
    c_align_bg = "#F5F3FF"
    c_norm = "#D97706"        # Amber / Orange
    c_norm_bg = "#FFFBEB"
    c_out = "#DC2626"         # Terracotta / Red
    c_out_bg = "#FEF2F2"
    c_gray = "#64748B"

    # =========================================================================
    # SECTION 1 (LEFT): Dual Multi-Modal Ingestion Streams (Graph vs Table)
    # =========================================================================
    # Stream A: Knowledge Graph
    box_g = patches.FancyBboxPatch((0.5, 4.6), 3.8, 3.4, boxstyle="round,pad=0.15",
                                   fc=c_graph_bg, ec=c_graph, lw=1.8)
    ax.add_patch(box_g)
    ax.text(2.4, 7.7, "3.3.1 Knowledge Graphs\n$G = (V, E)$ [DART / WebNLG]", 
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=c_graph)
    
    # Graph Details
    p1 = patches.FancyBboxPatch((0.8, 6.0), 3.2, 0.9, boxstyle="round,pad=0.08",
                                fc="#FFFFFF", ec=c_graph, lw=1.2)
    ax.add_patch(p1)
    ax.text(2.4, 6.45, "Undirected Projection\n$G_{\\mathrm{sym}} = (V, E_{\\mathrm{sym}})$", 
            ha="center", va="center", fontsize=8, color="#1E293B", fontweight="bold")
    
    p2 = patches.FancyBboxPatch((0.8, 4.8), 3.2, 1.0, boxstyle="round,pad=0.08",
                                fc="#FFFFFF", ec=c_graph, lw=1.2)
    ax.add_patch(p2)
    ax.text(2.4, 5.3, "BFS All-Pairs Shortest Path\n$d_G(u,v) = \\min_{\\pi} \\sum w(e)$\nDisconnected Cutoff: $\\max d_G + 1.0$", 
            ha="center", va="center", fontsize=7.5, color="#1E293B")

    # Stream B: 2D Relational Tables
    box_t = patches.FancyBboxPatch((0.5, 0.6), 3.8, 3.4, boxstyle="round,pad=0.15",
                                   fc=c_table_bg, ec=c_table, lw=1.8)
    ax.add_patch(box_t)
    ax.text(2.4, 3.7, "3.3.2 Relational Tables\n$T = \\{c_{r,k}\\}$ [ToTTo Grid]", 
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=c_table)

    p3 = patches.FancyBboxPatch((0.8, 2.0), 3.2, 0.9, boxstyle="round,pad=0.08",
                                fc="#FFFFFF", ec=c_table, lw=1.2)
    ax.add_patch(p3)
    ax.text(2.4, 2.45, "2D Matrix Coordinates\nCell $u=(r_u, c_u)$, Cell $v=(r_v, c_v)$", 
            ha="center", va="center", fontsize=8, color="#1E293B", fontweight="bold")

    p4 = patches.FancyBboxPatch((0.8, 0.8), 3.2, 1.0, boxstyle="round,pad=0.08",
                                fc="#FFFFFF", ec=c_table, lw=1.2)
    ax.add_patch(p4)
    ax.text(2.4, 1.3, "Manhattan Coordinate Metric\n$d_T(u,v) = |r_u - r_v| + |c_u - c_v|$\nAdjacent row/col cells $\\rightarrow d_T=1$", 
            ha="center", va="center", fontsize=7.5, color="#1E293B")

    # Connectors from Left to Middle
    ax.annotate("", xy=(4.9, 5.2), xytext=(4.3, 6.0),
                arrowprops=dict(arrowstyle="->", lw=2, color=c_graph))
    ax.annotate("", xy=(4.9, 3.8), xytext=(4.3, 2.5),
                arrowprops=dict(arrowstyle="->", lw=2, color=c_table))

    # =========================================================================
    # SECTION 2 (CENTER): Token-to-Node Span Alignment Engine mu(i)
    # =========================================================================
    box_m = patches.FancyBboxPatch((4.9, 2.2), 4.2, 4.4, boxstyle="round,pad=0.15",
                                   fc=c_align_bg, ec=c_align, lw=2.0)
    ax.add_patch(box_m)
    ax.text(7.0, 6.2, "Token-to-Node Alignment\nEngine $\\mu(i)$", 
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=c_align)

    # Content Token Mapping Box
    pm1 = patches.FancyBboxPatch((5.2, 4.5), 3.6, 1.2, boxstyle="round,pad=0.08",
                                 fc="#FFFFFF", ec="#10B981", lw=1.5)
    ax.add_patch(pm1)
    ax.text(7.0, 5.1, "Content / Subword Tokens ($x_i$)\n$\\mu(i) = u \\in V$\n$[s_i', e_i') \\cap [s_u, e_u) \\neq \\emptyset$",
            ha="center", va="center", fontsize=8, color="#047857", fontweight="bold")

    # Delimiter / Unmapped Box
    pm2 = patches.FancyBboxPatch((5.2, 2.8), 3.6, 1.4, boxstyle="round,pad=0.08",
                                 fc="#FFFFFF", ec="#EF4444", lw=1.5)
    ax.add_patch(pm2)
    ax.text(7.0, 3.5, "Structural Delimiters & Punctuation\n`[TRIPLE]`, `[ROW]`, `[CELL]`, `|`\n$\\mu(i) = \\emptyset$\n$\\rightarrow$ Maximum Penalty Assigned",
            ha="center", va="center", fontsize=8, color="#B91C1C", fontweight="bold")

    # Connectors from Center to Right Breakout
    ax.annotate("", xy=(9.8, 5.0), xytext=(9.1, 5.0),
                arrowprops=dict(arrowstyle="->", lw=2, color=c_align))

    # Dashed breakout projection lines (like Fig 4 in reference!)
    ax.plot([9.1, 9.8], [6.4, 7.8], linestyle="--", lw=1.3, color=c_gray, alpha=0.8)
    ax.plot([9.1, 9.8], [2.4, 0.8], linestyle="--", lw=1.3, color=c_gray, alpha=0.8)

    # =========================================================================
    # SECTION 3 (RIGHT BREAKOUT): Bounded Unit Normalization & Target Distance
    # =========================================================================
    box_r = patches.FancyBboxPatch((9.8, 0.6), 4.7, 7.4, boxstyle="round,pad=0.18",
                                   fc=c_norm_bg, ec=c_norm, lw=2.0)
    ax.add_patch(box_r)
    ax.text(12.15, 7.6, "3.3.3 Bounded Normalization Space $[0.0, 1.0]$\nToken-to-Token Distance Formulation", 
            ha="center", va="center", fontsize=10, fontweight="bold", color=c_norm)

    # Sub-block 1: Sequence Distance
    pr1 = patches.FancyBboxPatch((10.1, 5.8), 4.1, 1.4, boxstyle="round,pad=0.08",
                                 fc="#FFFFFF", ec="#B45309", lw=1.3)
    ax.add_patch(pr1)
    ax.text(12.15, 6.7, "Sequential Positional Metric:", fontsize=8.5, fontweight="bold", color="#B45309", ha="center")
    ax.text(12.15, 6.2, "$d_P^{\\mathrm{norm}}(i, j) = \\frac{|i - j|}{\\max(N - 1, 1)}$\nScales raw displacements (1 to 1000+) to $[0, 1]$", 
            fontsize=8, ha="center", va="center", color="#1E293B")

    # Sub-block 2: Graph Geodesic Normalization
    pr2 = patches.FancyBboxPatch((10.1, 3.7), 4.1, 1.9, boxstyle="round,pad=0.08",
                                 fc="#FFFFFF", ec=c_graph, lw=1.3)
    ax.add_patch(pr2)
    ax.text(12.15, 5.25, "Normalized Graph Topology Distance:", fontsize=8.5, fontweight="bold", color=c_graph, ha="center")
    ax.text(12.15, 4.45, 
            r"$\bullet\ \mu(i), \mu(j) \neq \emptyset:\ d_G^{\mathrm{norm}}(i, j) = \frac{d_G(\mu(i), \mu(j))}{\max_{u,v} d_G(u, v)}$" + "\n\n" +
            r"$\bullet\ \mu(i) = \emptyset\ \mathrm{or}\ \mu(j) = \emptyset:\ d_G^{\mathrm{norm}}(i, j) = 1.0$" + "\n" +
            r"   (Maximum penalty for unmapped / delimiters)",
            fontsize=7.8, ha="center", va="center", color="#1E293B")

    # Sub-block 3: Table Manhattan Normalization
    pr3 = patches.FancyBboxPatch((10.1, 1.6), 4.1, 1.9, boxstyle="round,pad=0.08",
                                 fc="#FFFFFF", ec=c_table, lw=1.3)
    ax.add_patch(pr3)
    ax.text(12.15, 3.15, "Normalized Table Topology Distance:", fontsize=8.5, fontweight="bold", color=c_table, ha="center")
    ax.text(12.15, 2.35, 
            r"$\bullet\ \mu(i), \mu(j) \neq \emptyset:\ d_T^{\mathrm{norm}}(i, j) = \frac{d_T(\mu(i), \mu(j))}{\max_{u,v} d_T(u, v)}$" + "\n\n" +
            r"$\bullet\ \mu(i) = \emptyset\ \mathrm{or}\ \mu(j) = \emptyset:\ d_T^{\mathrm{norm}}(i, j) = 1.0$" + "\n" +
            r"   (Maximum penalty for unmapped / delimiters)",
            fontsize=7.8, ha="center", va="center", color="#1E293B")

    # Bottom output highlight
    pr_out = patches.FancyBboxPatch((10.1, 0.8), 4.1, 0.65, boxstyle="round,pad=0.06",
                                    fc=c_out_bg, ec=c_out, lw=1.4)
    ax.add_patch(pr_out)
    ax.text(12.15, 1.12, "Feeds Directly to Composite Affinity:\n$R_{ij} = \\alpha S_{ij} - \\beta d_P^{\\mathrm{norm}}(i,j) - \\gamma d_{G/T}^{\\mathrm{norm}}(i,j)$", 
            ha="center", va="center", fontsize=8, fontweight="bold", color=c_out)

    fig.suptitle('Figure: Structural Overview of Multi-Modal Topological Distance Spaces and Token Span Projection', 
                 fontsize=12.5, fontweight='bold', y=0.97)

    out_path = os.path.join(OUT_DIR, "fig6_topological_distance_spaces.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    generate_figure_6()
