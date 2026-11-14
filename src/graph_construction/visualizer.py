"""
Graph Visualizer for Stage D: NetworkX example graph rendering.
"""
import os
import logging
from typing import List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger(__name__)

class GraphVisualizer:
    """Renders NetworkX DiGraph instances into PNG images for visual confirmation."""

    @staticmethod
    def draw_graph(G: nx.DiGraph, title: str, output_path: str):
        """Draws a NetworkX DiGraph and saves to output_path."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.figure(figsize=(8, 6))

        # Position nodes using spring layout with fallback to shell_layout
        try:
            pos = nx.spring_layout(G, seed=42, k=0.8)
        except Exception:
            pos = nx.shell_layout(G)

        # Labels for nodes (truncate text if long & filter non-ascii chars for matplotlib rendering)
        node_labels = {}
        for n, d in G.nodes(data=True):
            txt = str(d.get("text", n))
            clean_txt = txt.encode("ascii", "ignore").decode("ascii")
            if not clean_txt.strip():
                clean_txt = str(n)
            node_labels[n] = (clean_txt[:15] + "..") if len(clean_txt) > 15 else clean_txt

        # Edge labels
        edge_labels = {}
        for u, v, d in G.edges(data=True):
            rel = str(d.get("relation", ""))
            clean_rel = rel.encode("ascii", "ignore").decode("ascii")
            edge_labels[(u, v)] = clean_rel

        nx.draw_networkx_nodes(G, pos, node_color="#6baed6", node_size=1200, alpha=0.9)
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="#969696", width=1.5)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_family="sans-serif")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        # Clean title
        clean_title = title.encode("ascii", "ignore").decode("ascii")
        plt.title(clean_title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        logger.info(f"Rendered example graph image at {output_path}")
