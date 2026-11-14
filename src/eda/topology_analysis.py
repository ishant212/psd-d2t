"""Graph Topology and Structural Analysis for Phase 1 EDA.

Analyzes structural characteristics of tables and graphs across ToTTo, DART, and WebNLG:
- Table: rows, columns, cells, highlighted cells, aspect ratio
- Graph: nodes, edges, degree distribution, density, connected components, shortest paths.
"""

from typing import List, Dict, Any
import numpy as np
import networkx as nx
from src.data.unified_schema import UnifiedRecord


class TopologyAnalyzer:
    """Computes topological and structural metrics for tabular and graph records."""

    def __init__(self):
        pass

    def analyze_dataset(self, records: List[UnifiedRecord], dataset_name: str) -> Dict[str, Any]:
        """Computes structural metrics across records."""
        is_table = any(r.input_type == "table" for r in records)

        if is_table:
            return self._analyze_tables(records, dataset_name)
        else:
            return self._analyze_graphs(records, dataset_name)

    def _analyze_tables(self, records: List[UnifiedRecord], dataset_name: str) -> Dict[str, Any]:
        rows_list: List[int] = []
        cols_list: List[int] = []
        total_cells_list: List[int] = []
        hl_cells_list: List[int] = []

        for r in records:
            meta = r.metadata
            num_rows = meta.get("num_rows", 0)
            num_cols = meta.get("num_cols", 0)
            rows_list.append(num_rows)
            cols_list.append(num_cols)
            total_cells_list.append(len(r.nodes))
            hl_cells_list.append(len(r.highlighted_elements))

        return {
            "dataset": dataset_name,
            "type": "table",
            "total_examples": len(records),
            "avg_rows": float(np.mean(rows_list)) if rows_list else 0.0,
            "max_rows": int(np.max(rows_list)) if rows_list else 0,
            "avg_cols": float(np.mean(cols_list)) if cols_list else 0.0,
            "max_cols": int(np.max(cols_list)) if cols_list else 0,
            "avg_cells": float(np.mean(total_cells_list)) if total_cells_list else 0.0,
            "max_cells": int(np.max(total_cells_list)) if total_cells_list else 0,
            "avg_highlighted_cells": float(np.mean(hl_cells_list)) if hl_cells_list else 0.0,
            "max_highlighted_cells": int(np.max(hl_cells_list)) if hl_cells_list else 0,
            "pct_cells_highlighted": float(np.sum(hl_cells_list) / max(1, np.sum(total_cells_list)) * 100.0)
        }

    def _analyze_graphs(self, records: List[UnifiedRecord], dataset_name: str) -> Dict[str, Any]:
        nodes_list: List[int] = []
        edges_list: List[int] = []
        avg_degrees: List[float] = []
        max_degrees: List[int] = []
        densities: List[float] = []
        num_components: List[int] = []
        avg_path_lengths: List[float] = []

        for r in records:
            G = nx.DiGraph()
            for n in r.nodes:
                G.add_node(n["id"])
            for e in r.edges:
                G.add_edge(e["source"], e["target"])

            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            nodes_list.append(n_nodes)
            edges_list.append(n_edges)

            if n_nodes > 0:
                degrees = [d for _, d in G.degree()]
                avg_degrees.append(float(np.mean(degrees)))
                max_degrees.append(int(np.max(degrees)))
                densities.append(nx.density(G))
                
                # Weakly connected components
                wcc = list(nx.weakly_connected_components(G))
                num_components.append(len(wcc))

                # Shortest path statistics on largest component (undirected projection)
                G_undir = G.to_undirected()
                largest_cc = max(nx.connected_components(G_undir), key=len)
                subgraph = G_undir.subgraph(largest_cc)
                if len(subgraph) > 1:
                    try:
                        avg_path = nx.average_shortest_path_length(subgraph)
                        avg_path_lengths.append(avg_path)
                    except Exception:
                        pass
            else:
                avg_degrees.append(0.0)
                max_degrees.append(0)
                densities.append(0.0)
                num_components.append(0)

        return {
            "dataset": dataset_name,
            "type": "graph",
            "total_examples": len(records),
            "avg_nodes": float(np.mean(nodes_list)) if nodes_list else 0.0,
            "max_nodes": int(np.max(nodes_list)) if nodes_list else 0,
            "avg_edges": float(np.mean(edges_list)) if edges_list else 0.0,
            "max_edges": int(np.max(edges_list)) if edges_list else 0,
            "avg_degree": float(np.mean(avg_degrees)) if avg_degrees else 0.0,
            "max_degree": int(np.max(max_degrees)) if max_degrees else 0,
            "avg_density": float(np.mean(densities)) if densities else 0.0,
            "avg_components": float(np.mean(num_components)) if num_components else 0.0,
            "avg_shortest_path": float(np.mean(avg_path_lengths)) if avg_path_lengths else 0.0
        }
