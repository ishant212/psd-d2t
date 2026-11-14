"""Graph distance calculation module for Phase 2 PSD.

Computes topological distance matrices for graph records (DART, WebNLG)
via all-pairs shortest paths, and for tabular records (ToTTo) via
coordinate Manhattan distance.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import networkx as nx
from src.data.unified_schema import UnifiedRecord


class GraphDistanceCalculator:
    """Calculates pairwise topological distances between nodes in structured records."""

    def __init__(self, default_d_max: float = 10.0):
        self.default_d_max = default_d_max

    def compute_distance_matrix(self, record: UnifiedRecord) -> Tuple[np.ndarray, Dict[str, int]]:
        """Computes topological distance matrix D and node_id -> matrix index map.

        Args:
            record: UnifiedRecord containing nodes, edges, input_type.

        Returns:
            dist_matrix: np.ndarray of shape [num_nodes, num_nodes] where dist_matrix[i, j] is d(u_i, u_j).
            node_to_idx: Dict mapping node_id to row/column index in dist_matrix.
        """
        if record.input_type == "table":
            return self._compute_table_distance(record)
        else:
            return self._compute_graph_distance(record)

    def _compute_table_distance(self, record: UnifiedRecord) -> Tuple[np.ndarray, Dict[str, int]]:
        """Computes Manhattan distance between table cells: d(u, v) = |r_u - r_v| + |c_u - c_v|."""
        nodes = record.nodes
        num_nodes = len(nodes)

        if num_nodes == 0:
            return np.zeros((0, 0), dtype=np.float32), {}

        node_to_idx: Dict[str, int] = {}
        coords: Dict[str, Tuple[int, int]] = {}

        for idx, node in enumerate(nodes):
            nid = node["id"]
            node_to_idx[nid] = idx
            attrs = node.get("attributes", {})
            r = attrs.get("row", 0)
            c = attrs.get("col", 0)
            coords[nid] = (r, c)

        dist_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        for i, node_i in enumerate(nodes):
            r_i, c_i = coords[node_i["id"]]
            for j, node_j in enumerate(nodes):
                if i == j:
                    dist_matrix[i, j] = 0.0
                else:
                    r_j, c_j = coords[node_j["id"]]
                    # Manhattan distance on table grid
                    dist_matrix[i, j] = float(abs(r_i - r_j) + abs(c_i - c_j))

        return dist_matrix, node_to_idx

    def _compute_graph_distance(self, record: UnifiedRecord) -> Tuple[np.ndarray, Dict[str, int]]:
        """Computes shortest path distances on NetworkX graph for DART/WebNLG."""
        nodes = record.nodes
        edges = record.edges
        num_nodes = len(nodes)

        if num_nodes == 0:
            return np.zeros((0, 0), dtype=np.float32), {}

        node_to_idx: Dict[str, int] = {node["id"]: idx for idx, node in enumerate(nodes)}

        # Construct undirected graph for structural proximity
        G = nx.Graph()
        for node in nodes:
            G.add_node(node["id"])
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], relation=edge.get("relation", ""))

        # Compute all-pairs shortest path lengths
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))

        # Determine maximum observed finite distance across all components
        max_observed_dist = 1.0
        for src, targets in path_lengths.items():
            for tgt, dist in targets.items():
                if dist > max_observed_dist:
                    max_observed_dist = float(dist)

        # Finite cutoff for disconnected node pairs: max_observed_dist + 1
        d_cutoff = max_observed_dist + 1.0

        dist_matrix = np.full((num_nodes, num_nodes), d_cutoff, dtype=np.float32)

        for i, node_i in enumerate(nodes):
            nid_i = node_i["id"]
            if nid_i in path_lengths:
                for nid_j, dist in path_lengths[nid_i].items():
                    if nid_j in node_to_idx:
                        j = node_to_idx[nid_j]
                        dist_matrix[i, j] = float(dist)

            # Ensure exact self-distance is 0.0
            dist_matrix[i, i] = 0.0

        return dist_matrix, node_to_idx
