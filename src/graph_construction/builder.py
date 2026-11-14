"""
Graph Builder Module for Stage D: NetworkX DiGraph Construction & Serialization.
"""
import os
import csv
import json
import pickle
import argparse
import logging
from typing import Any, Dict, List, Tuple

import yaml
import networkx as nx

from src.graph_construction.visualizer import GraphVisualizer

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Constructs, validates, and serializes NetworkX DiGraphs from normalized records."""

    def __init__(self, dataset_name: str, config: Dict[str, Any]):
        self.dataset_name = dataset_name.lower()
        self.config = config
        self.output_dir = os.path.join(config.get("output_graphs_dir", "data/graphs"), self.dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.plots_dir = os.path.join(config.get("output_plots_dir", "reports/plots"), self.dataset_name, "example_graphs")
        os.makedirs(self.plots_dir, exist_ok=True)

    def build_graph(self, norm_dict: Dict[str, Any]) -> nx.DiGraph:
        """Constructs a networkx.DiGraph from a normalized record dictionary."""
        G = nx.DiGraph(id=norm_dict["id"], dataset=norm_dict["dataset"], source_type=norm_dict["source_type"])

        # Add Nodes
        for node in norm_dict.get("nodes", []):
            n_id = node["node_id"]
            G.add_node(n_id, text=node.get("text", ""), **node.get("attributes", {}))

        # Add Edges
        for edge in norm_dict.get("edges", []):
            src = edge["source"]
            tgt = edge["target"]
            rel = edge.get("relation", "")
            attrs = edge.get("attributes", {})
            G.add_edge(src, tgt, relation=rel, **attrs)

        return G

    def validate_graph(self, norm_dict: Dict[str, Any], G: nx.DiGraph) -> Tuple[bool, str]:
        """
        Validates constructed graph:
        1. Node count matching
        2. Unique edge count matching (DiGraph merges parallel edges between same u, v)
        3. No orphaned edge references missing nodes
        """
        expected_nodes = len(norm_dict.get("nodes", []))
        norm_edges = norm_dict.get("edges", [])
        unique_expected_edges = len(set((e["source"], e["target"]) for e in norm_edges))

        if G.number_of_nodes() != expected_nodes:
            return False, f"Node count mismatch: expected {expected_nodes}, got {G.number_of_nodes()}"

        if G.number_of_edges() != unique_expected_edges:
            return False, f"Edge count mismatch: expected {unique_expected_edges}, got {G.number_of_edges()}"

        # Check node set contains all edge endpoints
        node_set = set(G.nodes())
        for u, v in G.edges():
            if u not in node_set or v not in node_set:
                return False, f"Orphan edge detected between missing nodes ({u}, {v})"

        return True, "OK"

    def serialize_graph(self, G: nx.DiGraph, record_id: str) -> str:
        """Serializes DiGraph to disk as a .gpickle file."""
        clean_id = str(record_id).replace("/", "_").replace("\\", "_")
        gpickle_path = os.path.join(self.output_dir, f"{clean_id}.gpickle")
        with open(gpickle_path, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        return gpickle_path

    def run(self, normalized_file: str) -> Dict[str, Any]:
        """Runs graph construction pipeline over normalized JSONL file."""
        records: List[Dict[str, Any]] = []
        with open(normalized_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        logger.info(f"Building graphs for {len(records)} normalized records in dataset '{self.dataset_name}'")

        successful = 0
        failed = 0
        node_counts: List[int] = []
        edge_counts: List[int] = []
        constructed_graphs: List[Tuple[nx.DiGraph, str]] = []

        for rec in records:
            rec_id = rec["id"]
            try:
                G = self.build_graph(rec)
                is_valid, err_msg = self.validate_graph(rec, G)
                if is_valid:
                    self.serialize_graph(G, rec_id)
                    successful += 1
                    node_counts.append(G.number_of_nodes())
                    edge_counts.append(G.number_of_edges())
                    constructed_graphs.append((G, rec_id))
                else:
                    logger.warning(f"Graph validation failed for record {rec_id}: {err_msg}")
                    failed += 1
            except Exception as e:
                logger.error(f"Error building graph for record {rec_id}: {e}")
                failed += 1

        # Render Example Graphs (Small, Medium, Large)
        if constructed_graphs:
            # Sort by node count
            constructed_graphs.sort(key=lambda item: item[0].number_of_nodes())
            examples = []
            examples.append(("small_example", constructed_graphs[0]))
            examples.append(("medium_example", constructed_graphs[len(constructed_graphs) // 2]))
            examples.append(("large_example", constructed_graphs[-1]))

            for label, (g_ex, g_id) in examples:
                img_path = os.path.join(self.plots_dir, f"{label}_{g_ex.number_of_nodes()}nodes.png")
                GraphVisualizer.draw_graph(g_ex, f"{self.dataset_name.upper()} Graph — {label} (ID: {g_id})", img_path)

        stats = {
            "dataset": self.dataset_name,
            "total_records": len(records),
            "successful_graphs": successful,
            "failed_graphs": failed,
            "avg_nodes": float(sum(node_counts) / len(node_counts)) if node_counts else 0.0,
            "avg_edges": float(sum(edge_counts) / len(edge_counts)) if edge_counts else 0.0,
        }
        return stats

def write_graph_report(stats: Dict[str, Any], report_path: str):
    """Writes or appends to reports/graph_construction_report.csv."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    file_exists = os.path.exists(report_path)
    with open(report_path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "total_records", "successful_graphs", "failed_graphs", "avg_nodes", "avg_edges"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats)
    logger.info(f"Updated graph construction report at {report_path}")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Stage D - NetworkX Graph Construction")
    parser.add_argument("--dataset", required=True, choices=["totto", "webnlg", "dart"])
    parser.add_argument("--config", default="config/graph_construction.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    normalized_file = os.path.join(cfg.get("input_dir", "data/normalized"), f"{args.dataset}.jsonl")
    report_path = cfg.get("output_report_path", "reports/graph_construction_report.csv")

    builder = GraphBuilder(args.dataset, cfg)
    stats = builder.run(normalized_file)
    write_graph_report(stats, report_path)

if __name__ == "__main__":
    main()
