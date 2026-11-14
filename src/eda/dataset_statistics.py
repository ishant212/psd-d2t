"""Dataset Statistics Analyzer for Phase 1 EDA.

Calculates counts of examples, splits, entities, relations, tables/triplesets,
and reference distribution across ToTTo, DART, and WebNLG.
"""

from typing import List, Dict, Any, Set
import numpy as np
from src.data.unified_schema import UnifiedRecord


class DatasetStatisticsAnalyzer:
    """Computes high-level dataset statistics across unified records."""

    def __init__(self):
        pass

    def analyze_dataset(self, records: List[UnifiedRecord], dataset_name: str) -> Dict[str, Any]:
        """Computes comprehensive summary statistics for a dataset."""
        total_records = len(records)
        if total_records == 0:
            return {"dataset": dataset_name, "total_examples": 0}

        split_counts: Dict[str, int] = {}
        unique_entities: Set[str] = set()
        unique_relations: Set[str] = set()
        ref_counts: List[int] = []
        node_counts: List[int] = []
        edge_counts: List[int] = []
        highlighted_counts: List[int] = []

        for r in records:
            split_counts[r.split] = split_counts.get(r.split, 0) + 1
            ref_counts.append(len(r.reference_texts) if r.reference_texts else (1 if r.target_text else 0))
            node_counts.append(len(r.nodes))
            edge_counts.append(len(r.edges))
            highlighted_counts.append(len(r.highlighted_elements))

            for n in r.nodes:
                unique_entities.add(n.get("label", "").lower())

            for e in r.edges:
                unique_relations.add(e.get("relation", "").lower())

        return {
            "dataset": dataset_name,
            "total_examples": total_records,
            "train_count": split_counts.get("train", 0),
            "validation_count": split_counts.get("validation", 0) + split_counts.get("dev", 0),
            "test_count": split_counts.get("test", 0),
            "num_unique_entities": len(unique_entities),
            "num_unique_relations": len(unique_relations),
            "avg_references_per_example": float(np.mean(ref_counts)) if ref_counts else 0.0,
            "max_references_per_example": int(np.max(ref_counts)) if ref_counts else 0,
            "avg_nodes_per_example": float(np.mean(node_counts)) if node_counts else 0.0,
            "avg_edges_per_example": float(np.mean(edge_counts)) if edge_counts else 0.0,
            "avg_highlighted_elements": float(np.mean(highlighted_counts)) if highlighted_counts else 0.0
        }
