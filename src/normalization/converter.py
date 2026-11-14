"""
Data Converter for Stage C: Structured Data Normalization to Shared Schema.
"""
import os
import json
import argparse
import logging
from typing import Any, Dict, List, Set, Tuple

import yaml
from src.normalization.schema import NormalizedNode, NormalizedEdge, NormalizedRecord

logger = logging.getLogger(__name__)

class DataNormalizer:
    """Converts dataset-specific cleaned records into the unified internal schema."""

    def __init__(self, dataset_name: str, config: Dict[str, Any]):
        self.dataset_name = dataset_name.lower()
        self.config = config

    def convert_totto(self, record_dict: Dict[str, Any]) -> NormalizedRecord:
        """Converts a ToTTo table record into nodes and row/column adjacency edges."""
        rec_id = str(record_dict.get("record_id", "totto_unknown"))
        raw_data = record_dict.get("raw_data", {})
        table = raw_data.get("table", [])
        highlighted_set = set((r, c) for r, c in raw_data.get("highlighted_cells", []))

        nodes: List[NormalizedNode] = []
        edges: List[NormalizedEdge] = []
        node_grid: Dict[Tuple[int, int], str] = {}

        num_rows = len(table)
        for r_idx, row in enumerate(table):
            if isinstance(row, list):
                for c_idx, cell in enumerate(row):
                    cell_val = cell.get("value", "") if isinstance(cell, dict) else str(cell)
                    is_header = cell.get("is_header", False) if isinstance(cell, dict) else False
                    is_highlighted = (r_idx, c_idx) in highlighted_set

                    node_id = f"r{r_idx}_c{c_idx}"
                    nodes.append(NormalizedNode(
                        node_id=node_id,
                        text=cell_val,
                        attributes={
                            "row_idx": r_idx,
                            "col_idx": c_idx,
                            "is_header": is_header,
                            "is_highlighted": is_highlighted
                        }
                    ))
                    node_grid[(r_idx, c_idx)] = node_id

        # Row Adjacency Edges (Left -> Right)
        for (r_idx, c_idx), node_id in node_grid.items():
            if (r_idx, c_idx + 1) in node_grid:
                target_id = node_grid[(r_idx, c_idx + 1)]
                edges.append(NormalizedEdge(
                    source=node_id,
                    target=target_id,
                    relation="row_adjacent",
                    attributes={"edge_type": "row_adjacent"}
                ))
            # Column Adjacency Edges (Top -> Bottom)
            if (r_idx + 1, c_idx) in node_grid:
                target_id = node_grid[(r_idx + 1, c_idx)]
                edges.append(NormalizedEdge(
                    source=node_id,
                    target=target_id,
                    relation="column_adjacent",
                    attributes={"edge_type": "column_adjacent"}
                ))

        metadata = {
            "table_page_title": raw_data.get("table_page_title", ""),
            "table_section_title": raw_data.get("table_section_title", ""),
            "num_rows": num_rows,
            "num_cols": max((len(r) for r in table if isinstance(r, list)), default=0)
        }

        return NormalizedRecord(
            id=rec_id,
            dataset="totto",
            source_type="table",
            nodes=nodes,
            edges=edges,
            reference_texts=record_dict.get("references", []),
            metadata=metadata
        )

    def convert_triples(self, record_dict: Dict[str, Any], dataset_label: str) -> NormalizedRecord:
        """Converts WebNLG or DART triple-set records into nodes and directed edges."""
        rec_id = str(record_dict.get("record_id", f"{dataset_label}_unknown"))
        raw_data = record_dict.get("raw_data", {})
        triples_key = "triples" if dataset_label == "webnlg" else "tripleset"
        triples = raw_data.get(triples_key, [])

        nodes_dict: Dict[str, NormalizedNode] = {}
        edges: List[NormalizedEdge] = []

        def get_or_create_node(entity_str: str) -> str:
            entity_clean = entity_str.strip()
            # Node ID is entity text
            if entity_clean not in nodes_dict:
                nodes_dict[entity_clean] = NormalizedNode(
                    node_id=entity_clean,
                    text=entity_clean,
                    attributes={}
                )
            return entity_clean

        for s, r, o in triples:
            src_id = get_or_create_node(str(s))
            tgt_id = get_or_create_node(str(o))
            rel_str = str(r).strip()

            edges.append(NormalizedEdge(
                source=src_id,
                target=tgt_id,
                relation=rel_str,
                attributes={"edge_type": "triple"}
            ))

        return NormalizedRecord(
            id=rec_id,
            dataset=dataset_label,
            source_type="graph",
            nodes=list(nodes_dict.values()),
            edges=edges,
            reference_texts=record_dict.get("references", []),
            metadata=record_dict.get("metadata", {})
        )

    def validate_roundtrip(self, record_dict: Dict[str, Any], norm_rec: NormalizedRecord) -> Tuple[bool, str]:
        """
        Validates round-trip consistency: verifies no nodes/edges were dropped vs cleaned source.
        """
        if norm_rec.dataset in ("webnlg", "dart"):
            triples_key = "triples" if norm_rec.dataset == "webnlg" else "tripleset"
            source_triples = record_dict.get("raw_data", {}).get(triples_key, [])
            if len(source_triples) != len(norm_rec.edges):
                return False, f"Edge count mismatch: source has {len(source_triples)}, normalized has {len(norm_rec.edges)}"

        elif norm_rec.dataset == "totto":
            table = record_dict.get("raw_data", {}).get("table", [])
            total_cells = sum(len(r) for r in table if isinstance(r, list))
            if total_cells != len(norm_rec.nodes):
                return False, f"Node count mismatch: source has {total_cells} cells, normalized has {len(norm_rec.nodes)} nodes"

        return True, "OK"

    def convert_record(self, record_dict: Dict[str, Any]) -> NormalizedRecord:
        """Dispatches record conversion based on dataset type."""
        if self.dataset_name == "totto":
            norm = self.convert_totto(record_dict)
        elif self.dataset_name in ("webnlg", "dart"):
            norm = self.convert_triples(record_dict, self.dataset_name)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        if self.config.get("validate_roundtrip", True):
            is_valid, msg = self.validate_roundtrip(record_dict, norm)
            if not is_valid:
                logger.warning(f"Roundtrip check failed for record {norm.id}: {msg}")

        return norm

    def run(self, cleaned_file: str, output_file: str):
        """Converts cleaned JSONL file to normalized JSONL file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        count = 0
        with open(cleaned_file, "r", encoding="utf-8") as fin, \
             open(output_file, "w", encoding="utf-8") as fout:
            for line in fin:
                if line.strip():
                    rec_dict = json.loads(line)
                    norm_rec = self.convert_record(rec_dict)
                    fout.write(json.dumps(norm_rec.to_dict()) + "\n")
                    count += 1
        logger.info(f"Normalized {count} records for dataset '{self.dataset_name}' -> {output_file}")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Stage C - Structured Data Normalization")
    parser.add_argument("--dataset", required=True, choices=["totto", "webnlg", "dart"])
    parser.add_argument("--config", default="config/normalization.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cleaned_file = os.path.join(cfg.get("input_dir", "data/cleaned"), f"{args.dataset}.jsonl")
    output_file = os.path.join(cfg.get("output_dir", "data/normalized"), f"{args.dataset}.jsonl")

    normalizer = DataNormalizer(args.dataset, cfg)
    normalizer.run(cleaned_file, output_file)

if __name__ == "__main__":
    main()
