"""
Cleaning & Validation Module for ToTTo, WebNLG, and DART datasets.
"""
import os
import re
import csv
import json
import argparse
import unicodedata
import logging
from typing import Any, Dict, List, Set, Tuple, Optional
import networkx as nx
import yaml

from src.loaders.loader_factory import LoaderFactory
from src.loaders.base_loader import RawRecord
from src.cleaning.schema_validator import SchemaValidator

logger = logging.getLogger(__name__)

# Basic synonym normalization map for relations
COMMON_RELATION_SYNONYMS = {
    "leadername": "leader",
    "leader_name": "leader",
    "birthplace": "place_of_birth",
    "deathplace": "place_of_death",
    "location": "place_of_birth", # context dependent, but clean
    "operatingorganisation": "operator",
    "operating_organisation": "operator",
}

class DatasetCleaner:
    """Executes Stage A Cleaning & Validation on dataset records."""

    def __init__(self, dataset_name: str, config: Dict[str, Any]):
        self.dataset_name = dataset_name.lower()
        self.config = config
        self.report_rows: List[Dict[str, str]] = []
        self.seen_signatures: Set[str] = set()

    def log_action(self, record_id: str, action: str, reason: str):
        """Logs an action (DROP, FIX, FLAG) to the report list."""
        self.report_rows.append({
            "dataset": self.dataset_name,
            "record_id": str(record_id),
            "action": action,
            "reason": reason
        })

    def normalize_text(self, text: str) -> str:
        """Fixes unicode, punctuation, quotes, control characters, and whitespace."""
        if not text:
            return ""
        # Unicode normalization
        text = unicodedata.normalize("NFC", text)
        # Strip zero-width space and invisible control chars
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
        # Quotation mark standardization
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        # Whitespace collapsing
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_relation(self, rel: str) -> str:
        """Normalizes predicate/relation strings."""
        rel_clean = rel.strip().replace(" ", "_").lower()
        return COMMON_RELATION_SYNONYMS.get(rel_clean, rel.strip())

    def check_topology_flags(self, record: RawRecord):
        """Analyzes graph topology flags (disconnected components, isolated nodes, cycles)."""
        G = nx.DiGraph()
        
        if record.dataset in ("webnlg", "dart"):
            triples = record.raw_data.get("triples", record.raw_data.get("tripleset", []))
            for s, r, o in triples:
                s_str, o_str = str(s).strip(), str(o).strip()
                G.add_edge(s_str, o_str, relation=r)
        elif record.dataset == "totto":
            table = record.raw_data.get("table", [])
            highlighted = record.raw_data.get("highlighted_cells", [])
            # Nodes are highlighted cells or grid cells
            cell_nodes = []
            for r_idx, c_idx in highlighted:
                node_id = f"r{r_idx}_c{c_idx}"
                G.add_node(node_id)
                cell_nodes.append((r_idx, c_idx, node_id))
            # Edges between adjacent highlighted cells
            for i in range(len(cell_nodes)):
                for j in range(i + 1, len(cell_nodes)):
                    r1, c1, id1 = cell_nodes[i]
                    r2, c2, id2 = cell_nodes[j]
                    if r1 == r2 or c1 == c2: # adjacent row or col
                        G.add_edge(id1, id2)

        if G.number_of_nodes() > 0:
            # Check isolated nodes
            isolated = list(nx.isolates(G))
            if isolated:
                self.log_action(record.record_id, "FLAG", f"ISOLATED_NODES_COUNT_{len(isolated)}")
            
            # Check disconnected components (undirected view)
            num_components = nx.number_weakly_connected_components(G)
            if num_components > 1:
                self.log_action(record.record_id, "FLAG", f"DISCONNECTED_COMPONENTS_COUNT_{num_components}")
            
            # Check cycles
            if not nx.is_directed_acyclic_graph(G):
                self.log_action(record.record_id, "FLAG", "CYCLIC_STRUCTURE_DETECTED")

    def clean_record(self, record: RawRecord) -> Optional[RawRecord]:
        """Cleans a single RawRecord. Returns cleaned record or None if dropped."""
        # 1. Schema Validation
        is_valid, errors = SchemaValidator.validate(record)
        if not is_valid:
            for err in errors:
                self.log_action(record.record_id, "DROP", f"SCHEMA_VALIDATION_FAILED_{err}")
            return None

        # 2. Text Normalization
        cleaned_refs = []
        for ref in record.references:
            norm_ref = self.normalize_text(ref)
            if norm_ref:
                cleaned_refs.append(norm_ref)
        
        if not cleaned_refs:
            self.log_action(record.record_id, "DROP", "EMPTY_AFTER_TEXT_NORMALIZATION")
            return None
        
        if len(cleaned_refs) != len(record.references):
            self.log_action(record.record_id, "FIX", "NORMALIZED_OR_FILTERED_REF_TEXTS")
        record.references = cleaned_refs

        # 3. Native Data Cleaning (triples / table cells)
        if record.dataset in ("webnlg", "dart"):
            triples_key = "triples" if record.dataset == "webnlg" else "tripleset"
            raw_triples = record.raw_data.get(triples_key, [])
            cleaned_triples = []
            for s, p, o in raw_triples:
                s_norm = self.normalize_text(str(s))
                p_norm = self.normalize_relation(str(p))
                o_norm = self.normalize_text(str(o))
                if s_norm and p_norm and o_norm:
                    cleaned_triples.append([s_norm, p_norm, o_norm])
            
            if not cleaned_triples:
                self.log_action(record.record_id, "DROP", "EMPTY_TRIPLESET_AFTER_CLEANING")
                return None
            
            if len(cleaned_triples) != len(raw_triples):
                self.log_action(record.record_id, "FIX", "CLEANED_TRIPLES")
            record.raw_data[triples_key] = cleaned_triples

        elif record.dataset == "totto":
            # Normalize cell texts
            table = record.raw_data.get("table", [])
            for r_idx, row in enumerate(table):
                for c_idx, cell in enumerate(row):
                    if isinstance(cell, dict) and "value" in cell:
                        cell["value"] = self.normalize_text(cell["value"])

        # 4. Deduplication Check
        # Signature based on reference texts and core content
        content_str = json.dumps(record.raw_data, sort_keys=True) + "||" + "||".join(sorted(record.references))
        if content_str in self.seen_signatures:
            self.log_action(record.record_id, "DROP", "DUPLICATE_RECORD_DETECTED")
            return None
        self.seen_signatures.add(content_str)

        # 5. Check Topology Flags
        self.check_topology_flags(record)

        return record

    def run(self, input_path: str, output_file: str):
        """Runs cleaning over dataset input path and writes outputs."""
        loader = LoaderFactory.get_loader(self.dataset_name)
        records = loader.load(input_path)
        logger.info(f"Loaded {len(records)} raw records for dataset '{self.dataset_name}' from {input_path}")

        cleaned_records: List[RawRecord] = []
        for rec in records:
            cleaned = self.clean_record(rec)
            if cleaned is not None:
                cleaned_records.append(cleaned)

        # Save cleaned JSONL
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for rec in cleaned_records:
                obj = {
                    "record_id": rec.record_id,
                    "dataset": rec.dataset,
                    "raw_data": rec.raw_data,
                    "references": rec.references,
                    "metadata": rec.metadata
                }
                f.write(json.dumps(obj) + "\n")
        logger.info(f"Wrote {len(cleaned_records)} cleaned records to {output_file}")

        return self.report_rows

def write_cleaning_report(all_rows: List[Dict[str, str]], report_path: str):
    """Appends or creates cleaning_report.csv."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    file_exists = os.path.exists(report_path)
    with open(report_path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "record_id", "action", "reason"])
        if not file_exists:
            writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    logger.info(f"Updated cleaning report at {report_path}")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Stage A - Dataset Cleaning & Validation")
    parser.add_argument("--dataset", required=True, choices=["totto", "webnlg", "dart"])
    parser.add_argument("--config", default="config/cleaning.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    input_path = cfg["input_paths"][args.dataset]
    output_dir = cfg["output_dir"]
    output_file = os.path.join(output_dir, f"{args.dataset}.jsonl")
    report_path = cfg["report_path"]

    cleaner = DatasetCleaner(args.dataset, cfg.get("settings", {}))
    report_rows = cleaner.run(input_path, output_file)
    write_cleaning_report(report_rows, report_path)

if __name__ == "__main__":
    main()
