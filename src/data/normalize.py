"""Dataset Normalizer.

Converts dataset-specific records (ToTTo, DART, WebNLG) into the common
UnifiedRecord schema, preserving full structural topology (nodes, edges,
highlighted elements, coordinate mappings, and reference texts).
"""

import re
from typing import Dict, Any, List, Set, Tuple
from src.data.unified_schema import UnifiedRecord, Node, Edge


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\-_]+", "_", text.strip())
    return slug[:64] if slug else "empty"


def normalize_totto(raw_record: Dict[str, Any]) -> UnifiedRecord:
    """Converts a raw ToTTo record to UnifiedRecord."""
    table = raw_record.get("table", [])
    highlighted_cells = raw_record.get("highlighted_cells", [])
    highlighted_set = {tuple(rc) for rc in highlighted_cells}
    
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    highlighted_elements: List[str] = []

    header_cols: Dict[int, str] = {}
    num_rows = len(table)
    num_cols = max((len(row) for row in table), default=0)

    for r_idx, row in enumerate(table):
        for c_idx, cell in enumerate(row):
            node_id = f"r{r_idx}_c{c_idx}"
            val = str(cell.get("value", "")).strip()
            is_header = bool(cell.get("is_header", False) or r_idx == 0)
            is_highlighted = (r_idx, c_idx) in highlighted_set

            if is_header and c_idx not in header_cols:
                header_cols[c_idx] = node_id

            node_obj = Node(
                id=node_id,
                label=val,
                type="header" if is_header else "cell",
                attributes={
                    "row": r_idx,
                    "col": c_idx,
                    "is_header": is_header,
                    "is_highlighted": is_highlighted,
                    "column_span": cell.get("column_span", 1),
                    "row_span": cell.get("row_span", 1)
                }
            )
            nodes.append(node_obj.to_dict())
            if is_highlighted:
                highlighted_elements.append(node_id)

            # Horizontal edge to right neighbor
            if c_idx + 1 < len(row):
                edges.append(Edge(
                    source=node_id,
                    target=f"r{r_idx}_c{c_idx+1}",
                    relation="row_next",
                    type="row_adj",
                    attributes={"row": r_idx}
                ).to_dict())

            # Vertical edge to lower neighbor
            if r_idx + 1 < num_rows and c_idx < len(table[r_idx+1]):
                edges.append(Edge(
                    source=node_id,
                    target=f"r{r_idx+1}_c{c_idx}",
                    relation="col_next",
                    type="col_adj",
                    attributes={"col": c_idx}
                ).to_dict())

            # Edge from corresponding column header to data cell
            if not is_header and c_idx in header_cols:
                edges.append(Edge(
                    source=header_cols[c_idx],
                    target=node_id,
                    relation="header_of",
                    type="header_to_cell",
                    attributes={"col": c_idx}
                ).to_dict())

    return UnifiedRecord(
        dataset="totto",
        id=raw_record.get("id", ""),
        input_type="table",
        structured_input={
            "table": table,
            "highlighted_cells": highlighted_cells,
            "table_page_title": raw_record.get("table_page_title", ""),
            "table_section_title": raw_record.get("table_section_title", "")
        },
        nodes=nodes,
        edges=edges,
        highlighted_elements=highlighted_elements,
        target_text=raw_record.get("target_text", ""),
        reference_texts=raw_record.get("reference_texts", []),
        split=raw_record.get("split", "train"),
        metadata={
            "table_page_title": raw_record.get("table_page_title", ""),
            "table_section_title": raw_record.get("table_section_title", ""),
            "num_rows": num_rows,
            "num_cols": num_cols,
            "overlap_subset": raw_record.get("overlap_subset", False)
        }
    )


def normalize_triples_record(raw_record: Dict[str, Any], dataset_name: str) -> UnifiedRecord:
    """Converts a raw DART or WebNLG record to UnifiedRecord."""
    tripleset = raw_record.get("tripleset", [])
    
    nodes_map: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    highlighted_elements: List[str] = []

    def get_or_create_node(entity_label: str, role: str) -> str:
        clean_label = entity_label.strip()
        slug = _slugify(clean_label)
        node_id = f"ent_{slug}"
        if node_id not in nodes_map:
            nodes_map[node_id] = Node(
                id=node_id,
                label=clean_label,
                type="entity",
                attributes={"roles": {role}}
            ).to_dict()
        else:
            nodes_map[node_id]["attributes"]["roles"].add(role)
        return node_id

    for t_idx, triple in enumerate(tripleset):
        if not isinstance(triple, (list, tuple)) or len(triple) < 3:
            continue
        subj, rel, obj = str(triple[0]), str(triple[1]), str(triple[2])
        
        subj_id = get_or_create_node(subj, "subject")
        obj_id = get_or_create_node(obj, "object")

        edges.append(Edge(
            source=subj_id,
            target=obj_id,
            relation=rel.strip(),
            type="triple",
            attributes={"triple_index": t_idx}
        ).to_dict())

    # Convert roles set to list for JSON compatibility
    for node in nodes_map.values():
        if isinstance(node["attributes"].get("roles"), set):
            node["attributes"]["roles"] = sorted(list(node["attributes"]["roles"]))
        highlighted_elements.append(node["id"])

    metadata = {}
    if dataset_name == "webnlg":
        metadata["category"] = raw_record.get("category", "")
        metadata["shape"] = raw_record.get("shape", "")
        metadata["size"] = raw_record.get("size", len(tripleset))
    elif dataset_name == "dart":
        metadata["source"] = raw_record.get("source", "")
        metadata["subtree_was_extended"] = raw_record.get("subtree_was_extended", False)

    return UnifiedRecord(
        dataset=dataset_name,
        id=raw_record.get("id", ""),
        input_type="triples",
        structured_input=tripleset,
        nodes=list(nodes_map.values()),
        edges=edges,
        highlighted_elements=highlighted_elements,
        target_text=raw_record.get("target_text", ""),
        reference_texts=raw_record.get("reference_texts", []),
        split=raw_record.get("split", "train"),
        metadata=metadata
    )


def normalize_record(raw_record: Dict[str, Any], dataset: str) -> UnifiedRecord:
    """Entry point for normalizing any dataset record."""
    d = dataset.lower()
    if d == "totto":
        return normalize_totto(raw_record)
    elif d in ("dart", "webnlg"):
        return normalize_triples_record(raw_record, d)
    else:
        raise ValueError(f"Unknown dataset '{dataset}' for normalization.")
