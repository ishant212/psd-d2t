"""Deterministic Serializer with Token-to-Structure Mapping.

Converts UnifiedRecords into deterministic linear sequences for Transformer
models while strictly preserving character-level and token-level mappings
back to underlying nodes, edges, and cells for future PSD computation.
"""

from typing import Dict, Any, List, Optional, Tuple
from src.data.unified_schema import UnifiedRecord


class Serializer:
    """Deterministic serializer for tables and triples with token-to-structure mapping."""

    def __init__(self, table_prefix: bool = True):
        self.table_prefix = table_prefix

    def serialize_totto(self, record: UnifiedRecord) -> Tuple[str, List[Dict[str, Any]]]:
        """Serializes a ToTTo table record into text and character span annotations."""
        meta = record.metadata
        page_title = meta.get("table_page_title", "")
        section_title = meta.get("table_section_title", "")

        spans: List[Dict[str, Any]] = []
        parts: List[str] = []
        curr_pos = 0

        def append_chunk(token: str, span_type: str = "special", element_id: Optional[str] = None, label: Optional[str] = None):
            nonlocal curr_pos
            if not token:
                return
            start = curr_pos
            parts.append(token)
            curr_pos += len(token)
            end = curr_pos
            spans.append({
                "start": start,
                "end": end,
                "type": span_type,
                "id": element_id,
                "label": label or token
            })

        if page_title:
            append_chunk("[PAGE] ", "special")
            append_chunk(f"{page_title} ", "meta_page", "page_title", page_title)

        if section_title:
            append_chunk("[SECTION] ", "special")
            append_chunk(f"{section_title} ", "meta_section", "section_title", section_title)

        # Index nodes by row and col
        nodes_by_pos: Dict[Tuple[int, int], Dict[str, Any]] = {}
        headers: Dict[int, Dict[str, Any]] = {}
        for n in record.nodes:
            attrs = n.get("attributes", {})
            r = attrs.get("row", 0)
            c = attrs.get("col", 0)
            nodes_by_pos[(r, c)] = n
            if attrs.get("is_header", False):
                headers[c] = n

        num_rows = meta.get("num_rows", 0)
        num_cols = meta.get("num_cols", 0)

        # Emit headers if present
        if headers:
            append_chunk("[HEADER] ", "special")
            for c in range(num_cols):
                h_node = headers.get(c)
                if h_node:
                    append_chunk(f"{h_node['label']} ", "node", h_node["id"], h_node["label"])
                if c < num_cols - 1:
                    append_chunk("| ", "special")

        # Emit rows
        for r in range(num_rows):
            # Check if this row has highlighted cells or if we emit all cells
            row_has_highlight = any(
                nodes_by_pos.get((r, c), {}).get("attributes", {}).get("is_highlighted", False)
                for c in range(num_cols)
            )
            # Tag row
            append_chunk(f"[ROW {r}] ", "special")
            for c in range(num_cols):
                node = nodes_by_pos.get((r, c))
                if not node:
                    continue
                is_hl = node.get("attributes", {}).get("is_highlighted", False)
                is_head = node.get("attributes", {}).get("is_header", False)
                
                tag = "[HCELL] " if is_hl else ("[HEADER] " if is_head else "[CELL] ")
                append_chunk(tag, "special")
                append_chunk(f"{node['label']} ", "node", node["id"], node["label"])
                if c < num_cols - 1:
                    append_chunk("| ", "special")

        text = "".join(parts).strip()
        return text, spans

    def serialize_triples(self, record: UnifiedRecord) -> Tuple[str, List[Dict[str, Any]]]:
        """Serializes a DART or WebNLG tripleset into text and character span annotations."""
        spans: List[Dict[str, Any]] = []
        parts: List[str] = []
        curr_pos = 0

        def append_chunk(token: str, span_type: str = "special", element_id: Optional[str] = None, label: Optional[str] = None):
            nonlocal curr_pos
            if not token:
                return
            start = curr_pos
            parts.append(token)
            curr_pos += len(token)
            end = curr_pos
            spans.append({
                "start": start,
                "end": end,
                "type": span_type,
                "id": element_id,
                "label": label or token
            })

        # Process edges (triples) deterministically
        for idx, edge in enumerate(record.edges):
            s_id = edge["source"]
            t_id = edge["target"]
            rel = edge["relation"]
            
            # Find subject and target labels
            s_label = next((n["label"] for n in record.nodes if n["id"] == s_id), s_id)
            o_label = next((n["label"] for n in record.nodes if n["id"] == t_id), t_id)

            append_chunk("[TRIPLE] ", "special")
            append_chunk("[SUBJ] ", "special")
            append_chunk(f"{s_label} ", "node", s_id, s_label)
            append_chunk("[REL] ", "special")
            append_chunk(f"{rel} ", "edge", f"edge_{idx}", rel)
            append_chunk("[OBJ] ", "special")
            append_chunk(f"{o_label} ", "node", t_id, o_label)

        text = "".join(parts).strip()
        return text, spans

    def serialize_record(self, record: UnifiedRecord) -> str:
        """Returns just the serialized text string."""
        text, _ = self.serialize_with_spans(record)
        return text

    serialize = serialize_record

    def serialize_with_spans(self, record: UnifiedRecord) -> Tuple[str, List[Dict[str, Any]]]:
        """Serializes record and returns text and character spans."""
        if record.input_type == "table":
            return self.serialize_totto(record)
        else:
            return self.serialize_triples(record)

    def serialize_with_mapping(self, record: UnifiedRecord, tokenizer: Any = None) -> Dict[str, Any]:
        """Serializes record and produces complete token-to-node and token-to-edge mappings."""
        text, spans = self.serialize_with_spans(record)
        
        result: Dict[str, Any] = {
            "record_id": record.id,
            "dataset": record.dataset,
            "serialized_text": text,
            "char_spans": spans,
            "target_text": record.target_text,
            "reference_texts": record.reference_texts,
            "split": record.split
        }

        if tokenizer is not None:
            # Tokenize with offsets
            encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
            token_ids = encoding["input_ids"]
            offset_mapping = encoding.get("offset_mapping", [])

            token_to_node: List[Optional[str]] = []
            token_to_edge: List[Optional[str]] = []

            for start, end in offset_mapping:
                if start == end:  # special token like </s>
                    token_to_node.append(None)
                    token_to_edge.append(None)
                    continue

                matched_node: Optional[str] = None
                matched_edge: Optional[str] = None

                for sp in spans:
                    if sp["type"] == "node" and not (end <= sp["start"] or start >= sp["end"]):
                        matched_node = sp["id"]
                        break
                    elif sp["type"] == "edge" and not (end <= sp["start"] or start >= sp["end"]):
                        matched_edge = sp["id"]
                        break

                token_to_node.append(matched_node)
                token_to_edge.append(matched_edge)

            result["input_ids"] = token_ids
            result["token_to_node_map"] = token_to_node
            result["token_to_edge_map"] = token_to_edge

        return result
