"""Unit tests for Serialization and Token-to-Structure Mapping."""

import pytest
from src.data.unified_schema import UnifiedRecord
from src.preprocessing.serialize import Serializer


def test_serialize_triples_mapping():
    serializer = Serializer()
    rec = UnifiedRecord(
        dataset="dart",
        id="d1",
        input_type="triples",
        structured_input=[["Einstein", "born_in", "Ulm"]],
        nodes=[
            {"id": "ent_einstein", "label": "Einstein", "type": "entity"},
            {"id": "ent_ulm", "label": "Ulm", "type": "entity"}
        ],
        edges=[
            {"source": "ent_einstein", "target": "ent_ulm", "relation": "born in", "type": "triple"}
        ],
        target_text="Einstein was born in Ulm."
    )
    
    mapping = serializer.serialize_with_mapping(rec)
    text = mapping["serialized_text"]
    assert "[TRIPLE]" in text
    assert "[SUBJ] Einstein" in text
    assert "[REL] born in" in text
    assert "[OBJ] Ulm" in text
    
    spans = mapping["char_spans"]
    assert len(spans) > 0
    node_spans = [s for s in spans if s["type"] == "node"]
    assert len(node_spans) == 2
    assert node_spans[0]["id"] == "ent_einstein"
    assert node_spans[1]["id"] == "ent_ulm"


def test_serialize_table_mapping():
    serializer = Serializer()
    rec = UnifiedRecord(
        dataset="totto",
        id="t1",
        input_type="table",
        structured_input={},
        nodes=[
            {"id": "r0_c0", "label": "Driver", "attributes": {"row": 0, "col": 0, "is_header": True}},
            {"id": "r1_c0", "label": "Senna", "attributes": {"row": 1, "col": 0, "is_header": False, "is_highlighted": True}}
        ],
        edges=[],
        highlighted_elements=["r1_c0"],
        target_text="Ayrton Senna was a driver.",
        metadata={"table_page_title": "F1 Drivers", "num_rows": 2, "num_cols": 1}
    )
    
    mapping = serializer.serialize_with_mapping(rec)
    text = mapping["serialized_text"]
    assert "[PAGE] F1 Drivers" in text
    assert "[HCELL] Senna" in text
    
    spans = mapping["char_spans"]
    senna_span = next(s for s in spans if s["id"] == "r1_c0")
    assert senna_span["label"] == "Senna"
