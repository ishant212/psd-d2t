"""Unit tests for Unified Schema and Normalization."""

import pytest
from src.data.unified_schema import UnifiedRecord, Node, Edge
from src.data.normalize import normalize_record


def test_unified_record_roundtrip():
    node = Node(id="n1", label="Entity A", type="entity", attributes={"key": "val"})
    edge = Edge(source="n1", target="n2", relation="born_in", type="triple")
    
    rec = UnifiedRecord(
        dataset="test_ds",
        id="rec_1",
        input_type="triples",
        structured_input=[["Entity A", "born_in", "Entity B"]],
        nodes=[node.to_dict()],
        edges=[edge.to_dict()],
        highlighted_elements=["n1"],
        target_text="Entity A was born in Entity B.",
        reference_texts=["Entity A was born in Entity B."],
        split="train",
        metadata={"category": "test"}
    )
    
    d = rec.to_dict()
    assert d["dataset"] == "test_ds"
    assert len(d["nodes"]) == 1
    assert len(d["edges"]) == 1
    
    rec2 = UnifiedRecord.from_dict(d)
    assert rec2.id == rec.id
    assert rec2.nodes[0]["label"] == "Entity A"


def test_normalize_totto():
    sample_raw = {
        "id": "totto_100",
        "dataset": "totto",
        "split": "validation",
        "table": [
            [{"value": "Name", "is_header": True}, {"value": "Age", "is_header": True}],
            [{"value": "Alice", "is_header": False}, {"value": "30", "is_header": False}]
        ],
        "highlighted_cells": [[1, 0]],
        "table_page_title": "Test Page",
        "table_section_title": "Test Section",
        "target_text": "Alice is 30 years old.",
        "reference_texts": ["Alice is 30 years old."]
    }
    
    norm = normalize_record(sample_raw, "totto")
    assert norm.dataset == "totto"
    assert norm.input_type == "table"
    assert len(norm.nodes) == 4
    assert len(norm.highlighted_elements) == 1
    assert norm.highlighted_elements[0] == "r1_c0"
    assert norm.nodes[0]["label"] == "Name"
    assert norm.nodes[2]["label"] == "Alice"


def test_normalize_dart():
    sample_raw = {
        "id": "dart_1",
        "dataset": "dart",
        "split": "validation",
        "tripleset": [["Apollo 11", "crew", "Neil Armstrong"], ["Apollo 11", "destination", "Moon"]],
        "target_text": "Apollo 11 flew to the Moon with Neil Armstrong.",
        "reference_texts": ["Apollo 11 flew to the Moon with Neil Armstrong."],
        "source": "WikiTableQuestions"
    }
    
    norm = normalize_record(sample_raw, "dart")
    assert norm.dataset == "dart"
    assert norm.input_type == "triples"
    assert len(norm.nodes) == 3  # Apollo 11, Neil Armstrong, Moon
    assert len(norm.edges) == 2
    assert norm.edges[0]["relation"] == "crew"
