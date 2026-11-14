"""Unit tests for conservative cleaner."""

import pytest
from src.preprocessing.clean import clean_text, DataCleaner
from src.data.unified_schema import UnifiedRecord


def test_clean_text():
    raw_str = "  Hello \u200bWorld \t\n  with \xa0spaces   "
    cleaned = clean_text(raw_str)
    assert cleaned == "Hello World with spaces"


def test_cleaner_drop_empty():
    cleaner = DataCleaner(drop_empty_inputs=True, drop_empty_targets=False)
    
    empty_rec = UnifiedRecord(
        dataset="test",
        id="empty_1",
        input_type="triples",
        structured_input=[],
        nodes=[],
        edges=[],
        target_text="Some text",
        split="train"
    )
    
    res, reason = cleaner.clean_record(empty_rec)
    assert res is None
    assert reason == "DROP_EMPTY_STRUCTURED_INPUT"


def test_cleaner_valid_record():
    cleaner = DataCleaner(drop_empty_inputs=True)
    valid_rec = UnifiedRecord(
        dataset="test",
        id="v_1",
        input_type="triples",
        structured_input=[["A", "rel", "B"]],
        nodes=[{"id": "n1", "label": "  Node A \u200b "}],
        edges=[{"source": "n1", "target": "n2", "relation": " is_part_of "}],
        target_text="  Target text.  ",
        split="train"
    )
    
    res, reason = cleaner.clean_record(valid_rec)
    assert res is not None
    assert reason == "CLEAN_SUCCESS"
    assert res.target_text == "Target text."
    assert res.nodes[0]["label"] == "Node A"
    assert res.edges[0]["relation"] == "is_part_of"
