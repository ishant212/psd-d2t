"""Unit tests for dataset loaders (ToTTo, DART, WebNLG)."""

import pytest
from src.data.load_totto import ToTToLoader
from src.data.load_dart import DARTLoader
from src.data.load_webnlg import WebNLGLoader


def test_totto_loader_sample():
    loader = ToTToLoader()
    records = loader.load_split(split="validation", max_examples=5)
    assert len(records) == 5
    rec = records[0]
    assert rec["dataset"] == "totto"
    assert "table" in rec
    assert "highlighted_cells" in rec
    assert "table_page_title" in rec
    assert "sentence_annotations" in rec
    assert isinstance(rec["table"], list)


def test_dart_loader_sample():
    loader = DARTLoader()
    records = loader.load_split(split="validation", max_examples=5)
    assert len(records) == 5
    rec = records[0]
    assert rec["dataset"] == "dart"
    assert "tripleset" in rec
    assert "annotations" in rec
    assert isinstance(rec["tripleset"], list)
    assert len(rec["tripleset"]) > 0


def test_webnlg_loader_sample():
    loader = WebNLGLoader()
    records = loader.load_split(split="validation", max_examples=5)
    assert len(records) == 5
    rec = records[0]
    assert rec["dataset"] == "webnlg"
    assert "tripleset" in rec
    assert "reference_texts" in rec
    assert isinstance(rec["tripleset"], list)
