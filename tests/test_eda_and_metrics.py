"""Unit tests for EDA and Evaluation Metrics."""

import pytest
from src.data.unified_schema import UnifiedRecord
from src.eda.dataset_statistics import DatasetStatisticsAnalyzer
from src.eda.length_analysis import LengthAnalyzer
from src.eda.topology_analysis import TopologyAnalyzer
from src.baseline.evaluate_metrics import MetricEvaluator


def test_dataset_statistics_analyzer():
    analyzer = DatasetStatisticsAnalyzer()
    records = [
        UnifiedRecord(
            dataset="test",
            id="1",
            input_type="triples",
            structured_input=[],
            nodes=[{"id": "n1", "label": "Node1"}],
            edges=[{"source": "n1", "target": "n2", "relation": "rel1"}],
            target_text="Target one",
            split="train"
        )
    ]
    res = analyzer.analyze_dataset(records, "test")
    assert res["total_examples"] == 1
    assert res["train_count"] == 1
    assert res["num_unique_entities"] == 1
    assert res["num_unique_relations"] == 1


def test_length_analyzer():
    analyzer = LengthAnalyzer()
    records = [
        UnifiedRecord(
            dataset="test",
            id="1",
            input_type="triples",
            structured_input=[],
            nodes=[],
            edges=[],
            target_text="One two three four five",
            split="train"
        )
    ]
    res = analyzer.analyze(records, "test")
    assert res["total_analyzed"] == 1
    assert res["target_min"] == 5
    assert res["target_max"] == 5


def test_metric_evaluator():
    evaluator = MetricEvaluator()
    predictions = ["The dog sat on the mat."]
    references = [["The dog sat on the mat."]]
    metrics = evaluator.evaluate_batch(predictions, references)
    assert metrics["bleu"] > 90.0
    assert metrics["rouge1"] > 90.0
    assert metrics["rougeL"] > 90.0
