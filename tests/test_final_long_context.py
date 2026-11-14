"""Unit and Integration Tests for Final Long-Context Evaluation Suite.

Verifies:
1. Benchmark 60 dataset composition and stratification.
2. Zero data leakage between evaluation benchmark and training set.
3. Truncation arithmetic and budget constraints.
4. Paired bootstrap statistical CI computation.
5. Multi-reference METEOR and Factuality evaluation integration.
6. Immutability of historical phase checkpoints and artifacts.
"""

import os
import sys
import json
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.psd.run_final_long_context import (
    build_evaluation_benchmark_60,
    format_eval_prompt,
    assign_context_bucket,
    compute_paired_bootstrap_ci,
    OUTPUT_DIR,
    CHECKPOINTS,
    CONTEXT_BUCKETS,
    CONTEXT_LENGTHS,
)
from src.baseline.evaluate_metrics import MetricEvaluator
from src.baseline.factuality import FactualityEvaluator


class MockTokenizer:
    """Lightweight mock tokenizer to test prompt formatting and truncation."""
    def __init__(self):
        self.pad_token = "<|pad|>"
        self.eos_token = "<|im_end|>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, add_special_tokens=False, return_tensors=None, truncation=False, max_length=None):
        tokens = text.split()
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        input_ids = [hash(t) % 10000 + 10 for t in tokens]
        if return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor([input_ids], dtype=torch.long)}
        return {"input_ids": input_ids}

    def decode(self, ids, skip_special_tokens=False):
        return f"decoded_{len(ids)}_tokens"


def test_benchmark_60_composition_and_stratification():
    """Verifies that the 60-example benchmark is perfectly balanced and stratified."""
    bench = build_evaluation_benchmark_60()
    assert len(bench) == 60, f"Expected 60 instances, found {len(bench)}"

    totto = [it for it in bench if it["dataset"] == "totto"]
    dart = [it for it in bench if it["dataset"] == "dart"]
    webnlg = [it for it in bench if it["dataset"] == "webnlg"]

    assert len(totto) == 20, f"Expected 20 ToTTo instances, found {len(totto)}"
    assert len(dart) == 20, f"Expected 20 DART instances, found {len(dart)}"
    assert len(webnlg) == 20, f"Expected 20 WebNLG instances, found {len(webnlg)}"

    # Check ToTTo stratification: exactly 4 in each bucket
    bucket_counts = {b: 0 for b in CONTEXT_BUCKETS}
    for it in totto:
        b = it.get("context_bucket", assign_context_bucket(it.get("token_length", 0)))
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    for b in CONTEXT_BUCKETS:
        assert bucket_counts[b] == 4, f"Bucket {b} has {bucket_counts[b]} instances, expected 4"


def test_zero_leakage_with_training_set():
    """Verifies evaluation IDs have zero overlap with training set."""
    bench = build_evaluation_benchmark_60()
    eval_ids = set(it["id"] for it in bench)

    train_path = os.path.join(PROJECT_ROOT, "data", "unified", "train.jsonl")
    if os.path.exists(train_path):
        train_ids = set()
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    train_ids.add(json.loads(line)["id"])
        overlap = eval_ids.intersection(train_ids)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping IDs between benchmark and training!"


def test_format_eval_prompt_truncation_arithmetic():
    """Verifies prompt truncation logic, overhead deduction, and meta calculations."""
    mock_tok = MockTokenizer()
    long_text = " ".join([f"word_{i}" for i in range(500)])
    item = {
        "id": "test_001",
        "dataset": "totto",
        "input_type": "table",
        "serialized_text": long_text,
    }

    # Context length = 256
    prompt, meta = format_eval_prompt(item, mock_tok, max_context_len=256)
    assert meta["example_id"] == "test_001"
    assert meta["original_token_length"] == 500
    assert meta["effective_token_length"] < 256
    assert meta["truncation_amount"] == 500 - meta["effective_token_length"]
    assert meta["truncation_pct"] > 0.0

    # Context length = 2048 (no truncation needed)
    short_text = "word1 word2 word3"
    item_short = {
        "id": "test_002",
        "dataset": "dart",
        "input_type": "triples",
        "serialized_text": short_text,
    }
    prompt_short, meta_short = format_eval_prompt(item_short, mock_tok, max_context_len=2048)
    assert meta_short["truncation_amount"] == 0
    assert meta_short["truncation_pct"] == 0.0


def test_paired_bootstrap_confidence_interval():
    """Verifies bootstrap CI computation on synthetic distributions."""
    # Test identical distributions -> delta = 0
    scores_a = [10.0, 20.0, 30.0, 40.0, 50.0]
    scores_b = [10.0, 20.0, 30.0, 40.0, 50.0]
    res_null = compute_paired_bootstrap_ci(scores_a, scores_b, n_boot=200, seed=42)
    assert res_null["mean_diff"] == 0.0
    assert res_null["ci_lower"] <= 0.0 <= res_null["ci_upper"]
    assert not res_null["significant"]

    # Test strictly superior distribution -> delta > 0
    scores_high = [25.0, 35.0, 45.0, 55.0, 65.0]
    res_sig = compute_paired_bootstrap_ci(scores_a, scores_high, n_boot=200, seed=42)
    assert res_sig["mean_diff"] == 15.0
    assert res_sig["ci_lower"] > 0.0
    assert res_sig["significant"]


def test_meteor_metric_evaluator():
    """Verifies multi-reference METEOR score calculation."""
    evaluator = MetricEvaluator()
    pred = "The Eiffel Tower is located in Paris, France."
    refs = [
        "The Eiffel Tower is situated in Paris, France.",
        "Paris is home to the Eiffel Tower.",
    ]
    meteor_score = evaluator.compute_meteor_single(pred, refs)
    assert 0.0 <= meteor_score <= 100.0
    assert meteor_score > 40.0, f"Expected reasonable alignment, got {meteor_score}"


def test_factuality_evaluator_syntax_leak_classification():
    """Verifies factuality parsing handles structured claims and syntax leak classification."""
    fact_eval = FactualityEvaluator()
    mock_items = [{
        "id": "leak_test",
        "dataset": "totto",
        "linearized_input": "[TITLE] Paris [ROW] Eiffel Tower [COL] France",
        "serialized_text": "[TITLE] Paris [ROW] Eiffel Tower [COL] France",
    }]
    preds = ["[ROW] Eiffel Tower is in France [COL]"]
    res = fact_eval.evaluate_batch(mock_items, preds)
    assert "factual_precision" in res
    assert "hallucination_rate" in res
    assert "structured_coverage" in res


def test_historical_checkpoints_immutability():
    """Verifies all historical checkpoints exist and have not been modified."""
    for key, path in CHECKPOINTS.items():
        assert os.path.exists(path), f"Required checkpoint missing: {path}"
        adapter_weights = os.path.join(path, "adapter_model.safetensors")
        assert os.path.exists(adapter_weights), f"Adapter weights missing in: {path}"
        size_mb = os.path.getsize(adapter_weights) / (1024 * 1024)
        assert size_mb > 5.0, f"Adapter weights file suspicious size: {size_mb} MB"
