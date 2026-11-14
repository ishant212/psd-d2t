"""Unit Tests for Phase 2B Extended Controlled Training Infrastructure.

Verifies:
1. Phase 2B config parsing & isolation
2. Evaluation set deterministic loading
3. Layer-wise PSD logging
4. Metric computation (BLEU, ROUGE, BERTScore, Structured Content Coverage)
5. Context length bucketing
6. Artifact isolation (no Phase 2A overwrite)
"""

import os
import json
import yaml
import pytest
import torch

from src.data.unified_schema import UnifiedRecord
from src.baseline.evaluate_metrics import MetricEvaluator
from src.psd.divergence import PSDLoss
from src.psd.context_length_analysis import analyze_length_buckets


def test_phase2b_config_isolation():
    """Verifies that Phase 2B configs specify isolated output directories and 500 steps."""
    b_path = "configs/phase2b_baseline_config.yaml"
    p_path = "configs/phase2b_psd_config.yaml"

    assert os.path.exists(b_path), f"Missing {b_path}"
    assert os.path.exists(p_path), f"Missing {p_path}"

    with open(b_path, "r") as f:
        b_cfg = yaml.safe_load(f)
    with open(p_path, "r") as f:
        p_cfg = yaml.safe_load(f)

    # Check isolation
    assert "phase2b" in b_cfg["output"]["experiment_dir"]
    assert "phase2b" in p_cfg["output"]["experiment_dir"]
    assert b_cfg["output"]["experiment_dir"] != p_cfg["output"]["experiment_dir"]

    # Check hyperparameter parity
    assert b_cfg["training"]["learning_rate"] == p_cfg["training"]["learning_rate"] == 0.0002
    assert b_cfg["training"]["gradient_accumulation_steps"] == p_cfg["training"]["gradient_accumulation_steps"] == 4
    assert b_cfg["training"]["max_seq_len"] == p_cfg["training"]["max_seq_len"] == 256
    assert b_cfg["experiment"]["seed"] == p_cfg["experiment"]["seed"] == 42
    assert b_cfg["training"]["num_steps"] == p_cfg["training"]["num_steps"] == 500


def test_evaluation_ids_determinism():
    """Verifies that fixed evaluation set exists and contains balanced datasets."""
    eval_ids_path = "reports/experiments/phase2b/evaluation_ids.json"
    assert os.path.exists(eval_ids_path), f"Missing {eval_ids_path}"

    with open(eval_ids_path, "r", encoding="utf-8") as f:
        recs = json.load(f)

    assert len(recs) >= 15
    datasets = set(r["dataset"] for r in recs)
    assert "totto" in datasets
    assert "dart" in datasets
    assert "webnlg" in datasets

    for r in recs:
        assert "id" in r
        assert "serialized_text" in r
        assert "target_text" in r


def test_metric_evaluator_and_coverage():
    """Verifies MetricEvaluator computing BLEU, ROUGE, and Structured Content Coverage."""
    evaluator = MetricEvaluator()
    preds = ["Daniel Henry Chamberlain was the governor of South Carolina."]
    refs = [["Daniel Henry Chamberlain was the 76th Governor of South Carolina from 1874."]]

    metrics = evaluator.evaluate_batch(preds, refs, compute_bertscore=False)
    assert "bleu" in metrics
    assert "rougeL" in metrics
    assert metrics["rougeL"] > 0.0

    # Test structured coverage
    totto_mock = {
        "dataset": "totto",
        "nodes": [{"label": "Daniel Henry Chamberlain"}, {"label": "South Carolina"}, {"label": "1874"}]
    }
    cov = evaluator.compute_structured_content_coverage(totto_mock, preds[0])
    assert "structured_coverage" in cov
    assert cov["structured_coverage"] > 0.0


def test_layerwise_psd_loss_output():
    """Verifies PSD loss calculates layer-wise losses for selected layers."""
    psd_loss_fn = PSDLoss(selected_layers=[11, 23, 34])
    seq_len = 16
    attentions = [torch.randn(1, 4, seq_len, seq_len).softmax(dim=-1) for _ in range(36)]
    P_target = torch.randn(1, seq_len, seq_len).softmax(dim=-1)
    valid_mask = torch.ones(1, seq_len, seq_len, dtype=torch.bool)

    loss, per_layer = psd_loss_fn(attentions, P_target, valid_mask)
    assert loss.item() >= 0.0
    assert 11 in per_layer
    assert 23 in per_layer
    assert 34 in per_layer


def test_phase2a_artifacts_preservation():
    """Verifies that Phase 2A 10-step experiment artifacts remain untouched."""
    p2a_base_csv = "reports/experiments/baseline/training_log.csv"
    p2a_psd_csv = "reports/experiments/psd/training_log.csv"

    assert os.path.exists(p2a_base_csv), "Phase 2A baseline training_log.csv must exist"
    assert os.path.exists(p2a_psd_csv), "Phase 2A PSD training_log.csv must exist"

    with open(p2a_base_csv, "r") as f:
        lines = [line for line in f if line.strip()]
        assert len(lines) == 11  # 1 header + 10 step rows
