"""Unit tests for Phase 5 (METEOR Integration) and Factuality/Hallucination Evaluation."""

import os
import sys
import json
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.baseline.evaluate_metrics import MetricEvaluator
from src.baseline.factuality import FactualityEvaluator, FactUnit, FactualityResult


def test_meteor_metric_exact_match():
    """Verifies that an exact match achieves near 100% METEOR score."""
    evaluator = MetricEvaluator()
    assert evaluator.has_meteor, "METEOR score must be available via NLTK"

    hyp = "The dog sat on the mat."
    refs = [["The dog sat on the mat."]]
    metrics = evaluator.evaluate_batch([hyp], refs)
    assert metrics["meteor"] >= 95.0
    assert metrics["bleu"] >= 95.0
    assert metrics["rougeL"] >= 95.0


def test_meteor_multi_reference_alignment():
    """Verifies that METEOR aligns against the best matching reference among multiple references."""
    evaluator = MetricEvaluator()
    hyp = "Alma Jodorowsky starred in the film Kids in Love as Evelyn."
    refs = [
        "Alma Jodorowsky had the role of Evelyn in 2016 film Kids in Love.",
        "Alma Jodorowsky starred in the film Kids in Love as Evelyn.",  # Exact match here
        "In 2016, Alma Jodorowsky played Evelyn in Kids in Love."
    ]
    score = evaluator.compute_meteor_single(hyp, refs)
    assert score >= 95.0, f"Expected high score for exact multi-ref match, got {score}"


def test_meteor_empty_and_edge_cases():
    """Verifies graceful handling of empty hypothesis and reference strings."""
    evaluator = MetricEvaluator()
    assert evaluator.compute_meteor_single("", ["Some reference"]) == 0.0
    assert evaluator.compute_meteor_single("Some prediction", []) == 0.0
    assert evaluator.compute_meteor_single("   ", ["   "]) == 0.0

    batch_res = evaluator.evaluate_batch([], [])
    assert batch_res["meteor"] == 0.0


def test_factuality_fully_grounded():
    """Verifies that a fully grounded prediction receives 100% factual precision and 0% hallucination."""
    evaluator = FactualityEvaluator()
    source_record = {
        "dataset": "totto",
        "source": "[PAGE] Alma Jodorowsky [SECTION] Filmography [ROW 8] [HCELL] 2016 | [HCELL] Kids in Love | [HCELL] Evelyn",
        "nodes": [
            {"label": "Alma Jodorowsky"},
            {"label": "Kids in Love"},
            {"label": "2016"},
            {"label": "Evelyn"},
        ],
        "edges": [],
    }
    pred = "In 2016, Alma Jodorowsky played Evelyn in Kids in Love."
    res = evaluator.evaluate_single(source_record, pred)
    assert res.total_claims > 0
    assert res.unsupported_claims == 0
    assert res.factual_precision == 100.0
    assert res.hallucination_rate == 0.0


def test_factuality_unsupported_claims_detection():
    """Verifies that fabricated numbers and entities are flagged as unsupported claims."""
    evaluator = FactualityEvaluator()
    source_record = {
        "dataset": "dart",
        "source": "[TRIPLE] [SUBJ] Mars Hill College [REL] JOINED [OBJ] 1973 [TRIPLE] [SUBJ] Mars Hill College [REL] LOCATION [OBJ] Mars Hill, North Carolina",
        "nodes": [
            {"label": "Mars Hill College"},
            {"label": "1973"},
            {"label": "Mars Hill, North Carolina"},
        ],
        "edges": [
            {"relation": "JOINED"},
            {"relation": "LOCATION"},
        ],
    }
    # "Harvard University" and "1850" are fabricated/hallucinated
    pred = "Harvard University was established in 1850 in Boston."
    res = evaluator.evaluate_single(source_record, pred)
    assert res.unsupported_claims > 0
    assert res.hallucination_rate > 50.0


def test_factuality_syntax_leakage():
    """Verifies that leaked structural tokens (e.g. [CELL], [ROW 7]) are counted as hallucinations."""
    evaluator = FactualityEvaluator()
    source_record = {"dataset": "totto", "source": "Sample table", "nodes": [], "edges": []}
    pred = "| [CELL] John D. Kennedy | [ROW 7] [HEADER] 80"
    res = evaluator.evaluate_single(source_record, pred)
    syntax_claims = [c for c in res.claims if c.category == "syntax_leak"]
    assert len(syntax_claims) >= 2
    assert all(not c.supported for c in syntax_claims)


def test_historical_final_evaluation_artifacts_exist():
    """Verifies that the generated final evaluation artifacts exist and contain expected data."""
    eval_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "final_evaluation")
    assert os.path.exists(os.path.join(eval_dir, "final_metrics_with_meteor.csv")), "final_metrics_with_meteor.csv missing"
    assert os.path.exists(os.path.join(eval_dir, "final_metrics_with_meteor.json")), "final_metrics_with_meteor.json missing"
    assert os.path.exists(os.path.join(eval_dir, "meteor_analysis.csv")), "meteor_analysis.csv missing"
    assert os.path.exists(os.path.join(eval_dir, "factuality_results.csv")), "factuality_results.csv missing"
    assert os.path.exists(os.path.join(eval_dir, "verification.json")), "verification.json missing"
    assert os.path.exists(os.path.join(eval_dir, "model_verification.json")), "model_verification.json missing"

    with open(os.path.join(eval_dir, "final_metrics_with_meteor.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) >= 8, f"Expected at least 8 evaluated experiments, found {len(data)}"


def test_cross_model_registry_configuration():
    """Verifies that MODEL_REGISTRY contains complete configurations for Qwen, Llama, and Gemma."""
    from src.psd.run_cross_model_evaluation import MODEL_REGISTRY

    for key in ["qwen", "llama", "gemma"]:
        assert key in MODEL_REGISTRY, f"Missing {key} in MODEL_REGISTRY"
        cfg = MODEL_REGISTRY[key]
        assert "model_id" in cfg
        assert "display_name" in cfg
        assert "selected_layers" in cfg
        assert len(cfg["selected_layers"]) == 3, f"Expected 3 PSD layers, got {len(cfg['selected_layers'])}"
        assert "prompt_prefix" in cfg
        assert "prompt_suffix" in cfg
        assert "eos_token" in cfg
        assert "target_modules" in cfg

    # Verify specific layer indices
    assert MODEL_REGISTRY["qwen"]["selected_layers"] == [11, 23, 34]
    assert MODEL_REGISTRY["llama"]["selected_layers"] == [9, 18, 26]
    assert MODEL_REGISTRY["gemma"]["selected_layers"] == [11, 22, 32]


def test_format_d2t_training_example_all_models():
    """Verifies format_d2t_training_example_for_model across Qwen, Llama, and Gemma tokenizers."""
    from src.psd.run_cross_model_evaluation import MODEL_REGISTRY, format_d2t_training_example_for_model, load_dataset_records
    from src.preprocessing.serialize import Serializer
    from transformers import AutoTokenizer

    records = load_dataset_records(split="train", max_records=1)
    assert len(records) > 0, "Expected at least one training record"
    sample_record = records[0]
    serializer = Serializer()

    for key in ["qwen", "llama", "gemma"]:
        cfg = MODEL_REGISTRY[key]
        model_path = cfg["model_id"] if os.path.exists(cfg["model_id"]) else cfg.get("fallback_id", cfg["model_id"])
        tok = AutoTokenizer.from_pretrained(model_path)

        inp, lab, nmap = format_d2t_training_example_for_model(
            record=sample_record,
            serializer=serializer,
            tokenizer=tok,
            model_cfg=cfg,
            max_seq_len=128,
        )

        assert inp.shape[0] == 1
        assert lab.shape[0] == 1
        assert inp.shape[1] == lab.shape[1]
        assert len(nmap) == inp.shape[1]

        # Ensure prompt tokens are masked to -100
        lab_list = lab[0].tolist()
        assert -100 in lab_list, f"Expected masked prompt tokens in {key}"
        # Ensure target tokens are not all masked
        non_masked = [x for x in lab_list if x != -100]
        assert len(non_masked) > 0, f"Expected supervised target tokens in {key}"


def test_final_results_aggregation_and_gains():
    """Verifies that aggregate_all_results and compute_relative_gains run cleanly."""
    from src.baseline.generate_final_report_and_plots import aggregate_all_results, compute_relative_gains

    records = aggregate_all_results()
    assert len(records) >= 2, f"Expected at least Qwen Baseline and PSD, got {len(records)}"

    required_keys = ["model", "config", "steps", "bleu", "meteor", "rougeL", "bertscore_f1", "structured_coverage", "factual_precision", "hallucination_rate"]
    for r in records:
        for k in required_keys:
            assert k in r, f"Missing key {k} in aggregated record: {r}"

    gains = compute_relative_gains(records)
    assert len(gains) >= 1
    qwen_gain = next((g for g in gains if g["model"] == "Qwen3-4B"), None)
    assert qwen_gain is not None, "Expected Qwen3-4B in gains"
    assert "delta_bleu" in qwen_gain
    assert "rel_hallucination_reduction_pct" in qwen_gain

