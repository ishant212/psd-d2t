"""Long-Context Length Analysis for Phase 2B.

Categorizes evaluation examples into length buckets based on original serialized text length:
- <=256
- 257-512
- 513-1024
- >1024

Note on experimental constraint:
Model-visible training & evaluation sequence length is max_seq_len=256 (due to memory/speed controls).
Examples longer than 256 were truncated at 256. This analysis compares how Baseline vs PSD behave
on structured inputs of varying original complexity and length.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.baseline.evaluate_metrics import MetricEvaluator


def analyze_length_buckets(
    baseline_preds_path: str = "reports/experiments/phase2b/baseline/sample_predictions.json",
    psd_preds_path: str = "reports/experiments/phase2b/psd/sample_predictions.json",
    output_json_path: str = "reports/experiments/phase2b/length_bucket_analysis.json",
) -> Dict[str, Any]:
    """Computes paired metrics bucketed by original serialized input length."""
    evaluator = MetricEvaluator()

    with open(baseline_preds_path, "r", encoding="utf-8") as f:
        b_preds = json.load(f)

    with open(psd_preds_path, "r", encoding="utf-8") as f:
        p_preds = json.load(f)

    # Index by ID
    b_dict = {p["id"]: p for p in b_preds}
    p_dict = {p["id"]: p for p in p_preds}

    common_ids = [pid for pid in b_dict if pid in p_dict]

    bucket_definitions = {
        "<=256": (0, 256),
        "257-512": (257, 512),
        "513-1024": (513, 1024),
        ">1024": (1025, 100000),
    }

    bucket_data: Dict[str, Dict[str, Any]] = {
        k: {
            "count": 0,
            "sample_ids": [],
            "baseline_predictions": [],
            "psd_predictions": [],
            "references": [],
            "baseline_coverages": [],
            "psd_coverages": [],
        }
        for k in bucket_definitions
    }

    for cid in common_ids:
        b_item = b_dict[cid]
        p_item = p_dict[cid]

        # Determine original word/char length of source input
        source = b_item.get("source", "")
        orig_len = len(source.split())  # approximate token/word length

        matched_bucket = "<=256"
        for b_name, (low, high) in bucket_definitions.items():
            if low <= orig_len <= high:
                matched_bucket = b_name
                break

        b_sub = bucket_data[matched_bucket]
        b_sub["count"] += 1
        b_sub["sample_ids"].append(cid)
        b_sub["baseline_predictions"].append(b_item["prediction"])
        b_sub["psd_predictions"].append(p_item["prediction"])
        b_sub["references"].append(b_item["references"])
        b_sub["baseline_coverages"].append(b_item.get("structured_coverage", 0.0))
        b_sub["psd_coverages"].append(p_item.get("structured_coverage", 0.0))

    results: Dict[str, Any] = {
        "total_evaluated_samples": len(common_ids),
        "constraint_note": "Model-visible sequence length was max_seq_len=256. Original examples >256 were truncated at 256.",
        "buckets": {},
    }

    for b_name, b_info in bucket_data.items():
        cnt = b_info["count"]
        if cnt == 0:
            results["buckets"][b_name] = {
                "count": 0,
                "baseline": {},
                "psd": {},
                "delta": {},
            }
            continue

        b_metrics = evaluator.evaluate_batch(b_info["baseline_predictions"], b_info["references"], compute_bertscore=True)
        b_metrics["structured_coverage"] = round(sum(b_info["baseline_coverages"]) / cnt, 2)

        p_metrics = evaluator.evaluate_batch(b_info["psd_predictions"], b_info["references"], compute_bertscore=True)
        p_metrics["structured_coverage"] = round(sum(b_info["psd_coverages"]) / cnt, 2)

        delta = {}
        for m_key in b_metrics:
            diff = round(p_metrics.get(m_key, 0.0) - b_metrics.get(m_key, 0.0), 2)
            delta[m_key] = diff

        results["buckets"][b_name] = {
            "count": cnt,
            "sample_ids": b_info["sample_ids"],
            "baseline": b_metrics,
            "psd": p_metrics,
            "delta_psd_minus_baseline": delta,
        }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[LengthAnalysis] Saved bucket analysis to {output_json_path}")
    return results


if __name__ == "__main__":
    analyze_length_buckets()
