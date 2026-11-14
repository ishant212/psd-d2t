"""Compute METEOR and Factuality across Historical Experiments without Retraining.

Processes saved prediction files from Phase 2B, Phase 3A, Phase 3B, and Phase 4B.
Outputs consolidated tables to reports/experiments/final_evaluation/
without modifying any historical artifacts.
"""

import os
import sys
import json
import csv
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.baseline.evaluate_metrics import MetricEvaluator
from src.baseline.factuality import FactualityEvaluator


def load_json_predictions(path: str) -> List[Dict[str, Any]]:
    """Loads prediction entries from json file (handles both list and dict formats)."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "instances" in data:
        return data["instances"]
    return []


def run_historical_evaluation(output_dir: str = "reports/experiments/final_evaluation"):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    metric_evaluator = MetricEvaluator()
    fact_evaluator = FactualityEvaluator()

    # Define historical prediction artifacts to evaluate
    experiment_targets = [
        # Phase 2B (Primary Qwen3-4B 500-step runs)
        {
            "phase": "phase2b",
            "experiment": "phase2b_baseline",
            "model": "Qwen3-4B",
            "psd": False,
            "steps": 500,
            "context_len": 256,
            "path": "reports/experiments/phase2b/baseline/sample_predictions.json",
            "metrics_json": "reports/experiments/phase2b/baseline/metrics.json",
        },
        {
            "phase": "phase2b",
            "experiment": "phase2b_psd",
            "model": "Qwen3-4B",
            "psd": True,
            "steps": 500,
            "context_len": 256,
            "path": "reports/experiments/phase2b/psd/sample_predictions.json",
            "metrics_json": "reports/experiments/phase2b/psd/metrics.json",
        },
        # Phase 3A (Qwen3-4B 150-step screening runs)
        {
            "phase": "phase3a",
            "experiment": "phase3a_baseline",
            "model": "Qwen3-4B",
            "psd": False,
            "steps": 150,
            "context_len": 256,
            "path": "reports/experiments/phase3a/baseline/sample_predictions.json",
            "metrics_json": "reports/experiments/phase3a/baseline/metrics.json",
        },
        {
            "phase": "phase3a",
            "experiment": "phase3a_full_psd",
            "model": "Qwen3-4B",
            "psd": True,
            "steps": 150,
            "context_len": 256,
            "path": "reports/experiments/phase3a/full_psd/sample_predictions.json",
            "metrics_json": "reports/experiments/phase3a/full_psd/metrics.json",
        },
        # Phase 3B (Qwen3-4B 500-step confirmation runs)
        {
            "phase": "phase3b",
            "experiment": "phase3b_baseline",
            "model": "Qwen3-4B",
            "psd": False,
            "steps": 500,
            "context_len": 256,
            "path": "reports/experiments/phase3b/baseline/sample_predictions.json",
            "metrics_json": "reports/experiments/phase3b/baseline/metrics.json",
        },
        {
            "phase": "phase3b",
            "experiment": "phase3b_full_psd",
            "model": "Qwen3-4B",
            "psd": True,
            "steps": 500,
            "context_len": 256,
            "path": "reports/experiments/phase3b/full_psd/sample_predictions.json",
            "metrics_json": "reports/experiments/phase3b/full_psd/metrics.json",
        },
        # Phase 4B (Qwen3-4B 512-context scaling runs)
        {
            "phase": "phase4b",
            "experiment": "phase4b_baseline_512",
            "model": "Qwen3-4B",
            "psd": False,
            "steps": 500,
            "context_len": 512,
            "path": "reports/experiments/phase4b/baseline_512/sample_predictions.json",
            "metrics_json": "reports/experiments/phase4b/baseline_512/metrics.json",
        },
        {
            "phase": "phase4b",
            "experiment": "phase4b_psd_512",
            "model": "Qwen3-4B",
            "psd": True,
            "steps": 500,
            "context_len": 512,
            "path": "reports/experiments/phase4b/psd_512/sample_predictions.json",
            "metrics_json": "reports/experiments/phase4b/psd_512/metrics.json",
        },
    ]

    final_metrics_records = []
    meteor_analysis_records = []
    factuality_records = []
    by_dataset_records = []

    print(f"[Evaluation] Processing {len(experiment_targets)} historical experiments...")

    for target in experiment_targets:
        full_pred_path = os.path.join(PROJECT_ROOT, target["path"])
        full_metrics_path = os.path.join(PROJECT_ROOT, target["metrics_json"])
        
        preds_data = load_json_predictions(full_pred_path)
        if not preds_data:
            print(f"  [Warning] Missing predictions for {target['experiment']}")
            continue

        # Load existing metrics for reference (BLEU, ROUGE, BERTScore) to preserve exact values
        existing_metrics = {}
        if os.path.exists(full_metrics_path):
            with open(full_metrics_path, "r", encoding="utf-8") as f:
                raw_m = json.load(f)
                existing_metrics = raw_m.get("final_generation_metrics", {})

        predictions = [p.get("prediction", "") for p in preds_data]
        references_list = []
        for p in preds_data:
            refs = p.get("references", [])
            if not refs and "target" in p:
                refs = [p["target"]]
            references_list.append([r for r in refs if r.strip()])

        # 1. Compute METEOR
        eval_res = metric_evaluator.evaluate_batch(predictions, references_list, compute_bertscore=False)
        meteor_score = eval_res["meteor"]

        # Preserve existing metrics if present, otherwise use newly computed values
        bleu = existing_metrics.get("bleu", eval_res["bleu"])
        rouge1 = existing_metrics.get("rouge1", eval_res["rouge1"])
        rouge2 = existing_metrics.get("rouge2", eval_res["rouge2"])
        rougeL = existing_metrics.get("rougeL", eval_res["rougeL"])
        bertscore_f1 = existing_metrics.get("bertscore_f1", 0.0)

        # 2. Compute Factuality & Hallucination
        fact_res = fact_evaluator.evaluate_batch(preds_data, predictions)
        factual_precision = fact_res["factual_precision"]
        hallucination_rate = fact_res["hallucination_rate"]
        structured_coverage = fact_res["structured_coverage"]

        row = {
            "phase": target["phase"],
            "experiment": target["experiment"],
            "model": target["model"],
            "psd": "Yes" if target["psd"] else "No",
            "steps": target["steps"],
            "context_len": target["context_len"],
            "bleu_4": bleu,
            "rouge_1": rouge1,
            "rouge_2": rouge2,
            "rouge_l": rougeL,
            "meteor": meteor_score,
            "bertscore_f1": bertscore_f1,
            "structured_coverage": structured_coverage,
            "factual_precision": factual_precision,
            "hallucination_rate": hallucination_rate,
            "total_claims": fact_res["total_claims"],
            "supported_claims": fact_res["supported_claims"],
            "unsupported_claims": fact_res["unsupported_claims"],
        }
        final_metrics_records.append(row)

        meteor_analysis_records.append({
            "experiment": target["experiment"],
            "phase": target["phase"],
            "psd": row["psd"],
            "meteor": meteor_score,
            "bleu_4": bleu,
            "rouge_l": rougeL,
            "meteor_bleu_ratio": round(meteor_score / (bleu + 1e-4), 2),
        })

        factuality_records.append({
            "experiment": target["experiment"],
            "phase": target["phase"],
            "psd": row["psd"],
            "factual_precision": factual_precision,
            "hallucination_rate": hallucination_rate,
            "total_claims": fact_res["total_claims"],
            "supported_claims": fact_res["supported_claims"],
            "unsupported_claims": fact_res["unsupported_claims"],
            "structured_coverage": structured_coverage,
        })

        # Dataset-level breakdown
        datasets = ["totto", "dart", "webnlg"]
        for ds in datasets:
            ds_indices = [i for i, p in enumerate(preds_data) if p.get("dataset") == ds]
            if ds_indices:
                ds_preds = [predictions[i] for i in ds_indices]
                ds_refs = [references_list[i] for i in ds_indices]
                ds_records = [preds_data[i] for i in ds_indices]
                
                ds_eval = metric_evaluator.evaluate_batch(ds_preds, ds_refs, compute_bertscore=False)
                ds_fact = fact_evaluator.evaluate_batch(ds_records, ds_preds)
                
                by_dataset_records.append({
                    "experiment": target["experiment"],
                    "dataset": ds,
                    "psd": row["psd"],
                    "bleu_4": ds_eval["bleu"],
                    "rouge_l": ds_eval["rougeL"],
                    "meteor": ds_eval["meteor"],
                    "structured_coverage": ds_fact["structured_coverage"],
                    "factual_precision": ds_fact["factual_precision"],
                    "hallucination_rate": ds_fact["hallucination_rate"],
                })

    # Save final_metrics_with_meteor.csv
    csv_path = os.path.join(output_dir, "final_metrics_with_meteor.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if final_metrics_records:
            writer = csv.DictWriter(f, fieldnames=list(final_metrics_records[0].keys()))
            writer.writeheader()
            writer.writerows(final_metrics_records)

    # Save final_metrics_with_meteor.json
    json_path = os.path.join(output_dir, "final_metrics_with_meteor.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics_records, f, indent=2)

    # Save meteor_analysis.csv
    meteor_csv = os.path.join(output_dir, "meteor_analysis.csv")
    with open(meteor_csv, "w", newline="", encoding="utf-8") as f:
        if meteor_analysis_records:
            writer = csv.DictWriter(f, fieldnames=list(meteor_analysis_records[0].keys()))
            writer.writeheader()
            writer.writerows(meteor_analysis_records)

    # Save factuality_results.csv and json
    fact_csv = os.path.join(output_dir, "factuality_results.csv")
    with open(fact_csv, "w", newline="", encoding="utf-8") as f:
        if factuality_records:
            writer = csv.DictWriter(f, fieldnames=list(factuality_records[0].keys()))
            writer.writeheader()
            writer.writerows(factuality_records)

    fact_json = os.path.join(output_dir, "factuality_results.json")
    with open(fact_json, "w", encoding="utf-8") as f:
        json.dump(factuality_records, f, indent=2)

    # Save dataset_breakdown.csv
    ds_csv = os.path.join(output_dir, "dataset_breakdown.csv")
    with open(ds_csv, "w", newline="", encoding="utf-8") as f:
        if by_dataset_records:
            writer = csv.DictWriter(f, fieldnames=list(by_dataset_records[0].keys()))
            writer.writeheader()
            writer.writerows(by_dataset_records)

    # Save verification.json
    verif = {
        "status": "METEOR and Factuality Computed on Historical Predictions",
        "meteor_library": "nltk.translate.meteor_score",
        "scale": "0.0 - 100.0",
        "multi_reference_support": True,
        "historical_artifacts_modified": False,
        "num_experiments_evaluated": len(final_metrics_records),
        "files_generated": [
            "final_metrics_with_meteor.csv",
            "final_metrics_with_meteor.json",
            "meteor_analysis.csv",
            "factuality_results.csv",
            "factuality_results.json",
            "dataset_breakdown.csv",
            "verification.json",
        ]
    }
    with open(os.path.join(output_dir, "verification.json"), "w", encoding="utf-8") as f:
        json.dump(verif, f, indent=2)

    print(f"[Complete] Final evaluation metrics written to {output_dir}/")


if __name__ == "__main__":
    run_historical_evaluation()
