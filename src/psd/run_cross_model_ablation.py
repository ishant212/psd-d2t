"""Cross-Model Ablation Suite for Phase 6.

Evaluates across the top 3 ~3B models (Qwen2.5-3B, Llama-3.2-3B, Gemma-3-4B):
1. Loss Weight Sensitivity (lambda in [0.02, 0.05, 0.20])
2. Layer-Selective Supervision (Early vs. Middle vs. Deep bands)
3. Component Decomposition (Graph-Only vs. Semantic-Only vs. Full PSD)

Outputs are saved in reports/experiments/ablation/ with automatic checkpoint resumption.
"""

import os
import sys
import gc
import json
import csv
import time
import argparse
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.psd.run_cross_model_evaluation import (
    MODEL_REGISTRY,
    CrossModelTrainer,
    load_dataset_records,
)

LAYER_BANDS = {
    "gemma_4b": {
        "early": [3, 7, 11],
        "middle": [11, 16, 22],
        "deep": [24, 28, 32],
    },
    "llama_3b": {
        "early": [3, 6, 9],
        "middle": [9, 14, 18],
        "deep": [20, 23, 26],
    },
    "qwen_3b": {
        "early": [3, 7, 11],
        "middle": [12, 18, 23],
        "deep": [26, 30, 34],
    },
}


def run_ablation_suite(
    models: List[str],
    ablation_types: List[str],
    steps: int = 150,
) -> None:
    ablation_base = os.path.join(PROJECT_ROOT, "reports", "experiments", "ablation")
    os.makedirs(ablation_base, exist_ok=True)

    # 1. Load evaluation benchmark
    eval_path = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b", "evaluation_ids.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_items = json.load(f)
        print(f"Loaded fixed 15-instance benchmark from {eval_path}")
    else:
        val_records = load_dataset_records(split="validation", max_records=15)
        eval_items = [
            {
                "dataset": r.dataset,
                "id": r.id,
                "input_type": r.input_type,
                "target_text": r.target_text,
                "reference_texts": r.reference_texts,
                "serialized_text": r.serialized_text,
            }
            for r in val_records
        ]

    # 2. Load training data
    train_records = load_dataset_records(split="train")
    print(f"Loaded {len(train_records)} training records from unified dataset")

    summary_rows = []

    # =========================================================================
    # ABLATION 1: Lambda Loss Weight Sensitivity
    # =========================================================================
    if "lambda" in ablation_types:
        lambda_values = [0.02, 0.05, 0.20]
        print("\n" + "=" * 75)
        print("STARTING ABLATION SUITE: LOSS WEIGHT SENSITIVITY (LAMBDA)")
        print("=" * 75)

        for model_key in models:
            for l_val in lambda_values:
                exp_name = f"{model_key}_lambda_{str(l_val).replace('.', '_')}"
                exp_dir = os.path.join(ablation_base, exp_name)

                print(f"\n>>> Running: {exp_name.upper()} (lambda={l_val}, steps={steps})")
                trainer = CrossModelTrainer(
                    model_key=model_key,
                    mode="psd",
                    max_steps=steps,
                    lambda_psd=l_val,
                    output_dir=exp_dir,
                )
                trainer.train(train_records)
                metrics, preds, fact = trainer.evaluate_and_generate(eval_items)
                trainer.cleanup()

                summary_rows.append({
                    "ablation_type": "lambda_sensitivity",
                    "model": model_key,
                    "condition": f"lambda={l_val}",
                    "bleu": metrics.get("bleu", 0.0),
                    "rougeL": metrics.get("rougeL", 0.0),
                    "bertscore_f1": metrics.get("bertscore_f1", 0.0),
                    "factual_precision": metrics.get("factual_precision", 0.0),
                    "hallucination_rate": metrics.get("hallucination_rate", 0.0),
                })

    # =========================================================================
    # ABLATION 2: Layer-Selective Supervision (Early vs. Middle vs. Deep)
    # =========================================================================
    if "layer" in ablation_types:
        print("\n" + "=" * 75)
        print("STARTING ABLATION SUITE: LAYER-SELECTIVE SUPERVISION")
        print("=" * 75)

        for model_key in models:
            bands = LAYER_BANDS.get(model_key, {})
            for band_name, layers in bands.items():
                exp_name = f"{model_key}_layer_{band_name}"
                exp_dir = os.path.join(ablation_base, exp_name)

                print(f"\n>>> Running: {exp_name.upper()} (layers={layers}, steps={steps})")
                trainer = CrossModelTrainer(
                    model_key=model_key,
                    mode="psd",
                    max_steps=steps,
                    selected_layers=layers,
                    output_dir=exp_dir,
                )
                trainer.train(train_records)
                metrics, preds, fact = trainer.evaluate_and_generate(eval_items)
                trainer.cleanup()

                summary_rows.append({
                    "ablation_type": "layer_placement",
                    "model": model_key,
                    "condition": f"{band_name}_{layers}",
                    "bleu": metrics.get("bleu", 0.0),
                    "rougeL": metrics.get("rougeL", 0.0),
                    "bertscore_f1": metrics.get("bertscore_f1", 0.0),
                    "factual_precision": metrics.get("factual_precision", 0.0),
                    "hallucination_rate": metrics.get("hallucination_rate", 0.0),
                })

    # =========================================================================
    # ABLATION 3: Component Decomposition (Graph-Only vs. Semantic-Only)
    # =========================================================================
    if "component" in ablation_types:
        print("\n" + "=" * 75)
        print("STARTING ABLATION SUITE: COMPONENT DECOMPOSITION")
        print("=" * 75)

        comp_configs = {
            "graph_only": {"enable_semantic": False, "enable_positional": False, "enable_graph": True},
            "semantic_only": {"enable_semantic": True, "enable_positional": False, "enable_graph": False},
        }

        for model_key in models:
            for c_name, builder_kwargs in comp_configs.items():
                exp_name = f"{model_key}_comp_{c_name}"
                exp_dir = os.path.join(ablation_base, exp_name)

                print(f"\n>>> Running: {exp_name.upper()} (builder={builder_kwargs}, steps={steps})")
                trainer = CrossModelTrainer(
                    model_key=model_key,
                    mode="psd",
                    max_steps=steps,
                    target_builder_kwargs=builder_kwargs,
                    output_dir=exp_dir,
                )
                trainer.train(train_records)
                metrics, preds, fact = trainer.evaluate_and_generate(eval_items)
                trainer.cleanup()

                summary_rows.append({
                    "ablation_type": "component_decomposition",
                    "model": model_key,
                    "condition": c_name,
                    "bleu": metrics.get("bleu", 0.0),
                    "rougeL": metrics.get("rougeL", 0.0),
                    "bertscore_f1": metrics.get("bertscore_f1", 0.0),
                    "factual_precision": metrics.get("factual_precision", 0.0),
                    "hallucination_rate": metrics.get("hallucination_rate", 0.0),
                })

    # Save summary CSV
    summary_path = os.path.join(ablation_base, "cross_model_ablation_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ablation_type", "model", "condition", "bleu", "rougeL", "bertscore_f1", "factual_precision", "hallucination_rate"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved cross-model ablation summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Model Ablation Runner")
    parser.add_argument("--models", nargs="+", default=["gemma_4b", "llama_3b", "qwen_3b"])
    parser.add_argument("--ablation_types", nargs="+", default=["lambda", "layer"], choices=["lambda", "layer", "component", "all"])
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()

    types = ["lambda", "layer", "component"] if "all" in args.ablation_types else args.ablation_types
    print(f"Running Cross-Model Ablations:")
    print(f"  Models:         {args.models}")
    print(f"  Ablation Types: {types}")
    print(f"  Steps:          {args.steps}")

    run_ablation_suite(models=args.models, ablation_types=types, steps=args.steps)
