"""Position-Controlled Experiment Pipeline.

Constructs controlled positional variants (BEGINNING, MIDDLE, END)
for identical underlying structured records to test the positional
degradation / Lost-in-the-Middle hypothesis on Baseline vs PSD.
"""

import os
import sys
import json
import csv
import argparse
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.data.unified_schema import UnifiedRecord, Node, Edge
from src.preprocessing.serialize import Serializer
from src.psd.trainer import load_dataset_records
from src.psd.evaluate import ModelEvaluator, compute_structured_coverage
from src.baseline.evaluate_metrics import MetricEvaluator


def create_positional_variants(record: UnifiedRecord) -> Dict[str, UnifiedRecord]:
    """Creates BEGINNING, MIDDLE, and END serialized variants of a single record.

    For tabular data (ToTTo):
      Reorders rows / cell nodes such that highlighted cells appear at the beginning,
      middle (surrounded by unhighlighted context), or end.

    For graph/triple data (DART, WebNLG):
      Reorders triples / edges such that target-relevant triples appear at the beginning,
      middle (surrounded by distractors/context), or end.
    """
    variants = {}

    if record.dataset == "totto":
        highlighted_nodes = [n for n in record.nodes if n.get("type") == "cell"]
        # If we have table metadata with context cells, build filler
        filler_nodes = [
            {"id": f"ctx_{i}", "label": f"Context Header {i}: Value {i}", "type": "context_cell"}
            for i in range(1, 7)
        ]
        
        # 1. Beginning: Highlighted first, then context
        rec_beg = UnifiedRecord(
            dataset=record.dataset,
            id=f"{record.id}_pos_beginning",
            input_type=record.input_type,
            structured_input=record.structured_input,
            nodes=highlighted_nodes + filler_nodes,
            edges=record.edges,
            target_text=record.target_text,
            reference_texts=record.reference_texts,
        )
        variants["beginning"] = rec_beg

        # 2. Middle: Filler first half, highlighted, filler second half
        half = len(filler_nodes) // 2
        rec_mid = UnifiedRecord(
            dataset=record.dataset,
            id=f"{record.id}_pos_middle",
            input_type=record.input_type,
            structured_input=record.structured_input,
            nodes=filler_nodes[:half] + highlighted_nodes + filler_nodes[half:],
            edges=record.edges,
            target_text=record.target_text,
            reference_texts=record.reference_texts,
        )
        variants["middle"] = rec_mid

        # 3. End: Filler first, highlighted at the end
        rec_end = UnifiedRecord(
            dataset=record.dataset,
            id=f"{record.id}_pos_end",
            input_type=record.input_type,
            structured_input=record.structured_input,
            nodes=filler_nodes + highlighted_nodes,
            edges=record.edges,
            target_text=record.target_text,
            reference_texts=record.reference_texts,
        )
        variants["end"] = rec_end

    else:
        # Triples (DART / WebNLG)
        key_edges = record.edges
        key_nodes = record.nodes

        filler_edges = [
            {"source": f"Distractor_A{i}", "target": f"Distractor_B{i}", "relation": "associatedWith"}
            for i in range(1, 5)
        ]
        filler_nodes = [
            {"id": f"Distractor_A{i}", "label": f"Distractor Item A{i}", "type": "entity"}
            for i in range(1, 5)
        ] + [
            {"id": f"Distractor_B{i}", "label": f"Distractor Item B{i}", "type": "entity"}
            for i in range(1, 5)
        ]

        # Beginning
        rec_beg = UnifiedRecord(
            dataset=record.dataset,
            id=f"{record.id}_pos_beginning",
            input_type=record.input_type,
            structured_input=record.structured_input,
            nodes=key_nodes + filler_nodes,
            edges=key_edges + filler_edges,
            target_text=record.target_text,
            reference_texts=record.reference_texts,
        )
        variants["beginning"] = rec_beg

        # Middle
        half_e = len(filler_edges) // 2
        half_n = len(filler_nodes) // 2
        rec_mid = UnifiedRecord(
            dataset=record.dataset,
            id=f"{record.id}_pos_middle",
            input_type=record.input_type,
            structured_input=record.structured_input,
            nodes=filler_nodes[:half_n] + key_nodes + filler_nodes[half_n:],
            edges=filler_edges[:half_e] + key_edges + filler_edges[half_e:],
            target_text=record.target_text,
            reference_texts=record.reference_texts,
        )
        variants["middle"] = rec_mid

        # End
        rec_end = UnifiedRecord(
            dataset=record.dataset,
            id=f"{record.id}_pos_end",
            input_type=record.input_type,
            structured_input=record.structured_input,
            nodes=filler_nodes + key_nodes,
            edges=filler_edges + key_edges,
            target_text=record.target_text,
            reference_texts=record.reference_texts,
        )
        variants["end"] = rec_end

    return variants


def run_position_control_experiment(
    baseline_adapter_dir: str = "models/phase2a/baseline",
    psd_adapter_dir: str = "models/phase2a/psd",
    base_model_path: str = "models/qwen3_4b_hf",
    dataset_name: str = "totto",
    split: str = "validation",
    num_samples: int = 15,
    output_json: str = "reports/phase2a/position_controlled_results.json",
    output_csv: str = "reports/phase2a/position_controlled_results.csv",
) -> Dict[str, Any]:
    """Runs controlled position experiment evaluating beginning, middle, and end positions."""
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    records = load_dataset_records(dataset_name=dataset_name, split=split, max_records=num_samples)
    positions = ["beginning", "middle", "end"]

    # Generate positional variant record lists
    variant_records: Dict[str, List[UnifiedRecord]] = {pos: [] for pos in positions}
    for r in records:
        vars_dict = create_positional_variants(r)
        for pos in positions:
            if pos in vars_dict:
                variant_records[pos].append(vars_dict[pos])

    metric_evaluator = MetricEvaluator()
    results: Dict[str, Any] = {
        "dataset": dataset_name,
        "num_records": len(records),
        "positions": positions,
        "baseline_by_position": {},
        "psd_by_position": {},
        "delta_by_position": {},
    }

    # 1. Run Baseline on each position
    print(f"\n[PositionControl] Running Baseline on {dataset_name.upper()} positional variants...")
    b_evaluator = ModelEvaluator(base_model_path=base_model_path, lora_adapter_path=baseline_adapter_dir)
    for pos in positions:
        m, _ = b_evaluator.evaluate_records(variant_records[pos])
        results["baseline_by_position"][pos] = m

    del b_evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Run PSD on each position
    print(f"\n[PositionControl] Running PSD on {dataset_name.upper()} positional variants...")
    p_evaluator = ModelEvaluator(base_model_path=base_model_path, lora_adapter_path=psd_adapter_dir)
    for pos in positions:
        m, _ = p_evaluator.evaluate_records(variant_records[pos])
        results["psd_by_position"][pos] = m

    del p_evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Compute Delta and CSV rows
    csv_rows = []
    metric_keys = ["bleu", "rouge1", "rouge2", "rougeL", "triple_coverage"]

    for pos in positions:
        b_m = results["baseline_by_position"][pos]
        p_m = results["psd_by_position"][pos]
        results["delta_by_position"][pos] = {
            k: round(p_m.get(k, 0.0) - b_m.get(k, 0.0), 2) for k in metric_keys
        }

        for k in metric_keys:
            csv_rows.append({
                "dataset": dataset_name,
                "position": pos,
                "metric": k,
                "baseline": b_m.get(k, 0.0),
                "psd": p_m.get(k, 0.0),
                "delta": round(p_m.get(k, 0.0) - b_m.get(k, 0.0), 2),
            })

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save CSV
    if csv_rows:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "position", "metric", "baseline", "psd", "delta"])
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"[PositionControl] Saved positional experiment results to {output_json} and {output_csv}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Position-Controlled Experiment")
    parser.add_argument("--baseline_dir", type=str, default="models/phase2a/baseline")
    parser.add_argument("--psd_dir", type=str, default="models/phase2a/psd")
    parser.add_argument("--dataset", type=str, default="totto")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    run_position_control_experiment(
        baseline_adapter_dir=args.baseline_dir,
        psd_adapter_dir=args.psd_dir,
        dataset_name=args.dataset,
        num_samples=args.samples,
    )
