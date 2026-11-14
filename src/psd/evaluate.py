"""Controlled Evaluation Pipeline for Baseline vs PSD.

Phase 2A - Metric Comparison across ToTTo, DART, and WebNLG.
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.data.unified_schema import UnifiedRecord
from src.preprocessing.serialize import Serializer
from src.baseline.evaluate_metrics import MetricEvaluator
from src.psd.trainer import load_dataset_records


def compute_structured_coverage(record: UnifiedRecord, prediction: str) -> Dict[str, float]:
    """Computes structured element coverage in generated text."""
    pred_lower = prediction.lower()
    
    if record.dataset in ["dart", "webnlg"]:
        # Entity and relation coverage
        entities = [n.get("label", "").lower().strip() for n in record.nodes if n.get("label", "").strip()]
        relations = [e.get("relation", "").lower().strip().replace("_", " ") for e in record.edges if e.get("relation", "").strip()]
        
        ent_hits = sum(1 for ent in entities if ent in pred_lower)
        ent_cov = (ent_hits / len(entities)) if entities else 1.0
        
        rel_hits = sum(1 for rel in relations if rel in pred_lower)
        rel_cov = (rel_hits / len(relations)) if relations else 1.0
        
        triple_cov = 0.5 * ent_cov + 0.5 * rel_cov
        return {
            "entity_coverage": round(ent_cov * 100.0, 2),
            "relation_coverage": round(rel_cov * 100.0, 2),
            "triple_coverage": round(triple_cov * 100.0, 2),
        }
    elif record.dataset == "totto":
        # Highlighted table cell coverage
        cells = [n.get("label", "").lower().strip() for n in record.nodes if n.get("label", "").strip()]
        cell_hits = sum(1 for c in cells if c in pred_lower)
        cell_cov = (cell_hits / len(cells)) if cells else 1.0
        return {
            "highlighted_cell_coverage": round(cell_cov * 100.0, 2),
            "triple_coverage": round(cell_cov * 100.0, 2),
        }
    else:
        return {"triple_coverage": 0.0}


class ModelEvaluator:
    """Loads a fine-tuned LoRA checkpoint and generates D2T predictions."""

    def __init__(
        self,
        base_model_path: str = "models/qwen3_4b_hf",
        lora_adapter_path: Optional[str] = None,
        device: str = "cuda",
        max_seq_len: int = 256,
        max_new_tokens: int = 128,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.serializer = Serializer()

        # Load Tokenizer
        tokenizer_src = lora_adapter_path if (lora_adapter_path and os.path.exists(lora_adapter_path)) else base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Base Model in FP16 (or FP32 on CPU)
        print(f"[ModelEvaluator] Loading base model '{base_model_path}'...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation="eager",
        )

        if lora_adapter_path and os.path.exists(lora_adapter_path):
            print(f"[ModelEvaluator] Attaching LoRA adapter from '{lora_adapter_path}'...")
            self.model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        else:
            self.model = base_model

        self.model.eval()
        self.metric_engine = MetricEvaluator()

    def generate_single(self, record: UnifiedRecord) -> str:
        """Formats structured record into prompt and generates response."""
        serialized_text = self.serializer.serialize(record)
        prompt = f"<|im_start|>user\n{serialized_text}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_len).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0, input_len:]
        decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return decoded

    def evaluate_records(self, records: List[UnifiedRecord]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Runs batch evaluation over records, computing NLP and structured metrics."""
        predictions = []
        references_list = []
        detailed_samples = []
        coverage_scores: List[Dict[str, float]] = []

        for rec in records:
            pred_text = self.generate_single(rec)
            target = rec.target_text.strip()
            refs = [r.strip() for r in rec.reference_texts if r.strip()] or [target]

            predictions.append(pred_text)
            references_list.append(refs)

            cov = compute_structured_coverage(rec, pred_text)
            coverage_scores.append(cov)

            detailed_samples.append({
                "id": rec.id,
                "dataset": rec.dataset,
                "target": target,
                "references": refs,
                "prediction": pred_text,
                "coverage": cov,
            })

        # Compute standard text generation metrics
        metrics = self.metric_engine.evaluate_batch(predictions, references_list)

        # Average structured coverage
        if coverage_scores:
            avg_triple_cov = sum(c.get("triple_coverage", 0.0) for c in coverage_scores) / len(coverage_scores)
            metrics["triple_coverage"] = round(avg_triple_cov, 2)
            
            ent_covs = [c["entity_coverage"] for c in coverage_scores if "entity_coverage" in c]
            if ent_covs:
                metrics["entity_coverage"] = round(sum(ent_covs) / len(ent_covs), 2)
                
            rel_covs = [c["relation_coverage"] for c in coverage_scores if "relation_coverage" in c]
            if rel_covs:
                metrics["relation_coverage"] = round(sum(rel_covs) / len(rel_covs), 2)

            cell_covs = [c["highlighted_cell_coverage"] for c in coverage_scores if "highlighted_cell_coverage" in c]
            if cell_covs:
                metrics["cell_coverage"] = round(sum(cell_covs) / len(cell_covs), 2)

        return metrics, detailed_samples


def compare_baseline_vs_psd(
    baseline_adapter_dir: str = "models/phase2a/baseline",
    psd_adapter_dir: str = "models/phase2a/psd",
    base_model_path: str = "models/qwen3_4b_hf",
    datasets: List[str] = ["totto", "dart", "webnlg"],
    split: str = "validation",
    max_eval_samples: int = 15,
    output_json: str = "reports/phase2a/metrics_comparison.json",
    output_csv: str = "reports/phase2a/metrics_comparison.csv",
) -> Dict[str, Any]:
    """Runs rigorous controlled comparison between Baseline and PSD on exact same test records."""
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 1. Load evaluation records per dataset
    eval_records_by_dataset: Dict[str, List[UnifiedRecord]] = {}
    for ds in datasets:
        recs = load_dataset_records(dataset_name=ds, split=split, max_records=max_eval_samples)
        if recs:
            eval_records_by_dataset[ds] = recs

    comparison_results: Dict[str, Any] = {
        "datasets": datasets,
        "split": split,
        "max_samples_per_dataset": max_eval_samples,
        "by_dataset": {},
        "overall": {},
    }

    csv_rows = []

    # 2. Evaluate Baseline
    print("\n" + "=" * 60)
    print("Evaluating BASELINE Model...")
    print("=" * 60)
    baseline_evaluator = ModelEvaluator(base_model_path=base_model_path, lora_adapter_path=baseline_adapter_dir)

    baseline_metrics_by_ds = {}
    baseline_samples_by_ds = {}
    for ds, recs in eval_records_by_dataset.items():
        print(f"  Running Baseline on {ds.upper()} ({len(recs)} examples)...")
        m, s = baseline_evaluator.evaluate_records(recs)
        baseline_metrics_by_ds[ds] = m
        baseline_samples_by_ds[ds] = s

    # Clean GPU memory before loading PSD
    del baseline_evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Evaluate PSD
    print("\n" + "=" * 60)
    print("Evaluating PSD Model...")
    print("=" * 60)
    psd_evaluator = ModelEvaluator(base_model_path=base_model_path, lora_adapter_path=psd_adapter_dir)

    psd_metrics_by_ds = {}
    psd_samples_by_ds = {}
    for ds, recs in eval_records_by_dataset.items():
        print(f"  Running PSD on {ds.upper()} ({len(recs)} examples)...")
        m, s = psd_evaluator.evaluate_records(recs)
        psd_metrics_by_ds[ds] = m
        psd_samples_by_ds[ds] = s

    del psd_evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Aggregate Comparisons
    all_metrics_keys = ["bleu", "rouge1", "rouge2", "rougeL", "triple_coverage"]
    
    for ds in eval_records_by_dataset:
        b_m = baseline_metrics_by_ds.get(ds, {})
        p_m = psd_metrics_by_ds.get(ds, {})

        comparison_results["by_dataset"][ds] = {
            "baseline": b_m,
            "psd": p_m,
            "delta": {k: round(p_m.get(k, 0.0) - b_m.get(k, 0.0), 2) for k in all_metrics_keys},
        }

        for metric_k in all_metrics_keys:
            csv_rows.append({
                "dataset": ds,
                "metric": metric_k,
                "baseline": b_m.get(metric_k, 0.0),
                "psd": p_m.get(metric_k, 0.0),
                "delta": round(p_m.get(metric_k, 0.0) - b_m.get(metric_k, 0.0), 2),
            })

    # Overall averages
    overall_baseline = {
        k: round(sum(baseline_metrics_by_ds[ds].get(k, 0.0) for ds in eval_records_by_dataset) / max(len(eval_records_by_dataset), 1), 2)
        for k in all_metrics_keys
    }
    overall_psd = {
        k: round(sum(psd_metrics_by_ds[ds].get(k, 0.0) for ds in eval_records_by_dataset) / max(len(eval_records_by_dataset), 1), 2)
        for k in all_metrics_keys
    }
    overall_delta = {k: round(overall_psd[k] - overall_baseline[k], 2) for k in all_metrics_keys}

    comparison_results["overall"] = {
        "baseline": overall_baseline,
        "psd": overall_psd,
        "delta": overall_delta,
    }

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    # Save CSV
    if csv_rows:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "metric", "baseline", "psd", "delta"])
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\n[Evaluation] Saved metrics comparison to {output_json} and {output_csv}")
    return comparison_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline vs PSD")
    parser.add_argument("--baseline_dir", type=str, default="models/phase2a/baseline")
    parser.add_argument("--psd_dir", type=str, default="models/phase2a/psd")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    compare_baseline_vs_psd(
        baseline_adapter_dir=args.baseline_dir,
        psd_adapter_dir=args.psd_dir,
        max_eval_samples=args.samples,
    )
