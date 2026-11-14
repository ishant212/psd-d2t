"""Long-Context Robustness and Scaling Evaluation Engine.

Evaluates existing Qwen3-4B Baseline vs Full PSD checkpoints across L in [256, 512, 1024, 2048]
on a balanced 60-example benchmark (20 ToTTo stratified across 5 length buckets, 20 DART, 20 WebNLG).
Computes BLEU-4, ROUGE-1/2/L, METEOR, BERTScore, Content Coverage, Factuality, Hallucination,
truncation tracking, and paired bootstrap statistical confidence intervals.
"""

import os
import sys
import gc
import json
import csv
import time
import random
import argparse
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.data.unified_schema import UnifiedRecord
from src.preprocessing.serialize import Serializer
from src.psd.graph_distance import GraphDistanceCalculator
from src.psd.target_distribution import TargetDistributionBuilder
from src.psd.divergence import PSDLoss
from src.baseline.evaluate_metrics import MetricEvaluator
from src.baseline.factuality import FactualityEvaluator

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "experiments", "final_long_context")
CONTEXT_LENGTHS = [256, 512, 1024, 2048]
CONTEXT_BUCKETS = ["<=256", "257-512", "513-1024", "1025-2048", ">2048"]

CHECKPOINTS = {
    "baseline": os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b", "baseline", "checkpoints", "final"),
    "full_psd": os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b", "psd", "checkpoints", "final"),
}


def set_seed(seed: int = 42) -> None:
    """Sets random seed."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assign_context_bucket(token_length: int) -> str:
    """Assigns token length to standardized context bucket."""
    if token_length <= 256:
        return "<=256"
    elif token_length <= 512:
        return "257-512"
    elif token_length <= 1024:
        return "513-1024"
    elif token_length <= 2048:
        return "1025-2048"
    else:
        return ">2048"


def build_evaluation_benchmark_60() -> List[Dict[str, Any]]:
    """Builds a deterministic 60-example benchmark from evaluation_set_300.json."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bench_60_path = os.path.join(OUTPUT_DIR, "evaluation_set_60.json")

    if os.path.exists(bench_60_path):
        with open(bench_60_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if len(data) == 60:
                return data

    source_300_path = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase4_long_context", "evaluation_set_300.json")
    assert os.path.exists(source_300_path), f"evaluation_set_300.json not found at {source_300_path}"
    with open(source_300_path, "r", encoding="utf-8") as f:
        items_300 = json.load(f)

    # Stratified selection: 20 ToTTo (4 per bucket), 20 DART, 20 WebNLG
    totto_by_bucket = {b: [] for b in CONTEXT_BUCKETS}
    dart_items = []
    webnlg_items = []

    for it in items_300:
        if it["dataset"] == "totto":
            totto_by_bucket[it["context_bucket"]].append(it)
        elif it["dataset"] == "dart":
            dart_items.append(it)
        elif it["dataset"] == "webnlg":
            webnlg_items.append(it)

    selected = []
    # 1. ToTTo: 4 per bucket = 20
    for b in CONTEXT_BUCKETS:
        selected.extend(totto_by_bucket[b][:4])

    # 2. DART: 20
    selected.extend(dart_items[:20])

    # 3. WebNLG: 20
    selected.extend(webnlg_items[:20])

    assert len(selected) == 60, f"Expected 60, got {len(selected)}"
    with open(bench_60_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)

    print(f"Created deterministic 60-instance benchmark at {bench_60_path}")
    return selected


def format_eval_prompt(
    item: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_context_len: int,
) -> Tuple[str, Dict[str, Any]]:
    """Formats prompt preserving target overhead and tracking truncation."""
    serialized = item["serialized_text"]
    prompt_prefix = "<|im_start|>user\n"
    prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"

    prefix_ids = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(prompt_suffix, add_special_tokens=False)["input_ids"]
    ser_ids = tokenizer(serialized, add_special_tokens=False)["input_ids"]

    orig_tok_len = len(ser_ids)
    overhead = len(prefix_ids) + len(suffix_ids)
    available_budget = max(max_context_len - overhead, 16)

    eff_ser_ids = ser_ids[:available_budget]
    eff_tok_len = len(eff_ser_ids)
    trunc_amt = max(0, orig_tok_len - eff_tok_len)
    trunc_pct = round((trunc_amt / max(orig_tok_len, 1)) * 100.0, 2)

    eff_serialized_text = tokenizer.decode(eff_ser_ids)
    full_prompt = f"{prompt_prefix}{eff_serialized_text}{prompt_suffix}"

    meta = {
        "example_id": item["id"],
        "dataset": item["dataset"],
        "input_type": item["input_type"],
        "original_token_length": orig_tok_len,
        "effective_token_length": eff_tok_len,
        "truncation_amount": trunc_amt,
        "truncation_pct": trunc_pct,
        "context_bucket": item.get("context_bucket", assign_context_bucket(orig_tok_len)),
    }
    return full_prompt, meta


def compute_paired_bootstrap_ci(
    base_scores: List[float],
    psd_scores: List[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Computes paired bootstrap confidence interval for delta = PSD - Baseline."""
    rng = np.random.RandomState(seed)
    diffs = np.array(psd_scores) - np.array(base_scores)
    n = len(diffs)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_val": 1.0}

    boot_diffs = []
    for _ in range(n_boot):
        indices = rng.choice(n, size=n, replace=True)
        boot_diffs.append(float(np.mean(diffs[indices])))

    ci_lower = float(np.percentile(boot_diffs, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha / 2.0)))
    mean_diff = float(np.mean(diffs))

    # Two-sided empirical p-value
    if mean_diff >= 0:
        p_val = 2.0 * float(np.mean(np.array(boot_diffs) <= 0.0))
    else:
        p_val = 2.0 * float(np.mean(np.array(boot_diffs) >= 0.0))
    p_val = min(1.0, max(0.0, p_val))

    return {
        "mean_diff": round(mean_diff, 3),
        "ci_lower": round(ci_lower, 3),
        "ci_upper": round(ci_upper, 3),
        "p_value": round(p_val, 4),
        "significant": (ci_lower > 0.0 or ci_upper < 0.0),
    }


class FinalLongContextEvaluator:
    """Evaluates Baseline vs Full PSD across context lengths on the 60-example benchmark."""

    def __init__(self, base_model_path: str = "models/qwen3_4b_hf"):
        self.base_model_path = os.path.join(PROJECT_ROOT, base_model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.metric_eval = MetricEvaluator()
        self.fact_eval = FactualityEvaluator()

    def evaluate_model_at_context(
        self,
        model_key: str,
        checkpoint_dir: str,
        eval_items: List[Dict[str, Any]],
        context_len: int,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Generates predictions and computes all metrics for one model at one context length."""
        display_name = "Baseline" if model_key == "baseline" else "Full PSD"
        print(f"\n[Evaluator] Loading {display_name} from: {checkpoint_dir} (L={context_len})...", flush=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="eager",
        )

        # Attach trained adapter
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        model.eval()

        predictions = []
        references_list = []
        instance_records = []
        gen_times = []

        print(f"  Generating predictions for {len(eval_items)} instances...", flush=True)
        with torch.no_grad():
            for idx, item in enumerate(eval_items):
                prompt, meta = format_eval_prompt(item, self.tokenizer, context_len)
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_len).to(self.device)
                input_len = inputs["input_ids"].shape[1]

                t0 = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                gen_time = time.time() - t0
                gen_times.append(gen_time)

                gen_tokens = outputs[0, input_len:]
                decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

                target = item.get("target_text", "").strip()
                refs = [r.strip() for r in item.get("reference_texts", []) if r.strip()] or [target]

                predictions.append(decoded)
                references_list.append(refs)

                inst_res = {
                    **meta,
                    "model": "Qwen3-4B",
                    "condition": display_name,
                    "context_length": context_len,
                    "target_text": target,
                    "reference_texts": refs,
                    "generated_text": decoded,
                    "gen_time_sec": round(gen_time, 2),
                }
                instance_records.append(inst_res)

                if (idx + 1) % 15 == 0 or (idx + 1) == len(eval_items):
                    print(f"    [{idx+1:2d}/{len(eval_items)}] Done. Avg gen time: {np.mean(gen_times):.2f}s", flush=True)

        # 1. Lexical & Semantic Metrics
        lex_metrics = self.metric_eval.evaluate_batch(predictions, references_list, compute_bertscore=True)

        # 2. Factuality, Hallucination, and Content Coverage
        fact_summary = self.fact_eval.evaluate_batch(eval_items, predictions)

        # Attach per-instance factuality & BLEU/METEOR
        for inst, p, r, f_inst in zip(instance_records, predictions, references_list, fact_summary.get("detailed_results", [])):
            inst["factual_precision"] = f_inst.get("factual_precision", 0.0)
            inst["hallucination_rate"] = f_inst.get("hallucination_rate", 0.0)
            inst["structured_coverage"] = f_inst.get("structured_coverage", 0.0)
            inst["supported_claims"] = f_inst.get("supported_claims", 0)
            inst["unsupported_claims"] = f_inst.get("unsupported_claims", 0)
            inst["meteor"] = self.metric_eval.compute_meteor_single(p, r)

        # Calculate Syntax Leak Rate (markup leakage)
        total_claims = fact_summary.get("total_claims", 0)
        syntax_leaks = sum(
            1 for f_inst in fact_summary.get("detailed_results", [])
            for c in f_inst.get("claims", [])
            if c.get("category") == "syntax_leak"
        )
        leak_rate = round((syntax_leaks / max(total_claims, 1)) * 100.0, 2)

        overall_metrics = {
            "model": "Qwen3-4B",
            "condition": display_name,
            "context_length": context_len,
            "bleu_4": lex_metrics.get("bleu", 0.0),
            "rouge_1": lex_metrics.get("rouge1", 0.0),
            "rouge_2": lex_metrics.get("rouge2", 0.0),
            "rouge_l": lex_metrics.get("rougeL", 0.0),
            "meteor": lex_metrics.get("meteor", 0.0),
            "bertscore_f1": lex_metrics.get("bertscore_f1", 0.0),
            "structured_coverage": fact_summary.get("structured_coverage", 0.0),
            "factual_precision": fact_summary.get("factual_precision", 0.0),
            "hallucination_rate": fact_summary.get("hallucination_rate", 0.0),
            "syntax_leak_rate": leak_rate,
            "total_claims": total_claims,
            "supported_claims": fact_summary.get("supported_claims", 0),
            "unsupported_claims": fact_summary.get("unsupported_claims", 0),
            "avg_gen_time_sec": round(float(np.mean(gen_times)), 2),
        }

        # Memory cleanup
        del model
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return overall_metrics, instance_records


def run_final_long_context_study() -> Dict[str, Any]:
    """Executes complete evaluation suite across all 4 context lengths."""
    set_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Build or load 60-example benchmark
    benchmark_items = build_evaluation_benchmark_60()

    # Pre-execution Printout Requirement
    print("\n" + "=" * 75)
    print("FINAL LONG-CONTEXT ROBUSTNESS & SCALING EVALUATION")
    print("=" * 75)
    print("Checkpoints:")
    print(f"  Baseline: {CHECKPOINTS['baseline']}")
    print(f"  Full PSD: {CHECKPOINTS['full_psd']}")
    print(f"\nEvaluation Benchmark:")
    print(f"  Total examples: {len(benchmark_items)}")
    print(f"  ToTTo:          {sum(1 for it in benchmark_items if it['dataset'] == 'totto')}")
    print(f"  DART:           {sum(1 for it in benchmark_items if it['dataset'] == 'dart')}")
    print(f"  WebNLG:         {sum(1 for it in benchmark_items if it['dataset'] == 'webnlg')}")
    print(f"\nToTTo Length Buckets:")
    for b in CONTEXT_BUCKETS:
        cnt = sum(1 for it in benchmark_items if it['dataset'] == 'totto' and it['context_bucket'] == b)
        print(f"  {b:>9s}: {cnt} instances")
    print(f"\nGenerations: 60 examples * 4 context lengths * 2 models = 480 generations")
    print(f"Estimated Runtime: ~1.2 - 1.8 hours on NVIDIA RTX 4060 GPU")
    print("=" * 75 + "\n", flush=True)

    evaluator = FinalLongContextEvaluator()
    all_summary_results: List[Dict[str, Any]] = []
    all_instance_results: List[Dict[str, Any]] = []

    # Run loop
    for clen in CONTEXT_LENGTHS:
        for m_key in ["baseline", "full_psd"]:
            ckpt = CHECKPOINTS[m_key]
            m_res, inst_res = evaluator.evaluate_model_at_context(
                model_key=m_key,
                checkpoint_dir=ckpt,
                eval_items=benchmark_items,
                context_len=clen,
            )
            all_summary_results.append(m_res)
            all_instance_results.extend(inst_res)

    # 2. Compute Statistical Bootstrap Analysis
    stat_analysis: Dict[str, Any] = {}
    metrics_to_bootstrap = ["bleu_4", "meteor", "factual_precision", "hallucination_rate", "structured_coverage"]

    for clen in CONTEXT_LENGTHS:
        base_insts = [it for it in all_instance_results if it["condition"] == "Baseline" and it["context_length"] == clen]
        psd_insts = [it for it in all_instance_results if it["condition"] == "Full PSD" and it["context_length"] == clen]

        stat_analysis[str(clen)] = {}
        for m_name in ["meteor", "factual_precision", "hallucination_rate", "structured_coverage"]:
            b_vals = [it.get(m_name, 0.0) for it in base_insts]
            p_vals = [it.get(m_name, 0.0) for it in psd_insts]
            boot_res = compute_paired_bootstrap_ci(b_vals, p_vals)
            stat_analysis[str(clen)][m_name] = boot_res

    # 3. Context Length Summary Table
    length_summary_rows = []
    for clen in CONTEXT_LENGTHS:
        b_m = next((r for r in all_summary_results if r["condition"] == "Baseline" and r["context_length"] == clen), {})
        p_m = next((r for r in all_summary_results if r["condition"] == "Full PSD" and r["context_length"] == clen), {})

        b_bleu = b_m.get("bleu_4", 0.0)
        p_bleu = p_m.get("bleu_4", 0.0)
        b_met = b_m.get("meteor", 0.0)
        p_met = p_m.get("meteor", 0.0)
        b_rl = b_m.get("rouge_l", 0.0)
        p_rl = p_m.get("rouge_l", 0.0)
        b_bs = b_m.get("bertscore_f1", 0.0)
        p_bs = p_m.get("bertscore_f1", 0.0)
        b_fp = b_m.get("factual_precision", 0.0)
        p_fp = p_m.get("factual_precision", 0.0)
        b_hr = b_m.get("hallucination_rate", 0.0)
        p_hr = p_m.get("hallucination_rate", 0.0)

        length_summary_rows.append({
            "context_length": clen,
            "baseline_bleu": b_bleu,
            "psd_bleu": p_bleu,
            "delta_bleu": round(p_bleu - b_bleu, 2),
            "rel_gain_bleu_pct": round(((p_bleu - b_bleu) / max(b_bleu, 0.01)) * 100.0, 1),
            "baseline_meteor": b_met,
            "psd_meteor": p_met,
            "delta_meteor": round(p_met - b_met, 2),
            "baseline_rouge_l": b_rl,
            "psd_rouge_l": p_rl,
            "delta_rouge_l": round(p_rl - b_rl, 2),
            "baseline_bertscore": b_bs,
            "psd_bertscore": p_bs,
            "delta_bertscore": round(p_bs - b_bs, 2),
            "baseline_factual_prec": b_fp,
            "psd_factual_prec": p_fp,
            "delta_factual_prec": round(p_fp - b_fp, 2),
            "baseline_halluc_rate": b_hr,
            "psd_halluc_rate": p_hr,
            "delta_halluc_rate": round(p_hr - b_hr, 2),
        })

    # 4. Context Bucket Summary Table (Performance as function of original length)
    bucket_summary_rows = []
    for clen in CONTEXT_LENGTHS:
        for b in CONTEXT_BUCKETS:
            b_sub = [it for it in all_instance_results if it["condition"] == "Baseline" and it["context_length"] == clen and it["context_bucket"] == b]
            p_sub = [it for it in all_instance_results if it["condition"] == "Full PSD" and it["context_length"] == clen and it["context_bucket"] == b]

            if b_sub and p_sub:
                b_met = round(float(np.mean([it["meteor"] for it in b_sub])), 2)
                p_met = round(float(np.mean([it["meteor"] for it in p_sub])), 2)
                b_fp = round(float(np.mean([it["factual_precision"] for it in b_sub])), 2)
                p_fp = round(float(np.mean([it["factual_precision"] for it in p_sub])), 2)
                b_hr = round(float(np.mean([it["hallucination_rate"] for it in b_sub])), 2)
                p_hr = round(float(np.mean([it["hallucination_rate"] for it in p_sub])), 2)
                mean_trunc = round(float(np.mean([it["truncation_amount"] for it in b_sub])), 1)

                bucket_summary_rows.append({
                    "context_length": clen,
                    "context_bucket": b,
                    "num_instances": len(b_sub),
                    "mean_truncation_tokens": mean_trunc,
                    "baseline_meteor": b_met,
                    "psd_meteor": p_met,
                    "delta_meteor": round(p_met - b_met, 2),
                    "baseline_factual_prec": b_fp,
                    "psd_factual_prec": p_fp,
                    "delta_factual_prec": round(p_fp - b_fp, 2),
                    "baseline_halluc_rate": b_hr,
                    "psd_halluc_rate": p_hr,
                    "delta_halluc_rate": round(p_hr - b_hr, 2),
                })

    # Save artifacts
    # 1. evaluation_results.csv & .json
    eval_csv_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    with open(eval_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summary_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_summary_results)

    eval_json_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(all_summary_results, f, indent=2)

    # 2. context_length_summary.csv
    len_summary_csv = os.path.join(OUTPUT_DIR, "context_length_summary.csv")
    with open(len_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(length_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(length_summary_rows)

    # 3. context_bucket_summary.csv
    bkt_summary_csv = os.path.join(OUTPUT_DIR, "context_bucket_summary.csv")
    with open(bkt_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bucket_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bucket_summary_rows)

    # 4. per_example_results.jsonl
    jsonl_path = os.path.join(OUTPUT_DIR, "per_example_results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for it in all_instance_results:
            f.write(json.dumps(it) + "\n")

    # 5. generated_predictions.json
    preds_json_path = os.path.join(OUTPUT_DIR, "generated_predictions.json")
    with open(preds_json_path, "w", encoding="utf-8") as f:
        json.dump(all_instance_results, f, indent=2)

    # 6. statistical_analysis.json
    stat_path = os.path.join(OUTPUT_DIR, "statistical_analysis.json")
    with open(stat_path, "w", encoding="utf-8") as f:
        json.dump(stat_analysis, f, indent=2)

    # 7. verification.json
    verification_path = os.path.join(OUTPUT_DIR, "verification.json")
    with open(verification_path, "w", encoding="utf-8") as f:
        json.dump({
            "status": "COMPLETED",
            "benchmark_size": len(benchmark_items),
            "context_lengths": CONTEXT_LENGTHS,
            "checkpoints": CHECKPOINTS,
            "statistical_bootstrap_resamples": 1000,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }, f, indent=2)

    # 8. Automatically generate all 7 publication plots
    try:
        from src.psd.plot_final_long_context import generate_all_plots
        generate_all_plots(OUTPUT_DIR)
    except Exception as e:
        print(f"[Warning] Automatic plot generation encountered error: {e}", flush=True)

    print("\n[Final Long Context] All artifacts written successfully to:", OUTPUT_DIR)
    return {
        "summary": all_summary_results,
        "length_summary": length_summary_rows,
        "bucket_summary": bucket_summary_rows,
        "statistical_analysis": stat_analysis,
    }


if __name__ == "__main__":
    run_final_long_context_study()
