"""Phase 4 Long-Context Scaling Experiment Engine.

Topology-aware Positional Semantic Divergence (PSD) for Long-Context Data-to-Text Generation.
Evaluates Baseline vs Full PSD across sequence lengths L in [256, 512, 1024, 2048].
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
import torch.nn as nn
import torch.optim as optim
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from src.data.unified_schema import UnifiedRecord
from src.preprocessing.serialize import Serializer
from src.psd.graph_distance import GraphDistanceCalculator
from src.psd.target_distribution import TargetDistributionBuilder
from src.psd.divergence import PSDLoss
from src.psd.semantic_encoder import FrozenContextualSemanticEncoder
from src.baseline.evaluate_metrics import MetricEvaluator
from src.baseline.factuality import FactualityEvaluator

PHASE4_DIR = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase4_long_context")
CONFIGS_DIR = os.path.join(PHASE4_DIR, "configs")
PLOTS_DIR = os.path.join(PHASE4_DIR, "plots")
CONTEXT_LENGTHS = [256, 512, 1024, 2048]
CONTEXT_BUCKETS = ["<=256", "257-512", "513-1024", "1025-2048", ">2048"]


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
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


def load_unified_split(split: str = "validation") -> List[UnifiedRecord]:
    """Loads records from data/unified/{split}.jsonl."""
    path = os.path.join(PROJECT_ROOT, "data", "unified", f"{split}.jsonl")
    records = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    records.append(UnifiedRecord.from_dict(json.loads(line_str)))
    return records


def build_stratified_evaluation_benchmark(
    target_total: int = 300,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Builds a deterministic stratified benchmark from validation data.
    
    Stratification:
    - ToTTo: 100 instances evenly balanced across the 5 context length buckets (20 per bucket).
    - DART: 100 instances stratified across graph sizes / triple counts.
    - WebNLG: 100 instances stratified across 1-7 triple categories.
    """
    os.makedirs(PHASE4_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(PHASE4_DIR, "evaluation_set_300.json")

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if len(data) == target_total:
                return data

    print(f"[BenchmarkBuilder] Assembling stratified {target_total}-instance benchmark...")
    val_records = load_unified_split("validation")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(PROJECT_ROOT, "models", "qwen3_4b_hf"))
    serializer = Serializer()

    totto_by_bucket: Dict[str, List[Tuple[UnifiedRecord, int, str]]] = {b: [] for b in CONTEXT_BUCKETS}
    dart_records: List[Tuple[UnifiedRecord, int, str]] = []
    webnlg_records: List[Tuple[UnifiedRecord, int, str]] = []

    for rec in val_records:
        serialized = serializer.serialize_record(rec)
        tok_len = len(tokenizer(serialized, add_special_tokens=False)["input_ids"])
        b = assign_context_bucket(tok_len)

        if rec.dataset == "totto":
            totto_by_bucket[b].append((rec, tok_len, serialized))
        elif rec.dataset == "dart":
            dart_records.append((rec, tok_len, serialized))
        elif rec.dataset == "webnlg":
            webnlg_records.append((rec, tok_len, serialized))

    # Sample deterministically with fixed seed
    rng = random.Random(42)
    selected_items: List[Dict[str, Any]] = []

    # 1. ToTTo: target 100 (20 per bucket)
    per_bucket_target = target_total // 15  # 20 if target=300
    for b in CONTEXT_BUCKETS:
        pool = totto_by_bucket[b]
        rng.shuffle(pool)
        chosen = pool[:per_bucket_target]
        for rec, tok_len, ser in chosen:
            selected_items.append({
                "id": rec.id,
                "dataset": rec.dataset,
                "input_type": rec.input_type,
                "split": rec.split,
                "original_token_length": tok_len,
                "context_bucket": b,
                "target_text": rec.target_text,
                "reference_texts": rec.reference_texts,
                "serialized_text": ser,
                "nodes": rec.nodes,
                "edges": rec.edges,
            })

    # 2. DART: target 100
    dart_target = target_total // 3  # 100 if target=300
    dart_records.sort(key=lambda x: len(x[0].edges))
    stride_dart = max(1, len(dart_records) // dart_target)
    chosen_dart = [dart_records[i * stride_dart] for i in range(min(dart_target, len(dart_records)))]
    for rec, tok_len, ser in chosen_dart:
        selected_items.append({
            "id": rec.id,
            "dataset": rec.dataset,
            "input_type": rec.input_type,
            "split": rec.split,
            "original_token_length": tok_len,
            "context_bucket": assign_context_bucket(tok_len),
            "target_text": rec.target_text,
            "reference_texts": rec.reference_texts,
            "serialized_text": ser,
            "nodes": rec.nodes,
            "edges": rec.edges,
        })

    # 3. WebNLG: target 100
    webnlg_target = target_total // 3  # 100 if target=300
    webnlg_records.sort(key=lambda x: len(x[0].edges))
    stride_webnlg = max(1, len(webnlg_records) // webnlg_target)
    chosen_webnlg = [webnlg_records[i * stride_webnlg] for i in range(min(webnlg_target, len(webnlg_records)))]
    for rec, tok_len, ser in chosen_webnlg:
        selected_items.append({
            "id": rec.id,
            "dataset": rec.dataset,
            "input_type": rec.input_type,
            "split": rec.split,
            "original_token_length": tok_len,
            "context_bucket": assign_context_bucket(tok_len),
            "target_text": rec.target_text,
            "reference_texts": rec.reference_texts,
            "serialized_text": ser,
            "nodes": rec.nodes,
            "edges": rec.edges,
        })

    # Save 300-example benchmark
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_items, f, indent=2)
    print(f"Saved {len(selected_items)} benchmark items to {output_path}")

    # Also save a 150-example nested benchmark (first 50 of each dataset)
    sub_150_path = os.path.join(PHASE4_DIR, "evaluation_set_150.json")
    totto_sub = [it for it in selected_items if it["dataset"] == "totto"][:50]
    dart_sub = [it for it in selected_items if it["dataset"] == "dart"][:50]
    webnlg_sub = [it for it in selected_items if it["dataset"] == "webnlg"][:50]
    sub_150 = totto_sub + dart_sub + webnlg_sub
    with open(sub_150_path, "w", encoding="utf-8") as f:
        json.dump(sub_150, f, indent=2)
    print(f"Saved {len(sub_150)} nested benchmark items to {sub_150_path}")

    return selected_items


def format_example_for_scaling(
    record_dict_or_obj: Any,
    serializer: Serializer,
    tokenizer: AutoTokenizer,
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[str]], Dict[str, Any]]:
    """Formats training/evaluation example strictly tracking truncation details."""
    if isinstance(record_dict_or_obj, dict):
        rec = UnifiedRecord.from_dict(record_dict_or_obj)
    else:
        rec = record_dict_or_obj

    target_text = rec.target_text.strip()
    if not target_text and rec.reference_texts:
        target_text = rec.reference_texts[0].strip()

    eos_token = "<|im_end|>"
    target_formatted = f"{target_text}{eos_token}"
    target_ids = tokenizer(target_formatted, add_special_tokens=False)["input_ids"]

    prompt_prefix = "<|im_start|>user\n"
    prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"
    prefix_ids = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(prompt_suffix, add_special_tokens=False)["input_ids"]

    mapped_data = serializer.serialize_with_mapping(rec, tokenizer=tokenizer)
    serialized_ids = mapped_data["input_ids"]
    serialized_node_map = mapped_data["token_to_node_map"]

    original_tokens_count = len(serialized_ids)
    overhead = len(prefix_ids) + len(suffix_ids) + len(target_ids)
    available_budget = max(max_seq_len - overhead, 16)

    # Truncate serialized tokens from the right if exceeding available budget
    effective_serialized_ids = serialized_ids[:available_budget]
    effective_node_map = serialized_node_map[:available_budget]

    truncation_amount = max(0, original_tokens_count - len(effective_serialized_ids))
    truncation_pct = round((truncation_amount / max(original_tokens_count, 1)) * 100.0, 2)

    full_input_ids = prefix_ids + effective_serialized_ids + suffix_ids + target_ids
    prompt_len = len(prefix_ids) + len(effective_serialized_ids) + len(suffix_ids)
    full_labels = [-100] * prompt_len + target_ids
    full_node_map = [None] * len(prefix_ids) + effective_node_map + [None] * (len(suffix_ids) + len(target_ids))

    input_tensor = torch.tensor([full_input_ids], dtype=torch.long)
    labels_tensor = torch.tensor([full_labels], dtype=torch.long)

    stats = {
        "id": rec.id,
        "dataset": rec.dataset,
        "context_bucket": assign_context_bucket(original_tokens_count),
        "original_token_length": original_tokens_count,
        "effective_token_length": len(effective_serialized_ids),
        "truncation_amount": truncation_amount,
        "truncation_pct": truncation_pct,
        "full_seq_length": len(full_input_ids),
    }

    return input_tensor, labels_tensor, full_node_map, stats


class Phase4PreflightProfiler:
    """Micro-profiling engine to measure peak VRAM, throughput, and autograd limits."""

    def __init__(self, model_id: str = "models/qwen3_4b_hf"):
        self.model_id = os.path.join(PROJECT_ROOT, model_id) if not os.path.isabs(model_id) else model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.serializer = Serializer()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def run_preflight_for_length(
        self,
        context_len: int,
        mode: str = "psd",
        num_steps: int = 10,
        enable_grad_ckpt: bool = False,
    ) -> Dict[str, Any]:
        """Runs a 10-step micro-profile measuring peak VRAM, step time, and throughput."""
        print(f"\n[Preflight] Testing {mode.upper()} at L={context_len} (Grad Checkpoint: {enable_grad_ckpt})...", flush=True)
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="eager",
        )

        if enable_grad_ckpt:
            base_model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        model.train()

        optimizer = optim.AdamW(model.parameters(), lr=2e-4)
        target_builder = TargetDistributionBuilder(alpha=1.0, beta=0.5, gamma=0.5, tau=1.0)
        dist_calc = GraphDistanceCalculator()
        psd_loss_fn = PSDLoss(selected_layers=[11, 23, 34])

        val_records = load_unified_split("validation")[:num_steps]

        step_times = []
        tokens_processed = 0
        oom_occurred = False
        oom_error = ""

        try:
            for step_idx in range(num_steps):
                rec = val_records[step_idx % len(val_records)]
                inp_t, lab_t, node_map, stats = format_example_for_scaling(
                    record_dict_or_obj=rec,
                    serializer=self.serializer,
                    tokenizer=self.tokenizer,
                    max_seq_len=context_len,
                )
                inp_t = inp_t.to(self.device)
                lab_t = lab_t.to(self.device)
                tokens_processed += inp_t.shape[1]

                t0 = time.time()
                outputs = model(
                    input_ids=inp_t,
                    labels=lab_t,
                    output_attentions=(mode == "psd"),
                )
                lm_loss = outputs.loss

                if mode == "psd" and outputs.attentions is not None:
                    dist_matrix, node_to_idx = dist_calc.compute_distance_matrix(rec)
                    P_target, valid_mask = target_builder.build_target_distribution(
                        seq_len=inp_t.shape[1],
                        token_to_node_map=node_map,
                        graph_dist_matrix=dist_matrix,
                        node_to_idx=node_to_idx,
                        device=self.device,
                    )
                    psd_l, _ = psd_loss_fn(outputs.attentions, P_target, valid_mask)
                    total_loss = lm_loss + 0.1 * psd_l
                else:
                    total_loss = lm_loss

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                step_times.append(time.time() - t0)

        except torch.cuda.OutOfMemoryError as e:
            oom_occurred = True
            oom_error = str(e)
            print(f"  [OOM ERROR] at L={context_len}: {e}", flush=True)
        except Exception as e:
            oom_occurred = True
            oom_error = str(e)
            print(f"  [ERROR] at L={context_len}: {e}", flush=True)

        peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        peak_res_mb = torch.cuda.max_memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        avg_step_sec = (sum(step_times) / len(step_times)) if step_times else 0.0
        tok_per_sec = (tokens_processed / sum(step_times)) if (step_times and sum(step_times) > 0) else 0.0

        print(f"  Peak Alloc: {peak_alloc_mb:.1f} MB | Peak Res: {peak_res_mb:.1f} MB | Step: {avg_step_sec:.3f}s | Throughput: {tok_per_sec:.1f} tok/s", flush=True)

        # Thorough cleanup
        del model
        del base_model
        del optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "context_length": context_len,
            "mode": mode,
            "gradient_checkpointing": enable_grad_ckpt,
            "oom": oom_occurred,
            "error": oom_error,
            "peak_allocated_mb": round(peak_alloc_mb, 2),
            "peak_reserved_mb": round(peak_res_mb, 2),
            "avg_step_sec": round(avg_step_sec, 4),
            "tokens_per_sec": round(tok_per_sec, 1),
            "safe_under_8gb": (not oom_occurred and peak_res_mb < 7800.0),
        }


def run_full_preflight_suite() -> Dict[str, Any]:
    """Executes preflight profiling across all context lengths [256, 512, 1024, 2048] for Baseline and PSD."""
    profiler = Phase4PreflightProfiler()
    results = []

    configs_to_test = [
        # (Length, Mode, GradCheckpoint)
        (256, "baseline", False),
        (256, "psd", False),
        (512, "baseline", False),
        (512, "psd", False),
        (1024, "baseline", True),
        (1024, "psd", True),
        (2048, "baseline", True),
        (2048, "psd", True),
    ]

    for clen, mode, gckpt in configs_to_test:
        res = profiler.run_preflight_for_length(
            context_len=clen,
            mode=mode,
            num_steps=10,
            enable_grad_ckpt=gckpt,
        )
        results.append(res)
        time.sleep(1)

    os.makedirs(PHASE4_DIR, exist_ok=True)
    out_json = os.path.join(PHASE4_DIR, "preflight_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Preflight] Complete results saved to {out_json}")
    return {"results": results, "saved_to": out_json}


def generate_phase4_yaml_configs() -> List[str]:
    """Generates the 8 YAML config files under reports/experiments/phase4_long_context/configs/."""
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    created_configs = []

    for mode in ["baseline", "psd"]:
        for clen in CONTEXT_LENGTHS:
            cfg_name = f"{mode}_{clen}.yaml"
            cfg_path = os.path.join(CONFIGS_DIR, cfg_name)
            use_psd = (mode == "psd")
            use_grad_ckpt = (clen >= 1024)

            cfg_dict = {
                "experiment": {
                    "name": f"phase4_{mode}_{clen}",
                    "type": mode,
                    "phase": "phase4_long_context",
                    "seed": 42,
                    "context_length": clen,
                    "gradient_checkpointing": use_grad_ckpt,
                },
                "model": {
                    "model_id": "models/qwen3_4b_hf",
                    "torch_dtype": "float16",
                    "attn_implementation": "eager",
                },
                "lora": {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "target_modules": ["q_proj", "v_proj"],
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                },
                "psd": {
                    "enabled": use_psd,
                    "lambda_psd": 0.1 if use_psd else 0.0,
                    "alpha": 1.0,
                    "beta": 0.5,
                    "gamma": 0.5,
                    "tau": 1.0,
                    "selected_layers": [11, 23, 34],
                    "enable_semantic": True,
                    "enable_positional": True,
                    "enable_graph": True,
                },
                "training": {
                    "max_seq_len": clen,
                    "max_steps": 150,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 4,
                    "learning_rate": 0.0002,
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                    "eval_interval": 50,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                },
                "output": {
                    "experiment_dir": f"reports/experiments/phase4_long_context/{mode}/context_{clen}",
                    "checkpoint_dir": f"reports/experiments/phase4_long_context/{mode}/context_{clen}/checkpoints",
                    "training_log_csv": f"reports/experiments/phase4_long_context/{mode}/context_{clen}/training_log.csv",
                    "metrics_json": f"reports/experiments/phase4_long_context/{mode}/context_{clen}/metrics.json",
                    "sample_predictions_json": f"reports/experiments/phase4_long_context/{mode}/context_{clen}/sample_predictions.json",
                    "factuality_results_json": f"reports/experiments/phase4_long_context/{mode}/context_{clen}/factuality_results.json",
                },
            }

            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
            created_configs.append(cfg_path)

    print(f"Generated {len(created_configs)} configuration files in {CONFIGS_DIR}")
    return created_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Long-Context Scaling Engine")
    parser.add_argument("--build_benchmark", action="store_true", help="Build stratified 300-example benchmark")
    parser.add_argument("--generate_configs", action="store_true", help="Generate all YAML config files")
    parser.add_argument("--preflight", action="store_true", help="Run full preflight memory profiling suite")
    args = parser.parse_args()

    if args.build_benchmark:
        build_stratified_evaluation_benchmark(target_total=300)

    if args.generate_configs:
        generate_phase4_yaml_configs()

    if args.preflight:
        run_full_preflight_suite()
