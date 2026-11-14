"""Phase 4A: Long-Context Scaling Evaluation Engine.

Stress-tests trained Baseline and Full PSD checkpoints across context lengths:
L in [256, 512, 1024, 2048] tokens.

Calculates:
- BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1, Structured Content Coverage
- Truncation statistics: original_length, evaluated_length, truncated %
- Structural retention: node_retention, edge_retention, highlighted_cell_retention
- Forward-pass PSD KL divergence across context lengths
- Peak VRAM and runtime
"""

import os
import sys
import gc
import json
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.unified_schema import UnifiedRecord
from src.preprocessing.serialize import Serializer
from src.baseline.evaluate_metrics import MetricEvaluator
from src.psd.target_distribution import TargetDistributionBuilder
from src.psd.divergence import PSDLoss
from src.psd.trainer import load_dataset_records

PHASE4B_DIR = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase4b")
CONTEXT_LENGTHS = [256, 512, 1024, 2048]

CHECKPOINTS = {
    "baseline_512": os.path.join(PROJECT_ROOT, "reports", "experiments", "phase4b", "baseline_512", "checkpoints", "final"),
    "psd_512": os.path.join(PROJECT_ROOT, "reports", "experiments", "phase4b", "psd_512", "checkpoints", "final"),
}


def load_evaluation_instances() -> List[Dict[str, Any]]:
    """Loads the exact 15 fixed evaluation instances from Phase 2B."""
    eval_path = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b", "evaluation_ids.json")
    assert os.path.exists(eval_path), f"evaluation_ids.json missing at {eval_path}"
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_structural_retention(
    record_dict: Dict[str, Any],
    original_unified_record: Optional[UnifiedRecord],
    serializer: Serializer,
    tokenizer: AutoTokenizer,
    max_context_len: int,
) -> Dict[str, float]:
    """Calculates what percentage of nodes, edges, and highlighted elements remain in the context window."""
    if original_unified_record is None:
        if "structured_input" in record_dict:
            original_unified_record = UnifiedRecord.from_dict(record_dict)
        else:
            original_unified_record = UnifiedRecord(
                id=record_dict.get("id", ""),
                dataset=record_dict.get("dataset", ""),
                split=record_dict.get("split", "validation"),
                input_type=record_dict.get("input_type", "table"),
                structured_input={},
                target_text=record_dict.get("target_text", ""),
                reference_texts=record_dict.get("reference_texts", []),
                nodes=record_dict.get("nodes", []),
                edges=record_dict.get("edges", []),
            )

    mapping = serializer.serialize_with_mapping(original_unified_record, tokenizer=tokenizer)
    token_node_map = mapping.get("token_to_node_map", [])
    token_edge_map = mapping.get("token_to_edge_map", [])
    all_node_ids = set(n.get("id") for n in original_unified_record.nodes if n.get("id"))
    all_edge_ids = set(e.get("id") for e in original_unified_record.edges if e.get("id"))

    # Highlighted cells for ToTTo
    highlighted_nodes = set(
        n.get("id") for n in original_unified_record.nodes
        if n.get("attributes", {}).get("is_highlighted", False)
    )

    prefix_len = 3
    effective_budget = max(max_context_len - prefix_len - 6, 1)

    truncated_nodes_in_window = set(nid for nid in token_node_map[:effective_budget] if nid is not None)
    truncated_edges_in_window = set(eid for eid in token_edge_map[:effective_budget] if eid is not None)

    node_retention = (len(truncated_nodes_in_window.intersection(all_node_ids)) / max(len(all_node_ids), 1)) * 100.0 if all_node_ids else 100.0
    edge_retention = (len(truncated_edges_in_window.intersection(all_edge_ids)) / max(len(all_edge_ids), 1)) * 100.0 if all_edge_ids else 100.0

    if highlighted_nodes:
        hl_retention = (len(truncated_nodes_in_window.intersection(highlighted_nodes)) / len(highlighted_nodes)) * 100.0
    else:
        hl_retention = node_retention

    return {
        "node_retention_pct": round(node_retention, 2),
        "edge_retention_pct": round(edge_retention, 2),
        "highlighted_retention_pct": round(hl_retention, 2),
        "total_nodes": len(all_node_ids),
        "retained_nodes": len(truncated_nodes_in_window.intersection(all_node_ids)),
        "total_edges": len(all_edge_ids),
        "retained_edges": len(truncated_edges_in_window.intersection(all_edge_ids)),
    }


def load_model_and_tokenizer(model_type: str, checkpoint_path: str) -> Tuple[Any, AutoTokenizer]:
    """Loads base model with FP16 precision and attaches the LoRA adapter."""
    print(f"\n[ModelLoader] Loading {model_type.upper()} model from: {checkpoint_path}...")
    base_model_path = os.path.join(PROJECT_ROOT, "models", "qwen3_4b_hf")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
    )

    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    return model, tokenizer


def evaluate_model_at_context_length(
    model: Any,
    tokenizer: AutoTokenizer,
    eval_instances: List[Dict[str, Any]],
    record_map: Dict[str, UnifiedRecord],
    context_length: int,
    model_type: str,
    metric_evaluator: MetricEvaluator,
    serializer: Serializer,
    psd_builder: Optional[TargetDistributionBuilder] = None,
    psd_loss_fn: Optional[PSDLoss] = None,
) -> Dict[str, Any]:
    """Evaluates a single model at a specified context length across all 15 instances."""
    print(f"\n--- Evaluating [{model_type.upper()}] at Context Length: {context_length} tokens ---", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictions = []
    references_list = []
    instance_results = []
    coverage_list = []
    psd_losses = []
    layer_psds = {11: [], 23: [], 34: []}

    start_time = time.time()
    total_original_tokens = 0
    total_evaluated_tokens = 0
    total_truncated_count = 0
    oom_encountered = False

    for idx, inst in enumerate(eval_instances):
        rec_id = inst.get("id")
        serialized_text = inst.get("serialized_text", "")
        target = inst.get("target_text", "").strip()
        refs = [r.strip() for r in inst.get("reference_texts", []) if r.strip()] or [target]
        dataset = inst.get("dataset", "unknown")
        orig_rec = record_map.get(rec_id)

        # 1. Measure original full prompt length
        full_prompt = f"<|im_start|>user\n{serialized_text}<|im_end|>\n<|im_start|>assistant\n"
        full_tokens = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        orig_len = full_tokens.shape[1]
        total_original_tokens += orig_len

        # 2. Tokenize with evaluation-time truncation to context_length
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=context_length,
            add_special_tokens=False,
        ).to(device)

        eval_len = inputs["input_ids"].shape[1]
        total_evaluated_tokens += eval_len
        is_truncated = orig_len > eval_len
        if is_truncated:
            total_truncated_count += 1

        # 3. Calculate structural retention in this context window
        retention = calculate_structural_retention(
            record_dict=inst,
            original_unified_record=orig_rec,
            serializer=serializer,
            tokenizer=tokenizer,
            max_context_len=context_length,
        )

        # 4. Generate prediction
        decoded_pred = ""
        try:
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                gen_tokens = outputs[0, eval_len:]
                decoded_pred = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        except torch.cuda.OutOfMemoryError:
            oom_encountered = True
            print(f"  [OOM Warning] OutOfMemoryError on example {rec_id} at length {context_length}!")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            decoded_pred = "<OOM_ERROR>"

        predictions.append(decoded_pred)
        references_list.append(refs)

        # 5. Content coverage
        cov_dict = metric_evaluator.compute_structured_content_coverage(orig_rec or inst, decoded_pred)
        cov_score = cov_dict.get("structured_coverage", 0.0)
        coverage_list.append(cov_score)

        # 6. Forward-pass PSD loss computation for Full PSD if feasible
        inst_psd_loss = None
        inst_layer_psd = {}
        if model_type == "full_psd" and psd_builder and psd_loss_fn and not oom_encountered:
            try:
                with torch.inference_mode():
                    fwd_out = model(
                        **inputs,
                        output_attentions=True,
                        return_dict=True,
                    )
                    attns = fwd_out.attentions  # Tuple of [batch, heads, seq, seq]
                    if attns and orig_rec:
                        mapping = serializer.serialize_with_mapping(orig_rec, tokenizer=tokenizer)
                        prefix_node_map = [None] * 3 + mapping["token_to_node_map"][:eval_len - 3]
                        
                        P_target = psd_builder.build_target_distribution(
                            record=orig_rec,
                            token_to_node_map=prefix_node_map,
                            seq_len=eval_len,
                            device=device,
                            dtype=torch.float16,
                        )

                        valid_mask = torch.tensor([n is not None for n in prefix_node_map], dtype=torch.bool, device=device)
                        psd_val, l_dict = psd_loss_fn(
                            attentions=attns,
                            P_target=P_target,
                            valid_mask=valid_mask,
                        )
                        inst_psd_loss = round(psd_val.item(), 4)
                        inst_layer_psd = {k: round(v, 4) for k, v in l_dict.items()}
                        psd_losses.append(inst_psd_loss)
                        for lk, lv in l_dict.items():
                            layer_psds[lk].append(lv)
            except Exception as e:
                inst_psd_loss = None

        instance_results.append({
            "id": rec_id,
            "dataset": dataset,
            "original_length": orig_len,
            "evaluated_length": eval_len,
            "truncated": is_truncated,
            "structural_retention": retention,
            "prediction": decoded_pred,
            "target": target,
            "references": refs,
            "structured_coverage": cov_score,
            "psd_loss": inst_psd_loss,
            "layer_psd": inst_layer_psd,
        })

    elapsed_time = round(time.time() - start_time, 2)
    alloc_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    res_mb = torch.cuda.memory_reserved() / (1024**2) if torch.cuda.is_available() else 0.0

    robust_preds = [p if p.strip() and p != "<OOM_ERROR>" else "." for p in predictions]
    metrics = metric_evaluator.evaluate_batch(robust_preds, references_list, compute_bertscore=True)
    if coverage_list:
        metrics["structured_coverage"] = round(sum(coverage_list) / len(coverage_list), 2)

    avg_node_ret = round(sum(r["structural_retention"]["node_retention_pct"] for r in instance_results) / len(instance_results), 2)
    avg_edge_ret = round(sum(r["structural_retention"]["edge_retention_pct"] for r in instance_results) / len(instance_results), 2)
    avg_hl_ret = round(sum(r["structural_retention"]["highlighted_retention_pct"] for r in instance_results) / len(instance_results), 2)

    mean_psd = round(sum(psd_losses) / len(psd_losses), 4) if psd_losses else "NA"
    mean_l12 = round(sum(layer_psds[11]) / len(layer_psds[11]), 4) if layer_psds[11] else "NA"
    mean_l24 = round(sum(layer_psds[23]) / len(layer_psds[23]), 4) if layer_psds[23] else "NA"
    mean_l35 = round(sum(layer_psds[34]) / len(layer_psds[34]), 4) if layer_psds[34] else "NA"

    ds_breakdowns = {}
    for ds_name in ["totto", "dart", "webnlg"]:
        ds_indices = [i for i, r in enumerate(eval_instances) if r.get("dataset") == ds_name]
        if ds_indices:
            ds_preds = [robust_preds[i] for i in ds_indices]
            ds_refs = [references_list[i] for i in ds_indices]
            ds_m = metric_evaluator.evaluate_batch(ds_preds, ds_refs, compute_bertscore=True)
            ds_cov = [coverage_list[i] for i in ds_indices]
            ds_m["structured_coverage"] = round(sum(ds_cov) / len(ds_cov), 2) if ds_cov else 0.0
            ds_breakdowns[ds_name] = ds_m

    output_payload = {
        "model_type": model_type,
        "context_length": context_length,
        "num_evaluated": len(eval_instances),
        "mean_original_length": round(total_original_tokens / len(eval_instances), 2),
        "mean_evaluated_length": round(total_evaluated_tokens / len(eval_instances), 2),
        "truncation_percentage": round((total_truncated_count / len(eval_instances)) * 100.0, 2),
        "structural_retention": {
            "mean_node_retention_pct": avg_node_ret,
            "mean_edge_retention_pct": avg_edge_ret,
            "mean_highlighted_retention_pct": avg_hl_ret,
        },
        "metrics": metrics,
        "dataset_breakdowns": ds_breakdowns,
        "psd_divergence": {
            "mean_psd_loss": mean_psd,
            "layer_12_psd": mean_l12,
            "layer_24_psd": mean_l24,
            "layer_35_psd": mean_l35,
        },
        "system": {
            "elapsed_seconds": elapsed_time,
            "gpu_allocated_mb": round(alloc_mb, 2),
            "gpu_reserved_mb": round(res_mb, 2),
            "oom_encountered": oom_encountered,
        },
        "instances": instance_results,
    }

    print(
        f"  [{model_type.upper()} @ L={context_length:4d}] "
        f"BLEU: {metrics['bleu']:5.2f} | ROUGE-L: {metrics['rougeL']:5.2f} | BERTScore: {metrics['bertscore_f1']:5.2f}% | "
        f"Cov: {metrics['structured_coverage']:5.1f}% | Ret(HL): {avg_hl_ret:5.1f}% | VRAM: {alloc_mb:.1f}MB",
        flush=True,
    )

    return output_payload


def run_phase4b_evaluation_pipeline():
    """Main orchestrator for Phase 4B Long-Context Scaling Evaluation."""
    print("=" * 80)
    print("STARTING PHASE 4B: LONG-CONTEXT TRAINING SCALING EVALUATION")
    print("=" * 80, flush=True)

    os.makedirs(PHASE4B_DIR, exist_ok=True)
    os.makedirs(os.path.join(PHASE4B_DIR, "baseline_512"), exist_ok=True)
    os.makedirs(os.path.join(PHASE4B_DIR, "psd_512"), exist_ok=True)

    eval_instances = load_evaluation_instances()
    print(f"[Phase4B] Loaded {len(eval_instances)} fixed evaluation instances.")

    # Load validation records across all three datasets to populate record_map
    val_records = []
    for ds in ["totto", "dart", "webnlg"]:
        recs = load_dataset_records(dataset_name=ds, split="validation")
        val_records.extend(recs)
        print(f"[Phase4B] Loaded {len(recs)} validation records from {ds}.")

    record_map = {r.id: r for r in val_records}

    serializer = Serializer()
    metric_evaluator = MetricEvaluator()

    # PSD tools
    psd_builder = TargetDistributionBuilder(alpha=1.0, beta=0.5, gamma=0.5, tau=1.0)
    psd_loss_fn = PSDLoss(selected_layers=[11, 23, 34])

    # 1. Evaluate Baseline 512 Model
    baseline_results = {}
    base_model, base_tok = load_model_and_tokenizer("baseline_512", CHECKPOINTS["baseline_512"])
    for L in CONTEXT_LENGTHS:
        res = evaluate_model_at_context_length(
            model=base_model,
            tokenizer=base_tok,
            eval_instances=eval_instances,
            record_map=record_map,
            context_length=L,
            model_type="baseline_512",
            metric_evaluator=metric_evaluator,
            serializer=serializer,
        )
        baseline_results[L] = res
        save_path = os.path.join(PHASE4B_DIR, "baseline_512", f"context_{L}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

    del base_model
    del base_tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 2. Evaluate Full PSD 512 Model
    psd_results = {}
    psd_model, psd_tok = load_model_and_tokenizer("psd_512", CHECKPOINTS["psd_512"])
    for L in CONTEXT_LENGTHS:
        res = evaluate_model_at_context_length(
            model=psd_model,
            tokenizer=psd_tok,
            eval_instances=eval_instances,
            record_map=record_map,
            context_length=L,
            model_type="psd_512",
            metric_evaluator=metric_evaluator,
            serializer=serializer,
            psd_builder=psd_builder,
            psd_loss_fn=psd_loss_fn,
        )
        psd_results[L] = res
        save_path = os.path.join(PHASE4B_DIR, "psd_512", f"context_{L}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

    del psd_model
    del psd_tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n[Phase4B] Raw evaluation runs complete! Ready for analysis.", flush=True)


if __name__ == "__main__":
    run_phase4b_evaluation_pipeline()

