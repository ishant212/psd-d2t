"""Controlled Training Runner for Baseline and PSD Experiments.

Phase 2A - Experiment A (Baseline) and Experiment B (PSD).
"""

import os
import sys
import json
import csv
import shutil
import argparse
import random
import time
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
import yaml
from transformers import AutoTokenizer
from src.data.unified_schema import UnifiedRecord
from src.preprocessing.serialize import Serializer
from src.psd.model import PSDModelWrapper, PSDOutput
from src.baseline.evaluate_metrics import MetricEvaluator


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_records(
    dataset_name: Optional[str] = None,
    split: str = "train",
    max_records: Optional[int] = None,
) -> List[UnifiedRecord]:
    """Loads unified records from unified split or per-dataset split."""
    if dataset_name:
        path = os.path.join(PROJECT_ROOT, "data", "processed", dataset_name, f"{split}.jsonl")
        if not os.path.exists(path):
            path = os.path.join(PROJECT_ROOT, "data", "unified", f"{dataset_name}.jsonl")
    else:
        path = os.path.join(PROJECT_ROOT, "data", "unified", f"{split}.jsonl")
        if not os.path.exists(path):
            path = os.path.join(PROJECT_ROOT, "data", "processed", "totto", f"{split}.jsonl")

    records = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_records is not None and idx >= max_records:
                    break
                line_str = line.strip()
                if line_str:
                    records.append(UnifiedRecord.from_dict(json.loads(line_str)))
    return records


def format_d2t_training_example(
    record: UnifiedRecord,
    serializer: Serializer,
    tokenizer: AutoTokenizer,
    max_seq_len: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[str]]]:
    """Prepares structured D2T training example ensuring:
    1. Causal LM loss is evaluated ONLY on the target reference tokens (prompt masked to -100).
    2. Target/reference text is ALWAYS preserved (truncation only applies to the prompt).
    3. Token-to-node mapping is properly preserved for PSD computation.

    Returns:
        input_ids_tensor: [1, seq_len] tensor of token IDs.
        labels_tensor: [1, seq_len] tensor with prompt tokens masked to -100 and target tokens preserved.
        token_to_node_map: List of length seq_len with node IDs for prompt tokens and None for target/special tokens.
    """
    # 1. Target / Reference text
    target_text = record.target_text.strip()
    if not target_text and record.reference_texts:
        target_text = record.reference_texts[0].strip()

    # Format target with end-of-turn token
    target_formatted = f"{target_text}<|im_end|>"
    target_encoding = tokenizer(target_formatted, add_special_tokens=False)
    target_ids = target_encoding["input_ids"]

    # 2. Prompt Header and Footer
    prompt_prefix = "<|im_start|>user\n"
    prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"
    prefix_ids = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(prompt_suffix, add_special_tokens=False)["input_ids"]

    overhead_tokens = len(prefix_ids) + len(suffix_ids) + len(target_ids)

    # 3. Available budget for serialized structured text (target is never removed)
    max_serialized_tokens = max(max_seq_len - overhead_tokens, 16)

    # 4. Serialize with token mapping
    mapped_data = serializer.serialize_with_mapping(record, tokenizer=tokenizer)
    serialized_ids = mapped_data["input_ids"]
    serialized_node_map = mapped_data["token_to_node_map"]

    # Truncate prompt from the right if necessary to preserve 100% of target reference
    truncated_serialized_ids = serialized_ids[:max_serialized_tokens]
    truncated_node_map = serialized_node_map[:max_serialized_tokens]

    # 5. Concatenate full sequence: prefix + serialized_prompt + suffix + target
    full_input_ids = prefix_ids + truncated_serialized_ids + suffix_ids + target_ids
    prompt_len = len(prefix_ids) + len(truncated_serialized_ids) + len(suffix_ids)

    # 6. Labels: Mask all prompt tokens with -100 so LM loss is strictly computed over target tokens
    full_labels = [-100] * prompt_len + target_ids

    # 7. Token-to-node map: None for prefix/suffix/target, preserved for serialized tokens
    full_node_map = [None] * len(prefix_ids) + truncated_node_map + [None] * (len(suffix_ids) + len(target_ids))

    input_tensor = torch.tensor([full_input_ids], dtype=torch.long)
    labels_tensor = torch.tensor([full_labels], dtype=torch.long)

    return input_tensor, labels_tensor, full_node_map


class ExperimentTrainer:
    """Unified trainer for Phase 2A Baseline and PSD experiments."""

    def __init__(
        self,
        config_path: str,
        model_wrapper: Optional[PSDModelWrapper] = None,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path
        self.exp_name = self.config.get("experiment", {}).get("name", "experiment")
        self.exp_type = self.config.get("experiment", {}).get("type", "psd")
        self.seed = self.config.get("experiment", {}).get("seed", 42)
        set_seed(self.seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_seq_len = self.config.get("training", {}).get("max_seq_len", 256)
        self.batch_size = self.config.get("training", {}).get("batch_size", 1)
        self.grad_accum_steps = self.config.get("training", {}).get("gradient_accumulation_steps", 4)
        self.lr = float(self.config.get("training", {}).get("learning_rate", 2e-4))
        self.weight_decay = float(self.config.get("training", {}).get("weight_decay", 0.01))
        self.max_grad_norm = float(self.config.get("training", {}).get("max_grad_norm", 1.0))

        psd_cfg = self.config.get("psd", {})
        self.psd_enabled = psd_cfg.get("enabled", True) and (self.exp_type == "psd")
        self.lambda_psd = float(psd_cfg.get("lambda_psd", 0.1)) if self.psd_enabled else 0.0
        self.selected_layers = psd_cfg.get("selected_layers", [11, 23, 34])

        if model_wrapper is not None:
            self.model_wrapper = model_wrapper
        else:
            model_id = self.config["model"]["model_id"]
            self.model_wrapper = PSDModelWrapper(
                model_name_or_path=model_id,
                selected_layers=self.selected_layers,
                lora_r=self.config["lora"]["r"],
                lora_alpha=self.config["lora"]["lora_alpha"],
                lora_dropout=self.config["lora"]["lora_dropout"],
                target_modules=self.config["lora"]["target_modules"],
                alpha=psd_cfg.get("alpha", 1.0),
                beta=psd_cfg.get("beta", 0.5),
                gamma=psd_cfg.get("gamma", 0.5),
                tau=psd_cfg.get("tau", 1.0),
                lambda_psd=self.lambda_psd,
                psd_enabled=self.psd_enabled,
                enable_semantic=psd_cfg.get("enable_semantic", True),
                enable_positional=psd_cfg.get("enable_positional", True),
                enable_graph=psd_cfg.get("enable_graph", True),
                device=self.device,
            )

        self.eval_interval = self.config.get("training", {}).get("eval_interval", 50)
        self.checkpoint_interval = self.config.get("training", {}).get("checkpoint_interval", 100)

        trainable_params = [p for p in self.model_wrapper.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        self.serializer = Serializer()
        self.metric_evaluator = MetricEvaluator()

    def evaluate_validation_loss(
        self,
        val_records: List[UnifiedRecord],
        max_val_samples: int = 5,
    ) -> Tuple[float, float, float]:
        """Evaluates validation LM loss, PSD loss, and total loss."""
        if not val_records:
            return 0.0, 0.0, 0.0

        self.model_wrapper.eval()
        total_val_lm = 0.0
        total_val_psd = 0.0
        n_samples = min(len(val_records), max_val_samples)

        with torch.no_grad():
            for i in range(n_samples):
                rec = val_records[i]
                inp_t, lab_t, node_map = format_d2t_training_example(
                    record=rec,
                    serializer=self.serializer,
                    tokenizer=self.model_wrapper.tokenizer,
                    max_seq_len=self.max_seq_len,
                )
                inp_t = inp_t.to(self.device)
                lab_t = lab_t.to(self.device)

                output = self.model_wrapper(
                    input_ids=inp_t,
                    labels=lab_t,
                    token_to_node_map=node_map,
                    record=rec,
                    max_seq_len=self.max_seq_len,
                )
                total_val_lm += output.lm_loss.item()
                total_val_psd += output.psd_loss.item()

        self.model_wrapper.train()
        avg_val_lm = round(total_val_lm / n_samples, 4)
        avg_val_psd = round(total_val_psd / n_samples, 4) if self.psd_enabled else 0.0
        avg_val_total = round(avg_val_lm + self.lambda_psd * avg_val_psd, 4)
        return avg_val_lm, avg_val_psd, avg_val_total

    def generate_predictions(
        self,
        eval_records: List[Any],
        max_samples: int = 15,
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Generates sample predictions and calculates standard & structural evaluation metrics."""
        self.model_wrapper.eval()
        samples_to_eval = eval_records[:max_samples]
        predictions = []
        references_list = []
        sample_outputs = []
        coverage_list = []

        with torch.no_grad():
            for rec in samples_to_eval:
                if isinstance(rec, dict):
                    serialized = rec.get("serialized_text", "")
                    target = rec.get("target_text", "").strip()
                    refs = [r.strip() for r in rec.get("reference_texts", []) if r.strip()] or [target]
                    rec_id = rec.get("id", "unknown")
                    ds = rec.get("dataset", "unknown")
                    inp_t = rec.get("input_type", "unknown")
                else:
                    serialized = self.serializer.serialize_record(rec)
                    target = rec.target_text.strip()
                    refs = [r.strip() for r in rec.reference_texts if r.strip()] or [target]
                    rec_id = rec.id
                    ds = rec.dataset
                    inp_t = rec.input_type

                prompt = f"<|im_start|>user\n{serialized}<|im_end|>\n<|im_start|>assistant\n"
                inputs = self.model_wrapper.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_len).to(self.device)
                input_len = inputs["input_ids"].shape[1]

                outputs = self.model_wrapper.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.model_wrapper.tokenizer.pad_token_id,
                    eos_token_id=self.model_wrapper.tokenizer.eos_token_id,
                )

                gen_tokens = outputs[0, input_len:]
                decoded = self.model_wrapper.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

                predictions.append(decoded)
                references_list.append(refs)

                cov_dict = self.metric_evaluator.compute_structured_content_coverage(rec, decoded)
                coverage_list.append(cov_dict.get("structured_coverage", 0.0))

                sample_outputs.append({
                    "id": rec_id,
                    "dataset": ds,
                    "input_type": inp_t,
                    "source": serialized,
                    "target": target,
                    "references": refs,
                    "prediction": decoded,
                    "structured_coverage": cov_dict.get("structured_coverage", 0.0),
                })

        metrics = self.metric_evaluator.evaluate_batch(predictions, references_list, compute_bertscore=True)
        if coverage_list:
            metrics["structured_coverage"] = round(sum(coverage_list) / len(coverage_list), 2)
        else:
            metrics["structured_coverage"] = 0.0

        self.model_wrapper.train()
        return metrics, sample_outputs

    def train_steps(
        self,
        train_records: List[UnifiedRecord],
        val_records: Optional[List[UnifiedRecord]] = None,
        fixed_eval_records: Optional[List[Any]] = None,
        num_steps: int = 500,
    ) -> Dict[str, Any]:
        """Executes a controlled training run with strict CSV/JSON logging, periodic eval, and checkpointing."""
        out_cfg = self.config.get("output", {})
        exp_dir = out_cfg.get("experiment_dir", f"reports/experiments/{self.exp_type}")
        ckpt_dir = out_cfg.get("checkpoint_dir", os.path.join(exp_dir, "checkpoints"))
        train_csv_path = out_cfg.get("training_log_csv", os.path.join(exp_dir, "training_log.csv"))
        val_csv_path = out_cfg.get("validation_log_csv", os.path.join(exp_dir, "validation_log.csv"))
        metrics_json_path = out_cfg.get("metrics_json", os.path.join(exp_dir, "metrics.json"))
        gen_metrics_json_path = out_cfg.get("generation_metrics_json", os.path.join(exp_dir, "generation_metrics.json"))
        preds_json_path = out_cfg.get("sample_predictions_json", os.path.join(exp_dir, "sample_predictions.json"))
        summary_md_path = out_cfg.get("summary_md", os.path.join(exp_dir, "experiment_summary.md"))

        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Copy config.yaml into experiment directory for strict reproducibility
        dest_config_path = os.path.join(exp_dir, "config.yaml")
        if os.path.exists(self.config_path):
            shutil.copyfile(self.config_path, dest_config_path)

        # Load fixed evaluation set if available
        if fixed_eval_records is None:
            eval_ids_path = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b", "evaluation_ids.json")
            if os.path.exists(eval_ids_path):
                with open(eval_ids_path, "r", encoding="utf-8") as f:
                    fixed_eval_records = json.load(f)
            else:
                fixed_eval_records = val_records or train_records[:15]

        print(f"\n[ExperimentTrainer: {self.exp_name.upper()}] Starting {num_steps} optimizer steps (exp_type={self.exp_type}, psd_enabled={self.psd_enabled})...", flush=True)

        self.model_wrapper.train()
        self.optimizer.zero_grad()

        training_logs = []
        validation_logs = []
        periodic_generation_logs = []
        rec_idx = 0
        total_records = len(train_records)
        val_records = val_records or train_records[:10]

        start_time = time.time()

        for step in range(1, num_steps + 1):
            accum_lm_loss = 0.0
            accum_psd_loss = 0.0
            accum_total_loss = 0.0
            accum_l12 = 0.0
            accum_l24 = 0.0
            accum_l35 = 0.0

            # Gradient accumulation loop
            for _ in range(self.grad_accum_steps):
                record = train_records[rec_idx % total_records]
                rec_idx += 1

                inp_t, lab_t, node_map = format_d2t_training_example(
                    record=record,
                    serializer=self.serializer,
                    tokenizer=self.model_wrapper.tokenizer,
                    max_seq_len=self.max_seq_len,
                )
                inp_t = inp_t.to(self.device)
                lab_t = lab_t.to(self.device)

                output = self.model_wrapper(
                    input_ids=inp_t,
                    labels=lab_t,
                    token_to_node_map=node_map,
                    record=record,
                    max_seq_len=self.max_seq_len,
                )

                loss = output.total_loss / self.grad_accum_steps
                loss.backward()

                accum_lm_loss += output.lm_loss.item() / self.grad_accum_steps
                accum_psd_loss += output.psd_loss.item() / self.grad_accum_steps
                accum_total_loss += output.total_loss.item() / self.grad_accum_steps

                if self.psd_enabled:
                    accum_l12 += output.per_layer_psd.get(11, 0.0) / self.grad_accum_steps
                    accum_l24 += output.per_layer_psd.get(23, 0.0) / self.grad_accum_steps
                    accum_l35 += output.per_layer_psd.get(34, 0.0) / self.grad_accum_steps

            # Optimizer Step & Gradient Clipping
            trainable_params = [p for p in self.model_wrapper.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.max_grad_norm).item()

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Fast validation loss check
            val_lm, val_psd, val_total = self.evaluate_validation_loss(val_records, max_val_samples=3)

            # Memory tracking
            alloc_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
            res_mb = torch.cuda.memory_reserved() / (1024**2) if torch.cuda.is_available() else 0.0
            elapsed_now = round(time.time() - start_time, 2)

            # Log row for training_log.csv
            t_log = {
                "step": step,
                "epoch": round(step / max(total_records // self.grad_accum_steps, 1), 4),
                "train_lm_loss": round(accum_lm_loss, 4),
                "train_psd_loss": round(accum_psd_loss, 4) if self.psd_enabled else "NA",
                "total_loss": round(accum_total_loss, 4),
                "validation_lm_loss": val_lm,
                "validation_psd_loss": val_psd if self.psd_enabled else "NA",
                "psd_layer_12": round(accum_l12, 4) if self.psd_enabled else "NA",
                "psd_layer_24": round(accum_l24, 4) if self.psd_enabled else "NA",
                "psd_layer_35": round(accum_l35, 4) if self.psd_enabled else "NA",
                "gradient_norm": round(grad_norm, 4),
                "learning_rate": self.lr,
                "gpu_allocated_mb": round(alloc_mb, 2),
                "gpu_reserved_mb": round(res_mb, 2),
                "elapsed_time": elapsed_now,
            }
            training_logs.append(t_log)

            # Log row for validation_log.csv
            v_log = {
                "step": step,
                "val_lm_loss": val_lm,
                "val_psd_loss": val_psd if self.psd_enabled else "NA",
                "val_total_loss": val_total,
                "elapsed_time": elapsed_now,
            }
            validation_logs.append(v_log)

            # Periodic Generation Evaluation (e.g. every 50 steps, step 1, and final step)
            is_eval_step = (step % self.eval_interval == 0) or (step == 1) or (step == num_steps)
            if is_eval_step:
                print(f"  [Step {step:03d}/{num_steps:03d}] Running periodic generation evaluation on fixed set...", flush=True)
                step_metrics, step_preds = self.generate_predictions(fixed_eval_records, max_samples=15)
                gen_entry = {
                    "step": step,
                    "elapsed_time": elapsed_now,
                    "val_lm_loss": val_lm,
                    "val_psd_loss": val_psd if self.psd_enabled else "NA",
                    "metrics": step_metrics,
                }
                periodic_generation_logs.append(gen_entry)
                with open(gen_metrics_json_path, "w", encoding="utf-8") as f:
                    json.dump(periodic_generation_logs, f, indent=2)

            # Periodic Checkpoint saving (e.g. every 100 steps or milestones 1, 50, 100, 250, 500)
            if (step % self.checkpoint_interval == 0) or (step == num_steps) or (step in [1, 50, 100, 250, 500]):
                step_ckpt_dir = os.path.join(ckpt_dir, f"step_{step}")
                self.save_checkpoint(step_ckpt_dir)

            if self.psd_enabled:
                print(
                    f"  Step {step:03d}/{num_steps:03d} | "
                    f"Total: {accum_total_loss:.4f} (LM: {accum_lm_loss:.4f}, PSD: {accum_psd_loss:.4f} [L12:{accum_l12:.3f}, L24:{accum_l24:.3f}, L35:{accum_l35:.3f}]) | "
                    f"Val LM: {val_lm:.4f}, Val PSD: {val_psd:.4f} | "
                    f"Grad: {grad_norm:.4f} | VRAM: {alloc_mb:.1f}MB",
                    flush=True,
                )
            else:
                print(
                    f"  Step {step:03d}/{num_steps:03d} | "
                    f"LM Loss: {accum_lm_loss:.4f} | Val LM: {val_lm:.4f} | "
                    f"Grad: {grad_norm:.4f} | VRAM: {alloc_mb:.1f}MB",
                    flush=True,
                )

        total_elapsed = round(time.time() - start_time, 2)
        trainable_p = sum(p.numel() for p in self.model_wrapper.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in self.model_wrapper.parameters())

        # 1. Save training_log.csv
        with open(train_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(training_logs[0].keys()))
            writer.writeheader()
            writer.writerows(training_logs)

        # 2. Save validation_log.csv
        with open(val_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(validation_logs[0].keys()))
            writer.writeheader()
            writer.writerows(validation_logs)

        # 3. Save Final LoRA Checkpoint
        self.save_checkpoint(ckpt_dir)
        final_ckpt_dir = os.path.join(ckpt_dir, "final")
        self.save_checkpoint(final_ckpt_dir)

        # 4. Final sample predictions & metrics on fixed set
        final_metrics, sample_preds = self.generate_predictions(fixed_eval_records, max_samples=15)
        
        metrics_payload = {
            "experiment_name": self.exp_name,
            "experiment_type": self.exp_type,
            "psd_enabled": self.psd_enabled,
            "lambda_psd": self.lambda_psd,
            "num_steps": num_steps,
            "elapsed_seconds": total_elapsed,
            "initial_lm_loss": training_logs[0]["train_lm_loss"],
            "final_lm_loss": training_logs[-1]["train_lm_loss"],
            "initial_val_lm_loss": validation_logs[0]["val_lm_loss"],
            "final_val_lm_loss": validation_logs[-1]["val_lm_loss"],
            "peak_gpu_allocated_mb": max(l["gpu_allocated_mb"] for l in training_logs),
            "peak_gpu_reserved_mb": max(l["gpu_reserved_mb"] for l in training_logs),
            "final_generation_metrics": final_metrics,
            "periodic_evaluations": periodic_generation_logs,
        }

        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

        with open(preds_json_path, "w", encoding="utf-8") as f:
            json.dump(sample_preds, f, indent=2)

        # 5. Generate experiment_summary.md
        self.generate_summary_md(
            summary_md_path=summary_md_path,
            metrics_payload=metrics_payload,
            sample_preds=sample_preds,
            trainable_p=trainable_p,
            total_p=total_p,
        )

        print(f"[{self.exp_name.upper()}] All Phase 2B artifacts saved cleanly to {exp_dir}", flush=True)
        return metrics_payload

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """Saves LoRA adapter weights and tokenizer to directory."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model_wrapper.model.save_pretrained(checkpoint_dir)
        self.model_wrapper.tokenizer.save_pretrained(checkpoint_dir)
        meta_path = os.path.join(checkpoint_dir, "experiment_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "experiment_name": self.exp_name,
                "experiment_type": self.exp_type,
                "psd_enabled": self.psd_enabled,
                "lambda_psd": self.lambda_psd,
                "selected_layers": self.selected_layers,
                "max_seq_len": self.max_seq_len,
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.grad_accum_steps,
                "learning_rate": self.lr,
                "seed": self.seed,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, f, indent=2)
        print(f"[ExperimentTrainer] Checkpoint saved to {checkpoint_dir}", flush=True)

    def generate_summary_md(
        self,
        summary_md_path: str,
        metrics_payload: Dict[str, Any],
        sample_preds: List[Dict[str, Any]],
        trainable_p: int,
        total_p: int,
    ) -> None:
        """Writes markdown summary of the experiment."""
        gen_m = metrics_payload.get("final_generation_metrics", {})
        md_content = f"""# Experiment Summary: {self.exp_name.upper()}

- **Experiment Type:** `{self.exp_type}`
- **PSD Enabled:** `{self.psd_enabled}` (Lambda PSD: `{self.lambda_psd}`)
- **Steps Run:** `{metrics_payload['num_steps']}`
- **Elapsed Time:** `{metrics_payload['elapsed_seconds']}s`
- **Trainable Parameters:** `{trainable_p:,}` / `{total_p:,}` (`{round((trainable_p/total_p)*100, 4)}%`)
- **Peak VRAM Allocated:** `{metrics_payload['peak_gpu_allocated_mb']} MB`
- **Peak VRAM Reserved:** `{metrics_payload['peak_gpu_reserved_mb']} MB`

## Loss Progression
- **Initial Training LM Loss:** `{metrics_payload['initial_lm_loss']}` $\\rightarrow$ **Final Training LM Loss:** `{metrics_payload['final_lm_loss']}`
- **Initial Validation LM Loss:** `{metrics_payload['initial_val_lm_loss']}` $\\rightarrow$ **Final Validation LM Loss:** `{metrics_payload['final_val_lm_loss']}`

## Generation Metrics (Final Fixed Evaluation Set)
- **BLEU:** `{gen_m.get('bleu', 0.0)}`
- **ROUGE-1:** `{gen_m.get('rouge1', 0.0)}`
- **ROUGE-2:** `{gen_m.get('rouge2', 0.0)}`
- **ROUGE-L:** `{gen_m.get('rougeL', 0.0)}`
- **BERTScore Precision:** `{gen_m.get('bertscore_p', 0.0)}`
- **BERTScore Recall:** `{gen_m.get('bertscore_r', 0.0)}`
- **BERTScore F1:** `{gen_m.get('bertscore_f1', 0.0)}`
- **Structured Content Coverage:** `{gen_m.get('structured_coverage', 0.0)}%`

## Sample Output Preview
"""
        for i, s in enumerate(sample_preds[:5]):
            md_content += f"""
### Example {i+1} (`{s['id']}` - `{s['dataset']}`)
- **Source Input:** `{s.get('source', '')[:100]}...`
- **Target:** {s['target']}
- **Prediction:** {s['prediction']}
- **Coverage:** {s.get('structured_coverage', 0.0)}%
"""

        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline or PSD experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config yaml")
    parser.add_argument("--steps", type=int, default=None, help="Number of optimizer steps override")
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset or unified split")
    args = parser.parse_args()

    # Load large pool of records to support 500 steps (500*4=2000 micro-steps)
    train_recs = load_dataset_records(dataset_name=args.dataset, split="train", max_records=3000)
    val_recs = load_dataset_records(dataset_name=args.dataset, split="validation", max_records=100)

    trainer = ExperimentTrainer(config_path=args.config)
    steps_to_run = args.steps if args.steps is not None else trainer.config.get("training", {}).get("num_steps", 500)
    trainer.train_steps(train_records=train_recs, val_records=val_recs, num_steps=steps_to_run)

