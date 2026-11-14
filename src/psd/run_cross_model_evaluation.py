"""Cross-Model Evaluation Harness for Phase 6 Final Evaluation.

Supports controlled training (150 steps) and evaluation across:
1. Qwen3-4B (Baseline vs. Full PSD)
2. Llama 3.2 3B Instruct (Baseline vs. Full PSD)
3. Gemma 3 4B Instruct (Baseline vs. Full PSD)

Evaluates BLEU, ROUGE-1/2/L, METEOR, BERTScore, Structured Coverage,
and grounded Factuality (Factual Precision & Hallucination Rate).
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
import torch.optim as optim
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


# Model Configuration Registry
MODEL_REGISTRY = {
    "qwen": {
        "model_id": os.path.join(PROJECT_ROOT, "models", "qwen3_4b_hf"),
        "display_name": "Qwen3-4B",
        "total_layers": 36,
        "selected_layers": [11, 23, 34],
        "target_modules": ["q_proj", "v_proj"],
        "prompt_prefix": "<|im_start|>user\n",
        "prompt_suffix": "<|im_end|>\n<|im_start|>assistant\n",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
    },
    "llama": {
        "model_id": os.path.join(PROJECT_ROOT, "models", "llama_3_2_3b_hf"),
        "fallback_id": "meta-llama/Llama-3.2-3B-Instruct",
        "display_name": "Llama-3.2-3B-Instruct",
        "total_layers": 28,
        "selected_layers": [9, 18, 26],
        "target_modules": ["q_proj", "v_proj"],
        "prompt_prefix": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
        "prompt_suffix": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "eos_token": "<|eot_id|>",
        "pad_token": "<|finetune_right_pad_id|>",
    },
    "gemma": {
        "model_id": os.path.join(PROJECT_ROOT, "models", "gemma_3_4b_hf"),
        "fallback_id": "google/gemma-3-4b-it",
        "display_name": "Gemma-3-4B-Instruct",
        "total_layers": 34,
        "selected_layers": [11, 22, 32],
        "target_modules": ["q_proj", "v_proj"],
        "prompt_prefix": "<bos><start_of_turn>user\n",
        "prompt_suffix": "<end_of_turn>\n<start_of_turn>model\n",
        "eos_token": "<end_of_turn>",
        "pad_token": "<pad>",
    },
}


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_records(split: str = "train", max_records: Optional[int] = None) -> List[UnifiedRecord]:
    """Loads unified records from data/unified/{split}.jsonl."""
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


def format_d2t_training_example_for_model(
    record: UnifiedRecord,
    serializer: Serializer,
    tokenizer: AutoTokenizer,
    model_cfg: Dict[str, Any],
    max_seq_len: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[str]]]:
    """Prepares structured D2T training example adapted to model architecture:
    1. Causal LM loss evaluated ONLY on target reference tokens (prompt masked to -100).
    2. Target text preserved 100% (prompt serialized text truncated from right if budget exceeded).
    3. Token-to-node mapping preserved for PSD computation.
    """
    target_text = record.target_text.strip()
    if not target_text and record.reference_texts:
        target_text = record.reference_texts[0].strip()

    eos_token = model_cfg["eos_token"]
    target_formatted = f"{target_text}{eos_token}"
    target_ids = tokenizer(target_formatted, add_special_tokens=False)["input_ids"]

    prompt_prefix = model_cfg["prompt_prefix"]
    prompt_suffix = model_cfg["prompt_suffix"]
    prefix_ids = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(prompt_suffix, add_special_tokens=False)["input_ids"]

    overhead_tokens = len(prefix_ids) + len(suffix_ids) + len(target_ids)
    max_serialized_tokens = max(max_seq_len - overhead_tokens, 16)

    mapped_data = serializer.serialize_with_mapping(record, tokenizer=tokenizer)
    serialized_ids = mapped_data["input_ids"]
    serialized_node_map = mapped_data["token_to_node_map"]

    truncated_serialized_ids = serialized_ids[:max_serialized_tokens]
    truncated_node_map = serialized_node_map[:max_serialized_tokens]

    full_input_ids = prefix_ids + truncated_serialized_ids + suffix_ids + target_ids
    prompt_len = len(prefix_ids) + len(truncated_serialized_ids) + len(suffix_ids)

    full_labels = [-100] * prompt_len + target_ids
    full_node_map = [None] * len(prefix_ids) + truncated_node_map + [None] * (len(suffix_ids) + len(target_ids))

    input_tensor = torch.tensor([full_input_ids], dtype=torch.long)
    labels_tensor = torch.tensor([full_labels], dtype=torch.long)
    return input_tensor, labels_tensor, full_node_map


class CrossModelTrainer:
    """Manages training and evaluation of a specific model in Baseline or PSD mode."""

    def __init__(
        self,
        model_key: str,
        mode: str = "baseline",
        max_steps: int = 150,
        batch_size: int = 1,
        grad_accum_steps: int = 4,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        max_seq_len: int = 256,
        lambda_psd: float = 0.1,
        seed: int = 42,
    ):
        assert model_key in MODEL_REGISTRY, f"Unknown model key: {model_key}"
        assert mode in ["baseline", "psd"], f"Unknown mode: {mode}"

        self.model_key = model_key
        self.model_cfg = MODEL_REGISTRY[model_key]
        self.mode = mode
        self.psd_enabled = (mode == "psd")
        self.lambda_psd = lambda_psd if self.psd_enabled else 0.0
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_seq_len = max_seq_len
        self.seed = seed
        set_seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.selected_layers = self.model_cfg["selected_layers"]

        # Determine path to model checkpoint
        local_path = self.model_cfg["model_id"]
        safetensor_path = os.path.join(local_path, "model.safetensors")
        if os.path.exists(safetensor_path):
            self.model_load_path = local_path
        elif "fallback_id" in self.model_cfg:
            self.model_load_path = self.model_cfg["fallback_id"]
        else:
            self.model_load_path = local_path

        print(f"\n[{self.model_cfg['display_name']} - {self.mode.upper()}] Initializing...")
        print(f"  Model load path: {self.model_load_path}")
        print(f"  PSD Enabled: {self.psd_enabled} (lambda={self.lambda_psd}, layers={self.selected_layers})")

        # 1. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_load_path)
        if self.tokenizer.pad_token is None:
            pad = self.model_cfg.get("pad_token")
            if pad and pad in self.tokenizer.get_vocab():
                self.tokenizer.pad_token = pad
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Base Model with FP16/FP32
        print(f"  Loading base model with eager attention...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_load_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation="eager",
        )

        # 3. LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=self.model_cfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

        # 4. Helper modules
        self.serializer = Serializer()
        self.dist_calculator = GraphDistanceCalculator()
        self.target_builder = TargetDistributionBuilder()
        self.psd_loss_fn = PSDLoss(selected_layers=self.selected_layers)
        self.semantic_encoder = FrozenContextualSemanticEncoder(device="cpu")
        self.metric_evaluator = MetricEvaluator()
        self.factuality_evaluator = FactualityEvaluator()

        # 5. Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)

        # Output directory
        run_name = f"{self.model_key}_{self.mode}"
        self.output_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "final_evaluation", run_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def forward_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        token_to_node_map: List[Optional[str]],
        record: UnifiedRecord,
    ) -> Tuple[torch.Tensor, float, float]:
        """Computes causal LM loss and optional PSD divergence loss."""
        seq_len = input_ids.shape[1]
        device = input_ids.device

        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            output_attentions=self.psd_enabled,
            output_hidden_states=False,
        )

        lm_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)
        psd_loss = torch.tensor(0.0, device=device)

        if self.psd_enabled and outputs.attentions is not None:
            dist_matrix, node_to_idx = self.dist_calculator.compute_distance_matrix(record)
            tok_ids = input_ids[0].tolist()
            token_strings = [self.tokenizer.decode([tid]) for tid in tok_ids]

            sem_sim = self.semantic_encoder.compute_similarity_matrix(
                token_strings=token_strings,
                token_to_node_map=token_to_node_map,
                record_nodes=record.nodes,
            )

            P_target, valid_mask = self.target_builder.build_target_distribution(
                seq_len=seq_len,
                token_to_node_map=token_to_node_map,
                graph_dist_matrix=dist_matrix,
                node_to_idx=node_to_idx,
                semantic_sim_matrix=sem_sim,
                device=device,
            )

            psd_loss_val, _ = self.psd_loss_fn(
                attentions=outputs.attentions,
                P_target=P_target,
                valid_mask=valid_mask,
            )
            psd_loss = psd_loss_val

        total_loss = lm_loss + self.lambda_psd * psd_loss
        return total_loss, lm_loss.item(), psd_loss.item()

    def train(self, train_records: List[UnifiedRecord]) -> List[Dict[str, Any]]:
        """Executes controlled 150-step training loop with gradient accumulation."""
        log_csv_path = os.path.join(self.output_dir, "training_log.csv")
        adapter_path = os.path.join(self.output_dir, "adapter_model.safetensors")

        if os.path.exists(adapter_path) and os.path.exists(log_csv_path):
            with open(log_csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if len(rows) >= self.max_steps:
                print(f"\n[{self.model_cfg['display_name']} - {self.mode.upper()}] Found existing complete training run ({len(rows)} steps) in {self.output_dir}.")
                print("  Loading saved adapter weights and skipping training loop...")
                from safetensors.torch import load_file
                from peft import set_peft_model_state_dict
                adapter_weights = load_file(adapter_path)
                set_peft_model_state_dict(self.model, adapter_weights)
                return rows

        print(f"\nStarting training: {self.max_steps} optimizer steps (grad_accum={self.grad_accum_steps})...")
        self.model.train()
        self.optimizer.zero_grad()

        training_logs = []

        accum_lm_loss = 0.0
        accum_psd_loss = 0.0
        accum_total_loss = 0.0
        step = 0
        micro_step = 0
        record_idx = 0
        n_records = len(train_records)

        start_time = time.time()

        while step < self.max_steps:
            rec = train_records[record_idx % n_records]
            record_idx += 1

            inp_t, lab_t, node_map = format_d2t_training_example_for_model(
                record=rec,
                serializer=self.serializer,
                tokenizer=self.tokenizer,
                model_cfg=self.model_cfg,
                max_seq_len=self.max_seq_len,
            )
            inp_t = inp_t.to(self.device)
            lab_t = lab_t.to(self.device)

            loss, lm_val, psd_val = self.forward_step(inp_t, lab_t, node_map, rec)
            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            accum_lm_loss += lm_val
            accum_psd_loss += psd_val
            accum_total_loss += loss.item()
            micro_step += 1

            if micro_step % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                step += 1

                avg_lm = accum_lm_loss / self.grad_accum_steps
                avg_psd = accum_psd_loss / self.grad_accum_steps
                avg_tot = accum_total_loss / self.grad_accum_steps
                accum_lm_loss = 0.0
                accum_psd_loss = 0.0
                accum_total_loss = 0.0

                log_entry = {
                    "step": step,
                    "train_lm_loss": round(avg_lm, 4),
                    "train_psd_loss": round(avg_psd, 4),
                    "train_total_loss": round(avg_tot, 4),
                    "learning_rate": self.lr,
                    "elapsed_sec": round(time.time() - start_time, 2),
                }
                training_logs.append(log_entry)

                if step % 25 == 0 or step == self.max_steps:
                    print(
                        f"  Step {step:3d}/{self.max_steps} | "
                        f"LM Loss: {avg_lm:.4f} | "
                        f"PSD Loss: {avg_psd:.4f} | "
                        f"Total: {avg_tot:.4f} | "
                        f"Elapsed: {log_entry['elapsed_sec']:.1f}s"
                    )

        # Save training log CSV
        with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "train_lm_loss", "train_psd_loss", "train_total_loss", "learning_rate", "elapsed_sec"])
            writer.writeheader()
            writer.writerows(training_logs)
        print(f"Saved training log to {log_csv_path}")

        # Save LoRA adapter weights
        self.model.save_pretrained(self.output_dir)
        print(f"Saved model adapter to {self.output_dir}")

        return training_logs

    def evaluate_and_generate(
        self,
        eval_items: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, Any]]:
        """Generates predictions and evaluates full metric suite (BLEU, ROUGE, METEOR, BERTScore, Factuality)."""
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        preds_path = os.path.join(self.output_dir, "sample_predictions.json")
        fact_path = os.path.join(self.output_dir, "factuality_results.json")

        if os.path.exists(metrics_path) and os.path.exists(preds_path) and os.path.exists(fact_path):
            print(f"\n[{self.model_cfg['display_name']} - {self.mode.upper()}] Found existing evaluation files at {self.output_dir}. Loading...")
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            with open(preds_path, "r", encoding="utf-8") as f:
                sample_outputs = json.load(f)
            with open(fact_path, "r", encoding="utf-8") as f:
                factuality_summary = json.load(f)
            return metrics, sample_outputs, factuality_summary

        print(f"\nRunning evaluation on {len(eval_items)} benchmark instances...")
        self.model.eval()

        predictions = []
        references_list = []
        sample_outputs = []
        coverage_list = []

        with torch.no_grad():
            for item in eval_items:
                serialized = item["serialized_text"]
                target = item["target_text"].strip()
                refs = [r.strip() for r in item.get("reference_texts", []) if r.strip()] or [target]
                item_id = item["id"]
                ds = item.get("dataset", "unknown")
                inp_t = item.get("input_type", "unknown")

                prompt = f"{self.model_cfg['prompt_prefix']}{serialized}{self.model_cfg['prompt_suffix']}"
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_len).to(self.device)
                input_len = inputs["input_ids"].shape[1]

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                gen_tokens = outputs[0, input_len:]
                decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

                predictions.append(decoded)
                references_list.append(refs)

                cov_dict = self.metric_evaluator.compute_structured_content_coverage(item, decoded)
                coverage_list.append(cov_dict.get("structured_coverage", 0.0))

                sample_outputs.append({
                    "id": item_id,
                    "dataset": ds,
                    "input_type": inp_t,
                    "source": serialized,
                    "target": target,
                    "references": refs,
                    "prediction": decoded,
                    "structured_coverage": cov_dict.get("structured_coverage", 0.0),
                })

        # 1. Standard D2T Metrics (BLEU, ROUGE, METEOR, BERTScore)
        metrics = self.metric_evaluator.evaluate_batch(predictions, references_list, compute_bertscore=True)
        metrics["structured_coverage"] = round(sum(coverage_list) / len(coverage_list), 2) if coverage_list else 0.0

        # 2. Factuality & Hallucination Metrics
        factuality_summary = self.factuality_evaluator.evaluate_batch(eval_items, predictions)
        metrics["factual_precision"] = factuality_summary["factual_precision"]
        metrics["hallucination_rate"] = factuality_summary["hallucination_rate"]
        metrics["total_factual_claims"] = factuality_summary["supported_claims"]
        metrics["total_hallucinated_claims"] = factuality_summary["unsupported_claims"]

        # Attach per-instance factuality to sample_outputs
        for out, f_inst in zip(sample_outputs, factuality_summary.get("detailed_results", [])):
            out["factual_precision"] = f_inst.get("factual_precision", 0.0)
            out["hallucination_rate"] = f_inst.get("hallucination_rate", 0.0)
            out["num_supported_claims"] = f_inst.get("supported_claims", 0)
            out["num_unsupported_claims"] = f_inst.get("unsupported_claims", 0)

        # Save metrics.json
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

        # Save sample_predictions.json
        preds_path = os.path.join(self.output_dir, "sample_predictions.json")
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump(sample_outputs, f, indent=2)
        print(f"Saved predictions to {preds_path}")

        # Save factuality_results.json
        fact_path = os.path.join(self.output_dir, "factuality_results.json")
        with open(fact_path, "w", encoding="utf-8") as f:
            json.dump(factuality_summary, f, indent=2)
        print(f"Saved factuality analysis to {fact_path}")

        return metrics, sample_outputs, factuality_summary

    def cleanup(self) -> None:
        """Cleans up GPU memory thoroughly."""
        print(f"[{self.model_cfg['display_name']} - {self.mode.upper()}] Cleaning up GPU memory...")
        del self.model
        del self.optimizer
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("  GPU memory cleaned.")


def run_evaluation_pipeline(
    models_to_run: List[str],
    modes_to_run: List[str],
    steps: int = 150,
) -> Dict[str, Any]:
    """Runs sequential training and evaluation pipeline."""
    # Load evaluation benchmark
    eval_path = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b", "evaluation_ids.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_items = json.load(f)
    print(f"Loaded fixed 15-instance benchmark from {eval_path}")

    # Load training data
    train_records = load_dataset_records(split="train")
    print(f"Loaded {len(train_records)} training records from unified dataset")

    all_results = {}

    for model_key in models_to_run:
        for mode in modes_to_run:
            exp_id = f"{model_key}_{mode}"
            print("\n" + "=" * 70)
            print(f"STARTING RUN: {exp_id.upper()} ({steps} steps)")
            print("=" * 70)

            trainer = CrossModelTrainer(
                model_key=model_key,
                mode=mode,
                max_steps=steps,
            )

            # Train
            trainer.train(train_records)

            # Evaluate
            metrics, preds, fact = trainer.evaluate_and_generate(eval_items)
            all_results[exp_id] = metrics

            print(f"\n--- {exp_id.upper()} RESULTS ---")
            print(f"  BLEU-4:       {metrics.get('bleu', 0.0):.2f}")
            print(f"  ROUGE-L:      {metrics.get('rougeL', 0.0):.2f}")
            print(f"  METEOR:       {metrics.get('meteor', 0.0):.2f}")
            print(f"  BERTScore F1: {metrics.get('bertscore_f1', 0.0):.2f}")
            print(f"  Coverage:     {metrics.get('structured_coverage', 0.0):.2f}%")
            print(f"  Fact. Prec.:  {metrics.get('factual_precision', 0.0):.2f}%")
            print(f"  Halluc. Rate: {metrics.get('hallucination_rate', 0.0):.2f}%")

            # Thorough cleanup between runs
            trainer.cleanup()
            time.sleep(2)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Model Evaluation Harness")
    parser.add_argument("--models", nargs="+", default=["llama", "gemma"], choices=["qwen", "llama", "gemma", "all"])
    parser.add_argument("--modes", nargs="+", default=["baseline", "psd"], choices=["baseline", "psd", "all"])
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()

    models = list(MODEL_REGISTRY.keys()) if "all" in args.models else args.models
    modes = ["baseline", "psd"] if "all" in args.modes else args.modes

    print(f"Running Cross-Model Evaluation:")
    print(f"  Models: {models}")
    print(f"  Modes:  {modes}")
    print(f"  Steps:  {args.steps}")

    run_evaluation_pipeline(models_to_run=models, modes_to_run=modes, steps=args.steps)
