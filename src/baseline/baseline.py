"""Baseline Transformer Data-to-Text Model Pipeline.

Provides a clean, modular Sequence-to-Sequence baseline (e.g. T5) for
generating text from structured inputs, establishing the benchmark that
Phase 2 (topology-aware attention and PSD loss) will extend.
"""

import os
import json
import argparse
import logging
from typing import Dict, Any, List, Optional
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data.unified_schema import UnifiedRecord
from src.baseline.dataset import D2TDataset, d2t_collate_fn
from src.baseline.evaluate_metrics import MetricEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class D2TBaselineModel:
    """Sequence-to-sequence baseline generator using HuggingFace T5."""

    def __init__(
        self,
        model_name: str = "t5-small",
        device: Optional[str] = None,
        max_input_length: int = 512,
        max_target_length: int = 128
    ):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        logger.info(f"Initializing baseline model '{model_name}' on device '{self.device}'.")
        from transformers import T5Config, T5ForConditionalGeneration
        
        config = T5Config(
            vocab_size=32128,
            d_model=256,
            d_kv=64,
            d_ff=1024,
            num_layers=3,
            num_decoder_layers=3,
            num_heads=4,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
            
        self.model = T5ForConditionalGeneration(config).to(self.device)
        self.evaluator = MetricEvaluator()

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        gradient_accumulation_steps: int = 1
    ) -> float:
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc="Training Batch")):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / max(1, len(train_loader))
        return avg_loss

    def generate(
        self,
        eval_loader: DataLoader,
        num_beams: int = 4,
        max_new_tokens: int = 128
    ) -> List[Dict[str, Any]]:
        """Generates predictions on the evaluation dataloader."""
        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Generating Predictions"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=True
                )

                decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                for i in range(len(decoded_preds)):
                    results.append({
                        "id": batch["id"][i],
                        "dataset": batch["dataset"][i],
                        "source": batch["source_text"][i],
                        "prediction": decoded_preds[i].strip(),
                        "target": batch["target_text"][i],
                        "references": batch["references"][i] if isinstance(batch["references"][i], list) else [batch["references"][i]]
                    })

        return results

    def evaluate(self, eval_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculates evaluation metrics (BLEU, ROUGE) over predictions."""
        preds = [r["prediction"] for r in eval_results]
        refs = [r["references"] for r in eval_results]
        metrics = self.evaluator.evaluate_batch(preds, refs)
        return metrics

    def save_model(self, save_dir: str):
        """Saves model weights and tokenizer to directory."""
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Model saved to {save_dir}")


def run_baseline_experiment(
    train_records: List[UnifiedRecord],
    val_records: List[UnifiedRecord],
    model_name: str = "t5-small",
    batch_size: int = 4,
    epochs: int = 1,
    lr: float = 3e-4,
    max_input_length: int = 512,
    max_target_length: int = 128,
    output_predictions_path: str = "reports/phase1/sample_predictions.json"
) -> Dict[str, Any]:
    """Runs a complete training and evaluation experiment."""
    baseline = D2TBaselineModel(
        model_name=model_name,
        max_input_length=max_input_length,
        max_target_length=max_target_length
    )

    train_dataset = D2TDataset(train_records, baseline.tokenizer, max_input_length, max_target_length)
    val_dataset = D2TDataset(val_records, baseline.tokenizer, max_input_length, max_target_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=d2t_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=d2t_collate_fn)

    optimizer = AdamW(baseline.model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    logger.info(f"Starting baseline training for {epochs} epoch(s)...")
    for epoch in range(epochs):
        train_loss = baseline.train_epoch(train_loader, optimizer, scheduler)
        logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

    logger.info("Evaluating baseline model...")
    eval_predictions = baseline.generate(val_loader)
    metrics = baseline.evaluate(eval_predictions)
    logger.info(f"Validation Metrics: {metrics}")

    # Save predictions
    os.makedirs(os.path.dirname(output_predictions_path), exist_ok=True)
    with open(output_predictions_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "sample_predictions": eval_predictions[:50]
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved predictions and metrics to {output_predictions_path}")

    return {
        "metrics": metrics,
        "eval_predictions": eval_predictions
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Transformer D2T Pipeline")
    parser.add_argument("--model", type=str, default="t5-small")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
