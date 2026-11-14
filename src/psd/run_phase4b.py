"""Phase 4B: Long-Context Training Comparison Executor."""
import os
import sys
import json
import time
import torch
import yaml

FILE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.psd.trainer import ExperimentTrainer, load_dataset_records, set_seed
from src.baseline.evaluate_metrics import MetricEvaluator

def run_phase4b_training():
    print("=================================================")
    print("SHASE LONG-CONTEXT TRAINING COMPARISON")
    print("================================================")
    set_seed(42)

    reports_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase4b")
    os.makedirs(reports_dir, exist_ok=True)

    # Load datasets
    train_records = load_dataset_records(split="train")
    val_records = load_dataset_records(split="validation") or load_dataset_records(split="train", max_records=100)
    eval_ids_path = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b", "evaluation_ids.json")
    with open(eval_ids_path, "r", encoding="utf-8") as f:
        fixed_eval_records = json.load(f)

    print(f"Loaded {len(train_records)} training records, {len(val_records)} validation records, {len(fixed_eval_records)} fixed eval records.")

    experiments = [
        ("Baseline 512", "configs/phase4b_baseline_512.yaml"),
        ("Full PSD 512", "configs/phase4b_psd_512.yaml"),
    ]

    for exp_name, cfg_rel in experiments:
        print(f"\n--------------------------------------------------")
        print(f"[STARTING] Training {exp_name} using {cfg_rel}...")
        print(f"--------------------------------------------------", flush=True)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        cfg_full = os.path.join(PROJECT_ROOT, cfg_rel)
        trainer = ExperimentTrainer(config_path=cfg_full)
        
        result = trainer.train_steps(
            train_records=train_records,
            val_records=val_records,
            fixed_eval_records=fixed_eval_records,
            num_steps=500,
        )
        print(f"[COMPLETED] {exp_name} Training Finished.", flush=True)
        
        del trainer
        torch.cuda.empty_cache()

    print("\n=== PHASE 4B TRAINING RUNS COMPLETED ===")

if __name__ == "__main__":
    run_phase4b_training()
