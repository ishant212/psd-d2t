"""Master Orchestrator for Phase 3B Targeted 500-Step Confirmation Experiment.

Executes the 3 targeted confirmation variants for 500 optimizer steps:
1. phase3b_baseline
2. phase3b_position_only
3. phase3b_full_psd

Saves logs, checkpoints at 100/200/300/400/500/final, periodic evaluations at 1/50/100/150/200/300/400/500,
metrics, predictions, summary plots, and verification report.
"""

import os
import sys
import json
import time
import argparse
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.psd.trainer import ExperimentTrainer, load_dataset_records, set_seed
from src.psd.analyze_phase3b import compute_confirmation_summary
from src.psd.plot_phase3b import generate_all_phase3b_plots


CONFIRMATION_CONFIGS = [
    ("baseline", "configs/phase3b_baseline.yaml"),
    ("position_only", "configs/phase3b_position_only.yaml"),
    ("full_psd", "configs/phase3b_full_psd.yaml"),
]


def run_single_variant(variant_key: str, config_path: str, steps: int = 500, fixed_eval_records: list = None) -> None:
    """Executes a single confirmation variant."""
    print("\n" + "=" * 75)
    print(f"STARTING PHASE 3B CONFIRMATION: [{variant_key.upper()}] (Steps: {steps})")
    print("=" * 75, flush=True)

    set_seed(42)
    train_recs = load_dataset_records(split="train", max_records=2500)
    val_recs = load_dataset_records(split="validation", max_records=50)

    trainer = ExperimentTrainer(config_path=config_path)
    trainer.train_steps(
        train_records=train_recs,
        val_records=val_recs,
        fixed_eval_records=fixed_eval_records,
        num_steps=steps,
    )

    # Clean GPU memory between variants
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_all_confirmation(steps: int = 500, specific_variant: str = None) -> None:
    """Executes the 3 confirmation variants sequentially with strict scientific controls."""
    # 1. Safety check on Phase 2B and Phase 3A
    p2b_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b")
    p3a_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase3a")
    assert os.path.exists(p2b_dir), "Error: Phase 2B artifacts must exist before Phase 3B."
    assert os.path.exists(p3a_dir), "Error: Phase 3A artifacts must exist before Phase 3B."

    # 2. Load fixed evaluation set
    eval_ids_path = os.path.join(p2b_dir, "evaluation_ids.json")
    fixed_eval_records = None
    if os.path.exists(eval_ids_path):
        with open(eval_ids_path, "r", encoding="utf-8") as f:
            fixed_eval_records = json.load(f)
        print(f"[Phase3B] Loaded {len(fixed_eval_records)} fixed evaluation records from Phase 2B.")

    to_run = [(k, p) for k, p in CONFIRMATION_CONFIGS if (specific_variant is None or specific_variant == k)]

    for v_key, cfg_path in to_run:
        full_cfg_path = os.path.join(PROJECT_ROOT, cfg_path)
        run_single_variant(
            variant_key=v_key,
            config_path=full_cfg_path,
            steps=steps,
            fixed_eval_records=fixed_eval_records,
        )

    # 3. Post-run analysis & plotting if running full pipeline
    if specific_variant is None or len(to_run) == len(CONFIRMATION_CONFIGS):
        print("\n[Phase3B] Computing confirmation summary and generating plots...", flush=True)
        compute_confirmation_summary()
        generate_all_phase3b_plots()
        print("[Phase3B] All Phase 3B confirmation tasks completed successfully!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 3B Confirmation Experiments")
    parser.add_argument("--steps", type=int, default=500, help="Number of optimizer steps (default: 500)")
    parser.add_argument("--variant", type=str, default=None, choices=["baseline", "position_only", "full_psd"], help="Run single variant")
    args = parser.parse_args()

    run_all_confirmation(steps=args.steps, specific_variant=args.variant)
