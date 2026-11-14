"""Master Orchestrator for Phase 3A Ablation Screening.

Runs all 8 ablation variants for 150 optimizer steps:
1. phase3a_baseline
2. phase3a_semantic_only
3. phase3a_position_only
4. phase3a_graph_only
5. phase3a_semantic_position
6. phase3a_semantic_graph
7. phase3a_position_graph
8. phase3a_full_psd

Saves logs, checkpoints, metrics, and generates plots and reports.
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
from src.psd.analyze_phase3a import compute_ablation_summary
from src.psd.plot_phase3a import generate_all_phase3a_plots


VARIANTS_CONFIGS = [
    ("baseline", "configs/phase3a_baseline.yaml"),
    ("semantic_only", "configs/phase3a_semantic_only.yaml"),
    ("position_only", "configs/phase3a_position_only.yaml"),
    ("graph_only", "configs/phase3a_graph_only.yaml"),
    ("semantic_position", "configs/phase3a_semantic_position.yaml"),
    ("semantic_graph", "configs/phase3a_semantic_graph.yaml"),
    ("position_graph", "configs/phase3a_position_graph.yaml"),
    ("full_psd", "configs/phase3a_full_psd.yaml"),
]


def run_single_variant(variant_key: str, config_path: str, steps: int = 150, fixed_eval_records: list = None) -> None:
    """Executes a single ablation variant."""
    print("\n" + "=" * 75)
    print(f"STARTING PHASE 3A ABLATION: [{variant_key.upper()}] (Steps: {steps})")
    print("=" * 75, flush=True)

    set_seed(42)
    train_recs = load_dataset_records(split="train", max_records=1500)
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


def run_all_ablations(steps: int = 150, specific_variant: str = None) -> None:
    """Executes all 8 ablation variants sequentially with scientific rigor."""
    # 1. Safety check on Phase 2B
    p2b_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b")
    assert os.path.exists(p2b_dir), "Error: Phase 2B artifacts must exist before Phase 3A."

    # 2. Load fixed evaluation set
    eval_ids_path = os.path.join(p2b_dir, "evaluation_ids.json")
    fixed_eval_records = None
    if os.path.exists(eval_ids_path):
        with open(eval_ids_path, "r", encoding="utf-8") as f:
            fixed_eval_records = json.load(f)
        print(f"[Phase3A] Loaded {len(fixed_eval_records)} fixed evaluation records from Phase 2B.")

    to_run = [(k, p) for k, p in VARIANTS_CONFIGS if (specific_variant is None or specific_variant == k)]

    for v_key, cfg_path in to_run:
        full_cfg_path = os.path.join(PROJECT_ROOT, cfg_path)
        run_single_variant(
            variant_key=v_key,
            config_path=full_cfg_path,
            steps=steps,
            fixed_eval_records=fixed_eval_records,
        )

    # 3. Post-run analysis & plotting
    print("\n[Phase3A] Computing comprehensive ablation summary and generating plots...", flush=True)
    compute_ablation_summary()
    generate_all_phase3a_plots()
    print("[Phase3A] All Phase 3A tasks finished successfully!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 3A Ablation Screening")
    parser.add_argument("--steps", type=int, default=150, help="Number of optimizer steps per variant (default: 150)")
    parser.add_argument("--variant", type=str, default=None, help="Run a specific variant key")
    args = parser.parse_args()

    run_all_ablations(steps=args.steps, specific_variant=args.variant)
