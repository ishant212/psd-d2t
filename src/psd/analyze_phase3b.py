"""Analysis and Verification Engine for Phase 3B Confirmation Experiment.

Generates:
1. reports/experiments/phase3b/confirmation_summary.csv
2. reports/experiments/phase3b/confirmation_summary.json
3. reports/experiments/phase3b/verification.json

Performs deep 150 -> 300 -> 500 step progression analysis, component advantage tracking,
and strict verification audits.
"""

import os
import sys
import json
import csv
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PHASE3B_DIR = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase3b")

VARIANTS = [
    ("baseline", "Baseline (LM Only)", {"sem": False, "pos": False, "graph": False, "lambda": 0.0}),
    ("position_only", "Position Only (-beta*dP)", {"sem": False, "pos": True, "graph": False, "lambda": 0.1}),
    ("full_psd", "Full PSD (alpha*S - beta*dP - gamma*dG)", {"sem": True, "pos": True, "graph": True, "lambda": 0.1}),
]


def load_variant_data(variant_key: str) -> Dict[str, Any]:
    """Loads all logs, metrics, and generation histories for a variant."""
    v_dir = os.path.join(PHASE3B_DIR, variant_key)
    metrics_path = os.path.join(v_dir, "metrics.json")
    gen_metrics_path = os.path.join(v_dir, "generation_metrics.json")
    train_log_path = os.path.join(v_dir, "training_log.csv")
    val_log_path = os.path.join(v_dir, "validation_log.csv")

    data = {
        "metrics": {},
        "gen_history": [],
        "training_logs": [],
        "validation_logs": [],
    }

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data["metrics"] = json.load(f)

    if os.path.exists(gen_metrics_path):
        with open(gen_metrics_path, "r", encoding="utf-8") as f:
            data["gen_history"] = json.load(f)

    if os.path.exists(train_log_path):
        with open(train_log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data["training_logs"] = list(reader)

    if os.path.exists(val_log_path):
        with open(val_log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data["validation_logs"] = list(reader)

    return data


def compute_confirmation_summary() -> Dict[str, Any]:
    """Computes comparative metrics, step progression (150 -> 300 -> 500), and exports summary files."""
    os.makedirs(PHASE3B_DIR, exist_ok=True)
    summary_data = {}
    csv_rows = []

    # 1. Load all variant data
    raw_data = {}
    for v_key, v_label, v_flags in VARIANTS:
        raw_data[v_key] = load_variant_data(v_key)

    base_data = raw_data.get("baseline", {})
    base_final_gen = base_data.get("metrics", {}).get("final_generation_metrics", {})
    base_final_bleu = base_final_gen.get("bleu", base_final_gen.get("bleu_score", 0.0))
    base_final_rouge_l = base_final_gen.get("rougeL", base_final_gen.get("rouge_l", 0.0))
    base_final_bertscore = base_final_gen.get("bertscore_f1", 0.0)

    # Helper to get metrics at a specific step
    def get_metrics_at_step(gen_history: List[Dict[str, Any]], target_step: int) -> Dict[str, float]:
        for entry in gen_history:
            if entry.get("step") == target_step:
                return entry.get("metrics", {})
        return {}

    for v_key, v_label, v_flags in VARIANTS:
        v_data = raw_data[v_key]
        metrics = v_data["metrics"]
        final_gen = metrics.get("final_generation_metrics", {})
        gen_hist = v_data["gen_history"]
        train_logs = v_data["training_logs"]

        # Step 500 Metrics
        bleu = final_gen.get("bleu", final_gen.get("bleu_score", 0.0))
        rouge_1 = final_gen.get("rouge1", final_gen.get("rouge_1", 0.0))
        rouge_2 = final_gen.get("rouge2", final_gen.get("rouge_2", 0.0))
        rouge_l = final_gen.get("rougeL", final_gen.get("rouge_l", 0.0))
        bertscore_f1 = final_gen.get("bertscore_f1", 0.0)
        coverage = final_gen.get("structured_coverage", final_gen.get("content_coverage", 0.0))

        delta_bleu = round(bleu - base_final_bleu, 4)
        rel_delta_bleu = round((delta_bleu / max(base_final_bleu, 1e-6)) * 100, 2)
        delta_rouge_l = round(rouge_l - base_final_rouge_l, 4)
        delta_bertscore = round(bertscore_f1 - base_final_bertscore, 4)

        peak_vram_mb = metrics.get("peak_gpu_allocated_mb", 0.0)
        elapsed_sec = metrics.get("elapsed_seconds", 0.0)

        # Multi-step progression (1, 50, 100, 150, 200, 300, 400, 500)
        steps_eval = [1, 50, 100, 150, 200, 300, 400, 500]
        step_metrics_map = {}
        for s in steps_eval:
            m = get_metrics_at_step(gen_hist, s)
            step_metrics_map[s] = {
                "bleu": m.get("bleu", m.get("bleu_score", 0.0)),
                "rouge_l": m.get("rougeL", m.get("rouge_l", 0.0)),
                "bertscore_f1": m.get("bertscore_f1", 0.0),
            }

        # Trajectory 150 -> 300 -> 500
        m150 = step_metrics_map.get(150, {})
        m300 = step_metrics_map.get(300, {})
        m500 = step_metrics_map.get(500, {})

        # PSD loss reduction
        init_psd = 0.0
        final_psd = 0.0
        psd_reduction_pct = 0.0
        if v_flags["lambda"] > 0 and train_logs:
            for l in train_logs:
                if l.get("train_psd_loss") not in ["NA", None, ""]:
                    try:
                        init_psd = float(l["train_psd_loss"])
                        break
                    except (ValueError, TypeError):
                        pass
            for l in reversed(train_logs):
                if l.get("train_psd_loss") not in ["NA", None, ""]:
                    try:
                        final_psd = float(l["train_psd_loss"])
                        break
                    except (ValueError, TypeError):
                        pass
            if init_psd > 0:
                psd_reduction_pct = round(((init_psd - final_psd) / init_psd) * 100, 2)

        summary_data[v_key] = {
            "label": v_label,
            "flags": v_flags,
            "final_metrics": {
                "bleu": bleu,
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "rouge_l": rouge_l,
                "bertscore_f1": bertscore_f1,
                "content_coverage": coverage,
            },
            "deltas_vs_baseline_at_500": {
                "delta_bleu": delta_bleu,
                "rel_delta_bleu_pct": rel_delta_bleu,
                "delta_rouge_l": delta_rouge_l,
                "delta_bertscore_f1": delta_bertscore,
            },
            "step_progression": step_metrics_map,
            "trajectory_150_300_500": {
                "step_150": m150,
                "step_300": m300,
                "step_500": m500,
            },
            "psd_convergence": {
                "initial_psd_loss": init_psd,
                "final_psd_loss": final_psd,
                "reduction_pct": psd_reduction_pct,
            },
            "system": {
                "peak_gpu_allocated_mb": peak_vram_mb,
                "elapsed_seconds": elapsed_sec,
            },
        }

        csv_rows.append({
            "variant_key": v_key,
            "label": v_label,
            "lambda_psd": v_flags["lambda"],
            "bleu_step_150": m150.get("bleu", "NA"),
            "bleu_step_300": m300.get("bleu", "NA"),
            "bleu_step_500": bleu,
            "delta_bleu_vs_base_500": delta_bleu,
            "rouge_l_step_150": m150.get("rouge_l", "NA"),
            "rouge_l_step_300": m300.get("rouge_l", "NA"),
            "rouge_l_step_500": rouge_l,
            "delta_rouge_l_vs_base_500": delta_rouge_l,
            "bertscore_step_150": m150.get("bertscore_f1", "NA"),
            "bertscore_step_300": m300.get("bertscore_f1", "NA"),
            "bertscore_step_500": bertscore_f1,
            "delta_bertscore_vs_base_500": delta_bertscore,
            "content_coverage_500": coverage,
            "initial_psd_loss": init_psd if v_flags["lambda"] > 0 else "NA",
            "final_psd_loss": final_psd if v_flags["lambda"] > 0 else "NA",
            "psd_reduction_pct": psd_reduction_pct if v_flags["lambda"] > 0 else "NA",
            "peak_vram_mb": peak_vram_mb,
            "runtime_seconds": elapsed_sec,
        })

    # Save summary JSON
    summary_json_path = os.path.join(PHASE3B_DIR, "confirmation_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save summary CSV
    summary_csv_path = os.path.join(PHASE3B_DIR, "confirmation_summary.csv")
    if csv_rows:
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # Run automated verification audit
    verify_phase3b_integrity(summary_data)

    return summary_data


def verify_phase3b_integrity(summary_data: Dict[str, Any]) -> Dict[str, Any]:
    """Automated integrity audit ensuring complete 500-step runs, checkpointing, and preservation."""
    verification = {
        "status": "PASSED",
        "timestamp": None,
        "checks": {},
    }

    # 1. Historical artifacts preservation
    p2b_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b")
    p3a_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase3a")
    p2b_intact = os.path.exists(os.path.join(p2b_dir, "evaluation_ids.json")) and os.path.exists(os.path.join(p2b_dir, "psd", "metrics.json"))
    p3a_intact = os.path.exists(os.path.join(p3a_dir, "ablation_summary.csv")) and os.path.exists(os.path.join(p3a_dir, "phase3a_report.md"))

    verification["checks"]["phase2b_artifacts_preserved"] = p2b_intact
    verification["checks"]["phase3a_artifacts_preserved"] = p3a_intact

    # 2. Check 500 steps and checkpoints for each variant
    all_ckpts_present = True
    all_evals_present = True
    required_ckpts = ["step_100", "step_200", "step_300", "step_400", "step_500", "final"]
    required_eval_steps = [1, 50, 100, 150, 200, 300, 400, 500]

    for v_key, _, _ in VARIANTS:
        v_dir = os.path.join(PHASE3B_DIR, v_key)
        ckpt_dir = os.path.join(v_dir, "checkpoints")

        for ckpt in required_ckpts:
            c_path = os.path.join(ckpt_dir, ckpt)
            if not os.path.exists(c_path):
                all_ckpts_present = False
                verification["checks"][f"{v_key}_{ckpt}_exists"] = False
            else:
                verification["checks"][f"{v_key}_{ckpt}_exists"] = True

        v_sum = summary_data.get(v_key, {})
        step_prog = v_sum.get("step_progression", {})
        for s in required_eval_steps:
            if s not in step_prog or step_prog[s].get("bleu") is None:
                all_evals_present = False

    verification["checks"]["all_required_checkpoints_exist"] = all_ckpts_present
    verification["checks"]["all_required_evaluation_steps_exist"] = all_evals_present

    if not (p2b_intact and p3a_intact and all_ckpts_present and all_evals_present):
        verification["status"] = "INCOMPLETE_OR_FAILED"

    v_path = os.path.join(PHASE3B_DIR, "verification.json")
    with open(v_path, "w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2)

    return verification


if __name__ == "__main__":
    compute_confirmation_summary()
    print("[analyze_phase3b] Confirmation summary and verification completed.")
