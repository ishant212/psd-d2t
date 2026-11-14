"""Component Attribution and Analysis Engine for Phase 3A Ablation Screening.

Aggregates the 8 ablation experiments:
A. baseline
B. semantic_only
C. position_only
D. graph_only
E. semantic_position
F. semantic_graph
G. position_graph
H. full_psd

Computes:
1. Absolute and relative metric changes vs baseline
2. Marginal component contributions:
   - Semantic = Full_PSD - Position_Graph
   - Position = Full_PSD - Semantic_Graph
   - Graph = Full_PSD - Semantic_Position
3. Component-wise interaction analysis
"""

import os
import sys
import json
import csv
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


VARIANTS = [
    ("baseline", "Baseline (LM Only)", False, False, False),
    ("semantic_only", "Semantic Only", True, False, False),
    ("position_only", "Position Only", False, True, False),
    ("graph_only", "Graph Only", False, False, True),
    ("semantic_position", "Semantic + Position", True, True, False),
    ("semantic_graph", "Semantic + Graph", True, False, True),
    ("position_graph", "Position + Graph", False, True, True),
    ("full_psd", "Full PSD", True, True, True),
]


def load_variant_metrics(phase3a_dir: str = "reports/experiments/phase3a") -> Dict[str, Any]:
    """Loads metrics.json and generation_metrics.json for all 8 variants."""
    summary = {}
    for v_key, v_label, sem, pos, grp in VARIANTS:
        v_dir = os.path.join(phase3a_dir, v_key)
        metrics_path = os.path.join(v_dir, "metrics.json")
        gen_metrics_path = os.path.join(v_dir, "generation_metrics.json")
        train_csv_path = os.path.join(v_dir, "training_log.csv")

        data = {
            "variant_key": v_key,
            "label": v_label,
            "semantic": sem,
            "position": pos,
            "graph": grp,
            "initial_train_lm": 0.0,
            "final_train_lm": 0.0,
            "initial_val_lm": 0.0,
            "final_val_lm": 0.0,
            "initial_psd_loss": "NA",
            "final_psd_loss": "NA",
            "peak_vram_mb": 0.0,
            "elapsed_seconds": 0.0,
            "bleu_step1": 0.0,
            "bleu_step50": 0.0,
            "bleu_step100": 0.0,
            "bleu_step150": 0.0,
            "rougeL_step150": 0.0,
            "bertscore_f1_step150": 0.0,
            "coverage_step150": 100.0,
        }

        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                m_json = json.load(f)
            data["initial_train_lm"] = m_json.get("initial_lm_loss", 0.0)
            data["final_train_lm"] = m_json.get("final_lm_loss", 0.0)
            data["initial_val_lm"] = m_json.get("initial_val_lm_loss", 0.0)
            data["final_val_lm"] = m_json.get("final_val_lm_loss", 0.0)
            data["peak_vram_mb"] = m_json.get("peak_gpu_allocated_mb", 0.0)
            data["elapsed_seconds"] = m_json.get("elapsed_seconds", 0.0)

            final_gen = m_json.get("final_generation_metrics", {})
            data["bleu_step150"] = final_gen.get("bleu", 0.0)
            data["rougeL_step150"] = final_gen.get("rougeL", 0.0)
            data["bertscore_f1_step150"] = final_gen.get("bertscore_f1", 0.0)
            data["coverage_step150"] = final_gen.get("structured_coverage", 100.0)

        if os.path.exists(gen_metrics_path):
            with open(gen_metrics_path, "r", encoding="utf-8") as f:
                gen_list = json.load(f)
            for entry in gen_list:
                step = entry.get("step", 0)
                m = entry.get("metrics", {})
                if step == 1:
                    data["bleu_step1"] = m.get("bleu", 0.0)
                elif step == 50:
                    data["bleu_step50"] = m.get("bleu", 0.0)
                elif step == 100:
                    data["bleu_step100"] = m.get("bleu", 0.0)
                elif step == 150:
                    data["bleu_step150"] = m.get("bleu", 0.0)

        if os.path.exists(train_csv_path):
            with open(train_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    if rows[0].get("train_psd_loss") != "NA":
                        data["initial_psd_loss"] = float(rows[0]["train_psd_loss"])
                        data["final_psd_loss"] = float(rows[-1]["train_psd_loss"])

        summary[v_key] = data

    return summary


def compute_ablation_summary(
    phase3a_dir: str = "reports/experiments/phase3a",
    output_csv: str = "reports/experiments/phase3a/ablation_summary.csv",
    output_json: str = "reports/experiments/phase3a/ablation_summary.json",
) -> Dict[str, Any]:
    """Computes comprehensive ablation metrics and marginal attribution."""
    os.makedirs(phase3a_dir, exist_ok=True)
    var_metrics = load_variant_metrics(phase3a_dir)

    base = var_metrics.get("baseline", {})
    base_bleu = base.get("bleu_step150", 0.0)
    base_rl = base.get("rougeL_step150", 0.0)
    base_bs = base.get("bertscore_f1_step150", 0.0)

    rows = []
    for v_key, data in var_metrics.items():
        v_bleu = data["bleu_step150"]
        delta_bleu = round(v_bleu - base_bleu, 2)
        rel_bleu = round((delta_bleu / max(base_bleu, 0.01)) * 100.0, 1)

        row = {
            "variant": v_key,
            "label": data["label"],
            "semantic": "ON" if data["semantic"] else "OFF",
            "position": "ON" if data["position"] else "OFF",
            "graph": "ON" if data["graph"] else "OFF",
            "final_train_lm": data["final_train_lm"],
            "final_val_lm": data["final_val_lm"],
            "final_psd_loss": data["final_psd_loss"],
            "bleu_step1": data["bleu_step1"],
            "bleu_step50": data["bleu_step50"],
            "bleu_step100": data["bleu_step100"],
            "bleu_step150": data["bleu_step150"],
            "delta_bleu_vs_base": delta_bleu,
            "rel_bleu_vs_base_pct": rel_bleu,
            "rougeL_step150": data["rougeL_step150"],
            "bertscore_f1_step150": data["bertscore_f1_step150"],
            "coverage_step150": data["coverage_step150"],
            "peak_vram_mb": data["peak_vram_mb"],
            "runtime_s": data["elapsed_seconds"],
        }
        rows.append(row)

    # Compute marginal component contributions
    full_bleu = var_metrics.get("full_psd", {}).get("bleu_step150", 0.0)
    pos_grp_bleu = var_metrics.get("position_graph", {}).get("bleu_step150", 0.0)
    sem_grp_bleu = var_metrics.get("semantic_graph", {}).get("bleu_step150", 0.0)
    sem_pos_bleu = var_metrics.get("semantic_position", {}).get("bleu_step150", 0.0)

    marginal_attributions = {
        "semantic_marginal_contribution (Full - Pos+Graph)": round(full_bleu - pos_grp_bleu, 2),
        "position_marginal_contribution (Full - Sem+Graph)": round(full_bleu - sem_grp_bleu, 2),
        "graph_marginal_contribution (Full - Sem+Pos)": round(full_bleu - sem_pos_bleu, 2),
    }

    payload = {
        "variants": var_metrics,
        "summary_rows": rows,
        "marginal_attributions": marginal_attributions,
    }

    # Save CSV
    if rows:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[AnalyzePhase3A] Saved summary to {output_csv} and {output_json}")
    return payload


if __name__ == "__main__":
    compute_ablation_summary()
