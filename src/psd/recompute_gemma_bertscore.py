"""Recompute BERTScore for Gemma and update all summary tables and figures."""

import os
import sys
import json
import csv
import numpy as np
import torch
import bert_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "experiments", "final_totto_long_context")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

JSONL_PATH = os.path.join(DATA_DIR, "per_example_results.jsonl")

CONTEXT_LENGTHS = [256, 512, 1024, 2048]
CONTEXT_BUCKETS = ["257-512", "513-1024", "1025-2048", ">2048"]


def compute_paired_bootstrap_ci(
    base_scores,
    psd_scores,
    n_boot=1000,
    alpha=0.05,
    seed=42,
):
    rng = np.random.RandomState(seed)
    diffs = np.array(psd_scores) - np.array(base_scores)
    n = len(diffs)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0, "significant": False}

    boot_diffs = []
    for _ in range(n_boot):
        indices = rng.choice(n, size=n, replace=True)
        boot_diffs.append(float(np.mean(diffs[indices])))

    ci_lower = float(np.percentile(boot_diffs, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha / 2.0)))
    mean_diff = float(np.mean(diffs))

    if mean_diff >= 0:
        p_val = 2.0 * float(np.mean(np.array(boot_diffs) <= 0.0))
    else:
        p_val = 2.0 * float(np.mean(np.array(boot_diffs) >= 0.0))
    p_val = min(1.0, max(0.0, p_val))

    return {
        "mean_diff": round(mean_diff, 3),
        "ci_lower": round(ci_lower, 3),
        "ci_upper": round(ci_upper, 3),
        "p_value": round(p_val, 4),
        "significant": bool(ci_lower > 0.0 or ci_upper < 0.0),
    }


def main():
    print("Loading all 3,840 records from per_example_results.jsonl...")
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 3840, f"Expected 3840, got {len(records)}"

    gemma_indices = [i for i, r in enumerate(records) if r["model_key"] == "gemma"]
    print(f"Found {len(gemma_indices)} Gemma records. Computing BERTScore on GPU...")

    clean_preds = [records[i]["generated_text"] if records[i]["generated_text"].strip() else "." for i in gemma_indices]
    clean_refs = [records[i]["reference_texts"][0] if records[i]["reference_texts"][0].strip() else "." for i in gemma_indices]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    P, R, F1 = bert_score.score(clean_preds, clean_refs, lang="en", verbose=True, batch_size=64, device=device)

    f1_list = (F1 * 100.0).tolist()
    for idx_in_gemma, global_idx in enumerate(gemma_indices):
        if not records[global_idx]["generated_text"].strip():
            records[global_idx]["bertscore_f1"] = 0.0
        else:
            records[global_idx]["bertscore_f1"] = round(float(f1_list[idx_in_gemma]), 2)

    print(f"Gemma BERTScore mean: {np.mean(f1_list):.2f}%")

    # Save updated per_example_results.jsonl
    print("Writing updated per_example_results.jsonl...")
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUTPUT_DIR, "per_example_results.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Recompute summaries
    print("Recomputing Context Length Summary Table...")
    length_summary_rows = []
    models_dict = {"qwen": "Qwen3-4B", "llama": "Llama-3.2-3B", "gemma": "Gemma-3-4B"}

    for m_key, m_name in models_dict.items():
        for clen in CONTEXT_LENGTHS:
            for cond in ["Baseline", "Full PSD"]:
                sub = [it for it in records if it.get("model_key") == m_key and it.get("condition") == cond and it.get("context_length") == clen]
                bleu = np.mean([it.get("bleu_4", 0.0) for it in sub])
                rl = np.mean([it.get("rouge_l", 0.0) for it in sub])
                met = np.mean([it.get("meteor", 0.0) for it in sub])
                bs = np.mean([it.get("bertscore_f1", 0.0) for it in sub])
                fp = np.mean([it.get("factual_precision", 0.0) for it in sub])
                hr = np.mean([it.get("hallucination_rate", 0.0) for it in sub])
                cov = np.mean([it.get("structured_coverage", 0.0) for it in sub])
                trunc_rate = np.mean([100.0 if it.get("is_truncated") else 0.0 for it in sub])

                length_summary_rows.append({
                    "model": m_name,
                    "condition": cond,
                    "context_length": clen,
                    "bleu_4": round(float(bleu), 2),
                    "rouge_l": round(float(rl), 2),
                    "meteor": round(float(met), 2),
                    "bertscore_f1": round(float(bs), 2),
                    "factual_precision": round(float(fp), 2),
                    "hallucination_rate": round(float(hr), 2),
                    "structured_coverage": round(float(cov), 2),
                    "truncation_rate_pct": round(float(trunc_rate), 2),
                })

    summary_csv_path = os.path.join(ANALYSIS_DIR, "context_length_summary.csv")
    master_csv_path = os.path.join(DATA_DIR, "final_totto_results.csv")
    master_json_path = os.path.join(DATA_DIR, "final_totto_results.json")

    fieldnames = ["model", "condition", "context_length", "bleu_4", "rouge_l", "meteor", "bertscore_f1", "factual_precision", "hallucination_rate", "structured_coverage", "truncation_rate_pct"]
    for path in [summary_csv_path, master_csv_path, os.path.join(OUTPUT_DIR, "final_totto_results.csv"), os.path.join(OUTPUT_DIR, "context_length_summary.csv")]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(length_summary_rows)

    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(length_summary_rows, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "final_totto_results.json"), "w", encoding="utf-8") as f:
        json.dump(length_summary_rows, f, indent=2)

    # Recompute Bucket Summary
    print("Recomputing Source-Length Bucket Summary...")
    bucket_summary_rows = []
    for m_key, m_name in models_dict.items():
        for clen in CONTEXT_LENGTHS:
            for b in CONTEXT_BUCKETS:
                sub_base = [it for it in records if it.get("model_key") == m_key and it.get("condition") == "Baseline" and it.get("context_length") == clen and it.get("context_bucket") == b]
                sub_psd = [it for it in records if it.get("model_key") == m_key and it.get("condition") == "Full PSD" and it.get("context_length") == clen and it.get("context_bucket") == b]

                b_met = np.mean([it.get("meteor", 0.0) for it in sub_base])
                p_met = np.mean([it.get("meteor", 0.0) for it in sub_psd])
                b_bleu = np.mean([it.get("bleu_4", 0.0) for it in sub_base])
                p_bleu = np.mean([it.get("bleu_4", 0.0) for it in sub_psd])
                b_rl = np.mean([it.get("rouge_l", 0.0) for it in sub_base])
                p_rl = np.mean([it.get("rouge_l", 0.0) for it in sub_psd])
                b_bs = np.mean([it.get("bertscore_f1", 0.0) for it in sub_base])
                p_bs = np.mean([it.get("bertscore_f1", 0.0) for it in sub_psd])
                b_fp = np.mean([it.get("factual_precision", 0.0) for it in sub_base])
                p_fp = np.mean([it.get("factual_precision", 0.0) for it in sub_psd])
                b_hr = np.mean([it.get("hallucination_rate", 0.0) for it in sub_base])
                p_hr = np.mean([it.get("hallucination_rate", 0.0) for it in sub_psd])
                mean_trunc = np.mean([it.get("truncation_amount", 0) for it in sub_base])

                bucket_summary_rows.append({
                    "model": m_name,
                    "context_length": clen,
                    "source_bucket": b,
                    "n_examples": len(sub_base),
                    "mean_truncation_tokens": round(float(mean_trunc), 1),
                    "baseline_meteor": round(float(b_met), 2),
                    "psd_meteor": round(float(p_met), 2),
                    "delta_meteor": round(float(p_met - b_met), 2),
                    "baseline_bertscore": round(float(b_bs), 2),
                    "psd_bertscore": round(float(p_bs), 2),
                    "delta_bertscore": round(float(p_bs - b_bs), 2),
                    "baseline_fact_prec": round(float(b_fp), 2),
                    "psd_fact_prec": round(float(p_fp), 2),
                    "delta_fact_prec": round(float(p_fp - b_fp), 2),
                    "baseline_halluc": round(float(b_hr), 2),
                    "psd_halluc": round(float(p_hr), 2),
                    "delta_halluc": round(float(p_hr - b_hr), 2),
                    "delta_bleu": round(float(p_bleu - b_bleu), 2),
                    "delta_rouge_l": round(float(p_rl - b_rl), 2),
                })

    for path in [os.path.join(ANALYSIS_DIR, "context_bucket_summary.csv"), os.path.join(OUTPUT_DIR, "context_bucket_summary.csv")]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(bucket_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(bucket_summary_rows)

    # Recompute Statistical Bootstrap Analysis
    print("Recomputing Paired Bootstrap Confidence Intervals...")
    stat_analysis = {}
    for m_key, m_name in models_dict.items():
        stat_analysis[m_name] = {"by_context_length": {}, "by_source_bucket": {}}
        for clen in CONTEXT_LENGTHS:
            sub_base = [it for it in records if it.get("model_key") == m_key and it.get("condition") == "Baseline" and it.get("context_length") == clen]
            sub_psd = [it for it in records if it.get("model_key") == m_key and it.get("condition") == "Full PSD" and it.get("context_length") == clen]
            stat_analysis[m_name]["by_context_length"][str(clen)] = {}
            for metric in ["meteor", "bertscore_f1", "factual_precision", "hallucination_rate"]:
                b_vals = [it.get(metric, 0.0) for it in sub_base]
                p_vals = [it.get(metric, 0.0) for it in sub_psd]
                stat_analysis[m_name]["by_context_length"][str(clen)][metric] = compute_paired_bootstrap_ci(b_vals, p_vals)

        for b in CONTEXT_BUCKETS:
            sub_base = [it for it in records if it.get("model_key") == m_key and it.get("condition") == "Baseline" and it.get("context_bucket") == b]
            sub_psd = [it for it in records if it.get("model_key") == m_key and it.get("condition") == "Full PSD" and it.get("context_bucket") == b]
            stat_analysis[m_name]["by_source_bucket"][b] = {}
            for metric in ["meteor", "bertscore_f1", "factual_precision", "hallucination_rate"]:
                b_vals = [it.get(metric, 0.0) for it in sub_base]
                p_vals = [it.get(metric, 0.0) for it in sub_psd]
                stat_analysis[m_name]["by_source_bucket"][b][metric] = compute_paired_bootstrap_ci(b_vals, p_vals)

    for path in [os.path.join(ANALYSIS_DIR, "statistical_analysis.json"), os.path.join(OUTPUT_DIR, "statistical_analysis.json")]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stat_analysis, f, indent=2)

    print("Re-running figure generation...")
    from src.psd.plot_final_totto import plot_all
    plot_all()
    print("All updates complete!")


if __name__ == "__main__":
    main()
