import os
import glob
import json

def summarize():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")
    for suite in ["long_context", "final_evaluation"]:
        suite_path = os.path.join(base_dir, suite)
        if not os.path.exists(suite_path):
            continue
        metric_files = sorted(glob.glob(os.path.join(suite_path, "*", "metrics.json")))
        if not metric_files:
            continue
        print(f"\n### {suite.replace('_', ' ').title()} Suite ({len(metric_files)} runs)")
        print(f"| Model / Mode | BLEU-4 | ROUGE-L | METEOR | BERTScore F1 | Coverage | Factual Prec. | Halluc. Rate |")
        print(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for mf in metric_files:
            exp_name = os.path.basename(os.path.dirname(mf))
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                b = f"{data.get('bleu', 0.0):.2f}"
                r = f"{data.get('rougeL', 0.0):.2f}"
                m = f"{data.get('meteor', 0.0):.2f}"
                bs = f"{data.get('bertscore_f1', 0.0):.2f}"
                cov = f"{data.get('structured_coverage', 0.0):.2f}%"
                fp = f"{data.get('factual_precision', 0.0):.2f}%"
                hr = f"{data.get('hallucination_rate', 0.0):.2f}%"
                print(f"| `{exp_name}` | **{b}** | **{r}** | {m} | {bs} | {cov} | **{fp}** | **{hr}** |")
            except Exception as e:
                print(f"| `{exp_name}` | Error loading: {e} | | | | | | |")

if __name__ == "__main__":
    summarize()
