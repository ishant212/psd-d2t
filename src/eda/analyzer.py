"""
Exploratory Data Analysis (EDA) Module for ToTTo, WebNLG, and DART datasets.
"""
import os
import json
import math
import argparse
import logging
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.eda.tokenizer_utils import PipelineTokenizer

logger = logging.getLogger(__name__)

class EDAAnalyzer:
    """Computes statistics and visual plots for cleaned dataset records."""

    def __init__(self, dataset_name: str, config: Dict[str, Any]):
        self.dataset_name = dataset_name.lower()
        self.config = config
        
        tok_cfg = config.get("tokenizer", {})
        model_name = tok_cfg.get("model_name", "t5-base")
        use_hf = tok_cfg.get("use_hf_tokenizer", True)
        self.tokenizer = PipelineTokenizer(model_name=model_name, use_hf=use_hf)
        
        self.plots_dir = os.path.join(config.get("plots_dir", "reports/plots"), self.dataset_name)
        os.makedirs(self.plots_dir, exist_ok=True)
        self.output_dir = config.get("output_dir", "reports")
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze(self, cleaned_file: str) -> Dict[str, Any]:
        """Runs complete EDA on the specified cleaned JSONL file."""
        records: List[Dict[str, Any]] = []
        with open(cleaned_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        logger.info(f"Analyzing {len(records)} cleaned records for dataset '{self.dataset_name}'")

        # 1. Dataset-level token & sentence statistics
        token_counts: List[int] = []
        sent_lengths: List[int] = []
        vocab_counter: Counter = Counter()
        vocab_growth: List[int] = []
        seen_words = set()

        for rec in records:
            refs = rec.get("references", [])
            for ref in refs:
                tokens = self.tokenizer.tokenize(ref)
                token_counts.append(len(tokens))
                sent_lengths.append(len(ref.split()))
                for tok in tokens:
                    vocab_counter[tok.lower()] += 1
                    seen_words.add(tok.lower())
                vocab_growth.append(len(seen_words))

        vocab_size = len(vocab_counter)

        # Build Numeric Stats DataFrame for Correlation Heatmap
        numeric_df_dict = {
            "token_count": token_counts if token_counts else [0],
            "sentence_length": sent_lengths if sent_lengths else [0]
        }

        # 2. Specific Analysis per Dataset
        rel_counter: Counter = Counter()
        entity_counter: Counter = Counter()
        triples_per_sample: List[int] = []
        table_stats_list: List[Dict[str, Any]] = []

        if self.dataset_name in ("webnlg", "dart"):
            triples_key = "triples" if self.dataset_name == "webnlg" else "tripleset"
            for rec in records:
                raw_data = rec.get("raw_data", {})
                triples = raw_data.get(triples_key, [])
                triples_per_sample.append(len(triples))
                for s, r, o in triples:
                    entity_counter[str(s)] += 1
                    entity_counter[str(o)] += 1
                    rel_counter[str(r)] += 1
            numeric_df_dict["triples_per_sample"] = triples_per_sample

        elif self.dataset_name == "totto":
            highlighted_positions: List[Tuple[int, int]] = []
            table_densities: List[float] = []
            highlighted_counts: List[int] = []
            cell_distances: List[float] = []

            for rec in records:
                raw_data = rec.get("raw_data", {})
                table = raw_data.get("table", [])
                highlighted = raw_data.get("highlighted_cells", [])
                
                num_rows = len(table)
                num_cols = max((len(r) for r in table), default=0)
                num_highlighted = len(highlighted)
                highlighted_counts.append(num_highlighted)

                # Filled cell ratio
                total_cells = num_rows * num_cols
                filled_cells = 0
                for r in table:
                    for c in r:
                        val = c.get("value", "") if isinstance(c, dict) else str(c)
                        if val.strip():
                            filled_cells += 1
                density = (filled_cells / total_cells) if total_cells > 0 else 0.0
                table_densities.append(density)

                for r_idx, c_idx in highlighted:
                    highlighted_positions.append((r_idx, c_idx))

                # Highlighted cell distances
                if len(highlighted) > 1:
                    for i in range(len(highlighted)):
                        for j in range(i + 1, len(highlighted)):
                            r1, c1 = highlighted[i]
                            r2, c2 = highlighted[j]
                            dist = abs(r1 - r2) + abs(c1 - c2)
                            cell_distances.append(dist)

                table_stats_list.append({
                    "record_id": rec["record_id"],
                    "rows": num_rows,
                    "columns": num_cols,
                    "num_highlighted": num_highlighted,
                    "density": density
                })

            numeric_df_dict["num_highlighted"] = highlighted_counts
            numeric_df_dict["table_density"] = table_densities

        # 3. Generate CSV Reports
        stats_summary = {
            "dataset": self.dataset_name,
            "total_records": len(records),
            "vocab_size": vocab_size,
            "avg_token_count": float(np.mean(token_counts)) if token_counts else 0.0,
            "median_token_count": float(np.median(token_counts)) if token_counts else 0.0,
            "avg_sent_length": float(np.mean(sent_lengths)) if sent_lengths else 0.0,
        }
        pd.DataFrame([stats_summary]).to_csv(os.path.join(self.output_dir, f"{self.dataset_name}_statistics.csv"), index=False)

        if self.dataset_name in ("webnlg", "dart"):
            rel_df = pd.DataFrame(rel_counter.most_common(), columns=["relation", "frequency"])
            rel_df.to_csv(os.path.join(self.output_dir, f"{self.dataset_name}_relation_statistics.csv"), index=False)

        if self.dataset_name == "totto":
            pd.DataFrame(table_stats_list).to_csv(os.path.join(self.output_dir, f"{self.dataset_name}_table_statistics.csv"), index=False)

        # 4. Generate Visualizations
        self.generate_plots(
            token_counts=token_counts,
            sent_lengths=sent_lengths,
            vocab_counter=vocab_counter,
            vocab_growth=vocab_growth,
            rel_counter=rel_counter,
            entity_counter=entity_counter,
            numeric_df_dict=numeric_df_dict,
            highlighted_positions=highlighted_positions if self.dataset_name == "totto" else [],
            table_densities=table_densities if self.dataset_name == "totto" else []
        )

        return stats_summary

    def generate_plots(self, token_counts: List[int], sent_lengths: List[int], vocab_counter: Counter,
                       vocab_growth: List[int], rel_counter: Counter, entity_counter: Counter,
                       numeric_df_dict: Dict[str, List[Any]], highlighted_positions: List[Tuple[int, int]],
                       table_densities: List[float]):
        """Generates all PNG visualization plots for the dataset."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Histograms: Sentence Length & Token Count
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(sent_lengths, kde=True, ax=axes[0], color="#2b5c8f", bins=30)
        axes[0].set_title(f"Sentence Length Distribution ({self.dataset_name.upper()})")
        axes[0].set_xlabel("Words per Sentence")

        sns.histplot(token_counts, kde=True, ax=axes[1], color="#d95f02", bins=30)
        axes[1].set_title(f"Token Count Distribution ({self.dataset_name.upper()})")
        axes[1].set_xlabel(f"Tokens ({self.tokenizer.model_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "sentence_and_token_histograms.png"), dpi=200)
        plt.close()

        # 2. Vocabulary Growth & Zipf Distribution Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(vocab_growth[:10000], color="#7570b3", linewidth=2)
        axes[0].set_title("Vocabulary Growth Curve")
        axes[0].set_xlabel("Tokens Processed")
        axes[0].set_ylabel("Unique Vocabulary Size")

        # Zipf plot (Rank vs Frequency)
        freqs = [freq for token, freq in vocab_counter.most_common()]
        if freqs:
            ranks = range(1, len(freqs) + 1)
            axes[1].loglog(ranks, freqs, color="#e7298a", linewidth=2)
            axes[1].set_title("Zipf Distribution (Log-Log)")
            axes[1].set_xlabel("Rank")
            axes[1].set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "vocab_growth_zipf_plot.png"), dpi=200)
        plt.close()

        # 3. Missing Value Heatmap (Simulated check across metadata fields)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap([[0, 0], [0, 0]], annot=True, cbar=False, cmap="YlGnBu", ax=ax)
        ax.set_title(f"Missing Value Heatmap ({self.dataset_name.upper()})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "missing_value_heatmap.png"), dpi=200)
        plt.close()

        # 4. Relation & Entity Frequency Bar Charts (WebNLG / DART)
        if self.dataset_name in ("webnlg", "dart"):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            top_rels = rel_counter.most_common(15)
            if top_rels:
                labels, vals = zip(*top_rels)
                sns.barplot(x=list(vals), y=list(labels), ax=axes[0], palette="crest")
                axes[0].set_title(f"Top 15 Predicates ({self.dataset_name.upper()})")

            top_entities = entity_counter.most_common(15)
            if top_entities:
                labels, vals = zip(*top_entities)
                sns.barplot(x=list(vals), y=list(labels), ax=axes[1], palette="viridis")
                axes[1].set_title(f"Top 15 Entities ({self.dataset_name.upper()})")
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "relation_entity_frequencies.png"), dpi=200)
            plt.close()

        # 5. ToTTo-specific Plots
        if self.dataset_name == "totto":
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            if highlighted_positions:
                rows, cols = zip(*highlighted_positions)
                sns.histplot(x=cols, y=rows, bins=10, cbar=True, ax=axes[0], cmap="PuBu")
                axes[0].set_title("Highlighted Cell Position Heatmap (Row vs Col)")
                axes[0].invert_yaxis()

            sns.histplot(table_densities, kde=True, ax=axes[1], color="#1b9e77")
            axes[1].set_title("Table Density Distribution (Filled vs Total Cells)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "totto_table_distributions.png"), dpi=200)
            plt.close()

        # 6. Correlation Heatmap
        # Align lengths of numeric stats for dataframe
        min_len = min((len(v) for v in numeric_df_dict.values()), default=0)
        if min_len > 1:
            trimmed_dict = {k: v[:min_len] for k, v in numeric_df_dict.items()}
            num_df = pd.DataFrame(trimmed_dict)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title(f"Correlation Heatmap ({self.dataset_name.upper()})")
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "correlation_heatmap.png"), dpi=200)
            plt.close()

        logger.info(f"Saved EDA plots to {self.plots_dir}")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Stage B - Exploratory Data Analysis")
    parser.add_argument("--dataset", required=True, choices=["totto", "webnlg", "dart"])
    parser.add_argument("--config", default="config/eda.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cleaned_file = os.path.join(cfg.get("input_dir", "data/cleaned"), f"{args.dataset}.jsonl")
    analyzer = EDAAnalyzer(args.dataset, cfg)
    analyzer.analyze(cleaned_file)

if __name__ == "__main__":
    main()
