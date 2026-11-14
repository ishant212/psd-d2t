"""Length and Long-Context Analysis for Phase 1 EDA.

Analyzes input and output sequence length distributions, percentiles (50th, 90th, 95th, 99th),
and exact long-context breakdown (>512, >1024, >2048, >4096, >8192 tokens).
"""

from typing import List, Dict, Any, Optional
import numpy as np
from src.data.unified_schema import UnifiedRecord
from src.preprocessing.serialize import Serializer


class LengthAnalyzer:
    """Computes token and word length statistics and long-context distributions."""

    def __init__(self, tokenizer: Any = None, thresholds: Optional[List[int]] = None):
        self.tokenizer = tokenizer
        self.thresholds = thresholds or [512, 1024, 2048, 4096, 8192]
        self.serializer = Serializer()

    def _get_token_length(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer(text, add_special_tokens=True, truncation=False)
                return len(tokens["input_ids"])
            except Exception:
                pass
        # Whitespace word fallback
        return len(text.strip().split())

    def analyze(self, records: List[UnifiedRecord], dataset_name: str) -> Dict[str, Any]:
        """Computes statistical metrics for serialized input and target text lengths."""
        input_lengths: List[int] = []
        target_lengths: List[int] = []

        for r in records:
            ser_text = self.serializer.serialize_record(r)
            in_len = self._get_token_length(ser_text)
            out_len = self._get_token_length(r.target_text)

            input_lengths.append(in_len)
            target_lengths.append(out_len)

        def compute_stats(lengths: List[int], prefix: str) -> Dict[str, Any]:
            if not lengths:
                return {
                    f"{prefix}_min": 0,
                    f"{prefix}_mean": 0.0,
                    f"{prefix}_median": 0.0,
                    f"{prefix}_std": 0.0,
                    f"{prefix}_p90": 0.0,
                    f"{prefix}_p95": 0.0,
                    f"{prefix}_p99": 0.0,
                    f"{prefix}_max": 0
                }
            arr = np.array(lengths)
            return {
                f"{prefix}_min": int(np.min(arr)),
                f"{prefix}_mean": float(np.mean(arr)),
                f"{prefix}_median": float(np.median(arr)),
                f"{prefix}_std": float(np.std(arr)),
                f"{prefix}_p90": float(np.percentile(arr, 90)),
                f"{prefix}_p95": float(np.percentile(arr, 95)),
                f"{prefix}_p99": float(np.percentile(arr, 99)),
                f"{prefix}_max": int(np.max(arr))
            }

        input_stats = compute_stats(input_lengths, "input")
        target_stats = compute_stats(target_lengths, "target")

        # Long-context threshold statistics
        long_context_counts = {}
        total = len(input_lengths)
        for thresh in self.thresholds:
            count_above = sum(1 for l in input_lengths if l > thresh)
            pct = (count_above / total * 100.0) if total > 0 else 0.0
            long_context_counts[f"pct_gt_{thresh}"] = float(pct)
            long_context_counts[f"count_gt_{thresh}"] = count_above

        return {
            "dataset": dataset_name,
            "total_analyzed": total,
            **input_stats,
            **target_stats,
            **long_context_counts,
            "raw_input_lengths": input_lengths,
            "raw_target_lengths": target_lengths
        }
