"""Dataset preparation script for the Final ToTTo Long-Context Evaluation.

Extracts 160 stratified ToTTo examples (40 per bucket) from the project's
established validation pool, ensuring zero overlap with training data and
stratified sampling across token lengths within each bucket.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer
from src.data.unified_schema import UnifiedRecord
from src.preprocessing.serialize import Serializer

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "experiments", "final_totto_long_context", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "evaluation_set_160_totto.json")

CONTEXT_BUCKETS = ["257-512", "513-1024", "1025-2048", ">2048"]


def assign_context_bucket(tok_len: int) -> str:
    if tok_len <= 256:
        return "<=256"
    elif tok_len <= 512:
        return "257-512"
    elif tok_len <= 1024:
        return "513-1024"
    elif tok_len <= 2048:
        return "1025-2048"
    else:
        return ">2048"


def prepare_benchmark_160() -> List[Dict[str, Any]]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load train IDs to guarantee zero leakage
    train_path = os.path.join(PROJECT_ROOT, "data", "unified", "train.jsonl")
    train_ids = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                train_ids.add(json.loads(line_str)["id"])
    print(f"[DatasetBuilder] Loaded {len(train_ids)} train IDs for leakage exclusion.")

    # 2. Load validation records
    val_path = os.path.join(PROJECT_ROOT, "data", "unified", "validation.jsonl")
    val_records: List[UnifiedRecord] = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                val_records.append(UnifiedRecord.from_dict(json.loads(line_str)))

    totto_records = [r for r in val_records if r.dataset == "totto" and r.id not in train_ids]
    print(f"[DatasetBuilder] Loaded {len(totto_records)} candidate ToTTo validation records.")

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(PROJECT_ROOT, "models", "qwen3_4b_hf"))
    serializer = Serializer()

    # 3. Categorize into long-context buckets
    bucket_candidates: Dict[str, List[Dict[str, Any]]] = {b: [] for b in CONTEXT_BUCKETS}

    for r in totto_records:
        serialized = serializer.serialize_record(r)
        tok_len = len(tokenizer(serialized, add_special_tokens=False)["input_ids"])
        b = assign_context_bucket(tok_len)
        if b in CONTEXT_BUCKETS:
            bucket_candidates[b].append({
                "record": r,
                "token_length": tok_len,
                "serialized_text": serialized,
            })

    for b in CONTEXT_BUCKETS:
        print(f"[DatasetBuilder] Bucket '{b}' has {len(bucket_candidates[b])} candidate instances.")
        assert len(bucket_candidates[b]) >= 40, f"Bucket {b} has fewer than 40 instances ({len(bucket_candidates[b])})"

    # 4. Stratified selection: Sort by length and pick 40 linspace quantiles
    selected_items: List[Dict[str, Any]] = []

    for b in CONTEXT_BUCKETS:
        pool = bucket_candidates[b]
        pool.sort(key=lambda x: x["token_length"])
        
        # Select 40 evenly distributed indices
        indices = np.linspace(0, len(pool) - 1, 40, dtype=int)
        chosen = [pool[idx] for idx in indices]

        lengths = [x["token_length"] for x in chosen]
        print(f"  Selected 40 for '{b}': min={min(lengths)}, median={int(np.median(lengths))}, max={max(lengths)}")

        for item in chosen:
            r = item["record"]
            selected_items.append({
                "id": r.id,
                "dataset": r.dataset,
                "input_type": r.input_type,
                "split": r.split,
                "original_token_length": item["token_length"],
                "context_bucket": b,
                "target_text": r.target_text,
                "reference_texts": r.reference_texts,
                "serialized_text": item["serialized_text"],
                "nodes": r.nodes,
                "edges": r.edges,
            })

    assert len(selected_items) == 160, f"Expected 160, got {len(selected_items)}"

    # Check leakage again
    selected_ids = set(x["id"] for x in selected_items)
    overlap = selected_ids.intersection(train_ids)
    assert len(overlap) == 0, f"Critical error: {len(overlap)} overlapping IDs with train set!"

    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(selected_items, f, indent=2)

    print(f"\n[DatasetBuilder] Successfully generated and verified {len(selected_items)} ToTTo evaluation instances.")
    print(f"[DatasetBuilder] Saved to: {OUTPUT_FILE}")
    return selected_items


if __name__ == "__main__":
    prepare_benchmark_160()
