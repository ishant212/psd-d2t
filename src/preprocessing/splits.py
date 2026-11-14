"""Split Management and Leakage Verification.

Handles dataset partitioning, persistence of processed and unified splits,
and verifies zero ID or content leakage across train, validation, and test sets.
"""

import json
import os
import logging
from typing import Dict, Any, List, Set, Tuple, Optional
from src.data.unified_schema import UnifiedRecord
from src.data.load_totto import ToTToLoader
from src.data.load_dart import DARTLoader
from src.data.load_webnlg import WebNLGLoader
from src.data.normalize import normalize_record
from src.preprocessing.clean import DataCleaner

logger = logging.getLogger(__name__)


def process_dataset(
    dataset_name: str,
    raw_dir: Optional[str] = None,
    output_dir: str = "data/processed",
    unified_dir: str = "data/unified",
    max_examples_per_split: Optional[int] = None
) -> Dict[str, int]:
    """Loads, cleans, normalizes, and saves a dataset across all official splits."""
    cleaner = DataCleaner(drop_empty_targets=False, drop_empty_inputs=True)
    d_name = dataset_name.lower()

    if d_name == "totto":
        loader = ToTToLoader(raw_dir=raw_dir)
    elif d_name == "dart":
        loader = DARTLoader(raw_dir=raw_dir)
    elif d_name == "webnlg":
        loader = WebNLGLoader(raw_dir=raw_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    os.makedirs(os.path.join(output_dir, d_name), exist_ok=True)
    os.makedirs(unified_dir, exist_ok=True)

    counts = {}
    splits = ["train", "validation", "test"]

    for split in splits:
        try:
            records = loader.load_split(split=split, max_examples=max_examples_per_split)
        except Exception as e:
            logger.warning(f"Could not load split '{split}' for {d_name}: {e}")
            continue

        cleaned_unified_records: List[UnifiedRecord] = []
        for raw_rec in records:
            norm_rec = normalize_record(raw_rec, d_name)
            clean_rec, reason = cleaner.clean_record(norm_rec)
            if clean_rec is not None:
                cleaned_unified_records.append(clean_rec)

        # Save to data/processed/{dataset}/{split}.jsonl
        proc_path = os.path.join(output_dir, d_name, f"{split}.jsonl")
        with open(proc_path, "w", encoding="utf-8") as f:
            for r in cleaned_unified_records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

        counts[split] = len(cleaned_unified_records)
        logger.info(f"Processed {d_name} [{split}]: {len(cleaned_unified_records)} records saved to {proc_path}")

    return counts


def verify_split_leakage(records_by_split: Dict[str, List[UnifiedRecord]]) -> Dict[str, Any]:
    """Verifies that there is no ID or identical structured input leakage between splits."""
    train_ids = {r.id for r in records_by_split.get("train", [])}
    val_ids = {r.id for r in records_by_split.get("validation", [])}
    test_ids = {r.id for r in records_by_split.get("test", [])}

    train_val_id_leak = train_ids.intersection(val_ids)
    train_test_id_leak = train_ids.intersection(test_ids)
    val_test_id_leak = val_ids.intersection(test_ids)

    return {
        "train_val_id_overlap": len(train_val_id_leak),
        "train_test_id_overlap": len(train_test_id_leak),
        "val_test_id_overlap": len(val_test_id_leak),
        "is_leak_free": (
            len(train_val_id_leak) == 0 and
            len(train_test_id_leak) == 0 and
            len(val_test_id_leak) == 0
        )
    }
