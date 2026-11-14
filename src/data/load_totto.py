"""ToTTo Dataset Loader.

Loads Wikipedia table records from official ToTTo JSONL files, preserving
exact table matrices, cell metadata, highlighted cell coordinates, section titles,
page titles, and reference sentences without irreversible flattening.
"""

import json
import os
import logging
from typing import List, Dict, Any, Generator, Optional

logger = logging.getLogger(__name__)


class ToTToLoader:
    """Loader for the ToTTo dataset."""

    def __init__(self, raw_dir: Optional[str] = None):
        self.raw_dir = raw_dir or "totto_data"

    def _resolve_path(self, split: str, filepath: Optional[str] = None) -> str:
        if filepath and os.path.exists(filepath):
            return filepath
        
        split_map = {
            "train": ["totto_data/totto_train_data.jsonl", "data/raw/totto/totto_train_data.jsonl"],
            "validation": ["totto_data/totto_dev_data.jsonl", "data/raw/totto/totto_dev_data.jsonl", "totto_data/totto_val_data.jsonl"],
            "dev": ["totto_data/totto_dev_data.jsonl", "data/raw/totto/totto_dev_data.jsonl"],
            "test": ["totto_data/unlabeled_totto_test_data.jsonl", "data/raw/totto/unlabeled_totto_test_data.jsonl"]
        }
        
        candidates = split_map.get(split.lower(), [])
        if self.raw_dir:
            candidates.insert(0, os.path.join(self.raw_dir, f"totto_{split}_data.jsonl"))
            candidates.insert(1, os.path.join(self.raw_dir, f"totto_{split}.jsonl"))
            if split in ("validation", "dev"):
                candidates.insert(0, os.path.join(self.raw_dir, "totto_dev_data.jsonl"))

        for cand in candidates:
            if os.path.exists(cand):
                return cand
                
        raise FileNotFoundError(f"ToTTo data for split '{split}' not found in candidates: {candidates}")

    def load_split(self, split: str = "validation", filepath: Optional[str] = None, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Loads all records for a given split into memory."""
        records = list(self.stream_split(split=split, filepath=filepath, max_examples=max_examples))
        logger.info(f"Loaded {len(records)} records from ToTTo '{split}' split.")
        return records

    def stream_split(self, split: str = "validation", filepath: Optional[str] = None, max_examples: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """Streams records for a given split from disk."""
        target_path = self._resolve_path(split, filepath)
        norm_split = "validation" if split in ("validation", "dev") else split
        
        count = 0
        with open(target_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if max_examples is not None and count >= max_examples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON line {line_idx} in {target_path}: {e}")
                    continue

                # Extract reference sentences
                sentence_annotations = data.get("sentence_annotations", [])
                references = []
                for annot in sentence_annotations:
                    final_sent = annot.get("final_sentence", "").strip()
                    if final_sent:
                        references.append(final_sent)

                target_text = references[0] if references else ""

                record = {
                    "id": f"totto_{norm_split}_{data.get('example_id', line_idx)}",
                    "example_id": data.get("example_id", line_idx),
                    "dataset": "totto",
                    "split": norm_split,
                    "table": data.get("table", []),
                    "highlighted_cells": data.get("highlighted_cells", []),
                    "table_page_title": data.get("table_page_title", ""),
                    "table_section_title": data.get("table_section_title", ""),
                    "table_section_text": data.get("table_section_text", ""),
                    "table_webpage_url": data.get("table_webpage_url", ""),
                    "target_text": target_text,
                    "reference_texts": references,
                    "sentence_annotations": sentence_annotations,
                    "overlap_subset": data.get("overlap_subset", False)
                }
                yield record
                count += 1
