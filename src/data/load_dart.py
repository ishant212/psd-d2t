"""DART Dataset Loader.

Loads RDF triples and tables from official DART JSON files, preserving
tripleset structure, sources, subtree extensions, and reference annotations.
"""

import json
import os
import logging
from typing import List, Dict, Any, Generator, Optional

logger = logging.getLogger(__name__)


class DARTLoader:
    """Loader for the DART dataset."""

    def __init__(self, raw_dir: Optional[str] = None):
        self.raw_dir = raw_dir or "dart-master/data/v1.1.1"

    def _resolve_path(self, split: str, filepath: Optional[str] = None) -> str:
        if filepath and os.path.exists(filepath):
            return filepath

        split_map = {
            "train": [
                "dart-master/data/v1.1.1/dart-v1.1.1-full-train.json",
                "data/raw/dart/dart-v1.1.1-full-train.json",
                "dart-master/data/v1.1.1/dart_train.json"
            ],
            "validation": [
                "dart-master/data/v1.1.1/dart-v1.1.1-full-dev.json",
                "data/raw/dart/dart-v1.1.1-full-dev.json",
                "dart-master/data/v1.1.1/dart_dev.json"
            ],
            "dev": [
                "dart-master/data/v1.1.1/dart-v1.1.1-full-dev.json",
                "data/raw/dart/dart-v1.1.1-full-dev.json"
            ],
            "test": [
                "dart-master/data/v1.1.1/dart-v1.1.1-full-test.json",
                "data/raw/dart/dart-v1.1.1-full-test.json",
                "dart-master/data/v1.1.1/dart_test.json"
            ]
        }

        candidates = split_map.get(split.lower(), [])
        if self.raw_dir:
            cand_split = "dev" if split in ("validation", "dev") else split
            candidates.insert(0, os.path.join(self.raw_dir, f"dart-v1.1.1-full-{cand_split}.json"))
            candidates.insert(1, os.path.join(self.raw_dir, f"dart_{cand_split}.json"))

        for cand in candidates:
            if os.path.exists(cand):
                return cand

        raise FileNotFoundError(f"DART data for split '{split}' not found in candidates: {candidates}")

    def load_split(self, split: str = "validation", filepath: Optional[str] = None, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Loads all DART records for a given split into memory."""
        records = list(self.stream_split(split=split, filepath=filepath, max_examples=max_examples))
        logger.info(f"Loaded {len(records)} records from DART '{split}' split.")
        return records

    def stream_split(self, split: str = "validation", filepath: Optional[str] = None, max_examples: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """Streams DART records for a given split from disk."""
        target_path = self._resolve_path(split, filepath)
        norm_split = "validation" if split in ("validation", "dev") else split

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for idx, entry in enumerate(data):
            if max_examples is not None and count >= max_examples:
                break

            tripleset = entry.get("tripleset", [])
            annotations = entry.get("annotations", [])
            
            references = []
            sources = []
            for annot in annotations:
                text = annot.get("text", "").strip()
                if text:
                    references.append(text)
                src = annot.get("source", "").strip()
                if src:
                    sources.append(src)

            target_text = references[0] if references else ""
            primary_source = sources[0] if sources else "unknown"

            record = {
                "id": f"dart_{norm_split}_{idx}",
                "example_id": idx,
                "dataset": "dart",
                "split": norm_split,
                "tripleset": tripleset,
                "annotations": annotations,
                "reference_texts": references,
                "target_text": target_text,
                "source": primary_source,
                "sources": list(set(sources)),
                "subtree_was_extended": entry.get("subtree_was_extended", False)
            }
            yield record
            count += 1
