"""
Loader for the DART dataset (JSON format).
"""
import json
import logging
from typing import Any, Dict, List, Optional
from .base_loader import BaseLoader, RawRecord

logger = logging.getLogger(__name__)

class DARTloader(BaseLoader):
    """Loader for DART dataset files (JSON or XML format)."""

    def load(self, file_path: str) -> List[RawRecord]:
        records: List[RawRecord] = []
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for idx, raw_obj in enumerate(data):
                        record = self.parse_single_record(raw_obj, idx)
                        records.append(record)
            except Exception as e:
                logger.warning(f"Error loading DART JSON file {file_path}: {e}")
        return records

    def parse_single_record(self, raw_obj: Dict[str, Any], idx: Optional[int] = None) -> RawRecord:
        record_id = f"dart_{idx if idx is not None else 'idx'}"
        
        # Tripleset parsing
        tripleset = raw_obj.get("tripleset", [])
        
        # Annotations (reference texts)
        references: List[str] = []
        annotations = raw_obj.get("annotations", [])
        if isinstance(annotations, list):
            for ann in annotations:
                if isinstance(ann, dict) and "text" in ann:
                    references.append(ann["text"])
                elif isinstance(ann, str):
                    references.append(ann)

        metadata = {
            "subtree_was_extended": raw_obj.get("subtree_was_extended", False),
            "sources": [ann.get("source", "") for ann in annotations if isinstance(ann, dict)]
        }

        return RawRecord(
            record_id=record_id,
            dataset="dart",
            raw_data=raw_obj,
            references=references,
            metadata=metadata
        )
