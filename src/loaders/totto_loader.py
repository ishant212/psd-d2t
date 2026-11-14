"""
Loader for the ToTTo dataset (JSONL format).
"""
import json
import logging
from typing import Any, Dict, List, Optional
from .base_loader import BaseLoader, RawRecord

logger = logging.getLogger(__name__)

class ToTToloader(BaseLoader):
    """Loader for ToTTo JSONL dataset files."""
    
    def load(self, file_path: str) -> List[RawRecord]:
        records: List[RawRecord] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = self.parse_single_record(data, line_idx)
                    records.append(record)
                except Exception as e:
                    logger.warning(f"Error parsing ToTTo line {line_idx} in {file_path}: {e}")
        return records

    def parse_single_record(self, raw_obj: Dict[str, Any], idx: Optional[int] = None) -> RawRecord:
        record_id = str(raw_obj.get("example_id", idx if idx is not None else "unknown"))
        
        # Extract reference sentences
        references: List[str] = []
        sentence_annotations = raw_obj.get("sentence_annotations", [])
        if isinstance(sentence_annotations, list):
            for ann in sentence_annotations:
                if isinstance(ann, dict) and "final_sentence" in ann:
                    references.append(ann["final_sentence"])
                elif isinstance(ann, str):
                    references.append(ann)

        metadata = {
            "table_page_title": raw_obj.get("table_page_title", ""),
            "table_section_title": raw_obj.get("table_section_title", ""),
            "table_section_text": raw_obj.get("table_section_text", ""),
            "table_webpage_url": raw_obj.get("table_webpage_url", ""),
            "highlighted_cells": raw_obj.get("highlighted_cells", []),
            "overlap_subset": raw_obj.get("overlap_subset", False)
        }

        return RawRecord(
            record_id=record_id,
            dataset="totto",
            raw_data=raw_obj,
            references=references,
            metadata=metadata
        )
