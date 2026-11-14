"""PyTorch Dataset for Unified Data-to-Text Records.

Loads unified records from JSONL, serializes their structured input deterministically,
and encodes them into model input tensors with attention masks and target labels.
"""

import json
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from src.data.unified_schema import UnifiedRecord
from src.preprocessing.serialize import Serializer


class D2TDataset(Dataset):
    """PyTorch Dataset for serialized Data-to-Text records."""

    def __init__(
        self,
        records: List[UnifiedRecord],
        tokenizer: Any,
        max_input_length: int = 512,
        max_target_length: int = 128,
        is_training: bool = True
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.is_training = is_training
        self.serializer = Serializer()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        ser_text = self.serializer.serialize_record(record)
        target_text = record.target_text

        # Tokenize input
        inputs = self.tokenizer(
            ser_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {
            "id": record.id,
            "dataset": record.dataset,
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "source_text": ser_text,
            "target_text": target_text,
            "references": record.reference_texts or [target_text]
        }

        # Tokenize target labels if present
        if target_text:
            targets = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            labels = targets["input_ids"].squeeze(0)
            # Replace pad token id with -100 to ignore in loss
            labels[labels == self.tokenizer.pad_token_id] = -100
            item["labels"] = labels
        else:
            item["labels"] = torch.zeros(self.max_target_length, dtype=torch.long)
        return item

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: str,
        tokenizer: Any,
        max_input_length: int = 512,
        max_target_length: int = 128,
        max_records: Optional[int] = None
    ) -> "D2TDataset":
        """Loads records from a JSONL file and instantiates D2TDataset."""
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_records is not None and len(records) >= max_records:
                    break
                line = line.strip()
                if line:
                    data = json.loads(line)
                    records.append(UnifiedRecord.from_dict(data))
        return cls(records, tokenizer, max_input_length, max_target_length)


def d2t_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collator that batches tensors and preserves variable-length strings and reference lists."""
    return {
        "id": [item["id"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "source_text": [item["source_text"] for item in batch],
        "target_text": [item["target_text"] for item in batch],
        "references": [item["references"] for item in batch]
    }
