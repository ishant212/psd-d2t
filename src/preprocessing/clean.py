"""Conservative Data Cleaner.

Applies conservative, safe preprocessing:
- Unicode NFC normalization
- Non-printable control character removal
- Whitespace normalization
- Schema integrity and empty input/target validation
- Detailed logging of any dropped or modified records with reason codes.
"""

import unicodedata
import re
import logging
from typing import Dict, Any, Tuple, Optional, List
from src.data.unified_schema import UnifiedRecord

logger = logging.getLogger(__name__)


def clean_text(text: Optional[str]) -> str:
    """Safely normalizes Unicode, removes non-printable characters, collapses whitespace."""
    if not text:
        return ""
    # Unicode NFC normalization
    norm = unicodedata.normalize("NFC", str(text))
    # Replace zero-width spaces, special NBSPs, and non-printable control chars
    norm = re.sub(r"[\u200b\u200c\u200d\ufeff\xa0]", " ", norm)
    norm = "".join(ch for ch in norm if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", " "))
    # Collapse multiple whitespace characters (spaces, tabs, newlines) to single space
    norm = re.sub(r"\s+", " ", norm)
    return norm.strip()


class DataCleaner:
    """Applies conservative cleaning and integrity validation to UnifiedRecords."""

    def __init__(self, drop_empty_targets: bool = False, drop_empty_inputs: bool = True):
        self.drop_empty_targets = drop_empty_targets
        self.drop_empty_inputs = drop_empty_inputs
        self.log_entries: List[Dict[str, Any]] = []

    def clean_record(self, record: UnifiedRecord) -> Tuple[Optional[UnifiedRecord], Optional[str]]:
        """Cleans a UnifiedRecord.

        Returns (cleaned_record, action_reason) or (None, drop_reason).
        """
        # Validate ID
        if not record.id:
            reason = "DROP_MISSING_ID"
            self._log_action(record.id or "unknown", record.dataset, "DROP", reason)
            return None, reason

        # Clean target text & references
        cleaned_target = clean_text(record.target_text)
        cleaned_refs = [clean_text(r) for r in record.reference_texts if clean_text(r)]

        if not cleaned_refs and cleaned_target:
            cleaned_refs = [cleaned_target]
        elif cleaned_refs and not cleaned_target:
            cleaned_target = cleaned_refs[0]

        # Check empty target (for test sets without labels, target_text may be empty)
        if self.drop_empty_targets and not cleaned_target and record.split != "test":
            reason = "DROP_EMPTY_TARGET"
            self._log_action(record.id, record.dataset, "DROP", reason)
            return None, reason

        # Clean nodes
        cleaned_nodes = []
        for n in record.nodes:
            c_label = clean_text(n.get("label", ""))
            c_node = dict(n)
            c_node["label"] = c_label
            cleaned_nodes.append(c_node)

        # Clean edges
        cleaned_edges = []
        for e in record.edges:
            c_rel = clean_text(e.get("relation", ""))
            c_edge = dict(e)
            c_edge["relation"] = c_rel
            cleaned_edges.append(c_edge)

        # Check empty structured input
        if self.drop_empty_inputs and not cleaned_nodes:
            reason = "DROP_EMPTY_STRUCTURED_INPUT"
            self._log_action(record.id, record.dataset, "DROP", reason)
            return None, reason

        # Update record
        record.target_text = cleaned_target
        record.reference_texts = cleaned_refs
        record.nodes = cleaned_nodes
        record.edges = cleaned_edges

        # Clean metadata strings
        cleaned_meta = {}
        for k, v in record.metadata.items():
            if isinstance(v, str):
                cleaned_meta[k] = clean_text(v)
            else:
                cleaned_meta[k] = v
        record.metadata = cleaned_meta

        return record, "CLEAN_SUCCESS"

    def _log_action(self, record_id: str, dataset: str, action: str, reason: str):
        entry = {
            "record_id": record_id,
            "dataset": dataset,
            "action": action,
            "reason": reason
        }
        self.log_entries.append(entry)
        if action == "DROP":
            logger.warning(f"Cleaner [{dataset}] Record {record_id} DROPPED: {reason}")
