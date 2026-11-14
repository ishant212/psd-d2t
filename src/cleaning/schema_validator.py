"""
Schema Validator for ToTTo, WebNLG, and DART raw records.
"""
import logging
from typing import Dict, List, Tuple, Any
from src.loaders.base_loader import RawRecord

logger = logging.getLogger(__name__)

class SchemaValidator:
    """Validates raw records against dataset-specific structural schemas."""

    @staticmethod
    def validate(record: RawRecord) -> Tuple[bool, List[str]]:
        """
        Validates record structure.
        Returns: (is_valid: bool, errors: List[str])
        """
        errors: List[str] = []
        if not record.record_id:
            errors.append("MISSING_RECORD_ID")

        if record.dataset == "totto":
            table = record.raw_data.get("table")
            if not isinstance(table, list):
                errors.append("MALFORMED_TABLE_NOT_LIST")
            elif len(table) == 0:
                errors.append("EMPTY_TABLE")
            else:
                for row_idx, row in enumerate(table):
                    if not isinstance(row, list):
                        errors.append(f"MALFORMED_ROW_{row_idx}_NOT_LIST")
                        break

        elif record.dataset in ("webnlg", "dart"):
            triples = record.raw_data.get("triples", record.raw_data.get("tripleset"))
            if not isinstance(triples, list):
                errors.append("MALFORMED_TRIPLES_NOT_LIST")
            elif len(triples) == 0:
                errors.append("EMPTY_TRIPLESET")
            else:
                for idx, t in enumerate(triples):
                    if not isinstance(t, (list, tuple)) or len(t) != 3:
                        errors.append(f"INVALID_TRIPLE_FORMAT_INDEX_{idx}")
                        break

        # Check references
        if not record.references or not any(r.strip() for r in record.references if isinstance(r, str)):
            errors.append("MISSING_REFERENCES")

        is_valid = len(errors) == 0
        return is_valid, errors
