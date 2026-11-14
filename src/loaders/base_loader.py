"""
Base Loader interface and RawRecord dataclass definition.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class RawRecord:
    """Represents a raw, uncleaned record from a native dataset."""
    record_id: str
    dataset: str  # 'totto', 'webnlg', 'dart'
    raw_data: Dict[str, Any]
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseLoader(ABC):
    """Abstract base loader interface for parsing dataset files."""
    
    @abstractmethod
    def load(self, file_path: str) -> List[RawRecord]:
        """Loads and parses raw records from the specified file path."""
        pass
    
    @abstractmethod
    def parse_single_record(self, raw_obj: Any, idx: Optional[int] = None) -> RawRecord:
        """Parses a single raw object (dict/XML element) into a RawRecord."""
        pass
