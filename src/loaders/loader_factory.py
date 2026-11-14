"""
Factory for auto-detecting dataset format and instantiating the correct loader.
"""
import os
import json
import xml.etree.ElementTree as ET
import logging
from typing import Any, Dict, Optional, Type
from .base_loader import BaseLoader
from .totto_loader import ToTToloader
from .webnlg_loader import WebNLGloader
from .dart_loader import DARTloader

logger = logging.getLogger(__name__)

class LoaderFactory:
    """Factory for dataset loader auto-detection and instantiation."""

    LOADERS: Dict[str, Type[BaseLoader]] = {
        "totto": ToTToloader,
        "webnlg": WebNLGloader,
        "dart": DARTloader,
    }

    @classmethod
    def get_loader(cls, dataset_type: str) -> BaseLoader:
        """Gets loader instance by explicit dataset name."""
        dataset_type = dataset_type.lower()
        if dataset_type not in cls.LOADERS:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Supported: {list(cls.LOADERS.keys())}")
        return cls.LOADERS[dataset_type]()

    @classmethod
    def detect_dataset_type(cls, file_or_dir_path: str) -> str:
        """
        Auto-detects dataset type by inspecting file path, file extension, 
        and first sample contents.
        """
        path_lower = file_or_dir_path.lower()
        if "totto" in path_lower:
            return "totto"
        if "webnlg" in path_lower:
            return "webnlg"
        if "dart" in path_lower:
            return "dart"

        # Content-based schema detection
        if os.path.isdir(file_or_dir_path):
            # Check files in directory
            for root, _, files in os.walk(file_or_dir_path):
                for f in files:
                    if f.endswith(".xml"):
                        return "webnlg"
                    elif f.endswith(".jsonl"):
                        return "totto"
                    elif f.endswith(".json"):
                        return "dart"

        if os.path.isfile(file_or_dir_path):
            if file_or_dir_path.endswith(".xml"):
                return "webnlg"
            elif file_or_dir_path.endswith(".jsonl"):
                # Read first line
                with open(file_or_dir_path, "r", encoding="utf-8") as f:
                    line = f.readline()
                    if "highlighted_cells" in line or "table_page_title" in line:
                        return "totto"
            elif file_or_dir_path.endswith(".json"):
                with open(file_or_dir_path, "r", encoding="utf-8") as f:
                    content = f.read(2048)
                    if "tripleset" in content or "subtree_was_extended" in content:
                        return "dart"
                    elif "entries" in content or "modifiedtripleset" in content:
                        return "webnlg"

        raise ValueError(f"Could not auto-detect dataset type for path: {file_or_dir_path}")

    @classmethod
    def auto_load(cls, file_or_dir_path: str) -> BaseLoader:
        """Detects dataset format and returns the appropriate loader."""
        dataset_type = cls.detect_dataset_type(file_or_dir_path)
        logger.info(f"Auto-detected dataset '{dataset_type}' for path: {file_or_dir_path}")
        return cls.get_loader(dataset_type)
