"""
Loader for the WebNLG 2020 dataset (XML and JSON formats).
"""
import os
import xml.etree.ElementTree as ET
import json
import logging
from typing import Any, Dict, List, Optional
from .base_loader import BaseLoader, RawRecord

logger = logging.getLogger(__name__)

class WebNLGloader(BaseLoader):
    """Loader for WebNLG 2020 dataset files (XML or JSON)."""

    def load(self, path: str) -> List[RawRecord]:
        records: List[RawRecord] = []
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".xml") or file.endswith(".json"):
                        full_path = os.path.join(root, file)
                        records.extend(self._load_file(full_path))
        elif os.path.isfile(path):
            records.extend(self._load_file(path))
        return records

    def _load_file(self, file_path: str) -> List[RawRecord]:
        records: List[RawRecord] = []
        if file_path.endswith(".xml"):
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                entries = root.findall(".//entry")
                for idx, entry in enumerate(entries):
                    rec = self._parse_xml_entry(entry, file_path, idx)
                    if rec:
                        records.append(rec)
            except Exception as e:
                logger.warning(f"Failed to parse XML file {file_path}: {e}")
        elif file_path.endswith(".json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries = data.get("entries", data) if isinstance(data, dict) else data
                    for idx, entry in enumerate(entries):
                        records.append(self.parse_single_record(entry, idx))
            except Exception as e:
                logger.warning(f"Failed to parse JSON file {file_path}: {e}")
        return records

    def _parse_xml_entry(self, entry_elem: ET.Element, file_path: str, idx: int) -> RawRecord:
        eid = entry_elem.attrib.get("eid", f"webnlg_{idx}")
        category = entry_elem.attrib.get("category", "")
        size = entry_elem.attrib.get("size", "")
        
        # Unique ID combining category/file if eid is simple
        file_base = os.path.basename(file_path).replace(".xml", "")
        record_id = f"{file_base}_{category}_{eid}_{idx}"

        # Parse modifiedtripleset (or originaltripleset)
        triples: List[List[str]] = []
        mtriple_set = entry_elem.find("modifiedtripleset")
        if mtriple_set is None:
            mtriple_set = entry_elem.find("originaltripleset")
            
        if mtriple_set is not None:
            for elem in mtriple_set:
                text = (elem.text or "").strip()
                if text:
                    parts = [p.strip() for p in text.split("|")]
                    if len(parts) == 3:
                        triples.append(parts)

        # Parse reference lexicalizations
        references: List[str] = []
        for lex in entry_elem.findall("lex"):
            if lex.text:
                references.append(lex.text.strip())

        raw_obj = {
            "triples": triples,
            "category": category,
            "size": size,
            "eid": eid
        }

        return RawRecord(
            record_id=record_id,
            dataset="webnlg",
            raw_data=raw_obj,
            references=references,
            metadata={"category": category, "size": size, "file_source": file_path}
        )

    def parse_single_record(self, raw_obj: Dict[str, Any], idx: Optional[int] = None) -> RawRecord:
        record_id = str(raw_obj.get("id", raw_obj.get("eid", idx if idx is not None else "webnlg_unknown")))
        triples = raw_obj.get("triples", raw_obj.get("modifiedtripleset", []))
        
        references: List[str] = []
        lex_data = raw_obj.get("lexicalisations", raw_obj.get("lex", []))
        if isinstance(lex_data, list):
            for item in lex_data:
                if isinstance(item, str):
                    references.append(item)
                elif isinstance(item, dict) and "text" in item:
                    references.append(item["text"])

        return RawRecord(
            record_id=record_id,
            dataset="webnlg",
            raw_data=raw_obj,
            references=references,
            metadata={"category": raw_obj.get("category", "")}
        )
