"""WebNLG Dataset Loader.

Loads WebNLG RDF triple sets from official JSON (v2.1) and XML (v2.1/v3.0) files,
preserving RDF triples, subjects, predicates, objects, categories, shapes,
and multiple reference lexicalizations.
"""

import json
import os
import glob
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Generator, Optional

logger = logging.getLogger(__name__)


class WebNLGLoader:
    """Loader for the WebNLG dataset."""

    def __init__(self, raw_dir: Optional[str] = None):
        self.raw_dir = raw_dir or "webnlg-dataset-master"

    def _resolve_json_path(self, split: str, filepath: Optional[str] = None) -> Optional[str]:
        if filepath and os.path.exists(filepath):
            return filepath

        cand_split = "dev" if split in ("validation", "dev") else split
        candidates = [
            f"webnlg-dataset-master/release_v2.1_constrained/json/webnlg_release_v2.1_constrained_{cand_split}.json",
            f"data/raw/webnlg/webnlg_release_v2.1_constrained_{cand_split}.json",
            f"webnlg-dataset-master/release_v2.1_constrained/json/webnlg_{cand_split}.json"
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return None

    def _resolve_xml_dir(self, split: str) -> Optional[str]:
        cand_split = "dev" if split in ("validation", "dev") else split
        candidates = [
            f"webnlg-dataset-master/release_v3.0/en/{cand_split}",
            f"data/raw/webnlg/release_v3.0/en/{cand_split}",
            f"webnlg-dataset-master/webnlg_challenge_2017/{cand_split}",
            f"webnlg-dataset-master/release_v2_constrained/xml/{cand_split}"
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return None

    def load_split(self, split: str = "validation", filepath: Optional[str] = None, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Loads all WebNLG records for a given split into memory."""
        records = list(self.stream_split(split=split, filepath=filepath, max_examples=max_examples))
        logger.info(f"Loaded {len(records)} records from WebNLG '{split}' split.")
        return records

    def stream_split(self, split: str = "validation", filepath: Optional[str] = None, max_examples: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """Streams WebNLG records from disk, preferring JSON v2.1 with XML v3 fallback."""
        norm_split = "validation" if split in ("validation", "dev") else split
        json_path = self._resolve_json_path(split, filepath)

        if json_path and os.path.exists(json_path):
            yield from self._stream_json(json_path, norm_split, max_examples)
            return

        xml_dir = self._resolve_xml_dir(split)
        if xml_dir and os.path.exists(xml_dir):
            yield from self._stream_xml_dir(xml_dir, norm_split, max_examples)
            return

        raise FileNotFoundError(f"WebNLG data for split '{split}' not found.")

    def _stream_json(self, json_path: str, split: str, max_examples: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", [])
        count = 0

        for idx, item in enumerate(entries):
            if max_examples is not None and count >= max_examples:
                break

            # Handle possible nested wrapper dictionary {"1": {...}}
            entry = list(item.values())[0] if (isinstance(item, dict) and "category" not in item and len(item) == 1) else item

            category = entry.get("category", "")
            size = int(entry.get("size", 1))
            shape = entry.get("shape", "")
            shape_type = entry.get("shape_type", "")
            xml_id = entry.get("xml_id", f"Id{idx+1}")

            # Parse triples
            modifiedtriples = entry.get("modifiedtripleset", [])
            triples = []
            for t in modifiedtriples:
                s = t.get("subject", "").replace("_", " ").strip()
                p = t.get("property", "").strip()
                o = t.get("object", "").replace("_", " ").strip()
                triples.append([s, p, o])

            # Parse lexicalisations
            lex_entries = entry.get("lexicalisations", [])
            references = []
            for lex in lex_entries:
                text = lex.get("lex", "").strip()
                if text:
                    references.append(text)

            target_text = references[0] if references else ""

            record = {
                "id": f"webnlg_{split}_{xml_id}_{idx}",
                "example_id": xml_id,
                "dataset": "webnlg",
                "split": split,
                "tripleset": triples,
                "modifiedtripleset": modifiedtriples,
                "originaltriplesets": entry.get("originaltriplesets", {}),
                "category": category,
                "size": size,
                "shape": shape,
                "shape_type": shape_type,
                "reference_texts": references,
                "target_text": target_text,
                "lexicalisations": lex_entries
            }
            yield record
            count += 1

    def _stream_xml_dir(self, xml_dir: str, split: str, max_examples: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        xml_files = sorted(glob.glob(os.path.join(xml_dir, "**", "*.xml"), recursive=True))
        count = 0

        for file_path in xml_files:
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
            except Exception as e:
                logger.warning(f"Error parsing XML file {file_path}: {e}")
                continue

            for entry in root.findall(".//entry"):
                if max_examples is not None and count >= max_examples:
                    return

                category = entry.get("category", "")
                entry_id = entry.get("id", f"Id{count+1}")
                size = int(entry.get("size", 1))
                shape = entry.get("shape", "")

                triples = []
                for mtriple in entry.findall(".//modifiedtripleset/mtriple"):
                    text = mtriple.text or ""
                    parts = [p.strip().replace("_", " ") for p in text.split("|")]
                    if len(parts) == 3:
                        triples.append(parts)

                references = []
                for lex in entry.findall(".//lex"):
                    text = (lex.text or "").strip()
                    if text:
                        references.append(text)

                target_text = references[0] if references else ""

                record = {
                    "id": f"webnlg_{split}_{entry_id}_{count}",
                    "example_id": entry_id,
                    "dataset": "webnlg",
                    "split": split,
                    "tripleset": triples,
                    "category": category,
                    "size": size,
                    "shape": shape,
                    "reference_texts": references,
                    "target_text": target_text
                }
                yield record
                count += 1
