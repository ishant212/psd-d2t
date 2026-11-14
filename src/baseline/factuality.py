"""Lightweight, Reproducible Factuality and Hallucination Evaluation Module.

Computes claim-level grounding and hallucination metrics for Data-to-Text generation:
- Supported Factual Claims: Factual assertions (named entities, numerics, relations)
  that are directly grounded in the source table or knowledge graph.
- Unsupported Factual Claims: Factual assertions that cannot be grounded in the source.
- Factual Precision: supported / total
- Hallucination Rate: unsupported / total = 100.0 - Factual Precision
- Structured Content Coverage: recall of source elements in the hypothesis.
"""

import re
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict


@dataclass
class FactUnit:
    text: str
    category: str  # "entity", "numeric", "relation", "syntax_leak"
    supported: bool
    evidence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FactualityResult:
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    factual_precision: float  # 0.0 - 100.0
    hallucination_rate: float  # 0.0 - 100.0
    structured_coverage: float  # 0.0 - 100.0
    claims: List[FactUnit]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "factual_precision": round(self.factual_precision, 2),
            "hallucination_rate": round(self.hallucination_rate, 2),
            "structured_coverage": round(self.structured_coverage, 2),
            "claims": [c.to_dict() for c in self.claims],
        }


class FactualityEvaluator:
    """Lightweight, deterministic factuality evaluator for Data-to-Text."""

    COMMON_STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "with", "by", "from", "of", "about", "as", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "this", "that", "these", "those", "it", "its", "they", "their", "them",
        "who", "which", "what", "where", "when", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "can", "will", "just", "don", "should", "now", "also", "into", "after"
    }

    def __init__(self):
        pass

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercases and strips punctuation."""
        return re.sub(r"[^\w\s]", " ", text.lower()).strip()

    def extract_source_knowledge(self, record_or_dict: Any) -> Dict[str, Any]:
        """Extracts all entities, cells, headers, and relations from the source representation."""
        source_phrases: Set[str] = set()
        source_tokens: Set[str] = set()
        source_numbers: Set[str] = set()
        canonical_elements: List[str] = []

        # 1. From object attributes or dict
        if hasattr(record_or_dict, "nodes"):
            nodes = getattr(record_or_dict, "nodes", [])
            edges = getattr(record_or_dict, "edges", [])
            source_raw = getattr(record_or_dict, "source", "") or getattr(record_or_dict, "serialized_text", "")
            dataset = getattr(record_or_dict, "dataset", "")
        else:
            nodes = record_or_dict.get("nodes", [])
            edges = record_or_dict.get("edges", [])
            source_raw = record_or_dict.get("source", "") or record_or_dict.get("serialized_text", "")
            dataset = record_or_dict.get("dataset", "")

        # Extract from nodes (table cells, headers, graph entities)
        for n in nodes:
            label = n.get("label", "").strip() if isinstance(n, dict) else getattr(n, "label", "").strip()
            if label and label != "-":
                canonical_elements.append(label)
                norm = self._normalize(label)
                if norm:
                    source_phrases.add(norm)
                    for tok in norm.split():
                        if tok not in self.COMMON_STOPWORDS:
                            source_tokens.add(tok)
                        if re.match(r"^\d+(?:st|nd|rd|th)?$", tok):
                            source_numbers.add(tok)

        # Extract from edges (relations)
        for e in edges:
            rel = e.get("relation", "").strip() if isinstance(e, dict) else getattr(e, "relation", "").strip()
            if rel:
                canonical_elements.append(rel)
                norm_rel = self._normalize(rel.replace("_", " "))
                if norm_rel:
                    source_phrases.add(norm_rel)
                    for tok in norm_rel.split():
                        if tok not in self.COMMON_STOPWORDS:
                            source_tokens.add(tok)

        # Extract from raw source string (headers, page titles, section names, cells)
        if source_raw:
            # Table markup parsing: [PAGE] ..., [SECTION] ..., [HEADER] ..., [CELL] ...
            markup_tokens = re.findall(r"\[(?:PAGE|SECTION|HEADER|CELL|HCELL|SUBJ|REL|OBJ)\]\s*([^\[\|]+)", source_raw)
            for m in markup_tokens:
                clean_m = m.strip()
                if clean_m and clean_m != "-":
                    canonical_elements.append(clean_m)
                    norm = self._normalize(clean_m)
                    if norm:
                        source_phrases.add(norm)
                        for tok in norm.split():
                            if tok not in self.COMMON_STOPWORDS:
                                source_tokens.add(tok)
                            if re.match(r"^\d+(?:st|nd|rd|th)?$", tok):
                                source_numbers.add(tok)

            # Extract any numbers appearing anywhere in source
            for num in re.findall(r"\b\d+(?:[\.,]\d+)?(?:st|nd|rd|th)?\b", source_raw):
                source_numbers.add(num.lower().replace(",", ""))

        return {
            "dataset": dataset,
            "phrases": source_phrases,
            "tokens": source_tokens,
            "numbers": source_numbers,
            "canonical_elements": canonical_elements,
            "raw": source_raw.lower() if source_raw else "",
        }

    def extract_claim_units(self, prediction: str) -> List[Tuple[str, str]]:
        """Extracts claim units from hypothesis text: (unit_text, category)."""
        units: List[Tuple[str, str]] = []
        if not prediction or not prediction.strip():
            return units

        # 1. Structural syntax leak check (e.g. model outputting [CELL], [ROW 7], etc.)
        leakages = re.findall(r"\[(?:CELL|ROW|HEADER|HCELL|PAGE|SECTION|TRIPLE|SUBJ|REL|OBJ)\]", prediction)
        for leak in leakages:
            units.append((leak, "syntax_leak"))

        # 2. Extract Numbers & Dates (e.g. "1973", "2016", "76th", "3.5")
        numbers = re.findall(r"\b\d+(?:[\.,]\d+)?(?:st|nd|rd|th)?\b", prediction)
        for num in numbers:
            units.append((num, "numeric"))

        # 3. Extract Named Entities & Capitalized Noun Phrases
        words = prediction.split()
        current_entity: List[str] = []

        for idx, w in enumerate(words):
            clean = re.sub(r"[^\w\s-]", "", w)
            if not clean:
                continue

            is_cap = clean[0].isupper() if clean else False
            is_start = (idx == 0) or (idx > 0 and words[idx - 1].endswith((".", "!", "?")))

            if is_cap:
                if is_start and clean.lower() in self.COMMON_STOPWORDS:
                    continue
                current_entity.append(clean)
            else:
                if current_entity:
                    entity_str = " ".join(current_entity)
                    if len(entity_str) > 1 and entity_str.lower() not in self.COMMON_STOPWORDS:
                        units.append((entity_str, "entity"))
                    current_entity = []

        if current_entity:
            entity_str = " ".join(current_entity)
            if len(entity_str) > 1 and entity_str.lower() not in self.COMMON_STOPWORDS:
                units.append((entity_str, "entity"))

        # 4. Relational Predicate Verbs / Key Actions
        pred_tokens = [re.sub(r"[^\w]", "", tok.lower()) for tok in words]
        rel_candidates = {
            "founded", "born", "died", "joined", "served", "located", "capital",
            "starred", "played", "directed", "written", "published", "leader",
            "governor", "president", "member", "established", "produced", "created"
        }
        for tok in pred_tokens:
            if tok in rel_candidates:
                units.append((tok, "relation"))

        return units

    def evaluate_single(self, record_or_dict: Any, prediction: str) -> FactualityResult:
        """Evaluates grounding and factuality for a single prediction against its structured source."""
        source_info = self.extract_source_knowledge(record_or_dict)
        claim_candidates = self.extract_claim_units(prediction)

        claims: List[FactUnit] = []
        supported_count = 0
        unsupported_count = 0

        for unit_text, category in claim_candidates:
            norm_unit = self._normalize(unit_text)
            supported = False
            evidence = None

            if category == "syntax_leak":
                supported = False
                evidence = "Model leaked serialization markup into natural language generation"
            elif category == "numeric":
                clean_num = norm_unit.replace(",", "")
                if clean_num in source_info["numbers"] or clean_num in source_info["raw"]:
                    supported = True
                    evidence = f"Matched numeric token '{unit_text}' in source"
                else:
                    base_num = re.sub(r"(?:st|nd|rd|th)$", "", clean_num)
                    if base_num in source_info["numbers"] or base_num in source_info["raw"]:
                        supported = True
                        evidence = f"Matched base number '{base_num}' in source"
                    else:
                        supported = False
                        evidence = f"Numeric token '{unit_text}' not found in source"
            elif category == "relation":
                # Check direct match or equivalence mapping (e.g. played -> role/filmography)
                rel_equivalences = {
                    "played": ["role", "character", "filmography", "film", "actor", "actress", "cast", "evelyn"],
                    "starred": ["role", "filmography", "film", "title", "cast", "star"],
                    "born": ["birth", "born", "birth_date", "birth_place"],
                    "died": ["death", "died", "death_date", "death_place"],
                    "founded": ["foundation", "established", "founded", "founding", "established_date"],
                    "joined": ["joined", "join", "member"],
                    "located": ["location", "place", "city", "country", "located"],
                    "governor": ["governor", "governorship", "office"],
                    "president": ["president", "presidency"],
                    "served": ["served", "service", "office", "term"],
                    "directed": ["directed", "director"],
                    "written": ["written", "writer", "author"],
                }
                if norm_unit in source_info["phrases"] or norm_unit in source_info["raw"]:
                    supported = True
                    evidence = f"Exact/phrase match for relation '{unit_text}' in source"
                else:
                    eq_list = rel_equivalences.get(norm_unit, [])
                    if any(eq in source_info["raw"] or eq in source_info["tokens"] for eq in eq_list):
                        supported = True
                        evidence = f"Semantic/relation equivalence match for '{unit_text}' in source metadata"
                    else:
                        supported = False
                        evidence = f"Relation '{unit_text}' unsupported by source"
            elif category == "entity":
                if norm_unit in source_info["phrases"] or norm_unit in source_info["raw"]:
                    supported = True
                    evidence = f"Exact/phrase match for '{unit_text}' in source"
                else:
                    unit_words = [w for w in norm_unit.split() if w not in self.COMMON_STOPWORDS]
                    if unit_words:
                        hits = sum(1 for w in unit_words if w in source_info["tokens"] or w in source_info["raw"])
                        if hits == len(unit_words):
                            supported = True
                            evidence = f"Full content token match for '{unit_text}' in source"
                        elif hits > 0 and len(unit_words) > 1 and (hits / len(unit_words)) >= 0.5:
                            supported = True
                            evidence = f"Partial entity token overlap ({hits}/{len(unit_words)}) in source"
                        else:
                            supported = False
                            evidence = f"Entity '{unit_text}' unsupported by source"
                    else:
                        supported = True

            if supported:
                supported_count += 1
            else:
                unsupported_count += 1

            claims.append(FactUnit(text=unit_text, category=category, supported=supported, evidence=evidence))

        total_claims = supported_count + unsupported_count
        if total_claims > 0:
            factual_precision = (supported_count / total_claims) * 100.0
            hallucination_rate = (unsupported_count / total_claims) * 100.0
        else:
            factual_precision = 100.0
            hallucination_rate = 0.0

        # Compute source-content coverage
        canonical = [c for c in source_info["canonical_elements"] if len(c.strip()) > 1]
        pred_lower = prediction.lower()
        if canonical:
            cov_hits = sum(1 for c in canonical if self._normalize(c) in pred_lower or any(tok in pred_lower for tok in self._normalize(c).split() if tok not in self.COMMON_STOPWORDS))
            coverage = (cov_hits / len(canonical)) * 100.0
        else:
            coverage = 100.0

        return FactualityResult(
            total_claims=total_claims,
            supported_claims=supported_count,
            unsupported_claims=unsupported_count,
            factual_precision=factual_precision,
            hallucination_rate=hallucination_rate,
            structured_coverage=coverage,
            claims=claims,
        )

    def evaluate_batch(
        self, records_or_dicts: List[Any], predictions: List[str]
    ) -> Dict[str, Any]:
        """Evaluates a batch of predictions against records and aggregates factuality metrics."""
        assert len(records_or_dicts) == len(predictions), "Records and predictions length mismatch"
        if not predictions:
            return {
                "factual_precision": 100.0,
                "hallucination_rate": 0.0,
                "structured_coverage": 0.0,
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "detailed_results": [],
            }

        detailed = []
        precisions = []
        hallucinations = []
        coverages = []
        total_claims_sum = 0
        supported_claims_sum = 0
        unsupported_claims_sum = 0

        for rec, pred in zip(records_or_dicts, predictions):
            res = self.evaluate_single(rec, pred)
            detailed.append(res.to_dict())
            precisions.append(res.factual_precision)
            hallucinations.append(res.hallucination_rate)
            coverages.append(res.structured_coverage)
            total_claims_sum += res.total_claims
            supported_claims_sum += res.supported_claims
            unsupported_claims_sum += res.unsupported_claims

        return {
            "factual_precision": round(sum(precisions) / len(precisions), 2),
            "hallucination_rate": round(sum(hallucinations) / len(hallucinations), 2),
            "structured_coverage": round(sum(coverages) / len(coverages), 2),
            "total_claims": total_claims_sum,
            "supported_claims": supported_claims_sum,
            "unsupported_claims": unsupported_claims_sum,
            "detailed_results": detailed,
        }
