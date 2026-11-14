"""Unified Schema Definition for Phase 1 Data-to-Text Pipeline.

This module provides standard dataclasses representing nodes, edges,
and the UnifiedRecord schema shared across ToTTo, DART, and WebNLG.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class Node:
    """Represents an atomic entity or cell in structured input."""
    id: str
    label: str
    type: str  # "header", "cell", "entity", "subject", "object", etc.
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Edge:
    """Represents a relationship or topological adjacency between nodes."""
    source: str
    target: str
    relation: str
    type: str = "relation"  # "relation", "row_adj", "col_adj", etc.
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedRecord:
    """Common schema for all datasets in the project.

    Preserves both high-level structured input (tables, triples) and
    explicit graph topological components (nodes, edges, highlighted elements)
    required for future Positional Semantic Divergence (PSD) calculations.
    """
    dataset: str  # "totto", "dart", "webnlg"
    id: str
    input_type: str  # "table", "triples", "graph"
    structured_input: Any  # Preserved original structure
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    highlighted_elements: List[str] = field(default_factory=list)
    target_text: str = ""
    reference_texts: List[str] = field(default_factory=list)
    split: str = "train"  # "train", "validation", "test"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedRecord":
        return cls(
            dataset=data.get("dataset", ""),
            id=str(data.get("id", "")),
            input_type=data.get("input_type", "triples"),
            structured_input=data.get("structured_input"),
            nodes=data.get("nodes", []),
            edges=data.get("edges", []),
            highlighted_elements=data.get("highlighted_elements", []),
            target_text=data.get("target_text", ""),
            reference_texts=data.get("reference_texts", []),
            split=data.get("split", "train"),
            metadata=data.get("metadata", {})
        )
