"""
Shared Schema definition and dataclasses for Stage C Normalization.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class NormalizedNode:
    node_id: str
    text: str
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NormalizedEdge:
    source: str
    target: str
    relation: str
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NormalizedRecord:
    id: str
    dataset: str  # 'totto' | 'webnlg' | 'dart'
    source_type: str  # 'table' | 'graph' | 'mixed'
    nodes: List[NormalizedNode] = field(default_factory=list)
    edges: List[NormalizedEdge] = field(default_factory=list)
    reference_texts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "source_type": self.source_type,
            "nodes": [
                {"node_id": n.node_id, "text": n.text, "attributes": n.attributes}
                for n in self.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target, "relation": e.relation, "attributes": e.attributes}
                for e in self.edges
            ],
            "reference_texts": self.reference_texts,
            "metadata": self.metadata
        }
