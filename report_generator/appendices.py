from report_generator.styles import (
    add_chapter_heading, add_section_heading, add_subsection_heading,
    add_paragraph, add_custom_table, add_code_block
)

def build_appendices(doc):
    add_chapter_heading(doc, "APPENDICES")

    # =========================================================================
    # APPENDIX 1: CONFIGURATION SCHEMAS AND UNIFIED DATA MODELS
    # =========================================================================
    add_section_heading(doc, "Appendix 1: Complete Configuration Schemas and Unified Data Models")
    add_paragraph(
        doc,
        "This appendix provides the formal JSON schema definitions for the normalized intermediate representation, as well as the "
        "YAML configuration files governing text sanitization, graph construction, and parameter-efficient fine-tuning."
    )

    add_subsection_heading(doc, "A1.1 Canonical Intermediate Data Schema (JSON)")
    add_paragraph(
        doc,
        "Every raw instance from ToTTo, WebNLG 2020, and DART is mapped into the following canonical schema prior to graph serialization:"
    )

    schema_json = """{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "NormalizedStructuredD2TRecord",
  "type": "object",
  "required": ["id", "dataset", "source_type", "nodes", "edges", "reference_texts"],
  "properties": {
    "id": { "type": "string" },
    "dataset": { "type": "string", "enum": ["totto", "webnlg", "dart"] },
    "source_type": { "type": "string", "enum": ["table", "graph", "mixed"] },
    "nodes": {
      "type": "array",
      "items": {
        "required": ["node_id", "text"],
        "properties": {
          "node_id": { "type": "string" },
          "text": { "type": "string" },
          "attributes": { "type": "object" }
        }
      }
    },
    "edges": {
      "type": "array",
      "items": {
        "required": ["source", "target", "relation"],
        "properties": {
          "source": { "type": "string" }, "target": { "type": "string" }, "relation": { "type": "string" }
        }
      }
    },
    "reference_texts": { "type": "array", "items": { "type": "string" }, "minItems": 1 }
  }
}"""
    add_code_block(doc, schema_json, caption="Listing A1.1: Canonical intermediate JSON schema (src/normalization/schema.py).")

    add_subsection_heading(doc, "A1.2 Preprocessing and Graph Construction Configuration (YAML)")
    add_paragraph(doc, "The configuration files governing Stage A cleaning and Stage D graph construction are provided below:")

    config_yaml = """# config/cleaning.yaml & config/graph_construction.yaml
cleaning:
  fix_unicode: true; normalize_quotes: true; strip_zero_width_spaces: true
  max_source_tokens: 4096; min_target_tokens: 3; flag_unreachable_nodes: true

graph_construction:
  directed: true; multigraph: false
  table_connectivity: { row_adjacent: true, column_adjacent: true, header_to_cell: true }
  disconnected_component_penalty: 1.0; export_format: "gpickle" """
    add_code_block(doc, config_yaml, caption="Listing A1.2: Preprocessing and graph construction parameters.")

    # =========================================================================
    # APPENDIX 2: KEY SOURCE CODE LISTINGS AND CORE ALGORITHMS
    # =========================================================================
    add_section_heading(doc, "Appendix 2: Key Source Code Listings and Core Algorithms")
    add_paragraph(
        doc,
        "The following code excerpts are extracted directly from the verified project implementation in the project repository. "
        "They represent the foundational algorithmic engines of the PSD framework across data ingestion, cleaning, exploratory analysis, "
        "normalization, graph topological construction, deterministic serialization, target distribution construction, "
        "eager attention interception, LoRA parameter-efficient joint training, and claim-level factuality auditing."
    )

    # A2.1 LOADER FACTORY
    add_subsection_heading(doc, "A2.1 Multi-Modal Ingestion & Universal Loader Factory")
    add_paragraph(
        doc,
        "The LoaderFactory provides an extensible interface for automatically detecting raw input formats (Wikipedia HTML tables, "
        "RDF XML graphs, or OpenIE JSON records) and instantiating the corresponding dataset loader."
    )
    loader_code = """# src/loaders/loader_factory.py
import os
import json
import logging
from typing import Dict, Type
from .base_loader import BaseLoader
from .totto_loader import ToTToloader
from .webnlg_loader import WebNLGloader
from .dart_loader import DARTloader

logger = logging.getLogger(__name__)

class LoaderFactory:
    \"\"\"Factory for dataset loader auto-detection and instantiation.\"\"\"

    LOADERS: Dict[str, Type[BaseLoader]] = {
        "totto": ToTToloader,
        "webnlg": WebNLGloader,
        "dart": DARTloader,
    }

    @classmethod
    def get_loader(cls, dataset_type: str) -> BaseLoader:
        \"\"\"Instantiates and returns the loader matching the specified dataset key.\"\"\"
        dataset_type = dataset_type.lower()
        if dataset_type not in cls.LOADERS:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Supported: {list(cls.LOADERS.keys())}")
        return cls.LOADERS[dataset_type]()

    @classmethod
    def detect_dataset_type(cls, file_or_dir_path: str) -> str:
        \"\"\"Inspects file path heuristics, directory contents, and JSON keys to identify dataset.\"\"\"
        path_lower = file_or_dir_path.lower()
        if "totto" in path_lower:
            return "totto"
        if "webnlg" in path_lower:
            return "webnlg"
        if "dart" in path_lower:
            return "dart"

        if os.path.isdir(file_or_dir_path):
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
            if file_or_dir_path.endswith(".jsonl"):
                return "totto"
            if file_or_dir_path.endswith(".json"):
                with open(file_or_dir_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if "tripleset" in first_line or "annotations" in first_line:
                        return "dart"
                    return "totto"
        raise ValueError(f"Could not automatically detect dataset type for path: {file_or_dir_path}")"""
    add_code_block(doc, loader_code, caption="Listing A2.1: Multi-Modal Ingestion & Universal Loader Factory (src/loaders/loader_factory.py).")

    # A2.2 CLEANER & TEXT SANITIZATION
    add_subsection_heading(doc, "A2.2 Automated Text Sanitization, Unicode Normalization, and Anomaly Logger")
    add_paragraph(
        doc,
        "The DatasetCleaner implements Stage A text sanitization, enforcing Unicode NFC normalization, stripping non-printable "
        "and zero-width characters, standardizing quotation marks, and logging data hygiene anomalies into a structured CSV audit report."
    )
    cleaner_code = """# src/cleaning/cleaner.py (Excerpt)
import unicodedata
import re
from typing import List, Dict, Any, Set

class DatasetCleaner:
    \"\"\"Executes Stage A Cleaning, text sanitization, and structured anomaly logging.\"\"\"

    def __init__(self, dataset_name: str, config: Dict[str, Any]):
        self.dataset_name = dataset_name.lower()
        self.config = config
        self.report_rows: List[Dict[str, str]] = []
        self.seen_signatures: Set[str] = set()

    def log_action(self, record_id: str, action: str, reason: str) -> None:
        \"\"\"Appends structured audit action (DROP, FIX, FLAG) for provenance tracking.\"\"\"
        self.report_rows.append({
            "dataset": self.dataset_name,
            "record_id": str(record_id),
            "action": action,
            "reason": reason
        })

    def normalize_text(self, text: str) -> str:
        \"\"\"Applies NFC normalization, strips zero-width controls, and standardizes punctuation.\"\"\"
        if not text:
            return ""
        # 1. Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)
        # 2. Strip invisible zero-width and directional control characters
        text = re.sub(r"[\\u200b\\u200c\\u200d\\ufeff\\u200e\\u200f]", "", text)
        # 3. Standardize typographical quotation marks and primes
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        # 4. Collapse irregular whitespace and non-breaking spaces
        text = re.sub(r"[\\t\\r\\f\\v\\u00a0]+", " ", text)
        text = re.sub(r" +", " ", text)
        return text.strip()

    def clean_record(self, raw_record: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Cleans tabular and graph fields, dropping empty or corrupted samples.\"\"\"
        cleaned = dict(raw_record)
        rec_id = str(cleaned.get("id", "unknown"))
        text = cleaned.get("target_text", "")
        if not text or len(text.strip()) < self.config.get("min_target_tokens", 3):
            self.log_action(rec_id, "DROP", "Target verbalization empty or below minimum token threshold")
            return None
        cleaned["target_text"] = self.normalize_text(text)
        return cleaned"""
    add_code_block(doc, cleaner_code, caption="Listing A2.2: Automated Text Sanitization and Anomaly Logger (src/cleaning/cleaner.py).")

    # A2.3 LENGTH ANALYZER & TOKEN PROFILER
    add_subsection_heading(doc, "A2.3 Exploratory Data Analysis & Token Distribution Profiler")
    add_paragraph(
        doc,
        "The LengthAnalyzer profiles input sequence lengths, target verbalizations, and multi-horizon context distributions, "
        "computing percentiles (p50, p90, p95, p99) and quantifying long-context truncation risks across 256, 512, 1024, and 2048 token thresholds."
    )
    eda_code = """# src/eda/length_analysis.py
import numpy as np
from typing import List, Dict, Any, Optional
from src.data.unified_schema import UnifiedRecord
from src.preprocessing.serialize import Serializer

class LengthAnalyzer:
    \"\"\"Computes token length statistics, percentiles, and multi-horizon context distributions.\"\"\"

    def __init__(self, tokenizer: Any = None, thresholds: Optional[List[int]] = None):
        self.tokenizer = tokenizer
        self.thresholds = thresholds or [256, 512, 1024, 2048, 4096]
        self.serializer = Serializer()

    def _get_token_length(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer(text, add_special_tokens=True, truncation=False)
                return len(tokens["input_ids"])
            except Exception:
                pass
        return len(text.strip().split())

    def analyze(self, records: List[UnifiedRecord], dataset_name: str) -> Dict[str, Any]:
        \"\"\"Computes statistical metrics and percentile profiles for serialized input and targets.\"\"\"
        input_lens = [self._get_token_length(self.serializer.serialize_record(r)[0]) for r in records]
        target_lens = [self._get_token_length(r.target_text) for r in records]

        def compute_stats(lengths: List[int], prefix: str) -> Dict[str, Any]:
            if not lengths:
                return {f"{prefix}_{k}": 0 for k in ["min", "mean", "median", "p90", "p95", "p99", "max"]}
            arr = np.array(lengths)
            return {
                f"{prefix}_min": int(np.min(arr)),
                f"{prefix}_mean": float(np.mean(arr)),
                f"{prefix}_median": float(np.median(arr)),
                f"{prefix}_p90": float(np.percentile(arr, 90)),
                f"{prefix}_p95": float(np.percentile(arr, 95)),
                f"{prefix}_p99": float(np.percentile(arr, 99)),
                f"{prefix}_max": int(np.max(arr))
            }

        res = {"dataset": dataset_name, "num_records": len(records)}
        res.update(compute_stats(input_lens, "input"))
        res.update(compute_stats(target_lens, "target"))
        for th in self.thresholds:
            res[f"input_gt_{th}"] = int(sum(1 for l in input_lens if l > th))
            res[f"pct_gt_{th}"] = round(100.0 * res[f"input_gt_{th}"] / max(len(input_lens), 1), 2)
        return res"""
    add_code_block(doc, eda_code, caption="Listing A2.3: Token Length Distribution & Context Profiler (src/eda/length_analysis.py).")

    # A2.4 SCHEMA NORMALIZATION
    add_subsection_heading(doc, "A2.4 Canonical Intermediate Data Transformation & Schema Normalizer")
    add_paragraph(
        doc,
        "The DataNormalizer transforms dataset-specific structures into the unified canonical intermediate schema, establishing "
        "explicit node attributes and bidirectional horizontal (row) and vertical (column) adjacency edges for tables."
    )
    converter_code = """# src/normalization/converter.py (Excerpt)
from typing import Any, Dict, List, Tuple
from src.normalization.schema import NormalizedNode, NormalizedEdge, NormalizedRecord

class DataNormalizer:
    \"\"\"Maps dataset-specific representations into the unified canonical schema.\"\"\"

    def __init__(self, dataset_name: str, config: Dict[str, Any]):
        self.dataset_name = dataset_name.lower()
        self.config = config

    def convert_totto(self, record_dict: Dict[str, Any]) -> NormalizedRecord:
        \"\"\"Converts raw ToTTo table grids into relational nodes with bidirectional grid adjacencies.\"\"\"
        rec_id = str(record_dict.get("record_id", "totto_unknown"))
        raw_data = record_dict.get("raw_data", {})
        table = raw_data.get("table", [])
        highlighted_set = set((r, c) for r, c in raw_data.get("highlighted_cells", []))

        nodes: List[NormalizedNode] = []
        edges: List[NormalizedEdge] = []
        node_grid: Dict[Tuple[int, int], str] = {}

        # 1. Instantiate grid nodes with cell coordinates and header flags
        for r_idx, row in enumerate(table):
            for c_idx, cell in enumerate(row):
                cell_val = cell.get("value", "") if isinstance(cell, dict) else str(cell)
                is_header = cell.get("is_header", False) if isinstance(cell, dict) else False
                node_id = f"r{r_idx}_c{c_idx}"
                nodes.append(NormalizedNode(
                    node_id=node_id, text=cell_val,
                    attributes={"row": r_idx, "col": c_idx, "is_header": is_header,
                                "is_highlighted": (r_idx, c_idx) in highlighted_set}
                ))
                node_grid[(r_idx, c_idx)] = node_id

        # 2. Construct bidirectional row-wise and column-wise adjacency edges
        for (r_idx, c_idx), node_id in node_grid.items():
            if (r_idx, c_idx + 1) in node_grid:
                edges.append(NormalizedEdge(source=node_id, target=node_grid[(r_idx, c_idx + 1)], relation="row_adjacent"))
            if (r_idx + 1, c_idx) in node_grid:
                edges.append(NormalizedEdge(source=node_id, target=node_grid[(r_idx + 1, c_idx)], relation="col_adjacent"))

        return NormalizedRecord(id=rec_id, dataset="totto", source_type="table",
                                nodes=nodes, edges=edges, reference_texts=record_dict.get("target_texts", []))"""
    add_code_block(doc, converter_code, caption="Listing A2.4: Canonical Data Normalizer and Grid Adjacency Builder (src/normalization/converter.py).")

    # A2.5 GRAPH CONSTRUCTION & APSP BFS
    add_subsection_heading(doc, "A2.5 Topological Graph Construction & Geodesic APSP Shortest-Path BFS Matrix Engine")
    add_paragraph(
        doc,
        "The GraphBuilder constructs attributed NetworkX DiGraphs from normalized records and executes all-pairs shortest paths (APSP) "
        "via breadth-first search on undirected projections, with bounded penalty assignments for disconnected components."
    )
    builder_code = """# src/graph_construction/builder.py (Excerpt)
import networkx as nx
import numpy as np
from typing import Dict, Any, Tuple

class GraphBuilder:
    \"\"\"Constructs NetworkX DiGraphs and computes all-pairs shortest paths via BFS.\"\"\"

    def build_graph(self, norm_dict: Dict[str, Any]) -> nx.DiGraph:
        \"\"\"Constructs an attributed DiGraph from canonical schema dictionaries.\"\"\"
        G = nx.DiGraph(id=norm_dict["id"], dataset=norm_dict["dataset"], source_type=norm_dict["source_type"])
        for node in norm_dict.get("nodes", []):
            G.add_node(node["node_id"], text=node.get("text", ""), **node.get("attributes", {}))
        for edge in norm_dict.get("edges", []):
            G.add_edge(edge["source"], edge["target"], relation=edge.get("relation", ""), **edge.get("attributes", {}))
        return G

    @staticmethod
    def compute_apsp_matrix(graph: nx.DiGraph) -> Tuple[np.ndarray, Dict[str, int]]:
        \"\"\"Computes APSP geodesic distance matrix on undirected projection with disconnection penalties.\"\"\"
        undirected_g = graph.to_undirected()
        nodes = sorted(list(undirected_g.nodes()))
        node_to_idx = {nid: idx for idx, nid in enumerate(nodes)}
        n = len(nodes)
        dist_matrix = np.full((n, n), fill_value=-1.0, dtype=np.float32)
        np.fill_diagonal(dist_matrix, 0.0)

        # BFS shortest paths for each connected component
        for src in nodes:
            for tgt, length in nx.single_source_shortest_path_length(undirected_g, src).items():
                dist_matrix[node_to_idx[src], node_to_idx[tgt]] = float(length)

        # Assign bounded penalty cutoff for unreachable / disconnected nodes
        reachable = dist_matrix >= 0.0
        penalty = (np.max(dist_matrix[reachable]) + 1.0) if np.any(reachable) else 1.0
        dist_matrix[~reachable] = penalty
        return dist_matrix, node_to_idx"""
    add_code_block(doc, builder_code, caption="Listing A2.5: Graph Construction & APSP Geodesic Distance Engine (src/graph_construction/builder.py).")

    # A2.6 DETERMINISTIC SERIALIZATION & TOKEN-NODE MAPPING
    add_subsection_heading(doc, "A2.6 Deterministic Markup Serialization & Exact Token-to-Node Character Span Aligner")
    add_paragraph(
        doc,
        "The Serializer linearizes tables and knowledge graphs into unambiguous markup sequences while simultaneously constructing "
        "character-level offset intervals that allow exact projection of subword tokens back to underlying graph node identifiers."
    )
    serializer_code = """# src/preprocessing/serialize.py (Excerpt)
from typing import Dict, Any, List, Tuple

class Serializer:
    \"\"\"Linearizes tables and graphs into markup sequences while preserving token alignments.\"\"\"

    def serialize_record(self, record: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        \"\"\"Serializes tables or RDF triples and tracks character-level span boundaries.\"\"\"
        parts, spans, cur = [], [], 0

        if record.get("source_type") == "table":
            for node in record["nodes"]:
                is_hi = node.get("attributes", {}).get("is_highlighted", False)
                tag = "[HCELL]" if is_hi else "[CELL]"
                cell_str = f"{tag} {node['text']} "
                spans.append({
                    "node_id": node["node_id"],
                    "start": cur + len(tag) + 1,
                    "end": cur + len(cell_str) - 1,
                    "type": "table_cell"
                })
                parts.append(cell_str)
                cur += len(cell_str)
        else:
            for edge in record.get("edges", []):
                s, r, o = edge["source"], edge.get("relation", ""), edge["target"]
                t_str = f"[TRIPLE] [SUBJ] {s} [REL] {r} [OBJ] {o} "
                s_st = cur + len("[TRIPLE] [SUBJ] ")
                s_end = s_st + len(s)
                o_st = s_end + len(" [REL] ") + len(r) + len(" [OBJ] ")
                o_end = o_st + len(o)
                spans.extend([
                    {"node_id": s, "start": s_st, "end": s_end, "type": "subject"},
                    {"node_id": o, "start": o_st, "end": o_end, "type": "object"}
                ])
                parts.append(t_str)
                cur += len(t_str)

        return "".join(parts), spans"""
    add_code_block(doc, serializer_code, caption="Listing A2.6: Deterministic Markup Serialization & Span Aligner (src/preprocessing/serialize.py).")

    # A2.7 TARGET DISTRIBUTION BUILDER
    add_subsection_heading(doc, "A2.7 Structural-Semantic Target Attention Distribution Matrix Builder")
    add_paragraph(
        doc,
        "The TargetDistributionBuilder constructs the causal topological target distribution P_target(j|i) by synthesizing "
        "frozen semantic similarities with normalized positional penalties and graph geodesic distance matrices."
    )
    target_dist_code = """# src/psd/target_distribution.py (Excerpt)
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict

class TargetDistributionBuilder:
    \"\"\"Constructs topological-semantic target attention distributions P_target(j|i).\"\"\"

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.5, tau: float = 1.0):
        self.alpha, self.beta, self.gamma, self.tau = alpha, beta, gamma, tau

    def build_target_distribution(
        self, seq_len: int, token_to_node_map: List[Optional[str]],
        graph_dist_matrix: np.ndarray, node_to_idx: Dict[str, int],
        semantic_sim_matrix: Optional[torch.Tensor] = None, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        \"\"\"Synthesizes P_target(j|i) by combining semantic cosine similarity with positional and graph penalties.\"\"\"
        if seq_len == 0:
            return torch.zeros((0, 0), device=device), torch.zeros(0, dtype=torch.bool, device=device)

        # 1. Normalized sequence distance: dP_norm(i, j) = |i - j| / max(seq_len - 1, 1)
        dP_max = float(max(seq_len - 1, 1))
        pos = torch.arange(seq_len, dtype=torch.float32, device=device)
        dP_norm = torch.abs(pos.unsqueeze(1) - pos.unsqueeze(0)) / dP_max

        # 2. Normalized topological graph distance: dG_norm(i, j) = dG(u, v) / dG_max
        dG_norm = torch.zeros((seq_len, seq_len), dtype=torch.float32, device=device)
        valid_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        dG_max = float(np.max(graph_dist_matrix)) if graph_dist_matrix.size > 0 and np.max(graph_dist_matrix) > 0 else 1.0

        for i, u_id in enumerate(token_to_node_map):
            if u_id in node_to_idx:
                valid_mask[i] = True
                u_idx = node_to_idx[u_id]
                for j, v_id in enumerate(token_to_node_map):
                    if v_id in node_to_idx:
                        v_idx = node_to_idx[v_id]
                        dG_norm[i, j] = float(graph_dist_matrix[u_idx, v_idx]) / dG_max

        # 3. Target score composite: S_ij_hat = alpha * S_ij - beta * dP_norm - gamma * dG_norm
        S_sim = semantic_sim_matrix if semantic_sim_matrix is not None else torch.ones((seq_len, seq_len), device=device)
        scores = (self.alpha * S_sim - self.beta * dP_norm - self.gamma * dG_norm) / self.tau

        # 4. Strict causal masking: tokens can only attend to previous tokens (j <= i)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # 5. Numerically stable row-wise softmax normalization
        P_target = F.softmax(scores, dim=-1)
        P_target = torch.nan_to_num(P_target, nan=0.0)
        return P_target, valid_mask"""
    add_code_block(doc, target_dist_code, caption="Listing A2.7: Target Attention Distribution Matrix Builder (src/psd/target_distribution.py).")

    # A2.8 ATTENTION INTERCEPTION & PSD KL DIVERGENCE
    add_subsection_heading(doc, "A2.8 Eager Attention Interception & Multi-Layer KL Divergence Loss Engine")
    add_paragraph(
        doc,
        "The PSDLoss module averages attention distributions across attention heads on designated Transformer layers, enforces causal "
        "and valid token masking, and calculates the Kullback-Leibler divergence between P_target and empirical attention."
    )
    psd_loss_code = """# src/psd/divergence.py (Excerpt)
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional

class PSDLoss(nn.Module):
    \"\"\"Calculates layer-selective Positional Semantic Divergence (PSD) loss.\"\"\"

    def __init__(self, selected_layers: List[int] = [11, 23, 34], eps: float = 1e-9):
        super().__init__()
        self.selected_layers = selected_layers
        self.eps = eps

    def forward(
        self, attentions: Tuple[torch.Tensor, ...], P_target: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
        \"\"\"Averages attention heads and computes forward KL divergence D_KL(P_target || A_empirical).\"\"\"
        if P_target.dim() == 2:
            P_target = P_target.unsqueeze(0)  # [1, seq_len, seq_len]

        batch_size, seq_len, _ = P_target.shape
        device = P_target.device
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)).unsqueeze(0)
        p_target_safe = P_target.clamp(min=self.eps, max=1.0)

        layer_losses = []
        per_layer_dict = {}

        for l_idx in self.selected_layers:
            if l_idx >= len(attentions):
                continue
            # Extract [batch, heads, seq_len, seq_len] and average across attention heads
            layer_attn = attentions[l_idx].mean(dim=1)  # [batch, seq_len, seq_len]
            a_empirical_safe = layer_attn.clamp(min=self.eps, max=1.0)

            # Causal and structured token masking
            kl_elementwise = p_target_safe * (torch.log(p_target_safe) - torch.log(a_empirical_safe))
            kl_elementwise = kl_elementwise * causal_mask.float()

            if valid_mask is not None:
                mask_2d = valid_mask.unsqueeze(-1).expand_as(kl_elementwise)
                kl_elementwise = kl_elementwise * mask_2d.float()
                loss_val = kl_elementwise.sum() / valid_mask.sum().float().clamp(min=1.0)
            else:
                loss_val = kl_elementwise.sum() / float(batch_size * seq_len)

            layer_losses.append(loss_val)
            per_layer_dict[l_idx] = float(loss_val.detach().item())

        total_psd_loss = torch.stack(layer_losses).mean() if layer_losses else torch.tensor(0.0, device=device)
        return total_psd_loss, per_layer_dict"""
    add_code_block(doc, psd_loss_code, caption="Listing A2.8: Eager Attention Interception & Multi-Layer PSD Loss (src/psd/divergence.py).")

    # A2.9 PARAMETER-EFFICIENT FINE-TUNING TRAINER
    add_subsection_heading(doc, "A2.9 Parameter-Efficient Fine-Tuning (PEFT LoRA) Training Pipeline with Gradient Accumulation")
    add_paragraph(
        doc,
        "The training pipeline initializes PEFT LoRA adapters into query and value projections, manages half-precision execution, "
        "and coordinates the joint optimization objective L_total = L_LM + lambda_PSD * L_PSD with gradient accumulation."
    )
    trainer_code = """# src/psd/trainer.py (Excerpt)
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.psd.divergence import PSDLoss

def initialize_psd_training_pipeline(model_name: str, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05):
    \"\"\"Injects LoRA adapters and wraps backbone model for joint LM + PSD optimization.\"\"\"
    # 1. Load base causal language model in half precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, attn_implementation="eager", device_map="auto"
    )
    # 2. Inject LoRA adapters into query and value projections
    peft_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def train_psd_epoch(model, dataloader, optimizer, psd_loss_fn, lambda_psd: float = 0.10, grad_accum_steps: int = 4):
    \"\"\"Executes joint training epoch: L_total = L_LM + lambda_PSD * L_PSD with gradient accumulation.\"\"\"
    model.train()
    optimizer.zero_grad()
    running_loss, running_lm, running_psd = 0.0, 0.0, 0.0

    for step, batch in enumerate(dataloader):
        outputs = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            labels=batch["labels"], output_attentions=True
        )
        lm_loss = outputs.loss
        psd_loss, _ = psd_loss_fn(outputs.attentions, batch["P_target"], batch.get("valid_mask"))

        # Joint composite loss
        total_loss = (lm_loss + lambda_psd * psd_loss) / grad_accum_steps
        total_loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += total_loss.item() * grad_accum_steps
        running_lm += lm_loss.item()
        running_psd += psd_loss.item()

    return running_loss / len(dataloader), running_lm / len(dataloader), running_psd / len(dataloader)"""
    add_code_block(doc, trainer_code, caption="Listing A2.9: PEFT LoRA Training Engine with Gradient Accumulation (src/psd/trainer.py).")

    # A2.10 CLAIM-LEVEL FACTUALITY EVALUATOR
    add_subsection_heading(doc, "A2.10 Automated Claim-Level Factuality Evaluator & Hallucination Auditor")
    add_paragraph(
        doc,
        "The FactualityEvaluator decomposes generated verbalizations into atomic factual units (named entities, numerics, and relations) "
        "and checks them against the source table or knowledge graph cells, computing Factual Precision and Hallucination Rate."
    )
    factuality_code = """# src/baseline/factuality.py (Excerpt)
import re
from typing import Dict, Any, List, Set
from dataclasses import dataclass

@dataclass
class FactUnit:
    text: str
    category: str  # "entity", "numeric", "relation"
    supported: bool

class FactualityEvaluator:
    \"\"\"Audits generated verbalizations against structured source entities, numerics, and relations.\"\"\"

    def extract_candidate_facts(self, text: str) -> List[FactUnit]:
        \"\"\"Extracts candidate factual assertions using rule-based and entity boundary extraction.\"\"\"
        facts = []
        # Numeric values and measurements
        for match in re.finditer(r"\\b\\d+(?:[\\.,]\\d+)?%?\\b", text):
            facts.append(FactUnit(text=match.group(0), category="numeric", supported=False))
        # Capitalized multi-word named entities
        for match in re.finditer(r"\\b[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*\\b", text):
            token = match.group(0)
            if len(token) > 2 and token.lower() not in {"the", "this", "there", "then"}:
                facts.append(FactUnit(text=token, category="entity", supported=False))
        return facts

    def evaluate_factuality(self, hypothesis: str, source_record: Dict[str, Any]) -> Dict[str, float]:
        \"\"\"Verifies claims against source table/graph cells, computing Precision and Hallucination Rate.\"\"\"
        candidate_facts = self.extract_candidate_facts(hypothesis)
        if not candidate_facts:
            return {"factual_precision": 100.0, "hallucination_rate": 0.0, "total_claims": 0}

        # Build normalized source vocabulary
        source_tokens: Set[str] = set()
        for node in source_record.get("nodes", []):
            source_tokens.update(re.findall(r"\\w+", node.get("text", "").lower()))

        supported_count = 0
        for fact in candidate_facts:
            fact_tokens = set(re.findall(r"\\w+", fact.text.lower()))
            if fact_tokens and fact_tokens.issubset(source_tokens):
                fact.supported = True
                supported_count += 1

        total = len(candidate_facts)
        precision = (supported_count / total) * 100.0
        hallucination_rate = 100.0 - precision

        return {
            "total_claims": total,
            "supported_claims": supported_count,
            "factual_precision": round(precision, 2),
            "hallucination_rate": round(hallucination_rate, 2)
        }"""
    add_code_block(doc, factuality_code, caption="Listing A2.10: Automated Claim-Level Factuality Evaluator & Hallucination Auditor (src/baseline/factuality.py).")

    # =========================================================================
    # APPENDIX 3: EXPERIMENTAL PARAMETERS AND REPRODUCTION GUIDE
    # =========================================================================
    add_section_heading(doc, "Appendix 3: Experimental Parameters, Hyperparameter Grids, and Reproduction Guide")
    add_paragraph(
        doc,
        "Table A3.1 summarizes the complete hyperparameter grid governing the training, optimization, and evaluation of all "
        "experiments reported in this thesis."
    )

    tbl_param_headers = ["Parameter Category", "Hyperparameter Name", "Exact Value", "Operational Role"]
    tbl_param_rows = [
        ["Hardware Platform", "Primary Compute Cluster", "NVIDIA H100 PCIe (80 GB HBM3)", "Long-context scaling (L in {1024, 2048}) and 8-model cross-architecture benchmark"],
        ["Hardware Platform", "Local Validation Node", "RTX 4060 Laptop (8 GB GDDR6)", "Local data preprocessing, graph construction, and L = 256 prototyping"],
        ["Model Precision", "Base Model Precision", "float16 (FP16)", "Hardware-accelerated half-precision tensor execution across all SLM backbones"],
        ["Model Precision", "Attention Extraction", "Eager PyTorch", "Full multi-head attention matrix interception for PSD loss"],
        ["Sequence Context", "Evaluated Context Horizons (L)", "{256, 1024, 2048} tokens", "Three-point context scaling evaluation for attention dispersion and factuality"],
        ["LoRA PEFT", "Target Modules", "['q_proj', 'v_proj']", "Applies low-rank updates to query and value projections"],
        ["LoRA PEFT", "LoRA Rank (r)", "8", "Rank dimension of adaptation matrices"],
        ["LoRA PEFT", "LoRA Alpha (alpha_lora)", "16", "Scaling multiplier (alpha / r = 2.0)"],
        ["LoRA PEFT", "LoRA Dropout", "0.05", "Prevents overfitting on parameter adapters"],
        ["PSD Regularization", "Lambda PSD (lambda_PSD)", "0.10 (sweep: [0.02, 0.20])", "Balance between LM loss and attention KL divergence (optimal: 0.05–0.10)"],
        ["PSD Regularization", "Semantic Weight (alpha)", "1.0", "Weight assigned to frozen SBERT cosine similarity"],
        ["PSD Regularization", "Position Weight (beta)", "0.5", "Weight penalizing normalized sequence distance"],
        ["PSD Regularization", "Graph Weight (gamma)", "0.5", "Weight penalizing normalized graph/table distance"],
        ["PSD Regularization", "Temperature (tau)", "1.0", "Softmax scaling temperature for target distribution"],
        ["PSD Regularization", "Monitored Layers", "Middle & Deep Layers (34–100% depth)", "Intermediate and deep Transformer layers where relational bindings form"],
        ["Optimizer & Training", "Optimizer", "AdamW", "Decoupled weight decay optimization"],
        ["Optimizer & Training", "Learning Rate", "2e-4", "Peak learning rate with linear warmup over 50 steps"],
        ["Optimizer & Training", "Weight Decay", "0.01", "L2 weight decay penalty on non-bias parameters"],
        ["Optimizer & Training", "Effective Batch Size", "4", "Batch size = 1, Gradient accumulation = 4"],
        ["Evaluation", "Generation Decoding", "Greedy Decoding", "Deterministic greedy token selection (temperature = 0.0)"],
        ["Evaluation", "Max New Tokens", "128", "Maximum generated verbalization length"],
        ["Evaluation", "Repetition Penalty", "1.05", "Penalizes immediate token repetition loops"]
    ]
    add_custom_table(doc, tbl_param_headers, tbl_param_rows, caption="Table A3.1: Complete experimental parameter grid for training, LoRA PEFT, and inference.")

    add_subsection_heading(doc, "A3.2 Step-by-Step Reproduction Guide (PowerShell Commands)")
    add_paragraph(
        doc,
        "Every experimental finding, table, and figure reported in this thesis can be independently reproduced from the repository root "
        "using the following deterministic commands executed in PowerShell:"
    )

    repro_commands = """# 1. Run Complete Automated PyTest Suite
python -m pytest tests/ -v

# 2. Stage 0–D Multi-Stage Data Pipeline & Graph Serialization
python -m src.cleaning --dataset totto --config config/cleaning.yaml
python -m src.eda --dataset totto --config config/eda.yaml
python -m src.normalization --dataset totto --config config/normalization.yaml
python -m src.graph_construction --dataset totto --config config/graph_construction.yaml

# 3. Target Formulation Component Decomposition Ablation Suite
python -u src/psd/run_ablations.py --components all

# 4. Long-Context Scaling Runs across L in {1024, 2048} Tokens
python -u src/psd/run_long_context_scaling.py --context_lengths 1024 2048

# 5. Cross-Architecture Generalization Benchmark Suite (8 Models, L = 256)
python -u src/psd/run_cross_architecture_benchmark.py --models gemma3_4b gemma2_9b llama32_3b llama31_8b mistral_7b qwen25_3b qwen25_7b phi35_mini

# 6. Loss Weight (lambda_PSD) and Layer Placement Ablation Sweep
python -u src/psd/run_hyperparameter_sweep.py --lambda_sweep 0.02 0.05 0.10 0.20 --layers early middle deep

# 7. Automated Claim-Level SPO Factuality and Hallucination Audit
python -u src/psd/run_factuality_eval.py --datasets totto webnlg dart"""
    add_code_block(doc, repro_commands, caption="Listing A3.1: Complete PowerShell reproduction command sequence.")
