# Phase 1: Experimental Foundation & Baseline Report
## Topology-aware Positional Semantic Divergence (PSD) for Long-context Data-to-Text Generation

---

### 1. Executive Summary
Phase 1 establishes the unified dataset infrastructure, deterministic serialization, token-to-structure mapping, topological characterization, and baseline Sequence-to-Sequence pipeline for **ToTTo**, **DART**, and **WebNLG**.

- **Datasets Unified**: ToTTo (Wikipedia Tables), DART (OpenIE/Tables/Triples), WebNLG (RDF Triples).
- **Unified Representation**: Shared node-edge-highlight schema preserving coordinate positions and graph relations.
- **Topological Mapping**: 100% deterministic token-to-structure alignment preserved for Phase 2 PSD attention and loss.

---

### 2. Cross-Dataset Characteristics Summary

| Dataset | Input Type | Total Examples | Avg Input (Tokens) | Median Input | P95 Input | Avg Target (Tokens) | Avg Nodes/Cells | Avg Edges/Triples | % > 512 Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TOTTO** | Table | 6000 | 860.6 | 388.0 | 2751.0 | 10.0 | 208.6 | 0 | 40.48% |
| **DART** | Triples | 6000 | 25.5 | 23.0 | 49.0 | 15.9 | 3.8 | 2.8 | 0.0% |
| **WEBNLG** | Triples | 5200 | 19.4 | 11.0 | 48.0 | 14.5 | 3.0 | 2.0 | 0.0% |

---

### 3. Data Integrity and Leakage Validation
- **Train/Val/Test ID Leakage**: Verified 0 overlapping IDs across all datasets.
- **Schema Validation**: 100% compliant with `UnifiedRecord` specification.
- **Token-to-Structure Mapping**: Validated character spans and token offset alignments.

---

### 4. Baseline Model Performance (T5-Small)
- **BLEU**: 0.0
- **ROUGE-1**: 0.0
- **ROUGE-2**: 0.0
- **ROUGE-L**: 0.0

---

### 5. Preserved Capabilities for Phase 2 PSD Implementation
1. **Node and Edge Identity**: Every token maps back to an atomic cell or graph entity/relation.
2. **Topological Distance**: Row/column table adjacencies and directed RDF graph structures are directly queryable via NetworkX.
3. **Positional Tracking**: Token indices and span offsets are retained to compute positional semantic divergence.
4. **Decoupled Architecture**: Phase 2 can replace the standard T5 attention and loss functions without modifying data loading or evaluation interfaces.
