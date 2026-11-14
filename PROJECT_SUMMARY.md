# Topology-Aware Data-to-Text Pipeline — Development Summary

## Executive Overview

Today, we built and verified the complete **first stage (Stages 0 through D)** of a research pipeline for a paper on **topology-aware long-context Data-to-Text generation**. The pipeline processes three heterogeneous benchmark datasets with different native formats:

1. **ToTTo**: Wikipedia tables with highlighted cell coordinates and reference text annotations.
2. **WebNLG 2020**: RDF triple sets `(subject, relation, object)` with single/multiple reference texts.
3. **DART**: A heterogeneous mix of RDF triples, tables, and OpenIE triples spanning multiple domains.

All code was built with Python 3.10+ type hints, clean docstrings, robust logging, YAML configuration files, standalone stage CLI commands, and a full unit test suite (`pytest`).

---

## Repository Architecture

```
psd_pipeline/
├── config/
│   ├── loaders.yaml
│   ├── cleaning.yaml
│   ├── eda.yaml
│   ├── normalization.yaml
│   └── graph_construction.yaml
├── data/
│   ├── cleaned/              # Stage A output ({dataset}.jsonl)
│   ├── normalized/           # Stage C output ({dataset}.jsonl)
│   └── graphs/               # Stage D output ({dataset}/*.gpickle)
├── src/
│   ├── loaders/              # Stage 0: Native parsers & auto-detection factory
│   │   ├── base_loader.py
│   │   ├── totto_loader.py
│   │   ├── webnlg_loader.py
│   │   ├── dart_loader.py
│   │   └── loader_factory.py
│   ├── cleaning/             # Stage A: Schema validation, cleaner, topology flagger
│   │   ├── schema_validator.py
│   │   └── cleaner.py
│   ├── eda/                  # Stage B: Statistics, HuggingFace tokenizer, plots, HTML report
│   │   ├── tokenizer_utils.py
│   │   ├── analyzer.py
│   │   └── html_reporter.py
│   ├── normalization/        # Stage C: Shared JSON schema converter & roundtrip validator
│   │   ├── schema.py
│   │   └── converter.py
│   └── graph_construction/   # Stage D: NetworkX DiGraph builder, gpickle serializer, renderer
│       ├── visualizer.py
│       └── builder.py
├── reports/
│   ├── cleaning_report.csv
│   ├── graph_construction_report.csv
│   ├── {dataset}_statistics.csv
│   ├── {dataset}_table_statistics.csv
│   ├── {dataset}_relation_statistics.csv
│   ├── EDA_Report.html
│   └── plots/
│       └── {dataset}/
│           ├── sentence_and_token_histograms.png
│           ├── vocab_growth_zipf_plot.png
│           ├── missing_value_heatmap.png
│           ├── relation_entity_frequencies.png
│           ├── totto_table_distributions.png
│           ├── correlation_heatmap.png
│           └── example_graphs/
├── tests/
│   ├── test_loaders.py
│   ├── test_schema_validation.py
│   ├── test_normalization.py
│   └── test_graph_construction.py
├── PROJECT_SUMMARY.md
└── README.md
```

---

## Stage-by-Stage Implementation & Results

### Stage 0 — Native Loaders & Auto-Detection (`src/loaders/`)
- Built `BaseLoader` interface and `RawRecord` data structures.
- Created `ToTToloader` to parse JSONL table grids, cell highlight coordinates, section headers, and reference sentence annotations.
- Created `WebNLGloader` to recursively parse WebNLG XML (`<entry>`, `<mtriple>`, `<lex>`) and JSON formats.
- Created `DARTloader` to parse DART triple-set JSON structures and annotations.
- Implemented `LoaderFactory` to auto-detect dataset types directly from file paths or content schemas without requiring hardcoded flags.
- **Verification**: `tests/test_loaders.py` verified parsing and auto-detection.

### Stage A — Dataset Cleaning & Validation (`src/cleaning/`)
- Built `schema_validator.py` to reject malformed records.
- Built `cleaner.py` to:
  - Remove exact & near-duplicate records.
  - Remove empty samples (missing tables/triples or missing reference sentences).
  - Normalize Unicode (NFC), quotation marks, punctuation, zero-width space characters (`\u200b`), and whitespace.
  - Normalize entity and relation casing/synonyms (e.g. `leaderName` -> `leader`, `birthPlace` -> `place_of_birth`).
  - Flag graph topology characteristics (isolated nodes, disconnected components, cycles) without silently dropping valid records.
  - Log reason-coded actions (`DROP`, `FIX`, `FLAG`) to `reports/cleaning_report.csv`.
- **Output Data**:
  - `data/cleaned/totto.jsonl` (7,700 records)
  - `data/cleaned/webnlg.jsonl` (1,667 records)
  - `data/cleaned/dart.jsonl` (2,768 records)

### Stage B — Exploratory Data Analysis (`src/eda/`)
- Implemented `PipelineTokenizer` supporting HuggingFace `t5-base` (configurable in `config/eda.yaml`) with automatic BPE/whitespace fallback.
- Computed dataset-level statistics: sentence & token length distributions, Zipf rank-frequency distributions, vocabulary growth curves.
- Computed WebNLG & DART predicate/entity frequencies, triple counts, and relation diversity.
- Computed ToTTo table dimensions, density ratios, cell distance metrics, and highlighted cell position heatmaps.
- Rendered high-resolution PNG plots in `reports/plots/{dataset}/`.
- Generated CSV statistics tables and a unified, side-by-side HTML comparison report at `reports/EDA_Report.html`.

### Stage C — Structured Data Normalization (`src/normalization/`)
- Converted cleaned dataset records into one shared internal JSON schema:
  ```json
  {
    "id": "string",
    "dataset": "totto | webnlg | dart",
    "source_type": "table | graph | mixed",
    "nodes": [{"node_id": "string", "text": "string", "attributes": {}}],
    "edges": [{"source": "node_id", "target": "node_id", "relation": "string", "attributes": {}}],
    "reference_texts": ["string"],
    "metadata": {}
  }
  ```
- **ToTTo**: Cells represented as nodes (`r{row}_c{col}`), adjacent row/column cells linked via directed `row_adjacent` and `column_adjacent` edges.
- **WebNLG & DART**: Entities represented as nodes, triples linked via directed edges labeled with relations (`triple`).
- Verified 100% round-trip fidelity with zero node/edge loss into `data/normalized/{dataset}.jsonl`.

### Stage D — Graph Construction & NetworkX Serialization (`src/graph_construction/`)
- Built `networkx.DiGraph` instances attaching node attributes (`text`, `is_header`, `is_highlighted`, `row_idx`, `col_idx`) and edge attributes (`relation`, `edge_type`).
- Serialized 100% of constructed graphs into binary `.gpickle` format under `data/graphs/{dataset}/{record_id}.gpickle`.
- **Graph Construction Summary**:
  - **ToTTo**: 7,700 / 7,700 graphs (100% success, avg nodes: 31.42, avg edges: 51.58)
  - **WebNLG**: 1,667 / 1,667 graphs (100% success, avg nodes: 4.01, avg edges: 2.92)
  - **DART**: 2,768 / 2,768 graphs (100% success, avg nodes: 4.88, avg edges: 3.51)
- Logged construction summary metrics to `reports/graph_construction_report.csv`.
- Rendered visual sample graph PNGs (small, medium, large) under `reports/plots/{dataset}/example_graphs/*.png`.

---

## Unit Testing & Verification Results

Ran PyTest across all test suites:
```bash
python -m pytest tests/ -v
```
**Results**: `13 passed in 1.09s` (100% pass rate).

All outputs, reports, serialized graphs, visual plots, and documentation are ready for downstream pipeline stages (PSD Scoring, Attention Modification, Fine-Tuning).
