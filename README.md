# Topology-Aware Positional Semantic Divergence (PSD)

### Topology-Aware Auxiliary Attention Supervision for Long-Context Data-to-Text Generation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/Transformers-4.36%2B-yellow.svg)](https://huggingface.co/transformers/)
[![PEFT LoRA](https://img.shields.io/badge/PEFT-LoRA%20(r%3D8)-green.svg)](https://github.com/huggingface/peft)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](#)

---

## 1. Introduction

> **Bridging structural topologies and autoregressive decoders: Enforcing relational and semantic priors directly inside multi-head self-attention without architectural modifications.**

Autoregressive Small Language Models (SLMs) trained with next-token prediction loss ($\mathcal{L}_{\text{LM}}$) treat structured inputs as flat sequences of serialized tokens. Under long context horizons ($L \in [512, 1024, 2048]$), standard self-attention mechanisms suffer from **attention dispersion**—distributional entropy spikes, and intermediate attention heads lose focus on structurally bound entities. This dispersion directly causes:

1. **Structural Hallucinations**: Generating relationships that contradict or do not exist in the source graph.
2. **Positional Semantic Drift**: Prioritizing pretraining memorization over topological proximity in the input tables and graphs.
3. **Context Scaling Breakdown**: Rapid degradation of factual precision as prompt sequence length expands.

**Topology-Aware Positional Semantic Divergence (PSD)** addresses this limitation. Rather than requiring complex external Graph Neural Network (GNN) encoders or invasive Transformer architectural modifications, PSD constructs an external, differentiable structural-semantic prior ($P_{\text{target}}$) and aligns the model's internal attention distributions via an auxiliary **Kullback-Leibler (KL) divergence objective** during Parameter-Efficient Fine-Tuning (PEFT LoRA).

Because PSD regularizes only the self-attention weights during training, the final model runs with **zero inference latency overhead, zero extra parameters at runtime, and standard greedy or sampling decoders**.

---

## 2. Key Features and Contributions

- **Multi-Modal Topological Distance Spaces**:
  - **Knowledge Graphs (WebNLG, DART)**: Geodesic shortest-path distance matrix ($d_G$) computed via Breadth-First Search (BFS) over symmetric graph projections ($G_{\text{sym}}$) with bounded disconnected component penalties.
  - **2D Relational Tables (ToTTo)**: Manhattan grid coordinate distance ($d_T$) capturing row/column adjacency and cell locality.
- **Frozen Contextual Semantic Representation**:
  - Independent dense semantic embeddings via a frozen `all-MiniLM-L6-v2` encoder ($E_{\text{sem}}$), providing a strictly non-cyclical, stable optimization target ($\nabla_\theta S_{i,j} = 0$).
- **Strictly Causal Prior Synthesis**:
  - Autoregressive future key masking (sentinel value $-10^9$ for $j > i$) coupled with temperature-scaled row-wise softmax normalization ($\tau = 1.0$) ensuring valid probability measures over preceding keys.
- **Layer-Selective Attention Supervision**:
  - Selectively regularizes intermediate and deep Transformer layers (34%–100% depth percentile) where relational and factual binding occurs, preserving early syntactic token-processing layers.
- **Parameter-Efficient Adaptation**:
  - Native FP16 base backbone with trainable Low-Rank Adaptation (LoRA) on query and value projections ($r=8, \alpha=16$), modifying less than 0.08% of total network parameters.
- **Dual Evaluation and Factuality Audit**:
  - Combines standard lexical/semantic metrics (BLEU-4, ROUGE-L, METEOR, BERTScore F1) with an automated, dependency-guided Subject-Predicate-Object (SPO) factual audit.

---

## 3. System Architecture and Pipeline

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Raw Input Modalities                            │
│   Wikipedia Tables (ToTTo)  │  RDF Triples (WebNLG)  │  OpenIE (DART)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            Stage A & C: Conservative Normalization & Cleaning          │
│     Unicode NFC • Mojibake Resolution • Schema Integrity Validation    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│           Stage D: Multi-Modal Graph & Coordinate Ingestion            │
│  BFS All-Pairs Shortest Path (Graphs) │ Manhattan Coordinates (Tables) │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
       ┌────────────────────────┐      ┌───────────────────────────┐
       │ Dense Semantic Encoder │      │ Deterministic Linearizer  │
       │   all-MiniLM-L6-v2     │      │ Token-to-Node Mapping μ(i)│
       └────────────┬───────────┘      └─────────────┬─────────────┘
                    │                                │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│             Topological Semantic Prior Synthesis (P_target)            │
│          R_{i,j} = α S_{i,j} - β d_P(i,j) - γ d_G(i,j)                │
│          Mask future positions (j > i) with -10^9                      │
│          Causal Row-Wise Softmax Normalization (τ = 1.0)               │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│           Joint Layer-Selective PEFT Optimization (Algorithm 2)        │
│                                                                        │
│   Autoregressive Target Loss:          PSD Attention Supervision Loss: │
│     L_LM(θ) = -Σ log P_θ(y_m | X, y<m)   L_PSD(θ) = KL(P_target || A)  │
│                                                                        │
│         Total Loss: L_total = L_LM(θ) + λ_PSD · L_PSD(θ)               │
│         Gradients backpropagated strictly into LoRA (W_q, W_v)         │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Evaluation & Factual Grounding Audit                   │
│   Lexical Metrics (BLEU/ROUGE) • BERTScore • Dependency SPO Factuality │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Mathematical Formulation

### 4.1 Structural-Semantic Affinity Matrix

For all input prompt token pairs ($x_i, x_j$) at sequence positions $i, j \in \{1, \dots, N\}$, the composite structural-semantic affinity score is defined as:

$$
R_{i,j} = \alpha S_{i,j} - \beta d_P^{\text{norm}}(i,j) - \gamma d_G^{\text{norm}}(i,j)
$$

Where the individual components are formulated as:

1. **Contextual Semantic Similarity**: Computed using the cosine similarity of frozen contextual embeddings from $E_{\text{sem}}$ (`all-MiniLM-L6-v2`):

$$
S_{i,j} = \text{clamp}\left(\frac{\mathbf{h}_{\mu(i)}^\top \mathbf{h}_{\mu(j)} + 1}{2}, 0.0, 1.0\right)
$$

2. **Normalized Sequential Distance**: Penalizes linear token displacement to preserve local syntax:

$$
d_P^{\text{norm}}(i, j) = \frac{|i - j|}{\max(N - 1, 1)}
$$

3. **Normalized Topological Distance**: Computed over graph shortest paths or tabular Manhattan coordinates:

$$
d_G^{\text{norm}}(i, j) = \begin{cases} \frac{d_G(\mu(i), \mu(j))}{\max_{u,v} d_G(u, v)}, & \text{if } \mu(i) \ne \text{None} \text{ and } \mu(j) \ne \text{None} \\ 1.0, & \text{otherwise} \end{cases}
$$

Default hyperparameters: $\alpha = 1.0, \beta = 0.5, \gamma = 0.5$.

---

### 4.2 Causal Target Attention Distribution

Autoregressive self-attention enforces a strict causal constraint: each query token at position $i$ may only attend to preceding key tokens $j \le i$. Future key positions ($j > i$) are masked with a sentinel value $-10^9$:

$$
\tilde{R}_{i,j} = \begin{cases} \frac{R_{i,j}}{\tau}, & \text{if } j \le i \\ -10^9, & \text{if } j > i \end{cases}
$$

Where $\tau > 0$ denotes the temperature parameter (default: $\tau = 1.0$), and $\tilde{R}_{i,j}$ represents the causal masked logits. Applying row-wise softmax normalization yields the causal prior distribution:

$$
P_{\text{target}}(j \mid i) = \frac{\exp(\tilde{R}_{i,j})}{\sum_{k=1}^i \exp(\tilde{R}_{i,k})}, \quad \forall j \le i
$$

This ensures that $\sum_{j=1}^i P_{\text{target}}(j \mid i) = 1.0$ for every valid query token.

---

### 4.3 Layer-Selective Attention Supervision Objective

We supervise designated intermediate and deep Transformer layers (depth percentiles 34%–100%) across validly aligned prompt tokens via directional Kullback-Leibler divergence:

$$
\mathcal{L}_{\text{PSD}}(\theta) = \frac{1}{|\mathcal{L}_{\text{selected}}|} \sum_{l \in \mathcal{L}_{\text{selected}}} \frac{1}{|\mathcal{V}_{\text{valid}}|} \sum_{i \in \mathcal{V}_{\text{valid}}} D_{\text{KL}}\left(P_{\text{target}}(\cdot \mid i) \,\|\, \bar{A}^{(l)}(\cdot \mid i)\right)
$$

Where:
* $\mathcal{L}_{\text{selected}}$ denotes the set of monitored intermediate and deep layers.
* $\mathcal{V}_{\text{valid}} = \{i \mid \mu(i) \ne \text{None}\}$ denotes the set of tokens aligned to valid graph or table nodes.
* $\bar{A}^{(l)}$ is the head-averaged empirical attention at layer $l$:

$$
\bar{A}^{(l)}(j \mid i) = \frac{1}{H} \sum_{h=1}^H A_{l,h}(j \mid i)
$$

---

### 4.4 Joint Optimization Objective

The total loss jointly optimizes next-token language modeling and structural attention regularisation:

$$
\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{LM}}(\theta) + \lambda_{\text{PSD}} \cdot \mathcal{L}_{\text{PSD}}(\theta)
$$

Where prompt tokens are masked out from next-token cross-entropy loss ($\text{labels}_k = -100$ for $k \le N$).

---

## 5. Experimental Results

### 5.1 Long-Context Scaling Performance ($L = 1024$ and $L = 2048$ Tokens)

Evaluated on an NVIDIA H100 PCIe (80 GB VRAM) across leading Small Language Model architectures:

| Model Family & Variant | Parameters | Context | Mode | BLEU-4 | ROUGE-L | BERTScore | Factual Precision | Hallucination Rate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gemma 3** | 4B | $L = 1024$ | Baseline LoRA | 19.13 | 57.22 | 92.52 | 84.43% | 15.57% |
| **Gemma 3** | 4B | $L = 1024$ | **PSD (Ours)** | **19.73** | **60.74** | 92.26 | 83.16% | 16.84% |
| **Gemma 3** | 4B | $L = 2048$ | Baseline LoRA | 22.70 | 58.38 | 93.22 | 87.50% | 12.50% |
| **Gemma 3** | 4B | $L = 2048$ | **PSD (Ours)** | **27.47** *(+21.0%)* | **65.06** *(+6.68)* | **93.50** | **88.18%** | **11.82%** |
| **Llama 3.2** | 3B | $L = 1024$ | Baseline LoRA | 16.60 | 53.68 | 92.09 | 84.01% | 15.99% |
| **Llama 3.2** | 3B | $L = 1024$ | **PSD (Ours)** | 15.44 | 53.26 | 91.91 | **86.32%** | **13.68%** |
| **Llama 3.2** | 3B | $L = 2048$ | Baseline LoRA | 22.59 | 57.10 | 93.27 | 87.38% | 12.62% |
| **Llama 3.2** | 3B | $L = 2048$ | **PSD (Ours)** | 22.03 | 55.22 | 92.82 | **87.82%** | **12.18%** |
| **Qwen 2.5** | 3B | $L = 1024$ | Baseline LoRA | 12.31 | 51.03 | 91.31 | 80.00% | 20.00% |
| **Qwen 2.5** | 3B | $L = 1024$ | **PSD (Ours)** | **12.67** | **57.21** | **91.86** | **87.84%** | **12.16%** |
| **Qwen 2.5** | 3B | $L = 2048$ | Baseline LoRA | 19.76 | 57.33 | 93.08 | 88.48% | 11.52% |
| **Qwen 2.5** | 3B | $L = 2048$ | **PSD (Ours)** | 18.82 | **59.45** | **93.80** | **90.31%** | **9.69%** |

---

### 5.2 Cross-Architecture Generalization Benchmark ($L = 256$ Tokens)

To confirm architectural agnosticism, PSD was evaluated across 8 backbones spanning 3B to 9B parameters:

| Model Backbone | Parameters | Baseline BLEU-4 | PSD BLEU-4 | Baseline ROUGE-L | PSD ROUGE-L | Baseline Factuality | PSD Factuality |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gemma 3** | 4B | 4.68 | **7.09** | 30.73 | **47.42** | 76.98% | **83.93%** *(+6.95%)* |
| **Gemma 2** | 9B | 5.79 | **6.34** | 43.16 | **45.84** | 78.23% | 77.27% |
| **Llama 3.2** | 3B | 5.37 | **7.58** | 40.07 | **41.16** | 78.91% | 77.54% |
| **Llama 3.1** | 8B | 5.97 | **14.84** | 36.32 | **37.99** | 58.52% | **82.56%** *(+24.0%)* |
| **Mistral v0.3** | 7B | 9.60 | **14.07** | 49.04 | **50.58** | 76.25% | **79.71%** *(+3.46%)* |
| **Qwen 2.5** | 3B | 4.66 | 4.37 | 40.67 | 40.63 | 67.02% | 67.38% |
| **Qwen 2.5** | 7B | 6.79 | **7.01** | 46.25 | 46.19 | 78.36% | **83.46%** *(+5.10%)* |
| **Phi-3.5-mini**| 3.8B | 6.59 | 6.55 | 41.56 | 39.50 | 80.51% | **82.20%** *(+1.69%)* |

---

### 5.3 Target Distribution Component Decomposition Ablation

| Ablation Configuration | Semantic ($S$) | Position ($d_P$) | Graph ($d_G$) | BLEU-4 | ROUGE-L | BERTScore F1 | Delta BLEU-4 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (LM Loss Only)** | No | No | No | 7.25 | 46.42 | 89.76 | — |
| **Semantic Only** | Yes | No | No | 10.53 | 46.07 | 89.77 | +3.28 (+45.2%) |
| **Position Only** | No | Yes | No | **17.64** | 46.72 | 90.50 | +10.39 (+143%) |
| **Graph Only** | No | No | Yes | 17.21 | 45.73 | 89.79 | +9.96 (+137%) |
| **Semantic + Graph** | Yes | No | Yes | 12.25 | 45.86 | 89.78 | +5.00 (+69.0%) |
| **Position + Graph** | No | Yes | Yes | 10.64 | 45.13 | 89.47 | +3.39 (+46.8%) |
| **Full PSD (Ours)** | Yes | Yes | Yes | 12.15 | **49.72** | **90.91** | **+4.90 (+67.6%)** |

---

### 5.4 Qualitative Error Case Study (Factual Grounding)

| Method | Generated Verbalization | Factual Grounding Analysis |
|:---|:---|:---|
| **Input Graph** | `[Harry Kane] ──(Club)──> [Bayern Munich]`<br>`[Harry Kane] ──(Goals)──> [36]`<br>`[Harry Kane] ──(Season)──> [2023-24]`<br>`[Harry Kane] ──(Competition)──> [Bundesliga]` | Ground Truth Input: 4 explicit relational triples. |
| **Reference** | *"Harry Kane scored 36 Bundesliga goals for Bayern Munich in the 2023-24 season."* | 100% Entailed reference sentence. |
| **Baseline LoRA** | *"Harry Kane played for Bayern Munich and Tottenham Hotspur, scoring 36 goals in the Champions League."* | **[Hallucinated]**: Fused pretraining memorization ('Tottenham Hotspur', 'Champions League'). Relational drift occurred due to unconstrained self-attention. |
| **PSD (Ours)** | *"In the 2023-24 Bundesliga season, Harry Kane scored 36 goals for Bayern Munich."* | **[Factual]**: Attention heads anchored to geodesic shortest-paths; all 4 facts verbalized with zero ungrounded claims. |

---

## 6. Tech Stack and Requirements

* **Core Framework**: Python 3.10+, PyTorch 2.0+, Hugging Face Transformers 4.36+
* **Parameter-Efficient Tuning**: PEFT 0.7+ (LoRA), BitsAndBytes 0.41+
* **Topological & Graph Processing**: NetworkX 3.0+, SciPy 1.10+, NumPy 1.24+
* **Semantic Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Evaluation Suite**: `sacrebleu`, `rouge-score`, `bert-score`, `nltk`
* **Data & Schema**: `pydantic` v2, `pandas`, `pyyaml`

---

## 7. Repository Structure

```
psd-d2t/
├── config/                     # Preprocessing and stage YAML configurations
│   ├── cleaning.yaml
│   ├── eda.yaml
│   ├── graph_construction.yaml
│   ├── loaders.yaml
│   └── normalization.yaml
├── configs/                    # Training and ablation experiment configs
│   ├── baseline_config.yaml
│   ├── phase2b_baseline_config.yaml
│   ├── phase2b_psd_config.yaml
│   ├── phase3a_*.yaml          # 8 Component ablation configs
│   ├── phase3b_*.yaml          # Confirmation ablation configs
│   └── phase4b_*.yaml          # Long-context scaling configs (512, 1024)
├── src/
│   ├── baseline/               # Baseline seq2seq wrappers and evaluation
│   ├── base_model/             # Base model loading, quantization, memory benchmarking
│   ├── cleaning/               # Stage A: schema validation, conservative cleaner
│   ├── data/                   # Unified schema models, loaders (ToTTo, WebNLG, DART)
│   ├── eda/                    # Stage B: statistical analysis, length breakdown, plots
│   ├── graph_construction/     # Stage D: NetworkX graph builder, shortest paths
│   ├── loaders/                # Multi-dataset parsers and factory
│   ├── normalization/          # Shared schema normalizer and round-trip verification
│   ├── preprocessing/          # Sequence serialization and token-to-node mapping
│   └── psd/                    # Core PSD Engine:
│       ├── divergence.py       # KL divergence loss implementation
│       ├── evaluate.py         # Automated lexical + factual evaluation
│       ├── graph_distance.py   # BFS geodesic and Manhattan distance matrices
│       ├── semantic_encoder.py # Frozen contextual SBERT encoder
│       ├── target_distribution.py # Causal masked prior synthesis
│       └── trainer.py          # Joint PEFT LoRA training loop
├── tests/                      # Comprehensive test suite (100% passing)
│   ├── test_cleaner.py
│   ├── test_eda_and_metrics.py
│   ├── test_loaders.py
│   ├── test_psd.py
│   ├── test_phase2b.py
│   ├── test_phase3a.py
│   └── test_phase4b.py
├── scripts/                    # Auxiliary visualization and helper scripts
├── reports/                    # Benchmark metrics, CSV summaries, and ablation plots
├── requirements.txt            # Python dependencies
├── LICENSE                     # Apache 2.0 License
└── README.md                   # This documentation
```

---

## 8. Quickstart and Reproducibility Guide

### 8.1 Environment Setup

```bash
# Clone the repository
git clone https://github.com/ishant212/psd-d2t.git
cd psd-d2t

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 8.2 Run Test Suite
Validate the entire pipeline components (loaders, distance matrices, causal masking, divergence loss):
```bash
python -m pytest tests/ -v
```

### 8.3 Fast Verification Smoke Test (For Reviewers)
Run a quick, self-contained verification smoke test validating PSD attention extraction and joint loss computation without downloading multi-gigabyte backbones:
```bash
python -m pytest tests/test_psd.py -k "test_target_distribution or test_causal_masking" -v
```

### 8.4 Preprocessing and Graph Construction
To process raw tabular and graph datasets into serialized prompts with token alignments:
```bash
# Stage A: Clean and validate raw datasets
python -m src.cleaning

# Stage D: Construct NetworkX graphs and compute geodesic distance matrices
python src/graph_construction/graph_builder.py --config config/graph_construction.yaml
```

### 8.5 Training with Topology-Aware PSD
To launch parameter-efficient fine-tuning with PSD regularisation:
```bash
# Train Qwen-2.5 / Gemma-3 with PSD attention supervision
python -u src/psd/trainer.py --config configs/phase2b_psd_config.yaml --steps 500
```

### 8.6 Evaluation and Factual Audit
To run the dual evaluation suite (lexical metrics + claim-level SPO factuality audit):
```bash
python src/psd/evaluate.py \
    --model_dir reports/experiments/phase2b/psd/checkpoints/final \
    --benchmark totto \
    --eval_factuality
```

---

## 9. Citation

If you find this work, codebase, or methodology useful in your research, please cite our paper:

```bibtex
@article{psd2026topology,
  title  = {Topology-Aware Positional Semantic Divergence Modeling for Long-Context Data-to-Text Generation},
  author = {Ishant Shekhar Eeshu and Kulin Mathur and Lakshya Kumar Singh},
  year   = {2026},
  url    = {https://github.com/ishant212/psd-d2t}
}
```

---

## 10. License and Acknowledgements

* This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
* We acknowledge the creators of the **ToTTo** (Google Research), **WebNLG** (CNRS/Loria), and **DART** (Yale University) benchmark datasets.
