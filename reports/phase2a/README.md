# Phase 2A: Controlled Experiments Infrastructure & Verification Report

**Project:** Topology-aware Positional Semantic Divergence (PSD) for Long-context Data-to-Text Generation  
**Phase:** 2A — Controlled Baseline vs. PSD Scientific Experiment Setup & Smoke Test Verification  
**Hardware Profile:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, 7.77 GiB usable), 24 GB System RAM, Windows 11 (x86_64), PyTorch 2.11.0+cu128, Python 3.13.7.

---

## 1. Experiment Objective

The objective of Phase 2A is to transition from component-level implementation and verification to a rigorous, controlled scientific experiment comparing:
- **Experiment A (Baseline):** A standard causal LM Data-to-Text generator fine-tuned with PEFT LoRA on Qwen3-4B optimizing pure causal cross-entropy loss:
  $$\mathcal{L}_{\text{baseline}} = \mathcal{L}_{\text{LM}}$$
- **Experiment B (PSD):** The exact same Qwen3-4B + LoRA model fine-tuned with the joint Positional Semantic Divergence (PSD) objective:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \lambda_{\text{PSD}} \cdot \mathcal{L}_{\text{PSD}} \quad (\lambda_{\text{PSD}} = 0.1)$$

The scientific goal is to empirically test whether aligning empirical self-attention with a target topological-semantic affinity distribution reduces positional degradation (Lost-in-the-Middle) and improves generation faithfulness across diverse graph and tabular structures.

---

## 2. Experimental Configurations

Both experiments share 100% identical base checkpoints, quantization parameters, LoRA ranks/targets, tokenizers, data preprocessing, sequence lengths, batch sizes, optimizers, learning rates, and random seeds.

### 2.1 Experiment A: Baseline (`configs/baseline_config.yaml`)
- **Base Model:** `models/qwen3_4b_hf` (Qwen3-4B in Hugging Face Safetensors format)
- **Quantization:** BitsAndBytes 4-bit NF4 (`bnb_4bit_quant_type="nf4"`, double quant, float16 compute dtype)
- **Trainable Parameters:** PEFT LoRA on `["q_proj", "v_proj"]`, $r=8$, $\alpha=16$, dropout=0.05 (~2,949,120 parameters, 0.0733% of base model)
- **PSD Status:** Disabled ($\lambda_{\text{PSD}} = 0.0$, no target graph construction, no attention retention overhead)
- **Loss Function:** Pure causal cross-entropy $\mathcal{L}_{\text{LM}}$ over target tokens (prompt tokens masked to `-100`)
- **Checkpoints:** `models/phase2a/baseline/`
- **Reports:** `reports/phase2a/baseline_training.json`, `reports/phase2a/baseline_training.csv`

### 2.2 Experiment B: PSD (`configs/psd_experiment_config.yaml`)
- **Base Model & LoRA:** Identical to baseline (`models/qwen3_4b_hf`, NF4 4-bit, LoRA $r=8, \alpha=16$)
- **Selected Transformer Layers:** Zero-based indices `[11, 23, 34]` $\leftrightarrow$ 1-based Layers `[12, 24, 35]`
- **Affinity Formulation:**
  $$R_{ij} = \alpha S_{ij} - \beta d_{P,\text{norm}}(i, j) - \gamma d_{G,\text{norm}}(i, j)$$
  where $\alpha = 1.0$, $\beta = 0.5$, $\gamma = 0.5$, temperature $\tau = 1.0$, $\lambda_{\text{PSD}} = 0.1$.
- **Ablation Hooks:** Supports toggling components via `enable_semantic`, `enable_positional`, `enable_graph` in future runs.
- **Checkpoints:** `models/phase2a/psd/`
- **Reports:** `reports/phase2a/psd_training.json`, `reports/phase2a/psd_training.csv`

---

## 3. Dataset & Data Usage

- **Datasets:** ToTTo (tables), DART (triples/open-domain), WebNLG (triples/RDF)
- **Splits:** Cleaned splits from Phase 1 (`data/unified/train.jsonl`, `data/unified/validation.jsonl`, `data/unified/test.jsonl`)
- **Controlled Data Feeding:** Both baseline and PSD trainers initialize with fixed random seeds (`seed: 42`), ensuring both models encounter identical training examples in identical order.
- **Prompt Masking & Truncation Guarantee:**
  - Prompt tokens (structural markup, prefixes, suffixes) are assigned label `-100`.
  - Causal cross-entropy is evaluated strictly on target reference tokens.
  - Truncation budget preserves target text 100% intact, truncating only the serialized prompt from the right if sequence exceeds `max_seq_len`.

---

## 4. Smoke Test Results & GPU Resource Profile

Both 5-step smoke tests were executed on the local NVIDIA RTX 4060 Laptop GPU.

### 4.1 Baseline Smoke Test (5 Steps)
- **Initial Loss (Step 1):** LM Loss = 3.8706, Val LM Loss = 3.6352, Grad Norm = 2.3471
- **Final Loss (Step 5):** LM Loss = 2.7015, Val LM Loss = 2.8407, Grad Norm = 3.2844
- **Peak GPU Allocated VRAM:** **2,677.7 MB (~2.68 GB)**
- **Trainable Parameters:** 2,949,120 / 4,025,417,216 (0.0733%)
- **Status:** Gradients propagated cleanly to LoRA parameters; base model remained frozen.

### 4.2 PSD Smoke Test (5 Steps)
- **Initial Loss (Step 1):** Total Loss = 4.0477 (LM: 3.8706, PSD: 1.7707), Val LM = 3.6376, Val PSD = 1.7340, Grad Norm = 2.3478
- **Final Loss (Step 5):** Total Loss = 2.8556 (LM: 2.6849, PSD: 1.7070), Val LM = 2.8247, Val PSD = 1.7298, Grad Norm = 2.9920
- **Per-Layer PSD Divergence (Step 5):**
  - Layer 12 (0-indexed 11): 1.6371
  - Layer 24 (0-indexed 23): 1.7584
  - Layer 35 (0-indexed 34): 1.7255
- **Peak GPU Allocated VRAM:** **2,689.7 MB (~2.69 GB)**
- **Status:** Gradients backpropagated through attention layers to LoRA weights; VRAM remained stable under 2.7 GB.

---

## 5. Evaluation Methodology

Evaluation engine (`src/psd/evaluate.py`) computes standard Data-to-Text and structural faithfulness metrics on identical validation examples:
1. **BLEU:** Corpus BLEU score via `sacrebleu` (with smoothed sentence BLEU fallback).
2. **ROUGE-1, ROUGE-2, ROUGE-L:** Recall/precision/F1 LCS evaluation via `rouge-score`.
3. **Structured Faithfulness / Coverage:**
   - **Triple Coverage (DART / WebNLG):** Verifies presence of source entities and relations in output text.
   - **Highlighted Cell Coverage (ToTTo):** Verifies presence of highlighted table cell values in output text.
4. **Output Artifacts:** `reports/phase2a/metrics_comparison.json` and `reports/phase2a/metrics_comparison.csv`.

---

## 6. Position-Controlled Experiment Methodology

To directly test the positional degradation hypothesis, `src/psd/position_control.py` generates 3 controlled serialization variants for each record:
1. **BEGINNING:** Relevant facts / highlighted cells placed at the beginning of the prompt; context/filler follows.
2. **MIDDLE:** Context/distractors placed first, relevant facts placed in the middle, remainder placed after.
3. **END:** Context/filler placed first, relevant facts placed at the end of the prompt.

Both Baseline and PSD models are evaluated across all 3 variants. The goal is to determine whether PSD mitigates the performance drop observed when critical information is placed in the middle.
- **Output Artifacts:** `reports/phase2a/position_controlled_results.json` and `reports/phase2a/position_controlled_results.csv`.

---

## 7. Context Length Analysis Methodology

`src/psd/context_length_analysis.py` categorizes examples into standardized token length buckets:
- $\le 128$ tokens
- $129 - 256$ tokens
- $257 - 512$ tokens
- $513 - 1024$ tokens
- $> 1024$ tokens

Since Phase 1 established that ToTTo contains the largest long-context component (~20.1% >512 tokens), this analysis isolates how sequence length impacts baseline vs. PSD generation quality.
- **Output Artifacts:** `reports/phase2a/context_length_results.json` and `reports/phase2a/context_length_results.csv`.

---

## 8. Summary of Generated Files & Artifacts

| Component | Path | Description |
|---|---|---|
| **Baseline Config** | `configs/baseline_config.yaml` | Controls Experiment A (LM loss only) |
| **PSD Config** | `configs/psd_experiment_config.yaml` | Controls Experiment B (LM + 0.1 * PSD loss) |
| **Trainer Runner** | `src/psd/trainer.py` | Unified trainer with validation logging, CSV/JSON reporting |
| **Evaluation Engine** | `src/psd/evaluate.py` | Comparative evaluation across ToTTo, DART, WebNLG |
| **Position Controller** | `src/psd/position_control.py` | Positional degradation generator and evaluator |
| **Length Analyzer** | `src/psd/context_length_analysis.py` | Length bucketing and comparative analysis |
| **Plotter** | `src/psd/plot_results.py` | Generates 4 figures under `reports/phase2a/plots/` |
| **Unit Tests** | `tests/test_phase2a.py` | Comprehensive Phase 2A test suite (8 test suites) |
| **Baseline Checkpoint** | `models/phase2a/baseline/` | Saved baseline LoRA adapter and experiment metadata |
| **PSD Checkpoint** | `models/phase2a/psd/` | Saved PSD LoRA adapter and experiment metadata |
| **Baseline Report** | `reports/phase2a/baseline_training.json` | 5-step baseline training log and metrics |
| **PSD Report** | `reports/phase2a/psd_training.json` | 5-step PSD training log and layer-wise divergence |
| **Plot** | `reports/phase2a/plots/psd_by_layer.png` | Layer-wise PSD trajectory figure |

---

## 9. Exact Execution Commands

### Baseline Smoke Test / Training
```bash
python src/psd/trainer.py --config configs/baseline_config.yaml --steps 5
```

### PSD Smoke Test / Training
```bash
python src/psd/trainer.py --config configs/psd_experiment_config.yaml --steps 5
```

### Full Test Suite Execution
```bash
python -m pytest tests/ -v
```

### Evaluation Comparison
```bash
python src/psd/evaluate.py --baseline_dir models/phase2a/baseline --psd_dir models/phase2a/psd --samples 15
```

### Position-Controlled Experiment
```bash
python src/psd/position_control.py --baseline_dir models/phase2a/baseline --psd_dir models/phase2a/psd --dataset totto --samples 15
```

### Context Length Analysis
```bash
python src/psd/context_length_analysis.py --baseline_dir models/phase2a/baseline --psd_dir models/phase2a/psd --dataset totto --records 30
```

### Plot Generation
```bash
python src/psd/plot_results.py
```

---

## 10. Scientific Integrity & Known Limitations

> [!IMPORTANT]
> **Scientific Integrity Notice:**
> - A decrease in training/validation loss during a 5-step smoke test does **NOT** prove that PSD outperforms the baseline.
> - We do **NOT** assert that PSD solves "Lost-in-the-Middle" or eliminates hallucination until full-scale training and statistical evaluation across test sets are completed.
> - The smoke tests verify numerical stability, gradient flow, and memory safety only.

### Known Limitations & Hardware Considerations:
1. **Quantization Precision:** 4-bit NF4 weights are used to operate comfortably within 8 GB VRAM. Float16 compute activations are maintained.
2. **Batch Size:** Set to 1 with gradient accumulation (4 steps) to prevent VRAM spikes during long-context forward-backward passes.
3. **Training Steps:** Smoke tests (5 steps) do not reflect converged performance. Full training steps (e.g. 500–1000 steps) will be needed for final benchmark numbers.
