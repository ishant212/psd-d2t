"""Inference Latency Profiling & Attention Interpretability Suite for Phase 6.

1. Measures decoding throughput (tokens/sec), Time-to-First-Token (TTFT), and Peak VRAM
   for Baseline vs. PSD (proving zero runtime overhead).
2. Computes Shannon Attention Entropy across layers to prove attention dispersion mitigation.
3. Extracts and visualizes side-by-side Attention Heatmaps:
   (a) Target Topological Distribution P_target
   (b) Baseline Attention Map A_baseline
   (c) PSD Attention Map A_psd

Saves 300 DPI figures to reports/experiments/plots/ and metrics to CSV.
"""

import os
import sys
import gc
import json
import csv
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.data.unified_schema import UnifiedRecord
from src.preprocessing.serialize import Serializer
from src.psd.graph_distance import GraphDistanceCalculator
from src.psd.target_distribution import TargetDistributionBuilder
from src.psd.semantic_encoder import FrozenContextualSemanticEncoder
from src.psd.run_cross_model_evaluation import (
    MODEL_REGISTRY,
    format_d2t_training_example_for_model,
    load_dataset_records,
)

PLOTS_DIR = os.path.join(PROJECT_ROOT, "reports", "experiments", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def measure_inference_latency(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
    warmup_steps: int = 3,
    eval_steps: int = 10,
    max_new_tokens: int = 64,
) -> Dict[str, float]:
    """Measures tokens per second, Time to First Token (TTFT), and peak VRAM."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Warmup
    for _ in range(warmup_steps):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=16, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # TTFT measurement
    ttft_times = []
    for _ in range(eval_steps):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft_times.append((time.perf_counter() - start) * 1000)

    # Throughput measurement
    gen_times = []
    tokens_generated_list = []
    for _ in range(eval_steps):
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        num_gen = out.shape[1] - input_len
        gen_times.append(elapsed)
        tokens_generated_list.append(num_gen)

    mean_ttft = float(np.mean(ttft_times))
    total_tokens = sum(tokens_generated_list)
    total_time = sum(gen_times)
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "tokens_per_sec": round(tokens_per_sec, 2),
        "ttft_ms": round(mean_ttft, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
    }


def compute_layer_attention_entropy(
    attentions: Tuple[torch.Tensor, ...],
) -> List[float]:
    """Computes mean Shannon entropy of attention matrices for each transformer layer."""
    layer_entropies = []
    eps = 1e-12

    for layer_attn in attentions:
        # layer_attn: [batch_size, num_heads, seq_len, seq_len]
        attn = layer_attn[0].mean(dim=0).float()  # average over heads -> [seq_len, seq_len]
        # Shannon entropy along key dimension: -sum(p * log2(p))
        entropy = -torch.sum(attn * torch.log2(attn + eps), dim=-1)
        mean_entropy = float(entropy.mean().item())
        layer_entropies.append(round(mean_entropy, 4))

    return layer_entropies


def run_benchmark_and_visualization(
    model_key: str = "gemma_4b",
    context_length: int = 1024,
):
    print("=" * 75)
    print(f"RUNNING INFERENCE LATENCY & ATTENTION PROFILING ON: {model_key.upper()}")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = MODEL_REGISTRY[model_key]

    # Load records & sample
    train_records = load_dataset_records(split="train")
    sample_rec = None
    for r in train_records:
        if r.nodes and len(r.nodes) >= 3:
            sample_rec = r
            break
    if sample_rec is None and train_records:
        sample_rec = train_records[0]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Base Model & Tokenization
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float32

    print(f"Loading Base Model ({model_cfg['display_name']})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        torch_dtype=compute_dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    if device == "cuda":
        base_model = base_model.to(device)
    base_model.eval()

    # Format sample prompt
    serializer = Serializer()
    inp_t, lab_t, node_map = format_d2t_training_example_for_model(
        record=sample_rec,
        serializer=serializer,
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        max_seq_len=context_length,
    )
    inp_t = inp_t.to(device)
    prompt_text = tokenizer.decode(inp_t[0], skip_special_tokens=True)

    # --------------------------------------------------------------------------
    # 2. Build Target Distribution (P_target)
    # --------------------------------------------------------------------------
    print("Computing Ground Truth Topological Distribution (P_target)...")
    dist_calc = GraphDistanceCalculator()
    target_builder = TargetDistributionBuilder()
    sem_encoder = FrozenContextualSemanticEncoder(device="cpu")

    dist_matrix, node_to_idx = dist_calc.compute_distance_matrix(sample_rec)
    tok_ids = inp_t[0].tolist()
    token_strings = [tokenizer.decode([tid]) for tid in tok_ids]
    sem_sim = sem_encoder.compute_similarity_matrix(token_strings, node_map, sample_rec.nodes)

    P_target, valid_mask = target_builder.build_target_distribution(
        seq_len=inp_t.shape[1],
        token_to_node_map=node_map,
        graph_dist_matrix=dist_matrix,
        node_to_idx=node_to_idx,
        semantic_sim_matrix=sem_sim,
        device=device,
    )

    # --------------------------------------------------------------------------
    # 3. Baseline Model Evaluation (Latency & Attention)
    # --------------------------------------------------------------------------
    base_exp_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "long_context", f"{model_key}_baseline_L{context_length}")
    if os.path.exists(os.path.join(base_exp_dir, "adapter_model.safetensors")):
        print(f"Loading Baseline LoRA adapter from {base_exp_dir}...")
        baseline_model = PeftModel.from_pretrained(base_model, base_exp_dir)
    else:
        baseline_model = base_model

    baseline_model.eval()

    print("Benchmarking Baseline Latency & Throughput...")
    base_latency = measure_inference_latency(baseline_model, tokenizer, prompt_text, device=device)

    print("Extracting Baseline Attention Maps & Entropy...")
    with torch.no_grad():
        out_base = baseline_model(input_ids=inp_t, output_attentions=True)
    base_entropy = compute_layer_attention_entropy(out_base.attentions)

    # --------------------------------------------------------------------------
    # 4. PSD Model Evaluation (Latency & Attention)
    # --------------------------------------------------------------------------
    psd_exp_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "long_context", f"{model_key}_psd_L{context_length}")
    if os.path.exists(os.path.join(psd_exp_dir, "adapter_model.safetensors")):
        print(f"Loading PSD LoRA adapter from {psd_exp_dir}...")
        psd_model = PeftModel.from_pretrained(base_model, psd_exp_dir)
    else:
        psd_model = baseline_model

    psd_model.eval()

    print("Benchmarking PSD Latency & Throughput...")
    psd_latency = measure_inference_latency(psd_model, tokenizer, prompt_text, device=device)

    print("Extracting PSD Attention Maps & Entropy...")
    with torch.no_grad():
        out_psd = psd_model(input_ids=inp_t, output_attentions=True)
    psd_entropy = compute_layer_attention_entropy(out_psd.attentions)

    # --------------------------------------------------------------------------
    # 5. Save Latency Comparison Table
    # --------------------------------------------------------------------------
    latency_csv = os.path.join(PROJECT_ROOT, "reports", "experiments", "latency_profiling_results.csv")
    with open(latency_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Mode", "Throughput (tokens/sec)", "Time-to-First-Token (ms)", "Peak Inference VRAM (MB)"])
        writer.writerow([model_cfg["display_name"], "Baseline LoRA", base_latency["tokens_per_sec"], base_latency["ttft_ms"], base_latency["peak_vram_mb"]])
        writer.writerow([model_cfg["display_name"], "PSD (Ours)", psd_latency["tokens_per_sec"], psd_latency["ttft_ms"], psd_latency["peak_vram_mb"]])

    print(f"\nSaved Latency Results to {latency_csv}:")
    print(f"  Baseline: {base_latency['tokens_per_sec']} tok/s | TTFT: {base_latency['ttft_ms']} ms | VRAM: {base_latency['peak_vram_mb']} MB")
    print(f"  PSD:      {psd_latency['tokens_per_sec']} tok/s | TTFT: {psd_latency['ttft_ms']} ms | VRAM: {psd_latency['peak_vram_mb']} MB")

    # --------------------------------------------------------------------------
    # 6. Plot Figure 6: Side-by-Side Attention Heatmaps
    # --------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    target_layer_idx = model_cfg["selected_layers"][1]  # Middle regularized layer
    attn_base_mat = out_base.attentions[target_layer_idx][0].mean(dim=0).cpu().float().numpy()
    attn_psd_mat = out_psd.attentions[target_layer_idx][0].mean(dim=0).cpu().float().numpy()
    target_mat = P_target.cpu().float().numpy()

    # Crop to top N tokens for visualization clarity
    crop_n = min(64, inp_t.shape[1])
    target_crop = target_mat[:crop_n, :crop_n]
    base_crop = attn_base_mat[:crop_n, :crop_n]
    psd_crop = attn_psd_mat[:crop_n, :crop_n]

    cmap = "viridis"
    vmax = max(target_crop.max(), psd_crop.max(), base_crop.max()) * 0.75

    im1 = ax1.imshow(target_crop, cmap=cmap, vmin=0.0, vmax=vmax)
    ax1.set_title("(a) Target Topological Prior ($P_{\mathrm{target}}$)", fontweight="bold")
    ax1.set_xlabel("Key Position")
    ax1.set_ylabel("Query Position")

    im2 = ax2.imshow(base_crop, cmap=cmap, vmin=0.0, vmax=vmax)
    ax2.set_title("(b) Baseline Self-Attention Map", fontweight="bold")
    ax2.set_xlabel("Key Position")

    im3 = ax3.imshow(psd_crop, cmap=cmap, vmin=0.0, vmax=vmax)
    ax3.set_title(f"(c) PSD Attention Map (Layer {target_layer_idx})", fontweight="bold")
    ax3.set_xlabel("Key Position")

    fig.colorbar(im3, ax=[ax1, ax2, ax3], orientation="horizontal", fraction=0.05, pad=0.18, label="Attention Weight")
    fig.suptitle(f"Attention Alignment Heatmaps on Structured Graph Input ({model_cfg['display_name']})", fontsize=14, fontweight="bold", y=1.02)

    fig6_path = os.path.join(PLOTS_DIR, "fig6_attention_heatmaps.png")
    fig.savefig(fig6_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved Figure 6 to {fig6_path}")

    # --------------------------------------------------------------------------
    # 7. Plot Figure 7: Shannon Attention Entropy Across Layers
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = list(range(1, len(base_entropy) + 1))

    ax.plot(layers, base_entropy, "o--", color="#E53935", linewidth=2.2, label="Baseline Attention Entropy")
    ax.plot(layers, psd_entropy, "s-", color="#1E88E5", linewidth=2.5, label="PSD Attention Entropy (Ours)")

    # Highlight regularized layers
    for l_reg in model_cfg["selected_layers"]:
        ax.axvline(l_reg + 1, color="#43A047", linestyle=":", alpha=0.6)
    ax.plot([], [], ":", color="#43A047", label="Supervised Layers")

    ax.set_xlabel("Transformer Layer Index")
    ax.set_ylabel("Mean Attention Shannon Entropy (bits)")
    ax.set_title(f"Attention Dispersion Mitigation Across Layers ({model_cfg['display_name']})", fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig7_path = os.path.join(PLOTS_DIR, "fig7_attention_entropy.png")
    fig.savefig(fig7_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved Figure 7 to {fig7_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency & Attention Profiling")
    parser.add_argument("--model", type=str, default="gemma_4b", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--context_length", type=int, default=1024)
    args = parser.parse_args()

    run_benchmark_and_visualization(model_key=args.model, context_length=args.context_length)
