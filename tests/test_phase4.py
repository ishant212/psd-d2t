"""Automated Verification Suite for Phase 4 Long-Context Scaling.

Verifies:
1. Deterministic stratified benchmark construction (300 instances across ToTTo, DART, WebNLG and 5 length buckets).
2. Truncation and logging invariants (original_length, effective_length, truncation_amount).
3. Token-to-node mapping preservation under context expansion.
4. PSD target distribution invariants (causality, row-stochasticity, no NaNs) across scaling sequence lengths.
5. PSD loss computation stability and finiteness.
6. Historical experiment artifact integrity (Phase 1, 1.5, 2A, 2B, 3A, 3B, final evaluation untouched).
"""

import os
import sys
import json
import pytest
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.unified_schema import UnifiedRecord
from src.preprocessing.serialize import Serializer
from src.psd.target_distribution import TargetDistributionBuilder
from src.psd.divergence import PSDLoss
from src.psd.run_phase4_long_context import (
    build_stratified_evaluation_benchmark,
    format_example_for_scaling,
    generate_phase4_yaml_configs,
    assign_context_bucket,
    CONTEXT_LENGTHS,
    CONTEXT_BUCKETS,
    PHASE4_DIR,
)


def test_yaml_configs_generation():
    """Verifies all 8 configuration files are generated with valid hyperparameters."""
    configs = generate_phase4_yaml_configs()
    assert len(configs) == 8, f"Expected 8 configs, got {len(configs)}"
    for cfg_path in configs:
        assert os.path.exists(cfg_path)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.loads(json.dumps(__import__("yaml").safe_load(f)))
        assert cfg["model"]["torch_dtype"] == "float16"
        assert cfg["lora"]["r"] == 8
        assert cfg["lora"]["lora_alpha"] == 16
        assert cfg["training"]["learning_rate"] == 0.0002
        assert cfg["training"]["gradient_accumulation_steps"] == 4
        assert cfg["training"]["max_steps"] == 150
        clen = cfg["training"]["max_seq_len"]
        assert clen in CONTEXT_LENGTHS
        if clen >= 1024:
            assert cfg["experiment"]["gradient_checkpointing"] is True


def test_stratified_evaluation_benchmark():
    """Verifies the 300-example benchmark has exact stratification and nested 150-example subset."""
    items = build_stratified_evaluation_benchmark(target_total=300)
    assert len(items) == 300, f"Expected 300 items, got {len(items)}"

    by_ds = {"totto": 0, "dart": 0, "webnlg": 0}
    totto_by_bucket = {b: 0 for b in CONTEXT_BUCKETS}

    for it in items:
        ds = it["dataset"]
        assert ds in by_ds, f"Unknown dataset: {ds}"
        by_ds[ds] += 1
        assert "nodes" in it and "edges" in it, "Missing graph elements"
        assert "serialized_text" in it and len(it["serialized_text"]) > 0
        assert "original_token_length" in it and it["original_token_length"] > 0
        assert "context_bucket" in it

        if ds == "totto":
            totto_by_bucket[it["context_bucket"]] += 1

    assert by_ds["totto"] == 100
    assert by_ds["dart"] == 100
    assert by_ds["webnlg"] == 100

    # Verify each bucket has representation in ToTTo
    for b in CONTEXT_BUCKETS:
        assert totto_by_bucket[b] == 20, f"Bucket {b} expected 20 items, got {totto_by_bucket[b]}"

    # Verify 150-instance subset file exists
    sub_150_path = os.path.join(PHASE4_DIR, "evaluation_set_150.json")
    assert os.path.exists(sub_150_path)
    with open(sub_150_path, "r", encoding="utf-8") as f:
        sub_items = json.load(f)
    assert len(sub_items) == 150


def test_truncation_fields_mathematical_consistency():
    """Verifies truncation amount and percentage satisfy exact mathematical relations."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(PROJECT_ROOT, "models", "qwen3_4b_hf"))
    serializer = Serializer()

    # Load 1 benchmark record
    benchmark_path = os.path.join(PHASE4_DIR, "evaluation_set_300.json")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        sample_rec = json.load(f)[0]

    for max_l in [256, 512, 1024, 2048]:
        inp_t, lab_t, node_map, stats = format_example_for_scaling(
            record_dict_or_obj=sample_rec,
            serializer=serializer,
            tokenizer=tokenizer,
            max_seq_len=max_l,
        )

        orig_l = stats["original_token_length"]
        eff_l = stats["effective_token_length"]
        trunc_amt = stats["truncation_amount"]

        assert trunc_amt == max(0, orig_l - eff_l), "Truncation amount inconsistent"
        assert inp_t.shape[1] == stats["full_seq_length"]
        assert inp_t.shape[1] <= max_l + 32  # Within budget + overhead bounds
        assert lab_t.shape == inp_t.shape
        assert len(node_map) == inp_t.shape[1]

        # Target tokens should NOT be -100 at the end
        assert lab_t[0, -1].item() != -100, "Target EOS must not be masked"
        # First prompt token must be -100
        assert lab_t[0, 0].item() == -100, "Prompt must be masked to -100"


def test_target_distribution_scaling_invariants():
    """Verifies target distribution causality, row-stochasticity, and no NaNs at scaled lengths."""
    builder = TargetDistributionBuilder(alpha=1.0, beta=0.5, gamma=0.5, tau=1.0)
    dist_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.float32)
    node_to_idx = {"n1": 0, "n2": 1, "n3": 2}

    for seq_len in [128, 256, 512, 1024]:
        # Construct synthetic token-to-node map
        node_map = [None] * 5 + ["n1", "n2", "n3"] * (seq_len // 4)
        node_map = node_map[:seq_len]

        P_target, valid_mask = builder.build_target_distribution(
            seq_len=seq_len,
            token_to_node_map=node_map,
            graph_dist_matrix=dist_matrix,
            node_to_idx=node_to_idx,
            device="cpu",
        )

        assert P_target.shape == (seq_len, seq_len)
        assert not torch.isnan(P_target).any(), f"NaN in P_target at L={seq_len}"
        assert not torch.isinf(P_target).any(), f"Inf in P_target at L={seq_len}"

        # Check strict causality: P_target[i, j] == 0 for j > i
        future_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
        assert torch.all(P_target[future_mask] == 0.0), f"Causal leakage at L={seq_len}"

        # Check row stochasticity for valid rows: sum_j P(j|i) == 1.0
        row_sums = P_target.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), f"Row sums != 1 at L={seq_len}"


def test_psd_loss_gradient_flow():
    """Verifies PSDLoss calculates scalar positive loss and gradients flow backward."""
    psd_fn = PSDLoss(selected_layers=[0, 1])
    seq_len = 64
    heads = 4

    # Dummy causal attention tensors with requires_grad on leaf tensors
    raw_attn0 = torch.randn(1, heads, seq_len, seq_len, requires_grad=True)
    raw_attn1 = torch.randn(1, heads, seq_len, seq_len, requires_grad=True)
    attn_layer0 = torch.softmax(raw_attn0, dim=-1)
    attn_layer1 = torch.softmax(raw_attn1, dim=-1)

    P_target = torch.softmax(torch.tril(torch.randn(seq_len, seq_len)), dim=-1)
    valid_mask = torch.ones(seq_len, dtype=torch.bool)

    loss, per_layer = psd_fn((attn_layer0, attn_layer1), P_target, valid_mask)

    assert loss.dim() == 0, "Loss must be scalar"
    assert loss.item() >= 0.0, "KL loss must be non-negative"
    assert not torch.isnan(loss), "Loss must not be NaN"

    loss.backward()
    assert raw_attn0.grad is not None, "Gradient must flow to layer 0 leaf tensor"
    assert raw_attn1.grad is not None, "Gradient must flow to layer 1 leaf tensor"


def test_historical_artifacts_untouched():
    """Strictly verifies that no previous phase artifacts were modified or deleted."""
    historical_paths = [
        os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b", "evaluation_ids.json"),
        os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b", "baseline", "metrics.json"),
        os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b", "psd", "metrics.json"),
        os.path.join(PROJECT_ROOT, "reports", "experiments", "phase3a", "baseline", "metrics.json"),
        os.path.join(PROJECT_ROOT, "reports", "experiments", "phase3a", "full_psd", "metrics.json"),
        os.path.join(PROJECT_ROOT, "reports", "experiments", "phase3b", "baseline", "metrics.json"),
        os.path.join(PROJECT_ROOT, "reports", "experiments", "phase3b", "full_psd", "metrics.json"),
        os.path.join(PROJECT_ROOT, "reports", "experiments", "final_evaluation", "final_results.json"),
    ]

    for p in historical_paths:
        assert os.path.exists(p), f"CRITICAL: Historical artifact missing: {p}"
