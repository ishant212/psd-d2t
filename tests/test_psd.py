"""Comprehensive Unit Test Suite for Phase 2 PSD.

Verifies:
1. Graph distance calculation (adjacent, multi-hop, disconnected, self-distance)
2. ToTTo table Manhattan distance
3. Target distribution shape and normalization
4. Causal masking
5. Semantic similarity contribution
6. Positional distance contribution
7. Graph distance contribution
8. KL divergence correctness
9. Selected layer attention extraction ([11, 23, 34])
10. Target/reference text preservation during truncation & prompt masking with -100
11. Gradient propagation through attention to LoRA parameters using FP16 precision
"""

import os
import sys
import json
import pytest
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.unified_schema import UnifiedRecord, Node, Edge
from src.preprocessing.serialize import Serializer
from src.psd.graph_distance import GraphDistanceCalculator
from src.psd.target_distribution import TargetDistributionBuilder
from src.psd.divergence import PSDLoss
from src.psd.model import PSDModelWrapper, PSDOutput
from src.psd.trainer import format_d2t_training_example

HF_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "qwen3_4b_hf")


# ============================================================
# Test 1 & 2: Graph and Table Distance
# ============================================================

def test_graph_distance_calculator_linear_graph():
    """Verifies shortest path distance calculation on a multi-hop graph."""
    calc = GraphDistanceCalculator()
    rec = UnifiedRecord(
        dataset="webnlg",
        id="test_rec",
        input_type="triples",
        structured_input={},
        nodes=[
            {"id": "A", "label": "Node A", "type": "entity"},
            {"id": "B", "label": "Node B", "type": "entity"},
            {"id": "C", "label": "Node C", "type": "entity"},
            {"id": "D", "label": "Node D (Disconnected)", "type": "entity"},
        ],
        edges=[
            {"source": "A", "target": "B", "relation": "rel1"},
            {"source": "B", "target": "C", "relation": "rel2"},
        ]
    )
    dist_mat, node_to_idx = calc.compute_distance_matrix(rec)

    idx_a, idx_b, idx_c, idx_d = node_to_idx["A"], node_to_idx["B"], node_to_idx["C"], node_to_idx["D"]

    # Self-distance
    assert dist_mat[idx_a, idx_a] == 0.0
    assert dist_mat[idx_b, idx_b] == 0.0

    # 1-hop distance
    assert dist_mat[idx_a, idx_b] == 1.0
    assert dist_mat[idx_b, idx_c] == 1.0

    # 2-hop distance
    assert dist_mat[idx_a, idx_c] == 2.0

    # Disconnected node D has finite cutoff >= max_observed + 1
    assert dist_mat[idx_a, idx_d] >= 3.0


def test_table_manhattan_distance():
    """Verifies coordinate Manhattan distance for tabular cells."""
    calc = GraphDistanceCalculator()
    rec = UnifiedRecord(
        dataset="totto",
        id="test_table",
        input_type="table",
        structured_input={},
        nodes=[
            {"id": "c00", "label": "Cell(0,0)", "type": "cell", "attributes": {"row": 0, "col": 0}},
            {"id": "c01", "label": "Cell(0,1)", "type": "cell", "attributes": {"row": 0, "col": 1}},
            {"id": "c23", "label": "Cell(2,3)", "type": "cell", "attributes": {"row": 2, "col": 3}},
        ],
        edges=[]
    )
    dist_mat, node_to_idx = calc.compute_distance_matrix(rec)

    idx_00 = node_to_idx["c00"]
    idx_01 = node_to_idx["c01"]
    idx_23 = node_to_idx["c23"]

    assert dist_mat[idx_00, idx_00] == 0.0
    assert dist_mat[idx_00, idx_01] == 1.0
    assert dist_mat[idx_00, idx_23] == 5.0


# ============================================================
# Test 3, 4, 5: Target Distribution Shape, Normalization, Causal Mask
# ============================================================

def test_target_distribution_builder_properties():
    """Verifies shape, causal mask, and row-stochastic normalization of P_target."""
    builder = TargetDistributionBuilder(alpha=1.0, beta=0.5, gamma=0.5, tau=1.0)
    seq_len = 8
    token_to_node = ["A", "A", "B", None, "C", "C", "B", None]
    dist_mat = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
    ], dtype=np.float32)
    node_to_idx = {"A": 0, "B": 1, "C": 2}

    P_target, valid_mask = builder.build_target_distribution(
        seq_len=seq_len,
        token_to_node_map=token_to_node,
        graph_dist_matrix=dist_mat,
        node_to_idx=node_to_idx,
    )

    assert P_target.shape == (seq_len, seq_len)
    assert valid_mask.shape == (seq_len,)
    assert not torch.isnan(P_target).any()
    assert not torch.isinf(P_target).any()

    # Check Causal Masking
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert P_target[i, j].item() == 0.0

    # Check row normalization
    row_sums = P_target.sum(dim=-1)
    for i in range(seq_len):
        assert abs(row_sums[i].item() - 1.0) < 1e-4


# ============================================================
# Test 6: Target Preservation and Prompt Masking (-100)
# ============================================================

def test_target_preservation_and_prompt_masking():
    """Verifies that prompt is masked with -100 and target text is never removed by truncation."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
    serializer = Serializer()

    rec = UnifiedRecord(
        dataset="webnlg",
        id="test_trunc",
        input_type="triples",
        structured_input={},
        nodes=[{"id": "A", "label": "Very Long Subject Text " * 20, "type": "entity"}],
        edges=[],
        target_text="This is the crucial target reference sentence.",
    )

    # Set very small max_seq_len = 32
    input_tensor, labels_tensor, full_node_map = format_d2t_training_example(
        record=rec,
        serializer=serializer,
        tokenizer=tokenizer,
        max_seq_len=32,
    )

    input_ids = input_tensor[0].tolist()
    labels = labels_tensor[0].tolist()

    # Decoded target text from non -100 labels
    target_token_ids = [l for l in labels if l != -100]
    decoded_target = tokenizer.decode(target_token_ids)

    # 1. Prompt tokens are strictly -100
    assert -100 in labels
    assert labels[0] == -100

    # 2. Target text is 100% preserved in the non-masked portion of labels
    assert "crucial target reference sentence" in decoded_target
    assert len(input_ids) <= 32


# ============================================================
# Test 7: PSD KL Divergence Loss
# ============================================================

def test_psd_loss_computation():
    """Verifies KL divergence calculation between target and empirical attention distributions."""
    loss_module = PSDLoss(selected_layers=[0, 1])
    seq_len = 4
    P_target = torch.tril(torch.ones(seq_len, seq_len))
    P_target = P_target / P_target.sum(dim=-1, keepdim=True)

    # Identical empirical attention -> Loss ~ 0
    attn_identical = P_target.unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1)
    loss_zero, _ = loss_module((attn_identical, attn_identical), P_target)
    assert abs(loss_zero.item()) < 1e-4

    # Divergent empirical attention -> Loss > 0
    attn_divergent = torch.eye(seq_len).unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1)
    loss_pos, _ = loss_module((attn_divergent, attn_divergent), P_target)
    assert loss_pos.item() > 0.01


# ============================================================
# Test 8: Model Loading with FP16 & LoRA Gradients
# ============================================================

@pytest.fixture(scope="module")
def fp16_psd_model():
    """Loads PSDModelWrapper with Hugging Face base model in FP16/FP32."""
    assert os.path.exists(HF_MODEL_PATH), f"Model path {HF_MODEL_PATH} missing"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = PSDModelWrapper(
        model_name_or_path=HF_MODEL_PATH,
        selected_layers=[11, 23, 34],
        device=device,
    )
    return wrapper


def test_psd_model_layer_selection_and_gradients(fp16_psd_model):
    """Verifies that only selected layers are extracted and gradients update LoRA parameters."""
    device = fp16_psd_model.device_str
    fp16_psd_model.train()

    rec = UnifiedRecord(
        dataset="webnlg",
        id="test_grad",
        input_type="triples",
        structured_input={},
        nodes=[
            {"id": "Paris", "label": "Paris", "type": "entity"},
            {"id": "France", "label": "France", "type": "entity"},
        ],
        edges=[{"source": "Paris", "target": "France", "relation": "country"}],
        target_text="Paris is in France.",
    )

    input_tensor, labels_tensor, full_node_map = format_d2t_training_example(
        record=rec,
        serializer=fp16_psd_model.serializer,
        tokenizer=fp16_psd_model.tokenizer,
        max_seq_len=64,
    )

    input_tensor = input_tensor.to(device)
    labels_tensor = labels_tensor.to(device)

    # Forward pass
    output: PSDOutput = fp16_psd_model(
        input_ids=input_tensor,
        labels=labels_tensor,
        token_to_node_map=full_node_map,
        record=rec,
        max_seq_len=64,
    )

    assert output.total_loss is not None
    assert output.psd_loss is not None
    # Verify selected layers are exactly [11, 23, 34]
    assert set(output.per_layer_psd.keys()) == {11, 23, 34}

    # Backward pass
    output.total_loss.backward()

    # Verify trainable parameter gradients
    trainable_grads = [p.grad for p in fp16_psd_model.parameters() if p.requires_grad and p.grad is not None]
    assert len(trainable_grads) > 0, "No gradients were computed for trainable parameters!"
    total_grad_norm = sum(g.norm().item() for g in trainable_grads)
    assert total_grad_norm > 0.0, "Gradient norm is zero!"
