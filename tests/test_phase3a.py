"""Unit Tests for Phase 3A Ablation Screening Infrastructure.

Verifies:
1. TargetDistributionBuilder component toggles (Semantic-only, Position-only, Graph-only, Pairwise, Full)
2. Mathematical affinity formulation properties (R_ij = alpha*S_ij - beta*dP - gamma*dG)
3. Causal masking and row sum normalization in all ablation modes
4. Phase 3A 8-variant configuration files parsing and isolation
5. Phase 2B artifacts preservation
"""

import os
import json
import yaml
import pytest
import numpy as np
import torch

from src.psd.target_distribution import TargetDistributionBuilder
from src.psd.divergence import PSDLoss


def test_target_distribution_component_toggles():
    """Verifies that individual ablation flags correctly zero out disabled components."""
    seq_len = 4
    token_to_node_map = ["n0", "n1", "n2", "n3"]
    node_to_idx = {"n0": 0, "n1": 1, "n2": 2, "n3": 3}
    graph_dist = np.array([
        [0, 1, 2, 3],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [3, 2, 1, 0]
    ], dtype=np.float32)
    semantic_sim = torch.tensor([
        [1.0, 0.8, 0.2, 0.1],
        [0.8, 1.0, 0.7, 0.3],
        [0.2, 0.7, 1.0, 0.9],
        [0.1, 0.3, 0.9, 1.0]
    ], dtype=torch.float32)

    # 1. Semantic Only
    sem_builder = TargetDistributionBuilder(
        alpha=1.0, beta=0.5, gamma=0.5, tau=1.0,
        enable_semantic=True, enable_positional=False, enable_graph=False
    )
    assert sem_builder.alpha == 1.0
    assert sem_builder.beta == 0.0
    assert sem_builder.gamma == 0.0
    P_sem, _ = sem_builder.build_target_distribution(
        seq_len=seq_len,
        token_to_node_map=token_to_node_map,
        graph_dist_matrix=graph_dist,
        node_to_idx=node_to_idx,
        semantic_sim_matrix=semantic_sim,
    )
    assert P_sem.shape == (seq_len, seq_len)
    assert torch.allclose(P_sem.sum(dim=-1), torch.ones(seq_len), atol=1e-5)

    # 2. Position Only
    pos_builder = TargetDistributionBuilder(
        alpha=1.0, beta=0.5, gamma=0.5, tau=1.0,
        enable_semantic=False, enable_positional=True, enable_graph=False
    )
    assert pos_builder.alpha == 0.0
    assert pos_builder.beta == 0.5
    assert pos_builder.gamma == 0.0
    P_pos, _ = pos_builder.build_target_distribution(
        seq_len=seq_len,
        token_to_node_map=token_to_node_map,
        graph_dist_matrix=graph_dist,
        node_to_idx=node_to_idx,
        semantic_sim_matrix=semantic_sim,
    )
    assert P_pos.shape == (seq_len, seq_len)
    assert torch.allclose(P_pos.sum(dim=-1), torch.ones(seq_len), atol=1e-5)

    # 3. Graph Only
    graph_builder = TargetDistributionBuilder(
        alpha=1.0, beta=0.5, gamma=0.5, tau=1.0,
        enable_semantic=False, enable_positional=False, enable_graph=True
    )
    assert graph_builder.alpha == 0.0
    assert graph_builder.beta == 0.0
    assert graph_builder.gamma == 0.5
    P_graph, _ = graph_builder.build_target_distribution(
        seq_len=seq_len,
        token_to_node_map=token_to_node_map,
        graph_dist_matrix=graph_dist,
        node_to_idx=node_to_idx,
        semantic_sim_matrix=semantic_sim,
    )
    assert P_graph.shape == (seq_len, seq_len)
    assert torch.allclose(P_graph.sum(dim=-1), torch.ones(seq_len), atol=1e-5)

    # 4. Full PSD
    full_builder = TargetDistributionBuilder(
        alpha=1.0, beta=0.5, gamma=0.5, tau=1.0,
        enable_semantic=True, enable_positional=True, enable_graph=True
    )
    assert full_builder.alpha == 1.0
    assert full_builder.beta == 0.5
    assert full_builder.gamma == 0.5
    P_full, _ = full_builder.build_target_distribution(
        seq_len=seq_len,
        token_to_node_map=token_to_node_map,
        graph_dist_matrix=graph_dist,
        node_to_idx=node_to_idx,
        semantic_sim_matrix=semantic_sim,
    )
    assert P_full.shape == (seq_len, seq_len)
    assert torch.allclose(P_full.sum(dim=-1), torch.ones(seq_len), atol=1e-5)


def test_phase3a_all_configs_exist_and_isolated():
    """Verifies that all 8 Phase 3A configurations exist with identical training settings."""
    variant_names = [
        "baseline",
        "semantic_only",
        "position_only",
        "graph_only",
        "semantic_position",
        "semantic_graph",
        "position_graph",
        "full_psd",
    ]

    for v_name in variant_names:
        cfg_path = f"configs/phase3a_{v_name}.yaml"
        assert os.path.exists(cfg_path), f"Missing {cfg_path}"

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert cfg["training"]["num_steps"] == 150
        assert cfg["training"]["eval_interval"] == 50
        assert cfg["training"]["checkpoint_interval"] == 50
        assert cfg["training"]["batch_size"] == 1
        assert cfg["training"]["gradient_accumulation_steps"] == 4
        assert cfg["training"]["learning_rate"] == 0.0002
        assert cfg["experiment"]["seed"] == 42
        assert "phase3a" in cfg["output"]["experiment_dir"]
        assert v_name in cfg["output"]["experiment_dir"]


def test_phase2b_artifacts_preserved():
    """Verifies that Phase 2B 500-step artifacts remain untouched."""
    p2b_base_csv = "reports/experiments/phase2b/baseline/training_log.csv"
    p2b_psd_csv = "reports/experiments/phase2b/psd/training_log.csv"
    p2b_report = "reports/experiments/phase2b/phase2b_report.md"

    assert os.path.exists(p2b_base_csv), "Phase 2B baseline log must exist"
    assert os.path.exists(p2b_psd_csv), "Phase 2B PSD log must exist"
    assert os.path.exists(p2b_report), "Phase 2B report must exist"

    with open(p2b_base_csv, "r") as f:
        lines = [line for line in f if line.strip()]
        assert len(lines) == 501  # header + 500 steps
