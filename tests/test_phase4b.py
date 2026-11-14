"""Unit tests for Phase 4B: Long-Context Controlled Training Comparison."""
import os, sys, json, yaml, pytest, torch
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.psd.target_distribution import TargetDistributionBuilder

def test_phase4b_configs_exist_and_isolated():
    configs = [
        ("configs/phase4b_baseline_512.yaml", 512, False),
        ("configs/phase4b_psd_512.yaml", 512, True),
        ("configs/phase4b_baseline_1024.yaml", 1024, False),
        ("configs/phase4b_psd_1024.yaml", 1024, True),
    ]
    for cfg_rel, expected_len, expected_psd in configs:
        cfg_path = os.path.join(PROJECT_ROOT, cfg_rel)
        assert os.path.exists(cfg_path), f"Config missing: {cfg_rel}"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["training"]["max_seq_len"] == expected_len
        assert cfg["psd"]["enabled"] == expected_psd
        if expected_psd:
            assert cfg["psd"]["lambda_psd"] == 0.1
            assert cfg["psd"]["alpha"] == 1.0
            assert cfg["psd"]["beta"] == 0.5
            assert cfg["psd"]["gamma"] == 0.5

def test_phase4b_evaluation_ids_integrity():
    eval_ids_path = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b", "evaluation_ids.json")
    assert os.path.exists(eval_ids_path), "evaluation_ids.json missing"
    with open(eval_ids_path, "r", encoding="utf-8") as f:
        eval_ids = json.load(f)
    assert len(eval_ids) == 15
    ds_counts = {}
    for item in eval_ids:
        ds = item["dataset"]
        ds_counts[ds] = ds_counts.get(ds, 0) + 1
    assert ds_counts.get("totto") == 5
    assert ds_counts.get("dart") == 5
    assert ds_counts.get("webnlg") == 5

def test_phase4b_historical_artifacts_preserved():
    key_artifacts = [
        "reports/experiments/phase2b/psd/metrics.json",
        "reports/experiments/phase3a/phase3a_report.md",
        "reports/experiments/phase3b/phase3b_report.md",
        "reports/experiments/phase4a/phase4a_report.md",
        "reports/experiments/phase4a/context_length_results.csv",
    ]
    for rel_path in key_artifacts:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        assert os.path.exists(full_path), f"Historical artifact missing: {rel_path}"

def test_phase4b_target_distribution_long_context():
    import numpy as np
    builder = TargetDistributionBuilder(alpha=1.0, beta=0.5, gamma=0.5, tau=1.0)
    for seq_len in [512, 1024]:
        token_to_node = [f"n{i%10}" if i % 2 == 0 else None for i in range(seq_len)]
        dist_matrix = np.array([[abs(i - j) for j in range(10)] for i in range(10)], dtype=np.float32)
        node_to_idx = {f"n{i}": i for i in range(10)}
        sem_sim = torch.eye(seq_len, dtype=torch.float32)
        P_target, valid_mask = builder.build_target_distribution(seq_len=seq_len, token_to_node_map=token_to_node, graph_dist_matrix=dist_matrix, node_to_idx=node_to_idx, semantic_sim_matrix=sem_sim, device="cpu")
        assert P_target.shape == (seq_len, seq_len)
        assert valid_mask.shape == (seq_len,)
        assert not torch.isnan(P_target).any()
        assert not torch.isinf(P_target).any()



