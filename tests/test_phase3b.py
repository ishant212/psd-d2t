"""Unit Tests for Phase 3B Targeted 500-Step Confirmation Experiment.

Verifies:
1. Config integrity and isolation for all 3 Phase 3B variants (baseline, position_only, full_psd).
2. Correct component toggles and PSD hyperparameters.
3. Strict preservation of Phase 1, Phase 2A, Phase 2B, and Phase 3A artifacts.
"""

import os
import yaml
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PHASE3B_CONFIGS = [
    ("baseline", "configs/phase3b_baseline.yaml"),
    ("position_only", "configs/phase3b_position_only.yaml"),
    ("full_psd", "configs/phase3b_full_psd.yaml"),
]


def test_phase3b_configs_exist_and_isolated():
    """Verify all 3 Phase 3B configuration files exist and have consistent controlled parameters."""
    for variant_key, config_rel_path in PHASE3B_CONFIGS:
        cfg_path = os.path.join(PROJECT_ROOT, config_rel_path)
        assert os.path.exists(cfg_path), f"Missing config file: {config_rel_path}"

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert cfg["experiment"]["seed"] == 42
        assert cfg["training"]["learning_rate"] == 0.0002
        assert cfg["training"]["batch_size"] == 1
        assert cfg["training"]["gradient_accumulation_steps"] == 4
        assert cfg["training"]["max_seq_len"] == 256
        assert cfg["training"]["num_steps"] == 500
        assert cfg["training"]["eval_interval"] == 50
        assert cfg["training"]["checkpoint_interval"] == 100

        # Verify output directory is strictly isolated under reports/experiments/phase3b/
        exp_dir = cfg["output"]["experiment_dir"]
        assert exp_dir.startswith("reports/experiments/phase3b/"), (
            f"Config {config_rel_path} points to non-phase3b directory: {exp_dir}"
        )


def test_phase3b_component_toggles_and_lambdas():
    """Verify mathematical component flags and lambda values for the 3 confirmation variants."""
    expected_flags = {
        "baseline": {
            "psd_enabled": False,
            "lambda_psd": 0.0,
            "enable_semantic": False,
            "enable_positional": False,
            "enable_graph": False,
        },
        "position_only": {
            "psd_enabled": True,
            "lambda_psd": 0.1,
            "enable_semantic": False,
            "enable_positional": True,
            "enable_graph": False,
        },
        "full_psd": {
            "psd_enabled": True,
            "lambda_psd": 0.1,
            "enable_semantic": True,
            "enable_positional": True,
            "enable_graph": True,
        },
    }

    for variant_key, config_rel_path in PHASE3B_CONFIGS:
        cfg_path = os.path.join(PROJECT_ROOT, config_rel_path)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        psd_cfg = cfg.get("psd", {})
        exp = expected_flags[variant_key]

        assert psd_cfg.get("enabled", False) == exp["psd_enabled"]
        assert psd_cfg.get("lambda_psd", 0.0) == pytest.approx(exp["lambda_psd"])
        assert psd_cfg.get("enable_semantic", False) == exp["enable_semantic"]
        assert psd_cfg.get("enable_positional", False) == exp["enable_positional"]
        assert psd_cfg.get("enable_graph", False) == exp["enable_graph"]


def test_phase2b_and_phase3a_artifacts_preserved():
    """Verify historical artifacts from Phase 2B and Phase 3A remain intact and unmodified."""
    # Phase 2B verification
    p2b_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b")
    assert os.path.exists(p2b_dir), "Phase 2B directory missing!"
    assert os.path.exists(os.path.join(p2b_dir, "evaluation_ids.json")), "Phase 2B evaluation_ids.json missing!"
    assert os.path.exists(os.path.join(p2b_dir, "baseline", "metrics.json")), "Phase 2B baseline metrics missing!"
    assert os.path.exists(os.path.join(p2b_dir, "psd", "metrics.json")), "Phase 2B psd metrics missing!"

    # Phase 3A verification
    p3a_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase3a")
    assert os.path.exists(p3a_dir), "Phase 3A directory missing!"
    assert os.path.exists(os.path.join(p3a_dir, "ablation_summary.csv")), "Phase 3A ablation summary CSV missing!"
    assert os.path.exists(os.path.join(p3a_dir, "ablation_summary.json")), "Phase 3A ablation summary JSON missing!"
    assert os.path.exists(os.path.join(p3a_dir, "phase3a_report.md")), "Phase 3A report missing!"
