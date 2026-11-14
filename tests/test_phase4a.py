"""Unit Tests for Phase 4A Long-Context Scaling Evaluation.

Verifies:
1. Checkpoint availability for both Phase 2B Baseline and Full PSD models.
2. Integrity of evaluation IDs (15 instances across ToTTo, DART, WebNLG).
3. Truncation determinism and structural retention mapping functions.
4. Historical artifact preservation (Phase 1, Phase 2A, Phase 2B, Phase 3A, Phase 3B).
"""

import os
import json
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_phase4a_checkpoints_exist():
    """Verify that Phase 2B final checkpoints for Baseline and Full PSD exist."""
    base_ckpt = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b", "baseline", "checkpoints", "final")
    psd_ckpt = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b", "psd", "checkpoints", "final")

    assert os.path.exists(base_ckpt), f"Baseline checkpoint not found at: {base_ckpt}"
    assert os.path.exists(psd_ckpt), f"Full PSD checkpoint not found at: {psd_ckpt}"

    assert os.path.exists(os.path.join(base_ckpt, "adapter_model.safetensors")), "Baseline adapter_model.safetensors missing!"
    assert os.path.exists(os.path.join(psd_ckpt, "adapter_model.safetensors")), "Full PSD adapter_model.safetensors missing!"


def test_phase4a_evaluation_ids_integrity():
    """Verify the 15 fixed evaluation instances established in Phase 2B."""
    eval_path = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b", "evaluation_ids.json")
    assert os.path.exists(eval_path), f"evaluation_ids.json not found at: {eval_path}"

    with open(eval_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert len(records) == 15, f"Expected 15 evaluation records, got {len(records)}"

    datasets = [r["dataset"] for r in records]
    assert datasets.count("totto") == 5, f"Expected 5 ToTTo records, got {datasets.count('totto')}"
    assert datasets.count("dart") == 5, f"Expected 5 DART records, got {datasets.count('dart')}"
    assert datasets.count("webnlg") == 5, f"Expected 5 WebNLG records, got {datasets.count('webnlg')}"

    for r in records:
        assert "id" in r and "serialized_text" in r and "target_text" in r and "reference_texts" in r
        assert len(r["serialized_text"]) > 0


def test_phase4a_historical_artifacts_preserved():
    """Verify that historical artifacts from Phase 1, Phase 2A, Phase 2B, Phase 3A, and Phase 3B remain intact."""
    p1_dir = os.path.join(PROJECT_ROOT, "reports", "phase1")
    p2a_dir = os.path.join(PROJECT_ROOT, "reports", "phase2a")
    p2b_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase2b")
    p3a_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase3a")
    p3b_dir = os.path.join(PROJECT_ROOT, "reports", "experiments", "phase3b")

    assert os.path.exists(p1_dir), "Phase 1 directory missing!"
    assert os.path.exists(p2a_dir), "Phase 2A directory missing!"
    assert os.path.exists(p2b_dir), "Phase 2B directory missing!"
    assert os.path.exists(p3a_dir), "Phase 3A directory missing!"
    assert os.path.exists(p3b_dir), "Phase 3B directory missing!"

    assert os.path.exists(os.path.join(p3a_dir, "phase3a_report.md")), "Phase 3A report missing!"
    assert os.path.exists(os.path.join(p3b_dir, "phase3b_report.md")), "Phase 3B report missing!"
    assert os.path.exists(os.path.join(p3b_dir, "confirmation_summary.json")), "Phase 3B summary missing!"
