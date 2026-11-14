"""Unit tests for Phase 2A: Controlled Experiments Infrastructure.

Verifies:
1. Baseline produces LM-only loss.
2. PSD produces total_loss = lm_loss + lambda_psd * psd_loss.
3. Both experiments load identical records with same seed.
4. Target/reference masking remains correct (-100).
5. Target text is never truncated.
6. LoRA parameters receive gradients.
7. Base parameters remain frozen.
8. Checkpoints can be saved and reloaded.
9. Position-controlled examples preserve their target.
10. Metrics are computed consistently across generation outputs.
"""

import os
import sys
import tempfile
import json
import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.unified_schema import UnifiedRecord
from src.preprocessing.serialize import Serializer
from src.psd.model import PSDModelWrapper, PSDOutput
from src.psd.trainer import ExperimentTrainer, format_d2t_training_example, load_dataset_records, set_seed
from src.psd.position_control import create_positional_variants
from src.psd.evaluate import compute_structured_coverage
from src.baseline.evaluate_metrics import MetricEvaluator

HF_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "qwen3_4b_hf")


@pytest.fixture(scope="module")
def shared_model_wrapper():
    """Shared PSDModelWrapper on available device for unit testing."""
    assert os.path.exists(HF_MODEL_PATH), f"Model path {HF_MODEL_PATH} missing"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = PSDModelWrapper(
        model_name_or_path=HF_MODEL_PATH,
        selected_layers=[11, 23, 34],
        device=device,
    )
    return wrapper


def test_baseline_lm_only_loss(shared_model_wrapper):
    """Verifies that baseline mode computes only LM loss and PSD loss is 0."""
    rec = UnifiedRecord(
        dataset="webnlg",
        id="test_baseline_loss",
        input_type="triples",
        structured_input={},
        nodes=[{"id": "Rome", "label": "Rome", "type": "entity"}, {"id": "Italy", "label": "Italy", "type": "entity"}],
        edges=[{"source": "Rome", "target": "Italy", "relation": "country"}],
        target_text="Rome is in Italy.",
    )

    inp_t, lab_t, node_map = format_d2t_training_example(
        record=rec,
        serializer=shared_model_wrapper.serializer,
        tokenizer=shared_model_wrapper.tokenizer,
        max_seq_len=64,
    )

    device = shared_model_wrapper.device_str
    inp_t = inp_t.to(device)
    lab_t = lab_t.to(device)

    # Disable PSD
    shared_model_wrapper.psd_enabled = False
    shared_model_wrapper.lambda_psd = 0.0

    output = shared_model_wrapper(
        input_ids=inp_t,
        labels=lab_t,
        token_to_node_map=node_map,
        record=rec,
        max_seq_len=64,
    )

    assert output.lm_loss.item() > 0.0
    assert output.psd_loss.item() == 0.0
    assert abs(output.total_loss.item() - output.lm_loss.item()) < 1e-5


def test_psd_composite_loss(shared_model_wrapper):
    """Verifies that PSD mode computes L_total = L_LM + lambda_PSD * L_PSD."""
    rec = UnifiedRecord(
        dataset="webnlg",
        id="test_psd_comp_loss",
        input_type="triples",
        structured_input={},
        nodes=[{"id": "Tokyo", "label": "Tokyo", "type": "entity"}, {"id": "Japan", "label": "Japan", "type": "entity"}],
        edges=[{"source": "Tokyo", "target": "Japan", "relation": "country"}],
        target_text="Tokyo is the capital of Japan.",
    )

    inp_t, lab_t, node_map = format_d2t_training_example(
        record=rec,
        serializer=shared_model_wrapper.serializer,
        tokenizer=shared_model_wrapper.tokenizer,
        max_seq_len=64,
    )

    device = shared_model_wrapper.device_str
    inp_t = inp_t.to(device)
    lab_t = lab_t.to(device)

    shared_model_wrapper.psd_enabled = True
    shared_model_wrapper.lambda_psd = 0.1

    output = shared_model_wrapper(
        input_ids=inp_t,
        labels=lab_t,
        token_to_node_map=node_map,
        record=rec,
        max_seq_len=64,
    )

    assert output.lm_loss.item() > 0.0
    assert output.psd_loss.item() > 0.0
    expected_total = output.lm_loss.item() + 0.1 * output.psd_loss.item()
    assert abs(output.total_loss.item() - expected_total) < 1e-4


def test_data_sampling_consistency():
    """Verifies that seed control ensures identical record sequence across runs."""
    set_seed(42)
    recs_1 = load_dataset_records(dataset_name="dart", split="train", max_records=10)
    set_seed(42)
    recs_2 = load_dataset_records(dataset_name="dart", split="train", max_records=10)

    assert len(recs_1) == len(recs_2)
    for r1, r2 in zip(recs_1, recs_2):
        assert r1.id == r2.id
        assert r1.target_text == r2.target_text


def test_prompt_masking_and_target_preservation(shared_model_wrapper):
    """Verifies prompt masking with -100 and non-truncation of target text."""
    rec = UnifiedRecord(
        dataset="totto",
        id="test_masking",
        input_type="table",
        structured_input={},
        nodes=[{"id": f"c_{i}", "label": f"Cell Value {i}", "type": "cell"} for i in range(15)],
        edges=[],
        target_text="Unique target phrase that must be kept intact.",
    )

    inp_t, lab_t, node_map = format_d2t_training_example(
        record=rec,
        serializer=shared_model_wrapper.serializer,
        tokenizer=shared_model_wrapper.tokenizer,
        max_seq_len=64,
    )

    labels = lab_t[0].tolist()
    assert -100 in labels
    assert labels[0] == -100  # Prefix prompt masked

    target_tokens = [tid for tid in labels if tid != -100]
    decoded_target = shared_model_wrapper.tokenizer.decode(target_tokens)
    assert "Unique target phrase that must be kept intact" in decoded_target


def test_lora_gradients_and_frozen_base(shared_model_wrapper):
    """Verifies that base model parameters remain frozen while LoRA parameters receive gradients."""
    shared_model_wrapper.train()
    shared_model_wrapper.psd_enabled = True
    shared_model_wrapper.lambda_psd = 0.1

    rec = UnifiedRecord(
        dataset="dart",
        id="test_grads",
        input_type="triples",
        structured_input={},
        nodes=[{"id": "A", "label": "Alpha", "type": "entity"}, {"id": "B", "label": "Beta", "type": "entity"}],
        edges=[{"source": "A", "target": "B", "relation": "connects"}],
        target_text="Alpha connects to Beta.",
    )

    inp_t, lab_t, node_map = format_d2t_training_example(
        record=rec,
        serializer=shared_model_wrapper.serializer,
        tokenizer=shared_model_wrapper.tokenizer,
        max_seq_len=64,
    )

    device = shared_model_wrapper.device_str
    inp_t = inp_t.to(device)
    lab_t = lab_t.to(device)

    shared_model_wrapper.zero_grad()
    output = shared_model_wrapper(
        input_ids=inp_t,
        labels=lab_t,
        token_to_node_map=node_map,
        record=rec,
        max_seq_len=64,
    )
    output.total_loss.backward()

    trainable_params = [p for p in shared_model_wrapper.parameters() if p.requires_grad]
    frozen_params = [p for p in shared_model_wrapper.parameters() if not p.requires_grad]

    assert len(trainable_params) > 0
    assert len(frozen_params) > 0

    # Ensure trainable parameters have non-zero gradients
    grads = [p.grad for p in trainable_params if p.grad is not None]
    assert len(grads) == len(trainable_params)
    assert sum(g.norm().item() for g in grads) > 0.0


def test_checkpoint_saving_and_metadata(shared_model_wrapper):
    """Verifies that LoRA checkpoints and experiment metadata can be saved."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ExperimentTrainer(
            config_path="configs/baseline_config.yaml",
            model_wrapper=shared_model_wrapper,
        )
        trainer.save_checkpoint(tmp_dir)

        assert os.path.exists(os.path.join(tmp_dir, "adapter_config.json"))
        assert os.path.exists(os.path.join(tmp_dir, "adapter_model.safetensors")) or os.path.exists(os.path.join(tmp_dir, "adapter_model.bin"))
        assert os.path.exists(os.path.join(tmp_dir, "experiment_meta.json"))

        with open(os.path.join(tmp_dir, "experiment_meta.json"), "r") as f:
            meta = json.load(f)
        assert meta["experiment_type"] == "baseline"


def test_position_controlled_variants_preservation():
    """Verifies that positional variants preserve target text and facts."""
    rec = UnifiedRecord(
        dataset="webnlg",
        id="test_pos_var",
        input_type="triples",
        structured_input={},
        nodes=[{"id": "A", "label": "Subject A", "type": "entity"}, {"id": "B", "label": "Object B", "type": "entity"}],
        edges=[{"source": "A", "target": "B", "relation": "rel"}],
        target_text="Subject A is related to Object B.",
    )

    variants = create_positional_variants(rec)
    assert "beginning" in variants
    assert "middle" in variants
    assert "end" in variants

    for pos, v_rec in variants.items():
        assert v_rec.target_text == rec.target_text
        assert any(n["id"] == "A" for n in v_rec.nodes)
        assert any(n["id"] == "B" for n in v_rec.nodes)


def test_metrics_evaluator_consistency():
    """Verifies BLEU and ROUGE scoring on identical and divergent pairs."""
    evaluator = MetricEvaluator()
    predictions = ["The capital of France is Paris."]
    references = [["The capital of France is Paris."]]

    metrics_perf = evaluator.evaluate_batch(predictions, references)
    assert metrics_perf["rouge1"] == 100.0
    assert metrics_perf["rougeL"] == 100.0
    assert metrics_perf["bleu"] >= 90.0

    predictions_diff = ["Completely unrelated sentence."]
    metrics_diff = evaluator.evaluate_batch(predictions_diff, references)
    assert metrics_diff["rougeL"] < 30.0
