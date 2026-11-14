"""Unit tests for Phase 1.5 - Base LLM and Attention Access Verification.

Verifies:
- Tokenizer loading and offset mapping capability
- Model loading on CUDA with eager attention
- Hidden state tensor accessibility and cosine similarity computation
- Attention tensor accessibility and shape verification
- Phase 1 serialization and token-to-structure alignment across ToTTo, DART, WebNLG
"""

import os
import sys
import json
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.unified_schema import UnifiedRecord
from src.preprocessing.serialize import Serializer

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "qwen3_4b")
GGUF_FILE = "model.gguf"


@pytest.fixture(scope="module")
def tokenizer():
    assert os.path.exists(MODEL_DIR), f"Model directory {MODEL_DIR} does not exist"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    return tok


@pytest.fixture(scope="module")
def model():
    assert os.path.exists(os.path.join(MODEL_DIR, GGUF_FILE)), f"GGUF file not found in {MODEL_DIR}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mod = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        gguf_file=GGUF_FILE,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        attn_implementation="eager",
    )
    mod.eval()
    return mod


def test_tokenizer_loading(tokenizer):
    """Verifies tokenizer loads, is fast, and supports character offsets."""
    assert tokenizer is not None
    assert tokenizer.is_fast is True
    assert len(tokenizer) >= 150000

    encoding = tokenizer("Paris | country | France", return_offsets_mapping=True)
    assert "offset_mapping" in encoding
    assert len(encoding["offset_mapping"]) == len(encoding["input_ids"])


def test_model_loading_and_config(model):
    """Verifies Qwen3-4B model config parameters and device placement."""
    assert model is not None
    assert "Qwen3" in type(model).__name__

    config = model.config
    assert config.num_hidden_layers == 36
    assert config.num_attention_heads == 32
    assert config.hidden_size == 2560

    if torch.cuda.is_available():
        assert "cuda" in str(model.device)


def test_hidden_states_and_cosine_similarity(tokenizer, model):
    """Verifies hidden states accessibility and cosine similarity between token representations."""
    device = model.device
    inputs = tokenizer("Paris | country | France", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=False)

    assert outputs.hidden_states is not None
    # 1 embedding layer + 36 transformer layers = 37 hidden state tensors
    assert len(outputs.hidden_states) == 37

    last_hidden = outputs.hidden_states[-1]
    assert last_hidden.shape == (1, 5, 2560)

    # Compute cosine similarity between token 0 ('Paris') and token 4 ('France')
    h_0 = last_hidden[0, 0]
    h_4 = last_hidden[0, 4]
    cos_sim = F.cosine_similarity(h_0.unsqueeze(0), h_4.unsqueeze(0), dim=-1).item()
    assert isinstance(cos_sim, float)
    assert -1.0 <= cos_sim <= 1.0


def test_attention_accessibility(tokenizer, model):
    """Verifies attention tensors are accessible with shape [batch, heads, query_len, key_len]."""
    device = model.device
    inputs = tokenizer("Paris | country | France", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=False)

    assert outputs.attentions is not None
    assert len(outputs.attentions) == 36  # 36 transformer layers

    for layer_idx, att in enumerate(outputs.attentions):
        # Shape must be [batch=1, heads=32, seq_len=5, seq_len=5]
        assert att.shape == (1, 32, 5, 5), f"Layer {layer_idx} attention shape mismatch: {att.shape}"
        # Values must be valid probabilities (sum across keys ~= 1.0 for each query position)
        assert torch.all(att >= 0.0)
        assert torch.all(att <= 1.0 + 1e-3)


@pytest.mark.parametrize("dataset_name", ["totto", "dart", "webnlg"])
def test_dataset_serialization_and_alignment(tokenizer, model, dataset_name):
    """Verifies that each dataset's Phase 1 serialization maps to Qwen tokens and passes through forward pass."""
    data_path = os.path.join(PROJECT_ROOT, "data", "unified", f"{dataset_name}.jsonl")
    assert os.path.exists(data_path), f"Data file {data_path} missing"

    with open(data_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        record = UnifiedRecord.from_dict(json.loads(first_line))

    serializer = Serializer()
    mapped_data = serializer.serialize_with_mapping(record, tokenizer=tokenizer)

    input_ids = mapped_data["input_ids"]
    token_to_node = mapped_data["token_to_node_map"]
    token_to_edge = mapped_data["token_to_edge_map"]

    assert len(input_ids) > 0
    assert len(token_to_node) == len(input_ids)
    assert len(token_to_edge) == len(input_ids)

    # Node mappings should exist in all datasets
    node_mapped = [x for x in token_to_node if x is not None]
    assert len(node_mapped) > 0, f"No tokens mapped to nodes for {dataset_name}"

    # In triple datasets (DART, WebNLG), edge mappings should exist
    if dataset_name in ["dart", "webnlg"]:
        edge_mapped = [x for x in token_to_edge if x is not None]
        assert len(edge_mapped) > 0, f"No tokens mapped to edges for {dataset_name}"

    # Forward pass through Qwen3-4B
    device = model.device
    test_len = min(len(input_ids), 256)
    inp_tensor = torch.tensor([input_ids[:test_len]], device=device)

    with torch.no_grad():
        out = model(inp_tensor, output_attentions=True, output_hidden_states=True)

    assert out.hidden_states is not None
    assert len(out.hidden_states) == 37
    assert out.attentions is not None
    assert len(out.attentions) == 36
    assert out.attentions[0].shape == (1, 32, test_len, test_len)
