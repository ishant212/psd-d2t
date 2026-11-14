"""Standalone script to verify Qwen3-4B Hugging Face model, attention, and hidden state access.

Phases 1.5 - Steps 4, 5, 6.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_qwen_attention_test(
    model_dir: str = "models/qwen3_4b",
    gguf_file: str = "model.gguf",
    input_text: str = "Paris | country | France",
    output_report_path: str = "reports/phase1_5/attention_access_report.json",
    device: str = None,
) -> Dict[str, Any]:
    """Tests model loading, attention tensors, and hidden states extraction."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1/5] Loading tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print(f"      Tokenizer vocab size: {len(tokenizer)}")

    print(f"[2/5] Loading Qwen3-4B model onto {device} (eager attention)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        gguf_file=gguf_file,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        attn_implementation="eager",  # Required for output_attentions=True
    )
    model.eval()
    print(f"      Model loaded: {type(model).__name__}")
    print(f"      Device: {model.device}, Dtype: {model.dtype}")

    # Inspect model configuration
    config = model.config
    num_layers_cfg = getattr(config, "num_hidden_layers", None)
    num_heads_cfg = getattr(config, "num_attention_heads", None)
    hidden_size_cfg = getattr(config, "hidden_size", None)
    print(f"      Config detected: {num_layers_cfg} layers, {num_heads_cfg} heads, hidden_size={hidden_size_cfg}")

    # Tokenize input
    print(f"[3/5] Tokenizing test input: '{input_text}'")
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"][0].tolist()
    seq_len = len(input_ids)
    decoded_tokens = [tokenizer.decode([tid]) for tid in input_ids]

    print(f"      Sequence length: {seq_len}")
    print(f"      Token IDs: {input_ids}")
    print(f"      Decoded tokens: {decoded_tokens}")

    # Run forward pass requesting attentions and hidden states
    print("[4/5] Executing forward pass with output_attentions=True, output_hidden_states=True...")
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
        )

    # Inspect Hidden States
    hidden_states = outputs.hidden_states
    hidden_accessible = hidden_states is not None and len(hidden_states) > 0
    num_hidden_layers = len(hidden_states) if hidden_accessible else 0
    hidden_dim = hidden_states[-1].shape[-1] if hidden_accessible else None

    print(f"      Hidden states accessible: {hidden_accessible}")
    print(f"      Number of hidden state tensors: {num_hidden_layers} (includes embedding layer + {num_hidden_layers - 1} transformer layers)")
    if hidden_accessible:
        print(f"      Hidden state tensor shape: {list(hidden_states[-1].shape)} [batch, seq_len, hidden_dim]")

    # Hidden states token similarity verification (h_i, h_j)
    cosine_sim_example = None
    if hidden_accessible and seq_len >= 2:
        last_hidden = hidden_states[-1][0]  # [seq_len, hidden_dim]
        # Compare token 0 ("Paris") with token 4 ("France")
        pos_i = 0
        pos_j = seq_len - 1
        h_i = last_hidden[pos_i]
        h_j = last_hidden[pos_j]
        cos_sim = F.cosine_similarity(h_i.unsqueeze(0), h_j.unsqueeze(0), dim=-1).item()
        cosine_sim_example = {
            "token_i": {"position": pos_i, "token_id": input_ids[pos_i], "token_str": decoded_tokens[pos_i], "norm": round(torch.norm(h_i).item(), 4)},
            "token_j": {"position": pos_j, "token_id": input_ids[pos_j], "token_str": decoded_tokens[pos_j], "norm": round(torch.norm(h_j).item(), 4)},
            "cosine_similarity": round(cos_sim, 6),
        }
        print(f"      Cosine similarity between '{decoded_tokens[pos_i]}' (pos {pos_i}) and '{decoded_tokens[pos_j]}' (pos {pos_j}): {cos_sim:.6f}")

    # Inspect Attention Tensors
    attentions = outputs.attentions
    attention_accessible = attentions is not None and len(attentions) > 0
    num_attention_layers = len(attentions) if attention_accessible else 0

    per_layer_shapes = []
    if attention_accessible:
        first_layer_attn = attentions[0]
        # shape: [batch, heads, query_pos, key_pos]
        attn_shape = list(first_layer_attn.shape)
        batch_size, num_heads, query_len, key_len = attn_shape
        print(f"      Attention accessible: {attention_accessible}")
        print(f"      Number of attention layers: {num_attention_layers}")
        print(f"      Attention tensor dimensions: {attn_shape} [batch, heads, query_pos, key_pos]")
        for l_idx, att in enumerate(attentions):
            per_layer_shapes.append({
                "layer": l_idx,
                "shape": list(att.shape),
                "is_causal": bool(torch.all(att[0, :, 0, 1:] == 0.0).item()) if seq_len > 1 else True
            })
    else:
        batch_size, num_heads, query_len, key_len = 0, 0, 0, 0
        print("      Attention accessible: FALSE (attentions tuple is empty or None)")

    # Build report
    report: Dict[str, Any] = {
        "model_architecture": config.model_type,
        "model_class": type(model).__name__,
        "model_dir": model_dir,
        "device": str(model.device),
        "dtype": str(model.dtype),
        "attn_implementation": getattr(config, "_attn_implementation", "eager"),
        "input_text": input_text,
        "input_sequence_length": seq_len,
        "token_ids": input_ids,
        "decoded_tokens": decoded_tokens,
        "hidden_states": {
            "accessible": hidden_accessible,
            "total_layers": num_hidden_layers,
            "transformer_layers": num_hidden_layers - 1 if hidden_accessible else 0,
            "hidden_dimension": hidden_dim,
            "shape_format": "[batch, sequence_length, hidden_dimension]",
            "cosine_similarity_verification": cosine_sim_example,
        },
        "attention_tensors": {
            "accessible": attention_accessible,
            "number_of_layers": num_attention_layers,
            "number_of_heads": num_heads,
            "query_length": query_len,
            "key_length": key_len,
            "tensor_shape_format": "[batch, heads, query_position, key_position]",
            "first_layer_shape": list(attentions[0].shape) if attention_accessible else None,
            "last_layer_shape": list(attentions[-1].shape) if attention_accessible else None,
            "layer_samples_summary": per_layer_shapes[:3] + per_layer_shapes[-1:] if per_layer_shapes else [],
        },
    }

    if output_report_path:
        os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[5/5] Saved attention access report to: {output_report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Qwen3-4B attention and hidden state access")
    parser.add_argument("--model_dir", default="models/qwen3_4b", help="Path to model directory")
    parser.add_argument("--gguf_file", default="model.gguf", help="GGUF filename in model directory")
    parser.add_argument("--text", default="Paris | country | France", help="Input text to test")
    parser.add_argument("--output_report", default="reports/phase1_5/attention_access_report.json", help="Path to save report")
    parser.add_argument("--device", default=None, help="Device to run on (cuda or cpu)")
    args = parser.parse_args()

    run_qwen_attention_test(
        model_dir=args.model_dir,
        gguf_file=args.gguf_file,
        input_text=args.text,
        output_report_path=args.output_report,
        device=args.device,
    )
