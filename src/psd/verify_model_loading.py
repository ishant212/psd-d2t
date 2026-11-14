"""Model loading and LoRA verification script for Phase 2 PSD using Hugging Face FP16.

Phase 2 - Step 1.
"""

import os
import sys
import json
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def verify_model_loading(
    model_path: str = "models/qwen3_4b_hf",
    output_path: str = "reports/phase2/model_loading_report.json",
) -> dict:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("STEP 1: HUGGING FACE FP16 MODEL LOADING & LORA")
    print("=" * 60)

    # 1. Environment & CUDA
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    total_vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2) if cuda_avail else 0

    print(f"CUDA Available: {cuda_avail}")
    print(f"GPU Name: {gpu_name} ({total_vram_mb} MB VRAM)")

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load Model with FP16/FP32 precision
    print(f"Loading Hugging Face checkpoint from {model_path} with FP16 precision...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if cuda_avail else torch.float32,
        device_map="auto" if cuda_avail else None,
        attn_implementation="eager",
    )

    # 4. Attach PEFT LoRA
    target_modules = ["q_proj", "v_proj"]
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    # Count parameters
    num_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in peft_model.parameters())
    trainable_pct = round((num_trainable / num_total) * 100, 4)

    print(f"Total parameters: {num_total:,}")
    print(f"Trainable parameters: {num_trainable:,} ({trainable_pct}%)")

    # 5. One Forward Pass Test
    test_input = tokenizer("Paris | country | France", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = peft_model(**test_input, output_attentions=True)

    alloc_mb = round(torch.cuda.memory_allocated() / (1024**2), 2) if cuda_avail else 0.0
    res_mb = round(torch.cuda.memory_reserved() / (1024**2), 2) if cuda_avail else 0.0

    report = {
        "model_name": "Qwen/Qwen3-4B (Hugging Face)",
        "model_path": model_path,
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "gpu_total_vram_mb": total_vram_mb,
        "precision": "FP16 (Half Precision)",
        "linear_class": str(type(peft_model.model.model.layers[0].self_attn.q_proj)),
        "total_parameters": num_total,
        "trainable_parameters": num_trainable,
        "trainable_percentage": trainable_pct,
        "lora_modules": target_modules,
        "lora_rank": 8,
        "lora_alpha": 16,
        "forward_pass_success": outputs.logits is not None,
        "output_logits_shape": list(outputs.logits.shape),
        "attentions_returned": len(outputs.attentions) if outputs.attentions else 0,
        "allocated_gpu_memory_mb": alloc_mb,
        "reserved_gpu_memory_mb": res_mb,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved model loading report to {output_path}")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    verify_model_loading()
