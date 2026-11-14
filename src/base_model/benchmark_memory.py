"""Memory and long-context benchmarking for Qwen3-4B on RTX 4060 GPU.

Phases 1.5 - Steps 9 and 10.
"""

import os
import gc
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_memory_and_context_benchmarks(
    model_dir: str = "models/qwen3_4b",
    gguf_file: str = "model.gguf",
    device: str = "cuda",
):
    os.makedirs("reports/phase1_5", exist_ok=True)

    print("=" * 60)
    print("STEP 9: MEMORY / VRAM TEST (Short Sequence)")
    print("=" * 60)

    clear_gpu()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("Loading model for memory tests...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        gguf_file=gguf_file,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    base_allocated = torch.cuda.memory_allocated() / (1024**2)
    base_reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"Base Model Memory: Allocated={base_allocated:.1f} MB, Reserved={base_reserved:.1f} MB")

    # Short sequence test (e.g. 32 tokens)
    short_text = "The quick brown fox jumps over the lazy dog. " * 3
    short_inputs = tokenizer(short_text, return_tensors="pt").to(device)
    short_seq_len = short_inputs["input_ids"].shape[1]
    print(f"Short input sequence length: {short_seq_len} tokens")

    # Mode A: Standard inference (no hidden states, no attentions)
    clear_gpu()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out_a = model(**short_inputs, output_hidden_states=False, output_attentions=False)
    mem_a = {
        "mode": "A_standard_inference",
        "batch_size": 1,
        "sequence_length": short_seq_len,
        "allocated_vram_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
        "reserved_vram_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
        "peak_allocated_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
        "peak_reserved_vram_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 2),
    }
    print("Mode A:", mem_a)

    # Mode B: With hidden states
    clear_gpu()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out_b = model(**short_inputs, output_hidden_states=True, output_attentions=False)
    mem_b = {
        "mode": "B_with_hidden_states",
        "batch_size": 1,
        "sequence_length": short_seq_len,
        "allocated_vram_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
        "reserved_vram_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
        "peak_allocated_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
        "peak_reserved_vram_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 2),
        "num_hidden_tensors": len(out_b.hidden_states),
    }
    print("Mode B:", mem_b)

    # Mode C: With hidden states AND attention outputs
    clear_gpu()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out_c = model(**short_inputs, output_hidden_states=True, output_attentions=True)
    mem_c = {
        "mode": "C_with_hidden_states_and_attentions",
        "batch_size": 1,
        "sequence_length": short_seq_len,
        "allocated_vram_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
        "reserved_vram_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
        "peak_allocated_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
        "peak_reserved_vram_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 2),
        "num_hidden_tensors": len(out_c.hidden_states),
        "num_attention_tensors": len(out_c.attentions),
    }
    print("Mode C:", mem_c)

    memory_report = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2),
        "base_model_memory_mb": {
            "allocated": round(base_allocated, 2),
            "reserved": round(base_reserved, 2),
        },
        "experiments": [mem_a, mem_b, mem_c],
    }

    with open("reports/phase1_5/memory_report.json", "w", encoding="utf-8") as f:
        json.dump(memory_report, f, indent=2)
    print("Saved reports/phase1_5/memory_report.json")

    print("\n" + "=" * 60)
    print("STEP 10: LONG-CONTEXT SMOKE TEST (Progressive sequence lengths)")
    print("=" * 60)

    # Target sequence lengths: 256, 512, 1024, 2048
    target_lengths = [256, 512, 1024, 2048]
    long_context_results = []

    # Repeated seed text to achieve arbitrary token lengths
    seed = "In modern natural language processing, data-to-text generation aims to transform structured graph entities into fluent factual text. "

    for tgt_len in target_lengths:
        print(f"\nTesting sequence length ~{tgt_len}...")
        # Multiply text to get roughly tgt_len tokens
        mult = (tgt_len // 20) + 5
        text = (seed * mult)
        tokens = tokenizer(text, return_tensors="pt")
        # Slice to exact target length
        input_ids = tokens["input_ids"][:, :tgt_len].to(device)
        attention_mask = tokens["attention_mask"][:, :tgt_len].to(device)
        actual_len = input_ids.shape[1]

        clear_gpu()
        torch.cuda.reset_peak_memory_stats()
        success = False
        error_msg = None
        attn_returned = False
        hidden_returned = False
        attn_shape = None
        hidden_shape = None
        peak_alloc = 0.0
        peak_res = 0.0

        try:
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    output_hidden_states=True,
                )
            success = True
            hidden_returned = out.hidden_states is not None and len(out.hidden_states) > 0
            attn_returned = out.attentions is not None and len(out.attentions) > 0
            if hidden_returned:
                hidden_shape = list(out.hidden_states[-1].shape)
            if attn_returned:
                attn_shape = list(out.attentions[-1].shape)
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
            peak_res = torch.cuda.max_memory_reserved() / (1024**2)
            print(f"  Length {actual_len}: SUCCESS | Peak Allocated: {peak_alloc:.1f} MB | Peak Reserved: {peak_res:.1f} MB | Attn Shape: {attn_shape}")
        except torch.cuda.OutOfMemoryError as oom:
            success = False
            error_msg = f"CUDA OOM: {oom}"
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
            peak_res = torch.cuda.max_memory_reserved() / (1024**2)
            print(f"  Length {actual_len}: FAILED (OOM) | Peak: {peak_alloc:.1f} MB")
            clear_gpu()
        except Exception as e:
            success = False
            error_msg = str(e)
            print(f"  Length {actual_len}: FAILED ({e})")

        res_item = {
            "target_length": tgt_len,
            "actual_length": actual_len,
            "success": success,
            "error": error_msg,
            "peak_allocated_vram_mb": round(peak_alloc, 2),
            "peak_reserved_vram_mb": round(peak_res, 2),
            "attentions_returned": attn_returned,
            "attention_tensor_shape": attn_shape,
            "hidden_states_returned": hidden_returned,
            "hidden_state_shape": hidden_shape,
        }
        long_context_results.append(res_item)

        # Free output tensors
        del out
        clear_gpu()

    long_context_report = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2),
        "results": long_context_results,
    }

    with open("reports/phase1_5/long_context_report.json", "w", encoding="utf-8") as f:
        json.dump(long_context_report, f, indent=2)
    print("\nSaved reports/phase1_5/long_context_report.json")


if __name__ == "__main__":
    run_memory_and_context_benchmarks()
