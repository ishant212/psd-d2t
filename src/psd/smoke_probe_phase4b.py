"""Smoke and VRAM Safety Probe for Phase 4B (L=512 and L=1024)."""
import os
import sys
import time
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.psd.trainer import ExperimentTrainer, load_dataset_records, format_d2t_training_example

def run_smoke_probe():
    print("=== Phase 4B Smoke and VRAM Safety Probe ===")
    configs = [
        ("Baseline 512", "configs/phase4b_baseline_512.yaml", 512),
        ("Full PSD 512", "configs/phase4b_psd_512.yaml", 512),
        ("Baseline 1024", "configs/phase4b_baseline_1024.yaml", 1024),
        ("Full PSD1024", "configs/phase4b_psd_1024.yaml", 1024),
    ]
    
    train_records = load_dataset_records(split="train", max_records=20)
    print(f"Loaded {len(train_records)} training records for probe.")

    for name, cfg_path, seq_len in configs:
        print(f"\n--- Testing {name} (L{seq_len}) ---")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        trainer = ExperimentTrainer(config_path=os.path.join(PROJECT_ROOT, cfg_path))
        device = trainer.device
        print(f"Device: {device}")
        
        # Run 4 accumulation steps (1 optimizer step)
        trainer.optimizer.zero_grad()
        accum_loss = 0.0
        
        t0 = time.time()
        try:
            for step in range(4):
                rec = train_records[step % len(train_records)]
                inp_t, lab_t, node_map = format_d2t_training_example(
                    record=rec,
                    serializer=trainer.serializer,
                    tokenizer=trainer.model_wrapper.tokenizer,
                    max_seq_len=seq_len,
                )
                inp_t = inp_t.to(device)
                lab_t = lab_t.to(device)
                
                output = trainer.model_wrapper(
                    input_ids=inp_t,
                    labels=lab_t,
                    token_to_node_map=node_map,
                    record=rec,
                    max_seq_len=seq_len,
                )
                
                loss = output.total_loss / 4.0
                assert not torch.isnan(loss), f"NaN loss encountered in {name}!"
                loss.backward()
                accum_loss += output.total_loss.item()
                
                if trainer.psd_enabled:
                    print(f"  Step {step+1}/4 | LM Loss: {output.lm_loss.item():.4f} | PSD Loss: {output.psd_loss.item():.4f}")
                else:
                    print(f"  Step {step+1}/4 | LM Loss: {output.lm_loss.item():.4f}")
                    
            trainer.optimizer.step()
            elapsed = time.time() - t0
            
            alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            res_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
            print(f"[PROBE PASSED] {name} | 1 Opt Step Time: {elapsed:.2f}s | Peak Alloc VRAM: {alloc_mb:.1f} MB | Peak Reserved: {res_mb:.1f} MB")
        except torch.OutOfMemoryError as e:
            print(f"[OOM DETECTED] {name} (L={seq_len}) triggered CUDA OOM during backward pass!")
            print(f"  Detail: {e}")
        except Exception as e:
            print(f"[ERROR] {name} (L={seq_len}) failed with error: {e}")
        finally:
            del trainer
            torch.cuda.empty_cache()

    print("\n=== SMOKE PROBE RUN COMPLETED ===")

if __name__ == "__main__":
    run_smoke_probe()

