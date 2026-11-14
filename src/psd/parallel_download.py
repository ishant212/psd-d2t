"""Multi-Threaded Parallel Chunk Downloader for Hugging Face Safetensors.

Uses concurrent HTTP Range requests across 8 worker threads to download
large model checkpoints at maximum available aggregate bandwidth.
"""

import os
import sys
import time
import urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

MODEL_METADATA = {
    "llama": {
        "url": "https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit/resolve/main/model.safetensors",
        "output_path": os.path.join(PROJECT_ROOT, "models", "llama_3_2_3b_hf", "model.safetensors"),
        "total_bytes": 2242762780,
    },
    "gemma": {
        "url": "https://huggingface.co/unsloth/gemma-3-4b-it-bnb-4bit/resolve/main/model.safetensors",
        "output_path": os.path.join(PROJECT_ROOT, "models", "gemma_3_4b_hf", "model.safetensors"),
        "total_bytes": 3228933621,
    },
}


def download_chunk(
    url: str,
    output_path: str,
    start_byte: int,
    end_byte: int,
    chunk_idx: int,
    max_retries: int = 5,
) -> int:
    """Downloads a single byte range and writes it into output_path at start_byte."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Range": f"bytes={start_byte}-{end_byte}",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                expected_len = end_byte - start_byte + 1
                if len(data) != expected_len:
                    raise IOError(f"Incomplete chunk {chunk_idx}: got {len(data)}, expected {expected_len}")

                with open(output_path, "r+b") as out_f:
                    out_f.seek(start_byte)
                    out_f.write(data)
                return len(data)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"ERROR: Chunk {chunk_idx} [{start_byte}-{end_byte}] failed after {max_retries} attempts: {e}")
                raise
            time.sleep(1.5 * (attempt + 1))
    return 0


def parallel_download_model(
    model_name: str,
    num_workers: int = 8,
    chunk_size_mb: int = 20,
) -> bool:
    """Downloads model safetensors in parallel chunks."""
    info = MODEL_METADATA[model_name]
    url = info["url"]
    output_path = info["output_path"]
    total_bytes = info["total_bytes"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Initialize file size
    if not os.path.exists(output_path):
        with open(output_path, "wb") as f:
            f.seek(total_bytes - 1)
            f.write(b"\0")
        start_offset = 0
    else:
        existing_size = os.path.getsize(output_path)
        if existing_size >= total_bytes:
            print(f"[{model_name.upper()}] File is already fully downloaded: {existing_size / (1024*1024):.2f} MB")
            return True
        # If file is partially downloaded from sequential download, reuse existing contiguous prefix
        start_offset = existing_size
        with open(output_path, "r+b") as f:
            f.seek(total_bytes - 1)
            f.write(b"\0")

    chunk_size = chunk_size_mb * 1024 * 1024
    chunks: List[Tuple[int, int, int]] = []
    chunk_idx = 0
    cur = start_offset
    while cur < total_bytes:
        end = min(cur + chunk_size - 1, total_bytes - 1)
        chunks.append((cur, end, chunk_idx))
        chunk_idx += 1
        cur = end + 1

    remaining_bytes = total_bytes - start_offset
    print(f"\n[{model_name.upper()}] Starting Parallel Download:")
    print(f"  Target file:     {output_path}")
    print(f"  Total size:      {total_bytes / (1024*1024):.2f} MB")
    print(f"  Already on disk: {start_offset / (1024*1024):.2f} MB")
    print(f"  Remaining:       {remaining_bytes / (1024*1024):.2f} MB ({len(chunks)} chunks @ {chunk_size_mb} MB)")
    print(f"  Workers:         {num_workers} concurrent threads\n")

    t0 = time.time()
    downloaded_bytes = start_offset
    completed_chunks = 0
    total_chunks = len(chunks)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_chunk, url, output_path, s, e, idx): (s, e, idx)
            for s, e, idx in chunks
        }

        for future in as_completed(futures):
            n_bytes = future.result()
            downloaded_bytes += n_bytes
            completed_chunks += 1

            elapsed = time.time() - t0
            session_bytes = downloaded_bytes - start_offset
            speed = (session_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            pct = (downloaded_bytes / total_bytes) * 100
            eta_sec = ((total_bytes - downloaded_bytes) / (1024 * 1024)) / speed if speed > 0 else 0

            print(
                f"  [{completed_chunks:3d}/{total_chunks}] "
                f"{downloaded_bytes / (1024*1024):.1f} / {total_bytes / (1024*1024):.1f} MB "
                f"({pct:.1f}%) | {speed:.2f} MB/s | ETA: {eta_sec:.0f}s",
                flush=True,
            )

    total_time = time.time() - t0
    final_size = os.path.getsize(output_path)
    assert final_size == total_bytes, f"Size mismatch: {final_size} vs {total_bytes}"
    print(f"\n[{model_name.upper()}] Parallel download complete in {total_time:.1f}s ({final_size / (1024*1024):.2f} MB)!")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama", "gemma", "all"], default="llama")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    targets = ["llama", "gemma"] if args.model == "all" else [args.model]
    for target in targets:
        success = parallel_download_model(target, num_workers=args.workers)
        if not success:
            sys.exit(1)
