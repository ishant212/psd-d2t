"""Robust Resuming Downloader for Cross-Model Weights.

Downloads large safetensors files with automatic HTTP Range resume,
chunk flushing, and retry logic.
"""

import os
import sys
import time
import urllib.request
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def download_file_with_resume(
    url: str,
    output_path: str,
    max_retries: int = 100,
) -> bool:
    """Downloads a file using HTTP Range headers to support arbitrary resumption."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Determine total content length
    total_size = 0
    try:
        req_head = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_head, timeout=20) as resp:
            total_size = int(resp.headers.get("content-length", 0))
    except Exception as e:
        print(f"Warning: could not fetch content-length via HEAD: {e}")

    print(f"\nTarget File: {output_path}")
    print(f"Total Size:  {total_size / (1024 * 1024):.2f} MB" if total_size else "Total Size: Unknown")

    retries = 0
    while retries < max_retries:
        cur_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        if total_size > 0 and cur_size >= total_size:
            print(f"File is already fully downloaded: {cur_size / (1024 * 1024):.2f} MB")
            return True

        headers = {"User-Agent": "Mozilla/5.0"}
        mode = "wb"
        if cur_size > 0:
            headers["Range"] = f"bytes={cur_size}-"
            mode = "ab"
            print(f"Resuming download from byte {cur_size} ({cur_size / (1024 * 1024):.2f} MB)...")

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                # Handle 200 vs 206 status
                status = getattr(resp, "status", 200)
                if cur_size > 0 and status == 200:
                    # Server did not honor Range, reset to overwrite
                    mode = "wb"
                    cur_size = 0

                with open(output_path, mode) as out_f:
                    t0 = time.time()
                    downloaded_session = 0
                    last_log = cur_size

                    while True:
                        chunk = resp.read(1024 * 512)  # 512 KB
                        if not chunk:
                            break
                        out_f.write(chunk)
                        out_f.flush()
                        cur_size += len(chunk)
                        downloaded_session += len(chunk)

                        if cur_size - last_log >= 10 * 1024 * 1024:  # Log every 10 MB
                            elapsed = time.time() - t0
                            speed = (downloaded_session / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                            pct = (cur_size / total_size * 100) if total_size > 0 else 0
                            print(
                                f"  [{os.path.basename(output_path)}] "
                                f"{cur_size / (1024 * 1024):.1f} / {total_size / (1024 * 1024):.1f} MB "
                                f"({pct:.1f}%) | {speed:.2f} MB/s",
                                flush=True,
                            )
                            last_log = cur_size

            if total_size > 0 and cur_size >= total_size:
                print(f"Successfully completed download: {output_path} ({cur_size / (1024 * 1024):.2f} MB)")
                return True

        except Exception as e:
            retries += 1
            print(f"Connection interrupted ({e}). Resuming in 3s (retry {retries}/{max_retries})...", flush=True)
            time.sleep(3)

    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama", "gemma", "all"], default="llama")
    args = parser.parse_args()

    files = {
        "llama": (
            "https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit/resolve/main/model.safetensors",
            os.path.join(PROJECT_ROOT, "models", "llama_3_2_3b_hf", "model.safetensors"),
        ),
        "gemma": (
            "https://huggingface.co/unsloth/gemma-3-4b-it-bnb-4bit/resolve/main/model.safetensors",
            os.path.join(PROJECT_ROOT, "models", "gemma_3_4b_hf", "model.safetensors"),
        ),
    }

    targets = ["llama", "gemma"] if args.model == "all" else [args.model]
    for m in targets:
        url, path = files[m]
        print(f"\n{'=' * 60}\nDOWNLOADING: {m.upper()}\n{'=' * 60}")
        success = download_file_with_resume(url, path)
        if not success:
            print(f"Failed to download {m}")
            sys.exit(1)
    print("\nAll downloads finished successfully!")
