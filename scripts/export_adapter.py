#!/usr/bin/env python3
"""
Adapter Verification and Download Utility.

Connects to Hugging Face Hub to:
1. Verify the trained LoRA adapter files exist (adapter_config.json, adapter_model.safetensors)
2. Check tokenizer configuration
3. Optionally download the adapter weights locally to checkpoints/ for local inference
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, login


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and download fine-tuned LoRA adapter from Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hugging Face repo ID (e.g., username/phi3-legal-clause-qlora)",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default="./checkpoints/adapter",
        help="Local directory to store adapter files for local inference",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Only verify repo files on Hub without downloading locally",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    hub_model_id = args.hub_model_id or os.environ.get("HF_HUB_MODEL_ID")
    if not hub_model_id:
        print("[ERROR] Please provide --hub_model_id or set HF_HUB_MODEL_ID in .env")
        return

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    api = HfApi(token=token)

    print("=" * 70)
    print(f"🔍 VERIFYING HUGGING FACE HUB ADAPTER REPOSITORY: {hub_model_id}")
    print("=" * 70)

    try:
        repo_files = api.list_repo_files(repo_id=hub_model_id)
    except Exception as e:
        print(f"[ERROR] Could not access repo '{hub_model_id}': {e}")
        print("Please check your repository name and HF_TOKEN permissions.")
        return

    print(f"Files found in repository ({len(repo_files)}):")
    for f in repo_files:
        print(f"  ✓ {f}")

    essential_files = ["adapter_config.json"]
    has_weights = "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files

    if not all(ef in repo_files for ef in essential_files) or not has_weights:
        print("\n[WARNING] Repository does not contain complete LoRA adapter weights.")
        return

    print("\n[SUCCESS] Adapter repository verified! All essential weights and configs are present.")

    if not args.skip_download:
        dest = Path(args.download_dir)
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading adapter files to local directory: {dest}...")

        files_to_download = [
            f for f in repo_files if f.endswith((".json", ".safetensors", ".bin", ".txt", ".model"))
        ]

        for file_name in files_to_download:
            print(f"  Downloading: {file_name}...")
            hf_hub_download(
                repo_id=hub_model_id,
                filename=file_name,
                local_dir=str(dest),
                token=token,
            )

        print(f"\n✅ Adapter downloaded to: {dest.resolve()}")
        print("Ready for local inference testing on GTX 1050!")
    print("=" * 70)


if __name__ == "__main__":
    main()
