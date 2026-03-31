#!/usr/bin/env python3
"""
merge_adapter.py

Merges a trained LoRA adapter into the Qwen2.5-14B-Instruct base model
and saves the result as a full-weight HuggingFace model directory.

Run on the inference server inside the merge virtual environment.
The merged model directory is then converted to GGUF by llama.cpp.

Timeline:
  - Base model download (first run only): 20-40 min (~28GB)
  - Merge step:                            5-10 min
  - Disk space required:                  ~60GB system RAM, ~28GB output

Usage:
  source ~/merge-env/bin/activate
  python3 merge_adapter.py \
    --adapter ~/scratch/lora_adapter \
    --output  ~/scratch/merged_model

Requirements:
  pip install torch transformers peft accelerate
"""

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"


def human_size(path: Path) -> str:
    total = sum(
        f.stat().st_size
        for f in path.rglob("*")
        if f.is_file()
    )
    return f"{total / 1e9:.1f} GB"


def main():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into the Qwen2.5-14B base model"
    )
    parser.add_argument("--adapter",    required=True,
                        help="Path to the LoRA adapter directory")
    parser.add_argument("--output",     required=True,
                        help="Output path for the merged model")
    parser.add_argument("--base-model", default=BASE_MODEL,
                        help=f"Base model name or path (default: {BASE_MODEL})")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    output_path  = Path(args.output)

    if not adapter_path.exists():
        print(f"Error: adapter not found at {adapter_path}")
        raise SystemExit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Base model:   {args.base_model}")
    print(f"Adapter:      {adapter_path}")
    print(f"Output:       {output_path}")
    print()
    print("Loading base model (downloads ~28GB on first run)...")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype       = torch.float16,
        device_map        = "auto",
        low_cpu_mem_usage = True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))

    print(f"\nMerge complete. Output size: {human_size(output_path)}")
    print()
    print("Next steps:")
    print(f"  1. Convert to fp16 GGUF:")
    print(f"     python3 ~/llama.cpp/convert_hf_to_gguf.py \\")
    print(f"       {output_path} \\")
    print(f"       --outfile ~/scratch/energy-assistant-v1-f16.gguf \\")
    print(f"       --outtype f16")
    print()
    print(f"  2. Quantise to q4_k_m:")
    print(f"     ~/llama.cpp/build/bin/llama-quantize \\")
    print(f"       ~/scratch/energy-assistant-v1-f16.gguf \\")
    print(f"       ~/scratch/energy-assistant-v1-q4_k_m.gguf \\")
    print(f"       q4_k_m")


if __name__ == "__main__":
    main()
