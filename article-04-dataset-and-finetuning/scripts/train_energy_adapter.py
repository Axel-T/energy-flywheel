#!/usr/bin/env python3
"""
train_energy_adapter.py

Fine-tunes Qwen2.5-14B-Instruct on the energy QA dataset using
QLoRA via Unsloth. Run this on the rented cloud GPU instance.

Outputs a LoRA adapter to ./output/lora_adapter/ — archive and
rsync back to the homelab MinIO adapters/ bucket when complete.

Expected runtime on RTX 4090:
  ~90 min for ~420 examples · 3 epochs
  ~4 hrs for ~5,000 examples · 3 epochs

Usage:
  python3 train_energy_adapter.py \
    --data /workspace/data/energy_qa_v1.jsonl

Requirements (see requirements.txt):
  pip install -r requirements.txt
"""

import argparse

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# ── Configuration ───────────────────────────────────────────────────────────────

BASE_MODEL     = "Qwen/Qwen2.5-14B-Instruct"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT   = True   # QLoRA — reduces VRAM from ~28GB to ~10GB

ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{}\n\n"
    "### Input:\n{}\n\n"
    "### Response:\n{}"
)


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Qwen2.5-14B on energy QA data"
    )
    parser.add_argument("--data",    required=True,
                        help="Path to JSONL training file")
    parser.add_argument("--output",  default="./output",
                        help="Output directory for adapter and checkpoints "
                             "(default: ./output)")
    parser.add_argument("--epochs",  type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--lr",      type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    args = parser.parse_args()

    # ── Load model ──────────────────────────────────────────────────────────────
    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,        # auto-detect bfloat16
        load_in_4bit   = LOAD_IN_4BIT,
    )

    # ── Apply LoRA ──────────────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r              = 16,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha     = 16,
        lora_dropout   = 0,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 42,
    )

    # ── Dataset ─────────────────────────────────────────────────────────────────
    eos = tokenizer.eos_token

    def format_prompts(examples):
        return {
            "text": [
                ALPACA_PROMPT.format(inst, inp, out) + eos
                for inst, inp, out in zip(
                    examples["instruction"],
                    examples["input"],
                    examples["output"],
                )
            ]
        }

    raw = load_dataset("json", data_files=args.data, split="train")
    split = raw.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"].map(format_prompts, batched=True)
    eval_ds  = split["test"].map(format_prompts, batched=True)

    print(f"Training examples:   {len(train_ds)}")
    print(f"Validation examples: {len(eval_ds)}")

    # ── Training arguments ──────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                   = f"{args.output}/checkpoints",
        num_train_epochs             = args.epochs,
        per_device_train_batch_size  = 2,
        per_device_eval_batch_size   = 2,
        gradient_accumulation_steps  = 4,
        warmup_steps                 = 20,
        learning_rate                = args.lr,
        fp16                         = not torch.cuda.is_bf16_supported(),
        bf16                         = torch.cuda.is_bf16_supported(),
        logging_steps                = 10,
        eval_strategy                = "steps",
        eval_steps                   = 50,
        save_strategy                = "steps",
        save_steps                   = 100,
        load_best_model_at_end       = True,
        metric_for_best_model        = "eval_loss",
        optim                        = "adamw_8bit",
        weight_decay                 = 0.01,
        lr_scheduler_type            = "cosine",
        seed                         = 42,
        report_to                    = "none",
    )

    # ── Trainer ─────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = train_ds,
        eval_dataset       = eval_ds,
        dataset_text_field = "text",
        max_seq_length     = MAX_SEQ_LENGTH,
        dataset_num_proc   = 2,
        packing            = False,
        args               = training_args,
    )

    print("\nStarting training...")
    stats = trainer.train()

    print(f"\nTraining complete.")
    print(f"  Steps:         {stats.global_step}")
    print(f"  Training loss: {stats.training_loss:.4f}")
    print(f"  Runtime:       {stats.metrics['train_runtime']:.0f}s "
          f"({stats.metrics['train_runtime']/60:.1f} min)")

    # ── Save adapter ────────────────────────────────────────────────────────────
    adapter_dir = f"{args.output}/lora_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"\nLoRA adapter saved to {adapter_dir}")
    print("Next step: tar -czf lora_adapter.tar.gz output/lora_adapter/")
    print("Then rsync to your MinIO adapters/ bucket.")


if __name__ == "__main__":
    main()
