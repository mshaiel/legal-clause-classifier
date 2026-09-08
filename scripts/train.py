#!/usr/bin/env python3
"""
Modular QLoRA Instruction-Tuning Script for Phi-3-mini on LexGLUE/LEDGAR.

Executes:
1. Environment authentication (W&B and Hugging Face Hub)
2. Dataset loading and Phi-3 conversational formatting
3. Inverse-frequency class weight computation
4. 4-bit NF4 quantized base model loading & PEFT LoRA adapter wrapping
5. Training with WeightedSFTTrainer and completion token masking
6. Direct checkpoint upload to Hugging Face Hub
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)


class WeightedDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        class_ids = [f.pop("class_id") for f in features]
        batch = super().__call__(features)
        batch["class_id"] = torch.tensor(class_ids, dtype=torch.long)
        return batch


# Import internal modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.class_weights import compute_balanced_class_weights
from src.data.dataset import format_instruction_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Phi-3 on LEDGAR via QLoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qlora_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Override target Hugging Face Hub model ID (e.g. username/phi3-ledgar-adapter)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override per-device train batch size",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_auth():
    """Authenticate with Hugging Face Hub and Weights & Biases."""
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("[AUTH] Logging into Hugging Face Hub...")
        login(token=hf_token, add_to_git_credential=True)
    else:
        print("[WARN] HF_TOKEN environment variable not set. Push to Hub may fail if private.")

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        print("[AUTH] Logging into Weights & Biases...")
        import wandb

        wandb.login(key=wandb_key)
    else:
        print("[INFO] WANDB_API_KEY not found; wandb logging will run anonymously or offline.")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Allow CLI overrides
    if args.hub_model_id:
        cfg["hub"]["hub_model_id"] = args.hub_model_id
    if args.num_epochs:
        cfg["training"]["num_train_epochs"] = args.num_epochs
    if args.batch_size:
        cfg["training"]["per_device_train_batch_size"] = args.batch_size

    print("=" * 75)
    print("🚀 PHI-3-MINI QLORA CAUSAL INSTRUCTION TUNING — LEXGLUE/LEDGAR")
    print("=" * 75)

    setup_auth()

    # 1. Tokenizer
    model_id = cfg["model"]["base_model_id"]
    print(f"\n[1/6] Loading tokenizer from: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Dataset Preparation
    dataset_name = cfg["dataset"]["dataset_name"]
    subset = cfg["dataset"]["dataset_subset"]
    print(f"\n[2/6] Loading dataset '{dataset_name}' (subset: '{subset}')...")
    raw_dataset = load_dataset(dataset_name, subset)

    label_names = raw_dataset["train"].features["label"].names
    num_classes = len(label_names)
    print(f"Total provision categories: {num_classes}")

    # Compute balanced class weights from training labels
    print("Computing balanced class weights across 60,000 training examples...")
    train_labels = raw_dataset["train"]["label"]
    class_weights = compute_balanced_class_weights(
        labels=train_labels,
        num_classes=num_classes,
        max_clip=15.0,
        normalize=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Class weights ready: min={class_weights.min():.4f}, max={class_weights.max():.4f}")

    response_token_ids = tokenizer.encode("<|assistant|>\n", add_special_tokens=False)

    def tokenize_and_mask(example):
        cat_name = label_names[example["label"]]
        full_prompt = format_instruction_prompt(example["text"], cat_name)
        enc = tokenizer(
            full_prompt,
            max_length=cfg["dataset"]["max_seq_length"],
            truncation=True,
            padding=False,
        )
        input_ids = enc["input_ids"]
        labels = list(input_ids)
        r_len = len(response_token_ids)
        start_idx = -1
        for j in range(len(input_ids) - r_len + 1):
            if input_ids[j : j + r_len] == response_token_ids:
                start_idx = j + r_len
                break
        if start_idx != -1:
            labels[:start_idx] = [-100] * start_idx
        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "class_id": example["label"],
        }

    print("Tokenizing dataset and masking completion labels...")
    train_dataset = raw_dataset["train"].map(
        tokenize_and_mask,
        remove_columns=raw_dataset["train"].column_names,
        desc="Tokenizing train",
    )
    val_dataset = raw_dataset["validation"].map(
        tokenize_and_mask,
        remove_columns=raw_dataset["validation"].column_names,
        desc="Tokenizing val",
    )

    # 3. Quantization Configuration (4-bit NF4)
    q_cfg = cfg["quantization"]
    compute_dtype = getattr(torch, q_cfg["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_cfg["load_in_4bit"],
        bnb_4bit_quant_type=q_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # 4. Load Base Model with QLoRA
    print(f"\n[3/6] Loading 4-bit base model: {model_id}...")
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config.rope_scaling = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA Config
    l_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=l_cfg["r"],
        lora_alpha=l_cfg["lora_alpha"],
        lora_dropout=l_cfg["lora_dropout"],
        bias=l_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=l_cfg["target_modules"],
    )

    print("\n[4/6] Applying LoRA adapter...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Data Collator (Pads numeric tensors and packages class_id)
    collator = WeightedDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    # 6. Training Arguments & Trainer Setup
    t_cfg = cfg["training"]
    hub_cfg = cfg["hub"]

    import inspect

    try:
        from trl import SFTConfig

        ArgsClass = SFTConfig
    except ImportError:
        ArgsClass = TrainingArguments

    candidate_args = {
        "output_dir": t_cfg["output_dir"],
        "num_train_epochs": t_cfg["num_train_epochs"],
        "per_device_train_batch_size": t_cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": t_cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": t_cfg["gradient_accumulation_steps"],
        "gradient_checkpointing": t_cfg.get("gradient_checkpointing", True),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "learning_rate": float(t_cfg["learning_rate"]),
        "lr_scheduler_type": t_cfg["lr_scheduler_type"],
        "warmup_steps": 50,
        "weight_decay": t_cfg["weight_decay"],
        "max_grad_norm": t_cfg["max_grad_norm"],
        "optim": t_cfg["optim"],
        "bf16": t_cfg["bf16"] and torch.cuda.is_bf16_supported(),
        "fp16": t_cfg["fp16"] or (not torch.cuda.is_bf16_supported() and torch.cuda.is_available()),
        "logging_steps": t_cfg["logging_steps"],
        "eval_strategy": t_cfg["eval_strategy"],
        "eval_steps": t_cfg["eval_steps"],
        "save_strategy": t_cfg["save_strategy"],
        "save_steps": t_cfg["save_steps"],
        "save_total_limit": t_cfg["save_total_limit"],
        "load_best_model_at_end": t_cfg["load_best_model_at_end"],
        "metric_for_best_model": t_cfg["metric_for_best_model"],
        "greater_is_better": t_cfg["greater_is_better"],
        "seed": t_cfg["seed"],
        "remove_unused_columns": False,
        "report_to": "wandb" if os.environ.get("WANDB_API_KEY") else "none",
        "run_name": cfg["wandb"]["run_name"],
        "push_to_hub": hub_cfg["push_to_hub"],
        "hub_model_id": hub_cfg["hub_model_id"],
        "hub_private_repo": hub_cfg["hub_private_repo"],
        "max_length": cfg["dataset"]["max_seq_length"],
        "max_seq_length": cfg["dataset"]["max_seq_length"],
        "dataset_text_field": "text",
    }

    valid_params = set(inspect.signature(ArgsClass.__init__).parameters.keys())
    filtered_args = {k: v for k, v in candidate_args.items() if k in valid_params}
    training_args = ArgsClass(**filtered_args)

    from transformers import Trainer

    from src.training.trainer import WeightedTrainer

    trainer_kwargs = {
        "class_weights": class_weights,
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": collator,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)

    print("\n[5/6] Starting QLoRA Fine-Tuning...")
    trainer.train()

    print("\n[6/6] Pushing final adapter to Hugging Face Hub...")
    if hub_cfg["push_to_hub"]:
        trainer.push_to_hub(commit_message="End of fine-tuning: best adapter weights")
        tokenizer.push_to_hub(hub_cfg["hub_model_id"])
        print(f"🎉 Successfully pushed adapter to Hugging Face Hub: {hub_cfg['hub_model_id']}")

    trainer.save_model(f"{t_cfg['output_dir']}/final_adapter")
    tokenizer.save_pretrained(f"{t_cfg['output_dir']}/final_adapter")
    print(f"Saved local adapter checkpoint to: {t_cfg['output_dir']}/final_adapter")
    print("=" * 75)


if __name__ == "__main__":
    main()
