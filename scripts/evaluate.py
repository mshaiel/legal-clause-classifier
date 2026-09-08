#!/usr/bin/env python3
"""
Benchmark Evaluation Script for Fine-Tuned Phi-3 QLoRA on LexGLUE/LEDGAR.

Evaluates the fine-tuned adapter against the official LexGLUE/LEDGAR benchmark test set,
calculates Macro-F1, Micro-F1, Weighted-F1, and generates side-by-side comparisons
against published baselines (BERT-base, RoBERTa-base, LegalBERT).

Usage:
    python scripts/evaluate.py --num_samples 100
    python scripts/evaluate.py --split test --num_samples 1000
    python scripts/evaluate.py --split test  # evaluates full test set
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DynamicCache,
)

# Fix for remote Phi-3 code compatibility with modern Transformers (v4.44+)
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

# Internal module imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.dataset import (
    LEDGAR_PROVISION_CLASSES,
    SYSTEM_PROMPT,
)
from src.model.metrics import (
    compute_classification_metrics,
    map_text_to_label_id,
)

# Published LexGLUE benchmark baselines (Chalkidis et al., 2022)
PUBLISHED_LEXGLUE_BASELINES = {
    "BERT-base": {
        "macro_f1": 80.0,
        "micro_f1": 87.5,
        "model_type": "Encoder (110M)",
        "pretraining": "General Domain (Wikipedia + BookCorpus)",
    },
    "RoBERTa-base": {
        "macro_f1": 81.5,
        "micro_f1": 88.2,
        "model_type": "Encoder (125M)",
        "pretraining": "General Domain (160GB text)",
    },
    "LegalBERT": {
        "macro_f1": 82.5,
        "micro_f1": 88.4,
        "model_type": "Encoder (110M)",
        "pretraining": "Domain-Specific (12GB US/UK/EU legal text)",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phi-3 QLoRA on LexGLUE/LEDGAR")
    parser.add_argument(
        "--adapter_id",
        type=str,
        default="mshaiel2004/phi3-legal-clause-qlora",
        help="Hugging Face Hub repository ID or local path to LoRA adapter",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Base model ID on Hugging Face Hub",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for full split)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=256,
        help="Maximum input sequence length",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save benchmark metrics and comparison reports",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def load_model_and_tokenizer(
    base_model_id: str,
    adapter_id: str,
    device: str,
) -> tuple[Any, Any]:
    """Load base model with 4-bit quantization and merge LoRA adapter."""
    print(f"[1/4] Loading tokenizer for '{base_model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_id if Path(adapter_id).exists() else base_model_id,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/4] Loading model '{base_model_id}' on {device}...")
    config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    config.rope_scaling = None

    if device == "cuda" and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            config=config,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            config=config,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    from transformers.generation import GenerationMixin

    if not hasattr(model, "generate"):
        model.__class__ = type(model.__class__.__name__, (model.__class__, GenerationMixin), {})

    from transformers import GenerationConfig

    try:
        gen_config = GenerationConfig.from_pretrained(base_model_id)
    except Exception:
        gen_config = GenerationConfig()

    model.generation_config = gen_config

    print(f"[3/4] Attaching fine-tuned adapter: '{adapter_id}'...")
    model = PeftModel.from_pretrained(model, adapter_id)
    model.generation_config = gen_config
    model.eval()

    return model, tokenizer


def evaluate_dataset(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    max_seq_length: int,
    device: str,
) -> tuple[list[int], list[int], list[str], list[str], list[float]]:
    """Generate predictions and measure latency across test provisions."""
    y_true: list[int] = []
    y_pred: list[int] = []
    raw_preds: list[str] = []
    gold_categories: list[str] = []
    latencies: list[float] = []

    total = len(dataset)
    print(f"\n[4/4] Generating classifications across {total} provisions...")

    start_total_time = time.time()

    for idx, sample in enumerate(dataset):
        clause = sample["text"].strip()
        gold_id = sample["label"]
        gold_cat = LEDGAR_PROVISION_CLASSES[gold_id]

        prompt = (
            f"<|user|>\n"
            f"{SYSTEM_PROMPT}\n\n"
            f'Contract Provision:\n"{clause}"\n<|end|>\n'
            f"<|assistant|>\n"
        )

        inputs = tokenizer(
            prompt,
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt",
        )

        if device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=model.generation_config,
                max_new_tokens=24,
                do_sample=False,
                use_cache=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        pred_id = map_text_to_label_id(raw_text)

        y_true.append(gold_id)
        y_pred.append(pred_id)
        raw_preds.append(raw_text.strip())
        gold_categories.append(gold_cat)
        latencies.append(elapsed_ms)

        if (idx + 1) % max(1, total // 10) == 0 or (idx + 1) == total:
            interim_acc = (np.array(y_true) == np.array(y_pred)).mean() * 100.0
            print(
                f"  [{idx + 1:5d}/{total:5d}] ({((idx + 1) / total) * 100:5.1f}%) "
                f"| Current Accuracy: {interim_acc:5.2f}% "
                f"| Avg Latency: {np.mean(latencies):.1f}ms/clause"
            )

    total_eval_time = time.time() - start_total_time
    print(
        f"\nCompleted {total} evaluations in {total_eval_time:.1f}s ({total / total_eval_time:.2f} clauses/sec)"
    )

    return y_true, y_pred, raw_preds, gold_categories, latencies


def build_benchmark_comparison_table(
    our_macro_f1: float,
    our_micro_f1: float,
    our_accuracy: float,
    mean_latency_ms: float,
) -> str:
    """Format Markdown comparison table against published LexGLUE LEDGAR baselines."""
    lines = [
        "| Model Architecture | Parameter Count | Domain / Training | Macro-F1 (%) | Micro-F1 (%) |",
        "|:-------------------|:----------------|:------------------|:------------:|:------------:|",
    ]
    for model_name, data in PUBLISHED_LEXGLUE_BASELINES.items():
        lines.append(
            f"| {model_name} | {data['model_type']} | {data['pretraining']} "
            f"| {data['macro_f1']:.1f}% | {data['micro_f1']:.1f}% |"
        )

    lines.append(
        f"| **Phi-3-mini QLoRA (Ours)** | **3.8B (8M trainable)** | **Causal Instruction SFT + Class Weights** "
        f"| **{our_macro_f1 * 100:.1f}%** | **{our_micro_f1 * 100:.1f}%** |"
    )

    return "\n".join(lines)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("⚖️  LEXGLUE / LEDGAR BENCHMARK EVALUATION PIPELINE")
    print(f"Target Adapter: {args.adapter_id}")
    print(f"Base Model:     {args.base_model_id}")
    print(f"Split:          {args.split}")
    print(f"Device:         {device}")
    print("=" * 80)

    # 1. Load Dataset
    print(f"\nLoading LexGLUE/LEDGAR split: '{args.split}'...")
    dataset = load_dataset("coastalcph/lex_glue", "ledgar", split=args.split)
    if args.num_samples is not None and args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))
        print(f"Selected slice of {len(dataset)} samples for benchmarking.")

    # 2. Load Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer(
        base_model_id=args.base_model_id,
        adapter_id=args.adapter_id,
        device=device,
    )

    # 3. Run Evaluation
    y_true, y_pred, raw_preds, gold_cats, latencies = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_seq_length=args.max_seq_length,
        device=device,
    )

    # 4. Compute Comprehensive Metrics
    metrics = compute_classification_metrics(y_true, y_pred)
    macro_f1 = metrics["f1_macro"]
    micro_f1 = metrics["f1_micro"]
    weighted_f1 = metrics["f1_weighted"]
    accuracy = metrics["accuracy"]

    # Latency percentiles
    latency_summary = {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "throughput_samples_per_sec": float(1000.0 / np.mean(latencies))
        if len(latencies) > 0
        else 0.0,
    }

    # 5. Format and Print Comparison
    comparison_table_md = build_benchmark_comparison_table(
        our_macro_f1=macro_f1,
        our_micro_f1=micro_f1,
        our_accuracy=accuracy,
        mean_latency_ms=latency_summary["mean_ms"],
    )

    print("\n" + "=" * 80)
    print("📊 BENCHMARK COMPARISON RESULTS")
    print("=" * 80)
    print(comparison_table_md)
    print("\n" + "-" * 80)
    print(f"Macro-F1 (Primary LexGLUE Metric): {macro_f1 * 100:.2f}%")
    print(f"Micro-F1:                         {micro_f1 * 100:.2f}%")
    print(f"Weighted-F1:                      {weighted_f1 * 100:.2f}%")
    print(f"Accuracy:                         {accuracy * 100:.2f}%")
    print(f"Mean Latency per Clause:          {latency_summary['mean_ms']:.1f} ms")
    print("=" * 80)

    # 6. Save JSON & Markdown Reports
    report_dict = {
        "adapter_model_id": args.adapter_id,
        "base_model_id": args.base_model_id,
        "split": args.split,
        "evaluated_samples": len(dataset),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
            "weighted_f1": float(weighted_f1),
            "accuracy": float(accuracy),
        },
        "latency_profile": latency_summary,
        "lexglue_comparison": {
            "BERT-base": PUBLISHED_LEXGLUE_BASELINES["BERT-base"],
            "RoBERTa-base": PUBLISHED_LEXGLUE_BASELINES["RoBERTa-base"],
            "LegalBERT": PUBLISHED_LEXGLUE_BASELINES["LegalBERT"],
            "Phi-3-QLoRA-Ours": {
                "macro_f1": round(float(macro_f1 * 100), 2),
                "micro_f1": round(float(micro_f1 * 100), 2),
                "accuracy": round(float(accuracy * 100), 2),
            },
        },
        "sample_predictions": [
            {
                "index": i,
                "gold_category": gold_cats[i],
                "predicted_raw": raw_preds[i],
                "mapped_category": LEDGAR_PROVISION_CLASSES[y_pred[i]],
                "is_correct": bool(y_true[i] == y_pred[i]),
                "latency_ms": round(latencies[i], 2),
            }
            for i in range(min(15, len(dataset)))
        ],
    }

    metrics_json_path = out_dir / "benchmark_results.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[SAVED] Benchmark metrics saved to: {metrics_json_path.resolve()}")

    comparison_md_path = out_dir / "benchmark_comparison.md"
    with open(comparison_md_path, "w", encoding="utf-8") as f:
        f.write("# LexGLUE LEDGAR Benchmark Comparison\n\n")
        f.write(
            f"Evaluated Model: `{args.adapter_id}` on `{args.split}` set ({len(dataset)} samples)\n\n"
        )
        f.write(comparison_table_md)
        f.write("\n\n### Summary Metrics\n")
        f.write(f"- **Macro-F1**: `{macro_f1 * 100:.2f}%`\n")
        f.write(f"- **Micro-F1**: `{micro_f1 * 100:.2f}%`\n")
        f.write(f"- **Accuracy**: `{accuracy * 100:.2f}%`\n")
        f.write(f"- **Mean Latency**: `{latency_summary['mean_ms']:.1f} ms / clause`\n")
        f.write(
            f"- **Throughput**: `{latency_summary['throughput_samples_per_sec']:.2f} clauses / sec`\n"
        )
    print(f"[SAVED] Benchmark comparison table saved to: {comparison_md_path.resolve()}")


if __name__ == "__main__":
    main()
