#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for LexGLUE/LEDGAR Dataset.

Analyzes:
- Dataset split dimensions (train, validation, test)
- 100-class label distribution and class imbalance
- Gini coefficient of class distribution inequality
- Sequence length distribution (words, chars, estimated tokens)
- Previews balanced inverse-frequency class weights
- Exports statistical summary to JSON
"""

import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight

# Ensure robust UTF-8 printing across Windows and Linux terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def compute_gini(frequencies: np.ndarray) -> float:
    """Calculate Gini coefficient of inequality for class frequencies."""
    sorted_freqs = np.sort(frequencies)
    n = len(frequencies)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_freqs)) / (n * np.sum(sorted_freqs)) - (n + 1) / n)


def run_eda(output_dir: str = "data"):
    print("=" * 70)
    print("[EDA] LEXGLUE / LEDGAR DATASET EXPLORATION & CLASS IMBALANCE AUDIT")
    print("=" * 70)

    print("\n[1/5] Loading 'coastalcph/lex_glue' (subset: 'ledgar')...")
    raw_dataset = load_dataset("coastalcph/lex_glue", "ledgar")

    splits = {split: len(raw_dataset[split]) for split in raw_dataset}
    print(f"Splits loaded: {splits}")

    label_names = raw_dataset["train"].features["label"].names
    num_classes = len(label_names)
    print(f"Total provision classes: {num_classes}")

    print("\n[2/5] Analyzing train set class frequencies...")
    train_labels = raw_dataset["train"]["label"]
    counter = Counter(train_labels)

    freq_array = np.array([counter[i] for i in range(num_classes)])
    max_count = int(np.max(freq_array))
    min_count = int(np.min(freq_array))
    median_count = float(np.median(freq_array))
    imbalance_ratio = max_count / max(1, min_count)
    gini_index = compute_gini(freq_array)

    print(f"  - Total training samples : {len(train_labels):,}")
    print(f"  - Most frequent class count : {max_count:,}")
    print(f"  - Least frequent class count: {min_count:,}")
    print(f"  - Median class count        : {median_count:,.1f}")
    print(f"  - Imbalance Ratio (Max/Min) : {imbalance_ratio:.2f}x")
    print(f"  - Gini Index of inequality  : {gini_index:.4f} (1.0 = absolute disparity)")

    # Sort classes by frequency
    sorted_indices = np.argsort(-freq_array)
    top_10 = [(label_names[idx], int(freq_array[idx])) for idx in sorted_indices[:10]]
    bottom_10 = [(label_names[idx], int(freq_array[idx])) for idx in sorted_indices[-10:]]

    print("\n Top 10 Majority Classes:")
    for rank, (name, count) in enumerate(top_10, 1):
        pct = (count / len(train_labels)) * 100
        print(f"   {rank:2d}. {name:<35} | {count:5,d} ({pct:5.2f}%)")

    print("\n Bottom 10 Minority (Tail) Classes:")
    for rank, (name, count) in enumerate(bottom_10, 1):
        pct = (count / len(train_labels)) * 100
        print(f"   {rank:2d}. {name:<35} | {count:5,d} ({pct:5.2f}%)")

    print("\n[3/5] Analyzing sequence lengths (word counts on sample of 5,000)...")
    sample_texts = raw_dataset["train"]["text"][:5000]
    word_lengths = [len(text.split()) for text in sample_texts]
    char_lengths = [len(text) for text in sample_texts]

    p50_w = np.percentile(word_lengths, 50)
    p90_w = np.percentile(word_lengths, 90)
    p95_w = np.percentile(word_lengths, 95)
    p99_w = np.percentile(word_lengths, 99)

    print(f"  - Word length: Median={p50_w:.0f}, P90={p90_w:.0f}, P95={p95_w:.0f}, P99={p99_w:.0f}")
    print(f"  - Character length: Median={np.percentile(char_lengths, 50):.0f}")
    print(
        f"  - Max Sequence Length 512 coverage: ~{np.mean(np.array(word_lengths) <= 400) * 100:.1f}% within token limits"
    )

    print("\n[4/5] Computing balanced inverse-frequency class weights...")
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels,
    )
    # Normalize weights so mean is 1.0
    weights_normalized = weights / np.mean(weights)
    print(
        f"  - Weight range: min={weights_normalized.min():.4f}, max={weights_normalized.max():.4f}, mean={weights_normalized.mean():.4f}"
    )

    print("\n[5/5] Exporting EDA summary to JSON...")
    os.makedirs(output_dir, exist_ok=True)
    summary_path = Path(output_dir) / "eda_summary.json"

    summary_data = {
        "dataset": "coastalcph/lex_glue/ledgar",
        "splits": splits,
        "num_classes": num_classes,
        "imbalance_metrics": {
            "max_count": max_count,
            "min_count": min_count,
            "median_count": median_count,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "gini_index": round(gini_index, 4),
        },
        "sequence_percentiles_words": {
            "p50": round(p50_w, 1),
            "p90": round(p90_w, 1),
            "p95": round(p95_w, 1),
            "p99": round(p99_w, 1),
        },
        "top_10_classes": [{"class": name, "count": cnt} for name, cnt in top_10],
        "bottom_10_classes": [{"class": name, "count": cnt} for name, cnt in bottom_10],
        "label_names": label_names,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[SUCCESS] Saved EDA summary to: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_eda()
