"""
Evaluation metrics and label parsing for legal provision sequence classification.

Provides:
- Output cleaning and token/string normalization
- Exact & fuzzy label matcher mapping model generation to the 100 LEDGAR categories
- Metric calculations: Macro-F1, Micro-F1, Weighted-F1, Accuracy
"""

import difflib
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.data.dataset import LABEL2ID, LEDGAR_PROVISION_CLASSES


def clean_generated_label(raw_text: str) -> str:
    """
    Clean the model's generated text by stripping control tokens and whitespace.

    Extracts the candidate category name from responses such as:
    '<|assistant|>\nGoverning Laws<|end|>' -> 'Governing Laws'
    """
    text = raw_text.strip()

    # Strip conversational tags if present
    for token in ["<|assistant|>", "<|end|>", "<s>", "</s>", "<|user|>", "<|system|>"]:
        text = text.replace(token, "")

    # Extract first non-empty content line, skipping markdown code fence lines
    for line in text.splitlines():
        line_s = line.strip()
        if line_s.startswith("```"):
            continue
        line_s = line_s.strip("\"'`").strip()
        if line_s:
            return line_s

    return ""


def map_text_to_label_id(
    pred_text: str,
    candidate_labels: list[str] | None = None,
    default_id: int = 0,
) -> int:
    """
    Map arbitrary predicted text to the closest valid LEDGAR class integer ID (0-99).

    Uses:
    1. Exact match (case-insensitive)
    2. Substring match
    3. Difflib close-match fallback
    """
    if candidate_labels is None:
        candidate_labels = LEDGAR_PROVISION_CLASSES

    cleaned = clean_generated_label(pred_text)
    if not cleaned:
        return default_id

    cleaned_lower = cleaned.lower()

    # 1. Exact case-insensitive match
    for idx, name in enumerate(candidate_labels):
        if name.lower() == cleaned_lower:
            return idx

    # 2. Substring containment match
    for idx, name in enumerate(candidate_labels):
        if name.lower() in cleaned_lower or cleaned_lower in name.lower():
            return idx

    # 3. Closest fuzzy match via difflib
    matches = difflib.get_close_matches(cleaned, candidate_labels, n=1, cutoff=0.4)
    if matches:
        return LABEL2ID.get(matches[0], default_id)

    return default_id


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """
    Compute comprehensive classification metrics across LEDGAR provisions.

    Returns:
        Dict containing:
        - f1_macro: Macro-averaged F1 (primary LexGLUE benchmark metric)
        - f1_micro: Micro-averaged F1
        - f1_weighted: Class-frequency weighted F1
        - accuracy: Overall accuracy percentage
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)

    macro = f1_score(y_t, y_p, average="macro", zero_division=0) * 100
    micro = f1_score(y_t, y_p, average="micro", zero_division=0) * 100
    weighted = f1_score(y_t, y_p, average="weighted", zero_division=0) * 100
    acc = accuracy_score(y_t, y_p) * 100

    return {
        "f1_macro": round(float(macro), 2),
        "f1_micro": round(float(micro), 2),
        "f1_weighted": round(float(weighted), 2),
        "accuracy": round(float(acc), 2),
    }


def evaluate_generated_predictions(
    y_true_ids: Sequence[int],
    generated_texts: Sequence[str],
) -> dict[str, float]:
    """Evaluate a batch of generated text outputs against ground truth integer class IDs."""
    predicted_ids = [map_text_to_label_id(t) for t in generated_texts]
    metrics = compute_classification_metrics(y_true_ids, predicted_ids)
    return metrics
