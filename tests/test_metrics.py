"""Unit tests for metric calculations and text output cleaning."""

from src.model.metrics import (
    clean_generated_label,
    compute_classification_metrics,
    map_text_to_label_id,
)


def test_clean_generated_label():
    raw_1 = "<|assistant|>\nGoverning Laws<|end|>"
    assert clean_generated_label(raw_1) == "Governing Laws"

    raw_2 = '  "Indemnifications"  \nExtra text'
    assert clean_generated_label(raw_2) == "Indemnifications"

    raw_3 = "```\nTermination\n```"
    assert clean_generated_label(raw_3) == "Termination"


def test_map_text_to_label_id():
    # Exact match
    id_exact = map_text_to_label_id("Governing Laws")
    assert id_exact == 43  # ID of 'Governing Laws' in LEDGAR

    # Case insensitive
    id_lower = map_text_to_label_id("governing laws")
    assert id_lower == id_exact

    # Fuzzy match close typo
    id_fuzzy = map_text_to_label_id("Severabiliti")
    assert id_fuzzy == 78  # 'Severability'


def test_compute_classification_metrics():
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 0]  # 4 out of 5 correct

    metrics = compute_classification_metrics(y_true, y_pred)

    assert "f1_macro" in metrics
    assert "f1_micro" in metrics
    assert "f1_weighted" in metrics
    assert "accuracy" in metrics

    assert metrics["accuracy"] == 80.0
    assert 0.0 <= metrics["f1_macro"] <= 100.0
