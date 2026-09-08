"""Unit tests for WeightedSFTTrainer loss computation math."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from src.training.trainer import compute_weighted_completion_loss  # noqa: E402


def test_compute_weighted_completion_loss():
    batch_size = 2
    seq_len = 8
    vocab_size = 50
    num_classes = 10

    torch.manual_seed(42)

    # Random logits
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Labels: first 4 tokens are prompt (-100), next 4 are completion tokens
    labels = torch.tensor(
        [
            [-100, -100, -100, -100, 12, 15, 20, 2],
            [-100, -100, -100, -100, 5, 8, 2, -100],
        ]
    )

    class_ids = torch.tensor([0, 9])
    # Class 0 is frequent (weight 0.2), Class 9 is rare (weight 5.0)
    class_weights = torch.ones(num_classes)
    class_weights[0] = 0.2
    class_weights[9] = 5.0

    loss = compute_weighted_completion_loss(
        logits=logits,
        labels=labels,
        class_ids=class_ids,
        class_weights=class_weights,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, "Loss must be a scalar tensor"
    assert not torch.isnan(loss), "Loss must not be NaN"
    assert not torch.isinf(loss), "Loss must not be Inf"
    assert loss.item() > 0.0, "Cross-entropy loss must be positive"


def test_unweighted_fallback_loss():
    batch_size = 2
    seq_len = 6
    vocab_size = 30

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.tensor(
        [
            [-100, -100, 10, 12, 2, -100],
            [-100, -100, -100, 15, 2, -100],
        ]
    )

    loss = compute_weighted_completion_loss(
        logits=logits,
        labels=labels,
        class_ids=None,
        class_weights=None,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0.0
