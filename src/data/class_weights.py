"""
Balanced class weight calculation for handling severe class imbalance in LexGLUE/LEDGAR.

Formula:
    w_c = N / (C * N_c)
where:
    N   = total number of training samples (e.g. 60,000)
    C   = number of distinct classes (100)
    N_c = count of class c

Weights are normalized so mean(w) = 1.0, with optional upper-bound clipping
to ensure numerical stability during 4-bit QLoRA gradient accumulation.
"""

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

try:
    import torch
except ImportError:
    torch = None


def compute_balanced_class_weights(
    labels: Sequence[int] | np.ndarray,
    num_classes: int = 100,
    max_clip: float = 15.0,
    normalize: bool = True,
    device: Any = "cpu",
) -> Any:
    """
    Compute balanced inverse-frequency class weights for cross-entropy loss.

    Args:
        labels: Sequence of integer class IDs from the training split.
        num_classes: Total expected number of classes (100 for LEDGAR).
        max_clip: Maximum allowable weight value to prevent gradient explosions on tail classes.
        normalize: If True, scales weights so their mean equals 1.0.
        device: PyTorch target device ('cpu', 'cuda', etc.).

    Returns:
        torch.Tensor of shape (num_classes,) containing float32 weights.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    classes = np.arange(num_classes)

    # Compute standard sklearn balanced weights: N / (n_classes * count(c))
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_arr,
    ).astype(np.float32)

    # Apply clipping to upper bound if specified
    if max_clip is not None and max_clip > 0:
        raw_weights = np.clip(raw_weights, a_min=None, a_max=max_clip)

    # Normalize so mean weight is 1.0 (preserves effective learning rate scale)
    if normalize:
        mean_w = np.mean(raw_weights)
        if mean_w > 0:
            raw_weights = raw_weights / mean_w

    try:
        import torch

        return torch.tensor(raw_weights, dtype=torch.float32, device=device)
    except ImportError:
        return raw_weights


def get_class_frequencies(labels: Sequence[int], num_classes: int = 100) -> dict[int, int]:
    """Return frequency count dictionary for each class ID from 0 to num_classes - 1."""
    counts = Counter(labels)
    return {i: counts.get(i, 0) for i in range(num_classes)}
