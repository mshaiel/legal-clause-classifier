"""Unit tests for balanced inverse-frequency class weight computation."""

import numpy as np

from src.data.class_weights import compute_balanced_class_weights, get_class_frequencies


def test_balanced_class_weights_shape_and_mean():
    num_classes = 10
    # Skewed labels: class 0 has 1000 samples, class 9 has 10 samples
    labels = [0] * 1000 + [1] * 500 + [2] * 200 + [3] * 100 + [9] * 10
    # At least 1 sample for remaining classes
    for c in range(4, 9):
        labels.append(c)

    weights = compute_balanced_class_weights(
        labels=labels,
        num_classes=num_classes,
        max_clip=20.0,
        normalize=True,
    )

    weights_arr = np.asarray(weights)
    assert weights_arr.shape == (num_classes,)
    assert np.all(weights_arr > 0), "All weights must be strictly positive"

    # Rare class 9 must have significantly higher weight than majority class 0
    assert weights_arr[9] > weights_arr[0]

    # Mean normalized weight should be approximately 1.0
    assert np.isclose(float(weights_arr.mean()), 1.0, atol=1e-4)


def test_weights_clipping():
    num_classes = 5
    labels = [0] * 10000 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 1
    max_clip = 5.0

    weights = compute_balanced_class_weights(
        labels=labels,
        num_classes=num_classes,
        max_clip=max_clip,
        normalize=False,
    )

    weights_arr = np.asarray(weights)
    assert float(weights_arr.max()) <= max_clip + 1e-5


def test_get_class_frequencies():
    labels = [0, 0, 1, 2, 2, 2]
    freqs = get_class_frequencies(labels, num_classes=5)
    assert freqs[0] == 2
    assert freqs[1] == 1
    assert freqs[2] == 3
    assert freqs[3] == 0
    assert freqs[4] == 0
