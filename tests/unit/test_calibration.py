"""Tests for :mod:`deepvision.evaluation.calibration`."""

from __future__ import annotations

import numpy as np
import pytest

from deepvision.evaluation.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_diagram_data,
)


def _perfectly_calibrated_predictions(
    n: int = 1000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic perfectly-calibrated dataset.

    For each of n samples we draw a confidence ``p`` uniformly in (0, 1), then
    decide whether the prediction is correct with probability ``p``. The two
    classes are 0 (predicted) and 1 (correct/incorrect).
    """
    rng = np.random.default_rng(seed)
    confidences = rng.uniform(0.0, 1.0, size=n)
    is_correct = rng.uniform(0.0, 1.0, size=n) < confidences
    # Build 2-class probabilities: argmax = 0 always; correct iff label = 0.
    probs = np.column_stack([confidences, 1.0 - confidences])
    labels = np.where(is_correct, 0, 1)
    return probs, labels


def test_perfectly_calibrated_has_low_ece() -> None:
    probs, labels = _perfectly_calibrated_predictions(n=10_000, seed=42)
    ece = expected_calibration_error(probs, labels, n_bins=15)
    assert ece < 0.05


def test_overconfident_classifier_has_high_ece() -> None:
    """A classifier always saying 99 percent confidence but wrong half the time
    should have ECE close to 0.5."""
    n = 1000
    probs = np.full((n, 2), 0.01)
    probs[:, 0] = 0.99
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    ece = expected_calibration_error(probs, labels)
    assert 0.4 < ece < 0.6


def test_ece_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="probs must be 2D"):
        expected_calibration_error(np.array([0.5, 0.5]), np.array([0]))
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error(np.array([[0.5, 0.5]]), np.array([0]), n_bins=1)


def test_reliability_diagram_keys() -> None:
    probs, labels = _perfectly_calibrated_predictions(n=200)
    diagram = reliability_diagram_data(probs, labels, n_bins=10)
    assert set(diagram) == {"bin_centers", "bin_accuracies", "bin_confidences", "bin_counts"}
    assert len(diagram["bin_centers"]) == 10
    assert sum(diagram["bin_counts"]) == probs.shape[0]


def test_apply_temperature_changes_distribution_but_not_argmax() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(100, 10))
    softmax_t1 = apply_temperature(logits, temperature=1.0)
    softmax_t10 = apply_temperature(logits, temperature=10.0)
    np.testing.assert_array_equal(softmax_t1.argmax(axis=1), softmax_t10.argmax(axis=1))
    assert softmax_t10.max(axis=1).mean() < softmax_t1.max(axis=1).mean()


def test_apply_temperature_rejects_zero_or_negative() -> None:
    logits = np.zeros((1, 10))
    with pytest.raises(ValueError, match="temperature must be > 0"):
        apply_temperature(logits, temperature=0.0)
    with pytest.raises(ValueError, match="temperature must be > 0"):
        apply_temperature(logits, temperature=-1.0)


def test_fit_temperature_returns_finite_value_within_bounds() -> None:
    """``fit_temperature`` must return a positive finite scalar within bounds."""
    rng = np.random.default_rng(1)
    n, n_classes = 5_000, 10
    labels = rng.integers(0, n_classes, size=n)
    logits = rng.normal(0.0, 1.0, size=(n, n_classes))
    logits[np.arange(n), labels] += 4.0
    t = fit_temperature(logits, labels)
    assert np.isfinite(t)
    assert 0.05 <= t <= 10.0


def test_fit_temperature_does_not_increase_loss() -> None:
    """The fitted temperature should not be worse than T=1 in NLL."""
    rng = np.random.default_rng(2)
    n, n_classes = 1_000, 10
    labels = rng.integers(0, n_classes, size=n)
    logits = rng.normal(0.0, 0.5, size=(n, n_classes))
    logits[np.arange(n), labels] += 8.0

    def _cross_entropy(t: float) -> float:
        scaled = logits / t
        scaled -= scaled.max(axis=1, keepdims=True)
        log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return float(-log_probs[np.arange(n), labels].mean())

    t_fit = fit_temperature(logits, labels)
    assert _cross_entropy(t_fit) <= _cross_entropy(1.0) + 1e-6
