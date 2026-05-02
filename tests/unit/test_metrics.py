"""Tests for :mod:`deepvision.evaluation.metrics`."""

from __future__ import annotations

import numpy as np
import pytest

from deepvision.evaluation.metrics import _compute_categorical_crossentropy


def test_perfect_prediction_yields_near_zero_loss() -> None:
    """A model that always assigns ~1.0 to the true class has near-zero loss."""
    n, num_classes = 50, 10
    y_true = np.random.randint(0, num_classes, size=n)
    y_pred = np.full((n, num_classes), 1e-7)
    y_pred[np.arange(n), y_true] = 1.0 - 1e-7 * (num_classes - 1)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    loss = _compute_categorical_crossentropy(y_pred, y_true)
    assert loss < 1e-3


def test_uniform_prediction_yields_log_num_classes_loss() -> None:
    """A uniform softmax over K classes has loss ~= log(K) regardless of labels."""
    n, num_classes = 100, 10
    y_true = np.random.randint(0, num_classes, size=n)
    y_pred = np.full((n, num_classes), 1.0 / num_classes)
    loss = _compute_categorical_crossentropy(y_pred, y_true)
    # Use approx because float64 arithmetic over an N-element mean introduces
    # rounding error vs the closed-form log(K).
    assert loss == pytest.approx(float(np.log(num_classes)))


def test_evaluate_model_smoke() -> None:
    """Build a dummy probability matrix and verify the metrics dict shape."""

    # Use a fake "model" via a tiny wrapper exposing predict()
    class _FakeModel:
        def predict(self, x: np.ndarray, batch_size: int = 64, verbose: int = 0) -> np.ndarray:
            del batch_size, verbose  # signature must match Keras Model.predict
            n = x.shape[0]
            probs = np.full((n, 10), 0.05)
            probs[np.arange(n), 0] = 0.55  # always predict class 0
            return probs

    from deepvision.evaluation.metrics import evaluate_model

    # Cover all 10 classes so sklearn's classification_report doesn't complain
    # about a target_names size mismatch.
    n_per_class = 5
    x = np.zeros((10 * n_per_class, 32, 32, 3), dtype=np.float32)
    y = np.repeat(np.arange(10), n_per_class)
    metrics = evaluate_model(_FakeModel(), x, y)  # type: ignore[arg-type]

    assert "accuracy" in metrics
    assert "loss" in metrics
    assert "per_class_f1" in metrics
    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics
    # The fake model always predicts class 0, so only the n_per_class samples
    # of class 0 are correctly classified.
    assert metrics["accuracy"] == n_per_class / (10 * n_per_class)
