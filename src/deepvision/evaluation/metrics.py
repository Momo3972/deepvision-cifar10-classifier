"""
Model evaluation metrics for CIFAR-10 classification.

Provides a single :func:`evaluate_model` entry-point that returns a
JSON-friendly dictionary suitable for MLflow logging, plus helpers for
the underlying scikit-learn metrics. The interpretability metrics
(Grad-CAM, calibration ECE) are introduced in Phase 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from deepvision.constants import CLASS_NAMES_EN
from deepvision.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from tensorflow.keras import Model

log = get_logger(__name__)


def evaluate_model(
    model: Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    class_names: tuple[str, ...] = CLASS_NAMES_EN,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run inference on the test set and return a structured metrics dict.

    Parameters
    ----------
    model
        A trained Keras model with a softmax output of shape ``(batch, num_classes)``.
    x_test
        Test images as a NumPy array. Whether they should be normalized depends
        on the model: MLP/CNN expect ``[0, 1]`` floats, EfficientNet expects
        raw uint8.
    y_test
        Integer ground-truth labels of shape ``(N,)`` or ``(N, 1)``.
    class_names
        Names used for the classification report.
    batch_size
        Mini-batch size for ``model.predict``.

    Returns
    -------
    dict
        ``{"accuracy": float, "loss": float, "per_class_f1": {...},
        "classification_report": str, "confusion_matrix": list[list[int]]}``.
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
    )

    log.info("Predicting on %d test samples (batch_size=%d)…", x_test.shape[0], batch_size)
    y_pred_probs = model.predict(x_test, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.asarray(y_test).reshape(-1).astype(int)

    accuracy = float((y_pred == y_true).mean())
    loss = _compute_categorical_crossentropy(y_pred_probs, y_true)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    per_class_f1 = {cls: float(report_dict[cls]["f1-score"]) for cls in class_names}

    return {
        "accuracy": accuracy,
        "loss": loss,
        "per_class_f1": per_class_f1,
        "classification_report": report_text,
        "confusion_matrix": cm,
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
    }


def _compute_categorical_crossentropy(y_pred_probs: np.ndarray, y_true: np.ndarray) -> float:
    """Cross-entropy loss as a Python float, computed manually for safety."""
    eps = 1e-7
    clipped = np.clip(y_pred_probs, eps, 1.0 - eps)
    n = clipped.shape[0]
    return float(-np.log(clipped[np.arange(n), y_true]).mean())
