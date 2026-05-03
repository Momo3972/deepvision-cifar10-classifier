"""
Confidence calibration: Expected Calibration Error and temperature scaling.

A model is *calibrated* if, among the predictions it issues with confidence
``p``, exactly a fraction ``p`` are correct. EfficientNet and most modern
neural networks are over-confident out of the box; temperature scaling is a
single-parameter post-hoc fix that divides the pre-softmax logits by a scalar
``T``, which softens the output distribution without changing the ``argmax``.

References
----------
- Guo et al. (2017): "On Calibration of Modern Neural Networks", ICML.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.optimize import minimize_scalar

from deepvision.utils.logging import get_logger

log = get_logger(__name__)


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE) of a probabilistic classifier.

    Parameters
    ----------
    probs
        Array of shape ``(n_samples, n_classes)`` containing softmax outputs.
    labels
        Integer ground-truth labels of shape ``(n_samples,)``.
    n_bins
        Number of equal-width confidence bins used to approximate the
        reliability diagram. 15 is the value used by Guo et al. (2017).

    Returns
    -------
    float
        ECE value in ``[0, 1]``. Lower is better. A perfectly calibrated
        classifier has ECE ≈ 0.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (n, k), got shape {probs.shape}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == np.asarray(labels).reshape(-1)).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = float(probs.shape[0])
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Right-inclusive on the last bin so confidence == 1.0 falls in it.
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        avg_confidence = float(confidences[mask].mean())
        avg_accuracy = float(accuracies[mask].mean())
        ece += (n_bin / n_total) * abs(avg_confidence - avg_accuracy)
    return ece


def reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> dict[str, list[float] | list[int]]:
    """Return arrays suitable for plotting a reliability diagram.

    Returns
    -------
    dict
        ``{"bin_centers": ..., "bin_accuracies": ..., "bin_confidences": ...,
        "bin_counts": ...}``. Useful for matplotlib bar charts and for
        attaching the diagram as an MLflow artifact.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == np.asarray(labels).reshape(-1)).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_accs, bin_confs, bin_counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        n_bin = int(mask.sum())
        bin_centers.append(float((lo + hi) / 2.0))
        bin_counts.append(n_bin)
        if n_bin == 0:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
        else:
            bin_accs.append(float(accuracies[mask].mean()))
            bin_confs.append(float(confidences[mask].mean()))
    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
    }


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> float:
    """Optimal temperature ``T`` that minimizes the cross-entropy on a held-out set.

    Parameters
    ----------
    logits
        Pre-softmax outputs of shape ``(n_samples, n_classes)``.
    labels
        Integer ground-truth labels of shape ``(n_samples,)``.
    bounds
        Search bounds for ``T``. ``T < 1`` sharpens, ``T > 1`` softens.

    Returns
    -------
    float
        Best ``T`` found via scalar minimization.
    """
    labels_int = np.asarray(labels).reshape(-1).astype(int)
    n = logits.shape[0]

    def nll(temperature: float) -> float:
        if temperature <= 0:
            return float("inf")
        scaled = logits / temperature
        # log-softmax for numerical stability.
        scaled -= scaled.max(axis=1, keepdims=True)
        log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return float(-log_probs[np.arange(n), labels_int].mean())

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    log.info("Fitted temperature T = %.4f (NLL = %.4f)", result.x, result.fun)
    return float(result.x)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return softmax probabilities of ``logits / temperature``."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return cast(np.ndarray, exp / exp.sum(axis=1, keepdims=True))
