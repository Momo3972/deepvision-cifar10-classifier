"""Embedding drift detection -- Phase 8.

The monitoring service compares the distribution of the model's penultimate-layer
embeddings observed in production with the **baseline** distribution captured at
training time. A statistically significant shift signals that the input
distribution has moved away from what the model was trained on.

The audit (sections 7.2 and 9, Phase 8) prescribes either KL divergence or
the Wasserstein distance on the penultimate-layer features. We chose the
**1D Wasserstein distance computed independently per embedding dimension and
then aggregated**, because it is:

- symmetric (unlike KL),
- numerically stable on empty / sparse bins (unlike KL with an estimated PDF),
- analytically tractable on 1D samples via :func:`scipy.stats.wasserstein_distance`,
- bounded below by zero with a clear physical interpretation (the minimum
  amount of mass that must be moved to align the two distributions).

The functions in this module are pure NumPy and have no TensorFlow
dependency, so they can be unit-tested in milliseconds without loading a
model.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from scipy.stats import wasserstein_distance

#: Default percentile reported by :func:`summarize_drift`. The audit's alert
#: rule fires on a single scalar; we use the *mean* per-dimension distance as
#: the primary score and surface ``p95`` and ``max`` for diagnosability.
DEFAULT_PERCENTILE: Final[float] = 95.0


def wasserstein_drift(
    baseline: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    """Compute the 1D Wasserstein distance per embedding dimension.

    Parameters
    ----------
    baseline
        Reference embeddings, shape ``(N_b, D)``. Captured offline at
        training time and stored in :class:`deepvision.monitoring.baseline.Baseline`.
    current
        Live embeddings, shape ``(N_c, D)``. Both arrays must share the same
        feature dimension ``D`` (typically 1280 for EfficientNetB0).

    Returns
    -------
    np.ndarray
        Shape ``(D,)``. Each entry is the 1D Wasserstein distance between
        ``baseline[:, d]`` and ``current[:, d]``. Always non-negative.

    Raises
    ------
    ValueError
        If the two arrays have inconsistent shapes or contain no samples.
    """
    if baseline.ndim != 2 or current.ndim != 2:
        raise ValueError(
            f"Both arrays must be 2D (N, D); got "
            f"baseline.shape={baseline.shape}, current.shape={current.shape}."
        )
    if baseline.shape[1] != current.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: baseline has {baseline.shape[1]} dims, "
            f"current has {current.shape[1]} dims."
        )
    if baseline.shape[0] == 0 or current.shape[0] == 0:
        raise ValueError("Both arrays must contain at least one sample.")

    dim = baseline.shape[1]
    distances = np.empty(dim, dtype=np.float64)
    for d in range(dim):
        distances[d] = wasserstein_distance(baseline[:, d], current[:, d])
    return distances


def summarize_drift(
    per_dim_distances: np.ndarray,
    *,
    percentile: float = DEFAULT_PERCENTILE,
) -> dict[str, float]:
    """Aggregate per-dimension distances into a small set of headline numbers.

    Returns a dict with ``mean``, ``max``, ``p<percentile>`` and ``num_dims``.
    The ``mean`` is the canonical *drift score* exposed as a Prometheus
    gauge; ``max`` and the high percentile help operators investigate which
    feature dimensions diverged the most.

    Parameters
    ----------
    per_dim_distances
        Output of :func:`wasserstein_drift`, shape ``(D,)``.
    percentile
        Upper percentile to report. Defaults to ``95.0``.
    """
    if per_dim_distances.ndim != 1:
        raise ValueError(f"Expected a 1D array of distances, got shape {per_dim_distances.shape}.")
    if per_dim_distances.size == 0:
        raise ValueError("Cannot summarize an empty distance array.")
    if not 0.0 < percentile < 100.0:
        raise ValueError(f"percentile must be in (0, 100); got {percentile}.")

    return {
        "mean": float(per_dim_distances.mean()),
        "max": float(per_dim_distances.max()),
        f"p{int(percentile)}": float(np.percentile(per_dim_distances, percentile)),
        "num_dims": int(per_dim_distances.size),
    }


def drift_score(baseline: np.ndarray, current: np.ndarray) -> float:
    """One-shot helper returning the canonical scalar drift score.

    Equivalent to ``summarize_drift(wasserstein_drift(baseline, current))["mean"]``.
    Convenient for the Prometheus gauge update path in
    :mod:`deepvision.monitoring.server`.
    """
    return summarize_drift(wasserstein_drift(baseline, current))["mean"]
