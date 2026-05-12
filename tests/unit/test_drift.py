"""Unit tests for :mod:`deepvision.monitoring.drift`.

The drift module is pure NumPy / SciPy -- no TensorFlow needed -- so we
exercise the contract on synthetic distributions that make the expected
behaviour obvious:

* identical distributions -> distance ~ 0,
* shifted distributions -> distance > 0 and grows with the shift,
* shape-mismatched / empty / wrong-ndim inputs -> ``ValueError``.
"""

from __future__ import annotations

import numpy as np
import pytest

from deepvision.monitoring.drift import (
    drift_score,
    summarize_drift,
    wasserstein_drift,
)

# ---------------------------------------------------------------------------
# wasserstein_drift -- per-dimension distances
# ---------------------------------------------------------------------------


def test_identical_distributions_yield_zero_drift() -> None:
    rng = np.random.default_rng(seed=0)
    baseline = rng.normal(size=(500, 8))
    distances = wasserstein_drift(baseline, baseline.copy())
    assert distances.shape == (8,)
    np.testing.assert_allclose(distances, 0.0, atol=1e-12)


def test_shifted_distribution_yields_positive_drift() -> None:
    rng = np.random.default_rng(seed=1)
    baseline = rng.normal(size=(500, 4))
    current = baseline + 3.0  # uniform shift of 3 std devs
    distances = wasserstein_drift(baseline, current)
    # Wasserstein-1 of two N(mu, 1) Gaussians equals |mu_diff| in expectation.
    np.testing.assert_allclose(distances, 3.0, atol=0.5)


def test_drift_grows_monotonically_with_shift() -> None:
    rng = np.random.default_rng(seed=2)
    baseline = rng.normal(size=(400, 1))
    shifts = [0.0, 0.5, 1.0, 2.0, 5.0]
    means = [wasserstein_drift(baseline, baseline + shift).mean() for shift in shifts]
    assert means == sorted(means), f"Drift should grow with shift, got {means}"


def test_wasserstein_drift_rejects_dim_mismatch() -> None:
    baseline = np.zeros((10, 4))
    current = np.zeros((10, 5))
    with pytest.raises(ValueError, match=r"dimension mismatch|Feature dimension"):
        wasserstein_drift(baseline, current)


def test_wasserstein_drift_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        wasserstein_drift(np.zeros(10), np.zeros(10))


def test_wasserstein_drift_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match=r"at least one sample|empty"):
        wasserstein_drift(np.zeros((0, 4)), np.zeros((10, 4)))


# ---------------------------------------------------------------------------
# summarize_drift -- aggregation
# ---------------------------------------------------------------------------


def test_summarize_drift_returns_expected_keys() -> None:
    distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    summary = summarize_drift(distances)
    assert {"mean", "max", "p95", "num_dims"} <= summary.keys()


def test_summarize_drift_mean_max_correct() -> None:
    distances = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    summary = summarize_drift(distances)
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["max"] == pytest.approx(5.0)
    assert summary["num_dims"] == 5


def test_summarize_drift_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize_drift(np.array([]))


def test_summarize_drift_rejects_invalid_percentile() -> None:
    distances = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="percentile"):
        summarize_drift(distances, percentile=150.0)


# ---------------------------------------------------------------------------
# drift_score -- one-shot scalar shortcut used by the Prometheus exporter
# ---------------------------------------------------------------------------


def test_drift_score_returns_scalar() -> None:
    rng = np.random.default_rng(seed=3)
    baseline = rng.normal(size=(200, 16))
    score = drift_score(baseline, baseline.copy())
    assert isinstance(score, float)
    assert score == pytest.approx(0.0, abs=1e-10)


def test_drift_score_grows_with_shift() -> None:
    rng = np.random.default_rng(seed=4)
    baseline = rng.normal(size=(200, 16))
    s_low = drift_score(baseline, baseline + 0.1)
    s_high = drift_score(baseline, baseline + 2.0)
    assert s_low < s_high
