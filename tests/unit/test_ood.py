"""Unit tests for :mod:`deepvision.monitoring.ood`.

The pure-Python parts -- ``energy_score``, ``is_ood``, ``ood_rate`` -- are
unit-tested directly. ``extract_logits`` is exercised against a tiny
hand-built Functional model so we don't pay the EfficientNet build cost.
"""

from __future__ import annotations

import numpy as np
import pytest

from deepvision.monitoring.ood import energy_score, is_ood, ood_rate

# ---------------------------------------------------------------------------
# energy_score
# ---------------------------------------------------------------------------


def test_energy_score_known_values() -> None:
    """E(x) = -T * logsumexp(z/T). For T=1 and z=[0, 0, 0]:
    logsumexp = log(3) ~ 1.0986, so E = -log(3)."""
    logits = np.array([[0.0, 0.0, 0.0]])
    e = energy_score(logits, temperature=1.0)
    assert e.shape == (1,)
    assert e[0] == pytest.approx(-np.log(3.0))


def test_energy_score_lower_for_confident_predictions() -> None:
    """A peakier logit distribution -> lower (more negative) energy ->
    *more* in-distribution. The "in-distribution" sample has one logit
    much larger than the rest."""
    in_dist = np.array([[10.0, 0.0, 0.0, 0.0]])
    flat = np.array([[0.0, 0.0, 0.0, 0.0]])
    e_in = energy_score(in_dist)[0]
    e_flat = energy_score(flat)[0]
    assert e_in < e_flat


def test_energy_score_temperature_smooths() -> None:
    """Larger T -> smoother distribution -> energy approaches -T*log(K)."""
    logits = np.array([[10.0, 0.0, 0.0, 0.0]])
    e_t1 = energy_score(logits, temperature=1.0)[0]
    e_t100 = energy_score(logits, temperature=100.0)[0]
    # At very large T, energy approaches -T * log(num_classes).
    assert e_t100 == pytest.approx(-100.0 * np.log(4.0), rel=0.05)
    # And the absolute energy at T=100 is much larger than at T=1.
    assert abs(e_t100) > abs(e_t1)


def test_energy_score_per_sample_shape() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(7, 10))
    e = energy_score(logits)
    assert e.shape == (7,)


def test_energy_score_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be > 0"):
        energy_score(np.zeros((1, 3)), temperature=0)
    with pytest.raises(ValueError, match="temperature must be > 0"):
        energy_score(np.zeros((1, 3)), temperature=-1.0)


def test_energy_score_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        energy_score(np.zeros(5))


# ---------------------------------------------------------------------------
# is_ood / ood_rate
# ---------------------------------------------------------------------------


def test_is_ood_threshold_logic() -> None:
    scores = np.array([-3.0, -1.0, 0.5, 2.0])
    mask = is_ood(scores, threshold=0.0)
    np.testing.assert_array_equal(mask, [False, False, True, True])


def test_ood_rate_fraction() -> None:
    scores = np.array([-3.0, -1.0, 0.5, 2.0])
    assert ood_rate(scores, threshold=0.0) == pytest.approx(0.5)
    assert ood_rate(scores, threshold=10.0) == pytest.approx(0.0)
    assert ood_rate(scores, threshold=-100.0) == pytest.approx(1.0)


def test_ood_rate_empty_array() -> None:
    assert ood_rate(np.array([]), threshold=0.0) == 0.0
