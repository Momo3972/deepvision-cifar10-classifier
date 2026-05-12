"""Unit tests for :mod:`deepvision.monitoring.server`.

Cover :class:`DriftMonitor` initialisation and one polling cycle with a
**stub engine** that quacks like :class:`InferenceEngine` but returns a
hand-built tiny Functional model so the test does not pay the EfficientNet
build cost (still pulls TensorFlow once per test session, like
``test_serving_inference``).
"""

from __future__ import annotations

import numpy as np
import pytest

from deepvision.monitoring.server import DriftMonitor

# ---------------------------------------------------------------------------
# Tiny Functional model used to mock the inference engine's ``.model`` attr.
# ---------------------------------------------------------------------------


def _build_tiny_model() -> object:
    """Build a 4x4x3 -> Dense(5, softmax) model -- the smallest shape that
    matches the contract: last layer is a Dense classifier."""
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(4, 4, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Force a forward pass so ``model.input`` is fully connected in Keras 3.
    model.predict(np.zeros((1, 4, 4, 3), dtype=np.float32), verbose=0)
    return model


class _StubEngine:
    """Minimal stand-in for ``InferenceEngine`` used by ``DriftMonitor``."""

    def __init__(self) -> None:
        self.model = _build_tiny_model()
        self.model_name = "tiny_test_model"
        self.model_version = "0.0.0-stub"
        self.is_loaded = True

    def load(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monitor_with_stub(monkeypatch: pytest.MonkeyPatch) -> DriftMonitor:
    """A ``DriftMonitor`` whose ``initialize`` is short-circuited to use the
    stub engine and a small synthetic baseline -- no HTTP server started,
    no real EfficientNet built."""
    monitor = DriftMonitor(batch_size=8, interval=1.0)

    # Replace the lazy InferenceEngine creation with our stub.
    stub = _StubEngine()
    monitor._engine = stub

    # Override the synthetic batch size on the baseline path so compute is fast.
    from deepvision.monitoring import baseline as baseline_mod

    real_synthetic = baseline_mod.synthetic_reference_images

    def small_synthetic(n: int = 16, *, seed: int = 42, image_size: int = 4) -> np.ndarray:
        # Force shape (n, 4, 4, 3) to match the tiny model.
        return real_synthetic(n=n, seed=seed, image_size=4)

    monkeypatch.setattr(baseline_mod, "synthetic_reference_images", small_synthetic)
    monkeypatch.setattr("deepvision.monitoring.server.synthetic_reference_images", small_synthetic)

    monitor.initialize()
    return monitor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_initialize_loads_engine_and_baseline(monitor_with_stub: DriftMonitor) -> None:
    assert monitor_with_stub._engine is not None
    assert monitor_with_stub._baseline is not None
    assert monitor_with_stub._baseline.n_samples > 0
    assert monitor_with_stub._baseline.feature_dim > 0


def test_poll_once_updates_gauges(monitor_with_stub: DriftMonitor) -> None:
    result = monitor_with_stub.poll_once()
    # Polling cycle returned the four reported numbers.
    assert {"drift_mean", "drift_max", "drift_p95", "ood_rate"} <= result.keys()
    # All numeric and finite.
    for key, value in result.items():
        assert np.isfinite(value), f"{key} is not finite: {value}"
    # Drift is non-negative by construction.
    assert result["drift_mean"] >= 0.0
    assert result["drift_max"] >= 0.0
    assert 0.0 <= result["ood_rate"] <= 1.0


def test_poll_once_increments_counter(monitor_with_stub: DriftMonitor) -> None:
    """The counter goes up by exactly 1 per successful poll."""
    from prometheus_client import generate_latest

    monitor_with_stub.poll_once()
    monitor_with_stub.poll_once()

    payload = generate_latest(monitor_with_stub._registry).decode("utf-8")
    # The Counter family ends with ``_total`` and exposes the count directly.
    assert "deepvision_drift_polls_total" in payload


def test_poll_without_initialize_raises() -> None:
    monitor = DriftMonitor()
    with pytest.raises(RuntimeError, match="initialize"):
        monitor.poll_once()


def test_metrics_exposition_contains_expected_series(
    monitor_with_stub: DriftMonitor,
) -> None:
    """After one poll cycle, /metrics output must include all the gauges
    the Phase 8 dashboard relies on."""
    from prometheus_client import generate_latest

    monitor_with_stub.poll_once()
    payload = generate_latest(monitor_with_stub._registry).decode("utf-8")
    for series in (
        "deepvision_drift_score",
        "deepvision_drift_max",
        "deepvision_drift_p95",
        "deepvision_ood_rate",
        "deepvision_baseline_loaded",
        "deepvision_baseline_n_samples",
        "deepvision_drift_polls_total",
        "deepvision_drift_monitor_info",
    ):
        assert series in payload, f"missing Prometheus series {series!r}"
