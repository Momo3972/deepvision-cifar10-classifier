"""Tests for :mod:`deepvision.evaluation.benchmark`."""

from __future__ import annotations

import time

import numpy as np
import pytest

from deepvision.evaluation.benchmark import (
    LatencyResult,
    benchmark_callable,
)


def test_latency_result_fields_and_dict() -> None:
    result = LatencyResult(
        n_iterations=10,
        n_warmup=2,
        mean_ms=1.0,
        median_ms=1.0,
        p90_ms=1.2,
        p95_ms=1.3,
        p99_ms=1.5,
        min_ms=0.8,
        max_ms=2.0,
        throughput_imgs_per_sec=1000.0,
    )
    payload = result.as_dict()
    assert payload["n_iterations"] == 10
    assert payload["mean_ms"] == 1.0
    assert payload["throughput_imgs_per_sec"] == 1000.0


def test_benchmark_callable_returns_consistent_percentiles() -> None:
    """Benchmark a callable that sleeps for a fixed delay."""
    delay_ms = 5

    def fake_predict(_x: np.ndarray) -> None:
        time.sleep(delay_ms / 1000.0)

    sample = np.zeros((1, 32, 32, 3), dtype=np.float32)
    result = benchmark_callable(fake_predict, sample, n_iterations=20, n_warmup=2)

    # Each iteration takes approximately 5ms; allow a generous upper bound for
    # OS scheduling jitter.
    assert 4.0 < result.mean_ms < 30.0
    assert result.min_ms > 0
    assert result.max_ms >= result.mean_ms
    # Percentile ordering invariant.
    assert result.median_ms <= result.p90_ms <= result.p95_ms <= result.p99_ms


def test_benchmark_callable_rejects_zero_iterations() -> None:
    def noop(_x: np.ndarray) -> None:
        return None

    with pytest.raises(ValueError, match="n_iterations"):
        benchmark_callable(noop, np.zeros(1), n_iterations=0)


def test_benchmark_callable_rejects_negative_warmup() -> None:
    def noop(_x: np.ndarray) -> None:
        return None

    with pytest.raises(ValueError, match="n_warmup"):
        benchmark_callable(noop, np.zeros(1), n_iterations=5, n_warmup=-1)


def test_benchmark_callable_throughput_is_inverse_of_mean_latency() -> None:
    """throughput_imgs_per_sec should equal 1000 / mean_ms."""

    def quick(_x: np.ndarray) -> None:
        time.sleep(0.001)

    sample = np.zeros(1)
    result = benchmark_callable(quick, sample, n_iterations=10, n_warmup=1)
    expected_tps = 1000.0 / result.mean_ms
    assert abs(result.throughput_imgs_per_sec - expected_tps) < 1e-6
