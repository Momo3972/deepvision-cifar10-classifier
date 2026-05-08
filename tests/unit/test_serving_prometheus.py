"""Unit tests for ``deepvision.serving.prometheus``."""

from __future__ import annotations

import pytest
from prometheus_client import Counter, Gauge, Histogram

from deepvision.serving.prometheus import (
    DEFAULT_LATENCY_BUCKETS,
    HTTP_ERRORS,
    HTTP_REQUESTS,
    INFERENCE_LATENCY,
    MODEL_INFO,
    MODEL_LOADED,
    REGISTRY,
    build_registry,
    render_latest,
    status_class,
)


def test_default_buckets_strictly_ascending() -> None:
    assert list(DEFAULT_LATENCY_BUCKETS) == sorted(set(DEFAULT_LATENCY_BUCKETS))
    assert DEFAULT_LATENCY_BUCKETS[0] > 0


def test_metrics_are_correct_types() -> None:
    assert isinstance(INFERENCE_LATENCY, Histogram)
    assert isinstance(HTTP_REQUESTS, Counter)
    assert isinstance(HTTP_ERRORS, Counter)
    assert isinstance(MODEL_LOADED, Gauge)
    assert isinstance(MODEL_INFO, Gauge)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (100, "1xx"),
        (200, "2xx"),
        (302, "3xx"),
        (404, "4xx"),
        (599, "5xx"),
        (0, "???"),
        (999, "???"),
    ],
)
def test_status_class(code: int, expected: str) -> None:
    assert status_class(code) == expected


def test_build_registry_returns_independent_collector_registry() -> None:
    r1 = build_registry()
    r2 = build_registry()
    assert r1 is not r2


def test_render_latest_returns_bytes_with_metric_names() -> None:
    body = render_latest(REGISTRY)
    assert isinstance(body, bytes)
    text = body.decode()
    # Each metric should be visible in the exposition output.
    assert "deepvision_inference_latency_seconds" in text
    assert "deepvision_http_requests_total" in text
    assert "deepvision_model_loaded" in text


def test_counters_increment() -> None:
    before = HTTP_REQUESTS.labels(method="GET", endpoint="/health", status_class="2xx")._value.get()
    HTTP_REQUESTS.labels(method="GET", endpoint="/health", status_class="2xx").inc()
    after = HTTP_REQUESTS.labels(method="GET", endpoint="/health", status_class="2xx")._value.get()
    assert after == before + 1
