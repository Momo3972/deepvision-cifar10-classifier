"""
Prometheus instrumentation for the FastAPI serving layer.

The metrics exposed here are deliberately a small, stable set that maps onto
the four golden signals (Latency, Traffic, Errors, Saturation) and onto the
operational concerns of an inference service:

- :data:`INFERENCE_LATENCY` — histogram of end-to-end ``/predict`` latency.
- :data:`HTTP_REQUESTS` — counter of HTTP requests by route and status.
- :data:`HTTP_ERRORS` — counter of 4xx / 5xx responses.
- :data:`MODEL_LOADED` — 0/1 gauge flipped to 1 once the model is in memory.
- :data:`MODEL_INFO` — labels-only gauge that surfaces ``model_name`` and
  ``model_version`` in Grafana.

A dedicated :class:`prometheus_client.CollectorRegistry` (``REGISTRY``) is
used so test runs can be isolated and so importing this module does not
mutate the global default registry — important when the API is mounted
inside a larger application.
"""

from __future__ import annotations

from typing import Final

from prometheus_client import (
    CONTENT_TYPE_LATEST as _CONTENT_TYPE_LATEST,
)
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

#: Content-Type expected by Prometheus scrapers (re-exported for convenience).
CONTENT_TYPE_LATEST: Final[str] = _CONTENT_TYPE_LATEST

#: Latency buckets in seconds — covers <10 ms (warm GPU) up to 5 s (cold CPU).
DEFAULT_LATENCY_BUCKETS: Final[tuple[float, ...]] = (
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.250,
    0.500,
    1.000,
    2.500,
    5.000,
)


def build_registry() -> CollectorRegistry:
    """Build a fresh :class:`CollectorRegistry` populated with the project metrics.

    Returns a tuple-of-metrics layout that mirrors the module-level globals so
    that test code can spin a private registry per test and avoid double
    registration errors.
    """
    return CollectorRegistry()


# ---------------------------------------------------------------------------
# Default registry & metrics — used by the FastAPI app at runtime.
# ---------------------------------------------------------------------------

#: Shared registry used by the FastAPI app. Tests should not reuse this.
REGISTRY: Final[CollectorRegistry] = build_registry()

#: End-to-end latency of the ``/predict`` and ``/predict_batch`` endpoints.
INFERENCE_LATENCY: Final[Histogram] = Histogram(
    "deepvision_inference_latency_seconds",
    "End-to-end latency of the inference endpoints (decode + preprocess + forward).",
    labelnames=("endpoint", "model_name"),
    buckets=DEFAULT_LATENCY_BUCKETS,
    registry=REGISTRY,
)

#: Number of HTTP requests labeled by route, method and status code class.
HTTP_REQUESTS: Final[Counter] = Counter(
    "deepvision_http_requests_total",
    "Number of HTTP requests handled by the deepvision API.",
    labelnames=("method", "endpoint", "status_class"),
    registry=REGISTRY,
)

#: Errors counter (4xx + 5xx) — separated so dashboards can alert without
#: re-aggregating the requests counter.
HTTP_ERRORS: Final[Counter] = Counter(
    "deepvision_http_errors_total",
    "Number of 4xx/5xx responses returned by the deepvision API.",
    labelnames=("endpoint", "status_class", "error_code"),
    registry=REGISTRY,
)

#: 1 once the underlying Keras model has been loaded into memory, 0 otherwise.
MODEL_LOADED: Final[Gauge] = Gauge(
    "deepvision_model_loaded",
    "1 once the model is loaded and ready to serve, 0 otherwise.",
    registry=REGISTRY,
)

#: Static info gauge whose only purpose is to expose model_name / model_version
#: as labels — value is always 1. Mirrors ``mlflow_model_info`` conventions.
MODEL_INFO: Final[Gauge] = Gauge(
    "deepvision_model_info",
    "Static metadata about the served model exposed as labels.",
    labelnames=("model_name", "model_version"),
    registry=REGISTRY,
)


def status_class(status_code: int) -> str:
    """Bucket an HTTP status code into ``2xx`` / ``3xx`` / ``4xx`` / ``5xx``.

    Returns ``"???"`` for codes outside the 100-599 range so the metric
    cardinality stays bounded even on malformed responses.
    """
    if 100 <= status_code < 600:
        return f"{status_code // 100}xx"
    return "???"


def render_latest(registry: CollectorRegistry | None = None) -> bytes:
    """Render Prometheus exposition output for the given registry.

    Returns the bytes payload to use as the body of ``GET /metrics``.
    Defaults to the module-level :data:`REGISTRY`.
    """
    return generate_latest(registry if registry is not None else REGISTRY)


__all__ = [
    "CONTENT_TYPE_LATEST",
    "DEFAULT_LATENCY_BUCKETS",
    "HTTP_ERRORS",
    "HTTP_REQUESTS",
    "INFERENCE_LATENCY",
    "MODEL_INFO",
    "MODEL_LOADED",
    "REGISTRY",
    "build_registry",
    "render_latest",
    "status_class",
]
