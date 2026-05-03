"""
Inference latency benchmarking — p50 / p90 / p95 / p99.

Used to characterize how a model performs at serve-time on a given hardware,
and to compare runtimes (Keras vs ONNX Runtime vs TFLite INT8 — Phase 10).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from deepvision.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LatencyResult:
    """Results of a latency benchmark.

    All durations are in **milliseconds**.
    """

    n_iterations: int
    n_warmup: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_imgs_per_sec: float

    def as_dict(self) -> dict[str, float | int]:
        """JSON-friendly dict for MLflow logging."""
        return {
            "n_iterations": self.n_iterations,
            "n_warmup": self.n_warmup,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "throughput_imgs_per_sec": self.throughput_imgs_per_sec,
        }


def benchmark_callable(
    fn: Callable[[np.ndarray], object],
    sample_input: np.ndarray,
    *,
    n_iterations: int = 100,
    n_warmup: int = 10,
) -> LatencyResult:
    """Time a single-input prediction callable and return percentile latencies.

    Parameters
    ----------
    fn
        Callable taking a NumPy array (typically a single image) and producing
        a prediction. The return value is ignored.
    sample_input
        Pre-batched input array. Reused for every iteration to keep memory
        and cache hot — measures pure compute, not data loading.
    n_iterations
        Number of timed iterations. 100 is a sweet spot for stable percentiles.
    n_warmup
        Iterations to discard at the start (compiles XLA graphs, warms caches).

    Returns
    -------
    LatencyResult
        Aggregated latency statistics in milliseconds + throughput.
    """
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")
    if n_warmup < 0:
        raise ValueError(f"n_warmup must be >= 0, got {n_warmup}")

    # Warmup phase — discarded.
    for _ in range(n_warmup):
        fn(sample_input)

    durations_ms: list[float] = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        fn(sample_input)
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    arr = np.array(durations_ms)
    mean_ms = float(arr.mean())
    throughput = 1000.0 / mean_ms if mean_ms > 0 else float("inf")

    return LatencyResult(
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        mean_ms=mean_ms,
        median_ms=float(np.percentile(arr, 50)),
        p90_ms=float(np.percentile(arr, 90)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        throughput_imgs_per_sec=throughput,
    )


def benchmark_keras_model(
    model,
    image_shape: tuple[int, int, int] = (32, 32, 3),
    *,
    n_iterations: int = 100,
    n_warmup: int = 10,
    seed: int = 0,
) -> LatencyResult:
    """Convenience wrapper that benchmarks ``model.predict`` on a synthetic image.

    The synthetic image is generated with a fixed seed so latency runs are
    perfectly reproducible. Measurements isolate **inference compute**, not
    image preprocessing.
    """
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, 256, size=(1, *image_shape), dtype=np.uint8).astype(np.float32)

    def predict(x: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, model.predict(x, verbose=0))

    return benchmark_callable(
        predict,
        sample,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
    )
