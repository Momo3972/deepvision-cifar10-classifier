"""Multi-runtime latency benchmark -- Phase 10.

The audit (sections 7.2 and 9, Phase 10) prescribes a side-by-side
latency comparison of the model across the runtimes we export to. This
module provides a tiny framework that:

- abstracts each runtime behind the :class:`Runner` protocol so adding a
  fifth backend (TensorRT, OpenVINO, ...) is a 30-line affair;
- runs a configurable warmup pass followed by ``n_iter`` measured
  iterations per batch size;
- aggregates the timings into :class:`BenchmarkResult` records carrying
  p50 / p90 / p95 / p99 / mean / std / throughput, ready for
  serialization to pandas / MLflow / CHANGELOG tables.

Why these percentiles
=====================
p50 captures typical latency, p95 captures the user-visible tail, and
p99 captures the rare worst case that paging / GC pauses introduce.
Reporting only the mean would hide the fact that GIL pauses in Python
or PyArrow garbage collection can multiply p99 by 10x without budging
the mean -- a known trap when benchmarking model servers.

Why we time `runner.predict` and not the full request path
==========================================================
The benchmark measures the *kernel* of inference -- forward pass time
on a pre-allocated batch. Pre/post-processing (image decoding, softmax
post-treatment) is the same across runtimes and is benchmarked
separately in the FastAPI app's Prometheus histograms (see
``deepvision.serving.prometheus``).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np

from deepvision.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

log = get_logger(__name__)

#: Default warmup pass count. 100 iterations is more than enough to
#: prime the JIT caches of every runtime we benchmark and reach the
#: steady-state regime where measurement variance shrinks.
DEFAULT_WARMUP: Final[int] = 100

#: Default measurement pass count. 1000 iterations gives stable p99
#: estimates without exploding the wall-clock budget (~2 min total on
#: CPU across our four runtimes and three batch sizes).
DEFAULT_ITERATIONS: Final[int] = 1000

#: Default batch sizes. ``1`` and ``8`` cover the interactive serving
#: regime (per-image latency), ``32`` covers the batched training /
#: offline scoring regime where throughput matters more.
DEFAULT_BATCH_SIZES: Final[tuple[int, ...]] = (1, 8, 32)

#: Default input shape of a single sample (CIFAR-10 dimensions). Used
#: when no explicit ``input_shape`` is passed to :class:`LatencyBenchmark`.
DEFAULT_INPUT_SHAPE: Final[tuple[int, ...]] = (32, 32, 3)


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """A single ``(runtime, batch_size)`` latency measurement."""

    runtime: str
    batch_size: int
    n_iterations: int
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    throughput_ips: float
    """Images per second = ``batch_size / mean_seconds``."""

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-friendly dict suitable for ``pd.DataFrame``."""
        return {
            "runtime": self.runtime,
            "batch_size": self.batch_size,
            "n_iterations": self.n_iterations,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "throughput_ips": self.throughput_ips,
        }


# ---------------------------------------------------------------------------
# Runner protocol + concrete backends
# ---------------------------------------------------------------------------


class Runner(Protocol):
    """Minimal interface every benchmark backend must satisfy.

    The protocol is intentionally tiny: ``load`` materialises the model
    in memory, ``predict`` runs one forward pass on a pre-allocated
    NumPy batch, and ``close`` releases any handles. The benchmark
    framework owns the lifecycle (load -> warmup -> measure -> close)
    so backends only carry per-call state.
    """

    name: str

    def load(self) -> None: ...

    def predict(self, batch: np.ndarray) -> np.ndarray: ...

    def close(self) -> None: ...


class KerasRunner:
    """Runner that calls ``model.predict()`` on a ``.keras`` artefact."""

    name = "keras"

    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path)
        # Typed ``Any`` because the concrete object (``tf.keras.Model``)
        # comes from a third-party C extension whose stubs we deliberately
        # do not pin; the runtime contract is enforced by the load-before-
        # predict ``assert`` and by the unit tests.
        self._model: Any = None

    def load(self) -> None:
        import tensorflow as tf

        log.info("[%s] loading model from %s", self.name, self.model_path)
        self._model = tf.keras.models.load_model(str(self.model_path))

    def predict(self, batch: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Call load() before predict()."
        return np.asarray(self._model.predict(batch, verbose=0))

    def close(self) -> None:
        self._model = None


class TFSavedModelRunner:
    """Runner that calls a SavedModel's serving signature.

    Faster than :class:`KerasRunner` because it bypasses Keras's Python-side
    bookkeeping (callbacks, metrics, ...) and goes straight into the
    traced graph.

    Signature discovery
    -------------------
    The runner accepts SavedModels produced by either:

    - ``tf.saved_model.save(model, dir)`` -- legacy TF path, signature key
      is ``"serving_default"``,
    - ``model.export(dir)`` -- Keras 3 path, signature key is ``"serve"``.

    We pick the first signature available in the order ``"serve"``,
    ``"serving_default"``, then fall back to whichever signature key the
    SavedModel exposes first. Falling back rather than raising means the
    runner stays useful on hand-crafted SavedModels with custom
    signature names.
    """

    name = "tf_savedmodel"

    #: Signature keys we probe, in priority order. Keras 3's ``model.export``
    #: writes the first one; legacy ``tf.saved_model.save`` writes the second.
    _SIGNATURE_KEY_PRIORITY: tuple[str, ...] = ("serve", "serving_default")

    def __init__(self, savedmodel_dir: Path | str) -> None:
        self.savedmodel_dir = Path(savedmodel_dir)
        self._infer: Any = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._tf: Any = None

    def load(self) -> None:
        import tensorflow as tf

        log.info("[%s] loading SavedModel from %s", self.name, self.savedmodel_dir)
        loaded = tf.saved_model.load(str(self.savedmodel_dir))
        signature_key = self._select_signature_key(loaded.signatures.keys())
        log.info("[%s] using signature key %r", self.name, signature_key)
        self._infer = loaded.signatures[signature_key]
        # ``structured_input_signature`` is (args, kwargs); we want the kwargs dict.
        kwargs_sig = self._infer.structured_input_signature[1]
        self._input_name = next(iter(kwargs_sig.keys()))
        self._output_name = next(iter(self._infer.structured_outputs.keys()))
        self._tf = tf

    @classmethod
    def _select_signature_key(cls, available: Any) -> str:
        """Pick the best signature key from those exposed by the SavedModel.

        Raises :class:`ValueError` if the SavedModel exposes no signature at
        all -- which would indicate a corrupt artefact rather than a
        configuration issue.
        """
        keys = list(available)
        for preferred in cls._SIGNATURE_KEY_PRIORITY:
            if preferred in keys:
                return preferred
        if not keys:
            raise ValueError("SavedModel exposes no serving signature; cannot run inference.")
        # ``keys[0]`` is ``Any`` because ``available`` is ``Any``; cast to
        # ``str`` so mypy can prove the return type, and so a future SavedModel
        # that returns non-string keys would fail loudly at this very line.
        return str(keys[0])

    def predict(self, batch: np.ndarray) -> np.ndarray:
        assert self._infer is not None, "Call load() before predict()."
        assert self._tf is not None
        result = self._infer(**{self._input_name: self._tf.constant(batch)})
        return np.asarray(result[self._output_name].numpy())

    def close(self) -> None:
        self._infer = None
        self._tf = None


class OnnxRuntimeRunner:
    """Runner that delegates to ONNX Runtime on CPU."""

    name = "onnx_runtime"

    def __init__(self, onnx_path: Path | str) -> None:
        self.onnx_path = Path(onnx_path)
        self._sess: Any = None
        self._input_name: str | None = None

    def load(self) -> None:
        import onnxruntime as ort

        log.info("[%s] loading ONNX model from %s", self.name, self.onnx_path)
        # Use a deterministic single-thread session so the benchmark is
        # reproducible across machines and CI runners.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._sess.get_inputs()[0].name

    def predict(self, batch: np.ndarray) -> np.ndarray:
        assert self._sess is not None, "Call load() before predict()."
        # ONNX Runtime is strict on dtype; cast defensively.
        batch_f32 = batch if batch.dtype == np.float32 else batch.astype(np.float32)
        outputs = self._sess.run(None, {self._input_name: batch_f32})
        return np.asarray(outputs[0])

    def close(self) -> None:
        self._sess = None


class TFLiteRunner:
    """Runner that delegates to the TFLite interpreter.

    Handles variable batch sizes by re-allocating tensors on every shape
    change. The first ``predict()`` after a resize pays an allocation
    cost which the warmup pass absorbs.
    """

    name = "tflite"

    def __init__(self, tflite_path: Path | str) -> None:
        self.tflite_path = Path(tflite_path)
        self._interp: Any = None
        self._input_index: int | None = None
        self._output_index: int | None = None
        self._input_dtype: np.dtype | None = None
        self._last_shape: tuple[int, ...] | None = None

    def load(self) -> None:
        import tensorflow as tf

        log.info("[%s] loading TFLite model from %s", self.name, self.tflite_path)
        self._interp = tf.lite.Interpreter(model_path=str(self.tflite_path))
        self._interp.allocate_tensors()
        in_details = self._interp.get_input_details()[0]
        out_details = self._interp.get_output_details()[0]
        self._input_index = in_details["index"]
        self._output_index = out_details["index"]
        self._input_dtype = np.dtype(in_details["dtype"])
        self._last_shape = tuple(int(d) for d in in_details["shape"])

    def predict(self, batch: np.ndarray) -> np.ndarray:
        assert self._interp is not None, "Call load() before predict()."
        if batch.shape != self._last_shape:
            # Variable batch size -- ask the interpreter to re-allocate.
            self._interp.resize_tensor_input(self._input_index, batch.shape, strict=True)
            self._interp.allocate_tensors()
            self._last_shape = tuple(int(d) for d in batch.shape)
        # Cast to whatever dtype the interpreter expects (float32 for INT8
        # mode, int8 for INT8_STRICT mode).
        if batch.dtype != self._input_dtype:
            batch = batch.astype(self._input_dtype)
        self._interp.set_tensor(self._input_index, batch)
        self._interp.invoke()
        return np.asarray(self._interp.get_tensor(self._output_index))

    def close(self) -> None:
        self._interp = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class LatencyBenchmark:
    """Run a Cartesian product of ``runner x batch_size`` measurements.

    Attributes
    ----------
    runners
        Concrete :class:`Runner` instances to benchmark.
    n_warmup
        Warmup iterations per ``(runner, batch_size)`` pair. Defaults to
        :data:`DEFAULT_WARMUP`.
    n_iter
        Measured iterations per ``(runner, batch_size)`` pair. Defaults
        to :data:`DEFAULT_ITERATIONS`.
    batch_sizes
        Batch sizes to sweep. Defaults to :data:`DEFAULT_BATCH_SIZES`.
    input_shape
        Per-sample shape (without batch dim). Defaults to
        :data:`DEFAULT_INPUT_SHAPE` (CIFAR-10).
    seed
        Seed for the random input generator. Pinned so the benchmark is
        bit-reproducible across runs and CI runners.
    """

    runners: Sequence[Runner]
    n_warmup: int = DEFAULT_WARMUP
    n_iter: int = DEFAULT_ITERATIONS
    batch_sizes: Sequence[int] = field(default_factory=lambda: list(DEFAULT_BATCH_SIZES))
    input_shape: tuple[int, ...] = DEFAULT_INPUT_SHAPE
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_warmup < 0:
            raise ValueError(f"n_warmup must be >= 0, got {self.n_warmup}")
        if self.n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {self.n_iter}")
        if not self.runners:
            raise ValueError("At least one runner is required.")
        if not self.batch_sizes:
            raise ValueError("At least one batch_size is required.")
        for bs in self.batch_sizes:
            if bs < 1:
                raise ValueError(f"batch_size must be >= 1, got {bs}")

    def run(self) -> list[BenchmarkResult]:
        """Execute the full benchmark and return one result per pair."""
        results: list[BenchmarkResult] = []
        for runner in self.runners:
            runner.load()
            try:
                for batch_size in self.batch_sizes:
                    results.append(self._measure_one(runner, batch_size))
            finally:
                runner.close()
        return results

    def _measure_one(self, runner: Runner, batch_size: int) -> BenchmarkResult:
        """Run ``n_warmup + n_iter`` predictions on a single batch."""
        rng = np.random.default_rng(self.seed)
        batch = rng.uniform(-3.0, 3.0, size=(batch_size, *self.input_shape)).astype(np.float32)

        log.info(
            "[%s] warmup x%d, measure x%d, batch_size=%d",
            runner.name,
            self.n_warmup,
            self.n_iter,
            batch_size,
        )

        # Warmup pass -- discard timings.
        for _ in range(self.n_warmup):
            runner.predict(batch)

        # Measured pass.
        timings_ms = np.empty(self.n_iter, dtype=np.float64)
        for i in range(self.n_iter):
            t0 = time.perf_counter()
            runner.predict(batch)
            timings_ms[i] = (time.perf_counter() - t0) * 1000.0

        mean_ms = float(timings_ms.mean())
        throughput_ips = batch_size / (mean_ms / 1000.0) if mean_ms > 0 else float("inf")

        return BenchmarkResult(
            runtime=runner.name,
            batch_size=batch_size,
            n_iterations=self.n_iter,
            p50_ms=float(np.percentile(timings_ms, 50)),
            p90_ms=float(np.percentile(timings_ms, 90)),
            p95_ms=float(np.percentile(timings_ms, 95)),
            p99_ms=float(np.percentile(timings_ms, 99)),
            mean_ms=mean_ms,
            std_ms=float(timings_ms.std(ddof=1)) if self.n_iter > 1 else 0.0,
            throughput_ips=throughput_ips,
        )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def to_dataframe(results: Iterable[BenchmarkResult]) -> pd.DataFrame:
    """Convert a list of results to a tidy ``pandas.DataFrame``.

    Columns: ``runtime``, ``batch_size``, ``n_iterations``, ``p50_ms``,
    ``p90_ms``, ``p95_ms``, ``p99_ms``, ``mean_ms``, ``std_ms``,
    ``throughput_ips``.

    The function imports ``pandas`` lazily so this module can be used
    in pandas-free environments (e.g. notebooks that consume the raw
    list of :class:`BenchmarkResult`).
    """
    import pandas as pd

    return pd.DataFrame([r.to_dict() for r in results])
