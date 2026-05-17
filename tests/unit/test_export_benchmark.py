"""Unit tests for :mod:`deepvision.export.benchmark`.

The tests cover three concerns:

* contract of :class:`BenchmarkResult` (field semantics, ``to_dict``,
  pandas conversion),
* contract of every :class:`Runner` implementation (load -> predict ->
  close) against a real exported artefact,
* contract of :class:`LatencyBenchmark` (loop control, input validation,
  percentile maths).

To keep the suite fast we use a tiny MLP exported to ONNX / TFLite in a
session-scoped fixture; the per-runner predict calls then run in
microseconds and a 10-iteration benchmark finishes well under a second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepvision.export.benchmark import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_ITERATIONS,
    DEFAULT_WARMUP,
    BenchmarkResult,
    KerasRunner,
    LatencyBenchmark,
    OnnxRuntimeRunner,
    Runner,
    TFLiteRunner,
    to_dataframe,
)

# ---------------------------------------------------------------------------
# Tiny model + exported artefacts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_mlp_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Export a tiny MLP to .keras / .onnx / .tflite / SavedModel once per test module."""
    import tensorflow as tf

    from deepvision.export.onnx import export_to_onnx
    from deepvision.export.tflite import QuantizationMode, export_to_tflite

    tf.keras.utils.set_random_seed(0)
    inputs = tf.keras.Input(shape=(4,))
    x = tf.keras.layers.Dense(8, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    base = tmp_path_factory.mktemp("artifacts")
    keras_path = base / "tiny.keras"
    onnx_path = base / "tiny.onnx"
    tflite_path = base / "tiny_dyn.tflite"
    savedmodel_dir = base / "tiny_savedmodel"

    model.save(str(keras_path))
    # ``model.export`` writes the Keras 3 SavedModel format with the
    # ``"serve"`` signature key -- the very flavour that the
    # ``TFSavedModelRunner`` signature-discovery logic must cope with.
    model.export(str(savedmodel_dir))
    export_to_onnx(model, onnx_path, validate=False)
    export_to_tflite(model, tflite_path, quantization=QuantizationMode.DYNAMIC)

    return {
        "keras": keras_path,
        "onnx": onnx_path,
        "tflite": tflite_path,
        "savedmodel": savedmodel_dir,
    }


# ---------------------------------------------------------------------------
# BenchmarkResult -- pure dataclass behaviour
# ---------------------------------------------------------------------------


def test_benchmark_result_to_dict_roundtrip() -> None:
    r = BenchmarkResult(
        runtime="onnx_runtime",
        batch_size=1,
        n_iterations=10,
        p50_ms=1.0,
        p90_ms=1.2,
        p95_ms=1.3,
        p99_ms=1.5,
        mean_ms=1.1,
        std_ms=0.1,
        throughput_ips=909.0,
    )
    d = r.to_dict()
    assert d["runtime"] == "onnx_runtime"
    assert d["batch_size"] == 1
    assert d["n_iterations"] == 10
    assert {
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "mean_ms",
        "std_ms",
        "throughput_ips",
    } <= d.keys()


def test_benchmark_result_is_frozen() -> None:
    """Results must be immutable so downstream tables stay consistent."""
    r = BenchmarkResult(
        runtime="x",
        batch_size=1,
        n_iterations=1,
        p50_ms=1,
        p90_ms=1,
        p95_ms=1,
        p99_ms=1,
        mean_ms=1,
        std_ms=0,
        throughput_ips=1000,
    )
    with pytest.raises((AttributeError, Exception)):
        r.runtime = "y"  # type: ignore[misc]


def test_to_dataframe_columns() -> None:
    r = BenchmarkResult(
        runtime="keras",
        batch_size=1,
        n_iterations=1,
        p50_ms=1,
        p90_ms=1,
        p95_ms=1,
        p99_ms=1,
        mean_ms=1,
        std_ms=0,
        throughput_ips=1,
    )
    df = to_dataframe([r])
    expected = {
        "runtime",
        "batch_size",
        "n_iterations",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "mean_ms",
        "std_ms",
        "throughput_ips",
    }
    assert set(df.columns) >= expected
    assert len(df) == 1


# ---------------------------------------------------------------------------
# Runner backends -- load -> predict -> close
# ---------------------------------------------------------------------------


def _run_predict(runner: Runner, batch: np.ndarray) -> np.ndarray:
    runner.load()
    try:
        return runner.predict(batch)
    finally:
        runner.close()


def test_keras_runner_predict_shape(tiny_mlp_artifacts) -> None:
    runner = KerasRunner(tiny_mlp_artifacts["keras"])
    out = _run_predict(runner, np.zeros((3, 4), dtype=np.float32))
    assert out.shape == (3, 2)


def test_onnx_runtime_runner_predict_shape(tiny_mlp_artifacts) -> None:
    runner = OnnxRuntimeRunner(tiny_mlp_artifacts["onnx"])
    out = _run_predict(runner, np.zeros((3, 4), dtype=np.float32))
    assert out.shape == (3, 2)


def test_tf_savedmodel_runner_predict_shape(tiny_mlp_artifacts) -> None:
    """Validates the Keras 3 ``'serve'`` signature key path end-to-end."""
    from deepvision.export.benchmark import TFSavedModelRunner

    runner = TFSavedModelRunner(tiny_mlp_artifacts["savedmodel"])
    out = _run_predict(runner, np.zeros((3, 4), dtype=np.float32))
    assert out.shape == (3, 2)


def test_tf_savedmodel_runner_picks_serve_key_first(tiny_mlp_artifacts) -> None:
    """Keras 3 SavedModels expose ``'serve'``; the runner must pick it.

    Regression guard against the original implementation which hard-coded
    ``'serving_default'`` and crashed on Keras 3 exports.
    """
    from deepvision.export.benchmark import TFSavedModelRunner

    runner = TFSavedModelRunner(tiny_mlp_artifacts["savedmodel"])
    runner.load()
    try:
        # Probe the chosen signature via the loaded SavedModel directly.
        import tensorflow as tf

        loaded = tf.saved_model.load(str(tiny_mlp_artifacts["savedmodel"]))
        assert "serve" in loaded.signatures
        assert TFSavedModelRunner._select_signature_key(loaded.signatures.keys()) == "serve"
    finally:
        runner.close()


def test_tf_savedmodel_runner_signature_priority() -> None:
    """``serve`` wins over ``serving_default``; otherwise first key wins."""
    from deepvision.export.benchmark import TFSavedModelRunner

    # Both keys present: ``serve`` wins.
    assert TFSavedModelRunner._select_signature_key(["serving_default", "serve"]) == "serve"
    # Only legacy key: it wins.
    assert TFSavedModelRunner._select_signature_key(["serving_default"]) == "serving_default"
    # Neither preferred key: fall back to first available.
    assert TFSavedModelRunner._select_signature_key(["custom", "other"]) == "custom"
    # Empty: raise.
    with pytest.raises(ValueError, match="no serving signature"):
        TFSavedModelRunner._select_signature_key([])


def test_tflite_runner_handles_dynamic_batch(tiny_mlp_artifacts) -> None:
    """The TFLite interpreter must resize its tensors when the batch shape changes."""
    runner = TFLiteRunner(tiny_mlp_artifacts["tflite"])
    runner.load()
    try:
        out1 = runner.predict(np.zeros((1, 4), dtype=np.float32))
        out4 = runner.predict(np.zeros((4, 4), dtype=np.float32))
        assert out1.shape == (1, 2)
        assert out4.shape == (4, 2)
    finally:
        runner.close()


def test_runners_close_releases_state(tiny_mlp_artifacts) -> None:
    """After ``close()``, calling ``predict`` must fail rather than reuse stale state."""
    runner = KerasRunner(tiny_mlp_artifacts["keras"])
    runner.load()
    runner.close()
    with pytest.raises(AssertionError):
        runner.predict(np.zeros((1, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# LatencyBenchmark -- orchestration
# ---------------------------------------------------------------------------


def test_latency_benchmark_runs_all_runners(tiny_mlp_artifacts) -> None:
    bench = LatencyBenchmark(
        runners=[
            KerasRunner(tiny_mlp_artifacts["keras"]),
            OnnxRuntimeRunner(tiny_mlp_artifacts["onnx"]),
        ],
        n_warmup=2,
        n_iter=5,
        batch_sizes=[1, 2],
        input_shape=(4,),
    )
    results = bench.run()
    # 2 runtimes x 2 batch sizes = 4 records.
    assert len(results) == 4
    runtimes = {r.runtime for r in results}
    assert runtimes == {"keras", "onnx_runtime"}
    for r in results:
        assert r.n_iterations == 5
        assert r.p50_ms <= r.p95_ms <= r.p99_ms
        assert r.throughput_ips > 0


def test_latency_benchmark_rejects_empty_runners() -> None:
    with pytest.raises(ValueError, match="runner"):
        LatencyBenchmark(runners=[], batch_sizes=[1])


def test_latency_benchmark_rejects_invalid_warmup(tiny_mlp_artifacts) -> None:
    with pytest.raises(ValueError, match="n_warmup"):
        LatencyBenchmark(
            runners=[KerasRunner(tiny_mlp_artifacts["keras"])],
            n_warmup=-1,
        )


def test_latency_benchmark_rejects_invalid_iter(tiny_mlp_artifacts) -> None:
    with pytest.raises(ValueError, match="n_iter"):
        LatencyBenchmark(
            runners=[KerasRunner(tiny_mlp_artifacts["keras"])],
            n_iter=0,
        )


def test_latency_benchmark_rejects_invalid_batch_size(tiny_mlp_artifacts) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        LatencyBenchmark(
            runners=[KerasRunner(tiny_mlp_artifacts["keras"])],
            batch_sizes=[0],
        )


def test_latency_benchmark_rejects_empty_batch_sizes(tiny_mlp_artifacts) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        LatencyBenchmark(
            runners=[KerasRunner(tiny_mlp_artifacts["keras"])],
            batch_sizes=[],
        )


# ---------------------------------------------------------------------------
# Public constants -- guard against silent drift
# ---------------------------------------------------------------------------


def test_default_warmup_iter_match_audit() -> None:
    assert DEFAULT_WARMUP == 100
    assert DEFAULT_ITERATIONS == 1000


def test_default_batch_sizes_match_audit() -> None:
    assert DEFAULT_BATCH_SIZES == (1, 8, 32)
