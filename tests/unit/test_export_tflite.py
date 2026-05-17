"""Unit tests for :mod:`deepvision.export.tflite`.

These tests run real ``tf.lite.TFLiteConverter`` conversions on a tiny
CNN to make sure:

- every :class:`QuantizationMode` produces a loadable ``.tflite`` file,
- the Full INT8 mode requires a representative dataset and consumes one
  passed as either an ``ndarray`` or a generator factory,
- :func:`build_representative_dataset` enforces its shape contract,
- the ``int8_strict`` mode actually emits an INT8 input tensor.

A small CNN is used (rather than a pure MLP) so the converter exercises
its convolution / activation quantization paths -- the very paths that
fail silently when calibration data is wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepvision.export.tflite import (
    DEFAULT_REPRESENTATIVE_SAMPLES,
    QuantizationMode,
    build_representative_dataset,
    export_to_tflite,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_cnn():
    """A tiny CNN that converts to TFLite in <1 s on CPU."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    inputs = tf.keras.Input(shape=(8, 8, 3), name="image")
    x = tf.keras.layers.Conv2D(4, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="tiny_cnn")


@pytest.fixture(scope="module")
def representative_pool() -> np.ndarray:
    """A pool of synthetic 8x8x3 images for INT8 calibration."""
    rng = np.random.default_rng(seed=42)
    return rng.uniform(0.0, 1.0, size=(50, 8, 8, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# build_representative_dataset -- generator factory
# ---------------------------------------------------------------------------


def test_build_representative_dataset_yields_n_samples(representative_pool) -> None:
    gen_factory = build_representative_dataset(representative_pool, n_samples=10)
    samples = list(gen_factory())
    assert len(samples) == 10
    # Each yield is a list containing one (1, H, W, C) array.
    for item in samples:
        assert isinstance(item, list)
        assert len(item) == 1
        assert item[0].shape == (1, 8, 8, 3)
        assert item[0].dtype == np.float32


def test_build_representative_dataset_is_deterministic(representative_pool) -> None:
    """Two calls with the same seed must yield identical samples."""
    gen_a = build_representative_dataset(representative_pool, n_samples=5, seed=7)
    gen_b = build_representative_dataset(representative_pool, n_samples=5, seed=7)
    a = [item[0] for item in gen_a()]
    b = [item[0] for item in gen_b()]
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_build_representative_dataset_rejects_3d() -> None:
    bad = np.zeros((10, 8, 8))  # missing channel axis
    with pytest.raises(ValueError, match="4D"):
        build_representative_dataset(bad)


def test_build_representative_dataset_rejects_too_few_samples(
    representative_pool,
) -> None:
    with pytest.raises(ValueError, match="rows"):
        build_representative_dataset(representative_pool, n_samples=999)


def test_build_representative_dataset_rejects_zero_samples(
    representative_pool,
) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        build_representative_dataset(representative_pool, n_samples=0)


# ---------------------------------------------------------------------------
# QuantizationMode enum
# ---------------------------------------------------------------------------


def test_quantization_mode_from_string() -> None:
    assert QuantizationMode("int8") is QuantizationMode.INT8
    assert QuantizationMode("dynamic") is QuantizationMode.DYNAMIC
    assert QuantizationMode("fp16") is QuantizationMode.FP16


def test_quantization_mode_invalid_string_raises() -> None:
    with pytest.raises(ValueError, match="not a valid"):
        QuantizationMode("not-a-mode")


# ---------------------------------------------------------------------------
# export_to_tflite -- per-mode happy paths
# ---------------------------------------------------------------------------


def test_dynamic_export_produces_file(tiny_cnn, tmp_path: Path) -> None:
    output = tmp_path / "dynamic.tflite"
    result = export_to_tflite(tiny_cnn, output, quantization="dynamic")
    assert result == output.resolve()
    assert output.stat().st_size > 0


def test_fp16_export_produces_file(tiny_cnn, tmp_path: Path) -> None:
    output = tmp_path / "fp16.tflite"
    export_to_tflite(tiny_cnn, output, quantization=QuantizationMode.FP16)
    assert output.is_file()


def test_int8_export_with_ndarray(tiny_cnn, representative_pool, tmp_path: Path) -> None:
    output = tmp_path / "int8.tflite"
    export_to_tflite(
        tiny_cnn,
        output,
        quantization=QuantizationMode.INT8,
        representative_data=representative_pool,
        n_samples=20,
    )
    assert output.stat().st_size > 0


def test_int8_export_with_generator(tiny_cnn, representative_pool, tmp_path: Path) -> None:
    """Passing a pre-built generator factory must work just like an ndarray."""
    output = tmp_path / "int8_gen.tflite"
    gen_factory = build_representative_dataset(representative_pool, n_samples=15)
    export_to_tflite(
        tiny_cnn,
        output,
        quantization=QuantizationMode.INT8,
        representative_data=gen_factory,
    )
    assert output.is_file()


def test_int8_strict_produces_int8_io_tensors(
    tiny_cnn, representative_pool, tmp_path: Path
) -> None:
    """``int8_strict`` must yield a model whose input tensor is int8."""
    import tensorflow as tf

    output = tmp_path / "int8_strict.tflite"
    export_to_tflite(
        tiny_cnn,
        output,
        quantization=QuantizationMode.INT8_STRICT,
        representative_data=representative_pool,
        n_samples=10,
    )

    interp = tf.lite.Interpreter(model_path=str(output))
    interp.allocate_tensors()
    input_details = interp.get_input_details()[0]
    output_details = interp.get_output_details()[0]
    assert input_details["dtype"] == np.int8
    assert output_details["dtype"] == np.int8


# ---------------------------------------------------------------------------
# export_to_tflite -- error paths
# ---------------------------------------------------------------------------


def test_int8_requires_representative_data(tiny_cnn, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="representative_data"):
        export_to_tflite(tiny_cnn, tmp_path / "x.tflite", quantization="int8")


def test_int8_strict_requires_representative_data(tiny_cnn, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="representative_data"):
        export_to_tflite(tiny_cnn, tmp_path / "x.tflite", quantization="int8_strict")


def test_export_rejects_unknown_mode_string(tiny_cnn, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown quantization mode"):
        export_to_tflite(tiny_cnn, tmp_path / "x.tflite", quantization="garbage")


# ---------------------------------------------------------------------------
# Numerical accuracy -- INT8 must stay close to FP32 on the calibration set
# ---------------------------------------------------------------------------


def test_int8_export_predictions_close_to_keras(
    tiny_cnn, representative_pool, tmp_path: Path
) -> None:
    """Quantization to INT8 should not catastrophically degrade predictions.

    We require the per-image argmax to agree on at least 80 % of the
    calibration set. Anything below that signals a calibration bug --
    e.g. forgetting to pass the representative dataset, or feeding it
    images outside the activation range.
    """
    import tensorflow as tf

    output = tmp_path / "int8_acc.tflite"
    export_to_tflite(
        tiny_cnn,
        output,
        quantization=QuantizationMode.INT8,
        representative_data=representative_pool,
        n_samples=20,
    )

    keras_preds = np.argmax(np.asarray(tiny_cnn.predict(representative_pool, verbose=0)), axis=1)

    interp = tf.lite.Interpreter(model_path=str(output))
    interp.allocate_tensors()
    in_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]
    interp.resize_tensor_input(in_details["index"], representative_pool.shape, strict=True)
    interp.allocate_tensors()
    interp.set_tensor(in_details["index"], representative_pool.astype(in_details["dtype"]))
    interp.invoke()
    tflite_preds = np.argmax(interp.get_tensor(out_details["index"]), axis=1)

    agreement = (keras_preds == tflite_preds).mean()
    assert agreement >= 0.80, f"INT8 / FP32 agreement too low: {agreement:.2%}"


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


def test_default_representative_samples_is_reasonable() -> None:
    """The audit recommends 100-500 samples; the default must sit there."""
    assert 100 <= DEFAULT_REPRESENTATIVE_SAMPLES <= 500
