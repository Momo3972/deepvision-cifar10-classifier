"""Unit tests for :mod:`deepvision.export.onnx`.

The tests exercise real Keras -> ONNX conversion on a tiny MLP (so each
test stays under ~2 seconds even on a cold CPU) and then check the
round-trip equivalence numerically rather than just structurally. This
catches both:

- pure plumbing bugs (file not created, wrong opset, path normalization),
- silent numerical drift between Keras and ONNX Runtime (the kind of
  bug ``test_export_creates_file`` would miss).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepvision.export.onnx import (
    DEFAULT_OPSET,
    ONNX_VALIDATION_TOLERANCE,
    _infer_input_shape,
    _validate_onnx_export,
    export_to_onnx,
)

# ---------------------------------------------------------------------------
# Fixtures -- a tiny deterministic Keras model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    """Build a small deterministic Keras MLP that converts to ONNX in <1 s.

    Module-scoped so the four Keras imports + graph build happen once
    per test session instead of once per test, shaving ~6 s off the
    suite without weakening isolation (the model is stateless once
    built).
    """
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    inputs = tf.keras.Input(shape=(8,), name="features")
    x = tf.keras.layers.Dense(16, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="tiny_mlp")


# ---------------------------------------------------------------------------
# export_to_onnx -- happy paths
# ---------------------------------------------------------------------------


def test_export_creates_file(tiny_model, tmp_path: Path) -> None:
    output = tmp_path / "tiny.onnx"
    result = export_to_onnx(tiny_model, output)
    assert result == output.resolve()
    assert output.is_file()
    assert output.stat().st_size > 0


def test_export_creates_parent_dirs(tiny_model, tmp_path: Path) -> None:
    """The exporter must create missing parent directories."""
    output = tmp_path / "nested" / "subdir" / "tiny.onnx"
    assert not output.parent.exists()
    export_to_onnx(tiny_model, output, validate=False)
    assert output.is_file()


def test_export_passes_validation(tiny_model, tmp_path: Path) -> None:
    """Validation succeeds and returns numerics consistent with FP32."""
    output = tmp_path / "tiny.onnx"
    # validate=True is the default; calling it again should not raise.
    export_to_onnx(tiny_model, output)
    diffs = _validate_onnx_export(tiny_model, output)
    assert diffs["max_abs_diff"] < ONNX_VALIDATION_TOLERANCE
    assert diffs["mean_abs_diff"] < ONNX_VALIDATION_TOLERANCE


def test_export_respects_custom_opset(tiny_model, tmp_path: Path) -> None:
    output = tmp_path / "tiny_op15.onnx"
    export_to_onnx(tiny_model, output, opset=15)
    # Re-parse the file to check the opset is what we asked for.
    import onnx

    proto = onnx.load(str(output))
    onnx_opsets = {imp.domain: imp.version for imp in proto.opset_import}
    # Default domain ("") carries the canonical opset version.
    assert onnx_opsets.get("") == 15


def test_export_accepts_string_path(tiny_model, tmp_path: Path) -> None:
    """``output_path`` accepts ``str`` and ``Path`` interchangeably."""
    output = tmp_path / "tiny_str.onnx"
    result = export_to_onnx(tiny_model, str(output), validate=False)
    assert isinstance(result, Path)
    assert result.is_file()


# ---------------------------------------------------------------------------
# export_to_onnx -- error paths
# ---------------------------------------------------------------------------


def test_export_rejects_invalid_opset(tiny_model, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="opset"):
        export_to_onnx(tiny_model, tmp_path / "x.onnx", opset=0)


def test_export_rejects_negative_opset(tiny_model, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="opset"):
        export_to_onnx(tiny_model, tmp_path / "x.onnx", opset=-1)


# ---------------------------------------------------------------------------
# _validate_onnx_export -- numerical contract
# ---------------------------------------------------------------------------


def test_validate_detects_shape_mismatch(tiny_model, tmp_path: Path) -> None:
    """Manually corrupt the validation expectation to prove the guard fires.

    We can't easily produce a *real* shape-mismatched ONNX file from
    Keras, so we monkey-patch the runtime session output through a fake
    model that returns a different shape. The check is structural:
    swap the Keras model with a wrapper whose output shape diverges
    and confirm the assert fires.
    """
    import tensorflow as tf

    # First produce a valid ONNX file -- this is what ORT will run.
    output = tmp_path / "tiny.onnx"
    export_to_onnx(tiny_model, output, validate=False)

    # Build a Keras model with a deliberately different output shape.
    inputs = tf.keras.Input(shape=(8,))
    out = tf.keras.layers.Dense(5)(inputs)  # 5 classes instead of 3
    bad_shape_model = tf.keras.Model(inputs=inputs, outputs=out)

    with pytest.raises(AssertionError, match="shape mismatch"):
        _validate_onnx_export(bad_shape_model, output)


def test_validate_returns_zero_drift_on_clean_export(tiny_model, tmp_path: Path) -> None:
    """Sanity check: a clean export must yield essentially zero drift."""
    output = tmp_path / "tiny.onnx"
    export_to_onnx(tiny_model, output, validate=False)
    diffs = _validate_onnx_export(tiny_model, output, tolerance=ONNX_VALIDATION_TOLERANCE)
    # An MLP this small typically lands well under 1e-6 in FP32.
    assert diffs["max_abs_diff"] < 1e-5


def test_validate_raises_on_tight_tolerance(tiny_model, tmp_path: Path) -> None:
    """A tolerance tighter than FP32 round-off must fail loudly.

    Implementation note: a *negative* tolerance forces the guard to
    fire regardless of numerical luck. On some Python + tf2onnx
    combinations (e.g. CPython 3.12 + tf2onnx 1.17) the tiny MLP is so
    trivial that the conversion is *bit-identical* and
    ``max_abs_diff == 0.0``, so a tolerance of ``0.0`` would *not*
    fire (``0.0 > 0.0`` is ``False``). Using ``-1.0`` makes the test
    deterministic across runtimes while still exercising the failure
    path of the validator.
    """
    output = tmp_path / "tiny.onnx"
    export_to_onnx(tiny_model, output, validate=False)
    with pytest.raises(AssertionError, match="max_abs_diff"):
        _validate_onnx_export(tiny_model, output, tolerance=-1.0)


# ---------------------------------------------------------------------------
# _infer_input_shape -- batch-axis handling
# ---------------------------------------------------------------------------


def test_infer_input_shape_substitutes_batch(tiny_model) -> None:
    shape = _infer_input_shape(tiny_model, batch_size=7)
    # tiny_model has input shape (None, 8) -> (7, 8).
    assert shape == (7, 8)


def test_infer_input_shape_works_on_image_model() -> None:
    """A 4D input like (None, 32, 32, 3) must be resolved cleanly."""
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(32, 32, 3))
    out = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    shape = _infer_input_shape(model, batch_size=2)
    assert shape == (2, 32, 32, 3)


# ---------------------------------------------------------------------------
# Defaults -- make sure the public constants do not drift silently
# ---------------------------------------------------------------------------


def test_default_opset_is_documented_value() -> None:
    """The audit pins opset 17 -- a silent bump would invalidate the report."""
    assert DEFAULT_OPSET == 17


def test_validation_tolerance_is_reasonable() -> None:
    """Tolerance must be tight enough to catch a real bug, loose enough
    to absorb FP32 round-off across the MLP graph."""
    assert 1e-6 < ONNX_VALIDATION_TOLERANCE < 1e-2


def test_onnx_runtime_replay_matches_keras_numerically(tiny_model, tmp_path: Path) -> None:
    """End-to-end: ONNX Runtime output must match Keras on real inputs."""
    import onnxruntime as ort

    output = tmp_path / "tiny.onnx"
    export_to_onnx(tiny_model, output, validate=False)

    rng = np.random.default_rng(seed=123)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    keras_out = np.asarray(tiny_model.predict(x, verbose=0))

    sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    onnx_out = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0])

    np.testing.assert_allclose(keras_out, onnx_out, atol=ONNX_VALIDATION_TOLERANCE)
