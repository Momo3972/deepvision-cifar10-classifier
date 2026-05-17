"""Keras -> ONNX exporter -- Phase 10.

The audit (sections 7.2 and 9, Phase 10) prescribes ONNX export so the
trained EfficientNetB0 artefact can be served through *any* runtime that
speaks the ONNX Operator Set -- ONNX Runtime (CPU/GPU/mobile), Triton,
TensorRT, etc. The benchmark in :mod:`deepvision.export.benchmark` then
quantifies the latency gain over the native TensorFlow path.

Why opset 17
============
We default to **opset 17** because:

- It is the latest opset universally supported by ``onnxruntime>=1.13``
  and ``TensorRT>=8.6``.
- It exposes the fused operators ``LayerNormalization``, ``Gelu`` and
  ``HardSwish`` that EfficientNet leans on, avoiding the costly
  decomposition into elementary ops that older opsets imply.
- ``opset 20+`` exists but its ``StringNormalizer`` and
  ``GroupNormalization`` ops are not yet implemented in older
  ``onnxruntime`` builds we may encounter on customer hardware.

How the conversion happens
==========================
We call :func:`tf2onnx.convert.from_keras` directly with an explicit
``input_signature`` built from ``model.input_shape``. The explicit
signature is what makes the conversion robust on the Keras 3 / TF 2.21
stack: without it, tf2onnx falls back to introspecting the Keras
functional graph and occasionally fails on EfficientNet's
preprocessing BatchNormalization layers. Passing the signature
sidesteps that traversal entirely -- tf2onnx traces a fresh
``tf.function`` from the spec and the conversion succeeds even on
complex transfer-learning architectures.

Validation
==========
Every export ends with :func:`_validate_onnx_export`, which feeds random
uniform inputs through both the Keras model and an ONNX Runtime session
and asserts the maximum absolute difference is below
``ONNX_VALIDATION_TOLERANCE`` (default ``1e-4``). A silently broken
conversion therefore never escapes CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from deepvision.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from tensorflow.keras import Model

log = get_logger(__name__)

#: Default ONNX opset version targeted by :func:`export_to_onnx`. See the
#: module docstring for the rationale.
DEFAULT_OPSET: Final[int] = 17

#: Maximum acceptable absolute difference between a Keras forward pass
#: and the same forward pass replayed through ONNX Runtime, per element.
#: Empirically EfficientNetB0 on CIFAR-10 stays well below ``1e-5`` so
#: ``1e-4`` leaves a comfortable safety margin for future architectures.
ONNX_VALIDATION_TOLERANCE: Final[float] = 1e-4

#: Number of random samples drawn for the validation forward pass.
#: Small enough to keep the validation pass under 200 ms on CPU.
_VALIDATION_SAMPLES: Final[int] = 4


def export_to_onnx(
    model: Model,
    output_path: Path | str,
    *,
    opset: int = DEFAULT_OPSET,
    validate: bool = True,
    tolerance: float = ONNX_VALIDATION_TOLERANCE,
) -> Path:
    """Export a Keras model to an ONNX file via tf2onnx.

    Parameters
    ----------
    model
        A trained Keras 3 model. Must have a defined ``input_shape``.
    output_path
        Destination ``.onnx`` file. Parent directories are created if
        they do not exist.
    opset
        ONNX operator set version. Defaults to :data:`DEFAULT_OPSET`
        (17). Lower opsets degrade gracefully but lose access to the
        fused ops; higher opsets risk runtime incompatibility.
    validate
        When ``True`` (default), run :func:`_validate_onnx_export`
        immediately after the conversion. Disable only in throughput
        benchmarks where the validation overhead is unacceptable.
    tolerance
        Maximum allowed per-element absolute difference between the
        Keras and ONNX Runtime outputs. Passed through to
        :func:`_validate_onnx_export`.

    Returns
    -------
    pathlib.Path
        Absolute path of the written ``.onnx`` file.

    Raises
    ------
    ValueError
        If ``opset`` is not a positive integer.
    AssertionError
        If ``validate`` is ``True`` and the ONNX output does not match
        the Keras output within ``tolerance``.

    Examples
    --------
    >>> from deepvision.models.efficientnet import build_efficientnet
    >>> model = build_efficientnet(weights=None)
    >>> path = export_to_onnx(model, "models/exports/efficientnet.onnx")  # doctest: +SKIP
    """
    if opset < 1:
        raise ValueError(f"opset must be a positive integer, got {opset}")

    # Heavy imports deferred so simply importing this module stays cheap.
    import tensorflow as tf
    import tf2onnx

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # We pass an *explicit* input_signature instead of letting tf2onnx
    # introspect the Keras graph. Keras 3 + TF 2.21 occasionally leaks
    # ``None`` dimensions or names that tf2onnx mis-parses; building the
    # spec ourselves from ``model.input_shape`` eliminates that surface.
    input_signature = [
        tf.TensorSpec(
            shape=tuple(model.input_shape),
            dtype=tf.float32,
            name="input",
        )
    ]

    log.info("Converting Keras model to ONNX (opset=%d) -> %s", opset, output_path)
    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=str(output_path),
    )

    if not output_path.is_file():
        raise RuntimeError(f"tf2onnx did not produce a file at {output_path}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("ONNX export complete: %s (%.2f MB)", output_path, size_mb)

    if validate:
        diffs = _validate_onnx_export(model, output_path, tolerance=tolerance)
        log.info(
            "ONNX validation passed: max_abs_diff=%.2e, mean_abs_diff=%.2e",
            diffs["max_abs_diff"],
            diffs["mean_abs_diff"],
        )

    return output_path


def _validate_onnx_export(
    model: Model,
    onnx_path: Path,
    *,
    tolerance: float = ONNX_VALIDATION_TOLERANCE,
    n_samples: int = _VALIDATION_SAMPLES,
    seed: int = 42,
) -> dict[str, float]:
    """Compare a Keras forward pass to its ONNX Runtime replay.

    Parameters
    ----------
    model
        Source Keras model used for the export.
    onnx_path
        Path of the exported ONNX file to validate.
    tolerance
        Maximum allowed per-element absolute difference. The function
        raises :class:`AssertionError` if exceeded.
    n_samples
        Batch size of the random inputs. Defaults to ``4`` -- enough to
        catch shape / broadcasting bugs while staying fast.
    seed
        Seed for the random input generator. Pinned to ``42`` so the
        validation is bit-identical between runs and CI failures are
        reproducible.

    Returns
    -------
    dict[str, float]
        ``{"max_abs_diff": ..., "mean_abs_diff": ...}`` -- handy for
        callers that want to log the numerics without re-running the
        check.

    Raises
    ------
    AssertionError
        If ``max_abs_diff > tolerance``.
    """
    import onnxruntime as ort

    input_shape = _infer_input_shape(model, batch_size=n_samples)

    rng = np.random.default_rng(seed=seed)
    # Use a wide [-3, 3] range so the BatchNorm / activations see varied
    # inputs and any precision drift surfaces clearly.
    x = rng.uniform(-3.0, 3.0, size=input_shape).astype(np.float32)

    keras_out = np.asarray(model.predict(x, verbose=0))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_out = np.asarray(sess.run(None, {input_name: x})[0])

    if keras_out.shape != onnx_out.shape:
        raise AssertionError(
            f"Output shape mismatch: keras={keras_out.shape}, onnx={onnx_out.shape}"
        )

    diff = np.abs(keras_out - onnx_out)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    if max_diff > tolerance:
        raise AssertionError(
            f"ONNX export does not match Keras: "
            f"max_abs_diff={max_diff:.3e} exceeds tolerance={tolerance:.3e} "
            f"(mean_abs_diff={mean_diff:.3e})"
        )

    return {"max_abs_diff": max_diff, "mean_abs_diff": mean_diff}


def _infer_input_shape(model: Model, *, batch_size: int) -> tuple[int, ...]:
    """Return the concrete input shape for a forward pass.

    ``model.input_shape`` carries ``None`` for the batch axis (and
    sometimes for dynamic spatial dims). We replace every ``None`` with
    a concrete value so NumPy can allocate the tensor:

    - the leading ``None`` (batch) is replaced with ``batch_size``,
    - any remaining ``None`` -- rare, but possible on models with
      variable-resolution inputs -- is replaced with ``1`` so the
      forward pass still executes. The validation only needs a
      *consistent* shape between Keras and ONNX Runtime, not a
      production-grade one.

    Parameters
    ----------
    model
        Keras model.
    batch_size
        Concrete batch size for the validation forward pass.

    Returns
    -------
    tuple[int, ...]
        Fully concrete shape suitable for :func:`numpy.ndarray.reshape`.
    """
    raw = tuple(model.input_shape)
    if not raw:
        raise ValueError("Model has no input_shape; cannot infer a validation input.")

    # Replace batch axis (always first) with the requested batch size.
    concrete = [batch_size] + [int(d) if d is not None else 1 for d in raw[1:]]
    return tuple(concrete)
