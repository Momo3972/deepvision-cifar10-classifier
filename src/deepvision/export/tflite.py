"""Keras -> TFLite exporter with Full INT8 quantization -- Phase 10.

The audit (sections 7.2 and 9, Phase 10) prescribes a *quantized* TFLite
artefact so the model can be deployed on edge hardware -- mobile phones,
single-board computers, microcontrollers -- where memory bandwidth is
the bottleneck and FP32 inference is impractical.

We expose four quantization modes through :class:`QuantizationMode`:

- :attr:`QuantizationMode.DYNAMIC` -- weights are quantized to INT8 but
  activations stay in FP32. Smallest file *without* needing calibration
  data; latency gains are modest because activations still flow in
  floating point.
- :attr:`QuantizationMode.INT8` -- both weights and activations are
  quantized to INT8 using a representative dataset. I/O tensors stay in
  FP32 so the model is a drop-in replacement for the Keras model in the
  benchmark and in the FastAPI serving path. **This is the default** and
  matches what the audit calls "quantization complete".
- :attr:`QuantizationMode.INT8_STRICT` -- like ``INT8`` but I/O tensors
  are also INT8. Pure edge-deployment mode; the caller is responsible
  for input quantization and output dequantization.
- :attr:`QuantizationMode.FP16` -- weights in FP16, no calibration
  data. Useful for GPU edge runtimes (NNAPI on Android, Core ML on
  iOS) where FP16 maths is the fast path.

Why a representative dataset
============================
Full INT8 quantization needs *activation statistics* to pick the scale
and zero-point of every tensor. TFLite collects those by running the
converter on a generator that yields a handful of real inputs. The
audit recommends 100-500 samples; we default to
:data:`DEFAULT_REPRESENTATIVE_SAMPLES` (200), which empirically captures
the activation range of EfficientNetB0 on CIFAR-10 without making the
conversion noticeably slow.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from deepvision.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from tensorflow.keras import Model

log = get_logger(__name__)

#: Default size of the representative dataset used for INT8 calibration.
#: 200 samples is the sweet spot recommended by TensorFlow Lite docs
#: (https://www.tensorflow.org/lite/performance/post_training_quantization)
#: -- enough to cover the activation range of EfficientNetB0 on
#: CIFAR-10 without making the conversion appreciably slower.
DEFAULT_REPRESENTATIVE_SAMPLES: Final[int] = 200


class QuantizationMode(enum.StrEnum):
    """Post-training quantization strategy supported by :func:`export_to_tflite`."""

    DYNAMIC = "dynamic"
    INT8 = "int8"
    INT8_STRICT = "int8_strict"
    FP16 = "fp16"


#: Type alias for the generator function consumed by TFLite's
#: ``representative_dataset`` hook. The TFLite API expects a no-arg
#: callable that returns a fresh generator each time the converter calls
#: it -- typically once per quantization pass.
RepresentativeDataGen = Callable[[], Iterator[list[np.ndarray]]]


def build_representative_dataset(
    images: np.ndarray,
    *,
    n_samples: int = DEFAULT_REPRESENTATIVE_SAMPLES,
    seed: int = 42,
) -> RepresentativeDataGen:
    """Build a TFLite-compatible representative dataset generator.

    Parameters
    ----------
    images
        Source pool of images, shape ``(N, H, W, C)`` with ``N >= n_samples``.
        Typically a slice of the CIFAR-10 train set obtained from
        :func:`deepvision.data.loader.load_cifar10`.
    n_samples
        Number of distinct samples drawn (without replacement) from
        ``images``. Defaults to :data:`DEFAULT_REPRESENTATIVE_SAMPLES`.
    seed
        Seed for the shuffling RNG. Pinned so successive exports of the
        same model produce bit-identical TFLite files.

    Returns
    -------
    RepresentativeDataGen
        A no-arg callable that returns a fresh generator over the
        selected samples. Each yielded element is a one-element list
        ``[image[None, ...]]`` -- TFLite expects a list because models
        can have multiple inputs.

    Raises
    ------
    ValueError
        If ``images`` is malformed or does not contain at least
        ``n_samples`` rows.

    Examples
    --------
    >>> split = load_cifar10()  # doctest: +SKIP
    >>> gen = build_representative_dataset(split.x_train, n_samples=200)
    >>> # Pass ``gen`` to the converter -- TFLite calls ``gen()`` itself.
    """
    if images.ndim != 4:
        raise ValueError(f"images must be 4D (N, H, W, C); got shape {images.shape}.")
    if images.shape[0] < n_samples:
        raise ValueError(
            f"images has only {images.shape[0]} rows but n_samples={n_samples} were requested."
        )
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")

    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(images.shape[0], size=n_samples, replace=False)
    selected = images[indices].astype(np.float32)

    def _gen() -> Iterator[list[np.ndarray]]:
        # Yield one image at a time with an explicit batch dimension --
        # TFLite's calibrator processes a single sample per call.
        for img in selected:
            yield [img[None, ...]]

    return _gen


def export_to_tflite(
    model: Model,
    output_path: Path | str,
    *,
    quantization: QuantizationMode | str = QuantizationMode.INT8,
    representative_data: np.ndarray | RepresentativeDataGen | None = None,
    n_samples: int = DEFAULT_REPRESENTATIVE_SAMPLES,
) -> Path:
    """Export a Keras model to a quantized TFLite file.

    Parameters
    ----------
    model
        Trained Keras 3 model.
    output_path
        Destination ``.tflite`` file. Parent directories are created if
        they do not exist.
    quantization
        One of :class:`QuantizationMode`. Defaults to
        :attr:`QuantizationMode.INT8` (Full INT8 with FP32 I/O).
    representative_data
        Required for ``INT8`` and ``INT8_STRICT``. Either:

        - a :class:`numpy.ndarray` of images shape ``(N, H, W, C)`` --
          wrapped on the fly by :func:`build_representative_dataset`, or
        - a pre-built :data:`RepresentativeDataGen` callable.

        Ignored for ``DYNAMIC`` and ``FP16``.
    n_samples
        Used only when ``representative_data`` is an ``ndarray`` to size
        the generator. Defaults to :data:`DEFAULT_REPRESENTATIVE_SAMPLES`.

    Returns
    -------
    pathlib.Path
        Absolute path of the written ``.tflite`` file.

    Raises
    ------
    ValueError
        If a Full-INT8 mode is requested without ``representative_data``,
        or if ``quantization`` is not a valid mode string.

    Examples
    --------
    Full INT8 export of an EfficientNetB0::

        from deepvision.data.loader import load_cifar10

        split = load_cifar10()
        export_to_tflite(
            model,
            "models/exports/efficientnet_int8.tflite",
            quantization="int8",
            representative_data=split.x_train,
        )
    """
    # Normalise string mode to enum.
    if isinstance(quantization, str) and not isinstance(quantization, QuantizationMode):
        try:
            quantization = QuantizationMode(quantization)
        except ValueError as exc:
            valid = ", ".join(m.value for m in QuantizationMode)
            raise ValueError(
                f"Unknown quantization mode {quantization!r}; expected one of: {valid}"
            ) from exc

    needs_repr = quantization in (QuantizationMode.INT8, QuantizationMode.INT8_STRICT)
    if needs_repr and representative_data is None:
        raise ValueError(
            f"quantization={quantization.value!r} requires a representative_data argument "
            "(an ndarray of images or a generator factory)."
        )

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Heavy imports deferred -- keeps the module importable without TF.
    import tempfile

    import tensorflow as tf

    with tempfile.TemporaryDirectory(prefix="dv-tflite-") as tmp:
        sm_dir = Path(tmp) / "savedmodel"
        log.info("Exporting Keras model to transient SavedModel at %s", sm_dir)
        model.export(str(sm_dir))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(sm_dir))
        _apply_quantization(
            converter,
            mode=quantization,
            representative_data=representative_data,
            n_samples=n_samples,
            tf=tf,
        )

        log.info(
            "Converting SavedModel to TFLite (quantization=%s) -> %s",
            quantization.value,
            output_path,
        )
        tflite_bytes = converter.convert()

    output_path.write_bytes(tflite_bytes)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info(
        "TFLite export complete: %s (%.2f MB, mode=%s)",
        output_path,
        size_mb,
        quantization.value,
    )
    return output_path


def _apply_quantization(
    converter: object,  # tf.lite.TFLiteConverter -- typed loosely so this module stays importable without TF
    *,
    mode: QuantizationMode,
    representative_data: np.ndarray | RepresentativeDataGen | None,
    n_samples: int,
    tf: object,  # tensorflow module -- passed in to keep the heavy import scoped
) -> None:
    """Mutate a ``tf.lite.TFLiteConverter`` to match the requested mode.

    Implementation notes
    --------------------
    - All four modes start by enabling ``Optimize.DEFAULT``; TFLite then
      picks the actual quantization strategy from the additional
      properties we set below.
    - For Full INT8 modes we explicitly restrict the supported ops to
      ``TFLITE_BUILTINS_INT8`` so the converter raises a clear error
      instead of silently emitting an FP32 fallback for any unsupported
      op (which would cancel the benefit of quantization).
    - For ``INT8_STRICT`` we set ``inference_input_type`` /
      ``inference_output_type`` to ``int8``, which means the caller must
      quantize / dequantize manually -- this is the true edge mode.
    """
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # type: ignore[attr-defined]

    if mode == QuantizationMode.DYNAMIC:
        # Default optimisations alone == dynamic-range quantization.
        return

    if mode == QuantizationMode.FP16:
        converter.target_spec.supported_types = [tf.float16]  # type: ignore[attr-defined]
        return

    # Full INT8 modes from here on -- both need a representative dataset.
    if isinstance(representative_data, np.ndarray):
        gen = build_representative_dataset(representative_data, n_samples=n_samples)
    else:
        # Already a callable / generator factory -- trust the caller.
        gen = representative_data  # type: ignore[assignment]

    converter.representative_dataset = gen  # type: ignore[attr-defined]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # type: ignore[attr-defined]

    if mode == QuantizationMode.INT8_STRICT:
        converter.inference_input_type = tf.int8  # type: ignore[attr-defined]
        converter.inference_output_type = tf.int8  # type: ignore[attr-defined]
    # else mode == INT8 -- I/O tensors stay FP32 so the model is a drop-in.
