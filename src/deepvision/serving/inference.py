"""
Thin inference wrapper used by the FastAPI app.

The :class:`InferenceEngine` lazy-loads the model the first time
:meth:`predict` is called, so the API process can boot quickly and not pay
the TensorFlow startup cost when only ``/health`` and ``/meta`` are hit
(important for CI smoke tests and Kubernetes probes).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from deepvision.constants import CLASS_NAMES_EN
from deepvision.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from tensorflow.keras import Model

log = get_logger(__name__)

#: Default model identifier exposed in the response and the metrics labels.
DEFAULT_MODEL_NAME: Final[str] = "efficientnet_b0_transfer"

#: Default model version. The real semantic version will be set when the
#: trained artefact is registered in MLflow (Phase 5+).
DEFAULT_MODEL_VERSION: Final[str] = "0.0.0-untrained"


class InferenceEngine:
    """Single-model inference wrapper.

    Two ways to instantiate:

    - ``InferenceEngine()`` — builds an EfficientNetB0 with random weights
      (``weights=None``). Useful in CI and on machines that have not yet
      downloaded the ImageNet weights or trained on CIFAR-10. The predictions
      are statistically meaningless but shape-correct.
    - ``InferenceEngine(model_path=Path("models/best.keras"))`` — loads a
      previously trained model from disk via ``tf.keras.models.load_model``.

    The underlying model is loaded lazily the first time :meth:`predict` is
    called, so importing this module does not pull TensorFlow into memory.
    """

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        model_version: str = DEFAULT_MODEL_VERSION,
    ) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.model_version = model_version
        self._model: Model | None = None

    @property
    def is_loaded(self) -> bool:
        """``True`` once the underlying model is in memory."""
        return self._model is not None

    def load(self) -> None:
        """Force-load the model (idempotent)."""
        if self._model is not None:
            return
        if self.model_path is None:
            log.info("Loading EfficientNetB0 with random weights (no model_path provided).")
            from deepvision.models.efficientnet import build_efficientnet

            self._model = build_efficientnet(weights=None)
        else:
            import tensorflow as tf

            log.info("Loading model from %s", self.model_path)
            self._model = tf.keras.models.load_model(str(self.model_path))
        # Warmup forward pass so first user-facing request is not slow.
        self._warmup()

    def _warmup(self) -> None:
        if self._model is None:
            return
        dummy = np.zeros(self._input_shape(), dtype=np.float32)
        self._model.predict(dummy, verbose=0)
        log.info("Model warmup complete.")

    def _input_shape(self) -> tuple[int, ...]:
        # Default: (1, 32, 32, 3) — matches build_efficientnet's input.
        if self._model is None:
            return (1, 32, 32, 3)
        # Use model.input_shape if available, else fallback.
        try:
            shape = tuple(int(x) if x is not None else 1 for x in self._model.input_shape)
        except (AttributeError, TypeError):
            shape = (1, 32, 32, 3)
        if len(shape) == 4 and shape[0] != 1:
            shape = (1, *shape[1:])
        return shape

    def predict(
        self,
        image_batch: np.ndarray,
        *,
        top_k: int = 3,
    ) -> tuple[list[tuple[int, str, float]], float]:
        """Run inference on a batched image array.

        Parameters
        ----------
        image_batch
            Array of shape ``(1, H, W, C)`` ready for the model.
        top_k
            Number of top predictions to return.

        Returns
        -------
        (predictions, inference_time_ms)
            ``predictions`` is a list of ``(class_index, class_name, probability)``
            tuples sorted by descending probability.
        """
        if self._model is None:
            self.load()
        assert self._model is not None  # mypy

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if top_k > len(CLASS_NAMES_EN):
            top_k = len(CLASS_NAMES_EN)

        start = time.perf_counter()
        probs = self._model.predict(image_batch, verbose=0)[0]
        inference_time_ms = (time.perf_counter() - start) * 1000.0

        # ``argsort`` ascending; reverse to get top-k.
        ranked = np.argsort(probs)[::-1][:top_k]
        items = [(int(idx), CLASS_NAMES_EN[idx], float(probs[idx])) for idx in ranked]
        return items, inference_time_ms
