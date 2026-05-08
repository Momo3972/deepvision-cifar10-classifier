"""Unit tests for ``deepvision.serving.inference.InferenceEngine``.

The TensorFlow-backed branch is exercised; tests use ``weights=None`` so they
don't depend on ImageNet downloads, and run on CPU in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from deepvision.constants import CLASS_NAMES_EN, NUM_CLASSES
from deepvision.serving.inference import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_VERSION,
    InferenceEngine,
)


@pytest.fixture(scope="module")
def loaded_engine() -> InferenceEngine:
    """Fresh engine with a fast random-weights EfficientNetB0."""
    engine = InferenceEngine()
    engine.load()
    return engine


def test_engine_defaults() -> None:
    engine = InferenceEngine()
    assert engine.model_name == DEFAULT_MODEL_NAME
    assert engine.model_version == DEFAULT_MODEL_VERSION
    assert engine.is_loaded is False


def test_engine_load_marks_loaded(loaded_engine: InferenceEngine) -> None:
    assert loaded_engine.is_loaded is True


def test_engine_load_idempotent(loaded_engine: InferenceEngine) -> None:
    inner = loaded_engine._model
    loaded_engine.load()
    assert loaded_engine._model is inner


def test_engine_predict_returns_topk(loaded_engine: InferenceEngine) -> None:
    batch = np.zeros((1, 32, 32, 3), dtype=np.float32)
    items, latency_ms = loaded_engine.predict(batch, top_k=3)
    assert len(items) == 3
    indices = [idx for idx, _, _ in items]
    assert all(0 <= idx < NUM_CLASSES for idx in indices)
    names = [name for _, name, _ in items]
    assert all(name in CLASS_NAMES_EN for name in names)
    probs = [prob for _, _, prob in items]
    # Sorted descending.
    assert probs == sorted(probs, reverse=True)
    # Probabilities should be valid (random weights => softmax outputs in [0, 1]).
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert latency_ms >= 0.0


def test_engine_predict_clamps_topk(loaded_engine: InferenceEngine) -> None:
    batch = np.zeros((1, 32, 32, 3), dtype=np.float32)
    items, _ = loaded_engine.predict(batch, top_k=99)
    assert len(items) == NUM_CLASSES


def test_engine_predict_rejects_invalid_topk(loaded_engine: InferenceEngine) -> None:
    batch = np.zeros((1, 32, 32, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="top_k"):
        loaded_engine.predict(batch, top_k=0)


def test_engine_predict_lazy_loads() -> None:
    engine = InferenceEngine()
    assert engine.is_loaded is False
    batch = np.zeros((1, 32, 32, 3), dtype=np.float32)
    items, _ = engine.predict(batch, top_k=1)
    assert engine.is_loaded is True
    assert len(items) == 1
