"""Tests for the Phase 6 Streamlit refonte (``deepvision.streamlit_app``).

The Streamlit module must:

- Correct the four nominal bugs identified by the audit (sections 4.4 / 9):

  * **B1** -- no double softmax (probabilities come straight from the engine).
  * **B2** -- preprocessing targets ``IMG_SIZE_EFFICIENTNET`` (160), not 32x32.
  * **B3** -- RGB conversion is applied so RGBA / grayscale uploads do not
    crash with a shape mismatch.
  * **B4** -- the deprecated ``use_column_width`` is replaced by
    ``use_container_width``.

- Validate uploaded payloads using magic bytes (defence in depth on top of
  the file extension reported by the browser).

- Wire Grad-CAM via the existing ``deepvision.evaluation.interpretability``
  module rather than duplicating logic.

Each behavioural test asserts directly on the property guaranteed by the
audit so a regression on any of those bugs fails loudly.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from deepvision import streamlit_app
from deepvision.constants import (
    CLASS_NAMES_EN,
    IMG_SIZE_EFFICIENTNET,
    NUM_CLASSES,
)
from deepvision.serving.preprocess import ImageValidationError

# ---------------------------------------------------------------------------
# Stub engine -- mirrors tests/unit/test_serving_api.py to avoid pulling
# TensorFlow into Streamlit unit tests.
# ---------------------------------------------------------------------------


class StubEngine:
    """Deterministic stand-in for :class:`InferenceEngine` in Streamlit tests.

    Predicts class 0 with probability 1.0, the rest 0.0. Records every
    pre-processed batch on ``self.calls`` so tests can assert on shape.
    """

    def __init__(
        self,
        *,
        model_name: str = "stub_model",
        model_version: str = "0.0.0-stub",
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.is_loaded = True
        self.calls: list[tuple[int, ...]] = []
        # Sentinel object so a Grad-CAM test can verify the property is read.
        self.model = object()

    def load(self) -> None:
        self.is_loaded = True

    def predict(
        self,
        image_batch: np.ndarray,
        *,
        top_k: int = 3,
    ) -> tuple[list[tuple[int, str, float]], float]:
        self.calls.append(tuple(image_batch.shape))
        ranked: list[tuple[int, str, float]] = []
        for i in range(min(top_k, NUM_CLASSES)):
            prob = 1.0 if i == 0 else 0.0
            ranked.append((i, CLASS_NAMES_EN[i], prob))
        return ranked, 0.123


# ---------------------------------------------------------------------------
# Helpers -- synthetic images for B3 (RGB / RGBA / grayscale).
# ---------------------------------------------------------------------------


def _make_image(mode: str, *, size: int = 64) -> Image.Image:
    rng = np.random.default_rng(seed=0)
    if mode == "RGB":
        arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    elif mode == "RGBA":
        arr = rng.integers(0, 255, size=(size, size, 4), dtype=np.uint8)
    elif mode == "L":
        arr = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    else:  # pragma: no cover -- safety net
        raise ValueError(mode)
    return Image.fromarray(arr, mode=mode)


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Magic-byte detection
# ---------------------------------------------------------------------------


def test_detect_mime_recognises_png() -> None:
    payload = _to_png_bytes(_make_image("RGB", size=8))
    assert streamlit_app.detect_mime(payload) == "image/png"


def test_detect_mime_recognises_jpeg() -> None:
    buf = io.BytesIO()
    _make_image("RGB", size=8).save(buf, format="JPEG")
    assert streamlit_app.detect_mime(buf.getvalue()) == "image/jpeg"


def test_detect_mime_recognises_webp() -> None:
    buf = io.BytesIO()
    _make_image("RGB", size=8).save(buf, format="WEBP")
    assert streamlit_app.detect_mime(buf.getvalue()) == "image/webp"


def test_detect_mime_returns_none_for_unknown() -> None:
    # Plain text masquerading as an upload.
    assert streamlit_app.detect_mime(b"NOT_AN_IMAGE_AT_ALL_PLEASE") is None


def test_detect_mime_returns_none_for_payload_too_short() -> None:
    assert streamlit_app.detect_mime(b"\xff\xd8") is None


def test_validate_mime_rejects_text_renamed_as_image() -> None:
    """A ``.txt`` renamed ``image.png`` must still be rejected (defence in depth)."""
    with pytest.raises(ImageValidationError, match="Unsupported file format"):
        streamlit_app.validate_mime(b"hello, this is not a real PNG file at all\n")


def test_validate_mime_accepts_real_png() -> None:
    payload = _to_png_bytes(_make_image("RGB", size=8))
    assert streamlit_app.validate_mime(payload) == "image/png"


# ---------------------------------------------------------------------------
# predict_one -- core inference pipeline (B1, B2, B3 corrections)
# ---------------------------------------------------------------------------


def test_predict_one_resizes_to_efficientnet_size_not_32() -> None:
    """B2: the preprocessing must target 160x160, not the legacy 32x32."""
    engine = StubEngine()
    image = _make_image("RGB", size=200)  # arbitrary input size
    result = streamlit_app.predict_one(engine, image)
    # The engine recorded the batch shape it was fed.
    assert engine.calls, "predict() should have been called once"
    batch_shape = engine.calls[0]
    assert batch_shape == (1, IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET, 3)
    # And IMG_SIZE_EFFICIENTNET must not be 32 (regression guard).
    assert IMG_SIZE_EFFICIENTNET != 32
    # Returned batch has the expected shape too.
    assert result["batch"].shape == (1, IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET, 3)


def test_predict_one_handles_rgba_input() -> None:
    """B3: a PNG with alpha channel must not crash."""
    engine = StubEngine()
    image = _make_image("RGBA", size=80)
    result = streamlit_app.predict_one(engine, image)
    assert result["batch"].shape == (1, IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET, 3)


def test_predict_one_handles_grayscale_input() -> None:
    """B3: a single-channel 'L' image must not crash."""
    engine = StubEngine()
    image = _make_image("L", size=80)
    result = streamlit_app.predict_one(engine, image)
    assert result["batch"].shape == (1, IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET, 3)


def test_predict_one_returns_engine_probabilities_unmodified() -> None:
    """B1: the StubEngine returns prob=1.0 for class 0; no extra softmax must
    re-normalise that to a softer value."""
    engine = StubEngine()
    image = _make_image("RGB", size=64)
    result = streamlit_app.predict_one(engine, image)
    top_idx, top_name, top_prob = result["top_k"][0]
    assert top_idx == 0
    assert top_name == CLASS_NAMES_EN[0]
    assert top_prob == pytest.approx(1.0)


def test_predict_one_returns_top_3() -> None:
    engine = StubEngine()
    image = _make_image("RGB", size=64)
    result = streamlit_app.predict_one(engine, image)
    assert len(result["top_k"]) == 3


def test_predict_one_includes_inference_time() -> None:
    engine = StubEngine()
    image = _make_image("RGB", size=64)
    result = streamlit_app.predict_one(engine, image)
    assert "inference_time_ms" in result
    assert isinstance(result["inference_time_ms"], float)


# ---------------------------------------------------------------------------
# Grad-CAM wiring -- mocked so we don't pull TensorFlow into the test.
# ---------------------------------------------------------------------------


def test_render_gradcam_overlay_calls_interpretability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grad-CAM must be delegated to ``deepvision.evaluation.interpretability``.

    We monkeypatch both ``grad_cam`` and ``overlay_heatmap_on_image`` so the
    test runs without TensorFlow and checks that the correct arguments are
    forwarded.
    """
    from deepvision.evaluation import interpretability

    seen: dict[str, Any] = {}

    def fake_grad_cam(model: Any, image: np.ndarray, *, target_class: int) -> np.ndarray:
        seen["grad_cam_args"] = (model, image.shape, target_class)
        return np.zeros(image.shape[:2], dtype=np.float32)

    def fake_overlay(image: np.ndarray, heatmap: np.ndarray, *, alpha: float = 0.4) -> np.ndarray:
        seen["overlay_args"] = (image.shape, heatmap.shape, alpha)
        return np.zeros_like(image, dtype=np.uint8)

    monkeypatch.setattr(interpretability, "grad_cam", fake_grad_cam)
    monkeypatch.setattr(interpretability, "overlay_heatmap_on_image", fake_overlay)

    engine = StubEngine()
    batch = np.zeros((1, IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET, 3), dtype=np.uint8)
    overlay = streamlit_app.render_gradcam_overlay(engine, batch, target_class=3)

    # The mocked ``grad_cam`` was called with the engine's Keras model
    # (StubEngine.model is a sentinel object), the (H, W, C) image slice,
    # and the requested target class.
    assert seen["grad_cam_args"][0] is engine.model
    assert seen["grad_cam_args"][1] == (IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET, 3)
    assert seen["grad_cam_args"][2] == 3
    assert overlay.shape == (IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET, 3)


# ---------------------------------------------------------------------------
# Static checks on the module source -- regression guards against the four
# audit bugs and the duplicated CLASS_NAMES.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def module_source() -> str:
    return Path(streamlit_app.__file__).read_text(encoding="utf-8")


def test_module_does_not_use_deprecated_use_column_width(module_source: str) -> None:
    """B4: ``use_column_width`` was deprecated in Streamlit 1.32+.

    We look for the keyword-argument form ``use_column_width=`` so the test is
    not tripped by the docstring that legitimately discusses the deprecation.
    """
    assert "use_column_width=" not in module_source


def test_module_uses_use_container_width(module_source: str) -> None:
    """B4 corrective: the new flag must be present."""
    assert "use_container_width" in module_source


def test_module_does_not_apply_extra_softmax(module_source: str) -> None:
    """B1: the legacy ``tf.nn.softmax(predictions[0])`` call must be gone.

    The InferenceEngine returns probabilities already normalised by the
    model's final ``Dense(softmax)`` layer; re-applying softmax flattens the
    distribution and produces a misleading confidence indicator.
    """
    # Match the call form ``tf.nn.softmax(`` so the docstring mentioning the
    # legacy bug (``:func:`tf.nn.softmax```) is not a false positive.
    assert "tf.nn.softmax(" not in module_source
    # The module must not import tensorflow either; that's the engine's job.
    assert "import tensorflow" not in module_source


def test_module_does_not_resize_to_32x32(module_source: str) -> None:
    """B2: the legacy ``image.resize((32, 32))`` call must be gone."""
    assert "(32, 32)" not in module_source
    assert "(32,32)" not in module_source


def test_module_does_not_redefine_class_names(module_source: str) -> None:
    """The legacy ``app.py`` duplicated ``CLASS_NAMES`` -- a known divergence
    risk flagged by the audit (B12). The refonte must import them from
    ``deepvision.constants`` instead.
    """
    # No local literal list of CIFAR-10 class names.
    for name in ("Avion", "Voiture", "Grenouille", "airplane", "automobile"):
        assert f'"{name}"' not in module_source, (
            f"{name!r} is hard-coded in streamlit_app.py; import from constants."
        )
    # And the CLASS_NAMES_FR / _EN imports must be present.
    assert "from deepvision.constants import" in module_source
    assert "CLASS_NAMES_FR" in module_source


def test_find_examples_dir_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the optional examples folder is absent, the function returns None
    silently rather than raising."""
    monkeypatch.setattr(streamlit_app, "__file__", str(tmp_path / "fake" / "streamlit_app.py"))
    assert streamlit_app.find_examples_dir() is None
