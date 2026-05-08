"""Unit tests for ``deepvision.serving.preprocess``."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from deepvision.constants import IMG_SIZE_EFFICIENTNET, NUM_CHANNELS
from deepvision.serving.preprocess import (
    ACCEPTED_MIME_TYPES,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_SIDE,
    ImageValidationError,
    decode_image,
    load_image_for_inference,
    preprocess_for_efficientnet,
    validate_payload_size,
)


def _png_bytes(width: int, height: int, mode: str = "RGB") -> bytes:
    """Build a tiny PNG payload of the requested size and color mode."""
    rng = np.random.default_rng(seed=42)
    if mode == "RGB":
        arr = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    elif mode == "L":
        arr = rng.integers(0, 255, size=(height, width), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
    elif mode == "RGBA":
        arr = rng.integers(0, 255, size=(height, width, 4), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
    else:  # pragma: no cover — guarded by callers
        raise ValueError(mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# validate_payload_size
# ---------------------------------------------------------------------------


def test_validate_payload_size_accepts_normal_payload() -> None:
    payload = b"\x00" * 1024
    validate_payload_size(payload)  # should not raise


def test_validate_payload_size_rejects_empty() -> None:
    with pytest.raises(ImageValidationError, match="Empty"):
        validate_payload_size(b"")


def test_validate_payload_size_rejects_oversized() -> None:
    payload = b"\x00" * (MAX_IMAGE_BYTES + 1)
    with pytest.raises(ImageValidationError, match="too large"):
        validate_payload_size(payload)


def test_validate_payload_size_respects_custom_limit() -> None:
    payload = b"\x00" * 11
    with pytest.raises(ImageValidationError, match="too large"):
        validate_payload_size(payload, max_bytes=10)


# ---------------------------------------------------------------------------
# decode_image
# ---------------------------------------------------------------------------


def test_decode_image_returns_pillow_image() -> None:
    img = decode_image(_png_bytes(8, 8))
    assert isinstance(img, Image.Image)
    assert img.size == (8, 8)


def test_decode_image_rejects_garbage_bytes() -> None:
    with pytest.raises(ImageValidationError, match="Could not decode"):
        decode_image(b"this is not an image")


def test_decode_image_rejects_giant_dimensions() -> None:
    side = MAX_IMAGE_SIDE + 1
    with pytest.raises(ImageValidationError, match="too large"):
        decode_image(_png_bytes(side, 16))


# ---------------------------------------------------------------------------
# preprocess_for_efficientnet
# ---------------------------------------------------------------------------


def test_preprocess_for_efficientnet_returns_expected_shape() -> None:
    pil_image = Image.new("RGB", (40, 60), color=(255, 0, 0))
    arr = preprocess_for_efficientnet(pil_image)
    assert arr.shape == (1, IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET, NUM_CHANNELS)
    assert arr.dtype == np.uint8


def test_preprocess_for_efficientnet_converts_rgba_to_rgb() -> None:
    pil_image = Image.new("RGBA", (40, 40), color=(0, 255, 0, 128))
    arr = preprocess_for_efficientnet(pil_image)
    assert arr.shape[-1] == NUM_CHANNELS  # alpha dropped


def test_preprocess_for_efficientnet_converts_grayscale_to_rgb() -> None:
    pil_image = Image.new("L", (40, 40), color=128)
    arr = preprocess_for_efficientnet(pil_image)
    assert arr.shape[-1] == NUM_CHANNELS


def test_preprocess_for_efficientnet_skips_resize_when_already_correct() -> None:
    pil_image = Image.new("RGB", (IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET), color=(0, 0, 255))
    arr = preprocess_for_efficientnet(pil_image)
    # Last channel of a uniform blue should be 255 everywhere.
    assert int(arr[0, 0, 0, 2]) == 255


# ---------------------------------------------------------------------------
# load_image_for_inference (end-to-end)
# ---------------------------------------------------------------------------


def test_load_image_for_inference_happy_path_rgb() -> None:
    arr = load_image_for_inference(_png_bytes(20, 20))
    assert arr.shape == (1, IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET, NUM_CHANNELS)


def test_load_image_for_inference_handles_rgba_png() -> None:
    arr = load_image_for_inference(_png_bytes(20, 20, mode="RGBA"))
    assert arr.shape[-1] == 3  # alpha dropped


def test_load_image_for_inference_handles_grayscale_png() -> None:
    arr = load_image_for_inference(_png_bytes(20, 20, mode="L"))
    assert arr.shape[-1] == 3


def test_load_image_for_inference_rejects_text_payload() -> None:
    with pytest.raises(ImageValidationError):
        load_image_for_inference(b"definitely not a PNG")


def test_load_image_for_inference_rejects_empty_payload() -> None:
    with pytest.raises(ImageValidationError):
        load_image_for_inference(b"")


# ---------------------------------------------------------------------------
# Constants surface area
# ---------------------------------------------------------------------------


def test_accepted_mime_types_covers_jpeg_png_webp() -> None:
    assert "image/jpeg" in ACCEPTED_MIME_TYPES
    assert "image/png" in ACCEPTED_MIME_TYPES
    assert "image/webp" in ACCEPTED_MIME_TYPES


def test_max_image_bytes_is_at_least_one_megabyte() -> None:
    assert MAX_IMAGE_BYTES >= 1_000_000
