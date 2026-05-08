"""
Image upload validation and preprocessing for the FastAPI serving layer.

The audit (Bug B3) flagged that ``app.py`` crashed on PNG with alpha channel
or grayscale images. This module enforces three guarantees:

1. The uploaded payload is below ``MAX_IMAGE_BYTES``.
2. Pillow can decode it as an image.
3. The decoded image is converted to RGB before resizing.

Resize is done via Pillow's ``BILINEAR`` (acceptable for the 32x32-to-160x160
range we care about) to avoid pulling OpenCV into the serving image.
"""

from __future__ import annotations

import io
from typing import Final

import numpy as np
from PIL import Image, UnidentifiedImageError

from deepvision.constants import IMG_SIZE_EFFICIENTNET, NUM_CHANNELS

#: Hard upper bound on uploaded image size (10 MB).
MAX_IMAGE_BYTES: Final[int] = 10 * 1024 * 1024

#: Maximum image side length accepted (post-decode), to prevent decompression bombs.
MAX_IMAGE_SIDE: Final[int] = 4096

#: Tuple of MIME types accepted by the API. Keep aligned with schemas.MetaResponse.
ACCEPTED_MIME_TYPES: Final[tuple[str, ...]] = (
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
)


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation."""


def validate_payload_size(payload: bytes, *, max_bytes: int = MAX_IMAGE_BYTES) -> None:
    """Raise :class:`ImageValidationError` if ``payload`` exceeds the limit."""
    if len(payload) == 0:
        raise ImageValidationError("Empty payload.")
    if len(payload) > max_bytes:
        raise ImageValidationError(f"Payload too large: {len(payload)} bytes > {max_bytes} bytes.")


def decode_image(payload: bytes) -> Image.Image:
    """Decode ``payload`` into a Pillow Image, or raise.

    The image is **not** converted to RGB here so the caller can read the
    original mode if it wants to log the discovery (RGBA, L, etc.).
    """
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()  # force decoding so corrupt files raise here, not later
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError(f"Could not decode image: {exc}") from exc

    width, height = image.size
    if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
        raise ImageValidationError(
            f"Image too large: {width}x{height} (max {MAX_IMAGE_SIDE} px per side)."
        )
    return image


def preprocess_for_efficientnet(
    image: Image.Image,
    *,
    target_size: int = IMG_SIZE_EFFICIENTNET,
) -> np.ndarray:
    """Convert a Pillow image to a uint8 NumPy array ready for EfficientNet.

    EfficientNet expects raw uint8 inputs in ``[0, 255]``; the model integrates
    its own ``Rescaling`` layer.

    Returns
    -------
    np.ndarray
        Array of shape ``(1, target_size, target_size, NUM_CHANNELS)`` and dtype uint8.
    """
    rgb = image.convert("RGB")
    if rgb.size != (target_size, target_size):
        rgb = rgb.resize((target_size, target_size), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.shape != (target_size, target_size, NUM_CHANNELS):
        raise ImageValidationError(
            f"Unexpected post-resize shape: {arr.shape}, "
            f"expected ({target_size}, {target_size}, {NUM_CHANNELS})."
        )
    return arr[np.newaxis, ...]


def load_image_for_inference(payload: bytes) -> np.ndarray:
    """End-to-end: validate + decode + preprocess one image upload.

    Returns a batched uint8 array ready to feed the EfficientNet pipeline.
    Raises :class:`ImageValidationError` on any validation failure.
    """
    validate_payload_size(payload)
    image = decode_image(payload)
    return preprocess_for_efficientnet(image)
