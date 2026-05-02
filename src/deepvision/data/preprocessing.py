"""
Image and label preprocessing utilities.

The MLP and the custom CNN expect normalized float images in ``[0, 1]``,
whereas EfficientNet uses raw uint8 images and applies its own internal
normalization via ``Rescaling``. Both code paths share the same one-hot
label encoder.
"""

from __future__ import annotations

import numpy as np

from deepvision.constants import IMG_SIZE_NATIVE, NUM_CHANNELS, NUM_CLASSES


def normalize_to_unit(images: np.ndarray) -> np.ndarray:
    """Scale uint8 images in ``[0, 255]`` to float32 images in ``[0, 1]``.

    Parameters
    ----------
    images
        Array of shape ``(n, H, W, C)`` and dtype ``uint8``.

    Returns
    -------
    np.ndarray
        Same shape, dtype ``float32``, values in ``[0, 1]``.
    """
    validate_image_array(images)
    return images.astype(np.float32) / 255.0


def denormalize_to_uint8(images: np.ndarray) -> np.ndarray:
    """Inverse of :func:`normalize_to_unit`.

    Useful for displaying augmented samples or feeding EfficientNet, which
    expects uint8 inputs.
    """
    if images.dtype not in (np.float32, np.float64):
        raise TypeError(f"Expected float images, got dtype={images.dtype}")
    clipped = np.clip(images * 255.0, 0, 255)
    return clipped.astype(np.uint8)


def one_hot_encode(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Convert integer class labels into a one-hot float32 matrix.

    Parameters
    ----------
    labels
        Integer labels with shape ``(n,)`` or ``(n, 1)``.
    num_classes
        Number of distinct classes. Defaults to CIFAR-10's ten.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(n, num_classes)``.
    """
    flat = np.asarray(labels).reshape(-1).astype(np.int64)
    if flat.size == 0:
        return np.zeros((0, num_classes), dtype=np.float32)
    if flat.min() < 0 or flat.max() >= num_classes:
        raise ValueError(
            f"Labels must be in [0, {num_classes - 1}], got min={flat.min()} max={flat.max()}"
        )
    encoded = np.zeros((flat.size, num_classes), dtype=np.float32)
    encoded[np.arange(flat.size), flat] = 1.0
    return encoded


def validate_image_array(
    images: np.ndarray,
    *,
    expected_height: int = IMG_SIZE_NATIVE,
    expected_width: int = IMG_SIZE_NATIVE,
    expected_channels: int = NUM_CHANNELS,
) -> None:
    """Raise ``ValueError`` if ``images`` does not satisfy CIFAR-10's contract.

    A valid array has 4 dimensions ``(N, H, W, C)`` and the expected
    height / width / channel sizes.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected 4D array (N, H, W, C), got shape {images.shape}")
    _, height, width, channels = images.shape
    if height != expected_height or width != expected_width:
        raise ValueError(
            f"Expected image size {expected_height}x{expected_width}, got {height}x{width}"
        )
    if channels != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {channels}")
