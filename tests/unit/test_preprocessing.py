"""Tests for :mod:`deepvision.data.preprocessing`."""

from __future__ import annotations

import numpy as np
import pytest

from deepvision.data.preprocessing import (
    denormalize_to_uint8,
    normalize_to_unit,
    one_hot_encode,
    validate_image_array,
)


def _fake_images(n: int = 4) -> np.ndarray:
    return np.random.randint(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)


def test_normalize_to_unit_scales_to_unit_range() -> None:
    images = _fake_images(8)
    out = normalize_to_unit(images)
    assert out.dtype == np.float32
    assert out.shape == images.shape
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_normalize_then_denormalize_roundtrips() -> None:
    images = _fake_images(4)
    roundtrip = denormalize_to_uint8(normalize_to_unit(images))
    assert roundtrip.dtype == np.uint8
    assert np.array_equal(roundtrip, images)


def test_one_hot_encode_produces_correct_shape_and_sum() -> None:
    labels = np.array([0, 1, 2, 9])
    encoded = one_hot_encode(labels, num_classes=10)
    assert encoded.shape == (4, 10)
    assert encoded.dtype == np.float32
    np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(4))


def test_one_hot_encode_handles_nested_label_array() -> None:
    """The original notebook stores labels as ``(n, 1)``."""
    labels = np.array([[0], [3], [9]])
    encoded = one_hot_encode(labels, num_classes=10)
    assert encoded.shape == (3, 10)
    assert encoded[0, 0] == 1
    assert encoded[1, 3] == 1
    assert encoded[2, 9] == 1


def test_one_hot_encode_rejects_out_of_range_labels() -> None:
    with pytest.raises(ValueError, match="Labels must be in"):
        one_hot_encode(np.array([0, 10]), num_classes=10)


def test_one_hot_encode_handles_empty_input() -> None:
    encoded = one_hot_encode(np.array([], dtype=np.int64), num_classes=10)
    assert encoded.shape == (0, 10)


def test_validate_image_array_accepts_correct_shape() -> None:
    validate_image_array(_fake_images(2))  # should not raise


@pytest.mark.parametrize(
    ("bad_shape", "match"),
    [
        ((32, 32, 3), "Expected 4D"),
        ((1, 28, 28, 3), "Expected image size"),
        ((1, 32, 32, 1), "Expected .* channels"),
    ],
)
def test_validate_image_array_rejects_wrong_shape(bad_shape: tuple[int, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_image_array(np.zeros(bad_shape, dtype=np.uint8))


def test_denormalize_rejects_uint8_input() -> None:
    with pytest.raises(TypeError):
        denormalize_to_uint8(_fake_images(1))
