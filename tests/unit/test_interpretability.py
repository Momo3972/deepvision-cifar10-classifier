"""Tests for :mod:`deepvision.evaluation.interpretability` (Grad-CAM)."""

from __future__ import annotations

import numpy as np
import pytest

from deepvision.evaluation.interpretability import (
    find_last_conv_layer,
    grad_cam,
    overlay_heatmap_on_image,
)
from deepvision.models import build_cnn, build_efficientnet, build_mlp


def test_find_last_conv_layer_in_cnn() -> None:
    name = find_last_conv_layer(build_cnn())
    # Our VGG-style CNN names its last Conv2D "conv3_2".
    assert name == "conv3_2"


def test_find_last_conv_layer_in_efficientnet() -> None:
    name = find_last_conv_layer(build_efficientnet(weights=None))
    # EfficientNet's last Conv2D is the Conv2D right before global average pooling.
    # We don't pin a specific name (Keras may rename) — just check it exists.
    assert isinstance(name, str)
    assert len(name) > 0


def test_find_last_conv_layer_raises_on_pure_mlp() -> None:
    with pytest.raises(ValueError, match="No Conv2D layer"):
        find_last_conv_layer(build_mlp())


def test_grad_cam_returns_heatmap_of_input_resolution() -> None:
    """Heatmap shape must match the input spatial size."""
    model = build_cnn()
    image = np.random.rand(32, 32, 3).astype(np.float32)
    heatmap = grad_cam(model, image)
    assert heatmap.shape == (32, 32)
    assert heatmap.dtype == np.float32


def test_grad_cam_heatmap_is_normalized() -> None:
    model = build_cnn()
    image = np.random.rand(32, 32, 3).astype(np.float32)
    heatmap = grad_cam(model, image)
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0


def test_grad_cam_accepts_batched_singleton_input() -> None:
    model = build_cnn()
    image = np.random.rand(1, 32, 32, 3).astype(np.float32)
    heatmap = grad_cam(model, image)
    assert heatmap.shape == (32, 32)


def test_grad_cam_rejects_invalid_shape() -> None:
    model = build_cnn()
    with pytest.raises(ValueError, match="Expected"):
        grad_cam(model, np.zeros((10, 32, 32, 3), dtype=np.float32))


def test_overlay_returns_uint8_with_correct_shape() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    heatmap = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float32)
    out = overlay_heatmap_on_image(image, heatmap)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8
