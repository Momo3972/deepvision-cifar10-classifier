"""Tests for :mod:`deepvision.data.augmentation`."""

from __future__ import annotations

import numpy as np

from deepvision.data.augmentation import (
    AugmentationConfig,
    build_augmentation_pipeline,
)


def test_default_config_has_all_augmentations_enabled() -> None:
    config = AugmentationConfig()
    assert config.horizontal_flip is True
    assert config.rotation > 0
    assert config.zoom > 0
    assert config.contrast > 0


def test_pipeline_preserves_image_shape() -> None:
    pipeline = build_augmentation_pipeline()
    fake_batch = np.random.rand(2, 32, 32, 3).astype(np.float32)
    augmented = pipeline(fake_batch, training=True).numpy()
    assert augmented.shape == fake_batch.shape


def test_pipeline_with_disabled_options_has_fewer_layers() -> None:
    config = AugmentationConfig(horizontal_flip=False, rotation=0.0, zoom=0.0, contrast=0.0)
    pipeline = build_augmentation_pipeline(config)
    assert len(pipeline.layers) == 0
