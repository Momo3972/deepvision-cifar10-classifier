"""
Configurable image-augmentation pipeline.

Used by EfficientNet's training graph (and optionally by the custom CNN) to
improve robustness against orientation, scale and contrast variations.

Phase 4 will add Albumentations / Keras-CV based augmentations such as Mixup,
CutMix and RandAugment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Hyperparameters of the augmentation pipeline.

    Each field is the maximal magnitude of a Keras random transformation,
    expressed as a fraction of the image dimensions or pixel range.
    """

    horizontal_flip: bool = True
    rotation: float = 0.10
    zoom: float = 0.10
    contrast: float = 0.10


def build_augmentation_pipeline(config: AugmentationConfig | None = None):
    """Return a Keras Sequential implementing the augmentation pipeline.

    Parameters
    ----------
    config
        Override the default magnitudes.

    Returns
    -------
    tensorflow.keras.Sequential
        A model that can be inserted at the top of a training graph.

    Notes
    -----
    The Keras ``RandomFlip``/``RandomRotation``/``RandomZoom``/``RandomContrast``
    layers are deliberately used instead of preprocessing the dataset on disk:
    augmentation is then re-randomized at every epoch and runs on the GPU when
    one is available.
    """
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        RandomContrast,
        RandomFlip,
        RandomRotation,
        RandomZoom,
    )

    cfg = config or AugmentationConfig()
    layers = []
    if cfg.horizontal_flip:
        layers.append(RandomFlip("horizontal"))
    if cfg.rotation > 0:
        layers.append(RandomRotation(cfg.rotation))
    if cfg.zoom > 0:
        layers.append(RandomZoom(cfg.zoom))
    if cfg.contrast > 0:
        layers.append(RandomContrast(cfg.contrast))
    return Sequential(layers, name="data_augmentation")
