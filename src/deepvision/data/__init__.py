"""
Data loading, splitting, preprocessing and augmentation.

Public API
----------
- :class:`deepvision.data.loader.CifarSplit`: dataclass holding the split arrays.
- :func:`deepvision.data.loader.load_cifar10`: canonical entry-point.
- :func:`deepvision.data.loader.compute_dataset_hash`: traceability helper.
- :func:`deepvision.data.preprocessing.normalize_to_unit`: uint8 -> float [0, 1].
- :func:`deepvision.data.preprocessing.one_hot_encode`: integer labels -> one-hot.
- :func:`deepvision.data.preprocessing.validate_image_array`: shape contract.
- :class:`deepvision.data.augmentation.AugmentationConfig`: pipeline hyperparams.
- :func:`deepvision.data.augmentation.build_augmentation_pipeline`: Keras model.
"""

from __future__ import annotations

from deepvision.data.augmentation import (
    AugmentationConfig,
    build_augmentation_pipeline,
)
from deepvision.data.loader import (
    CifarSplit,
    compute_dataset_hash,
    load_cifar10,
)
from deepvision.data.preprocessing import (
    denormalize_to_uint8,
    normalize_to_unit,
    one_hot_encode,
    validate_image_array,
)

__all__ = [
    "AugmentationConfig",
    "CifarSplit",
    "build_augmentation_pipeline",
    "compute_dataset_hash",
    "denormalize_to_uint8",
    "load_cifar10",
    "normalize_to_unit",
    "one_hot_encode",
    "validate_image_array",
]
