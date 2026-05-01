"""
Project-wide constants for the deepvision package.

Single source of truth for CIFAR-10 class names, image dimensions and
training hyperparameters used across data loading, training, evaluation
and serving modules.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# CIFAR-10 dataset
# ---------------------------------------------------------------------------

#: Original CIFAR-10 image size in pixels (height = width).
IMG_SIZE_NATIVE: Final[int] = 32

#: Image size used as input of EfficientNetB0 in this project.
#: 160x160 is a tradeoff between fidelity (>32) and CPU latency (<224).
IMG_SIZE_EFFICIENTNET: Final[int] = 160

#: Number of color channels (RGB).
NUM_CHANNELS: Final[int] = 3

#: Number of classes in CIFAR-10.
NUM_CLASSES: Final[int] = 10

#: French human-readable class names — kept consistent with the original notebook.
CLASS_NAMES_FR: Final[tuple[str, ...]] = (
    "Avion",
    "Voiture",
    "Oiseau",
    "Chat",
    "Cerf",
    "Chien",
    "Grenouille",
    "Cheval",
    "Bateau",
    "Camion",
)

#: English class names — used in API responses and English README.
CLASS_NAMES_EN: Final[tuple[str, ...]] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

#: Default seed used for reproducibility across NumPy / TensorFlow / Python random.
DEFAULT_SEED: Final[int] = 42

#: Default batch size for training.
DEFAULT_BATCH_SIZE: Final[int] = 64

#: Default number of epochs (overridable via config).
DEFAULT_EPOCHS: Final[int] = 20
