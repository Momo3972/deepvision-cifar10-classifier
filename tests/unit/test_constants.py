"""
Tests for :mod:`deepvision.constants`.
"""

from __future__ import annotations

from deepvision.constants import (
    CLASS_NAMES_EN,
    CLASS_NAMES_FR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_SEED,
    IMG_SIZE_EFFICIENTNET,
    IMG_SIZE_NATIVE,
    NUM_CHANNELS,
    NUM_CLASSES,
)


def test_class_names_have_correct_length() -> None:
    """CIFAR-10 has exactly 10 classes in both languages."""
    assert len(CLASS_NAMES_FR) == NUM_CLASSES
    assert len(CLASS_NAMES_EN) == NUM_CLASSES


def test_class_names_are_unique() -> None:
    """No duplicates allowed in class names."""
    assert len(set(CLASS_NAMES_FR)) == NUM_CLASSES
    assert len(set(CLASS_NAMES_EN)) == NUM_CLASSES


def test_class_names_pairs_are_consistent() -> None:
    """The order of FR and EN class names must match for index-based lookups."""
    expected_fr_to_en = {
        "Avion": "airplane",
        "Voiture": "automobile",
        "Oiseau": "bird",
        "Chat": "cat",
        "Cerf": "deer",
        "Chien": "dog",
        "Grenouille": "frog",
        "Cheval": "horse",
        "Bateau": "ship",
        "Camion": "truck",
    }
    actual = dict(zip(CLASS_NAMES_FR, CLASS_NAMES_EN, strict=True))
    assert actual == expected_fr_to_en


def test_image_size_constants_are_positive() -> None:
    """Sanity checks on dimension constants."""
    assert IMG_SIZE_NATIVE == 32
    assert IMG_SIZE_EFFICIENTNET >= IMG_SIZE_NATIVE
    assert NUM_CHANNELS == 3


def test_default_hyperparameters_are_reasonable() -> None:
    """Bounds check: defaults must be within sane ranges."""
    assert DEFAULT_SEED >= 0
    assert 1 <= DEFAULT_BATCH_SIZE <= 1024
    assert 1 <= DEFAULT_EPOCHS <= 500
