"""
CIFAR-10 dataset loader.

Provides a single canonical entry-point ``load_cifar10`` that reproducibly
applies a stratified train / test split. The original notebook used the
native Keras 50 000 / 10 000 split for EfficientNet but a custom 80 / 20
stratified split for MLP and CNN, which led to data leakage between the
EfficientNet training set and the evaluation set used for the comparison
table. Phase 2 fixes this by making every model see the **same** train and
test partition.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from deepvision.constants import DEFAULT_SEED, NUM_CLASSES
from deepvision.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CifarSplit:
    """A reproducible train / test split of the CIFAR-10 dataset.

    Attributes
    ----------
    x_train, y_train
        Training images (uint8, shape ``(n_train, 32, 32, 3)``) and labels
        (uint8, shape ``(n_train, 1)``).
    x_test, y_test
        Test images and labels with the same shape conventions.
    seed
        Seed used for the stratified split.
    test_size
        Fraction of the data assigned to the test set.
    dataset_hash
        SHA-256 fingerprint of the union (images + labels), used for
        traceability in MLflow runs.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    seed: int
    test_size: float
    dataset_hash: str

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.x_test.shape[0])

    def summary(self) -> dict[str, int | float | str]:
        """Return a JSON-friendly summary suitable for MLflow logging."""
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "test_size": self.test_size,
            "seed": self.seed,
            "image_shape": str(tuple(self.x_train.shape[1:])),
            "dataset_hash": self.dataset_hash,
        }


def compute_dataset_hash(images: np.ndarray, labels: np.ndarray) -> str:
    """Return a stable SHA-256 fingerprint of an (images, labels) pair.

    Same arrays produce the same hash regardless of run; differing arrays
    produce different hashes with overwhelming probability. Useful for
    traceability ("what data did this run see?") and reproducibility checks.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(images).tobytes())
    h.update(np.ascontiguousarray(labels).tobytes())
    h.update(str(images.shape).encode())
    h.update(str(labels.shape).encode())
    return h.hexdigest()


def _load_cifar10_raw() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lazily import Keras and return the native CIFAR-10 split.

    Importing Keras is deferred to keep this module importable in environments
    without TensorFlow (e.g. lint-only CI jobs).
    """
    from tensorflow.keras.datasets import cifar10  # noqa: PLC0415 — lazy

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return x_train, y_train, x_test, y_test


def load_cifar10(
    test_size: float = 0.20,
    seed: int = DEFAULT_SEED,
    *,
    verify_balance: bool = True,
) -> CifarSplit:
    """Load CIFAR-10 and return a reproducible stratified train / test split.

    Parameters
    ----------
    test_size
        Fraction of the merged dataset used for the test set. Must be in
        ``(0, 1)``. Default is 0.20 (60 000 -> 48 000 train, 12 000 test).
    seed
        Random seed used by ``sklearn.train_test_split``.
    verify_balance
        When True, asserts that each class is approximately balanced in the
        resulting split (required for stratification to work as expected).

    Returns
    -------
    CifarSplit
        Frozen dataclass containing the four arrays plus traceability fields.

    Notes
    -----
    The native Keras CIFAR-10 split is **not** used directly because the
    original notebook mixed it with a custom 80/20 split, leading to
    train/test contamination between models. This function returns a single
    canonical split shared by all training pipelines (MLP, CNN, EfficientNet).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size!r}")

    log.info("Loading CIFAR-10 (this triggers a Keras download on first use)…")
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = _load_cifar10_raw()

    x_full = np.concatenate([x_train_raw, x_test_raw], axis=0)
    y_full = np.concatenate([y_train_raw, y_test_raw], axis=0)

    log.info(
        "Stratified split: %d total images, test_size=%.2f, seed=%d",
        x_full.shape[0],
        test_size,
        seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x_full,
        y_full,
        test_size=test_size,
        random_state=seed,
        stratify=y_full,
    )

    if verify_balance:
        _assert_balanced(y_train, "train")
        _assert_balanced(y_test, "test")

    dataset_hash = compute_dataset_hash(x_full, y_full)
    log.info(
        "Split ready: n_train=%d, n_test=%d, dataset_hash=%s",
        x_train.shape[0],
        x_test.shape[0],
        dataset_hash[:12] + "…",
    )

    return CifarSplit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        seed=seed,
        test_size=test_size,
        dataset_hash=dataset_hash,
    )


def _assert_balanced(labels: np.ndarray, name: str, *, tolerance: float = 0.02) -> None:
    """Assert that every class appears with at most ``tolerance`` deviation.

    For a balanced dataset like CIFAR-10 with 10 classes, each class should
    represent about 10 % of any subset. We tolerate a 2-percentage-point
    deviation by default (i.e. each class between 8 % and 12 %).
    """
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) != NUM_CLASSES:
        raise ValueError(
            f"{name} split contains {len(unique)} classes, expected {NUM_CLASSES}"
        )

    fractions = counts / counts.sum()
    expected = 1.0 / NUM_CLASSES
    deviation = float(np.max(np.abs(fractions - expected)))
    if deviation > tolerance:
        raise AssertionError(
            f"{name} split is unbalanced: max class deviation {deviation:.4f} "
            f"exceeds tolerance {tolerance}"
        )
