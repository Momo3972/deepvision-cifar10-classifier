"""Tests for :mod:`deepvision.data.loader`.

The "real" test that downloads CIFAR-10 from Keras is expensive
(~163 MB on first run). It is wrapped in a pytest mark so the suite stays fast
by default; run it explicitly with ``pytest -m integration``.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from deepvision.data.loader import (
    CifarSplit,
    _assert_balanced,
    compute_dataset_hash,
    load_cifar10,
)


def _fake_cifar(n_train: int = 100, n_test: int = 20):
    rng = np.random.default_rng(seed=0)
    x_train = rng.integers(0, 256, (n_train, 32, 32, 3), dtype=np.uint8)
    y_train = np.tile(np.arange(10, dtype=np.uint8), n_train // 10).reshape(-1, 1)
    x_test = rng.integers(0, 256, (n_test, 32, 32, 3), dtype=np.uint8)
    y_test = np.tile(np.arange(10, dtype=np.uint8), n_test // 10).reshape(-1, 1)
    return x_train, y_train, x_test, y_test


def test_compute_dataset_hash_is_deterministic() -> None:
    images = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    labels = np.zeros((4, 1), dtype=np.uint8)
    h1 = compute_dataset_hash(images, labels)
    h2 = compute_dataset_hash(images, labels)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_compute_dataset_hash_changes_with_content() -> None:
    images = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    labels = np.zeros((4, 1), dtype=np.uint8)
    h1 = compute_dataset_hash(images, labels)
    images[0, 0, 0, 0] = 1
    h2 = compute_dataset_hash(images, labels)
    assert h1 != h2


def test_load_cifar10_returns_stratified_split() -> None:
    fake = _fake_cifar(n_train=8000, n_test=2000)
    with patch("deepvision.data.loader._load_cifar10_raw", return_value=fake):
        split = load_cifar10(test_size=0.20, seed=42)
    assert isinstance(split, CifarSplit)
    assert split.n_train + split.n_test == 10_000
    assert split.n_test == pytest.approx(2_000, abs=2)
    assert split.x_train.shape[1:] == (32, 32, 3)


def test_load_cifar10_is_deterministic_with_seed() -> None:
    fake = _fake_cifar(n_train=8000, n_test=2000)
    with patch("deepvision.data.loader._load_cifar10_raw", return_value=fake):
        a = load_cifar10(seed=42)
        b = load_cifar10(seed=42)
    np.testing.assert_array_equal(a.x_train, b.x_train)
    assert a.dataset_hash == b.dataset_hash


def test_load_cifar10_rejects_invalid_test_size() -> None:
    with pytest.raises(ValueError, match="test_size"):
        load_cifar10(test_size=0.0)
    with pytest.raises(ValueError, match="test_size"):
        load_cifar10(test_size=1.0)


def test_assert_balanced_accepts_balanced_dataset() -> None:
    labels = np.tile(np.arange(10), 100).reshape(-1, 1)
    _assert_balanced(labels, name="train")  # should not raise


def test_assert_balanced_rejects_unbalanced_dataset() -> None:
    labels = np.array([[0]] * 1000 + [[1]] * 10)  # massively imbalanced
    with pytest.raises(ValueError, match="contains 2 classes"):
        _assert_balanced(labels, name="train")


def test_cifarsplit_summary_is_json_friendly() -> None:
    fake = _fake_cifar(n_train=8000, n_test=2000)
    with patch("deepvision.data.loader._load_cifar10_raw", return_value=fake):
        split = load_cifar10(seed=42)
    summary = split.summary()
    # Must contain the keys an MLflow run would log.
    for key in ("n_train", "n_test", "test_size", "seed", "image_shape", "dataset_hash"):
        assert key in summary


@pytest.mark.integration
def test_real_cifar10_load_smoke() -> None:
    """Slow smoke test that actually downloads CIFAR-10. Disabled by default."""
    split = load_cifar10()
    assert split.n_train == 48_000
    assert split.n_test == 12_000
