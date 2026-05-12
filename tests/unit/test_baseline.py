"""Unit tests for :mod:`deepvision.monitoring.baseline`.

The :class:`Baseline` dataclass and the synthetic-images helper are tested
without TensorFlow. The full ``compute_baseline`` path exercises model
surgery, so it is covered indirectly via
``test_monitoring_server.test_synthetic_baseline_round_trip`` to keep the
TF import out of this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepvision.constants import IMG_SIZE_EFFICIENTNET, NUM_CHANNELS
from deepvision.monitoring.baseline import Baseline, synthetic_reference_images

# ---------------------------------------------------------------------------
# Baseline dataclass
# ---------------------------------------------------------------------------


def _make_baseline(n: int = 32, d: int = 16) -> Baseline:
    rng = np.random.default_rng(0)
    return Baseline(
        embeddings=rng.normal(size=(n, d)),
        energies=rng.normal(size=(n,)),
        model_name="stub_model",
        model_version="0.0.0-stub",
    )


def test_baseline_properties() -> None:
    b = _make_baseline(n=128, d=64)
    assert b.n_samples == 128
    assert b.feature_dim == 64
    assert b.model_name == "stub_model"
    assert b.model_version == "0.0.0-stub"


def test_baseline_save_and_load_roundtrip(tmp_path: Path) -> None:
    original = _make_baseline()
    target = tmp_path / "baseline.npz"
    original.save(target)
    assert target.exists()

    restored = Baseline.load(target)
    np.testing.assert_array_equal(restored.embeddings, original.embeddings)
    np.testing.assert_array_equal(restored.energies, original.energies)
    assert restored.model_name == original.model_name
    assert restored.model_version == original.model_version
    assert restored.n_samples == original.n_samples
    assert restored.feature_dim == original.feature_dim


def test_baseline_save_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "subdir" / "baseline.npz"
    _make_baseline().save(target)
    assert target.exists()


def test_baseline_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Baseline.load(tmp_path / "does_not_exist.npz")


# ---------------------------------------------------------------------------
# synthetic_reference_images
# ---------------------------------------------------------------------------


def test_synthetic_images_have_expected_shape_and_dtype() -> None:
    images = synthetic_reference_images(n=4)
    assert images.shape == (4, IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET, NUM_CHANNELS)
    assert images.dtype == np.uint8


def test_synthetic_images_are_deterministic() -> None:
    a = synthetic_reference_images(n=8, seed=42)
    b = synthetic_reference_images(n=8, seed=42)
    np.testing.assert_array_equal(a, b)


def test_synthetic_images_seed_changes_output() -> None:
    a = synthetic_reference_images(n=8, seed=1)
    b = synthetic_reference_images(n=8, seed=2)
    assert not np.array_equal(a, b)
