"""
Tests for :func:`deepvision.utils.seed.set_seed`.
"""

from __future__ import annotations

import os
import random

import numpy as np

from deepvision.utils.seed import set_seed


def test_set_seed_returns_seed() -> None:
    """``set_seed`` returns the seed it applied."""
    assert set_seed(123) == 123


def test_set_seed_makes_python_random_deterministic() -> None:
    """Two consecutive calls with the same seed produce identical random sequences."""
    set_seed(42)
    sequence_a = [random.random() for _ in range(5)]
    set_seed(42)
    sequence_b = [random.random() for _ in range(5)]
    assert sequence_a == sequence_b


def test_set_seed_makes_numpy_deterministic() -> None:
    """Same expectation on NumPy's default RNG."""
    set_seed(7)
    array_a = np.random.rand(4, 4)
    set_seed(7)
    array_b = np.random.rand(4, 4)
    np.testing.assert_array_equal(array_a, array_b)


def test_set_seed_writes_pythonhashseed() -> None:
    """``PYTHONHASHSEED`` is exported so subprocesses inherit the seed."""
    set_seed(2024)
    assert os.environ.get("PYTHONHASHSEED") == "2024"
