"""
Project-wide pytest configuration and fixtures.

Add reusable fixtures here. Module-specific fixtures should live next to
their tests (e.g. ``tests/unit/conftest.py``).
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _silence_tf_warnings() -> None:
    """Silence TensorFlow's chatty C++ logs during tests.

    Set BEFORE TensorFlow is imported. Auto-applied to every test for clean output.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
