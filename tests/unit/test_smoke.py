"""
Smoke tests — the package imports cleanly and exposes basic metadata.
"""

from __future__ import annotations

import re

import deepvision


def test_package_version_is_defined() -> None:
    """The package must expose a ``__version__`` attribute."""
    assert hasattr(deepvision, "__version__")
    assert isinstance(deepvision.__version__, str)
    assert len(deepvision.__version__) > 0


def test_package_version_is_pep440_compliant() -> None:
    """Version string follows MAJOR.MINOR.PATCH with optional pre-release suffix."""
    pattern = r"^\d+\.\d+\.\d+([a-zA-Z0-9.+-]*)?$"
    assert re.match(pattern, deepvision.__version__), (
        f"Version {deepvision.__version__!r} is not PEP 440 compliant"
    )


def test_package_metadata_is_defined() -> None:
    """Package exposes author and license."""
    assert deepvision.__author__
    assert deepvision.__license__ == "MIT"


def test_subpackages_importable() -> None:
    """Each declared subpackage must be importable (even if empty in Phase 1)."""
    import importlib

    for sub in (
        "deepvision.data",
        "deepvision.models",
        "deepvision.training",
        "deepvision.evaluation",
        "deepvision.serving",
        "deepvision.monitoring",
        "deepvision.utils",
    ):
        module = importlib.import_module(sub)
        assert module is not None
