"""Tests for :mod:`deepvision.config`.

Focuses on the regression discovered during the Phase 7 Docker smoke-test:
``docker compose`` substitutes ``${DEEPVISION_MODEL_PATH:-}`` into the
container environment as the empty string when the host ``.env`` line is
blank, and pydantic used to coerce ``""`` into ``Path(".")`` -- a truthy
path that pointed at the current working directory and caused Keras 3 to
fail with ``ValueError: File format not supported`` when the inference
engine tried to load it.

The :func:`Settings._empty_string_is_none` validator now normalises empty /
whitespace strings back to ``None`` for ``model_path``. The tests below
pin that contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deepvision.config import Settings, get_settings


def test_settings_default_model_path_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPVISION_MODEL_PATH", raising=False)
    settings = Settings()
    assert settings.model_path is None


def test_empty_string_env_var_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Phase 7 docker-compose substitutes ``""`` -- it must NOT become Path('.')."""
    monkeypatch.setenv("DEEPVISION_MODEL_PATH", "")
    settings = Settings()
    assert settings.model_path is None, (
        "Empty env var should be normalised to None to avoid loading the cwd as a model"
    )


def test_whitespace_only_env_var_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPVISION_MODEL_PATH", "   ")
    settings = Settings()
    assert settings.model_path is None


def test_non_empty_env_var_yields_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A real path must be preserved as a :class:`pathlib.Path`."""
    target = tmp_path / "best.keras"
    monkeypatch.setenv("DEEPVISION_MODEL_PATH", str(target))
    settings = Settings()
    assert settings.model_path == target
    assert isinstance(settings.model_path, Path)


def test_get_settings_returns_fresh_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_settings`` should re-read the environment so tests can override env vars."""
    monkeypatch.delenv("DEEPVISION_MODEL_PATH", raising=False)
    first = get_settings()
    assert first.model_path is None

    monkeypatch.setenv("DEEPVISION_MODEL_PATH", "/tmp/model.keras")
    second = get_settings()
    assert second.model_path == Path("/tmp/model.keras")


def test_serving_defaults_match_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spot-check the serving defaults stayed aligned with the audit prescription."""
    for var in (
        "DEEPVISION_API_HOST",
        "DEEPVISION_API_PORT",
        "DEEPVISION_API_KEY",
        "DEEPVISION_MAX_IMAGE_BYTES",
        "DEEPVISION_MAX_BATCH_SIZE",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = Settings()
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.api_key is None
    assert settings.max_image_bytes == 10 * 1024 * 1024  # 10 MB cap from audit 7.4.
    assert settings.max_batch_size == 16
