"""
Centralized application configuration.

Uses ``pydantic-settings`` so values can be overridden by environment variables
or a local ``.env`` file. The skeleton stays minimal in Phase 1 — actual fields
will grow during Phases 3 (training), 5 (serving) and 8 (monitoring).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from deepvision.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_SEED,
    IMG_SIZE_EFFICIENTNET,
)

#: Repository root, derived from the location of this file
#: (``src/deepvision/config.py`` → repo root is two parents up from ``src``).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Top-level configuration object.

    Override any field via environment variables (prefix ``DEEPVISION_``)
    or via a ``.env`` file at the repo root.

    Examples
    --------
    >>> import os
    >>> os.environ["DEEPVISION_SEED"] = "1234"
    >>> Settings().seed
    1234
    """

    model_config = SettingsConfigDict(
        env_prefix="DEEPVISION_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----------------- Reproducibility -----------------
    seed: int = Field(
        default=DEFAULT_SEED,
        description="Global random seed for Python, NumPy and TensorFlow.",
    )

    # ----------------- Paths --------------------------
    repo_root: Path = Field(
        default=REPO_ROOT,
        description="Absolute path to the repository root.",
    )
    models_dir: Path = Field(
        default=REPO_ROOT / "models",
        description="Local directory holding training artifacts (gitignored).",
    )
    data_dir: Path = Field(
        default=REPO_ROOT / "data",
        description="Local directory holding datasets (gitignored).",
    )
    mlruns_dir: Path = Field(
        default=REPO_ROOT / "mlruns",
        description="Local MLflow tracking store (gitignored).",
    )

    # ----------------- Training (used in Phase 3) -----
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, le=1024)
    epochs: int = Field(default=DEFAULT_EPOCHS, ge=1, le=500)
    image_size: int = Field(default=IMG_SIZE_EFFICIENTNET, ge=32, le=512)

    # ----------------- Logging ------------------------
    log_level: str = Field(
        default="INFO",
        description="Root logger level (DEBUG / INFO / WARNING / ERROR).",
    )
    log_format: str = Field(
        default="text",
        description="Log output format: 'text' for development, 'json' for production.",
    )


def get_settings() -> Settings:
    """Return a freshly instantiated Settings object.

    Notes
    -----
    Re-instantiating allows tests to override env vars without holding stale state.
    For runtime callers the cost is negligible (Settings reads the environment lazily).
    """
    return Settings()
