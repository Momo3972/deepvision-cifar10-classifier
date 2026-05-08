"""
Centralized application configuration.

Uses ``pydantic-settings`` so values can be overridden by environment variables
or a local ``.env`` file. The skeleton stays minimal in Phase 1; actual fields
grow during Phases 3 (training), 5 (serving) and 8 (monitoring).
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
#: (``src/deepvision/config.py`` -> repo root is two parents up from ``src``).
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

    # ----------------- Serving (Phase 5) --------------
    api_host: str = Field(
        default="0.0.0.0",
        description="Host interface the FastAPI server binds to.",
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="TCP port exposed by the FastAPI server.",
    )
    api_reload: bool = Field(
        default=False,
        description="Enable uvicorn's auto-reload (development only).",
    )
    api_key: str | None = Field(
        default=None,
        description=(
            "Optional API key. When set, every /predict* request must carry "
            "the matching value in the 'X-API-Key' header. "
            "Leave unset to disable authentication."
        ),
    )
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allow-origin list. Use ['*'] for an open demo, restrict for prod.",
    )
    max_image_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1024,
        description="Hard cap on the size of an uploaded image, in bytes.",
    )
    max_batch_size: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Maximum number of images accepted by /predict_batch in one call.",
    )
    model_path: Path | None = Field(
        default=None,
        description=(
            "Optional path to a trained .keras / SavedModel artefact. "
            "When unset, the API serves an EfficientNetB0 with random weights "
            "(useful for CI smoke tests and image build verification)."
        ),
    )
    serving_model_name: str = Field(
        default="efficientnet_b0_transfer",
        description="Identifier surfaced in API responses and Prometheus labels.",
    )
    serving_model_version: str = Field(
        default="0.0.0-untrained",
        description="Semantic version surfaced in API responses and Prometheus labels.",
    )

    # ----------------- Streamlit (Phase 6) ------------
    streamlit_host: str = Field(
        default="0.0.0.0",
        description="Host interface the Streamlit demo binds to.",
    )
    streamlit_port: int = Field(
        default=8501,
        ge=1,
        le=65535,
        description="TCP port exposed by the Streamlit demo.",
    )

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
