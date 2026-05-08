"""
Pydantic v2 schemas for the FastAPI serving layer.

The schemas are intentionally tight: every public field has a description so
that the auto-generated OpenAPI documentation (Swagger / ReDoc) is meaningful
for any consumer of the API.

Note on Pydantic v2 protected namespaces
----------------------------------------
Pydantic v2 reserves names starting with ``model_`` for its own machinery
(``model_dump``, ``model_validate``, ``model_config``...). To still surface
``model_name`` / ``model_version`` in the OpenAPI schema (consumers expect
those exact field names) we whitelist them per-class via
``model_config = {"protected_namespaces": ()}``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from deepvision import __version__
from deepvision.constants import CLASS_NAMES_EN


class PredictionItem(BaseModel):
    """One ranked prediction returned by ``/predict``."""

    rank: int = Field(..., ge=1, description="1-based rank of this prediction.")
    class_index: int = Field(..., ge=0, description="Class index in [0, num_classes).")
    class_name: str = Field(..., description="Human-readable class label.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Softmax probability.")


class PredictResponse(BaseModel):
    """Response returned by the ``/predict`` endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(..., description="Identifier of the model used.")
    model_version: str = Field(..., description="Semantic version of the model.")
    top_k: int = Field(..., ge=1, description="Number of returned predictions.")
    predictions: list[PredictionItem] = Field(
        ..., description="Top-k predictions sorted by descending probability."
    )
    inference_time_ms: float = Field(
        ..., ge=0.0, description="Wall-clock inference time in milliseconds."
    )


class BatchPredictionItem(BaseModel):
    """One image's worth of predictions inside a batch response."""

    index: int = Field(..., ge=0, description="0-based index of this image in the batch.")
    filename: str = Field(default="", description="Original filename, if provided by the client.")
    predictions: list[PredictionItem] = Field(..., description="Top-k predictions for this image.")


class BatchPredictResponse(BaseModel):
    """Response returned by the ``/predict_batch`` endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(..., description="Identifier of the model used.")
    model_version: str = Field(..., description="Semantic version of the model.")
    top_k: int = Field(..., ge=1, description="Number of returned predictions per image.")
    items: list[BatchPredictionItem] = Field(..., description="Per-image predictions.")
    inference_time_ms: float = Field(
        ..., ge=0.0, description="Total wall-clock inference time in milliseconds."
    )


class HealthResponse(BaseModel):
    """Liveness response returned by ``/health``."""

    status: str = Field(default="ok", description="Always 'ok' if the process responds.")
    package_version: str = Field(default=__version__, description="deepvision package version.")


class ReadyResponse(BaseModel):
    """Readiness response returned by ``/ready``.

    A serving instance is "ready" when the model is loaded and able to serve
    inference. ``model_loaded`` is False until the first warmup call succeeds.
    """

    model_config = ConfigDict(protected_namespaces=())

    ready: bool = Field(..., description="True when the model is loaded and ready.")
    model_loaded: bool = Field(..., description="True when the model artefact is in memory.")
    model_name: str = Field(..., description="Identifier of the model.")
    model_version: str = Field(..., description="Semantic version of the model.")


class MetaResponse(BaseModel):
    """Static metadata returned by ``/meta``."""

    package: str = Field(default="deepvision")
    version: str = Field(default=__version__)
    num_classes: int = Field(default=len(CLASS_NAMES_EN))
    class_names: list[str] = Field(default_factory=lambda: list(CLASS_NAMES_EN))
    expected_image_format: list[str] = Field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp"],
        description="Image MIME types accepted by /predict.",
    )
    max_image_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Hard limit on uploaded image size, in bytes.",
    )


class ErrorResponse(BaseModel):
    """Standard error envelope used for non-2xx responses."""

    error: str = Field(..., description="Short error code (e.g. 'invalid_image').")
    detail: str = Field(..., description="Human-readable explanation.")
