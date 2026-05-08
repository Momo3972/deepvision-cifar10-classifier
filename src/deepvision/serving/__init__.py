"""
Serving layer: FastAPI REST API.

Public API
----------
- ``deepvision.serving.api``        : FastAPI app and ``create_app`` factory.
- ``deepvision.serving.schemas``    : Pydantic request/response schemas.
- ``deepvision.serving.preprocess`` : image validation + RGB conversion + resize.
- ``deepvision.serving.inference``  : thin wrapper around the loaded model.
- ``deepvision.serving.prometheus`` : Prometheus instrumentation.

Convenience re-exports
----------------------
``from deepvision.serving import create_app, InferenceEngine, ImageValidationError``
"""

from __future__ import annotations

from deepvision.serving.inference import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_VERSION,
    InferenceEngine,
)
from deepvision.serving.preprocess import (
    ACCEPTED_MIME_TYPES,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_SIDE,
    ImageValidationError,
    decode_image,
    load_image_for_inference,
    preprocess_for_efficientnet,
    validate_payload_size,
)
from deepvision.serving.schemas import (
    BatchPredictionItem,
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    MetaResponse,
    PredictionItem,
    PredictResponse,
    ReadyResponse,
)

__all__ = [
    "ACCEPTED_MIME_TYPES",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_VERSION",
    "MAX_IMAGE_BYTES",
    "MAX_IMAGE_SIDE",
    "BatchPredictResponse",
    "BatchPredictionItem",
    "ErrorResponse",
    "HealthResponse",
    "ImageValidationError",
    "InferenceEngine",
    "MetaResponse",
    "PredictResponse",
    "PredictionItem",
    "ReadyResponse",
    "decode_image",
    "load_image_for_inference",
    "preprocess_for_efficientnet",
    "validate_payload_size",
]


def __getattr__(name: str):
    """Lazy-load FastAPI symbols.

    ``api.py`` imports FastAPI which is heavy; importing the serving package
    just for the schemas should not pull it in. ``create_app`` and ``app`` are
    exposed via this PEP 562 hook so they only load when explicitly requested.
    """
    if name in {"create_app", "app", "make_api_key_dependency"}:
        from deepvision.serving import api

        return getattr(api, name)
    raise AttributeError(f"module 'deepvision.serving' has no attribute {name!r}")
