"""Unit tests for ``deepvision.serving.schemas``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepvision import __version__
from deepvision.constants import CLASS_NAMES_EN, NUM_CLASSES
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


def test_prediction_item_valid() -> None:
    item = PredictionItem(rank=1, class_index=3, class_name="cat", probability=0.7)
    assert item.rank == 1
    assert item.probability == 0.7


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"rank": 0, "class_index": 1, "class_name": "x", "probability": 0.5}, "rank"),
        (
            {"rank": 1, "class_index": -1, "class_name": "x", "probability": 0.5},
            "class_index",
        ),
        (
            {"rank": 1, "class_index": 0, "class_name": "x", "probability": 1.5},
            "probability",
        ),
        (
            {"rank": 1, "class_index": 0, "class_name": "x", "probability": -0.1},
            "probability",
        ),
    ],
)
def test_prediction_item_invalid(kwargs: dict, field: str) -> None:
    with pytest.raises(ValidationError) as info:
        PredictionItem(**kwargs)
    assert field in str(info.value)


def test_predict_response_round_trip() -> None:
    payload = PredictResponse(
        model_name="m",
        model_version="0.1",
        top_k=2,
        predictions=[
            PredictionItem(rank=1, class_index=0, class_name="a", probability=0.9),
            PredictionItem(rank=2, class_index=1, class_name="b", probability=0.05),
        ],
        inference_time_ms=12.3,
    )
    dumped = payload.model_dump()
    assert dumped["model_name"] == "m"
    assert dumped["top_k"] == 2
    assert len(dumped["predictions"]) == 2


def test_health_response_defaults() -> None:
    h = HealthResponse()
    assert h.status == "ok"
    assert h.package_version == __version__


def test_ready_response_payload() -> None:
    r = ReadyResponse(ready=True, model_loaded=True, model_name="m", model_version="0.1")
    assert r.ready is True
    assert r.model_loaded is True


def test_meta_response_defaults() -> None:
    m = MetaResponse()
    assert m.package == "deepvision"
    assert m.version == __version__
    assert m.num_classes == NUM_CLASSES
    assert m.class_names == list(CLASS_NAMES_EN)
    assert "png" in m.expected_image_format


def test_batch_predict_response_round_trip() -> None:
    payload = BatchPredictResponse(
        model_name="m",
        model_version="0.1",
        top_k=1,
        items=[
            BatchPredictionItem(
                index=0,
                filename="a.png",
                predictions=[
                    PredictionItem(rank=1, class_index=0, class_name="a", probability=0.5)
                ],
            ),
        ],
        inference_time_ms=42.0,
    )
    dumped = payload.model_dump()
    assert dumped["items"][0]["filename"] == "a.png"


def test_error_response_minimal() -> None:
    err = ErrorResponse(error="bad", detail="payload too small")
    assert err.error == "bad"
    assert "small" in err.detail
