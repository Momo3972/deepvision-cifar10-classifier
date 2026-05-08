"""End-to-end tests for the FastAPI serving layer.

We use FastAPI's :class:`TestClient` (built on :mod:`httpx`) so the test path
exercises the same ASGI pipeline uvicorn would in production: middlewares,
dependency injection, and the Prometheus instrumentation.

The real :class:`~deepvision.serving.inference.InferenceEngine` would pull
TensorFlow into every test; we use a small ``StubEngine`` instead to exercise
the API logic alone (TF integration is covered in ``test_serving_inference``).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from deepvision import __version__
from deepvision.config import Settings
from deepvision.constants import CLASS_NAMES_EN, NUM_CLASSES
from deepvision.serving.api import create_app

if TYPE_CHECKING:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Stub inference engine — quacks like InferenceEngine without TensorFlow.
# ---------------------------------------------------------------------------


class StubEngine:
    """Deterministic stand-in for :class:`InferenceEngine` in API tests.

    Always predicts class 0 with probability 1.0 and uniform 0.0 elsewhere.
    Records every batch it sees on ``self.calls`` for assertions.
    """

    def __init__(
        self,
        *,
        model_name: str = "stub_model",
        model_version: str = "0.0.0-stub",
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.is_loaded = True
        self.calls: list[tuple[int, ...]] = []

    def load(self) -> None:
        self.is_loaded = True

    def predict(
        self,
        image_batch: np.ndarray,
        *,
        top_k: int = 3,
    ) -> tuple[list[tuple[int, str, float]], float]:
        self.calls.append(tuple(image_batch.shape))
        ranked: list[tuple[int, str, float]] = []
        for i in range(min(top_k, NUM_CLASSES)):
            prob = 1.0 if i == 0 else 0.0
            ranked.append((i, CLASS_NAMES_EN[i], prob))
        return ranked, 0.123


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _png_bytes(width: int = 16, height: int = 16, mode: str = "RGB") -> bytes:
    rng = np.random.default_rng(seed=0)
    if mode == "RGB":
        arr = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    elif mode == "RGBA":
        arr = rng.integers(0, 255, size=(height, width, 4), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
    else:  # pragma: no cover — safety net
        raise ValueError(mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def stub_engine() -> StubEngine:
    return StubEngine()


@pytest.fixture
def client(stub_engine: StubEngine) -> TestClient:
    settings = Settings()
    app = create_app(settings=settings, engine=stub_engine)  # type: ignore[arg-type]
    return TestClient(app)


# ---------------------------------------------------------------------------
# Operational endpoints
# ---------------------------------------------------------------------------


def test_index_renders_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert __version__ in response.text


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["package_version"] == __version__


def test_ready_reflects_engine_state(client: TestClient, stub_engine: StubEngine) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["model_loaded"] is True
    assert body["model_name"] == stub_engine.model_name


def test_meta_exposes_classes_and_limits(client: TestClient) -> None:
    response = client.get("/meta")
    assert response.status_code == 200
    body = response.json()
    assert body["num_classes"] == NUM_CLASSES
    assert body["class_names"] == list(CLASS_NAMES_EN)
    assert body["max_image_bytes"] >= 1_000_000


def test_metrics_exposes_prometheus_format(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "deepvision_http_requests_total" in response.text


def test_openapi_schema_available(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["version"] == __version__
    assert "/predict" in schema["paths"]
    assert "/predict_batch" in schema["paths"]


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


def test_predict_happy_path(client: TestClient, stub_engine: StubEngine) -> None:
    payload = _png_bytes()
    response = client.post(
        "/predict",
        files={"file": ("a.png", payload, "image/png")},
        params={"top_k": 2},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_name"] == stub_engine.model_name
    assert body["top_k"] == 2
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["rank"] == 1
    assert body["predictions"][0]["class_index"] == 0
    assert body["predictions"][0]["probability"] == 1.0
    assert body["inference_time_ms"] >= 0.0
    assert len(stub_engine.calls) == 1


def test_predict_handles_rgba_png(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("a.png", _png_bytes(mode="RGBA"), "image/png")},
    )
    assert response.status_code == 200, response.text


def test_predict_rejects_unsupported_mime(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("doc.pdf", b"%PDF-1.4 dummy", "application/pdf")},
    )
    assert response.status_code == 415
    body = response.json()
    assert body["error"] == "unsupported_media_type"


def test_predict_rejects_garbage_image(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("a.png", b"not a real image", "image/png")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "invalid_image"


def test_predict_rejects_invalid_top_k(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("a.png", _png_bytes(), "image/png")},
        params={"top_k": 0},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_top_k"


def test_predict_clamps_top_k_to_num_classes(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("a.png", _png_bytes(), "image/png")},
        params={"top_k": 99},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["top_k"] == NUM_CLASSES
    assert len(body["predictions"]) == NUM_CLASSES


# ---------------------------------------------------------------------------
# /predict_batch
# ---------------------------------------------------------------------------


def test_predict_batch_happy_path(client: TestClient, stub_engine: StubEngine) -> None:
    files = [
        ("files", ("a.png", _png_bytes(), "image/png")),
        ("files", ("b.png", _png_bytes(), "image/png")),
    ]
    response = client.post("/predict_batch", files=files, params={"top_k": 1})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["items"]) == 2
    assert body["items"][0]["filename"] == "a.png"
    assert body["items"][0]["index"] == 0
    assert body["items"][1]["filename"] == "b.png"
    assert len(stub_engine.calls) == 2


def test_predict_batch_rejects_empty_batch(client: TestClient) -> None:
    response = client.post("/predict_batch", files=[])
    # FastAPI's File(...) makes the parameter required → 422.
    assert response.status_code in (400, 422)


def test_predict_batch_rejects_oversized_batch() -> None:
    settings = Settings(max_batch_size=2)
    app = create_app(settings=settings, engine=StubEngine())  # type: ignore[arg-type]
    client = TestClient(app)
    files = [("files", (f"img_{i}.png", _png_bytes(), "image/png")) for i in range(3)]
    response = client.post("/predict_batch", files=files)
    assert response.status_code == 413
    assert response.json()["error"] == "batch_too_large"


def test_predict_batch_propagates_unsupported_mime(client: TestClient) -> None:
    files = [
        ("files", ("a.png", _png_bytes(), "image/png")),
        ("files", ("b.pdf", b"%PDF-", "application/pdf")),
    ]
    response = client.post("/predict_batch", files=files)
    assert response.status_code == 415


# ---------------------------------------------------------------------------
# Auth (X-API-Key)
# ---------------------------------------------------------------------------


def _auth_client(api_key: str) -> TestClient:
    settings = Settings(api_key=api_key)
    app = create_app(settings=settings, engine=StubEngine())  # type: ignore[arg-type]
    return TestClient(app)


def test_predict_requires_key_when_configured() -> None:
    client = _auth_client("s3cret")
    response = client.post(
        "/predict",
        files={"file": ("a.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 401


def test_predict_accepts_correct_key() -> None:
    client = _auth_client("s3cret")
    response = client.post(
        "/predict",
        files={"file": ("a.png", _png_bytes(), "image/png")},
        headers={"X-API-Key": "s3cret"},
    )
    assert response.status_code == 200


def test_health_does_not_require_key() -> None:
    client = _auth_client("s3cret")
    response = client.get("/health")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_cors_preflight_allows_get() -> None:
    settings = Settings(cors_allow_origins=["https://example.com"])
    app = create_app(settings=settings, engine=StubEngine())  # type: ignore[arg-type]
    client = TestClient(app)
    response = client.options(
        "/health",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://example.com"
