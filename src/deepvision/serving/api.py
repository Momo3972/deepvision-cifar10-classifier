"""
FastAPI inference service for deepvision.

Endpoints
---------
- ``GET  /``               — minimal HTML index pointing to ``/docs``.
- ``GET  /health``         — liveness probe (responds 200 as long as the
  process is alive). No model touch.
- ``GET  /ready``          — readiness probe; reports whether the model is
  loaded and able to serve traffic.
- ``GET  /meta``           — static metadata: package version, class names,
  accepted MIME types, payload limits.
- ``POST /predict``        — single-image prediction (multipart/form-data).
- ``POST /predict_batch``  — multi-image prediction (multipart/form-data).
- ``GET  /metrics``        — Prometheus exposition (text/plain v0.0.4).

Design notes
------------
* The Keras model is loaded **lazily** through :class:`InferenceEngine` so
  the process starts in milliseconds. ``/health`` and ``/meta`` therefore
  do not pay the TensorFlow import cost — important for Kubernetes liveness
  probes and CI smoke tests.
* Authentication is **opt-in**: setting the ``DEEPVISION_API_KEY`` environment
  variable enables a header-based check on the prediction endpoints. When
  unset, the API is open (matches the demo deployment story).
* Errors return a uniform :class:`ErrorResponse` envelope; their counts are
  tracked via :data:`HTTP_ERRORS` for Grafana alerting.
* Logs are structured key=value pairs through :func:`utils.logging.get_logger`.

The ``app`` ASGI object is what ``uvicorn`` boots; the ``create_app`` factory
exists so tests can spin up an isolated app with a stub :class:`InferenceEngine`
or a custom :class:`Settings` instance.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from deepvision import __version__
from deepvision.config import Settings, get_settings
from deepvision.constants import CLASS_NAMES_EN
from deepvision.serving.inference import InferenceEngine
from deepvision.serving.preprocess import (
    ACCEPTED_MIME_TYPES,
    ImageValidationError,
    load_image_for_inference,
)
from deepvision.serving.prometheus import (
    CONTENT_TYPE_LATEST,
    HTTP_ERRORS,
    HTTP_REQUESTS,
    INFERENCE_LATENCY,
    MODEL_INFO,
    MODEL_LOADED,
    REGISTRY,
    render_latest,
    status_class,
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
from deepvision.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _err_payload(code: str, detail: str) -> dict[str, str]:
    """Build the JSON body of an :class:`ErrorResponse`."""
    return ErrorResponse(error=code, detail=detail).model_dump()


def _record_metrics(
    *,
    method: str,
    endpoint: str,
    status_code: int,
    error_code: str | None = None,
) -> None:
    """Increment the request and (optionally) error Prometheus counters."""
    sc = status_class(status_code)
    HTTP_REQUESTS.labels(method=method, endpoint=endpoint, status_class=sc).inc()
    if status_code >= 400:
        HTTP_ERRORS.labels(
            endpoint=endpoint,
            status_class=sc,
            error_code=error_code or f"http_{status_code}",
        ).inc()


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def make_api_key_dependency(settings: Settings) -> Callable[..., None]:
    """Return a FastAPI dependency that validates ``X-API-Key`` if configured.

    When ``settings.api_key`` is ``None`` the dependency is a no-op so the
    API remains usable on a public demo. Tests that want to exercise the
    enforcement branch should build a :class:`Settings` with ``api_key`` set.
    """

    expected = settings.api_key

    def _check(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        if expected is None:
            return
        if x_api_key != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid X-API-Key header.",
            )

    return _check


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    settings: Settings | None = None,
    engine: InferenceEngine | None = None,
) -> FastAPI:
    """Build a FastAPI application configured for the deepvision API.

    Parameters
    ----------
    settings
        Optional :class:`Settings` override. Defaults to :func:`get_settings`.
    engine
        Optional pre-built :class:`InferenceEngine`. Defaults to one
        constructed from ``settings``. Tests can pass a stub engine to skip
        the TensorFlow dependency entirely.
    """

    if settings is None:
        settings = get_settings()
    if engine is None:
        engine = InferenceEngine(
            model_path=settings.model_path,
            model_name=settings.serving_model_name,
            model_version=settings.serving_model_version,
        )

    app = FastAPI(
        title="DeepVision CIFAR-10 Inference API",
        version=__version__,
        description=(
            "REST API serving an EfficientNetB0 fine-tuned on CIFAR-10. "
            "Built as Phase 5 of the industrial refactor described in "
            "`Audit_DeepVision_CIFAR10.docx`."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Stash on app.state so dependencies and tests can introspect.
    app.state.settings = settings
    app.state.engine = engine

    # Initialize the static "model_info" gauge to 1 with the configured labels
    # — Grafana queries can then group by model_version without ever seeing
    # an empty series.
    MODEL_INFO.labels(
        model_name=settings.serving_model_name,
        model_version=settings.serving_model_version,
    ).set(1)
    MODEL_LOADED.set(1 if engine.is_loaded else 0)

    api_key_required = make_api_key_dependency(settings)

    # ---------------- Middleware: Prometheus + access logs --------------
    @app.middleware("http")
    async def _observe(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        response: Response
        try:
            response = await call_next(request)
        except Exception as exc:  # pragma: no cover — fallback path
            log.exception("Unhandled exception in request pipeline: %s", exc)
            response = JSONResponse(
                status_code=500,
                content=_err_payload("internal_error", "Unexpected server error."),
            )
        elapsed = time.perf_counter() - start
        endpoint = request.url.path
        _record_metrics(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        )
        log.info(
            "method=%s path=%s status=%d duration_ms=%.2f",
            request.method,
            endpoint,
            response.status_code,
            elapsed * 1000.0,
        )
        return response

    # ------------------------ Routes ------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> HTMLResponse:
        return HTMLResponse(
            "<html><body style='font-family:sans-serif'>"
            f"<h1>DeepVision CIFAR-10 API v{__version__}</h1>"
            "<p>Try <a href='/docs'>/docs</a> for interactive Swagger UI, "
            "or <a href='/redoc'>/redoc</a> for ReDoc.</p>"
            "</body></html>"
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["operational"],
        summary="Liveness probe.",
    )
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get(
        "/ready",
        response_model=ReadyResponse,
        tags=["operational"],
        summary="Readiness probe.",
    )
    async def ready() -> ReadyResponse:
        return ReadyResponse(
            ready=engine.is_loaded,
            model_loaded=engine.is_loaded,
            model_name=engine.model_name,
            model_version=engine.model_version,
        )

    @app.get(
        "/meta",
        response_model=MetaResponse,
        tags=["operational"],
        summary="Static metadata.",
    )
    async def meta() -> MetaResponse:
        return MetaResponse(
            max_image_bytes=settings.max_image_bytes,
        )

    @app.get(
        "/metrics",
        response_class=PlainTextResponse,
        tags=["operational"],
        summary="Prometheus metrics endpoint.",
        responses={200: {"content": {CONTENT_TYPE_LATEST: {}}}},
    )
    async def metrics() -> Response:
        body = render_latest(REGISTRY)
        return Response(content=body, media_type=CONTENT_TYPE_LATEST)

    # ---------------- Predict (single) ----------------------------------

    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["inference"],
        summary="Run inference on a single image.",
        responses={
            400: {"model": ErrorResponse, "description": "Invalid image upload."},
            413: {"model": ErrorResponse, "description": "Image too large."},
            415: {"model": ErrorResponse, "description": "Unsupported MIME type."},
        },
    )
    async def predict(
        file: UploadFile = File(..., description="Image file (jpg, png, webp)."),
        top_k: int = 3,
        _: None = Depends(api_key_required),
    ) -> PredictResponse | JSONResponse:
        return await _run_single_prediction(file=file, top_k=top_k, engine=engine)

    # ---------------- Predict (batch) ----------------------------------

    @app.post(
        "/predict_batch",
        response_model=BatchPredictResponse,
        tags=["inference"],
        summary="Run inference on a batch of images.",
        responses={
            400: {"model": ErrorResponse, "description": "Invalid batch."},
            413: {"model": ErrorResponse, "description": "Image too large."},
            415: {"model": ErrorResponse, "description": "Unsupported MIME type."},
        },
    )
    async def predict_batch(
        files: list[UploadFile] = File(..., description="Up to ``max_batch_size`` images."),
        top_k: int = 3,
        _: None = Depends(api_key_required),
    ) -> BatchPredictResponse | JSONResponse:
        return await _run_batch_prediction(
            files=files,
            top_k=top_k,
            engine=engine,
            max_batch_size=settings.max_batch_size,
        )

    return app


# ---------------------------------------------------------------------------
# Per-endpoint logic (extracted for clarity and unit testing)
# ---------------------------------------------------------------------------


async def _run_single_prediction(
    *,
    file: UploadFile,
    top_k: int,
    engine: InferenceEngine,
) -> PredictResponse | JSONResponse:
    payload, mime_error = await _read_and_validate_upload(file)
    if mime_error is not None:
        return mime_error

    try:
        image_batch = load_image_for_inference(payload)
    except ImageValidationError as exc:
        return _validation_failed_response(str(exc))
    except Exception as exc:  # pragma: no cover — defensive
        log.exception("Unexpected error preprocessing /predict upload")
        return JSONResponse(
            status_code=400,
            content=_err_payload("invalid_image", f"Could not process image: {exc}"),
        )

    if top_k < 1:
        return JSONResponse(
            status_code=400,
            content=_err_payload("invalid_top_k", "top_k must be >= 1."),
        )
    top_k = min(top_k, len(CLASS_NAMES_EN))

    start = time.perf_counter()
    items, _model_ms = engine.predict(image_batch, top_k=top_k)
    elapsed_s = time.perf_counter() - start
    INFERENCE_LATENCY.labels(
        endpoint="/predict",
        model_name=engine.model_name,
    ).observe(elapsed_s)

    return PredictResponse(
        model_name=engine.model_name,
        model_version=engine.model_version,
        top_k=top_k,
        predictions=[
            PredictionItem(
                rank=i + 1,
                class_index=idx,
                class_name=name,
                probability=prob,
            )
            for i, (idx, name, prob) in enumerate(items)
        ],
        inference_time_ms=elapsed_s * 1000.0,
    )


async def _run_batch_prediction(
    *,
    files: list[UploadFile],
    top_k: int,
    engine: InferenceEngine,
    max_batch_size: int,
) -> BatchPredictResponse | JSONResponse:
    if len(files) == 0:
        return JSONResponse(
            status_code=400,
            content=_err_payload("empty_batch", "At least one image required."),
        )
    if len(files) > max_batch_size:
        return JSONResponse(
            status_code=413,
            content=_err_payload(
                "batch_too_large",
                f"Got {len(files)} files; max allowed is {max_batch_size}.",
            ),
        )

    if top_k < 1:
        return JSONResponse(
            status_code=400,
            content=_err_payload("invalid_top_k", "top_k must be >= 1."),
        )
    top_k = min(top_k, len(CLASS_NAMES_EN))

    items_per_image: list[list[PredictionItem]] = []
    filenames: list[str] = []

    start = time.perf_counter()
    for upload in files:
        payload, mime_error = await _read_and_validate_upload(upload)
        if mime_error is not None:
            return mime_error
        try:
            image_batch = load_image_for_inference(payload)
        except ImageValidationError as exc:
            return _validation_failed_response(str(exc))
        ranked, _ = engine.predict(image_batch, top_k=top_k)
        items_per_image.append(
            [
                PredictionItem(
                    rank=i + 1,
                    class_index=idx,
                    class_name=name,
                    probability=prob,
                )
                for i, (idx, name, prob) in enumerate(ranked)
            ]
        )
        filenames.append(upload.filename or "")
    elapsed_s = time.perf_counter() - start
    INFERENCE_LATENCY.labels(
        endpoint="/predict_batch",
        model_name=engine.model_name,
    ).observe(elapsed_s)

    return BatchPredictResponse(
        model_name=engine.model_name,
        model_version=engine.model_version,
        top_k=top_k,
        items=[
            BatchPredictionItem(
                index=i,
                filename=filename,
                predictions=preds,
            )
            for i, (filename, preds) in enumerate(zip(filenames, items_per_image, strict=True))
        ],
        inference_time_ms=elapsed_s * 1000.0,
    )


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


async def _read_and_validate_upload(
    upload: UploadFile,
) -> tuple[bytes, JSONResponse | None]:
    """Read an :class:`UploadFile`, returning either bytes or a 415/400 response."""
    if upload.content_type not in ACCEPTED_MIME_TYPES:
        return b"", JSONResponse(
            status_code=415,
            content=_err_payload(
                "unsupported_media_type",
                f"Unsupported MIME type: {upload.content_type!r}. "
                f"Accepted: {list(ACCEPTED_MIME_TYPES)}.",
            ),
        )
    payload = await upload.read()
    return payload, None


def _validation_failed_response(detail: str) -> JSONResponse:
    """Return a 400 :class:`ErrorResponse` for image validation failures."""
    return JSONResponse(
        status_code=400,
        content=_err_payload("invalid_image", detail),
    )


# ---------------------------------------------------------------------------
# Re-export a default app instance — uvicorn entrypoint.
# ---------------------------------------------------------------------------

#: Default ASGI app — used by ``uvicorn deepvision.serving.api:app``.
app: FastAPI = create_app()


def _accepted_mimes_in_meta() -> Iterable[str]:  # pragma: no cover — convenience
    """Hook so other modules can import the canonical accepted MIME list."""
    return ACCEPTED_MIME_TYPES


__all__ = [
    "app",
    "create_app",
    "make_api_key_dependency",
]
