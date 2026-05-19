# Serving

This tutorial covers the FastAPI inference service: how to boot it,
which endpoints it exposes, how to enable optional API-key auth, and
how to scrape its Prometheus metrics.

## Boot the server

```bash
python -m deepvision serve --host 0.0.0.0 --port 8000 --workers 2
```

The server uses [uvicorn](https://www.uvicorn.org/) under the hood.
With `--workers 2` you get two independent processes sharing the port.
Auto-reload is available with `--reload` for development:

```bash
python -m deepvision serve --reload
```

!!! warning "`--reload` forces a single worker"
    The CLI silently caps `--workers` to 1 when `--reload` is set --
    uvicorn cannot watch files across multiple processes. Use
    `--reload` for development only.

By default the API serves an **EfficientNetB0 with random weights**
(predictions are statistically meaningless but shape-correct), which
keeps CI smoke tests cheap. Point at a real artefact with:

```bash
export DEEPVISION_MODEL_PATH=models/efficientnet_best.keras
python -m deepvision serve
```

## Endpoints

### `GET /health`

Liveness probe. Returns `{"status": "ok"}` as soon as the FastAPI
process is up -- the model is **not** loaded yet (lazy initialisation).

```bash
curl http://localhost:8000/health
```

### `GET /meta`

Model identity. Returns the configured name and version:

```bash
$ curl http://localhost:8000/meta
{"name": "efficientnet_b0_transfer", "version": "0.11.0"}
```

The `DEEPVISION_SERVING_MODEL_NAME` and `DEEPVISION_SERVING_MODEL_VERSION`
env vars let you override these without rebuilding.

### `POST /predict`

Single-image inference. Send a multipart upload:

```bash
curl -X POST http://localhost:8000/predict \
    -F "file=@photo.jpg" \
    -F "top_k=3"
```

Response:

```json
{
  "predictions": [
    {"class_index": 5, "class_name": "dog", "probability": 0.92},
    {"class_index": 3, "class_name": "cat", "probability": 0.05},
    {"class_index": 6, "class_name": "frog", "probability": 0.01}
  ],
  "inference_time_ms": 18.4,
  "model_name": "efficientnet_b0_transfer",
  "model_version": "0.11.0"
}
```

The upload is validated by **magic bytes** (JPEG/PNG/WEBP/GIF/BMP/TIFF)
in [`deepvision.serving.preprocess.validate_payload_size`][deepvision.serving.preprocess.validate_payload_size]
-- the file extension is ignored. Size cap is
`DEEPVISION_MAX_IMAGE_BYTES` (10 MB by default); files larger than
that yield a `413 Request Entity Too Large`.

### `POST /predict_batch`

Same payload format, accepts up to `DEEPVISION_MAX_BATCH_SIZE` files
in one call (default 16). The response is a list of per-image
prediction objects in the same order as the uploaded files.

### `GET /metrics`

Prometheus exposition format. Sample metrics:

```
# HELP deepvision_predictions_total Total number of /predict[_batch] calls
# TYPE deepvision_predictions_total counter
deepvision_predictions_total{class_name="dog",model_version="0.11.0"} 42

# HELP deepvision_inference_latency_seconds Inference wall-clock per call
# TYPE deepvision_inference_latency_seconds histogram
deepvision_inference_latency_seconds_bucket{le="0.05"} 35
deepvision_inference_latency_seconds_bucket{le="0.1"} 41
...
```

Add `localhost:8000` to your Prometheus scrape config (already done
in the bundled `monitoring/prometheus.yml`) and you immediately get
request rate, error rate and latency percentiles for free.

### `GET /docs` and `GET /redoc`

Two interactive API browsers shipped by FastAPI for free. `/docs`
uses Swagger UI; `/redoc` uses ReDoc.

## Optional API-key auth

Set `DEEPVISION_API_KEY` to any non-empty string and every
`/predict*` request must carry an `X-API-Key` header matching it:

```bash
export DEEPVISION_API_KEY="<YOUR_API_KEY_HERE>"
python -m deepvision serve

# Now this request returns 401:
curl -X POST http://localhost:8000/predict -F "file=@photo.jpg"

# And this one returns 200:
curl -X POST http://localhost:8000/predict \
    -H "X-API-Key: <YOUR_API_KEY_HERE>" \
    -F "file=@photo.jpg"
```

The middleware in [`deepvision.serving.api`][deepvision.serving.api]
returns a clean `{"detail": "Invalid or missing API key"}` payload
rather than the FastAPI default. Health and meta endpoints stay open
so probes still work.

!!! tip "Auth is off by default"
    The empty string is normalised to `None` by the
    `_empty_string_is_none` validator in `Settings`, so leaving
    `DEEPVISION_API_KEY` blank or unset disables auth completely.
    This is the right behaviour inside a private Docker network; in
    production you should always set it.

## Common pitfalls

!!! warning "Don't expose the API to the public internet without auth"
    The model returns predictions regardless of the input image
    content. Without auth, anyone can flood `/predict` with requests
    and rack up your compute bill. Set `DEEPVISION_API_KEY` and put
    a TLS-terminating proxy in front of the service.

!!! tip "Use `--workers 2` (not more) on a 2-vCPU instance"
    TensorFlow grabs every available core for matrix multiplications
    inside one prediction. Two worker processes saturate a 2-vCPU
    instance without context-switching overhead; four workers would
    actually be *slower*.

## Next steps

- Live drift detection: [Monitoring tutorial](monitoring.md).
- Faster inference: [Export & benchmark tutorial](export.md).
