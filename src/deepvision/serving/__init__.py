"""
Serving layer: FastAPI REST API and Streamlit demo.

Will be populated in Phase 5 with:

- :mod:`deepvision.serving.api`:        FastAPI app exposing /predict, /predict_batch, /explain, /metrics.
- :mod:`deepvision.serving.schemas`:    Pydantic request/response schemas.
- :mod:`deepvision.serving.preprocess`: image validation + RGB conversion + resize.
- :mod:`deepvision.serving.inference`:  thin wrapper around the loaded model.
- :mod:`deepvision.serving.prometheus`: Prometheus instrumentation (latency histogram, error counter).
"""
