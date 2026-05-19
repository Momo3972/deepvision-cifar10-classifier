# DeepVision -- CIFAR-10

**Industrial Computer Vision pipeline for CIFAR-10.** Compares MLP, custom
CNN and EfficientNetB0 (transfer learning + data augmentation) with a
full MLOps lifecycle: MLflow tracking, FastAPI serving, Docker images,
drift monitoring, and CI/CD.

[![CI](https://github.com/Momo3972/deepvision-cifar10-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Momo3972/deepvision-cifar10-classifier/actions/workflows/ci.yml)
[![Security](https://github.com/Momo3972/deepvision-cifar10-classifier/actions/workflows/security.yml/badge.svg)](https://github.com/Momo3972/deepvision-cifar10-classifier/actions/workflows/security.yml)
[![Codecov](https://codecov.io/gh/Momo3972/deepvision-cifar10-classifier/branch/main/graph/badge.svg)](https://codecov.io/gh/Momo3972/deepvision-cifar10-classifier)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why this project

CIFAR-10 is a *teaching* dataset. The interesting question is not
"can we hit 90 % accuracy?" -- the answer has been "yes" since 2015 --
but **"can we wrap the model in everything industry needs to actually
deploy and operate it?"**. This project demonstrates the answer with a
single coherent codebase covering every MLOps surface:

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } **Three model families**

    ---

    MLP baseline, custom CNN with BatchNorm + dropout, and an
    EfficientNetB0 transfer-learning head with optional fine-tuning.
    All trained with the same reproducible CIFAR-10 split.

-   :material-rocket-launch:{ .lg .middle } **FastAPI inference service**

    ---

    Async REST API with Pydantic v2 schemas, magic-bytes file validation,
    optional `X-API-Key` auth, Prometheus instrumentation, and a
    `/predict_batch` endpoint capped at 16 images per call.

-   :material-chart-bell-curve:{ .lg .middle } **Drift & OOD monitoring**

    ---

    Wasserstein-1D drift detection on penultimate-layer embeddings plus
    energy-based out-of-distribution scoring, exported to Prometheus and
    visualised in a pre-provisioned Grafana dashboard.

-   :material-export:{ .lg .middle } **Multi-runtime export**

    ---

    One-command export to ONNX (opset 17, forward-pass validated) and
    TFLite (Full INT8 quantization with CIFAR-10 calibration). Bundled
    benchmark reports p50 / p90 / p95 / p99 latency across runtimes.

-   :material-docker:{ .lg .middle } **Six-service Docker stack**

    ---

    `api`, `streamlit`, `mlflow`, `prometheus`, `grafana`,
    `drift-monitor` -- all wired through a private network with
    healthchecks. Production-style boot via `docker compose up`.

-   :material-cog:{ .lg .middle } **Full CI/CD**

    ---

    Four GitHub Actions workflows enforce `ruff`, `mypy`, `pytest`,
    `bandit`, `pip-audit`, `gitleaks`, `codeql`, plus `trivy` image
    scans on every tag.

</div>

---

## Quickstart

```bash
# 1. Clone + install
git clone https://github.com/Momo3972/deepvision-cifar10-classifier.git
cd deepvision-cifar10-classifier
pip install -e ".[dev]"

# 2. Train a tiny model (CPU-friendly smoke run)
python -m deepvision train --model efficientnet --quick

# 3. Serve it
python -m deepvision serve --port 8000
# -> http://localhost:8000/docs

# 4. Or launch the full stack via Docker
docker compose up -d
```

See the [Getting started](getting-started.md) page for the full walkthrough,
or jump straight to a [tutorial](tutorials/index.md) for a specific task.

---

## Project status

| Phase | Scope | Status |
|---|---|---|
| 1 -- Packaging | `pyproject.toml`, CLI, logging, config | :material-check-circle:{ .green } shipped |
| 2 -- Data | Reproducible split, augmentation pipeline | :material-check-circle:{ .green } shipped |
| 3 -- Training | MLP / CNN / EfficientNet + MLflow tracking | :material-check-circle:{ .green } shipped |
| 4 -- Evaluation | Metrics, calibration, robustness, Grad-CAM | :material-check-circle:{ .green } shipped |
| 5 -- Serving | FastAPI app + Prometheus middleware | :material-check-circle:{ .green } shipped |
| 6 -- Streamlit | Demo UI with Grad-CAM and batch upload | :material-check-circle:{ .green } shipped |
| 7 -- Docker | Multi-stage images + 6-service compose | :material-check-circle:{ .green } shipped |
| 8 -- Monitoring | Drift, OOD, Prometheus, Grafana | :material-check-circle:{ .green } shipped |
| 9 -- CI/CD | 4 GitHub Actions + Dependabot + templates | :material-check-circle:{ .green } shipped |
| 10 -- Export | ONNX / TFLite / latency benchmark | :material-check-circle:{ .green } shipped |
| 11 -- Documentation | This site, model card, bilingual EN/FR | :material-progress-clock:{ .blue } in progress |
| 12 -- Final acceptance | End-to-end recipe + sign-off | :material-clock-outline:{ .grey } planned |

---

## License

Released under the [MIT License](https://github.com/Momo3972/deepvision-cifar10-classifier/blob/main/LICENSE).

## Author

**Mohamed Lamine OULD BOUYA** -- AI/ML engineer  
[GitHub](https://github.com/Momo3972) · [LinkedIn](https://www.linkedin.com/in/mohamed-lamine-ould-bouya)
