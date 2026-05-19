# Getting started

This page walks you through cloning the repo, installing the project,
and running each top-level command at least once. It targets a fresh
machine with **Python 3.11 or 3.12** and either Linux or Windows
PowerShell.

If you only want to *try* the model without installing anything, the
Docker stack (see [the Docker section below](#with-docker)) launches
the API, the Streamlit demo, MLflow, Prometheus and Grafana in one
command.

## Prerequisites

| Tool | Version | Why |
|---|---|---|
| Python | 3.11 or 3.12 | CI matrix; avoid 3.13 -- TensorFlow wheels are not yet shipped |
| `pip` | >= 24.0 | PEP 668 + editable install support |
| `git` | any recent | `git rev-parse` is logged in every MLflow run |
| Docker (optional) | >= 24 | only needed for the multi-service compose stack |

GPU is **not required**. The whole pipeline runs on CPU; training one
EfficientNet epoch on a modern laptop CPU takes ~3 minutes for the
`--quick` smoke profile.

## Install from source

```bash
# 1. Clone
git clone https://github.com/Momo3972/deepvision-cifar10-classifier.git
cd deepvision-cifar10-classifier

# 2. Create + activate a venv
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# 3. Install runtime + dev dependencies
pip install -r requirements.txt -r requirements-dev.txt
pip install -e . --no-deps

# 4. Sanity check
python -m deepvision --help
deepvision info
```

The `deepvision info` command prints the package version, resolved
config paths, and the active Python interpreter. If something is off,
this is the first place to look.

## Configure (optional)

All knobs are exposed via the `Settings` class in
[`deepvision.config`][deepvision.config]. Override any field with an
environment variable prefixed by `DEEPVISION_`:

```bash
export DEEPVISION_SEED=1234
export DEEPVISION_BATCH_SIZE=128
export DEEPVISION_MODEL_PATH=models/efficientnet_best.keras
```

A `.env` file at the repo root works too -- see `.env.example` for the
canonical template.

## Train your first model

A tiny smoke run on 1 000 images and one epoch (~30 s on CPU):

```bash
python -m deepvision train --model efficientnet --quick
```

This will:

1. Download CIFAR-10 on first use (~170 MB to `~/.keras/datasets/`).
2. Build an EfficientNetB0 with random weights.
3. Train one epoch on a stratified 1 000-image subset.
4. Log the run, metrics and artefacts to a local MLflow tracking store at
   `./mlruns/`.

Open the MLflow UI to inspect the run:

```bash
mlflow ui --backend-store-uri ./mlruns
# -> http://localhost:5000
```

For a full training (10 epochs + 5 fine-tuning epochs):

```bash
python -m deepvision train --model efficientnet --epochs 10 --fine-tune-epochs 5
```

See the [training tutorial](tutorials/training.md) for hyperparameter
recommendations and tips for running on Colab.

## Serve the trained model

```bash
python -m deepvision serve --port 8000
```

The FastAPI server boots in ~2 s (heavy imports are deferred) and
exposes:

- `GET /health` -- liveness probe
- `GET /meta` -- model name and version
- `POST /predict` -- single image, returns top-K classes
- `POST /predict_batch` -- up to 16 images
- `GET /metrics` -- Prometheus exposition format
- `GET /docs` -- interactive Swagger UI

Without a trained model the API serves an EfficientNetB0 with **random
weights** -- predictions will be meaningless but the request path is
correct. Point at a real artefact with:

```bash
export DEEPVISION_MODEL_PATH=models/efficientnet_best.keras
python -m deepvision serve
```

## Launch the Streamlit demo

```bash
python -m deepvision streamlit --port 8501
# -> http://localhost:8501
```

The demo lets you upload one or several CIFAR-10-style images, see the
top-K classes, and overlay a Grad-CAM heatmap on the input.

## With Docker

The full six-service stack is in `docker-compose.yml`. Bring it up with:

```bash
docker compose up -d
```

Then visit:

| Service | URL | Note |
|---|---|---|
| FastAPI | <http://localhost:8000/docs> | inference API |
| Streamlit | <http://localhost:8501> | demo UI |
| MLflow | <http://localhost:5000> | training runs |
| Prometheus | <http://localhost:9090> | metrics + alert rules |
| Grafana | <http://localhost:3000> | dashboards (login `admin` / `admin`) |
| Drift exporter | <http://localhost:9091/metrics> | Wasserstein + OOD scores |

Bring it down with `docker compose down -v`.

## Run the test suite

```bash
pytest -n auto
```

You should see **356+ tests passed** in ~40 s on a recent CPU. The
suite includes structural checks for the GitHub workflows, the Docker
artefacts, the monitoring stack, the export pipeline and every
internal module.

## Next steps

- Architecture overview: see [Architecture](architecture.md).
- Specific tasks: pick a [tutorial](tutorials/index.md).
- Model details: read the [Model card](model-card.md).
- API reference: see [API reference](reference/index.md).
- Contribute: read [Contributing](contributing.md).
