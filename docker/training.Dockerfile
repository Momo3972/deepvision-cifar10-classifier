# syntax=docker/dockerfile:1.7
# =============================================================================
# docker/training.Dockerfile - CPU-only training image (Phase 7).
#
# Differences with ``api.Dockerfile``:
#   * Embeds ``requirements-dev.txt`` so MLflow autologging dependencies and
#     test scaffolding are available -- training runs typically need them.
#   * Entrypoint is ``python -m deepvision train`` (the audit prescribes a
#     CUDA variant; we ship the CPU variant first since Colab handles GPU
#     training in this project's actual workflow).
#   * No HEALTHCHECK -- training jobs are short-lived batch workloads.
#
# Build:
#   docker build -f docker/training.Dockerfile -t deepvision-training:dev .
#
# Run a quick smoke training:
#   docker run --rm -v %cd%/mlruns:/app/mlruns deepvision-training:dev \
#       --model efficientnet --quick
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 - builder
# -----------------------------------------------------------------------------
FROM python:3.14-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Order matters for layer caching: install runtime first, dev next.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt -r requirements-dev.txt

# -----------------------------------------------------------------------------
# Stage 2 - runtime
# -----------------------------------------------------------------------------
FROM python:3.14-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    DEEPVISION_LOG_LEVEL=info \
    MLFLOW_TRACKING_URI=file:/app/mlruns \
    MPLCONFIGDIR=/tmp/matplotlib

RUN groupadd --system --gid 10001 deepvision \
    && useradd --system --uid 10001 --gid deepvision --shell /usr/sbin/nologin deepvision

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-deps --no-cache-dir .

# Pre-create mount points so the bind volumes inherit the correct ownership.
RUN mkdir -p /app/mlruns /app/models /app/data \
    && chown -R deepvision:deepvision /app

USER deepvision

# No EXPOSE - training jobs do not listen.
# No HEALTHCHECK - they are short-lived.

ENTRYPOINT ["python", "-m", "deepvision", "train"]
CMD ["--model", "efficientnet", "--quick"]
