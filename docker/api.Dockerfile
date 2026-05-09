# syntax=docker/dockerfile:1.7
# =============================================================================
# docker/api.Dockerfile - FastAPI inference service (Phase 7).
#
# Multi-stage build:
#   1. ``builder``  - installs runtime Python deps into an isolated venv so
#                     the final image carries no compiler toolchain or pip
#                     cache.
#   2. ``runtime``  - copies the venv + the application source, runs as a
#                     non-root user, exposes :8000 and a HEALTHCHECK that
#                     hits ``/health``.
#
# Build:
#   docker build -f docker/api.Dockerfile -t deepvision-api:dev .
#
# Run (smoke test, random weights):
#   docker run --rm -p 8000:8000 deepvision-api:dev
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 - builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build tools needed by some wheels (numpy/scipy/pillow have manylinux
# wheels so we mostly need them only for transitive C extensions).
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated venv so we can copy *just* it into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Install deps first - layer is cached as long as requirements.txt is unchanged.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2 - runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    DEEPVISION_API_HOST=0.0.0.0 \
    DEEPVISION_API_PORT=8000 \
    DEEPVISION_LOG_LEVEL=info

# Non-root user. Use a fixed UID/GID so the bind-mounted ./models keeps
# predictable ownership across hosts.
RUN groupadd --system --gid 10001 deepvision \
    && useradd --system --uid 10001 --gid deepvision --shell /usr/sbin/nologin deepvision

WORKDIR /app

# Copy the populated virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy only the package source + the project metadata. ``.dockerignore``
# already prunes tests/, docs/, notebooks/, mlruns/, models/.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install the project itself (editable not needed - we want a frozen layer).
RUN pip install --no-deps --no-cache-dir .

# Drop privileges before any user-facing code runs.
RUN chown -R deepvision:deepvision /app
USER deepvision

EXPOSE 8000

# Liveness probe - uses stdlib so no extra package is pulled into the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else sys.exit(1)"

ENTRYPOINT ["python", "-m", "deepvision", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
