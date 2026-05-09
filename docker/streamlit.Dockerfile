# syntax=docker/dockerfile:1.7
# =============================================================================
# docker/streamlit.Dockerfile - Streamlit demo UI (Phase 7).
#
# Same multi-stage pattern as ``api.Dockerfile``: a builder stage installs
# runtime Python dependencies into ``/opt/venv``; the runtime stage copies
# the venv + the package source and runs a non-root user.
#
# The CLI entrypoint (``python -m deepvision streamlit``) is responsible for
# spawning ``streamlit run`` against ``deepvision.streamlit_app`` -- so the
# image stays consistent with the CLI surface and the unit tests.
#
# Build:
#   docker build -f docker/streamlit.Dockerfile -t deepvision-streamlit:dev .
#
# Run:
#   docker run --rm -p 8501:8501 deepvision-streamlit:dev
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 - builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

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
    DEEPVISION_STREAMLIT_HOST=0.0.0.0 \
    DEEPVISION_STREAMLIT_PORT=8501 \
    DEEPVISION_LOG_LEVEL=info \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN groupadd --system --gid 10001 deepvision \
    && useradd --system --uid 10001 --gid deepvision --shell /usr/sbin/nologin deepvision

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-deps --no-cache-dir .

RUN chown -R deepvision:deepvision /app
USER deepvision

EXPOSE 8501

# Streamlit ships a built-in liveness endpoint at ``/_stcore/health``.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0) if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).status == 200 else sys.exit(1)"

ENTRYPOINT ["python", "-m", "deepvision", "streamlit"]
CMD ["--host", "0.0.0.0", "--port", "8501"]
