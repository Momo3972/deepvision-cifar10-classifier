"""Structural tests for the Phase 7 containerization artefacts.

These tests do **not** invoke Docker. They parse the artefacts as text/YAML
and assert on the contract prescribed by the audit (section 7.5):

* three multi-stage Dockerfiles with a non-root ``USER`` and an explicit
  ``ENTRYPOINT``;
* the ``api`` and ``streamlit`` images expose a HEALTHCHECK;
* ``docker-compose.yml`` declares the six prescribed services with health
  probes, named volumes and an explicit network;
* ``.dockerignore`` excludes the local artefacts that must never enter an
  image (``.venv``, ``mlruns``, ``models``, ``data``, ``tests`` ...);
* ``.env.example`` documents the variables read by ``docker-compose.yml``.

The point is to catch regressions on the contract without depending on a
running Docker daemon -- so the tests stay fast and run in CI even where
Docker is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Fixtures -- read each artefact once per test module.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compose_data() -> dict:
    path = REPO_ROOT / "docker-compose.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def api_dockerfile() -> str:
    return (REPO_ROOT / "docker" / "api.Dockerfile").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def streamlit_dockerfile() -> str:
    return (REPO_ROOT / "docker" / "streamlit.Dockerfile").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def training_dockerfile() -> str:
    return (REPO_ROOT / "docker" / "training.Dockerfile").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dockerignore() -> str:
    return (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def env_example() -> str:
    return (REPO_ROOT / ".env.example").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def prometheus_config() -> dict:
    path = REPO_ROOT / "monitoring" / "prometheus.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# docker-compose.yml -- the 6-service contract.
# ---------------------------------------------------------------------------


REQUIRED_COMPOSE_SERVICES: Final[frozenset[str]] = frozenset(
    {
        "api",
        "streamlit",
        "mlflow",
        "prometheus",
        "grafana",
        "drift-monitor",
    }
)


def test_compose_declares_the_six_required_services(compose_data: dict) -> None:
    """Audit section 7.5 prescribes exactly these six services."""
    services = set(compose_data["services"].keys())
    assert REQUIRED_COMPOSE_SERVICES.issubset(services), (
        f"Missing services: {REQUIRED_COMPOSE_SERVICES - services}"
    )


def test_compose_api_exposes_8000(compose_data: dict) -> None:
    api = compose_data["services"]["api"]
    assert "8000:8000" in api["ports"]


def test_compose_streamlit_exposes_8501(compose_data: dict) -> None:
    streamlit = compose_data["services"]["streamlit"]
    assert "8501:8501" in streamlit["ports"]


def test_compose_mlflow_exposes_5000(compose_data: dict) -> None:
    mlflow = compose_data["services"]["mlflow"]
    assert "5000:5000" in mlflow["ports"]


def test_compose_prometheus_exposes_9090(compose_data: dict) -> None:
    prom = compose_data["services"]["prometheus"]
    assert "9090:9090" in prom["ports"]


def test_compose_grafana_exposes_3000(compose_data: dict) -> None:
    grafana = compose_data["services"]["grafana"]
    assert "3000:3000" in grafana["ports"]


@pytest.mark.parametrize("service", sorted(REQUIRED_COMPOSE_SERVICES))
def test_compose_long_running_services_have_a_healthcheck(compose_data: dict, service: str) -> None:
    """Every long-running service must declare a healthcheck.

    Since Phase 8, the ``drift-monitor`` placeholder is replaced by the real
    Prometheus exporter that exposes ``/metrics`` on :9091 and probes itself
    via stdlib ``urllib`` -- no service is exempt anymore.
    """
    spec = compose_data["services"][service]
    assert "healthcheck" in spec, f"{service} is missing a healthcheck"
    assert "test" in spec["healthcheck"], f"{service} healthcheck has no test"


def test_compose_has_named_volumes(compose_data: dict) -> None:
    """Stateful services must persist data across `compose down`."""
    expected = {"mlflow_data", "prometheus_data", "grafana_data"}
    assert expected.issubset(set(compose_data.get("volumes", {}).keys()))


def test_compose_has_dedicated_network(compose_data: dict) -> None:
    networks = compose_data.get("networks", {})
    assert "deepvision" in networks, "Stack should run on its own bridge network"


def test_compose_streamlit_depends_on_api(compose_data: dict) -> None:
    streamlit = compose_data["services"]["streamlit"]
    deps = streamlit.get("depends_on", {})
    # ``depends_on`` may be a list (basic) or a mapping (with conditions).
    if isinstance(deps, dict):
        assert "api" in deps
        assert deps["api"].get("condition") == "service_healthy"
    else:
        assert "api" in deps


def test_compose_api_depends_on_mlflow(compose_data: dict) -> None:
    api = compose_data["services"]["api"]
    deps = api.get("depends_on", {})
    if isinstance(deps, dict):
        assert "mlflow" in deps
    else:
        assert "mlflow" in deps


# ---------------------------------------------------------------------------
# Dockerfiles -- multi-stage + non-root + entrypoint contract.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dockerfile_fixture", "expected_entry", "expects_healthcheck"),
    [
        ("api_dockerfile", "deepvision", True),
        ("streamlit_dockerfile", "streamlit", True),
        # Training jobs are short-lived batch workloads; no healthcheck.
        ("training_dockerfile", "train", False),
    ],
)
def test_dockerfile_contract(
    request: pytest.FixtureRequest,
    dockerfile_fixture: str,
    expected_entry: str,
    expects_healthcheck: bool,
) -> None:
    src: str = request.getfixturevalue(dockerfile_fixture)

    # Multi-stage build: at least one ``FROM ... AS builder`` and a runtime stage.
    from_lines = [line for line in src.splitlines() if line.startswith("FROM ")]
    assert len(from_lines) >= 2, "Dockerfile must be multi-stage (>= 2 FROM)"
    assert any("AS builder" in line for line in from_lines)
    assert any("AS runtime" in line for line in from_lines)

    # Non-root user.
    assert "USER deepvision" in src, "Dockerfile must drop privileges"
    assert "useradd" in src or "adduser" in src

    # Entrypoint references the deepvision CLI.
    assert "ENTRYPOINT" in src
    assert "deepvision" in src
    assert expected_entry in src

    # Healthcheck only where applicable. We look for the *directive* (a line
    # starting with ``HEALTHCHECK``) to avoid false positives on the word
    # appearing inside a comment such as ``# No HEALTHCHECK - short-lived``.
    has_hc = any(line.strip().startswith("HEALTHCHECK") for line in src.splitlines())
    if expects_healthcheck:
        assert has_hc, "Long-running service must declare a HEALTHCHECK"
    else:
        assert not has_hc, "Short-lived training image must not declare a HEALTHCHECK"


def test_api_dockerfile_exposes_8000(api_dockerfile: str) -> None:
    assert "EXPOSE 8000" in api_dockerfile


def test_streamlit_dockerfile_exposes_8501(streamlit_dockerfile: str) -> None:
    assert "EXPOSE 8501" in streamlit_dockerfile


# ---------------------------------------------------------------------------
# .dockerignore -- the audit's "must not enter the image" list.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern",
    [
        ".venv",
        "mlruns",
        "models",
        "data",
        "tests",
        "docs",
        "Notebooks",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".env",
    ],
)
def test_dockerignore_excludes(dockerignore: str, pattern: str) -> None:
    """Local artefacts that must never enter a Docker image."""
    assert pattern in dockerignore, f"{pattern!r} should be excluded by .dockerignore"


def test_dockerignore_keeps_env_example(dockerignore: str) -> None:
    """``.env`` is excluded but the template ``.env.example`` is whitelisted."""
    assert "!.env.example" in dockerignore


# ---------------------------------------------------------------------------
# .env.example -- documents the compose-time variables.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "var",
    [
        "DEEPVISION_API_KEY",
        "DEEPVISION_MODEL_PATH",
        "DEEPVISION_LOG_LEVEL",
        "MLFLOW_TRACKING_URI",
        "GF_SECURITY_ADMIN_PASSWORD",
    ],
)
def test_env_example_documents(env_example: str, var: str) -> None:
    assert f"{var}=" in env_example, f"{var} missing from .env.example"


# ---------------------------------------------------------------------------
# monitoring/prometheus.yml -- scrape contract.
# ---------------------------------------------------------------------------


def test_prometheus_scrapes_the_api(prometheus_config: dict) -> None:
    jobs = {job["job_name"] for job in prometheus_config.get("scrape_configs", [])}
    assert "deepvision-api" in jobs


def test_prometheus_api_target_uses_compose_hostname(prometheus_config: dict) -> None:
    """Inside the compose network, the API is reachable at ``api:8000``.

    Targeting ``localhost`` would silently break scraping.
    """
    api_job = next(
        job for job in prometheus_config["scrape_configs"] if job["job_name"] == "deepvision-api"
    )
    targets = [t for sc in api_job["static_configs"] for t in sc["targets"]]
    assert "api:8000" in targets


def test_prometheus_metrics_path_is_metrics(prometheus_config: dict) -> None:
    api_job = next(
        job for job in prometheus_config["scrape_configs"] if job["job_name"] == "deepvision-api"
    )
    assert api_job["metrics_path"] == "/metrics"
