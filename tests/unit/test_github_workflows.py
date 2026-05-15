"""Structural tests for the Phase 9 GitHub Actions artefacts.

These tests do **not** run any workflow. They parse the YAML and Markdown
files under ``.github/`` and assert on the contract prescribed by the
audit (section 7.5):

* four workflows exist with the right names and triggers,
* each workflow declares the prescribed jobs,
* ``dependabot.yml`` tracks the three ecosystems (pip, docker,
  github-actions) on a weekly cadence,
* the four issue templates + the PR template are valid YAML / Markdown,
* the README badges point at the GitHub Actions endpoints we just wrote.

Running these in unit tests (rather than relying on the workflows
themselves) keeps the contract visible from a fresh ``pytest`` run and
prevents accidental drift between what the PR template promises and what
the CI actually enforces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
GH_DIR: Final[Path] = REPO_ROOT / ".github"
WF_DIR: Final[Path] = GH_DIR / "workflows"
ISSUE_DIR: Final[Path] = GH_DIR / "ISSUE_TEMPLATE"


# ---------------------------------------------------------------------------
# Fixtures -- read each artefact once per test module.
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ci() -> dict:
    return _load(WF_DIR / "ci.yml")


@pytest.fixture(scope="module")
def security() -> dict:
    return _load(WF_DIR / "security.yml")


@pytest.fixture(scope="module")
def docker() -> dict:
    return _load(WF_DIR / "docker.yml")


@pytest.fixture(scope="module")
def docs() -> dict:
    return _load(WF_DIR / "docs.yml")


@pytest.fixture(scope="module")
def dependabot() -> dict:
    return _load(GH_DIR / "dependabot.yml")


@pytest.fixture(scope="module")
def pr_template() -> str:
    return (GH_DIR / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def readme() -> str:
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helper -- PyYAML loads the bare-word ``on:`` key as Python ``True``, since
# YAML 1.1 treats it as a boolean. The workflow file is valid YAML and
# GitHub Actions reads it correctly; we just need to handle both shapes in
# the test.
# ---------------------------------------------------------------------------


def _triggers(wf: dict) -> dict:
    return wf.get("on") or wf.get(True) or {}


# ---------------------------------------------------------------------------
# ci.yml
# ---------------------------------------------------------------------------


def test_ci_name(ci: dict) -> None:
    assert ci["name"] == "ci"


def test_ci_triggers_push_and_pr(ci: dict) -> None:
    triggers = _triggers(ci)
    assert "push" in triggers
    assert "pull_request" in triggers
    assert "main" in triggers["push"]["branches"]
    assert "main" in triggers["pull_request"]["branches"]


def test_ci_has_required_jobs(ci: dict) -> None:
    required = {"lint", "typecheck", "test"}
    assert required <= set(ci["jobs"].keys())


def test_ci_test_job_uses_python_matrix(ci: dict) -> None:
    matrix = ci["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    assert "3.11" in matrix
    assert "3.12" in matrix


def test_ci_test_job_uploads_to_codecov(ci: dict) -> None:
    steps = ci["jobs"]["test"]["steps"]
    assert any("codecov" in step.get("uses", "").lower() for step in steps), (
        "Expected a codecov upload step in the test job"
    )


def test_ci_concurrency_cancels_in_progress(ci: dict) -> None:
    """Force-pushes should not burn stale CI minutes."""
    assert ci["concurrency"]["cancel-in-progress"] is True


# ---------------------------------------------------------------------------
# security.yml
# ---------------------------------------------------------------------------


REQUIRED_SECURITY_JOBS: Final[frozenset[str]] = frozenset(
    {"bandit", "pip-audit", "gitleaks", "codeql", "trivy"}
)


def test_security_name(security: dict) -> None:
    assert security["name"] == "security"


def test_security_has_required_jobs(security: dict) -> None:
    missing = REQUIRED_SECURITY_JOBS - set(security["jobs"].keys())
    assert not missing, f"Missing security jobs: {missing}"


def test_security_runs_on_schedule(security: dict) -> None:
    """Weekly cron catches newly disclosed CVEs even without code changes."""
    triggers = _triggers(security)
    assert "schedule" in triggers


def test_security_trivy_runs_only_on_tags(security: dict) -> None:
    """Image scans on every push would burn ~15 minutes per push."""
    trivy = security["jobs"]["trivy"]
    assert "if" in trivy
    assert "refs/tags/v" in trivy["if"]


# ---------------------------------------------------------------------------
# docker.yml
# ---------------------------------------------------------------------------


def test_docker_name(docker: dict) -> None:
    assert docker["name"] == "docker"


def _image_registry(image_ref: str) -> str:
    """Return the registry hostname of an OCI image reference.

    Strips any trailing tag (``:v1``) or digest (``@sha256:...``), then
    keeps only the part before the first ``/`` -- which is the registry
    by the OCI distribution spec, e.g. ``ghcr.io/momo3972/deepvision-api``
    -> ``ghcr.io``. A reference without a slash (e.g. ``python:3.12``)
    has the implicit Docker Hub registry; we return an empty string in
    that case so the caller can detect it.
    """
    # Drop digest suffix first, then tag suffix.
    image_ref = image_ref.split("@", 1)[0]
    image_ref = image_ref.split(":", 1)[0]
    if "/" not in image_ref:
        return ""
    return image_ref.split("/", 1)[0]


def test_docker_pushes_to_ghcr(docker: dict) -> None:
    """The audit prescribes GHCR (free, native GITHUB_TOKEN auth).

    Inspects the ``images:`` input of the ``docker/metadata-action`` step
    and validates that **every** image reference resolves to the
    ``ghcr.io`` registry. Parsing the structured field is more robust
    than a substring search and avoids CodeQL's
    *Incomplete URL substring sanitization* warning.
    """
    steps = docker["jobs"]["build-push"]["steps"]

    image_refs: list[str] = []
    for step in steps:
        with_block = step.get("with")
        if not isinstance(with_block, dict):
            continue
        images_field = with_block.get("images")
        if not images_field:
            continue
        # The action accepts a single string or a YAML block-list of strings.
        if isinstance(images_field, str):
            image_refs.extend(line.strip() for line in images_field.splitlines() if line.strip())
        elif isinstance(images_field, list):
            image_refs.extend(str(item).strip() for item in images_field if str(item).strip())

    assert image_refs, "Expected docker/metadata-action to declare at least one image"
    for ref in image_refs:
        assert _image_registry(ref) == "ghcr.io", f"Image reference {ref!r} does not target ghcr.io"


def test_docker_builds_all_three_images(docker: dict) -> None:
    images = docker["jobs"]["build-push"]["strategy"]["matrix"]["image"]
    names = {img["name"] for img in images}
    assert {"api", "streamlit", "training"} <= names


def test_docker_has_packages_write_permission(docker: dict) -> None:
    """Required to push to GHCR with the GITHUB_TOKEN."""
    assert docker["permissions"]["packages"] == "write"


# ---------------------------------------------------------------------------
# docs.yml
# ---------------------------------------------------------------------------


def test_docs_skips_until_mkdocs_present(docs: dict) -> None:
    """Phase 11 populates docs/; until then docs.yml must stay green."""
    jobs = docs["jobs"]
    assert "check-docs-source" in jobs
    # ``build`` runs only when the gate detects mkdocs.yml.
    assert jobs["build"]["if"].strip().startswith("needs.check-docs-source.outputs.ready")


def test_docs_deploys_only_from_main(docs: dict) -> None:
    deploy = docs["jobs"]["deploy"]
    assert "refs/heads/main" in deploy["if"]


# ---------------------------------------------------------------------------
# dependabot.yml
# ---------------------------------------------------------------------------


def test_dependabot_version(dependabot: dict) -> None:
    assert dependabot["version"] == 2


def test_dependabot_tracks_three_ecosystems(dependabot: dict) -> None:
    ecosystems = {entry["package-ecosystem"] for entry in dependabot["updates"]}
    assert {"pip", "docker", "github-actions"} <= ecosystems


@pytest.mark.parametrize("ecosystem", ["pip", "docker", "github-actions"])
def test_dependabot_weekly_schedule(dependabot: dict, ecosystem: str) -> None:
    entry = next(e for e in dependabot["updates"] if e["package-ecosystem"] == ecosystem)
    assert entry["schedule"]["interval"] == "weekly"


# ---------------------------------------------------------------------------
# Issue templates + PR template
# ---------------------------------------------------------------------------


REQUIRED_ISSUE_FORMS: Final[frozenset[str]] = frozenset(
    {"bug_report.yml", "feature_request.yml", "model_issue.yml", "drift_report.yml"}
)


def test_all_issue_templates_present() -> None:
    files = {p.name for p in ISSUE_DIR.iterdir() if p.is_file()}
    missing = REQUIRED_ISSUE_FORMS - files
    assert not missing, f"Missing issue templates: {missing}"


@pytest.mark.parametrize("form", sorted(REQUIRED_ISSUE_FORMS))
def test_issue_template_parses_and_has_name(form: str) -> None:
    data = _load(ISSUE_DIR / form)
    assert data.get("name"), f"{form}: missing 'name'"
    assert data.get("description"), f"{form}: missing 'description'"


def test_issue_config_disables_blank_issues() -> None:
    cfg = _load(ISSUE_DIR / "config.yml")
    assert cfg["blank_issues_enabled"] is False


def test_pr_template_lists_quality_gates(pr_template: str) -> None:
    """The PR template must remind contributors to run ruff / mypy / pytest."""
    for gate in ("ruff check", "ruff format", "mypy", "pytest"):
        assert gate in pr_template, f"PR template missing gate: {gate!r}"


def test_pr_template_mentions_changelog(pr_template: str) -> None:
    assert "CHANGELOG.md" in pr_template


# ---------------------------------------------------------------------------
# README badges -- ensure they point at the workflows we just wrote.
# ---------------------------------------------------------------------------


def test_readme_has_ci_badge(readme: str) -> None:
    assert "actions/workflows/ci.yml/badge.svg" in readme


def test_readme_has_security_badge(readme: str) -> None:
    assert "actions/workflows/security.yml/badge.svg" in readme


def test_readme_has_codecov_badge(readme: str) -> None:
    assert "codecov.io/gh/Momo3972/deepvision-cifar10-classifier" in readme


def test_readme_advertises_python_matrix(readme: str) -> None:
    """The Python badge must reflect the CI matrix (3.11 + 3.12)."""
    assert "3.11" in readme
    assert "3.12" in readme
