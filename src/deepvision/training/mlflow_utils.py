"""
MLflow integration helpers.

Provides a small, side-effect-free API around ``mlflow`` so the training
script stays readable and testable. The core utilities are:

- :func:`setup_mlflow`: configure tracking URI + experiment.
- :func:`log_dataset_metadata`: log :class:`~deepvision.data.loader.CifarSplit` summary.
- :func:`log_environment_metadata`: log Python/TensorFlow/git provenance.
- :func:`log_classification_artifacts`: persist confusion matrix + per-class report
  as files inside the run.
"""

from __future__ import annotations

import json
import os
import platform

# Note: subprocess is only used to read the current git SHA via a fixed
# argv list (no shell, no untrusted input). Bandit B404/B603/B607 are skipped
# globally in pyproject.toml's [tool.bandit] section.
import subprocess  # nosec
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepvision.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator

    from deepvision.data.loader import CifarSplit

log = get_logger(__name__)


def setup_mlflow(
    experiment_name: str = "deepvision-cifar10",
    tracking_uri: str | None = None,
) -> str:
    """Configure MLflow tracking URI and experiment.

    Parameters
    ----------
    experiment_name
        Name of the MLflow experiment (created if missing).
    tracking_uri
        Optional override. Defaults to a local ``./mlruns`` directory
        rooted at the current working directory.

    Returns
    -------
    str
        The tracking URI actually used (resolved to absolute path).
    """
    import mlflow

    if tracking_uri is None:
        tracking_uri = _local_mlruns_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    log.info("MLflow tracking URI = %s, experiment = %s", tracking_uri, experiment_name)
    return tracking_uri


def _local_mlruns_uri() -> str:
    """Return a ``file://`` URI pointing to ``./mlruns/`` (Windows-safe)."""
    mlruns_dir = (Path.cwd() / "mlruns").resolve()
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    return mlruns_dir.as_uri()


@contextmanager
def start_run(run_name: str, tags: dict[str, str] | None = None) -> Iterator[Any]:
    """Context manager wrapping :func:`mlflow.start_run` with safe cleanup."""
    import mlflow

    with mlflow.start_run(run_name=run_name, tags=tags or {}) as active_run:
        log.info("MLflow run started: %s (id=%s)", run_name, active_run.info.run_id)
        yield active_run


def log_dataset_metadata(split: CifarSplit) -> None:
    """Log the dataset summary as MLflow params + a JSON artifact."""
    import mlflow

    summary = split.summary()
    for key, value in summary.items():
        mlflow.log_param(f"dataset.{key}", value)

    artifact_path = Path("dataset_summary.json")
    artifact_path.write_text(json.dumps(summary, indent=2))
    mlflow.log_artifact(str(artifact_path))
    artifact_path.unlink(missing_ok=True)


def log_environment_metadata() -> None:
    """Log Python / TensorFlow versions and git commit (best-effort)."""
    import mlflow

    mlflow.log_param("env.python_version", platform.python_version())
    mlflow.log_param("env.platform", platform.platform())

    tf_version = _safe_tf_version()
    if tf_version:
        mlflow.log_param("env.tensorflow_version", tf_version)

    git_commit = _safe_git_commit()
    if git_commit:
        mlflow.log_param("env.git_commit", git_commit)


def log_classification_artifacts(
    classification_report: str,
    confusion_matrix: list[list[int]],
    class_names: list[str] | tuple[str, ...],
) -> None:
    """Persist per-class report and confusion matrix as MLflow artifacts.

    The classification report is logged as plain text; the confusion matrix
    is serialized to a JSON file with class labels for easy downstream usage.
    """
    import mlflow

    report_file = Path("classification_report.txt")
    report_file.write_text(classification_report)
    mlflow.log_artifact(str(report_file))
    report_file.unlink(missing_ok=True)

    cm_file = Path("confusion_matrix.json")
    cm_payload = {
        "class_names": list(class_names),
        "matrix": confusion_matrix,
    }
    cm_file.write_text(json.dumps(cm_payload, indent=2))
    mlflow.log_artifact(str(cm_file))
    cm_file.unlink(missing_ok=True)


def _safe_tf_version() -> str | None:
    try:
        import tensorflow as tf

        return str(tf.__version__)
    except ImportError:
        return None


def _safe_git_commit() -> str | None:
    """Return the current git commit SHA, or None if outside a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            cwd=os.getcwd(),
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
