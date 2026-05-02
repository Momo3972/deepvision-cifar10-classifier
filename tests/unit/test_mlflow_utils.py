"""Tests for :mod:`deepvision.training.mlflow_utils`.

Real MLflow operations are exercised through a temporary tracking URI
(``file:///tmp/...``) so the tests don't pollute the user's ``./mlruns``.
"""

from __future__ import annotations

from pathlib import Path

from deepvision.training.mlflow_utils import (
    _safe_git_commit,
    _safe_tf_version,
    setup_mlflow,
)


def test_safe_tf_version_returns_string_when_tf_installed() -> None:
    version = _safe_tf_version()
    # TF is a hard dependency of the project, so this must succeed in CI.
    assert version is not None
    assert isinstance(version, str)
    assert "." in version  # e.g. "2.21.0"


def test_safe_git_commit_returns_str_or_none() -> None:
    commit = _safe_git_commit()
    if commit is not None:
        assert isinstance(commit, str)
        assert 6 <= len(commit) <= 12  # short SHA


def test_setup_mlflow_creates_local_directory(tmp_path: Path, monkeypatch) -> None:
    """``setup_mlflow`` with no URI defaults to ``./mlruns``."""
    monkeypatch.chdir(tmp_path)
    uri = setup_mlflow(experiment_name="test-experiment")
    assert uri.startswith("file:")
    assert (tmp_path / "mlruns").exists()


def test_setup_mlflow_accepts_explicit_file_uri(tmp_path: Path) -> None:
    """``tracking_uri`` is honored when an explicit ``file://`` URI is provided."""
    target = (tmp_path / "explicit_mlruns").as_uri()
    returned_uri = setup_mlflow(experiment_name="test-explicit", tracking_uri=target)
    assert returned_uri == target
