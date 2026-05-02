"""
Training pipeline.

Public API
----------
- :class:`deepvision.training.train.TrainConfig`: hyperparams of a run.
- :class:`deepvision.training.train.TrainResult`: output of a run.
- :func:`deepvision.training.train.run_training`: end-to-end training + MLflow run.
- :func:`deepvision.training.callbacks.build_default_callbacks`: shared callbacks.
- :func:`deepvision.training.mlflow_utils.setup_mlflow`: tracking configuration.
- :func:`deepvision.training.mlflow_utils.start_run`: context-manager wrapper.
- :func:`deepvision.training.mlflow_utils.log_dataset_metadata`: split summary -> MLflow.
- :func:`deepvision.training.mlflow_utils.log_environment_metadata`: env -> MLflow.
- :func:`deepvision.training.mlflow_utils.log_classification_artifacts`: report + CM.
"""

from __future__ import annotations

from deepvision.training.callbacks import build_default_callbacks
from deepvision.training.mlflow_utils import (
    log_classification_artifacts,
    log_dataset_metadata,
    log_environment_metadata,
    setup_mlflow,
    start_run,
)
from deepvision.training.train import TrainConfig, TrainResult, run_training

__all__ = [
    "TrainConfig",
    "TrainResult",
    "build_default_callbacks",
    "log_classification_artifacts",
    "log_dataset_metadata",
    "log_environment_metadata",
    "run_training",
    "setup_mlflow",
    "start_run",
]
