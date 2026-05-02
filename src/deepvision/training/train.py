"""
End-to-end training pipeline orchestrated by MLflow.

The pipeline is identical for the three architectures (mlp, cnn, efficientnet)
except for two specifics:

- MLP and CNN are trained on normalized inputs in ``[0, 1]``.
- EfficientNet is trained on raw uint8 inputs (Rescaling is internal) with an
  optional fine-tuning stage that unfreezes the top layers and lowers the lr.

Calling ``run_training`` returns a :class:`TrainResult` dataclass that
contains both the metrics dict (suitable for MLflow logging) and the MLflow
``run_id``, so downstream tools can fetch the artifact later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from deepvision.constants import CLASS_NAMES_EN, DEFAULT_BATCH_SIZE, DEFAULT_SEED
from deepvision.data.augmentation import AugmentationConfig, build_augmentation_pipeline
from deepvision.data.loader import CifarSplit, load_cifar10
from deepvision.data.preprocessing import normalize_to_unit, one_hot_encode
from deepvision.evaluation.metrics import evaluate_model
from deepvision.models.efficientnet import unfreeze_top_layers
from deepvision.models.registry import get_model
from deepvision.training.callbacks import build_default_callbacks
from deepvision.training.mlflow_utils import (
    log_classification_artifacts,
    log_dataset_metadata,
    log_environment_metadata,
    setup_mlflow,
    start_run,
)
from deepvision.utils.logging import get_logger
from deepvision.utils.seed import set_seed

if TYPE_CHECKING:  # pragma: no cover
    from tensorflow.keras import Model

log = get_logger(__name__)

ModelName = Literal["mlp", "cnn", "efficientnet"]


@dataclass(slots=True)
class TrainConfig:
    """Hyperparameters of a single training run.

    Attributes
    ----------
    model_name
        One of ``"mlp"``, ``"cnn"``, ``"efficientnet"``.
    epochs
        Number of epochs (feature-extraction stage for EfficientNet).
    batch_size
        Mini-batch size.
    learning_rate
        Initial Adam learning rate.
    seed
        Global seed for reproducibility.
    fine_tune_epochs
        EfficientNet only: extra epochs spent fine-tuning after unfreezing.
        Set to 0 to skip the fine-tuning stage.
    fine_tune_lr
        Learning rate used during the fine-tuning stage.
    quick
        If True, train on a 1 000-image subset and ignore ``epochs`` (use 1).
        Used by ``python -m deepvision train --quick`` to validate the
        pipeline locally on weak hardware.
    experiment_name
        MLflow experiment name.
    augmentation
        Optional augmentation hyperparameters (EfficientNet only).
    """

    model_name: ModelName = "mlp"
    epochs: int = 5
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = 1e-3
    seed: int = DEFAULT_SEED
    fine_tune_epochs: int = 0
    fine_tune_lr: float = 1e-5
    quick: bool = False
    experiment_name: str = "deepvision-cifar10"
    augmentation: AugmentationConfig | None = field(default_factory=AugmentationConfig)


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Output of a training run."""

    run_id: str
    model_name: str
    metrics: dict[str, Any]


def run_training(config: TrainConfig) -> TrainResult:
    """Train, evaluate and log a model end-to-end.

    Parameters
    ----------
    config
        A :class:`TrainConfig` instance describing the run.

    Returns
    -------
    TrainResult
        ``run_id``, ``model_name``, and a JSON-friendly metrics dict.
    """
    log.info("=== run_training: model=%s, quick=%s ===", config.model_name, config.quick)
    set_seed(config.seed)

    split = load_cifar10(seed=config.seed)
    if config.quick:
        split = _shrink_split(split, n_train=1_000, n_test=200)

    # Build inputs depending on the architecture's preprocessing contract.
    use_normalized = config.model_name in {"mlp", "cnn"}
    x_train_in, x_test_in = _prepare_inputs(split, use_normalized=use_normalized)
    y_train_oh = one_hot_encode(split.y_train)

    setup_mlflow(experiment_name=config.experiment_name)

    run_name = f"{config.model_name}-{'quick' if config.quick else 'full'}"
    with start_run(run_name=run_name, tags={"model": config.model_name}) as active_run:
        run_id = active_run.info.run_id

        # ---- log params + dataset + environment provenance --------------
        _log_config_params(config)
        log_dataset_metadata(split)
        log_environment_metadata()

        # ---- build the model --------------------------------------------
        builder_kwargs: dict[str, Any] = {}
        if config.model_name == "efficientnet":
            builder_kwargs["augmentation"] = (
                build_augmentation_pipeline(config.augmentation)
                if config.augmentation is not None
                else None
            )
            # ImageNet weights are downloaded only when not running in quick mode.
            builder_kwargs["weights"] = None if config.quick else "imagenet"

        model = get_model(config.model_name, **builder_kwargs)
        _compile(model, learning_rate=config.learning_rate)
        log.info("Model built: %d params", model.count_params())

        # ---- stage 1 — feature extraction or full training -------------
        epochs = 1 if config.quick else config.epochs
        history = model.fit(
            x_train_in,
            y_train_oh,
            epochs=epochs,
            batch_size=config.batch_size,
            validation_split=0.10,
            callbacks=build_default_callbacks(),
            verbose=2,
        )
        _log_history(history.history, prefix="stage1")

        # ---- stage 2 — fine-tuning (EfficientNet only, opt-in) ---------
        if config.model_name == "efficientnet" and config.fine_tune_epochs > 0 and not config.quick:
            log.info("Starting fine-tuning stage (lr=%g)", config.fine_tune_lr)
            unfreeze_top_layers(model)
            _compile(model, learning_rate=config.fine_tune_lr)
            ft_history = model.fit(
                x_train_in,
                y_train_oh,
                epochs=config.fine_tune_epochs,
                batch_size=config.batch_size,
                validation_split=0.10,
                callbacks=build_default_callbacks(),
                verbose=2,
            )
            _log_history(ft_history.history, prefix="stage2_finetune")

        # ---- evaluation -------------------------------------------------
        log.info("Evaluating on the held-out test set…")
        metrics = evaluate_model(
            model,
            x_test_in,
            split.y_test,
            class_names=CLASS_NAMES_EN,
            batch_size=config.batch_size,
        )
        _log_eval_metrics(metrics)
        log_classification_artifacts(
            classification_report=metrics["classification_report"],
            confusion_matrix=metrics["confusion_matrix"],
            class_names=CLASS_NAMES_EN,
        )

        # ---- model serialization ---------------------------------------
        _log_model_artifact(model, model_name=config.model_name)

    log.info(
        "Run finished: model=%s, accuracy=%.4f, run_id=%s",
        config.model_name,
        metrics["accuracy"],
        run_id,
    )
    return TrainResult(run_id=run_id, model_name=config.model_name, metrics=metrics)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_inputs(split: CifarSplit, *, use_normalized: bool) -> tuple[np.ndarray, np.ndarray]:
    if use_normalized:
        return normalize_to_unit(split.x_train), normalize_to_unit(split.x_test)
    return split.x_train, split.x_test


def _shrink_split(split: CifarSplit, *, n_train: int, n_test: int) -> CifarSplit:
    """Return a smaller split used by ``--quick``."""
    return CifarSplit(
        x_train=split.x_train[:n_train],
        y_train=split.y_train[:n_train],
        x_test=split.x_test[:n_test],
        y_test=split.y_test[:n_test],
        seed=split.seed,
        test_size=split.test_size,
        dataset_hash=split.dataset_hash + "-quick",
    )


def _compile(model: Model, learning_rate: float) -> None:
    from tensorflow.keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def _log_config_params(config: TrainConfig) -> None:
    import mlflow

    mlflow.log_params(
        {
            "model_name": config.model_name,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "quick": config.quick,
            "fine_tune_epochs": config.fine_tune_epochs,
            "fine_tune_lr": config.fine_tune_lr,
        }
    )


def _log_history(history: dict[str, list[float]], prefix: str) -> None:
    import mlflow

    for metric_name, values in history.items():
        for epoch_idx, value in enumerate(values):
            mlflow.log_metric(f"{prefix}.{metric_name}", float(value), step=epoch_idx)


def _log_eval_metrics(metrics: dict[str, Any]) -> None:
    import mlflow

    mlflow.log_metric("test_accuracy", metrics["accuracy"])
    mlflow.log_metric("test_loss", metrics["loss"])
    mlflow.log_metric("test_macro_f1", metrics["macro_f1"])
    mlflow.log_metric("test_weighted_f1", metrics["weighted_f1"])
    for class_name, f1 in metrics["per_class_f1"].items():
        mlflow.log_metric(f"f1.{class_name}", f1)


def _log_model_artifact(model: Model, *, model_name: str) -> None:
    """Save the model under MLflow's artifact store.

    Uses :func:`mlflow.tensorflow.log_model` when available so that the model
    can later be loaded with ``mlflow.tensorflow.load_model`` from any process.
    Falls back to a plain ``model.save`` artifact if the optional flavor isn't
    installed (keeps the function importable on minimal envs).
    """
    import mlflow

    try:
        import mlflow.tensorflow as mlf_tf

        mlf_tf.log_model(model, artifact_path=f"models/{model_name}")
    except Exception as exc:
        log.warning("mlflow.tensorflow.log_model failed (%s); falling back to plain save", exc)
        from pathlib import Path

        out = Path(f"_tmp_{model_name}.keras")
        model.save(out)
        mlflow.log_artifact(str(out), artifact_path="model_fallback")
        out.unlink(missing_ok=True)
