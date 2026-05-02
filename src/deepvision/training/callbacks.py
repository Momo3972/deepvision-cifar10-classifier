"""
Keras callbacks factory for training pipelines.

Centralized so that every model (MLP, CNN, EfficientNet) shares the same
EarlyStopping / ReduceLROnPlateau policy by default. Custom callbacks
(MLflow metric logger) live in :mod:`deepvision.training.mlflow_utils`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tensorflow.keras.callbacks import Callback


def build_default_callbacks(
    *,
    monitor: str = "val_loss",
    early_stopping_patience: int = 5,
    reduce_lr_patience: int = 3,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    restore_best_weights: bool = True,
    verbose: int = 1,
) -> list[Callback]:
    """Return the default Keras callbacks used during training.

    Parameters
    ----------
    monitor
        Quantity to monitor for both callbacks. Default ``val_loss``.
    early_stopping_patience
        Number of epochs without improvement before stopping.
    reduce_lr_patience
        Number of epochs without improvement before reducing the lr.
    reduce_lr_factor
        Multiplicative factor applied to the learning rate when triggered.
    min_lr
        Lower bound for the learning rate.
    restore_best_weights
        Whether EarlyStopping restores the weights of the best epoch.
    verbose
        Verbosity level forwarded to both callbacks.

    Returns
    -------
    list[Callback]
        ``[EarlyStopping, ReduceLROnPlateau]``.
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    return [
        EarlyStopping(
            monitor=monitor,
            patience=early_stopping_patience,
            restore_best_weights=restore_best_weights,
            verbose=verbose,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            patience=reduce_lr_patience,
            factor=reduce_lr_factor,
            min_lr=min_lr,
            verbose=verbose,
        ),
    ]
