"""Tests for :mod:`deepvision.training.callbacks`."""

from __future__ import annotations

from deepvision.training.callbacks import build_default_callbacks


def test_build_default_callbacks_returns_two_entries() -> None:
    callbacks = build_default_callbacks()
    assert len(callbacks) == 2


def test_default_callbacks_contain_early_stopping_and_lr_reducer() -> None:
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    types = [type(cb).__name__ for cb in build_default_callbacks()]
    assert EarlyStopping.__name__ in types
    assert ReduceLROnPlateau.__name__ in types


def test_callbacks_use_provided_monitor() -> None:
    callbacks = build_default_callbacks(monitor="val_accuracy")
    assert all(cb.monitor == "val_accuracy" for cb in callbacks)


def test_callbacks_use_provided_patience() -> None:
    callbacks = build_default_callbacks(
        early_stopping_patience=10,
        reduce_lr_patience=4,
    )
    early_stopping, reduce_lr = callbacks
    assert early_stopping.patience == 10
    assert reduce_lr.patience == 4
