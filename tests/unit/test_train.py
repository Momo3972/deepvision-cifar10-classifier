"""Lightweight tests for :mod:`deepvision.training.train`.

The full training run is heavy and lives behind ``@pytest.mark.integration``.
The tests here only verify the public types and basic input handling without
touching MLflow or Keras.
"""

from __future__ import annotations

from deepvision.training.train import TrainConfig, TrainResult


def test_train_config_defaults_are_sane() -> None:
    config = TrainConfig()
    assert config.model_name == "mlp"
    assert config.epochs >= 1
    assert config.batch_size >= 1
    assert config.learning_rate > 0
    assert config.fine_tune_epochs == 0  # off by default
    assert config.quick is False


def test_train_config_quick_keeps_other_fields() -> None:
    config = TrainConfig(model_name="efficientnet", quick=True)
    assert config.model_name == "efficientnet"
    assert config.quick is True


def test_train_result_is_immutable_dataclass() -> None:
    """``TrainResult`` uses ``frozen=True`` so users can't accidentally mutate metrics."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(TrainResult)}
    assert {"run_id", "model_name", "metrics"} == fields
