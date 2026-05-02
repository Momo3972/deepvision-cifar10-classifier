"""
Tests for :mod:`deepvision.models`.

The EfficientNet builder is heavy (it instantiates 5.3 M parameters even with
``weights=None``). To keep the suite fast we only build it once via a
session-scoped fixture and check structural properties (output shape, layer
naming) rather than running ``predict`` on real data.
"""

from __future__ import annotations

import pytest

from deepvision.models import (
    DEFAULT_UNFREEZE_LAYERS,
    MODEL_REGISTRY,
    available_models,
    build_cnn,
    build_efficientnet,
    build_mlp,
    get_model,
    unfreeze_top_layers,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_lists_three_models() -> None:
    assert set(available_models()) == {"mlp", "cnn", "efficientnet"}


def test_registry_keys_match_module_keys() -> None:
    assert set(MODEL_REGISTRY.keys()) == set(available_models())


def test_get_model_with_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="Unknown model"):
        get_model("transformer")


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


def test_build_mlp_has_correct_io_shape() -> None:
    model = build_mlp()
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 10)


def test_build_mlp_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError, match="dropout"):
        build_mlp(dropout=1.5)


def test_get_model_mlp_returns_same_as_build_mlp() -> None:
    via_factory = get_model("mlp")
    via_direct = build_mlp()
    assert via_factory.count_params() == via_direct.count_params()
    assert via_factory.output_shape == via_direct.output_shape


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------


def test_build_cnn_has_correct_io_shape() -> None:
    model = build_cnn()
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 10)


def test_build_cnn_rejects_wrong_dropout_count() -> None:
    with pytest.raises(ValueError, match="dropouts must be 4"):
        build_cnn(dropouts=(0.2, 0.3, 0.4))  # type: ignore[arg-type]


def test_build_cnn_has_three_pooling_layers() -> None:
    model = build_cnn()
    pooling = [layer for layer in model.layers if "pool" in layer.name]
    assert len(pooling) == 3


# ---------------------------------------------------------------------------
# EfficientNet (heavier — built once for the module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def efficientnet_model():
    """Build a randomly initialized EfficientNet (no ImageNet download)."""
    return build_efficientnet(weights=None)


def test_efficientnet_has_correct_io_shape(efficientnet_model) -> None:
    assert efficientnet_model.input_shape == (None, 32, 32, 3)
    assert efficientnet_model.output_shape == (None, 10)


def test_efficientnet_includes_resize_layer(efficientnet_model) -> None:
    layer_names = [layer.name for layer in efficientnet_model.layers]
    assert "resize" in layer_names


def test_unfreeze_top_layers_returns_count(efficientnet_model) -> None:
    unfrozen = unfreeze_top_layers(efficientnet_model, n_unfrozen=DEFAULT_UNFREEZE_LAYERS)
    assert unfrozen == DEFAULT_UNFREEZE_LAYERS


def test_unfreeze_with_zero_keeps_backbone_frozen(efficientnet_model) -> None:
    unfrozen = unfreeze_top_layers(efficientnet_model, n_unfrozen=0)
    assert unfrozen == 0


def test_efficientnet_rejects_target_size_smaller_than_input() -> None:
    with pytest.raises(ValueError, match="target_size"):
        build_efficientnet(weights=None, target_size=16)
