"""
Factory registry mapping model names to builder functions.

Used by the training pipeline (Phase 3) and the CLI (`python -m deepvision
train --model {mlp,cnn,efficientnet}`) to select an architecture by name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from deepvision.models.cnn import build_cnn
from deepvision.models.efficientnet import build_efficientnet
from deepvision.models.mlp import build_mlp

if TYPE_CHECKING:  # pragma: no cover — type-check time only
    from tensorflow.keras import Model

#: Mapping of public model names to their builder functions.
MODEL_REGISTRY: dict[str, Callable[..., "Model"]] = {
    "mlp": build_mlp,
    "cnn": build_cnn,
    "efficientnet": build_efficientnet,
}


def available_models() -> list[str]:
    """Return the list of registered model names."""
    return sorted(MODEL_REGISTRY)


def get_model(name: str, **kwargs: Any) -> "Model":
    """Build a model from its registered name.

    Parameters
    ----------
    name
        One of :data:`MODEL_REGISTRY`'s keys.
    **kwargs
        Forwarded to the underlying builder.

    Raises
    ------
    KeyError
        If ``name`` is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Available: {available_models()}"
        )
    builder = MODEL_REGISTRY[name]
    return builder(**kwargs)
