"""
Neural-network architectures.

Public API
----------
- :func:`deepvision.models.mlp.build_mlp`: dense baseline (~48 % accuracy).
- :func:`deepvision.models.cnn.build_cnn`: VGG-style custom CNN (~75 %).
- :func:`deepvision.models.efficientnet.build_efficientnet`: transfer learning model (~93 %).
- :func:`deepvision.models.efficientnet.unfreeze_top_layers`: fine-tuning helper.
- :data:`deepvision.models.registry.MODEL_REGISTRY`: name -> builder mapping.
- :func:`deepvision.models.registry.get_model`: factory used by the training CLI.
- :func:`deepvision.models.registry.available_models`: list of registered names.
"""

from __future__ import annotations

from deepvision.models.cnn import build_cnn
from deepvision.models.efficientnet import (
    DEFAULT_UNFREEZE_LAYERS,
    build_efficientnet,
    unfreeze_top_layers,
)
from deepvision.models.mlp import build_mlp
from deepvision.models.registry import (
    MODEL_REGISTRY,
    available_models,
    get_model,
)

__all__ = [
    "DEFAULT_UNFREEZE_LAYERS",
    "MODEL_REGISTRY",
    "available_models",
    "build_cnn",
    "build_efficientnet",
    "build_mlp",
    "get_model",
    "unfreeze_top_layers",
]
