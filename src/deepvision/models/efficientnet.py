"""
EfficientNetB0 transfer-learning model for CIFAR-10.

Two-stage strategy
------------------
1. **Feature extraction**: the ImageNet-pretrained backbone is frozen and only
   the classification head is trained at a normal learning rate.
2. **Fine-tuning**: the top ``n_unfrozen`` layers of the backbone are unfrozen
   and the whole model is re-trained at a much smaller learning rate
   (typically 1e-5).

Phase 4 will add Grad-CAM interpretability on top of this model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepvision.constants import (
    IMG_SIZE_EFFICIENTNET,
    IMG_SIZE_NATIVE,
    NUM_CHANNELS,
    NUM_CLASSES,
)

if TYPE_CHECKING:  # pragma: no cover — type-check time only
    from tensorflow.keras import Model

#: Default number of trainable layers from the top during fine-tuning.
DEFAULT_UNFREEZE_LAYERS = 40


def build_efficientnet(
    *,
    input_height: int = IMG_SIZE_NATIVE,
    input_width: int = IMG_SIZE_NATIVE,
    input_channels: int = NUM_CHANNELS,
    target_size: int = IMG_SIZE_EFFICIENTNET,
    num_classes: int = NUM_CLASSES,
    augmentation = None,  # noqa: ANN001  (Keras Sequential | None)
    weights: str | None = "imagenet",
    head_dropout: float = 0.20,
    name: str = "efficientnet_b0_transfer",
):
    """Build the transfer-learning EfficientNetB0 model.

    Parameters
    ----------
    augmentation
        Optional Keras model inserted at the top of the graph (typically a
        :func:`deepvision.data.augmentation.build_augmentation_pipeline` output).
    weights
        Either ``"imagenet"`` to load pretrained weights or ``None`` for a
        randomly initialized backbone (used in unit tests to avoid the download).
    target_size
        EfficientNet works better on larger inputs than CIFAR-10's native 32.
        We upscale to ``target_size`` (default 160) inside the graph.

    Returns
    -------
    tensorflow.keras.Model
        Functional API model with a frozen backbone — call
        :func:`unfreeze_top_layers` to switch to the fine-tuning stage.
    """
    if not 0.0 <= head_dropout < 1.0:
        raise ValueError(f"head_dropout must be in [0, 1), got {head_dropout}")
    if target_size < input_height:
        raise ValueError(
            f"target_size ({target_size}) must be >= input_height ({input_height})"
        )

    from tensorflow.keras import Model  # noqa: PLC0415
    from tensorflow.keras.applications import EfficientNetB0  # noqa: PLC0415
    from tensorflow.keras.layers import (  # noqa: PLC0415
        BatchNormalization,
        Dense,
        Dropout,
        GlobalAveragePooling2D,
        Input,
        Resizing,
    )

    inputs = Input(shape=(input_height, input_width, input_channels), name="input")
    x = augmentation(inputs) if augmentation is not None else inputs
    x = Resizing(target_size, target_size, name="resize")(x)

    # Build the backbone as a self-contained sub-model (NOT via input_tensor),
    # so it appears as a single nested layer in the parent model — this is what
    # makes ``unfreeze_top_layers`` able to find it later.
    backbone = EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(target_size, target_size, input_channels),
    )
    backbone.trainable = False

    features = backbone(x)
    features = GlobalAveragePooling2D(name="gap")(features)
    features = BatchNormalization(name="bn_head")(features)
    features = Dropout(head_dropout, name="dropout_head")(features)
    outputs = Dense(num_classes, activation="softmax", name="output")(features)

    return Model(inputs=inputs, outputs=outputs, name=name)


def unfreeze_top_layers(
    model: "Model",
    n_unfrozen: int = DEFAULT_UNFREEZE_LAYERS,
) -> int:
    """Unfreeze the top ``n_unfrozen`` layers of the EfficientNet backbone.

    Parameters
    ----------
    model
        A model returned by :func:`build_efficientnet`.
    n_unfrozen
        Number of layers from the top that become trainable. Lower layers
        (which encode generic features like edges and textures) stay frozen.

    Returns
    -------
    int
        Number of layers actually marked trainable (useful for logging).

    Notes
    -----
    The caller is responsible for re-compiling the model with a smaller
    learning rate (``Adam(1e-5)`` is a sound default) after this call.
    """
    backbone = _find_efficientnet_backbone(model)
    backbone.trainable = True

    if n_unfrozen <= 0:
        for layer in backbone.layers:
            layer.trainable = False
        return 0

    n_layers = len(backbone.layers)
    if n_unfrozen > n_layers:
        n_unfrozen = n_layers
    for layer in backbone.layers[: n_layers - n_unfrozen]:
        layer.trainable = False
    return n_unfrozen


def _find_efficientnet_backbone(model: "Model"):
    """Return the EfficientNet sub-model embedded in ``model``."""
    for layer in model.layers:
        if "efficientnet" in layer.name.lower() and hasattr(layer, "layers"):
            return layer
    raise ValueError(
        "No EfficientNet backbone found inside the provided model — "
        "did you build it with build_efficientnet()?"
    )
