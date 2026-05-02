"""
Baseline Multi-Layer Perceptron for CIFAR-10.

Used as a reference scoreboard entry: by flattening 32x32x3 = 3 072 features,
the MLP cannot capture spatial structure, which caps its accuracy around
~48 %. Its purpose in the project is to make the CNN and EfficientNet gains
quantifiable.
"""

from __future__ import annotations

from deepvision.constants import IMG_SIZE_NATIVE, NUM_CHANNELS, NUM_CLASSES


def build_mlp(
    *,
    input_height: int = IMG_SIZE_NATIVE,
    input_width: int = IMG_SIZE_NATIVE,
    input_channels: int = NUM_CHANNELS,
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.30,
    name: str = "mlp_baseline",
):
    """Return a compiled-ready Keras ``Sequential`` MLP.

    Architecture
    ------------
    Flatten -> Dense(512, relu) -> BN -> Dropout
            -> Dense(256, relu) -> BN -> Dropout
            -> Dense(num_classes, softmax)
    """
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")

    from tensorflow.keras import Sequential  # noqa: PLC0415
    from tensorflow.keras.layers import (  # noqa: PLC0415
        BatchNormalization,
        Dense,
        Dropout,
        Flatten,
        Input,
    )

    return Sequential(
        [
            Input(shape=(input_height, input_width, input_channels), name="input"),
            Flatten(name="flatten"),
            Dense(512, activation="relu", name="dense_1"),
            BatchNormalization(name="bn_1"),
            Dropout(dropout, name="dropout_1"),
            Dense(256, activation="relu", name="dense_2"),
            BatchNormalization(name="bn_2"),
            Dropout(dropout, name="dropout_2"),
            Dense(num_classes, activation="softmax", name="output"),
        ],
        name=name,
    )
