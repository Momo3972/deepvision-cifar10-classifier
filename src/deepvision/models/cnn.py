"""
Custom VGG-style Convolutional Neural Network for CIFAR-10.

Three convolutional blocks (32 / 64 / 128 filters), each ending with a
``MaxPooling2D`` and a ``Dropout`` layer, followed by a dense classifier.
This architecture tops out around ~75 % accuracy, illustrating the gain
brought by spatial inductive bias compared to the MLP, but also the
ceiling of training from scratch on 50 000 images.
"""

from __future__ import annotations

from deepvision.constants import IMG_SIZE_NATIVE, NUM_CHANNELS, NUM_CLASSES


def build_cnn(
    *,
    input_height: int = IMG_SIZE_NATIVE,
    input_width: int = IMG_SIZE_NATIVE,
    input_channels: int = NUM_CHANNELS,
    num_classes: int = NUM_CLASSES,
    dropouts: tuple[float, float, float, float] = (0.20, 0.30, 0.40, 0.50),
    name: str = "cnn_vgg_style",
):
    """Return a Keras ``Sequential`` CNN with three convolutional blocks.

    Parameters
    ----------
    dropouts
        Four dropout rates: after pool 1, pool 2, pool 3, and the dense head.

    Architecture
    ------------
    Block 1: Conv(32) -> BN -> Conv(32) -> BN -> Pool -> Dropout
    Block 2: Conv(64) -> BN -> Conv(64) -> BN -> Pool -> Dropout
    Block 3: Conv(128) -> BN -> Conv(128) -> BN -> Pool -> Dropout
    Head:    Flatten -> Dense(128) -> BN -> Dropout -> Dense(num_classes, softmax)
    """
    if len(dropouts) != 4 or any(not 0.0 <= d < 1.0 for d in dropouts):
        raise ValueError(f"dropouts must be 4 values in [0, 1), got {dropouts}")

    from tensorflow.keras import Sequential  # noqa: PLC0415
    from tensorflow.keras.layers import (  # noqa: PLC0415
        BatchNormalization,
        Conv2D,
        Dense,
        Dropout,
        Flatten,
        Input,
        MaxPooling2D,
    )

    d1, d2, d3, d4 = dropouts
    return Sequential(
        [
            Input(shape=(input_height, input_width, input_channels), name="input"),
            # Block 1
            Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1"),
            BatchNormalization(name="bn1_1"),
            Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2"),
            BatchNormalization(name="bn1_2"),
            MaxPooling2D((2, 2), name="pool1"),
            Dropout(d1, name="drop1"),
            # Block 2
            Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1"),
            BatchNormalization(name="bn2_1"),
            Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2"),
            BatchNormalization(name="bn2_2"),
            MaxPooling2D((2, 2), name="pool2"),
            Dropout(d2, name="drop2"),
            # Block 3
            Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1"),
            BatchNormalization(name="bn3_1"),
            Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_2"),
            BatchNormalization(name="bn3_2"),
            MaxPooling2D((2, 2), name="pool3"),
            Dropout(d3, name="drop3"),
            # Head
            Flatten(name="flatten"),
            Dense(128, activation="relu", name="dense"),
            BatchNormalization(name="bn_dense"),
            Dropout(d4, name="drop_dense"),
            Dense(num_classes, activation="softmax", name="output"),
        ],
        name=name,
    )
