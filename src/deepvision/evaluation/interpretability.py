"""
Model interpretability -- Grad-CAM heatmaps.

Grad-CAM (Selvaraju et al., 2017) highlights the spatial regions of an input
image that contributed the most to a target class score. The method consists
of three steps:

1. Compute the gradient of the target class score w.r.t. the activations of a
   target convolutional layer.
2. Average the gradient over the spatial dimensions to obtain importance
   weights ``alpha_k`` for each feature map ``k``.
3. Compute the weighted sum ``sum_k alpha_k * A^k`` and apply ReLU to obtain
   the final heatmap, which is finally upsampled to the input resolution.

The implementation is pure TensorFlow/Keras and works on any model that
contains at least one ``Conv2D`` layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from deepvision.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from tensorflow.keras import Model

log = get_logger(__name__)


def find_last_conv_layer(model: Model) -> str:
    """Return the name of the last 2D-convolutional layer in ``model``.

    Walks the model recursively because EfficientNetB0 nests its backbone as
    a sub-model.

    Raises
    ------
    ValueError
        If no ``Conv2D`` layer is found.
    """
    last_name: str | None = None

    def _walk(submodel: Model) -> None:
        nonlocal last_name
        for layer in submodel.layers:
            cls_name = type(layer).__name__
            if cls_name == "Conv2D":
                last_name = layer.name
            if hasattr(layer, "layers"):
                _walk(layer)

    _walk(model)
    if last_name is None:
        raise ValueError("No Conv2D layer found inside the provided model.")
    return last_name


def grad_cam(
    model: Model,
    image: np.ndarray,
    *,
    target_class: int | None = None,
    layer_name: str | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for ``image`` and ``model``.

    Parameters
    ----------
    model
        A trained Keras model with a Conv2D layer somewhere in its graph.
    image
        Single image with shape ``(H, W, C)`` or batched ``(1, H, W, C)``.
    target_class
        Class index to explain. When ``None``, uses the class predicted by
        the model itself.
    layer_name
        Name of the convolutional layer whose activations are used.
        When ``None``, falls back to :func:`find_last_conv_layer`.
    eps
        Numerical stabilizer to avoid division by zero when normalizing.

    Returns
    -------
    np.ndarray
        Heatmap of shape ``(H, W)`` with values normalized to ``[0, 1]``.
    """
    import tensorflow as tf

    if image.ndim == 3:
        batch = np.expand_dims(image, axis=0)
    elif image.ndim == 4 and image.shape[0] == 1:
        batch = image
    else:
        raise ValueError(f"Expected (H, W, C) or (1, H, W, C), got {image.shape}")

    inputs_tensor = tf.convert_to_tensor(batch.astype(np.float32))

    if layer_name is None:
        layer_name = find_last_conv_layer(model)
    log.info("Grad-CAM target conv layer: %s", layer_name)

    target_layer = _find_layer(model, layer_name)

    # Build a grad model that exposes both the target conv activations and
    # the final predictions. We re-trace through the layers symbolically so
    # the approach works for both Functional models (EfficientNet) and
    # Sequential models (custom CNN), even on Keras 3 where ``model.output``
    # is unavailable on Sequential graphs that were only called eagerly.
    grad_model = _build_grad_model(model, inputs_tensor.shape[1:], target_layer)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(inputs_tensor, training=False)
        if target_class is None:
            target_class = int(tf.argmax(predictions[0]).numpy())
        class_score = predictions[:, target_class]

    grads = tape.gradient(class_score, conv_output)  # (1, h, w, k)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # (k,)

    cam = tf.reduce_sum(conv_output[0] * weights, axis=-1)  # (h, w)
    cam = tf.nn.relu(cam).numpy()

    # Normalize to [0, 1].
    cam_min, cam_max = float(cam.min()), float(cam.max())
    cam = (cam - cam_min) / (cam_max - cam_min + eps)

    # Resize to input spatial size.
    target_h, target_w = batch.shape[1:3]
    cam_resized = _resize_heatmap(cam, target_h, target_w)
    return cam_resized.astype(np.float32)


def _find_layer(model: Model, name: str):
    """Return the layer with ``name`` (searching recursively into sub-models)."""
    for layer in model.layers:
        if layer.name == name:
            return layer
        if hasattr(layer, "layers"):
            try:
                return _find_layer(layer, name)
            except ValueError:
                continue
    raise ValueError(f"Layer {name!r} not found in model {model.name!r}.")


def _build_grad_model(model: Model, input_shape, target_layer):
    """Return a Functional model exposing ``[target_conv_output, predictions]``.

    Tries the symbolic ``layer.output`` approach first (works for Functional
    models like EfficientNet), then falls back to a layer-by-layer re-tracing
    that also handles Sequential models (Keras 3 leaves ``model.output``
    undefined for Sequential graphs that have only been called eagerly).
    """
    import tensorflow as tf

    # Fast path: Functional model whose layers expose symbolic outputs.
    try:
        return tf.keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )
    except (AttributeError, ValueError):
        pass

    # Fallback: re-trace symbolically through the model's layers.
    new_inputs = tf.keras.Input(shape=input_shape)
    x = new_inputs
    target_tensor = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == target_layer.name:
            target_tensor = x
    if target_tensor is None:
        raise ValueError(
            f"Could not capture activations of layer {target_layer.name!r} during re-tracing."
        )
    return tf.keras.Model(inputs=new_inputs, outputs=[target_tensor, x])


def _resize_heatmap(cam: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize using TensorFlow (CPU is enough for tiny heatmaps)."""
    import tensorflow as tf

    resized = tf.image.resize(cam[..., None], (height, width), method="bilinear")
    return cast(np.ndarray, resized.numpy().squeeze(-1))


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an image for human inspection.

    Returns a uint8 RGB array of the same spatial size as ``image``.
    The heatmap is colorized using a viridis-like inferno gradient
    implemented inline (no matplotlib dependency).
    """
    if image.dtype == np.uint8:
        base = image.astype(np.float32) / 255.0
    else:
        base = np.clip(image, 0.0, 1.0).astype(np.float32)

    heatmap_rgb = _inferno_colormap(heatmap)
    blended = (1 - alpha) * base + alpha * heatmap_rgb
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


def _inferno_colormap(values: np.ndarray) -> np.ndarray:
    """Map a [0, 1] array to RGB floats using an inferno-style ramp.

    Avoids the matplotlib dependency at runtime.
    """
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 * v, 0.0, 1.0)
    g = np.clip(1.5 * v - 0.5, 0.0, 1.0)
    b = np.clip(2.0 * v - 1.0, 0.0, 1.0) * (v > 0.6)
    return np.stack([r, g, b], axis=-1).astype(np.float32)
