"""Out-of-distribution detection via the free energy score.

Phase 8 of the industrial refactor (audit section 7.2):

    "Detection d'OOD (out-of-distribution) via score d'energie ou
    Mahalanobis distance --- utile car CIFAR-10 est petit et l'utilisateur
    peut envoyer n'importe quoi."

We use the **free energy score** of Liu et al. (2020, NeurIPS):

    E(x) = -T * logsumexp(z(x) / T)

where ``z(x)`` are the model's logits and ``T`` is a temperature
hyper-parameter. Lower energy = more likely in-distribution. A threshold
``E*`` is calibrated on a held-out in-distribution set so that, e.g., 95 %
of ID samples have ``E < E*``; samples with ``E > E*`` are flagged as OOD.

Compared to the Mahalanobis-distance alternative the audit also mentions,
the energy score:

- needs no per-class mean / covariance estimation (an EfficientNetB0 head
  exposes 1280-D features -- the inverse covariance matrix is 1.6 M
  parameters before regularisation),
- requires only the existing forward pass (no Gaussian fitting at boot),
- is a single scalar per sample, easy to plot in Grafana and to alert on.

Reference
---------
Liu, Wang, Owens, Li (2020). *Energy-based Out-of-distribution Detection*.
Advances in Neural Information Processing Systems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.special import logsumexp

if TYPE_CHECKING:  # pragma: no cover
    from tensorflow.keras import Model


def energy_score(logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    """Compute the per-sample free energy score.

    .. math::

       E(x) = - T \\cdot \\mathrm{logsumexp}(z(x) / T)

    Parameters
    ----------
    logits
        Pre-softmax outputs, shape ``(N, num_classes)``.
    temperature
        ``T`` in the formula above. Defaults to ``1.0`` (the standard
        energy). Larger values smooth the distribution, reducing
        sensitivity to a single dominant logit.

    Returns
    -------
    np.ndarray
        Per-sample energy, shape ``(N,)``. Lower = more in-distribution.

    Raises
    ------
    ValueError
        If ``temperature <= 0`` or ``logits`` is not 2D.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits to be a 2D (N, num_classes) array; got shape {logits.shape}."
        )
    result = -temperature * logsumexp(logits / temperature, axis=1)
    return np.asarray(result, dtype=np.float64)


def is_ood(scores: np.ndarray, *, threshold: float) -> np.ndarray:
    """Boolean mask: ``True`` for samples flagged as out-of-distribution.

    Higher-energy samples are flagged. The threshold is typically
    calibrated on a held-out ID set so that a chosen false-positive rate
    (e.g. 5 %) of ID samples is mistakenly flagged.
    """
    return np.asarray(scores) > threshold


def ood_rate(scores: np.ndarray, *, threshold: float) -> float:
    """Fraction of samples in ``scores`` that exceed ``threshold``.

    Convenience wrapper used by the Prometheus exporter to publish a
    single gauge value per polling cycle.
    """
    if scores.size == 0:
        return 0.0
    return float(is_ood(scores, threshold=threshold).mean())


def extract_logits(model: Model, batch: np.ndarray) -> np.ndarray:
    """Return the pre-softmax logits for ``batch``.

    Strategy
    --------
    Our classifier head is the canonical ``... -> Dense(num_classes,
    activation='softmax')`` shape. Rather than rebuilding the head with a
    linear activation (which is fragile in Keras 3 when the model has
    already been called eagerly), we exploit the fact that

    .. math::

       z(x) = \\phi(x) \\cdot W + b

    where :math:`\\phi(x)` is the penultimate-layer activation feeding the
    final ``Dense``. We extract :math:`\\phi(x)` via a sub-model and apply
    the kernel and bias as a NumPy matmul -- no TF model surgery required.

    Parameters
    ----------
    model
        Keras model whose last layer is a Dense classifier (with any
        activation).
    batch
        Input batch shaped for the model.

    Returns
    -------
    np.ndarray
        Logits, shape ``(N, num_classes)``, dtype ``float64``.

    Raises
    ------
    TypeError
        If the last layer is not a Dense.
    """
    import tensorflow as tf

    final = model.layers[-1]
    if not isinstance(final, tf.keras.layers.Dense):
        raise TypeError(
            "extract_logits expects the last layer to be a Dense classifier; "
            f"got {type(final).__name__}."
        )

    penultimate = tf.keras.Model(inputs=model.input, outputs=final.input)
    # Eager call (model(batch)) instead of ``predict()`` avoids the
    # ``tf.function`` retracing warning that fires whenever the drift monitor
    # alternates between baseline-size and poll-size batches.
    output = penultimate(tf.convert_to_tensor(batch), training=False)
    features = output.numpy().astype(np.float64)

    weights, bias = final.get_weights()
    logits = features @ weights.astype(np.float64) + bias.astype(np.float64)
    return cast(np.ndarray, logits)
