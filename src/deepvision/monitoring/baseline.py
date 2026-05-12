"""Baseline embedding distribution -- compute, persist, reload.

A *baseline* is the snapshot of the model's penultimate-layer activations
(and energy scores) on a representative reference set, captured offline
**once**. The :class:`~deepvision.monitoring.server.DriftMonitor` reloads
the baseline at startup and compares every fresh batch against it via
:func:`deepvision.monitoring.drift.wasserstein_drift`.

Persistence format
------------------
A single ``.npz`` file with the following keys:

==================  ==========  ====================================
key                 dtype       contents
==================  ==========  ====================================
``embeddings``      float64     ``(N, D)`` penultimate features
``energies``        float64     ``(N,)``   per-sample energy scores
``model_name``      ``<U64``    one-element array, model name
``model_version``   ``<U64``    one-element array, model version
``n_samples``       int64       one-element array, ``N``
``feature_dim``     int64       one-element array, ``D``
==================  ==========  ====================================

The format is intentionally minimal so the file can be opened with bare
``np.load`` from any operator workstation, MLflow artifact store, or
ad-hoc analysis notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from deepvision.constants import IMG_SIZE_EFFICIENTNET, NUM_CHANNELS
from deepvision.monitoring.ood import energy_score, extract_logits
from deepvision.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from tensorflow.keras import Model

log = get_logger(__name__)


@dataclass(frozen=True)
class Baseline:
    """In-memory representation of a baseline distribution snapshot."""

    embeddings: np.ndarray  # (N, D), float64
    energies: np.ndarray  # (N,),   float64
    model_name: str
    model_version: str

    @property
    def n_samples(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def save(self, path: Path) -> None:
        """Persist this baseline as a single ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            embeddings=self.embeddings.astype(np.float64),
            energies=self.energies.astype(np.float64),
            model_name=np.array([self.model_name]),
            model_version=np.array([self.model_version]),
            n_samples=np.array([self.n_samples], dtype=np.int64),
            feature_dim=np.array([self.feature_dim], dtype=np.int64),
        )
        log.info(
            "Saved baseline (n=%d, D=%d, model=%s@%s) -> %s",
            self.n_samples,
            self.feature_dim,
            self.model_name,
            self.model_version,
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Baseline:
        """Reload a baseline previously saved with :meth:`save`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Baseline file not found: {path}")
        data = np.load(path, allow_pickle=False)
        baseline = cls(
            embeddings=data["embeddings"].astype(np.float64),
            energies=data["energies"].astype(np.float64),
            model_name=str(data["model_name"][0]),
            model_version=str(data["model_version"][0]),
        )
        log.info(
            "Loaded baseline (n=%d, D=%d, model=%s@%s) <- %s",
            baseline.n_samples,
            baseline.feature_dim,
            baseline.model_name,
            baseline.model_version,
            path,
        )
        return baseline


def extract_embeddings(model: Model, batch: np.ndarray) -> np.ndarray:
    """Return penultimate-layer activations for ``batch``.

    Identifies the penultimate layer as the **input tensor of the final
    Dense classifier** -- a definition that survives variations in the head
    (extra Dropout, BatchNorm, etc.) as long as the last layer is a
    classifier ``Dense``. For our ``build_efficientnet`` head this is the
    1280-D global-average-pooled feature vector.

    Returns
    -------
    np.ndarray
        Embedding matrix shaped ``(N, D)`` and cast to ``float64`` so the
        downstream Wasserstein computation keeps double precision.

    Raises
    ------
    TypeError
        If the last layer is not a Dense.
    """
    import tensorflow as tf

    final = model.layers[-1]
    if not isinstance(final, tf.keras.layers.Dense):
        raise TypeError(
            "extract_embeddings expects the last layer to be a Dense classifier; "
            f"got {type(final).__name__}."
        )

    penultimate = tf.keras.Model(inputs=model.input, outputs=final.input)
    # Eager call avoids the ``tf.function`` retracing that ``model.predict``
    # triggers every time the batch size changes -- the drift monitor compares
    # 256-sample baseline batches with 64-sample poll batches, which would
    # otherwise spam the logs with retracing warnings.
    output = penultimate(tf.convert_to_tensor(batch), training=False)
    embeddings = output.numpy().astype(np.float64)
    return cast(np.ndarray, embeddings)


def compute_baseline(
    model: Model,
    images: np.ndarray,
    *,
    model_name: str,
    model_version: str,
) -> Baseline:
    """Compute a fresh baseline from ``images``.

    Parameters
    ----------
    model
        Keras model whose last layer is a Dense classifier.
    images
        Reference images, shape ``(N, H, W, C)``.
    model_name, model_version
        Identity tags persisted alongside the embeddings, used by the
        Grafana dashboard to sanity-check baseline / live-model agreement.
    """
    embeddings = extract_embeddings(model, images)
    logits = extract_logits(model, images)
    energies = energy_score(logits)
    return Baseline(
        embeddings=embeddings,
        energies=energies,
        model_name=model_name,
        model_version=model_version,
    )


def synthetic_reference_images(
    n: int = 256,
    *,
    seed: int = 42,
    image_size: int = IMG_SIZE_EFFICIENTNET,
) -> np.ndarray:
    """Generate deterministic synthetic images for smoke-testing the monitor.

    Returns
    -------
    np.ndarray
        Shape ``(n, image_size, image_size, NUM_CHANNELS)``, dtype
        ``uint8``, drawn from a fixed-seed uniform distribution. Statistically
        meaningless, but shape-correct for an EfficientNetB0 input -- enough
        to exercise the Prometheus exporter end-to-end without a real
        dataset attached to the container.
    """
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, 255, size=(n, image_size, image_size, NUM_CHANNELS), dtype=np.uint8)
