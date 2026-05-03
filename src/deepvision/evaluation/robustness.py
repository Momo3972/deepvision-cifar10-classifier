"""
Robustness benchmarking with CIFAR-10-C.

CIFAR-10-C (Hendrycks & Dietterich, 2019) is a benchmark suite that applies
15 common corruptions (Gaussian noise, motion blur, fog, snow, jpeg, …) at
5 severity levels each to the original CIFAR-10 test set. A model robust to
distribution shift should keep its accuracy under these corruptions.

Dataset
-------
Hosted on Zenodo: https://zenodo.org/records/2535967

Each corruption is provided as a NumPy array of shape ``(50_000, 32, 32, 3)``
where the first 10 000 images are severity 1, the next 10 000 are severity 2,
etc. — i.e. the same 10 000 test images replayed 5 times with growing
corruption intensity.

Status
------
Phase 4 ships the **harness** (corruption discovery, evaluator,
mean-corruption-error aggregator). The actual download is **not** triggered
automatically because each corruption is ~30 MB and the full benchmark is
~12 GB. Users opt in via :func:`download_cifar10c` once the upstream
host (cs.toronto.edu) is reachable again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from deepvision.constants import IMG_SIZE_NATIVE, NUM_CHANNELS, NUM_CLASSES
from deepvision.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from tensorflow.keras import Model

log = get_logger(__name__)

#: All 15 standard CIFAR-10-C corruption names (Hendrycks & Dietterich 2019).
STANDARD_CORRUPTIONS: tuple[str, ...] = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)

#: Number of severity levels per corruption (always 5 in the canonical benchmark).
N_SEVERITIES: int = 5

#: Number of test images per severity level (always 10 000 in CIFAR-10).
IMAGES_PER_SEVERITY: int = 10_000


@dataclass(slots=True)
class RobustnessReport:
    """Aggregate result of a CIFAR-10-C run.

    Attributes
    ----------
    clean_accuracy
        Accuracy on the unperturbed CIFAR-10 test set.
    per_corruption
        ``{corruption_name: [acc_severity_1, ..., acc_severity_5]}``.
    per_corruption_mean
        ``{corruption_name: mean_accuracy_over_severities}``.
    mean_corruption_accuracy
        Average over all 15 corruptions x 5 severities.
    mean_corruption_error
        ``1 - mean_corruption_accuracy``. Lower is better.
    """

    clean_accuracy: float
    per_corruption: dict[str, list[float]] = field(default_factory=dict)
    per_corruption_mean: dict[str, float] = field(default_factory=dict)
    mean_corruption_accuracy: float = 0.0
    mean_corruption_error: float = 1.0

    def summarize(self) -> dict[str, float]:
        """Return a flat dict suitable for MLflow logging."""
        out: dict[str, float] = {
            "clean_accuracy": self.clean_accuracy,
            "mean_corruption_accuracy": self.mean_corruption_accuracy,
            "mean_corruption_error": self.mean_corruption_error,
        }
        for name, mean_acc in self.per_corruption_mean.items():
            out[f"corruption.{name}.mean_acc"] = mean_acc
        return out


def discover_corruptions(corruptions_dir: Path) -> list[str]:
    """List corruption names found locally, given a directory of ``*.npy`` files.

    Parameters
    ----------
    corruptions_dir
        Directory expected to contain ``<corruption>.npy`` files plus
        ``labels.npy`` (the ground truth, identical to CIFAR-10 test labels).

    Returns
    -------
    list[str]
        Sorted list of corruption names (without the ``.npy`` suffix), excluding
        ``labels``.
    """
    if not corruptions_dir.is_dir():
        return []
    return sorted(path.stem for path in corruptions_dir.glob("*.npy") if path.stem != "labels")


def evaluate_corruption(
    model: Model,
    corruption_path: Path,
    labels_path: Path,
    *,
    batch_size: int = 64,
    use_normalized_input: bool = False,
) -> list[float]:
    """Return the per-severity accuracy for one corruption file.

    Parameters
    ----------
    model
        Trained Keras model.
    corruption_path
        Path to the ``*.npy`` file containing 50 000 images
        (10 000 per severity, 5 severities concatenated).
    labels_path
        Path to ``labels.npy`` (10 000 ground-truth labels; reused for every
        severity per the CIFAR-10-C convention).
    batch_size
        Inference batch size.
    use_normalized_input
        ``True`` for MLP/CNN (expect ``[0, 1]`` floats), ``False`` for
        EfficientNet (expects raw uint8).

    Returns
    -------
    list[float]
        Five accuracies, one per severity level (1 → 5).
    """
    images = np.load(corruption_path)
    labels = np.load(labels_path).astype(int).reshape(-1)
    if images.shape[0] != N_SEVERITIES * IMAGES_PER_SEVERITY:
        raise ValueError(
            f"Expected {N_SEVERITIES * IMAGES_PER_SEVERITY} images in "
            f"{corruption_path}, got {images.shape[0]}"
        )

    accuracies: list[float] = []
    for severity in range(N_SEVERITIES):
        start = severity * IMAGES_PER_SEVERITY
        end = start + IMAGES_PER_SEVERITY
        x = images[start:end]
        x = x.astype(np.float32) / 255.0 if use_normalized_input else x.astype(np.float32)
        y_pred = np.argmax(model.predict(x, batch_size=batch_size, verbose=0), axis=1)
        acc = float((y_pred == labels).mean())
        accuracies.append(acc)
        log.info("  severity %d: accuracy = %.4f", severity + 1, acc)
    return accuracies


def evaluate_robustness(
    model: Model,
    corruptions_dir: Path,
    *,
    clean_images: np.ndarray,
    clean_labels: np.ndarray,
    batch_size: int = 64,
    use_normalized_input: bool = False,
    corruptions: tuple[str, ...] | None = None,
) -> RobustnessReport:
    """Run the full CIFAR-10-C benchmark and return an aggregated report.

    Parameters
    ----------
    corruptions_dir
        Directory containing ``<corruption>.npy`` files plus ``labels.npy``.
    clean_images, clean_labels
        Original (unperturbed) test set used to compute the clean accuracy.
    corruptions
        Subset of corruption names to evaluate. ``None`` means all 15 standard
        corruptions present in ``corruptions_dir``.

    Notes
    -----
    Requires CIFAR-10-C downloaded locally. Calling this function before
    :func:`download_cifar10c` will raise ``FileNotFoundError`` on missing files.
    """
    labels_path = corruptions_dir / "labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} not found. Download CIFAR-10-C first via "
            "deepvision.evaluation.robustness.download_cifar10c()."
        )

    available = set(discover_corruptions(corruptions_dir))
    if corruptions is None:
        target = tuple(c for c in STANDARD_CORRUPTIONS if c in available)
    else:
        target = corruptions
    if not target:
        raise FileNotFoundError(f"No usable corruption files found in {corruptions_dir}.")

    # ---- Clean accuracy -----------------------------------------------------
    if use_normalized_input:
        clean_x = clean_images.astype(np.float32) / 255.0
    else:
        clean_x = clean_images.astype(np.float32)
    clean_pred = np.argmax(model.predict(clean_x, batch_size=batch_size, verbose=0), axis=1)
    clean_acc = float((clean_pred == np.asarray(clean_labels).reshape(-1)).mean())
    log.info("Clean accuracy: %.4f", clean_acc)

    # ---- Per-corruption evaluation -----------------------------------------
    per_corruption: dict[str, list[float]] = {}
    per_corruption_mean: dict[str, float] = {}
    for name in target:
        log.info("Evaluating corruption '%s'…", name)
        accs = evaluate_corruption(
            model,
            corruptions_dir / f"{name}.npy",
            labels_path,
            batch_size=batch_size,
            use_normalized_input=use_normalized_input,
        )
        per_corruption[name] = accs
        per_corruption_mean[name] = float(np.mean(accs))

    overall_mean = float(np.mean([np.mean(a) for a in per_corruption.values()]))
    return RobustnessReport(
        clean_accuracy=clean_acc,
        per_corruption=per_corruption,
        per_corruption_mean=per_corruption_mean,
        mean_corruption_accuracy=overall_mean,
        mean_corruption_error=1.0 - overall_mean,
    )


def download_cifar10c(target_dir: Path) -> None:
    """Trigger the (heavy) CIFAR-10-C download into ``target_dir``.

    Not implemented in Phase 4 — the dataset is ~12 GB total and the upstream
    host (Zenodo + cs.toronto.edu mirror) was unreachable when this module
    was written. Use ``deepvision`` from a Colab notebook with internet
    access, or download manually from
    https://zenodo.org/records/2535967, then point
    :func:`evaluate_robustness` at the resulting directory.

    Raises
    ------
    NotImplementedError
        Always — see the rationale above.
    """
    raise NotImplementedError(
        "Automatic CIFAR-10-C download is intentionally deferred to Phase 5+. "
        "Download manually from https://zenodo.org/records/2535967 and "
        f"extract the .npy files into {target_dir}."
    )


def assert_cifar10c_shape(images: np.ndarray) -> None:
    """Raise if ``images`` does not have the canonical CIFAR-10-C shape."""
    expected_shape = (
        N_SEVERITIES * IMAGES_PER_SEVERITY,
        IMG_SIZE_NATIVE,
        IMG_SIZE_NATIVE,
        NUM_CHANNELS,
    )
    if images.shape != expected_shape:
        raise ValueError(
            f"Expected CIFAR-10-C images of shape {expected_shape}, got {images.shape}"
        )
    if NUM_CLASSES != 10:  # tautological, but clarifies intent in docs
        raise ValueError("CIFAR-10-C is only defined for the 10-class CIFAR-10.")
