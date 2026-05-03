"""
Model evaluation, interpretability and benchmarking.

Public API
----------
- :func:`deepvision.evaluation.metrics.evaluate_model`: full classification metrics.
- :func:`deepvision.evaluation.calibration.expected_calibration_error`: ECE.
- :func:`deepvision.evaluation.calibration.fit_temperature`: temperature scaling.
- :func:`deepvision.evaluation.calibration.apply_temperature`: temperature scaling.
- :func:`deepvision.evaluation.calibration.reliability_diagram_data`: plotting helper.
- :func:`deepvision.evaluation.interpretability.grad_cam`: Grad-CAM heatmap.
- :func:`deepvision.evaluation.interpretability.find_last_conv_layer`: helper.
- :func:`deepvision.evaluation.interpretability.overlay_heatmap_on_image`: visualization.
- :class:`deepvision.evaluation.benchmark.LatencyResult`: latency dataclass.
- :func:`deepvision.evaluation.benchmark.benchmark_callable`: generic benchmark.
- :func:`deepvision.evaluation.benchmark.benchmark_keras_model`: Keras helper.
- :class:`deepvision.evaluation.robustness.RobustnessReport`: aggregate report.
- :func:`deepvision.evaluation.robustness.evaluate_robustness`: full CIFAR-10-C run.
- :data:`deepvision.evaluation.robustness.STANDARD_CORRUPTIONS`: 15 official names.
"""

from __future__ import annotations

from deepvision.evaluation.benchmark import (
    LatencyResult,
    benchmark_callable,
    benchmark_keras_model,
)
from deepvision.evaluation.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_diagram_data,
)
from deepvision.evaluation.interpretability import (
    find_last_conv_layer,
    grad_cam,
    overlay_heatmap_on_image,
)
from deepvision.evaluation.metrics import evaluate_model
from deepvision.evaluation.robustness import (
    IMAGES_PER_SEVERITY,
    N_SEVERITIES,
    STANDARD_CORRUPTIONS,
    RobustnessReport,
    discover_corruptions,
    evaluate_corruption,
    evaluate_robustness,
)

__all__ = [
    "IMAGES_PER_SEVERITY",
    "N_SEVERITIES",
    "STANDARD_CORRUPTIONS",
    "LatencyResult",
    "RobustnessReport",
    "apply_temperature",
    "benchmark_callable",
    "benchmark_keras_model",
    "discover_corruptions",
    "evaluate_corruption",
    "evaluate_model",
    "evaluate_robustness",
    "expected_calibration_error",
    "find_last_conv_layer",
    "fit_temperature",
    "grad_cam",
    "overlay_heatmap_on_image",
    "reliability_diagram_data",
]
