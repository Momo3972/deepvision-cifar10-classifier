"""
Model evaluation, interpretability and benchmarking.

Public API
----------
- :func:`deepvision.evaluation.metrics.evaluate_model`: full classification metrics.

Phase 4 will add:
- :mod:`deepvision.evaluation.calibration`: temperature scaling + ECE.
- :mod:`deepvision.evaluation.robustness`: CIFAR-10-C corruptions.
- :mod:`deepvision.evaluation.interpretability`: Grad-CAM and Integrated Gradients.
- :mod:`deepvision.evaluation.benchmark`: latency p50 / p90 / p95 / p99.
"""

from __future__ import annotations

from deepvision.evaluation.metrics import evaluate_model

__all__ = ["evaluate_model"]
