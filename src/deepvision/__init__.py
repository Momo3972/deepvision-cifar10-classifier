"""
deepvision - Industrial Computer Vision pipeline for CIFAR-10.

Compares MLP, custom CNN and EfficientNetB0 (Transfer Learning + Data Augmentation)
on CIFAR-10, with full MLOps lifecycle: MLflow tracking & registry, FastAPI serving,
Docker, monitoring (drift + Prometheus + Grafana) and CI/CD.

See ``Audit_DeepVision_CIFAR10.docx`` at the repository root for the full audit
and 13-phase roadmap.
"""

from __future__ import annotations

__version__ = "0.9.0"
__author__ = "Mohamed Lamine OULD BOUYA"
__license__ = "MIT"

__all__ = ["__author__", "__license__", "__version__"]
