"""Model export package -- Phase 10.

Public surface of :mod:`deepvision.export`. The actual work happens in:

- :mod:`deepvision.export.onnx`     -- Keras -> ONNX conversion + validation
- :mod:`deepvision.export.tflite`   -- Keras -> TFLite (Full INT8) quantization
- :mod:`deepvision.export.benchmark` -- multi-runtime latency benchmark

Heavy dependencies (``tf2onnx``, ``onnxruntime``, ``tensorflow.lite``) are
imported lazily inside each submodule so that simply importing
``deepvision.export`` stays cheap.
"""

from __future__ import annotations

from deepvision.export.benchmark import (
    BenchmarkResult,
    KerasRunner,
    LatencyBenchmark,
    OnnxRuntimeRunner,
    Runner,
    TFLiteRunner,
    TFSavedModelRunner,
)
from deepvision.export.onnx import (
    DEFAULT_OPSET,
    ONNX_VALIDATION_TOLERANCE,
    export_to_onnx,
)
from deepvision.export.tflite import (
    DEFAULT_REPRESENTATIVE_SAMPLES,
    QuantizationMode,
    build_representative_dataset,
    export_to_tflite,
)

__all__ = [
    "DEFAULT_OPSET",
    "DEFAULT_REPRESENTATIVE_SAMPLES",
    "ONNX_VALIDATION_TOLERANCE",
    "BenchmarkResult",
    "KerasRunner",
    "LatencyBenchmark",
    "OnnxRuntimeRunner",
    "QuantizationMode",
    "Runner",
    "TFLiteRunner",
    "TFSavedModelRunner",
    "build_representative_dataset",
    "export_to_onnx",
    "export_to_tflite",
]
