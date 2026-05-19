# API reference

The API reference is **auto-generated** from the NumPy-style
docstrings present throughout `src/deepvision/`. Every module exposes
its public surface here, with cross-links to the source and to the
relevant tutorial.

## How to navigate

The sidebar groups modules by feature area:

- **Configuration** -- the central `Settings` object and runtime
  constants.
- **Data** -- CIFAR-10 loader, preprocessing utilities, augmentation
  layers.
- **Models** -- the three model families and their registry.
- **Training** -- the `run_training` entrypoint, MLflow utilities, and
  Keras callbacks.
- **Evaluation** -- metrics, calibration, robustness, Grad-CAM,
  generic latency benchmark.
- **Serving** -- the FastAPI app, the lazy InferenceEngine, the
  Pydantic schemas and the Prometheus middleware.
- **Monitoring** -- drift detection, OOD scoring, baseline I/O, the
  Prometheus exporter.
- **Export** -- ONNX / TFLite conversion and the multi-runtime
  benchmark (Phase 10).

## Quick links

| You want to... | Look at |
|---|---|
| Configure the package | [`deepvision.config`][deepvision.config] |
| Load CIFAR-10 reproducibly | [`deepvision.data.loader`][deepvision.data.loader] |
| Build a model | [`deepvision.models.registry`][deepvision.models.registry] |
| Train end-to-end | [`deepvision.training.train`][deepvision.training.train] |
| Run inference programmatically | [`deepvision.serving.inference`][deepvision.serving.inference] |
| Measure drift | [`deepvision.monitoring.drift`][deepvision.monitoring.drift] |
| Export to ONNX | [`deepvision.export.onnx`][deepvision.export.onnx] |
| Benchmark runtimes | [`deepvision.export.benchmark`][deepvision.export.benchmark] |

---

## `deepvision.config`

::: deepvision.config

## `deepvision.constants`

::: deepvision.constants

## `deepvision.data.loader`

::: deepvision.data.loader

## `deepvision.data.preprocessing`

::: deepvision.data.preprocessing

## `deepvision.models.registry`

::: deepvision.models.registry

## `deepvision.training.train`

::: deepvision.training.train

## `deepvision.evaluation.robustness`

::: deepvision.evaluation.robustness

## `deepvision.serving.api`

::: deepvision.serving.api

## `deepvision.serving.inference`

::: deepvision.serving.inference

## `deepvision.serving.preprocess`

::: deepvision.serving.preprocess

## `deepvision.monitoring.drift`

::: deepvision.monitoring.drift

## `deepvision.monitoring.ood`

::: deepvision.monitoring.ood

## `deepvision.monitoring.baseline`

::: deepvision.monitoring.baseline

## `deepvision.export.onnx`

::: deepvision.export.onnx

## `deepvision.export.tflite`

::: deepvision.export.tflite

## `deepvision.export.benchmark`

::: deepvision.export.benchmark
