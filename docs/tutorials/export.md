# Export & benchmark

This tutorial covers Phase 10: converting your trained `.keras` model
to ONNX and TFLite, then benchmarking inference latency across the
four supported runtimes.

## Why export at all

The trained Keras artefact is wedded to TensorFlow. Two scenarios
force you to leave:

1. **Edge deployment** -- mobile, embedded, microcontrollers. TFLite
   with Full INT8 quantization shrinks the model by ~4x and speeds
   inference up by 3-4x.
2. **Cross-framework portability** -- you need to run inside
   ONNX Runtime, Triton, TensorRT, OpenVINO, or any other runtime
   that speaks the ONNX Operator Set.

The export pipeline gives you both with a single command each, plus a
**latency benchmark** that lets you pick the right runtime for your
hardware before you commit to a deployment.

## Export to ONNX

```bash
python -m deepvision export onnx \
    --model-path models/efficientnet_best.keras \
    --output models/exports/efficientnet.onnx \
    --opset 17
```

The exporter uses `tf2onnx.convert.from_keras` with an explicit
`input_signature` built from `model.input_shape`. After the conversion
it runs a **forward-pass equivalence check**: random uniform inputs
are fed through both the Keras model and an ONNX Runtime session,
and the maximum absolute difference must stay below `1e-4`. If it
exceeds the tolerance the command fails -- a silently broken
conversion can never reach CI.

Disable the check (faster, less safe -- only useful in tight CI
matrices):

```bash
python -m deepvision export onnx --model-path X.keras --output Y.onnx --no-validate
```

### Why opset 17

| Opset | Pros | Cons |
|---|---|---|
| 15 | Maximum compatibility with legacy runtimes | No `LayerNormalization`/`Gelu` fused ops |
| **17** (default) | Stable, fused ops, supported by `onnxruntime >= 1.13` | -- |
| 20+ | Newer fusions, marginal perf gain | May break older `onnxruntime` builds |

## Export to TFLite

```bash
python -m deepvision export tflite \
    --model-path models/efficientnet_best.keras \
    --output models/exports/efficientnet_int8.tflite \
    --quantization int8 \
    --n-samples 200
```

The `--quantization` flag accepts four modes:

| Mode | What it does | Calibration data? |
|---|---|---|
| `dynamic` | Weights INT8, activations FP32 | No |
| `int8` (**default**) | Weights + activations INT8, I/O FP32 | Yes (200 CIFAR-10 images by default) |
| `int8_strict` | INT8 everywhere including I/O tensors | Yes |
| `fp16` | Weights FP16 | No |

For Full INT8 modes the command loads CIFAR-10 automatically and uses
the first `--n-samples` train images as the
[representative dataset][deepvision.export.tflite.build_representative_dataset].
The audit recommends 100-500 samples; 200 is the default sweet spot.

!!! tip "Use `int8` for drop-in replacement"
    Mode `int8` keeps I/O tensors as FP32, so the resulting `.tflite`
    file is a drop-in replacement for the FP32 `.keras` model -- you
    can swap one for the other in the serving stack without touching
    the preprocessing code. Use `int8_strict` only for pure edge
    deployments where you control quantization and dequantization on
    both sides.

## Benchmark latency across runtimes

```bash
python -m deepvision export benchmark \
    --keras-path models/efficientnet_best.keras \
    --savedmodel-dir models/exports/efficientnet_savedmodel \
    --onnx-path models/exports/efficientnet.onnx \
    --tflite-path models/exports/efficientnet_int8.tflite \
    --n-warmup 100 \
    --n-iter 1000 \
    --batch-sizes 1,8,32 \
    --output-csv reports/phase10_benchmark.csv
```

The benchmark loops over every `(runner, batch_size)` pair, runs
`--n-warmup` warmup iterations followed by `--n-iter` measured
iterations, and reports:

- `p50`, `p90`, `p95`, `p99` -- latency percentiles in milliseconds.
- `mean`, `std` -- distribution centre + spread.
- `throughput_ips` -- images per second, computed as
  `batch_size / mean_seconds`.

Sample output (CPU, 2 vCPU):

```
   runtime       batch_size  n_iter  p50_ms  p95_ms  p99_ms  mean_ms  throughput_ips
   keras          1          1000     19.2    24.1    31.4     19.8         50.5
   tf_savedmodel  1          1000     14.7    18.2    23.0     15.1         66.2
   onnx_runtime   1          1000      8.3    10.4    12.7      8.6        116.3
   tflite         1          1000      4.2     5.1     6.3      4.4        227.3
   ...
```

ONNX Runtime is typically 1.5x-2x faster than the native TF SavedModel
path on CPU, and TFLite is another 2x-3x on top thanks to INT8
arithmetic. Your mileage will vary -- run the benchmark on your
target hardware.

## Use the exports in your own code

```python
from pathlib import Path

from deepvision.export.benchmark import (
    OnnxRuntimeRunner,
    TFLiteRunner,
)

# ONNX
onnx = OnnxRuntimeRunner(Path("models/exports/efficientnet.onnx"))
onnx.load()
predictions = onnx.predict(images_batch)  # shape: (N, 10) softmax

# TFLite
tflite = TFLiteRunner(Path("models/exports/efficientnet_int8.tflite"))
tflite.load()
predictions = tflite.predict(images_batch)
```

The runner protocol is identical across runtimes -- you can swap one
for another with a single line change.

## Common pitfalls

!!! warning "Calibration data matters"
    Full INT8 quantization picks scales and zero-points based on the
    activation ranges it sees during calibration. If you feed it 200
    images of a single class, or 200 images that are systematically
    too dark, the resulting model will mis-quantize. The default
    helper samples uniformly from the CIFAR-10 train set with a
    pinned seed for reproducibility.

!!! warning "`tf2onnx.convert.from_keras` is fragile with Keras 3"
    The exporter passes an explicit `input_signature` built from
    `model.input_shape` to side-step Keras 3's graph-walking
    quirks. If you bypass the wrapper and call `from_keras` directly
    on a complex transfer-learning model, you may hit cryptic errors
    on BatchNormalization layers.

!!! tip "Bench on the deployment hardware"
    p95 latency depends on CPU model, RAM bandwidth, and concurrent
    load. The numbers above came from a 2-vCPU GitHub Actions
    runner; a Mac M1 Pro is ~3x faster across the board, and a
    Raspberry Pi 5 is ~5x slower. Always benchmark on the actual
    target.

## Next steps

- Architecture overview: [Architecture](../architecture.md).
- Model details: [Model card](../model-card.md).
