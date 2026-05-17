"""Standalone CLI wrapper around :func:`deepvision.export.tflite.export_to_tflite`.

Quantize a Keras model to TFLite with four possible modes:

- ``dynamic``     -- weights INT8, activations FP32. No calibration data
                     needed.
- ``int8``        -- weights + activations INT8, I/O tensors FP32.
                     Drop-in replacement for the FP32 model. **Default.**
- ``int8_strict`` -- weights + activations + I/O all INT8. Pure edge mode.
- ``fp16``        -- weights FP16. Useful for GPU edge runtimes.

For ``int8`` and ``int8_strict`` modes the script downloads CIFAR-10
once via :func:`deepvision.data.loader.load_cifar10` and uses the first
``--n-samples`` train images as the representative dataset.

Examples
--------
Full INT8 export with 200 calibration images::

    python scripts/quantize_tflite.py \\
        --model-path models/efficientnet_best.keras \\
        --output models/exports/efficientnet_int8.tflite \\
        --quantization int8

Dynamic-range quantization (no calibration)::

    python scripts/quantize_tflite.py \\
        --model-path models/efficientnet_best.keras \\
        --output models/exports/efficientnet_dyn.tflite \\
        --quantization dynamic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Keras model to TFLite with optional Full INT8 / FP16 quantization. "
            "INT8 modes calibrate on a CIFAR-10 representative dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Path to the source .keras file or SavedModel directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination .tflite file. Parent directories are created if missing.",
    )
    parser.add_argument(
        "--quantization",
        "-q",
        choices=["dynamic", "int8", "int8_strict", "fp16"],
        default="int8",
        help="Post-training quantization mode.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Representative dataset size (only used for INT8 modes).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args(argv)

    import tensorflow as tf

    from deepvision.export.tflite import QuantizationMode, export_to_tflite

    print(f"Loading Keras model from {args.model_path}")
    model = tf.keras.models.load_model(str(args.model_path))

    mode = QuantizationMode(args.quantization)

    representative_data = None
    if mode in (QuantizationMode.INT8, QuantizationMode.INT8_STRICT):
        from deepvision.data.loader import load_cifar10

        print(f"Loading CIFAR-10 for representative dataset (n_samples={args.n_samples})")
        split = load_cifar10()
        representative_data = split.x_train

    output_path = export_to_tflite(
        model,
        args.output,
        quantization=mode,
        representative_data=representative_data,
        n_samples=args.n_samples,
    )
    print(f"TFLite export written to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
