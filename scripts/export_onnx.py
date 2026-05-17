"""Standalone CLI wrapper around :func:`deepvision.export.onnx.export_to_onnx`.

This script is a thin convenience layer for users who want to convert a
Keras model to ONNX without remembering the ``python -m deepvision export
onnx ...`` invocation. It exposes the same arguments via ``argparse`` so
it can also be wired into shell pipelines or CI matrix jobs without
pulling Typer into the dependency graph.

Examples
--------
Convert a trained EfficientNetB0::

    python scripts/export_onnx.py \\
        --model-path models/efficientnet_best.keras \\
        --output models/exports/efficientnet.onnx

Disable the forward-pass equivalence validation (faster, less safe)::

    python scripts/export_onnx.py \\
        --model-path models/efficientnet_best.keras \\
        --output models/exports/efficientnet.onnx \\
        --no-validate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Keras model to ONNX format via tf2onnx. "
            "Conversion goes through a transient SavedModel for robustness "
            "on the Keras 3 / TF 2.21+ stack."
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
        help="Destination .onnx file. Parent directories are created if missing.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (1-22). Default 17.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the forward-pass equivalence check after export.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args(argv)

    # Imports deferred so ``--help`` returns instantly.
    import tensorflow as tf

    from deepvision.export.onnx import export_to_onnx

    print(f"Loading Keras model from {args.model_path}")
    model = tf.keras.models.load_model(str(args.model_path))

    output_path = export_to_onnx(
        model,
        args.output,
        opset=args.opset,
        validate=not args.no_validate,
    )
    print(f"ONNX export written to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
