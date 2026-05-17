"""Standalone CLI wrapper around :class:`deepvision.export.benchmark.LatencyBenchmark`.

Benchmark inference latency for any combination of the four supported
runtimes: Keras, TensorFlow SavedModel, ONNX Runtime (CPU) and TFLite.
Each runtime is benchmarked on a sweep of batch sizes; the script
reports p50 / p90 / p95 / p99 / mean / std / throughput for every
``(runtime, batch_size)`` pair.

Examples
--------
Four-way comparison of a trained EfficientNet::

    python scripts/benchmark_latency.py \\
        --keras-path models/efficientnet_best.keras \\
        --savedmodel-dir models/exports/efficientnet_savedmodel \\
        --onnx-path models/exports/efficientnet.onnx \\
        --tflite-path models/exports/efficientnet_int8.tflite \\
        --output-csv reports/phase10_benchmark.csv

Quick smoke benchmark (single runtime, single batch size, low iterations)::

    python scripts/benchmark_latency.py \\
        --onnx-path models/exports/efficientnet.onnx \\
        --batch-sizes 1 \\
        --n-warmup 5 \\
        --n-iter 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _parse_batch_sizes(raw: str) -> list[int]:
    """Parse a comma-separated batch-size string into a list of positive ints."""
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("--batch-sizes must contain at least one integer.")
    parsed: list[int] = []
    for item in items:
        try:
            value = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Non-integer batch size: {item!r}") from exc
        if value < 1:
            raise argparse.ArgumentTypeError(f"Batch size must be >= 1, got {value}")
        parsed.append(value)
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark inference latency across Keras / SavedModel / ONNX Runtime / TFLite. "
            "At least one of --keras-path, --savedmodel-dir, --onnx-path, --tflite-path is required."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--keras-path",
        type=Path,
        default=None,
        help="Path to a .keras file to benchmark with the Keras runtime.",
    )
    parser.add_argument(
        "--savedmodel-dir",
        type=Path,
        default=None,
        help="Path to a SavedModel directory to benchmark with the native TF runtime.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=None,
        help="Path to a .onnx file to benchmark with ONNX Runtime (CPU).",
    )
    parser.add_argument(
        "--tflite-path",
        type=Path,
        default=None,
        help="Path to a .tflite file to benchmark with the TFLite interpreter.",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=100,
        help="Warmup iterations per (runner, batch_size) pair.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1000,
        help="Measured iterations per (runner, batch_size) pair.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_parse_batch_sizes,
        default=_parse_batch_sizes("1,8,32"),
        help="Comma-separated list of batch sizes to sweep.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path to persist the result table.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args(argv)

    from deepvision.export.benchmark import (
        KerasRunner,
        LatencyBenchmark,
        OnnxRuntimeRunner,
        Runner,
        TFLiteRunner,
        TFSavedModelRunner,
        to_dataframe,
    )

    runners: list[Runner] = []
    if args.keras_path:
        runners.append(KerasRunner(args.keras_path))
    if args.savedmodel_dir:
        runners.append(TFSavedModelRunner(args.savedmodel_dir))
    if args.onnx_path:
        runners.append(OnnxRuntimeRunner(args.onnx_path))
    if args.tflite_path:
        runners.append(TFLiteRunner(args.tflite_path))

    if not runners:
        print(
            "Error: at least one of --keras-path, --savedmodel-dir, --onnx-path, "
            "--tflite-path must be provided.",
            file=sys.stderr,
        )
        return 2

    print(
        f"Running benchmark: {len(runners)} runtime(s) x "
        f"{len(args.batch_sizes)} batch size(s) x {args.n_iter} iterations "
        f"(after {args.n_warmup} warmup)."
    )

    bench = LatencyBenchmark(
        runners=runners,
        n_warmup=args.n_warmup,
        n_iter=args.n_iter,
        batch_sizes=args.batch_sizes,
    )
    results = bench.run()
    df = to_dataframe(results)

    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nResults written to {args.output_csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
