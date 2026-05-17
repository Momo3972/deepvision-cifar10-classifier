"""
Command-line interface for the deepvision package.

Subcommands
-----------
- ``version``:   print the package version and exit.
- ``info``:      print package metadata and resolved configuration paths.
- ``train``:     train one of the registered models with full MLflow tracking.
- ``serve``:     launch the FastAPI inference service via uvicorn.
- ``streamlit``: launch the Streamlit demo UI (Phase 6 refonte).
- ``drift-monitor``: launch the Prometheus drift + OOD exporter (Phase 8).

Usage
-----
.. code-block:: bash

    python -m deepvision --help
    python -m deepvision version
    python -m deepvision info
    python -m deepvision train --model efficientnet --epochs 1 --quick
    python -m deepvision serve --host 0.0.0.0 --port 8000
    python -m deepvision streamlit --port 8501
    python -m deepvision drift-monitor --port 9091 --interval 60
"""

from __future__ import annotations

import json
import sys
from typing import Any

import typer

from deepvision import __author__, __license__, __version__
from deepvision.config import get_settings

app = typer.Typer(
    name="deepvision",
    help="Industrial Computer Vision pipeline for CIFAR-10.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the deepvision package version and exit."""
    typer.echo(f"deepvision {__version__}")


@app.command()
def info(
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON instead of human-readable text.",
    ),
) -> None:
    """Print package metadata and resolved configuration paths."""
    settings = get_settings()
    payload: dict[str, Any] = {
        "package": "deepvision",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "python": sys.version.split()[0],
        "settings": {
            "seed": settings.seed,
            "repo_root": str(settings.repo_root),
            "models_dir": str(settings.models_dir),
            "data_dir": str(settings.data_dir),
            "mlruns_dir": str(settings.mlruns_dir),
            "log_level": settings.log_level,
            "log_format": settings.log_format,
        },
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    typer.echo(f"deepvision {__version__}")
    typer.echo(f"  author : {__author__}")
    typer.echo(f"  license: {__license__}")
    typer.echo(f"  python : {sys.version.split()[0]}")
    typer.echo("")
    typer.echo("Settings:")
    for key, value in payload["settings"].items():
        typer.echo(f"  {key:11s} : {value}")


@app.command()
def train(
    model: str = typer.Option(
        "mlp",
        "--model",
        "-m",
        help="Model name: mlp, cnn, or efficientnet.",
    ),
    epochs: int = typer.Option(5, "--epochs", "-e", min=1, help="Number of training epochs."),
    batch_size: int = typer.Option(64, "--batch-size", "-b", min=1, help="Mini-batch size."),
    learning_rate: float = typer.Option(
        1e-3,
        "--learning-rate",
        "--lr",
        help="Initial Adam learning rate.",
    ),
    fine_tune_epochs: int = typer.Option(
        0,
        "--fine-tune-epochs",
        help="EfficientNet only: extra epochs spent fine-tuning after unfreezing.",
        min=0,
    ),
    fine_tune_lr: float = typer.Option(
        1e-5,
        "--fine-tune-lr",
        help="Learning rate used during the fine-tuning stage.",
    ),
    seed: int = typer.Option(42, "--seed", help="Global random seed."),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Tiny smoke run on 1 000 images and 1 epoch (useful on weak CPUs).",
    ),
    experiment_name: str = typer.Option(
        "deepvision-cifar10",
        "--experiment",
        help="MLflow experiment name.",
    ),
) -> None:
    """Train a model end-to-end with MLflow tracking.

    Examples
    --------
    Tiny smoke training on a weak CPU::

        python -m deepvision train --model efficientnet --quick

    Full training on Colab::

        python -m deepvision train --model efficientnet --epochs 10 --fine-tune-epochs 5
    """
    # Heavy imports deferred so `--help` stays fast and importable in CI.
    from deepvision.training.train import TrainConfig, run_training

    config = TrainConfig(
        model_name=model,  # type: ignore[arg-type]
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        fine_tune_epochs=fine_tune_epochs,
        fine_tune_lr=fine_tune_lr,
        seed=seed,
        quick=quick,
        experiment_name=experiment_name,
    )
    result = run_training(config)

    typer.echo("")
    typer.echo("=" * 60)
    typer.echo(f"Run finished -- model={result.model_name}, run_id={result.run_id}")
    typer.echo(f"Test accuracy : {result.metrics['accuracy']:.4f}")
    typer.echo(f"Test loss     : {result.metrics['loss']:.4f}")
    typer.echo(f"Macro F1      : {result.metrics['macro_f1']:.4f}")
    typer.echo(f"Weighted F1   : {result.metrics['weighted_f1']:.4f}")
    typer.echo("=" * 60)
    typer.echo("Open the MLflow UI:  mlflow ui  -> http://localhost:5000")


@app.command()
def serve(
    host: str = typer.Option(
        "0.0.0.0",  # nosec B104 - bind-all is intentional inside containers; operators override via --host or DEEPVISION_API_HOST.
        "--host",
        help="Interface uvicorn binds to.",
    ),
    port: int = typer.Option(8000, "--port", "-p", min=1, max=65535, help="TCP port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code change (dev only)."),
    workers: int = typer.Option(
        1, "--workers", "-w", min=1, max=16, help="Number of uvicorn worker processes."
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        help="Uvicorn log level: critical / error / warning / info / debug / trace.",
    ),
) -> None:
    """Launch the FastAPI inference server.

    Examples
    --------
    Local development with hot reload::

        python -m deepvision serve --reload

    Production-style boot (matches the Docker image entrypoint)::

        python -m deepvision serve --host 0.0.0.0 --port 8000 --workers 2
    """
    # Heavy import deferred so `--help` stays cheap.
    import uvicorn

    typer.echo(f"deepvision API v{__version__} -- http://{host}:{port}/docs")
    uvicorn.run(
        "deepvision.serving.api:app",
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,
        log_level=log_level,
    )


@app.command()
def streamlit(
    host: str = typer.Option(
        "0.0.0.0",  # nosec B104 - bind-all is intentional inside containers; operators override via --host or DEEPVISION_STREAMLIT_HOST.
        "--host",
        "--server-address",
        help="Interface the Streamlit server binds to.",
    ),
    port: int = typer.Option(
        8501,
        "--port",
        "-p",
        min=1,
        max=65535,
        help="TCP port exposed by the Streamlit demo.",
    ),
    headless: bool = typer.Option(
        True,
        "--headless/--no-headless",
        help="Run without auto-opening a browser tab (recommended in containers).",
    ),
) -> None:
    """Launch the Streamlit demo UI.

    Examples
    --------
    Local development::

        python -m deepvision streamlit --no-headless

    Production-style boot inside the future Streamlit container::

        python -m deepvision streamlit --host 0.0.0.0 --port 8501
    """
    # Heavy imports deferred so ``--help`` stays cheap and the CLI remains
    # importable without Streamlit installed.
    import sys
    from pathlib import Path

    from streamlit.web import cli as stcli

    target = Path(__file__).resolve().parent / "streamlit_app.py"
    typer.echo(f"deepvision Streamlit v{__version__} -- http://{host}:{port}")
    sys.argv = [
        "streamlit",
        "run",
        str(target),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true" if headless else "false",
    ]
    sys.exit(stcli.main())


@app.command(name="drift-monitor")
def drift_monitor(
    port: int = typer.Option(
        9091,
        "--port",
        "-p",
        min=1,
        max=65535,
        help="TCP port exposed by the Prometheus exporter.",
    ),
    interval: float = typer.Option(
        60.0,
        "--interval",
        "-i",
        min=1.0,
        help="Seconds between two drift polling cycles.",
    ),
    baseline: str = typer.Option(
        "",
        "--baseline",
        help=(
            "Optional path to a baseline .npz file. When omitted (or the file is "
            "missing) the monitor computes a synthetic baseline at boot - useful "
            "for smoke tests but statistically meaningless."
        ),
    ),
    ood_threshold: float = typer.Option(
        -2.0,
        "--ood-threshold",
        help="Energy threshold above which a sample is flagged out-of-distribution.",
    ),
) -> None:
    """Launch the Phase 8 drift + OOD Prometheus exporter.

    Examples
    --------
    Local run with a real baseline computed at training time::

        python -m deepvision drift-monitor --baseline ./models/baseline.npz

    Smoke test (no baseline, synthetic data)::

        python -m deepvision drift-monitor --port 9091 --interval 30
    """
    # Heavy imports deferred so ``--help`` stays cheap.
    from pathlib import Path

    from deepvision.monitoring.server import run

    typer.echo(f"deepvision drift-monitor v{__version__} -- Prometheus exporter on :{port}")
    run(
        port=port,
        interval=interval,
        baseline_path=Path(baseline) if baseline.strip() else None,
        ood_threshold=ood_threshold,
    )


# ---------------------------------------------------------------------------
# `deepvision export ...`  -- Phase 10 sub-app
# ---------------------------------------------------------------------------
# Grouped under a sub-Typer so the three related commands (onnx / tflite /
# benchmark) share a clean namespace and a single ``--help`` listing. Each
# sub-command defers its heavy import (tensorflow, onnxruntime, ...) so
# ``deepvision export --help`` returns instantly.

export_app = typer.Typer(
    name="export",
    help="Export the trained model to ONNX/TFLite and benchmark latency.",
    no_args_is_help=True,
)
app.add_typer(export_app, name="export")


@export_app.command("onnx")
def export_onnx_cmd(
    model_path: str = typer.Option(
        ...,
        "--model-path",
        "-m",
        help="Path to the source .keras file or a SavedModel directory.",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Destination .onnx file.",
    ),
    opset: int = typer.Option(
        17,
        "--opset",
        min=1,
        max=22,
        help="ONNX opset version. Default 17.",
    ),
    no_validate: bool = typer.Option(
        False,
        "--no-validate",
        help="Skip the forward-pass equivalence check after export.",
    ),
) -> None:
    """Convert a Keras model to ONNX.

    Examples
    --------
    Export the trained EfficientNetB0::

        python -m deepvision export onnx \\
            --model-path models/efficientnet_best.keras \\
            --output models/exports/efficientnet.onnx
    """
    # Heavy imports deferred so ``--help`` stays cheap.
    from pathlib import Path

    import tensorflow as tf

    from deepvision.export.onnx import export_to_onnx

    typer.echo(f"Loading Keras model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    out_path = export_to_onnx(
        model,
        Path(output),
        opset=opset,
        validate=not no_validate,
    )
    typer.echo(f"ONNX export written to {out_path}")


@export_app.command("tflite")
def export_tflite_cmd(
    model_path: str = typer.Option(
        ...,
        "--model-path",
        "-m",
        help="Path to the source .keras file or a SavedModel directory.",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Destination .tflite file.",
    ),
    quantization: str = typer.Option(
        "int8",
        "--quantization",
        "-q",
        help="Quantization mode: dynamic, int8, int8_strict, or fp16.",
    ),
    n_samples: int = typer.Option(
        200,
        "--n-samples",
        min=1,
        help="Representative dataset size for INT8 calibration.",
    ),
) -> None:
    """Convert a Keras model to TFLite with optional quantization.

    For ``int8`` / ``int8_strict`` modes the command loads CIFAR-10
    train images automatically (via :func:`deepvision.data.loader.load_cifar10`)
    and uses ``--n-samples`` of them as the representative dataset.

    Examples
    --------
    Full INT8 export with 200 calibration images::

        python -m deepvision export tflite \\
            --model-path models/efficientnet_best.keras \\
            --output models/exports/efficientnet_int8.tflite \\
            --quantization int8

    Quick dynamic-range export (no calibration needed)::

        python -m deepvision export tflite \\
            --model-path models/efficientnet_best.keras \\
            --output models/exports/efficientnet_dyn.tflite \\
            --quantization dynamic
    """
    from pathlib import Path

    import tensorflow as tf

    from deepvision.export.tflite import QuantizationMode, export_to_tflite

    typer.echo(f"Loading Keras model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Validate the mode early so we fail before the slow CIFAR-10 download.
    mode = QuantizationMode(quantization)

    representative_data = None
    if mode in (QuantizationMode.INT8, QuantizationMode.INT8_STRICT):
        from deepvision.data.loader import load_cifar10

        typer.echo(f"Loading CIFAR-10 for the representative dataset (n_samples={n_samples})")
        split = load_cifar10()
        representative_data = split.x_train

    out_path = export_to_tflite(
        model,
        Path(output),
        quantization=mode,
        representative_data=representative_data,
        n_samples=n_samples,
    )
    typer.echo(f"TFLite export written to {out_path}")


@export_app.command("benchmark")
def benchmark_cmd(
    keras_path: str = typer.Option(
        "",
        "--keras-path",
        help="Path to a .keras file to benchmark with the Keras runtime.",
    ),
    savedmodel_dir: str = typer.Option(
        "",
        "--savedmodel-dir",
        help="Path to a SavedModel directory to benchmark with the TF runtime.",
    ),
    onnx_path: str = typer.Option(
        "",
        "--onnx-path",
        help="Path to a .onnx file to benchmark with ONNX Runtime (CPU).",
    ),
    tflite_path: str = typer.Option(
        "",
        "--tflite-path",
        help="Path to a .tflite file to benchmark with the TFLite interpreter.",
    ),
    n_warmup: int = typer.Option(
        100,
        "--n-warmup",
        min=0,
        help="Warmup iterations per (runner, batch_size) pair.",
    ),
    n_iter: int = typer.Option(
        1000,
        "--n-iter",
        min=1,
        help="Measured iterations per (runner, batch_size) pair.",
    ),
    batch_sizes: str = typer.Option(
        "1,8,32",
        "--batch-sizes",
        help="Comma-separated list of batch sizes to sweep.",
    ),
    output_csv: str = typer.Option(
        "",
        "--output-csv",
        help="Optional CSV path to persist the result table.",
    ),
) -> None:
    """Benchmark inference latency across the available runtimes.

    Pass any combination of ``--keras-path``, ``--savedmodel-dir``,
    ``--onnx-path`` and ``--tflite-path``; the command benchmarks every
    runtime that was given a path. At least one is required.

    Examples
    --------
    Four-way comparison::

        python -m deepvision export benchmark \\
            --keras-path models/efficientnet_best.keras \\
            --savedmodel-dir models/exports/efficientnet_savedmodel \\
            --onnx-path models/exports/efficientnet.onnx \\
            --tflite-path models/exports/efficientnet_int8.tflite \\
            --output-csv reports/phase10_benchmark.csv
    """
    from pathlib import Path

    from deepvision.export.benchmark import (
        KerasRunner,
        LatencyBenchmark,
        OnnxRuntimeRunner,
        Runner,
        TFLiteRunner,
        TFSavedModelRunner,
        to_dataframe,
    )

    bs_list = [int(x.strip()) for x in batch_sizes.split(",") if x.strip()]
    if not bs_list:
        raise typer.BadParameter("--batch-sizes must contain at least one positive integer.")

    runners: list[Runner] = []
    if keras_path:
        runners.append(KerasRunner(keras_path))
    if savedmodel_dir:
        runners.append(TFSavedModelRunner(savedmodel_dir))
    if onnx_path:
        runners.append(OnnxRuntimeRunner(onnx_path))
    if tflite_path:
        runners.append(TFLiteRunner(tflite_path))

    if not runners:
        raise typer.BadParameter(
            "At least one of --keras-path, --savedmodel-dir, --onnx-path, --tflite-path "
            "must be provided."
        )

    typer.echo(
        f"Running benchmark: {len(runners)} runtime(s) x {len(bs_list)} batch size(s) "
        f"x {n_iter} iterations (after {n_warmup} warmup)."
    )

    bench = LatencyBenchmark(
        runners=runners,
        n_warmup=n_warmup,
        n_iter=n_iter,
        batch_sizes=bs_list,
    )
    results = bench.run()
    df = to_dataframe(results)

    typer.echo("")
    typer.echo(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        typer.echo(f"\nResults written to {out}")


def main() -> None:  # pragma: no cover -- entrypoint
    """Entrypoint used by ``python -m deepvision`` and console scripts."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
