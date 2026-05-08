"""
Command-line interface for the deepvision package.

Subcommands
-----------
- ``version``:   print the package version and exit.
- ``info``:      print package metadata and resolved configuration paths.
- ``train``:     train one of the registered models with full MLflow tracking.
- ``serve``:     launch the FastAPI inference service via uvicorn.
- ``streamlit``: launch the Streamlit demo UI (Phase 6 refonte).

Usage
-----
.. code-block:: bash

    python -m deepvision --help
    python -m deepvision version
    python -m deepvision info
    python -m deepvision train --model efficientnet --epochs 1 --quick
    python -m deepvision serve --host 0.0.0.0 --port 8000
    python -m deepvision streamlit --port 8501
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
        "0.0.0.0",
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
        "0.0.0.0",
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


def main() -> None:  # pragma: no cover -- entrypoint
    """Entrypoint used by ``python -m deepvision`` and console scripts."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
