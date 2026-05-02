"""
Command-line interface for the deepvision package.

Subcommands
-----------
- ``version``: print the package version and exit.
- ``info``:    print package metadata and resolved configuration paths.
- ``train``:   train one of the registered models with full MLflow tracking.

Usage
-----
.. code-block:: bash

    python -m deepvision --help
    python -m deepvision version
    python -m deepvision info
    python -m deepvision train --model efficientnet --epochs 1 --quick
    python -m deepvision train --model efficientnet --epochs 10 --fine-tune-epochs 5
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
        help="Tiny smoke run on 1 000 images and 1 epoch — useful on weak CPUs.",
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
    typer.echo(f"Run finished — model={result.model_name}, run_id={result.run_id}")
    typer.echo(f"Test accuracy : {result.metrics['accuracy']:.4f}")
    typer.echo(f"Test loss     : {result.metrics['loss']:.4f}")
    typer.echo(f"Macro F1      : {result.metrics['macro_f1']:.4f}")
    typer.echo(f"Weighted F1   : {result.metrics['weighted_f1']:.4f}")
    typer.echo("=" * 60)
    typer.echo("Open the MLflow UI:  mlflow ui  → http://localhost:5000")


def main() -> None:  # pragma: no cover — entrypoint
    """Entrypoint used by ``python -m deepvision`` and console scripts."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
