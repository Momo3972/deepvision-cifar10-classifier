"""
Command-line interface for the deepvision package.

Phase 1 ships a minimal CLI built with `Typer <https://typer.tiangolo.com/>`__.
Subcommands ``train``, ``evaluate``, ``serve`` etc. will be added in subsequent
phases.

Usage
-----
.. code-block:: bash

    python -m deepvision --help
    python -m deepvision version
    python -m deepvision info
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


def main() -> None:  # pragma: no cover — entrypoint
    """Entrypoint used by ``python -m deepvision`` and console scripts."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
