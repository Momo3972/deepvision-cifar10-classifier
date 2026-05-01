"""
Logging configuration for the deepvision package.

Phase 1 uses the standard ``logging`` module with a sensible default handler.
A migration to ``structlog`` (JSON output for production) is planned for Phase 5
when the FastAPI service is introduced.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

_DEFAULT_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | int = "INFO", *, fmt: str | None = None) -> None:
    """Configure the root logger of the deepvision package.

    Parameters
    ----------
    level
        Log level for the deepvision logger. Accepts strings (``"INFO"``)
        or integers (``logging.INFO``).
    fmt
        Optional log format string. Defaults to a human-readable layout.

    Notes
    -----
    Idempotent: calling this function multiple times only adds handlers
    if none exist on the deepvision logger.
    """
    logger = logging.getLogger("deepvision")
    if logger.handlers:
        return  # Already configured.

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt or _DEFAULT_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logs via root handler.


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the deepvision namespace.

    Parameters
    ----------
    name
        Name suffix appended to the ``deepvision`` namespace.
        Pass ``__name__`` from any module inside ``deepvision``.

    Returns
    -------
    logging.Logger
        A configured logger suitable for module-level use.

    Examples
    --------
    >>> log = get_logger(__name__)
    >>> log.info("training started")  # doctest: +SKIP
    """
    if not name.startswith("deepvision"):
        name = f"deepvision.{name}"
    return logging.getLogger(name)
