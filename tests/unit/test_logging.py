"""
Tests for :mod:`deepvision.utils.logging`.
"""

from __future__ import annotations

import logging

from deepvision.utils.logging import get_logger, setup_logging


def test_get_logger_returns_namespaced_logger() -> None:
    """A logger fetched with a bare name is auto-prefixed with ``deepvision.``."""
    log = get_logger("my_module")
    assert log.name == "deepvision.my_module"


def test_get_logger_respects_already_prefixed_name() -> None:
    """If the name already starts with ``deepvision``, no double prefix is applied."""
    log = get_logger("deepvision.training.train")
    assert log.name == "deepvision.training.train"


def test_setup_logging_is_idempotent() -> None:
    """Calling ``setup_logging`` multiple times must not stack handlers."""
    setup_logging("INFO")
    handlers_count = len(logging.getLogger("deepvision").handlers)
    setup_logging("INFO")
    setup_logging("DEBUG")
    assert len(logging.getLogger("deepvision").handlers) == handlers_count
