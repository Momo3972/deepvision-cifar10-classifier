"""
Reproducibility helpers.

Fixing seeds is a prerequisite for benchmarking and side-by-side model comparisons.
This module guarantees deterministic behavior across Python's ``random``, NumPy,
the ``PYTHONHASHSEED`` environment variable, and TensorFlow if it is importable.
"""

from __future__ import annotations

import importlib
import os
import random

from deepvision.constants import DEFAULT_SEED


def set_seed(seed: int = DEFAULT_SEED, *, deterministic_tf: bool = False) -> int:
    """Seed all random number generators that this project relies upon.

    Parameters
    ----------
    seed
        Integer seed to apply. Defaults to :data:`deepvision.constants.DEFAULT_SEED`.
    deterministic_tf
        If True and TensorFlow is installed, also enable
        ``tf.config.experimental.enable_op_determinism()`` (TF >= 2.10).
        This may slow training but guarantees byte-identical output across runs.

    Returns
    -------
    int
        The seed actually applied (useful when callers pass ``None`` in the future).

    Side effects
    ------------
    - Sets ``PYTHONHASHSEED`` (process-level).
    - Seeds ``random.seed`` and ``numpy.random.seed``.
    - Seeds ``tf.random.set_seed`` if TensorFlow is importable.

    Examples
    --------
    >>> set_seed(123)
    123
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy is a hard runtime dependency of the project but kept optional here
    # so this helper can be unit-tested in isolation.
    try:
        np = importlib.import_module("numpy")
        np.random.seed(seed)
    except ImportError:  # pragma: no cover — numpy missing only in tooling envs
        pass

    # TensorFlow is heavier; only seed it if it is available.
    try:
        tf = importlib.import_module("tensorflow")
        tf.random.set_seed(seed)
        if deterministic_tf:
            tf.config.experimental.enable_op_determinism()
    except ImportError:  # pragma: no cover
        pass

    return seed
