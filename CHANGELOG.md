# Changelog

All notable changes to **deepvision-cifar10-classifier** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Notes

Industrial refactor in progress. The dépôt is being transformed from an academic
deliverable into a production-grade Computer Vision pipeline aligned with industry
standards (MLflow, FastAPI, Docker, CI/CD, monitoring).

The roadmap is detailed in `Audit_DeepVision_CIFAR10.docx` (13 phases).

---

## [0.2.0] — Phase 1 — Packaging Python (2026-05-01)

The project is now a proper Python package, with quality tooling wired in.

### Added
- **`src/deepvision/` package** with declared subpackages:
  - `data/`, `models/`, `training/`, `evaluation/`, `serving/`, `monitoring/`,
    `utils/` (each with a docstring announcing the phase that fills it).
  - `__init__.py` exposing `__version__`, `__author__`, `__license__`.
  - `__main__.py` with a Typer CLI (`python -m deepvision version|info`).
  - `config.py` with a `pydantic-settings` `Settings` class (env-driven).
  - `constants.py` with `CLASS_NAMES_FR`, `CLASS_NAMES_EN`, `IMG_SIZE_*`,
    `NUM_CLASSES`, `DEFAULT_SEED`, etc.
  - `utils/logging.py` with idempotent `setup_logging` and `get_logger`.
  - `utils/seed.py` with `set_seed` covering Python random, NumPy, and TF.
  - `py.typed` marker (PEP 561) for downstream type checkers.
- **`tests/` directory** with `conftest.py` and unit tests covering the package's
  smoke imports, constants, seeding, and logging.
- **`requirements-dev.txt`** with ruff, mypy, pytest, pytest-cov, pytest-xdist,
  Hypothesis, pre-commit, bandit, pip-audit.
- **`.pre-commit-config.yaml`** with ruff (lint + format), gitleaks (secret
  detection), and standard hygiene hooks (large files, merge conflicts,
  YAML/JSON/TOML validation, EOL fix, trailing whitespace).
- **`Makefile`** with targets: `install`, `install-dev`, `lint`, `format`,
  `typecheck`, `security`, `test`, `test-fast`, `check`, `diagnose`, `clean`.
- **`tasks.ps1`** PowerShell equivalent for Windows-native users.
- **Console script** `deepvision = deepvision.__main__:main`
  (so `deepvision --help` works once the package is installed in editable mode).

### Changed
- **`pyproject.toml`** now declares the package layout (`src/`),
  `[project.scripts]`, the dev-deps optional group, and full configurations
  for ruff, mypy, pytest, coverage, and bandit.
- **`requirements.txt`** ships two new runtime deps:
  `typer==0.21.0` and `pydantic-settings==2.13.1`, both required by the package.
- **Version bumped** from `0.1.0` to `0.2.0`.

---

## [0.1.0] — Phase 0 — Baseline (2026-04-26)

First commit of the industrial refactor.
This release introduces the foundations on which all subsequent phases will build.

### Added
- `LICENSE` (MIT) — formal open-source licensing.
- `pyproject.toml` (PEP 621) — unified Python project configuration with skeleton
  for ruff, pytest, and coverage (will be activated in Phase 1).
- `CHANGELOG.md` — this file, following Keep a Changelog conventions.
- `scripts/check_machine.py` — local hardware diagnostic utility (CPU, RAM, disk,
  GPU, ML libraries, inference latency benchmark).

### Changed
- `requirements.txt` — pinned versions for reproducibility
  (TensorFlow 2.21.0, Keras 3.14.0, NumPy 2.4.4, scikit-learn 1.8.0,
  Pillow 12.2.0, Streamlit 1.57.0, psutil 7.2.2). Was unpinned previously.
- `.gitignore` — replaced the minimal version with an exhaustive ignore list
  covering Python, Jupyter, ML artifacts (.h5, .keras, .onnx, .tflite),
  MLflow runs, quality tool caches (ruff, mypy, pytest), IDE files, secrets,
  and OS metadata.

### Removed
- `models/best_model_efficientnet_aug.h5` — removed from Git tracking.
  The 33 MB binary was committed to the repository, which is an anti-pattern.
  It is preserved on the `archive/legacy-h5` branch for historical access and
  remains usable on disk for Colab-based training.
  In Phase 2, models will be versioned via MLflow Model Registry.

### Infrastructure
- New branching strategy: one branch per phase
  (`feature/phase-N-<topic>`) merged via Pull Request into `main`.
- New archive branch: `archive/legacy-h5` containing the last commit before
  removal of the binary model.

---

## Project history (pre-refactor)

### [0.0.x] — Academic version (December 2025)
- Initial deliverable for the EBDE Master's degree (UTT).
- Single Jupyter notebook covering MLP, CNN custom, and EfficientNetB0 transfer
  learning on CIFAR-10. Final test accuracy: ~93% (note: data leakage between
  the 80/20 stratified split used for MLP/CNN and the native CIFAR-10 split used
  for EfficientNet — to be corrected in Phase 3).
- Streamlit demo (`app.py`) with live image classification.

[Unreleased]: https://github.com/Momo3972/deepvision-cifar10-classifier/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Momo3972/deepvision-cifar10-classifier/releases/tag/v0.1.0
