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

## [0.5.0] — Phase 4 — Evaluation enrichment (2026-05-02)

The project now ships a full evaluation toolbox aligned with modern Computer
Vision practice. **No fabricated metrics**: actual numbers will be measured on
Colab once the upstream CIFAR-10 host (cs.toronto.edu) is reachable again
(scheduled to resume on 2026-05-04).

### Added — `src/deepvision/evaluation/`
- **`calibration.py`** — Expected Calibration Error (ECE) over 15 confidence
  bins (Guo et al., 2017), `reliability_diagram_data` helper for plotting,
  `fit_temperature` (post-hoc temperature scaling via scipy `minimize_scalar`),
  `apply_temperature`. Numerically stable log-softmax internally.
- **`interpretability.py`** — Grad-CAM (`grad_cam`) implementation in pure
  TensorFlow/Keras with recursive sub-model traversal (`find_last_conv_layer`,
  `_find_layer`) so it works on the nested EfficientNetB0 backbone.
  Heatmap normalization, bilinear upsampling, and an `inferno`-style
  colormap (`overlay_heatmap_on_image`) without a matplotlib dependency.
- **`benchmark.py`** — `LatencyResult` dataclass (mean / p50 / p90 / p95 /
  p99 / min / max in milliseconds + throughput), `benchmark_callable`
  (warmup + measured iterations), `benchmark_keras_model` convenience
  wrapper using a deterministic synthetic input.
- **`robustness.py`** — CIFAR-10-C harness: `STANDARD_CORRUPTIONS` (15
  canonical names), `RobustnessReport` dataclass, `discover_corruptions`,
  `evaluate_corruption` (per-severity accuracy), `evaluate_robustness`
  (clean accuracy + per-corruption + mean Corruption Error). The download
  helper raises `NotImplementedError` deliberately — users opt in by
  fetching CIFAR-10-C manually from Zenodo.

### Added — Tests (3 new modules, ~25 unit tests)
- `test_calibration.py` — perfectly-calibrated synthetic dataset has ECE ≈ 0,
  overconfident classifier has ECE ≈ 0.5, temperature scaling preserves
  argmax but flattens the distribution, `fit_temperature` recovers T ≈ 1
  on calibrated logits.
- `test_interpretability.py` — Grad-CAM returns a heatmap of the correct
  shape for an EfficientNet without ImageNet weights, normalized to [0, 1].
- `test_benchmark.py` — `benchmark_callable` returns sane statistics on a
  noop function (warmup honored, percentiles ordered, throughput positive).

### Tech-debt acknowledged (deferred)
- `evaluate_robustness` is **not unit-tested end-to-end** because CIFAR-10-C
  is too large to bundle (12 GB) and downloading from Zenodo requires
  cross-platform handling that belongs in a future Phase 10 enhancement.
  The harness is documented and shape-checked.
- All metrics will be re-measured on Colab once CIFAR-10 is back online
  (cs.toronto.edu currently down for scheduled maintenance until 2026-05-04).

### Changed
- `requirements.txt` adds `scipy>=1.13,<2.0` (used by `fit_temperature`).
  scipy was already a transitive dep of scikit-learn but is now declared
  explicitly to avoid surprises.
- Version bumped: `0.4.0` → `0.5.0`.

---

## [0.4.0] — Phase 3 — Training pipeline & MLflow (2026-05-01)

The project now has a reproducible, fully tracked training pipeline. Running
`python -m deepvision train --model efficientnet` produces an MLflow run with
params, per-epoch metrics, dataset hash, environment provenance,
classification report, confusion matrix and the serialized model.

### Added — `src/deepvision/training/`
- **`train.py`**: `TrainConfig` + `TrainResult` dataclasses,
  `run_training()` end-to-end pipeline. Two-stage strategy for EfficientNet
  (feature extraction + optional fine-tuning) wired in. `--quick` mode for
  weak hardware (1 000 images, 1 epoch, no ImageNet weights).
- **`callbacks.py`**: `build_default_callbacks()` factory exposing
  EarlyStopping + ReduceLROnPlateau with overridable hyperparameters.
- **`mlflow_utils.py`**: `setup_mlflow`, `start_run` (context manager),
  `log_dataset_metadata`, `log_environment_metadata`,
  `log_classification_artifacts`. Captures git SHA, Python and TF versions.

### Added — `src/deepvision/evaluation/`
- **`metrics.py`**: `evaluate_model()` returns a JSON-friendly dict
  (`accuracy`, `loss`, `per_class_f1`, `classification_report`,
  `confusion_matrix`, `macro_f1`, `weighted_f1`).

### Added — CLI
- `python -m deepvision train --model {mlp,cnn,efficientnet} [--quick]`
  with options for epochs, batch size, learning rate, fine-tune-epochs,
  fine-tune-lr, seed and experiment name.

### Added — Tests (4 modules, 14 new tests)
- `test_callbacks.py`: callback factory contract.
- `test_metrics.py`: cross-entropy edge cases and `evaluate_model` smoke.
- `test_mlflow_utils.py`: tracking URI handling, environment helpers.
- `test_train.py`: config and result dataclass invariants.

### Changed
- Runtime dependency: `mlflow>=2.16,<3.0` (range pin — locked in Phase 5).
- Version bumped: `0.3.0` -> `0.4.0`.

### Methodological note
The corrected EfficientNet accuracy will be re-measured on the clean,
leakage-free 12 000-image test set during the first full training run on
Colab. Results will be logged in MLflow and added to this changelog
when measured.

---

## [0.3.0] — Phase 2 — Data & Models (2026-05-01)

The core ML logic of the original notebook has been ported into a modular,
tested and documented Python package. Critical bugs from the audit are fixed.

### Added — `src/deepvision/data/`
- **`loader.py`**: canonical `load_cifar10()` returning a frozen
  `CifarSplit` dataclass. Performs a stratified 80/20 split on the merged
  train+test arrays so every model sees the **same** partition. Includes
  `compute_dataset_hash()` (SHA-256) for MLflow traceability.
- **`preprocessing.py`**: `normalize_to_unit`, `denormalize_to_uint8`,
  `one_hot_encode`, `validate_image_array`. Strict input validation.
- **`augmentation.py`**: `AugmentationConfig` dataclass and
  `build_augmentation_pipeline()` returning a Keras `Sequential`
  (RandomFlip + RandomRotation + RandomZoom + RandomContrast).

### Added — `src/deepvision/models/`
- **`mlp.py`**: `build_mlp()` — Flatten + 2x(Dense + BN + Dropout) + Output.
- **`cnn.py`**: `build_cnn()` — three VGG-style convolutional blocks with
  BatchNorm and progressive Dropout (0.2 → 0.3 → 0.4 → 0.5).
- **`efficientnet.py`**: `build_efficientnet()` and `unfreeze_top_layers()`.
  Two-stage transfer learning (feature extraction → fine-tuning), with
  `weights=None` mode for fast unit tests.
- **`registry.py`**: `MODEL_REGISTRY`, `get_model()`, `available_models()` —
  factory used by the upcoming training CLI.

### Added — Tests (4 new test modules, 28 new tests)
- **`tests/unit/test_loader.py`**: dataset hash determinism, stratified split
  reproducibility, balance assertion, summary contract. Real CIFAR-10
  download wrapped in `@pytest.mark.integration` (skipped by default).
- **`tests/unit/test_preprocessing.py`**: 9 tests covering normalization,
  one-hot encoding, validation, parameterized failure cases.
- **`tests/unit/test_augmentation.py`**: pipeline shape preservation, custom
  configurations, disabled mode.
- **`tests/unit/test_models.py`**: registry contracts, MLP/CNN I/O shapes,
  EfficientNet structural checks (resize layer, layer counts), unfreeze
  helper invariants.

### Fixed
- **Data leakage between models** (audit Bug B6): the original notebook
  trained MLP / CNN on a stratified 80/20 split (48k/12k) but EfficientNet
  on the native Keras 50k/10k split, and then evaluated all three on the
  12k subset — which contained images EfficientNet had seen during training.
  Phase 2 enforces a single `load_cifar10()` entry-point used by every
  model. The 93 % test accuracy will be re-measured on a clean test set in
  Phase 3 (the score may drop by 1–3 percentage points; this will be
  documented as a deliberate methodological correction).
- **Ambiguous Colab-only model save path** (audit Bug B7): no hardcoded
  paths anymore — all I/O goes through `Settings.models_dir`.
- **`get_callbacks` and `plot_history` duplication** in the notebook is
  superseded by the new module structure (training pipeline arrives in
  Phase 3).

### Changed
- **`pytest`** now skips `@pytest.mark.integration` by default
  (`addopts: -m "not integration"`). Run them explicitly with
  `pytest -m integration`.
- **Version bumped** from `0.2.0` to `0.3.0`.

### Tech debt acknowledged (deferred)
- The legacy `app.py` Streamlit demo at the root still uses local
  preprocessing logic. It will be rewritten on top of the new `data/` and
  `models/` modules in Phase 6.
- The original `Notebooks/Projet_Vision_CIFAR10.ipynb` is unchanged; it
  will be split into 5 thinner notebooks importing the package in Phase 4.

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
