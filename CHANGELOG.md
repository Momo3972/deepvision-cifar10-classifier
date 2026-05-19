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

## [0.12.0] — Phase 11 — Documentation (mkdocs-material, bilingue EN/FR) (2026-05-17)

Phase 11 of the industrial refactor roadmap (audit section 7.6). The
project gets a real documentation site, published on GitHub Pages from
`main`, served bilingual (English + French) with a language switcher
in the header. The `docs.yml` workflow that has been in graceful-skip
mode since Phase 9 now detects `mkdocs.yml` and builds + deploys on
every push to `main`.

### Added — `mkdocs.yml` and tooling
- **`mkdocs.yml`** -- single source of truth for the site: Material
  theme with light/dark palette toggle, JetBrains Mono + Roboto fonts,
  navigation tabs / instant-back-to-top / search suggest, mkdocstrings
  Python handler configured for NumPy-style docstrings, the
  `pymdownx.*` family of extensions (admonitions, tabbed content,
  fenced code annotations, Mermaid diagrams, footnotes).
- **`mkdocs-static-i18n` plugin** -- bilingual EN/FR with the
  `suffix` strategy: `page.md` for English (default), `page.fr.md`
  for the French variant. Navigation labels translate via the
  `nav_translations` map; the search index is reconfigured to handle
  both locales.
- **`mkdocstrings[python]` plugin** -- auto-generated API reference
  from the 300+ NumPy-style docstrings already present in
  `src/deepvision/`. The reference page lists 14 modules grouped by
  feature area.

### Added — `docs/` content (English, default)
- **`index.md`** -- homepage with hero, feature grid, project status
  matrix tracking the 12 phases.
- **`getting-started.md`** -- full installation walkthrough,
  configuration via env vars, training / serving / Docker happy paths.
- **`architecture.md`** -- system diagram (Mermaid), source-tree map,
  runtime topology of the six-service Docker stack, data flow
  narratives.
- **`tutorials/{index,training,serving,monitoring,export}.md`** --
  four task-oriented walkthroughs with snippets, common pitfalls
  callouts, and cross-links into the API reference.
- **`reference/index.md`** -- auto-generated API documentation for
  14 modules across config / data / models / training / evaluation /
  serving / monitoring / export.
- **`model-card.md`** -- Hugging Face-template model card with all
  prescribed sections: Model Details, Intended Use, Out-of-Scope
  Uses, Bias / Risks / Limitations, How to Get Started, Training
  Details (data + procedure + speeds-sizes-times), Evaluation
  Metrics + Per-Class Confusions, Environmental Impact, Technical
  Specifications, Citation.
- **`contributing.md`** -- branching model, local quality gates,
  Conventional Commits convention, Dependabot triage rules, code
  of conduct.

### Added — `docs/` content (French translations)
- **Full translations** of the homepage (`index.fr.md`), the getting
  started page (`getting-started.fr.md`) and the model card
  (`model-card.fr.md`) -- the three most visible pages.
- **Bilingual placeholders** for the architecture, contributing,
  reference and four tutorial pages: short French intro + summary
  table + "see English version" call-to-action button. Keeps the
  language switcher functional everywhere without doubling the
  maintenance load on detail-heavy pages.
- **`reference/index.fr.md`** explicitly redirects to the English
  API reference to avoid duplicating mkdocstrings output (which
  would clash with the global autorefs index).

### Added — `README.md` updates
- **Language switcher block** at the top right linking to the EN and
  FR landings of the published site.
- **`Docs` badge** wired to the `docs.yml` workflow.
- **"Full documentation -- bilingual EN/FR" block** with a call-out
  pointing to the published GitHub Pages site.
- **Bilingual `🇫🇷 Présentation` / `🇬🇧 Overview`** intro paragraphs.
- Existing French README content preserved -- the site is now the
  canonical detailed documentation, README stays as a high-level
  overview.

### Added — Tests (`tests/unit/test_documentation.py`)
- **26 new structural tests** validating: `mkdocs.yml` parses to a
  valid Material config; theme palette exposes both `default` and
  `slate` schemes; `navigation.instant` is *not* enabled (would
  break the i18n switcher); required plugins (`search`, `i18n`,
  `mkdocstrings`) are declared; the `i18n` plugin declares EN and
  FR with EN as default; mkdocstrings uses NumPy docstring style;
  every page in the `nav` exists on disk; the 11 required EN pages
  exist; the 11 required FR variants exist and are not empty; the
  model card has every Hugging Face top-level section; the README
  links to the docs site, has the language switcher, and exposes
  the docs badge.

### Added — Dependencies (`requirements-dev.txt`)
- `mkdocs-material>=9.5,<10.0`
- `mkdocs-static-i18n>=1.2,<2.0`
- `mkdocstrings[python]>=0.26,<1.0`
- `mkdocs-gen-files>=0.5,<1.0`
- `mkdocs-literate-nav>=0.6,<1.0`
- `mkdocs-section-index>=0.3,<1.0`
- `pymdown-extensions>=10.7,<11.0`

### Fixed
- **Dead cross-reference** `validate_image_bytes` -- the actual
  function in `deepvision.serving.preprocess` is `validate_payload_size`.
  Two markdown pages (`architecture.md`, `tutorials/serving.md`)
  fixed.

### Changed
- `pyproject.toml`: `version` bumped `0.11.0` -> `0.12.0`.
- `src/deepvision/__init__.py`: `__version__` bumped `0.11.0` -> `0.12.0`.

### Notes
- Build is **`--strict`-clean**: `mkdocs build --strict` succeeds in
  ~5 seconds on the standard runner with zero warnings (only the
  Material team's marketing notice about the upcoming MkDocs 2.0
  re-architecture, which is informational, not a warning).
- Some `INFO`-level messages surface on relative links from FR pages
  pointing to the EN site (`../reference/`, `../training/`, ...).
  These are intentional: from the published FR site, those links
  resolve to the EN sibling page so users keep their language choice
  when crossing into pages that have no full French translation yet.

### Known issues
- **Two new `mlflow 2.22.5` CVEs** surfaced after Phase 9 closed:
  `CVE-2026-2393` (fix in mlflow 3.9.0) and `CVE-2026-2614` (fix in
  mlflow 3.10.0). Both are scoped to the MLflow Server UI / REST
  surface, which our deployment exposes only on the internal
  `deepvision` Docker network -- it never reaches untrusted callers,
  so the practical risk is nil. Added to `pip-audit --ignore-vuln`
  in `.github/workflows/security.yml`. The mlflow 3.x bump that
  resolves them all is queued as a follow-up PR (it is a major
  version bump and requires migrating away from the deprecated
  Models API).

---

## [0.11.0] — Phase 10 — Export ONNX/TFLite + benchmark latence (2026-05-17)

Cross-runtime portability layer prescribed by the audit (sections 7.2 and
9, Phase 10). The trained EfficientNetB0 can now be served through any
of four runtimes — Keras, TensorFlow SavedModel, ONNX Runtime (CPU),
TFLite (Full INT8) — and the new latency benchmark quantifies the
trade-off on the operator's hardware before they pick a deployment
target.

### Added — `src/deepvision/export/`
- **`onnx.py`** — `export_to_onnx(model, output_path, *, opset=17,
  validate=True)`. Conversion runs through `tf2onnx.convert.from_keras`
  with an explicit `input_signature` built from `model.input_shape`;
  passing the signature is what makes the conversion robust on the
  Keras 3 / TF 2.21 stack (the implicit graph-walking path occasionally
  fails on EfficientNet's preprocessing BatchNormalization layers).
  Every export ends with a forward-pass equivalence check that asserts
  `max_abs_diff < 1e-4` between the Keras and ONNX Runtime outputs on
  random inputs — a silently broken conversion can never escape CI.
- **`tflite.py`** — `export_to_tflite(model, output_path, *,
  quantization, representative_data, n_samples)` with four modes
  exposed via the `QuantizationMode` `StrEnum`:
  - `DYNAMIC` — weights INT8, activations FP32, no calibration needed.
  - `INT8` (**default**) — weights + activations INT8, I/O FP32
    (drop-in replacement for the FP32 model). Calibrates on 200
    CIFAR-10 train images by default.
  - `INT8_STRICT` — INT8 everywhere including I/O tensors (pure edge
    deployment; caller handles quantization/dequantization).
  - `FP16` — weights FP16 for GPU edge runtimes.
- **`benchmark.py`** — `LatencyBenchmark` orchestrator running a
  Cartesian product of `(runner, batch_size)` measurements. Four
  pluggable runners satisfy the `Runner` protocol: `KerasRunner`,
  `TFSavedModelRunner`, `OnnxRuntimeRunner` (single-threaded for
  reproducibility), `TFLiteRunner` (auto-resizes tensors on shape
  change). Each measurement reports p50 / p90 / p95 / p99 / mean / std
  / throughput; the `to_dataframe()` helper produces a tidy pandas
  table suitable for reports and MLflow logging.

### Added — CLI (`python -m deepvision export ...`)
- **`onnx`** — `--model-path`, `--output`, `--opset`, `--no-validate`.
- **`tflite`** — `--model-path`, `--output`, `--quantization`,
  `--n-samples`. For `int8` / `int8_strict` modes the command loads
  CIFAR-10 automatically as the calibration source.
- **`benchmark`** — accepts any combination of `--keras-path`,
  `--savedmodel-dir`, `--onnx-path`, `--tflite-path`, plus
  `--n-warmup`, `--n-iter`, `--batch-sizes`, `--output-csv`.

### Added — Standalone scripts (`scripts/`)
- **`export_onnx.py`**, **`quantize_tflite.py`**,
  **`benchmark_latency.py`** — thin argparse wrappers around the
  modules so they can be wired into shell pipelines or CI matrix jobs
  without pulling Typer.

### Added — Dependencies
- `tf2onnx>=1.16,<2.0`, `onnx>=1.16,<3.0`, `onnxruntime>=1.18,<3.0` in
  `requirements.txt`. Upper bounds left loose so Dependabot can ship
  minor bumps without breaking the export pipeline.

### Added — Tests (`tests/unit/test_export_{onnx,tflite,benchmark}.py`)
- **50 new tests**, 97.58 % branch coverage on `src/deepvision/export/`
  (target was ≥ 90 %). Highlights:
  - `tflite.py`: 100 % coverage. All four quantization modes exercised
    on a tiny CNN; INT8 round-trip checks argmax agreement ≥ 80 % vs
    Keras; `INT8_STRICT` verifies the I/O tensors are emitted as
    `int8`.
  - `onnx.py`: 93.55 %. Real Keras → ONNX → ONNX Runtime round trip on
    a tiny MLP with numerical equivalence asserted; opset persistence
    verified by re-parsing the ONNX proto; shape-mismatch and tight
    tolerance both raise as designed.
  - `benchmark.py`: 97.73 %. Every runner exercised end-to-end against
    a real exported artefact; `TFSavedModelRunner` validates the
    signature-key priority (`"serve"` from Keras 3 wins over the
    legacy `"serving_default"`, with a clean fallback to the first
    available key); `LatencyBenchmark` input validation tested across
    all four argument boundaries.

### Fixed
- **`TFSavedModelRunner` no longer hard-codes `"serving_default"`** —
  Keras 3's `model.export()` writes the serving signature under the
  `"serve"` key, which the original implementation didn't know about.
  The runner now probes a priority list (`"serve"`,
  `"serving_default"`) and falls back to the first available
  signature, so it works on both legacy `tf.saved_model.save()`
  outputs and Keras 3 `model.export()` outputs.

### Changed
- `pyproject.toml`: `version` bumped `0.10.0` → `0.11.0`.
- `src/deepvision/__init__.py`: `__version__` bumped `0.10.0` → `0.11.0`.

### Notes — Deferred warnings
- The TFLite path emits a `UserWarning` from `tf.lite.Interpreter`
  about deprecation in TF 2.20 (in favour of the upcoming `ai_edge_litert`
  package). Migration is queued for the post-TF-2.20 dependabot bump;
  the current API stays functional and produces bit-identical INT8
  models.
- A `DeprecationWarning` from Keras 3's `__array__` adapter surfaces
  when `KerasRunner.predict()` converts the tensor back to NumPy. It
  originates inside Keras itself (no caller-side fix possible) and
  goes away with the NumPy 2.x compatibility shim shipped in Keras
  3.15+.

---

## [0.10.0] — Phase 9 — CI/CD GitHub Actions (2026-05-14)

End of the manual-PR era: every push to `main` and every pull request now
triggers the four GitHub Actions workflows prescribed by the audit (section
7.5). The previous "Checks 0" badge on every PR is replaced by green check
marks from `ci`, `security` and (release-only) `docker` workflows.

### Added — `.github/workflows/`
- **`ci.yml`** — three parallel jobs on every push/PR to `main`:
  - `lint` — `ruff check` + `ruff format --check` (Python 3.12, ~30 s).
  - `typecheck` — `mypy src/deepvision` on a Python 3.11 + 3.12 matrix.
  - `test` — `pytest -q` with coverage on the same matrix. Coverage XML
    from the 3.12 entry is uploaded to Codecov for the README badge.
  Uses `cancel-in-progress` concurrency so force-pushes don't burn stale
  CI minutes.
- **`security.yml`** — five scanners on every push/PR + a weekly Monday
  06:00 UTC cron so newly disclosed CVEs are caught even when no code
  changes:
  - `bandit` (static analysis, MEDIUM+ severity), `pip-audit` (CVE on
    resolved deps), `gitleaks` (secrets in git history), `codeql`
    (GitHub native SAST with `security-and-quality` queries).
  - `trivy` (container vuln scan) **only on tag `v*`** — building +
    scanning the three ~3 GB images is too expensive for every push;
    findings are uploaded as SARIF to the GitHub Security tab.
- **`docker.yml`** — multi-stage build + push to GHCR
  (`ghcr.io/momo3972/deepvision-{api,streamlit,training}`) on push to
  `main` and on tag `v*`. PRs run a build-only validation pass.
  Tags emitted: `latest` (default branch), `<short-sha>`, semver
  `{major.minor}` + `{version}`. Auth uses the native `GITHUB_TOKEN`
  (no manual secret). Buildx cache via `type=gha` for speed.
- **`docs.yml`** — `mkdocs build --strict` + deploy to GitHub Pages.
  Phase 11 will populate `docs/` and `mkdocs.yml`; until then the
  `check-docs-source` gate exits gracefully so the workflow stays green
  on `main`.

### Added — `.github/`
- **`dependabot.yml`** — three ecosystems tracked weekly on Monday at
  06:00 UTC Europe/Paris: pip (with curated `ml-stack`, `serving-stack`
  and `dev-tooling` groups so dependent bumps land in a single PR),
  docker (the three Dockerfile bases) and github-actions (workflow SHA
  pins). Each PR gets the `dependencies` label and a conventional
  `chore(...)` commit prefix.
- **`PULL_REQUEST_TEMPLATE.md`** — summary / related / type-of-change
  checklist, quality gates checklist (ruff + mypy + pytest +
  CHANGELOG), audit-alignment checklist for phase-delivery PRs, smoke
  test field.
- **`ISSUE_TEMPLATE/`** — four structured forms (bug_report,
  feature_request, model_issue, drift_report — the last one wires up
  with the Phase 8 monitoring), `config.yml` disables blank issues and
  surfaces the audit + CHANGELOG as contact links.

### Added — Documentation & tests
- **`docs/contributing/branch-protection.md`** — operator guide for
  configuring `main` branch protection rules in GitHub Settings
  (required reviews, required status checks, linear history,
  conversation resolution).
- **`tests/unit/test_github_workflows.py`** — parses every `.github`
  YAML artefact and asserts on the contract: four workflows exist
  with the prescribed `name`, each declares the expected triggers and
  jobs, `dependabot.yml` covers the three ecosystems, all five issue
  templates parse, the PR template references the quality gates.

### Changed — README
- Badge row replaced. The legacy "Status: Terminé" sticker is gone;
  six dynamic badges now reflect the real state: CI, Security, Codecov
  coverage, Python 3.11 / 3.12 support, MIT licence, and a link to the
  GHCR container packages.

### Known issues
- **PYSEC-2025-52** (mlflow 2.22.5 → fix in 3.1.0). We pin
  `mlflow>=2.16,<3.0` to avoid a mid-phase major version bump, and
  whitelist this CVE in `pip-audit` via `--ignore-vuln PYSEC-2025-52`
  inside `security.yml`. The vulnerability's scope (MLflow Server UI /
  REST) does not match our threat model: the mlflow container is only
  reachable on the internal `deepvision` Docker network and never
  exposed to untrusted callers. Bump tracked as a follow-up PR (will
  also be proposed automatically by dependabot's `ml-stack` group).

---

## [0.9.0] — Phase 8 — Monitoring & Drift (2026-05-10)

The Phase 7 placeholder `drift-monitor` is replaced by a real Prometheus
exporter that periodically computes embedding drift and OOD rate against a
baseline distribution. The audit prescriptions of section 7.2
(Computer Vision specifics) and section 9 (roadmap, Phase 8) are met.

### Added — `src/deepvision/monitoring/`
- **`drift.py`** — pure-NumPy Wasserstein-1D drift on penultimate-layer
  embeddings. Functions: `wasserstein_drift(baseline, current)`,
  `summarize_drift(distances, percentile=95)`, `drift_score(baseline,
  current)` shortcut for the Prometheus gauge update path.
- **`ood.py`** — energy-score OOD detection (Liu et al. 2020,
  `E(x) = -T·logsumexp(z/T)`), with `is_ood(scores, threshold)`,
  `ood_rate(scores, threshold)`, and an `extract_logits(model, batch)`
  helper that recovers logits via penultimate-features × Dense weights
  (no fragile in-place model surgery).
- **`baseline.py`** — `Baseline` dataclass with `save`/`load` to a
  single `.npz`, `compute_baseline(model, images, ...)` that pairs
  embeddings with energy scores, and a `synthetic_reference_images()`
  helper for smoke tests.
- **`server.py`** — `DriftMonitor` class wrapping a
  `prometheus_client.start_http_server` on :9091. Polls every interval,
  samples a synthetic batch (or operator-provided stream later), updates
  drift + OOD gauges and counters. Reuses the FastAPI `InferenceEngine`
  so a single canonical model load serves both serving and monitoring.

### Added — CLI
- `python -m deepvision drift-monitor [--port 9091] [--interval 60]
  [--baseline ./models/baseline.npz] [--ood-threshold -2.0]` boots the
  exporter. Same Typer pattern as `serve` and `streamlit`.

### Added — Monitoring artefacts (`monitoring/`)
- **`alerts.yml`** — eight Prometheus alert rules grouped in two
  families: `deepvision-api` (DeepvisionApiDown, ModelNotLoaded,
  HighInferenceLatencyP95, HighErrorRate) and `deepvision-drift`
  (BaselineMissing, EmbeddingDriftHigh, EmbeddingDriftCritical,
  OodRateHigh).
- **`grafana/provisioning/datasources/prometheus.yml`** — Grafana
  auto-loads the in-cluster Prometheus on http://prometheus:9090 with a
  stable UID `deepvision-prometheus`.
- **`grafana/provisioning/dashboards/dashboards.yml`** — provider that
  scans `/var/lib/grafana/dashboards` on every restart.
- **`grafana/dashboards/deepvision.json`** — six-panel dashboard:
  model_loaded stat, RPS, error rate, OOD rate, latency p50/p95/p99
  timeseries, embedding drift mean/p95/max timeseries.
- **`prometheus.yml`** — adds `rule_files: [/etc/prometheus/alerts.yml]`
  and a new scrape job `deepvision-drift-monitor` targeting
  `drift-monitor:9091`.

### Changed — `docker-compose.yml`
- `drift-monitor` now uses `deepvision-api:dev` (the same image already
  built for the api service) with `entrypoint: python -m deepvision
  drift-monitor`. No fourth Docker image to ship.
- `prometheus` mounts `./monitoring/alerts.yml` read-only so the rules
  load at boot.
- `grafana` mounts `./monitoring/grafana/provisioning` and
  `./monitoring/grafana/dashboards` read-only so the datasource and
  dashboard import without any UI clicking.

### Added — Tests
- **`tests/unit/test_drift.py`** — 11 tests on the Wasserstein
  primitives (zero on identical, monotone in shift, scalar shortcut,
  shape/empty validation).
- **`tests/unit/test_ood.py`** — 9 tests on the energy score
  (closed-form value at zero logits, lower for confident predictions,
  temperature smoothing, threshold logic, OOD rate fraction).
- **`tests/unit/test_baseline.py`** — 7 tests: dataclass properties,
  `.npz` round-trip, parent-dir creation, synthetic image determinism.
- **`tests/unit/test_monitoring_server.py`** — 5 end-to-end tests using
  a tiny hand-built Functional model and a stub `InferenceEngine`:
  initialize loads engine + baseline, `poll_once` updates the gauges,
  `/metrics` exposes the expected eight series.
- **`tests/unit/test_monitoring_artifacts.py`** — ~20 structural tests
  on `alerts.yml` (required alerts present with severity + summary),
  the dashboard JSON (≥6 panels, datasource UID matches provisioning,
  panels query `deepvision_*` metrics), the Grafana provisioning files,
  and the new `prometheus.yml` additions (rule_files, drift-monitor
  scrape job and target).
- **`tests/unit/test_docker_artifacts.py`** — drops the `drift-monitor`
  exemption now that the service has a real healthcheck.

---

## [0.8.0] — Phase 7 — Conteneurisation (2026-05-09)

The project now ships as a six-service Docker stack. Audit section 7.5
prescriptions are met: multi-stage Dockerfiles with non-root runtime,
per-service healthchecks, `.dockerignore` that prunes the local artefacts,
named volumes for stateful services, and a dedicated bridge network.

### Added — Docker artefacts (`docker/`)
- **`api.Dockerfile`** — multi-stage build (builder + runtime) on
  `python:3.12-slim-bookworm`. Installs runtime deps into an isolated venv,
  drops the build toolchain, runs as a fixed UID/GID `deepvision` (10001),
  exposes `:8000`, and wires a `HEALTHCHECK` that hits `/health` via
  stdlib `urllib` (no extra package pulled into the image).
- **`streamlit.Dockerfile`** — same builder pattern; entrypoint shells out
  to `python -m deepvision streamlit`. Healthcheck on
  `/_stcore/health` (Streamlit's built-in liveness endpoint).
  `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false` baked in.
- **`training.Dockerfile`** — CPU-only batch image. Installs both
  `requirements.txt` and `requirements-dev.txt` so MLflow autologging and
  test harness work inside the container. Entrypoint is
  `python -m deepvision train`. No `HEALTHCHECK` (jobs are short-lived).

### Added — Orchestration
- **`docker-compose.yml`** — six services declared:
  `api` (`:8000`), `streamlit` (`:8501`), `mlflow` (`:5000`),
  `prometheus` (`:9090`), `grafana` (`:3000`),
  and a `drift-monitor` placeholder that sleeps until Phase 8 replaces it
  with the real drift detector. `depends_on` uses
  `condition: service_healthy` so the stack boots in dependency order.
  Named volumes (`mlflow_data`, `prometheus_data`, `grafana_data`)
  preserve state across `compose down`. Dedicated `deepvision` bridge
  network keeps DNS predictable.
- **`monitoring/prometheus.yml`** — minimal scrape config: `deepvision-api`
  job hitting `api:8000/metrics` every 15 s, plus a Prometheus self-scrape.
  Phase 8 enriches with recording rules, alerts and the drift-monitor job.
- **`.dockerignore`** — excludes `.venv`, `mlruns`, `models`, `data`,
  `tests`, `docs`, `Notebooks`, `.git`, tooling caches, IDE settings, and
  any local `.env`. Whitelists `.env.example` so `compose up` works
  out of the box for newcomers.
- **`.env.example`** — documents every variable consumed by the compose
  file (API key, model path, log level, MLflow URI, Grafana credentials).

### Added — Tests
- **`tests/unit/test_docker_artifacts.py`** — ~25 structural tests that
  parse the YAML/Dockerfile artefacts (no Docker daemon required). Asserts
  the six-service contract, the per-service healthcheck contract, the
  multi-stage build contract, the non-root user, the exposed ports, the
  scrape configuration, the `.dockerignore` exclusions and the
  `.env.example` documented variables. These tests run in CI even where
  Docker is unavailable.

### Notes
- The `training` image is CPU-only by design — GPU training stays on Colab
  for this project. Phase 10 (ONNX/TFLite export) will revisit whether a
  CUDA variant is worth shipping.
- The `drift-monitor` slot is intentionally a sleeping placeholder; it is
  promoted to a real service in Phase 8.

---

## [0.7.0] — Phase 6 — Refonte Streamlit (2026-05-08)

The Streamlit demo, previously a 68-line `app.py` at the repository root, has
been replaced by a structured module that reuses the Phase 5 serving layer
and the Phase 4 interpretability helpers. The four critical bugs flagged by
the audit (sections 4.4 and 9) are corrected with dedicated regression tests:

- **B1 — Double softmax** removed; probabilities come straight from
  `InferenceEngine.predict`.
- **B2 — Destructive 32×32 resize** replaced by `preprocess_for_efficientnet`
  targeting the 160×160 native input.
- **B3 — Missing RGB conversion** handled by the same shared preprocessor;
  RGBA and grayscale uploads no longer crash.
- **B4 — Deprecated `use_column_width`** replaced everywhere by
  `use_container_width`.

### Added — `src/deepvision/streamlit_app.py`
- Full Streamlit module: page config, sidebar with model name/version (B12
  fix — class names now imported from `deepvision.constants`), file uploader
  with multi-image support, per-image prediction card with top-3 probabilities,
  inference latency, Grad-CAM expander (lazy TensorFlow import).
- `detect_mime` / `validate_mime` for header-based MIME validation on top of
  the browser-reported extension.
- `predict_one(engine, image)` is the testable core: it runs preprocess +
  inference and returns the top-k items, the inference time, and the
  pre-processed batch (used as input for Grad-CAM).
- `render_gradcam_overlay` delegates to
  `deepvision.evaluation.interpretability.grad_cam` and
  `overlay_heatmap_on_image` so there is one canonical Grad-CAM
  implementation across serving and demo.
- Optional clickable example gallery sourced from
  `assets/streamlit/examples/` (silently skipped when the folder is absent).

### Added — `src/deepvision/serving/inference.py`
- New public `InferenceEngine.model` property returning the underlying Keras
  model (auto-loads on first access). Used by the Streamlit demo to feed
  Grad-CAM without touching the private `_model` attribute.

### Added — CLI
- `python -m deepvision streamlit [--host 0.0.0.0] [--port 8501]
  [--headless/--no-headless]` shells out to `streamlit.web.cli` to launch
  `streamlit_app.py`. Streamlit is lazy-imported so `--help` stays instant.

### Added — Configuration (`Settings`, prefix `DEEPVISION_`)
- `streamlit_host` (default `0.0.0.0`) and `streamlit_port` (default `8501`).

### Added — Tests (`tests/unit/test_streamlit_app.py`, ~17 tests)
- Magic-byte detection: PNG / JPEG / WebP recognised, unknown / too-short
  payloads return `None`.
- `validate_mime` rejects a text payload renamed `image.png` (defence in
  depth).
- `predict_one` regression tests:
  - target size is 160×160, **not** 32×32 (B2);
  - RGBA and grayscale inputs do not crash (B3);
  - the engine's probabilities pass through unchanged — no second softmax
    (B1);
  - top-3 length and `inference_time_ms` shape contract.
- `render_gradcam_overlay` calls
  `deepvision.evaluation.interpretability.grad_cam` with the engine's model,
  the (H, W, C) image slice, and the requested target class
  (TensorFlow-free, monkeypatched).
- Static guards on the module source: no `use_column_width`, no
  `tf.nn.softmax`, no `(32, 32)` literal, no hard-coded class names —
  catches regressions before they ship.

### Removed
- Legacy `app.py` at the repository root deleted (`git rm app.py`). The
  refonte lives in `src/deepvision/streamlit_app.py` and is launched via the
  CLI.

### Changed
- Version bumped: `0.6.0` → `0.7.0`.

### Audit bugs addressed
- **B1**, **B2**, **B3**, **B4** (table of bugs, section 5).
- **B12** — duplicated `CLASS_NAMES` between `app.py` and the notebook —
  fixed: the demo imports `CLASS_NAMES_FR` from `deepvision.constants`.

### Tech-debt acknowledged (deferred)
- Sample images for the clickable gallery are **not** bundled in the repo
  yet (`assets/streamlit/examples/` is optional). They will land alongside
  the documentation polish in Phase 11.
- The MLflow Registry round-trip for the served model version is still
  driven by `Settings.serving_model_version`; auto-discovery via
  `MlflowClient.get_latest_versions` is a Phase 9/11 candidate.

---

## [0.6.0] — Phase 5 — Serving FastAPI (2026-05-08)

The project now ships a production-shaped REST API. The serving layer
follows the audit prescriptions (sections 4.4, 7.4 and table 25): strict
input validation, Pydantic v2 schemas, Prometheus instrumentation, optional
API-key auth, and a CLI entrypoint that boots uvicorn.

### Added — `src/deepvision/serving/`
- **`api.py`** — FastAPI app with `/`, `/health`, `/ready`, `/meta`,
  `/predict`, `/predict_batch`, `/metrics` and `/openapi.json`. The
  `create_app(*, settings, engine)` factory builds an isolated app for tests
  and lets us inject a stub engine without touching TensorFlow. CORS is
  configurable via `Settings.cors_allow_origins`. Optional `X-API-Key`
  header enforcement when `DEEPVISION_API_KEY` is set. Each request goes
  through a Prometheus + structured-log middleware that records latency,
  status class, and per-endpoint counters.
- **`prometheus.py`** — dedicated `CollectorRegistry`, latency histogram
  (`deepvision_inference_latency_seconds`, 10 buckets from 5 ms to 5 s),
  request and error counters (`deepvision_http_requests_total`,
  `deepvision_http_errors_total`), `model_loaded` gauge, and a label-only
  `model_info` gauge for Grafana.
- **`schemas.py`** — added `BatchPredictionItem` and `BatchPredictResponse`,
  whitelisted Pydantic v2 protected namespaces so `model_name` /
  `model_version` keep their canonical names in the OpenAPI document.
- **`__init__.py`** — re-exports the public surface and lazy-loads the
  FastAPI symbols via PEP 562 (importing `deepvision.serving` no longer
  pulls FastAPI into memory).

### Added — CLI
- `python -m deepvision serve [--host 0.0.0.0] [--port 8000] [--reload]
  [--workers N] [--log-level info]` boots uvicorn against
  `deepvision.serving.api:app`.

### Added — Configuration (`Settings`, prefix `DEEPVISION_`)
- `api_host`, `api_port`, `api_reload`, `api_key`, `cors_allow_origins`,
  `max_image_bytes`, `max_batch_size`, `model_path`, `serving_model_name`,
  `serving_model_version`. All overridable via env or `.env`.

### Added — Tests (5 modules, ~50 unit tests)
- `test_serving_preprocess.py` — RGB/RGBA/grayscale → uint8(1,160,160,3),
  payload-size and decompression-bomb guards, end-to-end happy path.
- `test_serving_inference.py` — lazy load, idempotency, top-k contract,
  random-weights softmax sanity. Uses `weights=None` so it runs on a CPU.
- `test_serving_schemas.py` — round-trips, Pydantic validation errors,
  default values match the package version and class catalogue.
- `test_serving_prometheus.py` — metric types, bucket monotonicity,
  `status_class` mapping, exposition output sanity.
- `test_serving_api.py` — `TestClient` end-to-end coverage of every
  endpoint (happy path + 400/401/413/415/422 branches), CORS preflight,
  X-API-Key enforcement, OpenAPI schema sanity.

### Changed
- `requirements.txt` adds `fastapi>=0.115,<1.0`, `uvicorn[standard]>=0.32`,
  `python-multipart>=0.0.12`, `prometheus-client>=0.21`. `requirements-dev.txt`
  adds `httpx>=0.27` (FastAPI TestClient backend).
- Version bumped: `0.5.0` → `0.6.0`.

### Audit bugs addressed (originally flagged in `Audit_DeepVision_CIFAR10.docx`)
- B3 — *No RGB conversion* — `preprocess_for_efficientnet` always coerces to
  RGB, covered by dedicated unit tests for both RGBA-PNG and grayscale-PNG
  uploads.
- B14 — *No input validation* — payload size cap, decompression-bomb guard
  on dimensions, MIME-type allowlist, hard 401 path on missing/invalid key.
- B13 — *No structured logging* — every request emits a `method=… path=…
  status=… duration_ms=…` line through `deepvision.utils.logging.get_logger`.

### Tech-debt acknowledged (deferred)
- The `/explain` Grad-CAM endpoint is **not** part of Phase 5 — it lands in
  Phase 6 alongside the Streamlit refresh, so the same preprocessing path is
  reused in both UIs.
- Rate limiting (slowapi) is deferred to Phase 7 (containerisation), where
  it will be paired with an Nginx reverse-proxy in the Docker Compose stack.
- pip-tools lock-files are still pending; Phase 5 only adds upper-bounded
  ranges to keep the runtime footprint stable until then.

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
