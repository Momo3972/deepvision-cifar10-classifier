# Contributing

Thanks for your interest in contributing to **deepvision-cifar10-classifier**.
This page is a quick orientation for anyone opening a pull request --
whether you're shipping a bug fix, a new feature, or a doc tweak.

## Branching model

We use a simple trunk-based flow:

- `main` is **always green** -- CI passes, every commit is shippable.
- Feature work happens on `feat/<short-description>` branches.
- Hotfixes land on `fix/<short-description>` branches.
- Phase work follows the audit roadmap: `feat/phase-<N>-<scope>`.

Branch protection (configured in
[`docs/contributing/branch-protection.md`](https://github.com/Momo3972/deepvision-cifar10-classifier/blob/main/docs/contributing/branch-protection.md)
on the repo) requires the CI status checks below to pass before
anything can land on `main`:

- `ci/lint`
- `ci/typecheck (py3.11)` and `ci/typecheck (py3.12)`
- `ci/test (py3.11)` and `ci/test (py3.12)`
- `security/bandit`, `security/pip-audit`, `security/gitleaks`,
  `security/codeql`

## Local quality gates

Before opening the PR, run the same four gates the CI will:

```bash
# 1. Lint + format
ruff check .
ruff format --check .

# 2. Static types
mypy src

# 3. Tests (parallel)
pytest -n auto

# 4. Coverage (optional but appreciated for new modules)
pytest --cov=deepvision --cov-report=term-missing
```

We use a `pre-commit` hook that runs `ruff` automatically on staged
files. Install it once:

```bash
pre-commit install
```

## Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org/).
A canonical example:

```
feat(export): add TFLite Full INT8 quantization mode

- New QuantizationMode.INT8 in src/deepvision/export/tflite.py
- 200-sample CIFAR-10 calibration by default
- CLI flag: --quantization int8
- 12 new unit tests, coverage 100% on tflite.py

Closes #N
```

Allowed prefixes: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`,
`perf`, `ci`. The scope in parentheses is optional but encouraged.

## Pull request checklist

Every PR opened against `main` auto-loads the template at
[`.github/PULL_REQUEST_TEMPLATE.md`](https://github.com/Momo3972/deepvision-cifar10-classifier/blob/main/.github/PULL_REQUEST_TEMPLATE.md).
The non-negotiable items:

- [ ] CI status checks all green
- [ ] `CHANGELOG.md` updated under `[Unreleased]` (or a new version
      entry if you're shipping a phase)
- [ ] New code is covered by tests
- [ ] Public API additions are documented (docstrings + a sentence in
      the right `docs/` page)
- [ ] No new dependencies without justification in the PR description

## Documentation contributions

The site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
in `mkdocs.yml` and the markdown pages live under `docs/`. Preview
your changes locally with:

```bash
mkdocs serve
# -> http://127.0.0.1:8000/
```

The site is **bilingual**:

- English files live at the regular path: `docs/foo.md`.
- French translations live next to them with the `.fr.md` suffix:
  `docs/foo.fr.md`.
- The `i18n` plugin in `mkdocs.yml` wires the language switcher.

If you only touch the English version, please add a `TODO: French
translation pending` line at the top of the FR variant so the team
knows to follow up.

## Reporting bugs

Open an issue using the **Bug report** template (or **Model issue** /
**Drift report** if those fit better). The four templates live in
[`.github/ISSUE_TEMPLATE/`](https://github.com/Momo3972/deepvision-cifar10-classifier/tree/main/.github/ISSUE_TEMPLATE).
Blank issues are disabled on purpose -- the templates ask for the
information we'll need anyway, in a structured way.

## Triage of Dependabot PRs

Dependabot opens grouped PRs once a week. Triage rules:

1. **`github-actions` updates** -- merge on green CI without further
   review. Action versions are pinned to tags, so the risk is minimal.
2. **`docker` updates** -- read the base-image changelog (debian
   point releases ship security fixes), then merge if CI is green.
3. **`pip` updates in the `dev-tooling` group** (`ruff`, `mypy`,
   `pytest`, ...) -- merge in batch on green CI. Major bumps may
   surface new lint/type errors; fix them in the same PR if trivial,
   open a follow-up otherwise.
4. **`pip` updates in the `serving-stack` group** (`fastapi`,
   `pydantic`, `uvicorn`, `streamlit`) -- read the release notes for
   breaking changes; sometimes a behaviour tweak warrants a code
   adjustment.
5. **`pip` updates in the `ml-stack` group** (`tensorflow`, `keras`,
   `numpy`, `scikit-learn`) -- the most sensitive group. Always look
   at the release notes; bump locally first to make sure training
   still converges before merging.

## Code of conduct

Be kind, be technical, be specific. Disagreements happen -- when they
do, default to writing more, not less. The audit document
(`Audit_DeepVision_CIFAR10.docx`) is the source of truth for
prescriptive decisions; if you think the audit is wrong, open a
discussion before opening the PR.

## Next steps

- Read the [Architecture overview](architecture.md).
- Pick a tutorial: [Tutorials](tutorials/index.md).
- Browse the [API reference](reference/index.md).
