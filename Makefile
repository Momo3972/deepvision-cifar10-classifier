# ============================================================================
# Makefile - deepvision-cifar10-classifier
# Cross-platform task runner (Linux, macOS, Git Bash on Windows).
# Windows-native users: see tasks.ps1 for the equivalent PowerShell script.
# ============================================================================

# Use bash on Linux/macOS, sh-like on Git Bash. Avoid /bin/sh for portability.
SHELL := /bin/bash

# Python interpreter — overridable: make PY=python3.12 install
PY     ?= python
PIP    ?= $(PY) -m pip
PYTEST ?= $(PY) -m pytest
RUFF   ?= $(PY) -m ruff
MYPY   ?= $(PY) -m mypy

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help: ## Display this help message.
	@echo ""
	@echo "deepvision-cifar10-classifier - available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
.PHONY: install
install: ## Install runtime dependencies + the package in editable mode.
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

.PHONY: install-dev
install-dev: install ## Install dev tooling (ruff, mypy, pytest, pre-commit).
	$(PIP) install -r requirements-dev.txt
	pre-commit install

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------
.PHONY: lint
lint: ## Lint with ruff (no auto-fix).
	$(RUFF) check src tests scripts

.PHONY: format
format: ## Auto-format and auto-fix with ruff.
	$(RUFF) format src tests scripts
	$(RUFF) check --fix src tests scripts

.PHONY: typecheck
typecheck: ## Static type-check the package with mypy.
	$(MYPY) src

.PHONY: security
security: ## Static security scan with bandit + dependency CVEs with pip-audit.
	$(PY) -m bandit -r src -q
	$(PY) -m pip_audit --strict --requirement requirements.txt

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
.PHONY: test
test: ## Run the full test suite with coverage.
	$(PYTEST) --cov --cov-report=term-missing

.PHONY: test-fast
test-fast: ## Run tests in parallel, no coverage (quicker feedback).
	$(PYTEST) -n auto -q

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
.PHONY: check
check: lint typecheck test ## Run lint + typecheck + tests (CI gate).

# ---------------------------------------------------------------------------
# Hardware diagnostic
# ---------------------------------------------------------------------------
.PHONY: diagnose
diagnose: ## Run scripts/check_machine.py to introspect the local environment.
	$(PY) scripts/check_machine.py

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
.PHONY: clean
clean: ## Remove caches, build artifacts and coverage reports.
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
