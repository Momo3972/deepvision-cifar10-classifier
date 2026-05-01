# ============================================================================
# tasks.ps1 - deepvision-cifar10-classifier
# Equivalent PowerShell du Makefile, pour Windows natif (sans WSL/Git Bash).
# ----------------------------------------------------------------------------
# Usage :
#   .\tasks.ps1 help
#   .\tasks.ps1 install-dev
#   .\tasks.ps1 lint
#   .\tasks.ps1 test
#   .\tasks.ps1 check
# ============================================================================

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet(
        "help",
        "install", "install-dev",
        "lint", "format", "typecheck", "security",
        "test", "test-fast",
        "check",
        "diagnose",
        "clean"
    )]
    [string]$Task = "help"
)

$ErrorActionPreference = "Stop"

# Couleurs sympas pour la lisibilite
function Write-Section([string]$msg) { Write-Host "`n>>> $msg" -ForegroundColor Cyan }
function Write-Success([string]$msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Failure([string]$msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red }

# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------
function Invoke-Help {
    Write-Host ""
    Write-Host "deepvision-cifar10-classifier - tasks disponibles :" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  install        Dependencies runtime + package en mode editable"
    Write-Host "  install-dev    + outils de developpement (ruff, mypy, pytest, pre-commit)"
    Write-Host "  lint           Lint avec ruff"
    Write-Host "  format         Auto-formatage et auto-fix avec ruff"
    Write-Host "  typecheck      Type-checking statique avec mypy"
    Write-Host "  security       Scans de securite (bandit + pip-audit)"
    Write-Host "  test           Suite de tests avec couverture"
    Write-Host "  test-fast      Tests en parallele, sans couverture"
    Write-Host "  check          lint + typecheck + test (gate CI)"
    Write-Host "  diagnose       Lance scripts/check_machine.py"
    Write-Host "  clean          Supprime les caches et artifacts"
    Write-Host ""
    Write-Host "Exemple : .\tasks.ps1 install-dev" -ForegroundColor Gray
    Write-Host ""
}

# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------
function Invoke-Install {
    Write-Section "Installation des dependencies runtime"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .
    Write-Success "Dependencies runtime installees"
}

function Invoke-InstallDev {
    Invoke-Install
    Write-Section "Installation des dependencies dev"
    python -m pip install -r requirements-dev.txt
    Write-Section "Installation des hooks pre-commit"
    pre-commit install
    Write-Success "Environnement de developpement pret"
}

# ---------------------------------------------------------------------------
# lint / format / typecheck / security
# ---------------------------------------------------------------------------
function Invoke-Lint {
    Write-Section "Lint avec ruff"
    python -m ruff check src tests scripts
}

function Invoke-Format {
    Write-Section "Auto-format avec ruff"
    python -m ruff format src tests scripts
    python -m ruff check --fix src tests scripts
}

function Invoke-TypeCheck {
    Write-Section "Type-check avec mypy"
    python -m mypy src
}

function Invoke-Security {
    Write-Section "Scan de securite avec bandit"
    python -m bandit -r src -q
    Write-Section "Audit des dependencies avec pip-audit"
    python -m pip_audit --strict --requirement requirements.txt
}

# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------
function Invoke-Test {
    Write-Section "Suite de tests avec couverture"
    python -m pytest --cov --cov-report=term-missing
}

function Invoke-TestFast {
    Write-Section "Tests rapides (parallele, sans couverture)"
    python -m pytest -n auto -q
}

# ---------------------------------------------------------------------------
# aggregat
# ---------------------------------------------------------------------------
function Invoke-Check {
    Invoke-Lint
    Invoke-TypeCheck
    Invoke-Test
    Write-Success "Tous les checks ont passe"
}

# ---------------------------------------------------------------------------
# diagnostic & cleanup
# ---------------------------------------------------------------------------
function Invoke-Diagnose {
    python scripts\check_machine.py
}

function Invoke-Clean {
    Write-Section "Nettoyage des caches et artifacts"
    $patterns = @(
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "htmlcov",
        ".coverage",
        "build",
        "dist"
    )
    foreach ($p in $patterns) {
        if (Test-Path $p) {
            Remove-Item -Recurse -Force $p
            Write-Host "  removed $p"
        }
    }
    # Caches __pycache__ recursifs
    Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Force -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            Remove-Item -Recurse -Force $_.FullName
            Write-Host "  removed $($_.FullName)"
        }
    Get-ChildItem -Path . -Filter "*.egg-info" -Recurse -Force -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            Remove-Item -Recurse -Force $_.FullName
            Write-Host "  removed $($_.FullName)"
        }
    Write-Success "Nettoyage termine"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
switch ($Task) {
    "help"        { Invoke-Help }
    "install"     { Invoke-Install }
    "install-dev" { Invoke-InstallDev }
    "lint"        { Invoke-Lint }
    "format"      { Invoke-Format }
    "typecheck"   { Invoke-TypeCheck }
    "security"   { Invoke-Security }
    "test"        { Invoke-Test }
    "test-fast"   { Invoke-TestFast }
    "check"       { Invoke-Check }
    "diagnose"    { Invoke-Diagnose }
    "clean"       { Invoke-Clean }
}
