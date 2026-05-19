"""Structural tests for the Phase 11 documentation artefacts.

These tests do **not** run ``mkdocs build`` (too slow for the unit
tier; the CI workflow ``.github/workflows/docs.yml`` does that). They
parse the YAML and Markdown files under ``docs/`` and ``mkdocs.yml``
and assert on the contract:

* ``mkdocs.yml`` is valid YAML with the expected site metadata,
  plugins, and theme settings;
* every page referenced in the ``nav`` exists on disk;
* the bilingual ``.fr.md`` variant of each user-facing page is
  present (so the FR site does not fall back silently);
* the Hugging Face model card has every prescribed top-level section;
* the README links to the published documentation site.

Running these in unit tests (rather than relying on the workflow)
keeps the contract visible from a fresh ``pytest`` run and prevents
silent drift between what the docs nav promises and what the
``docs/`` folder actually contains.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest
import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DOCS_DIR: Final[Path] = REPO_ROOT / "docs"
MKDOCS_YML: Final[Path] = REPO_ROOT / "mkdocs.yml"
README: Final[Path] = REPO_ROOT / "README.md"

# Pages we expect to exist in EN. The corresponding ``.fr.md`` variants
# are tested in :func:`test_french_variants_exist` below.
REQUIRED_EN_PAGES: Final[frozenset[str]] = frozenset(
    {
        "index.md",
        "getting-started.md",
        "architecture.md",
        "tutorials/index.md",
        "tutorials/training.md",
        "tutorials/serving.md",
        "tutorials/monitoring.md",
        "tutorials/export.md",
        "reference/index.md",
        "model-card.md",
        "contributing.md",
    }
)

# Pages where a FR variant must exist (placeholder or full translation).
# ``contributing/branch-protection.md`` is allowed to fall back to EN
# because it is a Phase 9 operator handbook with little FR-specific value.
REQUIRED_FR_VARIANTS: Final[frozenset[str]] = frozenset(
    {
        "index.fr.md",
        "getting-started.fr.md",
        "architecture.fr.md",
        "tutorials/index.fr.md",
        "tutorials/training.fr.md",
        "tutorials/serving.fr.md",
        "tutorials/monitoring.fr.md",
        "tutorials/export.fr.md",
        "reference/index.fr.md",
        "model-card.fr.md",
        "contributing.fr.md",
    }
)

# Hugging Face model card -- every level-2 heading the template
# prescribes. We grep the markdown for these to catch silent removals.
HF_MODEL_CARD_SECTIONS: Final[frozenset[str]] = frozenset(
    {
        "Model details",
        "Intended use",
        "Bias, risks, and limitations",
        "How to get started",
        "Training details",
        "Evaluation",
        "Environmental impact",
        "Technical specifications",
        "Citation",
    }
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mkdocs_config() -> dict:
    """Parse ``mkdocs.yml`` once per test module.

    We use ``yaml.unsafe_load`` because mkdocs supports the
    ``!!python/name:`` tag (for example to reference Pygments
    formatters in ``pymdownx.superfences``). The file is repository-
    controlled and reviewed, so loading is safe.
    """
    return yaml.unsafe_load(MKDOCS_YML.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model_card_en() -> str:
    return (DOCS_DIR / "model-card.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# mkdocs.yml -- site metadata
# ---------------------------------------------------------------------------


def test_mkdocs_yml_is_present() -> None:
    assert MKDOCS_YML.is_file(), "mkdocs.yml missing at repo root"


def test_site_metadata(mkdocs_config: dict) -> None:
    assert mkdocs_config["site_name"].startswith("DeepVision")
    assert mkdocs_config["site_url"].startswith("https://")
    assert "github.com/Momo3972" in mkdocs_config["repo_url"]
    assert mkdocs_config["repo_name"].startswith("Momo3972/")


def test_theme_is_material(mkdocs_config: dict) -> None:
    theme = mkdocs_config["theme"]
    assert theme["name"] == "material"
    # Must offer a light/dark palette toggle.
    palette = theme["palette"]
    assert isinstance(palette, list)
    assert any(p.get("scheme") == "slate" for p in palette)
    assert any(p.get("scheme") == "default" for p in palette)


def test_theme_features_do_not_include_navigation_instant(mkdocs_config: dict) -> None:
    """``navigation.instant`` breaks the mkdocs-static-i18n switcher.

    Regression guard: a future contributor adding ``navigation.instant``
    would degrade the language toggle without noticing.
    """
    features = mkdocs_config["theme"].get("features", [])
    assert "navigation.instant" not in features


# ---------------------------------------------------------------------------
# mkdocs.yml -- plugins
# ---------------------------------------------------------------------------


def _plugin_names(config: dict) -> set[str]:
    """Return the set of plugin names from the ``plugins`` block.

    Each entry is either a bare string or a dict ``{name: options}``.
    """
    names: set[str] = set()
    for entry in config.get("plugins", []):
        if isinstance(entry, str):
            names.add(entry)
        elif isinstance(entry, dict):
            names.update(entry.keys())
    return names


def test_required_plugins_are_declared(mkdocs_config: dict) -> None:
    required = {"search", "i18n", "mkdocstrings"}
    missing = required - _plugin_names(mkdocs_config)
    assert not missing, f"Missing mkdocs plugins: {missing}"


def test_i18n_has_en_and_fr_languages(mkdocs_config: dict) -> None:
    """The i18n plugin must declare EN (default) and FR."""
    for entry in mkdocs_config.get("plugins", []):
        if isinstance(entry, dict) and "i18n" in entry:
            languages = entry["i18n"]["languages"]
            locales = {lang["locale"] for lang in languages}
            assert {"en", "fr"} <= locales
            # English must be the default.
            assert any(lang["locale"] == "en" and lang.get("default") for lang in languages)
            return
    pytest.fail("i18n plugin block not found")


def test_mkdocstrings_uses_numpy_docstring_style(mkdocs_config: dict) -> None:
    """The whole code base uses NumPy docstrings -- the config must match."""
    for entry in mkdocs_config.get("plugins", []):
        if isinstance(entry, dict) and "mkdocstrings" in entry:
            options = entry["mkdocstrings"]["handlers"]["python"]["options"]
            assert options["docstring_style"] == "numpy"
            return
    pytest.fail("mkdocstrings plugin block not found")


# ---------------------------------------------------------------------------
# mkdocs.yml -- navigation contract
# ---------------------------------------------------------------------------


def _flatten_nav(nav: list) -> list[str]:
    """Return every relative markdown path mentioned in the nav tree."""
    paths: list[str] = []
    for item in nav:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str):
                    paths.append(value)
                elif isinstance(value, list):
                    paths.extend(_flatten_nav(value))
    return paths


def test_every_nav_entry_exists_on_disk(mkdocs_config: dict) -> None:
    """Catches typos in the nav block before mkdocs does."""
    missing: list[str] = []
    for rel in _flatten_nav(mkdocs_config["nav"]):
        if not (DOCS_DIR / rel).is_file():
            missing.append(rel)
    assert not missing, f"Nav references missing pages: {missing}"


def test_all_required_en_pages_present() -> None:
    missing = {p for p in REQUIRED_EN_PAGES if not (DOCS_DIR / p).is_file()}
    assert not missing, f"Missing EN pages: {missing}"


# ---------------------------------------------------------------------------
# Bilingual coverage
# ---------------------------------------------------------------------------


def test_all_required_fr_variants_present() -> None:
    missing = {p for p in REQUIRED_FR_VARIANTS if not (DOCS_DIR / p).is_file()}
    assert not missing, f"Missing FR variants: {missing}"


@pytest.mark.parametrize("fr_page", sorted(REQUIRED_FR_VARIANTS))
def test_fr_pages_are_not_empty(fr_page: str) -> None:
    content = (DOCS_DIR / fr_page).read_text(encoding="utf-8").strip()
    assert len(content) > 50, f"{fr_page} looks too empty (<50 chars)"


# ---------------------------------------------------------------------------
# Model card -- Hugging Face template
# ---------------------------------------------------------------------------


def test_model_card_has_all_hf_sections(model_card_en: str) -> None:
    """Every prescribed level-2 heading must appear, case-insensitive.

    We match on the *raw* markdown text so the test catches both
    ``## Model details`` and ``## Model Details``.
    """
    lowered = model_card_en.lower()
    missing = [
        section for section in HF_MODEL_CARD_SECTIONS if f"## {section.lower()}" not in lowered
    ]
    assert not missing, f"Model card missing HF sections: {missing}"


def test_model_card_declares_version(model_card_en: str) -> None:
    """The model card must surface the package version explicitly."""
    assert re.search(r"\*\*Version:\*\*\s*`0\.\d+\.\d+`", model_card_en), (
        "Model card must surface the version in the Model details section"
    )


# ---------------------------------------------------------------------------
# README contract
# ---------------------------------------------------------------------------


def test_readme_links_to_docs_site(readme_text: str) -> None:
    assert "momo3972.github.io/deepvision-cifar10-classifier" in readme_text


def test_readme_has_language_switcher(readme_text: str) -> None:
    """The bilingual switcher block at the top must reference both locales."""
    assert "🇬🇧 English" in readme_text
    assert "🇫🇷 Français" in readme_text


def test_readme_has_docs_badge(readme_text: str) -> None:
    """The docs.yml workflow badge must be present so users see build status."""
    assert "actions/workflows/docs.yml/badge.svg" in readme_text
