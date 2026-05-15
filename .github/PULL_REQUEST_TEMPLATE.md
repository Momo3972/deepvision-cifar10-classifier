<!--
  Phase 9 - PR template. The audit (section 7.5) prescribes a PR template
  alongside the issue templates. Tick every box before requesting review.
-->

## Summary

<!-- One paragraph describing WHAT changes and WHY (link to the audit
section, the GitHub issue, or the roadmap phase if applicable). -->

## Related

<!-- "Closes #123" / "Refs #456" / "Phase X of Audit_DeepVision_CIFAR10.docx" -->

## Type of change

<!-- Tick what applies. Multiple boxes are fine. -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Refactor / cleanup (no functional change)
- [ ] Documentation
- [ ] CI / tooling
- [ ] Dependency upgrade
- [ ] Phase delivery (one of the 13 audit phases)

## Quality checklist

- [ ] `ruff check .` passes locally
- [ ] `ruff format --check .` passes locally
- [ ] `mypy src/deepvision` passes locally
- [ ] `pytest -q` passes locally
- [ ] New / changed code is covered by tests
- [ ] `CHANGELOG.md` updated (under `[Unreleased]` or a new version section)
- [ ] If a public API changes: docstrings updated, version bumped in
      `pyproject.toml` and `src/deepvision/__init__.py`

## Audit alignment (for phase-delivery PRs)

- [ ] Section(s) of `Audit_DeepVision_CIFAR10.docx` addressed: <!-- e.g. 7.5 + 9 -->
- [ ] Audit prescriptions met or explicitly deferred (with rationale)

## Smoke test

<!-- Brief description of what you tested manually after the change.
     For service changes: which endpoints / pages / commands you exercised. -->

## Screenshots / logs (optional)

<!-- For UI / dashboard changes; for debugging traces; for CLI output. -->
