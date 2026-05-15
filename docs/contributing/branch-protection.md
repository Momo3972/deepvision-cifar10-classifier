# Branch protection rules

GitHub branch protection rules are **not stored in the repository** — they
have to be configured once, by hand, in the GitHub UI. This document is the
checklist that mirrors what the Phase 9 CI/CD workflows (`ci.yml`,
`security.yml`, `docker.yml`, `docs.yml`) expect from the host platform.

Configure them at:

> **Settings → Branches → Add branch protection rule** (or **Edit** an
> existing rule on `main`).

## Rule for `main`

| Field                                          | Value                                  |
| ---------------------------------------------- | -------------------------------------- |
| **Branch name pattern**                        | `main`                                 |
| Require a pull request before merging          | ✅ on                                   |
| ↳ Require approvals                            | ✅ **1** reviewer                       |
| ↳ Dismiss stale approvals on new commit        | ✅ on                                   |
| ↳ Require review from Code Owners              | ⛔ off (no CODEOWNERS in this repo yet) |
| Require status checks to pass                  | ✅ on                                   |
| ↳ Require branches to be up to date            | ✅ on                                   |
| ↳ **Required status checks** (see list below)  | configured                             |
| Require conversation resolution before merging | ✅ on                                   |
| Require signed commits                         | ⛔ off (optional, off for solo work)    |
| Require linear history                         | ✅ on                                   |
| Require deployments to succeed                 | ⛔ off                                  |
| Lock branch                                    | ⛔ off                                  |
| Do not allow bypassing the above settings      | ✅ on (also applies to admins)          |
| Allow force pushes                             | ⛔ off                                  |
| Allow deletions                                | ⛔ off                                  |

### Required status checks

Pick the entries below from the **search box** once they have run at least
once on the repository (GitHub only surfaces checks it has already seen):

- `lint`
- `typecheck (py3.11)`
- `typecheck (py3.12)`
- `test (py3.11)`
- `test (py3.12)`
- `bandit`
- `pip-audit`
- `gitleaks`
- `codeql`

Do **not** require `trivy (image scan)` — it only runs on `tag v*` and would
permanently block the merge of every non-release PR.

Do **not** require the `docker` workflow — its build-only PR job is a soft
validation; failures should not block a merge if the rest of CI is green.

Do **not** require `docs` — the workflow short-circuits via
`check-docs-source` until Phase 11 populates the `docs/` directory.

## After the first successful CI run

The required-check search box stays empty until a workflow with a given job
name has actually completed once. So the order is:

1. Merge Phase 9 (this PR) **without** the required-check rule.
2. Wait for the first `ci` + `security` runs on `main` to finish (≈ 15 min).
3. Add the branch protection rule and tick all the checks listed above.

After that, every subsequent PR will require those checks to pass before
the **Merge** button becomes available.

## Programmatic alternative

If you prefer Infrastructure-as-Code, mirror this rule with the GitHub CLI:

```bash
gh api -X PUT \
  repos/Momo3972/deepvision-cifar10-classifier/branches/main/protection \
  -F required_status_checks.strict=true \
  -F 'required_status_checks.contexts[]=lint' \
  -F 'required_status_checks.contexts[]=typecheck (py3.11)' \
  -F 'required_status_checks.contexts[]=typecheck (py3.12)' \
  -F 'required_status_checks.contexts[]=test (py3.11)' \
  -F 'required_status_checks.contexts[]=test (py3.12)' \
  -F 'required_status_checks.contexts[]=bandit' \
  -F 'required_status_checks.contexts[]=pip-audit' \
  -F 'required_status_checks.contexts[]=gitleaks' \
  -F 'required_status_checks.contexts[]=codeql' \
  -F required_pull_request_reviews.required_approving_review_count=1 \
  -F required_pull_request_reviews.dismiss_stale_reviews=true \
  -F required_conversation_resolution=true \
  -F required_linear_history=true \
  -F enforce_admins=true \
  -F allow_force_pushes=false \
  -F allow_deletions=false
```
