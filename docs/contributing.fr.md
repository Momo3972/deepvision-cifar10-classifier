# Contribuer

Merci de votre intérêt pour contribuer à
**deepvision-cifar10-classifier**. Cette page est une orientation
rapide pour quiconque ouvre une pull request -- correction de bug,
nouvelle fonctionnalité, ou modification de la documentation.

!!! info "Guide complet disponible en anglais"
    La version détaillée (modèle de branching, conventions de
    messages de commit, checklist PR, triage Dependabot) est
    actuellement disponible uniquement en anglais. La traduction
    française intégrale est en cours.

[:material-arrow-right: Lire le guide complet (anglais)](../contributing/){ .md-button .md-button--primary }

## Workflow en bref

- `main` est **toujours vert** -- CI passe, chaque commit est
  déployable.
- Le travail se fait sur des branches `feat/<description-courte>`.
- Les hotfixes sur `fix/<description-courte>`.
- Les phases suivent la feuille de route audit :
  `feat/phase-<N>-<scope>`.

## Quality gates locaux

Avant d'ouvrir la PR, lancer les quatre gates que la CI imposera :

```bash
ruff check .
ruff format --check .
mypy src
pytest -n auto
```

Un hook `pre-commit` lance `ruff` automatiquement sur les fichiers
stagés :

```bash
pre-commit install
```

## Messages de commit

On suit
[Conventional Commits](https://www.conventionalcommits.org/).
Préfixes autorisés : `feat`, `fix`, `chore`, `docs`, `refactor`,
`test`, `perf`, `ci`.

## Signaler un bug

Ouvrir une issue via le template **Bug report** (ou **Model issue** /
**Drift report** si plus approprié). Les templates sont dans
[`.github/ISSUE_TEMPLATE/`](https://github.com/Momo3972/deepvision-cifar10-classifier/tree/main/.github/ISSUE_TEMPLATE).
Les issues blanches sont désactivées : les templates demandent les
informations dont on aura besoin de toute façon.

## Protection de branche

Voir [Protection des branches](contributing/branch-protection.md)
pour la configuration manuelle des règles de protection sur `main`
(reviews requises, status checks requis, historique linéaire,
résolution des conversations).
