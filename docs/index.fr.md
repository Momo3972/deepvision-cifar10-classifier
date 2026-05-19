# DeepVision -- CIFAR-10

**Pipeline industriel de Computer Vision pour CIFAR-10.** Compare MLP,
CNN custom et EfficientNetB0 (transfer learning + data augmentation)
avec un cycle MLOps complet : suivi MLflow, service FastAPI, images
Docker, supervision de dérive et CI/CD.

[![CI](https://github.com/Momo3972/deepvision-cifar10-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Momo3972/deepvision-cifar10-classifier/actions/workflows/ci.yml)
[![Security](https://github.com/Momo3972/deepvision-cifar10-classifier/actions/workflows/security.yml/badge.svg)](https://github.com/Momo3972/deepvision-cifar10-classifier/actions/workflows/security.yml)
[![Codecov](https://codecov.io/gh/Momo3972/deepvision-cifar10-classifier/branch/main/graph/badge.svg)](https://codecov.io/gh/Momo3972/deepvision-cifar10-classifier)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Pourquoi ce projet

CIFAR-10 est un dataset *pédagogique*. La question intéressante n'est
pas « peut-on atteindre 90 % de précision ? » -- la réponse est « oui »
depuis 2015 -- mais **« peut-on emballer le modèle avec tout ce que
l'industrie attend pour réellement le déployer et l'opérer ? »**.
Ce projet répond à la question avec une base de code unique qui couvre
toutes les surfaces MLOps :

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } **Trois familles de modèles**

    ---

    Baseline MLP, CNN custom avec BatchNorm + dropout, et tête
    transfer-learning EfficientNetB0 avec fine-tuning optionnel.
    Tous entraînés sur le même split CIFAR-10 reproductible.

-   :material-rocket-launch:{ .lg .middle } **Service d'inférence FastAPI**

    ---

    API REST asynchrone avec schémas Pydantic v2, validation des
    fichiers par magic-bytes, auth optionnelle `X-API-Key`,
    instrumentation Prometheus, et endpoint `/predict_batch` capé à
    16 images par appel.

-   :material-chart-bell-curve:{ .lg .middle } **Supervision dérive & OOD**

    ---

    Détection de dérive Wasserstein-1D sur les embeddings de
    l'avant-dernière couche + scoring out-of-distribution basé énergie,
    exportés vers Prometheus et visualisés dans un tableau de bord
    Grafana pré-provisionné.

-   :material-export:{ .lg .middle } **Export multi-runtime**

    ---

    Export en une commande vers ONNX (opset 17, validé en forward) et
    TFLite (quantization Full INT8 avec calibration CIFAR-10).
    Benchmark intégré qui reporte la latence p50 / p90 / p95 / p99
    pour chaque runtime.

-   :material-docker:{ .lg .middle } **Stack Docker six services**

    ---

    `api`, `streamlit`, `mlflow`, `prometheus`, `grafana`,
    `drift-monitor` -- tous reliés par un réseau privé avec
    healthchecks. Démarrage production via `docker compose up`.

-   :material-cog:{ .lg .middle } **CI/CD complet**

    ---

    Quatre workflows GitHub Actions imposent `ruff`, `mypy`, `pytest`,
    `bandit`, `pip-audit`, `gitleaks`, `codeql`, et `trivy` sur les
    tags pour le scan des images.

</div>

---

## Démarrage rapide

```bash
# 1. Cloner + installer
git clone https://github.com/Momo3972/deepvision-cifar10-classifier.git
cd deepvision-cifar10-classifier
pip install -e ".[dev]"

# 2. Entraîner un mini modèle (compatible CPU)
python -m deepvision train --model efficientnet --quick

# 3. Le servir
python -m deepvision serve --port 8000
# -> http://localhost:8000/docs

# 4. Ou lancer la stack complète via Docker
docker compose up -d
```

Voir la page [Démarrage rapide](getting-started.md) pour le pas-à-pas
complet, ou aller directement à un [tutoriel](tutorials/index.md) pour
une tâche précise.

---

## Statut du projet

| Phase | Périmètre | Statut |
|---|---|---|
| 1 -- Packaging | `pyproject.toml`, CLI, logs, config | :material-check-circle:{ .green } livré |
| 2 -- Données | Split reproductible, pipeline d'augmentation | :material-check-circle:{ .green } livré |
| 3 -- Entraînement | MLP / CNN / EfficientNet + tracking MLflow | :material-check-circle:{ .green } livré |
| 4 -- Évaluation | Métriques, calibration, robustesse, Grad-CAM | :material-check-circle:{ .green } livré |
| 5 -- Service | App FastAPI + middleware Prometheus | :material-check-circle:{ .green } livré |
| 6 -- Streamlit | UI démo avec Grad-CAM et upload batch | :material-check-circle:{ .green } livré |
| 7 -- Docker | Images multi-stage + compose 6 services | :material-check-circle:{ .green } livré |
| 8 -- Supervision | Dérive, OOD, Prometheus, Grafana | :material-check-circle:{ .green } livré |
| 9 -- CI/CD | 4 GitHub Actions + Dependabot + templates | :material-check-circle:{ .green } livré |
| 10 -- Export | ONNX / TFLite / benchmark latence | :material-check-circle:{ .green } livré |
| 11 -- Documentation | Ce site, fiche modèle, bilingue EN/FR | :material-progress-clock:{ .blue } en cours |
| 12 -- Recette finale | Validation end-to-end + signature | :material-clock-outline:{ .grey } prévu |

---

## Licence

Publié sous [Licence MIT](https://github.com/Momo3972/deepvision-cifar10-classifier/blob/main/LICENSE).

## Auteur

**Mohamed Lamine OULD BOUYA** -- Ingénieur AI/ML  
[GitHub](https://github.com/Momo3972) · [LinkedIn](https://www.linkedin.com/in/mohamed-lamine-ould-bouya)
