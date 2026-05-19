# Architecture

Cette page donne une vue synthétique de l'architecture runtime du
projet : la chaîne d'entraînement, le service d'inférence, la
supervision de dérive et l'export multi-runtime.

Pour des walkthroughs orientés tâches, voir les
[tutoriels](tutorials/index.md). Pour la liste exhaustive des modules
et leurs interfaces, voir la [référence API](reference/index.md).

!!! info "Documentation en cours de traduction"
    La version détaillée de cette page (diagramme système, arbre des
    sources, topologie Docker, flux de données et quality gates) est
    disponible pour l'instant uniquement en anglais. La traduction
    française est en cours pour une prochaine version.

[:material-arrow-right: Lire la version anglaise complète](../architecture/){ .md-button .md-button--primary }

## Aperçu rapide

Le projet est structuré en huit packages sous `src/deepvision/` :

| Package | Rôle |
|---|---|
| `config`, `constants`, `utils` | Configuration centrale, constantes, logging et seeds |
| `data` | Loader CIFAR-10 reproductible, preprocessing, augmentation |
| `models` | MLP / CNN / EfficientNet + registry |
| `training` | `run_training` + utils MLflow + callbacks Keras |
| `evaluation` | Métriques, calibration, robustesse, Grad-CAM, benchmark |
| `serving` | App FastAPI + InferenceEngine + schemas Pydantic |
| `monitoring` | Détection dérive Wasserstein + OOD énergie + exporter Prometheus |
| `export` | Conversion ONNX / TFLite + benchmark multi-runtime (Phase 10) |

La stack Docker complète boote six services (`api`, `streamlit`,
`mlflow`, `prometheus`, `grafana`, `drift-monitor`) sur un réseau
privé avec healthchecks.
