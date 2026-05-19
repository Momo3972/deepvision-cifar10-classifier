# Démarrage rapide

Cette page vous accompagne pour cloner le dépôt, installer le projet
et exécuter chaque commande de premier niveau au moins une fois. Elle
vise une machine vierge avec **Python 3.11 ou 3.12** sous Linux ou
Windows PowerShell.

Si vous voulez juste *essayer* le modèle sans rien installer, la stack
Docke (voir [la section Docker plus bas](#avec-docker)) lance l'API,
la démo Streamlit, MLflow, Prometheus et Grafana en une seule
commande.

## Prérequis

| Outil | Version | Pourquoi |
|---|---|---|
| Python | 3.11 ou 3.12 | matrice CI ; évitez 3.13 -- TensorFlow ne fournit pas encore de wheels |
| `pip` | >= 24.0 | support PEP 668 + install éditable |
| `git` | récent | `git rev-parse` est loggé dans chaque run MLflow |
| Docker (optionnel) | >= 24 | uniquement pour la stack compose multi-services |

Pas besoin de **GPU**. Tout le pipeline tourne sur CPU ; entraîner une
époque EfficientNet sur un CPU laptop moderne prend ~3 minutes en
profil smoke `--quick`.

## Installer depuis les sources

```bash
# 1. Cloner
git clone https://github.com/Momo3972/deepvision-cifar10-classifier.git
cd deepvision-cifar10-classifier

# 2. Créer + activer un venv
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# 3. Installer dépendances runtime + dev
pip install -r requirements.txt -r requirements-dev.txt
pip install -e . --no-deps

# 4. Vérification rapide
python -m deepvision --help
deepvision info
```

La commande `deepvision info` affiche la version du package, les
chemins de configuration résolus, et l'interpréteur Python actif. Si
quelque chose cloche, c'est le premier endroit à regarder.

## Configurer (optionnel)

Tous les paramètres sont exposés via la classe `Settings` dans
[`deepvision.config`][deepvision.config]. Override n'importe quel
champ via une variable d'environnement préfixée `DEEPVISION_` :

```bash
export DEEPVISION_SEED=1234
export DEEPVISION_BATCH_SIZE=128
export DEEPVISION_MODEL_PATH=models/efficientnet_best.keras
```

Un fichier `.env` à la racine du dépôt marche aussi -- voir
`.env.example` pour le template canonique.

## Entraîner votre premier modèle

Un run smoke sur 1 000 images et une époque (~30 s sur CPU) :

```bash
python -m deepvision train --model efficientnet --quick
```

Va effectuer :

1. Télécharger CIFAR-10 (~170 Mo vers `~/.keras/datasets/`).
2. Construire un EfficientNetB0 avec poids aléatoires.
3. Entraîner une époque sur 1 000 images stratifiées.
4. Logger run, métriques et artefacts vers un store MLflow local
   `./mlruns/`.

Ouvrir l'UI MLflow pour inspecter :

```bash
mlflow ui --backend-store-uri ./mlruns
# -> http://localhost:5000
```

Pour un entraînement complet (10 époques + 5 époques de fine-tuning) :

```bash
python -m deepvision train --model efficientnet --epochs 10 --fine-tune-epochs 5
```

Voir le [tutoriel d'entraînement](tutorials/training.md) pour les
recommandations d'hyperparamètres et les astuces pour exécuter sur
Colab.

## Servir le modèle entraîné

```bash
python -m deepvision serve --port 8000
```

Le serveur FastAPI démarre en ~2 s (les imports lourds sont différés)
et expose :

- `GET /health` -- liveness probe
- `GET /meta` -- nom et version du modèle
- `POST /predict` -- image unique, retourne les top-K classes
- `POST /predict_batch` -- jusqu'à 16 images
- `GET /metrics` -- format Prometheus
- `GET /docs` -- Swagger UI interactif

Sans modèle entraîné, l'API sert un EfficientNetB0 à **poids
aléatoires** -- les prédictions n'auront pas de sens mais le chemin
de requête est correct. Pointer vers un vrai artefact avec :

```bash
export DEEPVISION_MODEL_PATH=models/efficientnet_best.keras
python -m deepvision serve
```

## Lancer la démo Streamlit

```bash
python -m deepvision streamlit --port 8501
# -> http://localhost:8501
```

La démo permet de uploader une ou plusieurs images de type CIFAR-10,
voir les top-K classes, et superposer une heatmap Grad-CAM.

## Avec Docker

La stack complète six services est dans `docker-compose.yml`. Démarrer :

```bash
docker compose up -d
```

Puis visiter :

| Service | URL | Note |
|---|---|---|
| FastAPI | <http://localhost:8000/docs> | API d'inférence |
| Streamlit | <http://localhost:8501> | UI démo |
| MLflow | <http://localhost:5000> | runs d'entraînement |
| Prometheus | <http://localhost:9090> | métriques + règles d'alerte |
| Grafana | <http://localhost:3000> | tableaux de bord (login `admin` / `admin`) |
| Drift exporter | <http://localhost:9091/metrics> | scores Wasserstein + OOD |

Pour arrêter : `docker compose down -v`.

## Lancer la suite de tests

```bash
pytest -n auto
```

Vous devriez voir **356+ tests passés** en ~40 s sur un CPU récent.
La suite inclut des vérifications structurelles pour les workflows
GitHub, les artefacts Docker, la stack de supervision, le pipeline
d'export et chaque module interne.

## Étapes suivantes

- Vue d'ensemble architecture : voir [Architecture](architecture.md).
- Tâche spécifique : choisir un [tutoriel](tutorials/index.md).
- Détails du modèle : lire la [Fiche modèle](model-card.md).
- Référence API : voir [Référence API](reference/index.md).
- Contribuer : lire [Contribuer](contributing.md).
