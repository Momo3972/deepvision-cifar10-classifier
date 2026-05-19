# Entraînement

Tutoriel pour entraîner un modèle avec tracking MLflow.

!!! info "Disponible en anglais"
    Le tutoriel complet (trois familles de modèles, recommandations
    d'hyperparamètres, exécution sur Colab, pièges courants) est
    actuellement disponible uniquement en anglais. La traduction
    française intégrale est en cours.

[:material-arrow-right: Lire le tutoriel complet (anglais)](../training/){ .md-button .md-button--primary }

## Commande de référence

Run d'entraînement smoke (~30 s sur CPU) :

```bash
python -m deepvision train --model efficientnet --quick
```

Entraînement complet (10 époques + 5 fine-tuning) :

```bash
python -m deepvision train \
    --model efficientnet \
    --epochs 10 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --fine-tune-epochs 5 \
    --fine-tune-lr 1e-5 \
    --seed 42
```

## Inspecter les runs

```bash
mlflow ui --backend-store-uri ./mlruns
# -> http://localhost:5000
```

Chaque run logge : params (config, dataset hash, git SHA), métriques
(loss/accuracy par époque + test final), artefacts (`.keras`, matrice
de confusion, classification report).
