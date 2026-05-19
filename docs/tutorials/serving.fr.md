# Service d'inférence

Tutoriel pour démarrer le service FastAPI et appeler ses endpoints.

!!! info "Disponible en anglais"
    Le tutoriel complet (auth API-key, scraping Prometheus, pièges
    déploiement) est actuellement disponible uniquement en anglais.
    La traduction française intégrale est en cours.

[:material-arrow-right: Lire le tutoriel complet (anglais)](../serving/){ .md-button .md-button--primary }

## Démarrer le serveur

```bash
python -m deepvision serve --host 0.0.0.0 --port 8000 --workers 2
```

Avec un modèle entraîné :

```bash
export DEEPVISION_MODEL_PATH=models/efficientnet_best.keras
python -m deepvision serve
```

## Endpoints

| Endpoint | Usage |
|---|---|
| `GET /health` | Liveness probe |
| `GET /meta` | Nom + version du modèle |
| `POST /predict` | Image unique, retourne top-K classes |
| `POST /predict_batch` | Jusqu'à 16 images |
| `GET /metrics` | Format Prometheus |
| `GET /docs` | Swagger UI interactif |

## Exemple d'appel

```bash
curl -X POST http://localhost:8000/predict \
    -F "file=@photo.jpg" \
    -F "top_k=3"
```

Les uploads sont validés par **magic bytes** (JPEG/PNG/WEBP/GIF/BMP/TIFF)
dans
[`deepvision.serving.preprocess.validate_payload_size`][deepvision.serving.preprocess.validate_payload_size].
