# Supervision

Tutoriel pour la stack de supervision Phase 8 : détection de dérive,
scoring out-of-distribution, exporter Prometheus, dashboards Grafana.

!!! info "Disponible en anglais"
    Le tutoriel complet (capture du baseline, calibration du seuil
    OOD, tuning des règles d'alerte) est actuellement disponible
    uniquement en anglais. La traduction française intégrale est en
    cours.

[:material-arrow-right: Lire le tutoriel complet (anglais)](../monitoring/){ .md-button .md-button--primary }

## Ce que la stack mesure

| Signal | Indication | Algorithme |
|---|---|---|
| **Score de dérive** | La distribution d'entrée live s'est éloignée de l'entraînement | Distance Wasserstein-1D par dim d'embedding, agrégée |
| **Ratio OOD** | Fraction des inputs n'appartenant à aucune classe d'entraînement | Score énergie (Liu et al. 2020) seuillé |
| **Taux de requêtes** | Trafic dans l'API | Compteur Prometheus `deepvision_predictions_total` |
| **Latence p95** | Latence de queue de `/predict` | Histogramme `deepvision_inference_latency_seconds` |

## Démarrer la stack

```bash
docker compose up -d prometheus grafana drift-monitor
```

Puis :
- **Prometheus** : <http://localhost:9090>
- **Grafana** : <http://localhost:3000> (login `admin` / `admin`)
- **Drift exporter** : <http://localhost:9091/metrics>

## Capturer un baseline

Voir le script Python dans le tutoriel anglais pour capturer 5 000+
embeddings de référence depuis le training set et les persister dans
`models/baseline.npz`.
