# Tutoriels

Walkthroughs pas-à-pas pour les tâches les plus courantes.

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } **[Entraînement](training.md)**

    ---

    Entraîner un modèle de bout en bout avec tracking MLflow, sur CPU
    ou sur Colab. Recommandations d'hyperparamètres, fine-tuning, et
    comment inspecter les runs dans l'UI MLflow.

-   :material-rocket-launch:{ .lg .middle } **[Service d'inférence](serving.md)**

    ---

    Démarrer le service FastAPI, appeler `/predict` et `/predict_batch`,
    activer l'auth optionnelle API-key, et scraper les métriques
    Prometheus.

-   :material-chart-bell-curve:{ .lg .middle } **[Supervision](monitoring.md)**

    ---

    Capturer un baseline, lancer l'exporter de dérive, router les
    alertes Prometheus et ouvrir le tableau de bord Grafana
    pré-provisionné.

-   :material-export:{ .lg .middle } **[Export & benchmark](export.md)**

    ---

    Convertir l'artefact `.keras` vers ONNX (opset 17) et TFLite
    (Full INT8), puis benchmarker p50 / p95 / p99 sur les quatre
    runtimes supportés.

</div>

Chaque tutoriel est autonome -- pas besoin de les lire dans l'ordre.
Tous supposent que vous avez suivi la page
[Démarrage rapide](../getting-started.md) et avez un environnement
virtuel fonctionnel avec le projet installé en mode éditable.

!!! info "Tutoriels détaillés en anglais"
    Les pages tutoriels détaillées sont disponibles en anglais. La
    traduction française intégrale est en cours pour une prochaine
    version. Les boutons « Entraînement », « Service d'inférence »,
    etc. ci-dessus pointent vers les pages anglaises complètes.
