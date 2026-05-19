# Tutorials

Step-by-step walkthroughs for the most common operator tasks.

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } **[Training](training.md)**

    ---

    Train a model end-to-end with MLflow tracking, on a CPU or on Colab.
    Hyperparameter recommendations, fine-tuning, and how to inspect the
    runs in the MLflow UI.

-   :material-rocket-launch:{ .lg .middle } **[Serving](serving.md)**

    ---

    Boot the FastAPI service, call `/predict` and `/predict_batch`,
    enable optional API-key auth, and scrape the Prometheus metrics.

-   :material-chart-bell-curve:{ .lg .middle } **[Monitoring](monitoring.md)**

    ---

    Capture a baseline, launch the drift exporter, route Prometheus
    alerts and open the pre-provisioned Grafana dashboard.

-   :material-export:{ .lg .middle } **[Export & benchmark](export.md)**

    ---

    Convert the trained `.keras` artefact to ONNX (opset 17) and TFLite
    (Full INT8), then benchmark p50 / p95 / p99 latency across the four
    supported runtimes.

</div>

Each tutorial is self-contained -- you do not need to read them in
order. They all assume you have followed the
[Getting started](../getting-started.md) page and have a working
virtual environment with the project installed in editable mode.
