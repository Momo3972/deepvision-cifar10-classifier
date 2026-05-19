# Monitoring

This tutorial covers the Phase 8 monitoring stack: drift detection on
the model's penultimate-layer embeddings, energy-based
out-of-distribution scoring, and the Prometheus / Grafana plumbing
that surfaces both as alerts and dashboards.

## What the stack measures

| Signal | What it tells you | Algorithm |
|---|---|---|
| **Drift score** | The live input distribution has moved away from training | 1D Wasserstein distance per embedding dim, aggregated |
| **OOD ratio** | The fraction of live inputs that don't belong to any training class | Energy score (Liu et al. 2020) thresholded |
| **Request rate** | Traffic going through the API | Prometheus counter `deepvision_predictions_total` |
| **Latency p95** | Tail latency of `/predict` | `deepvision_inference_latency_seconds` histogram |

Drift and OOD are computed offline by the dedicated `drift-monitor`
service on a configurable interval; rate and latency come straight
from the FastAPI Prometheus middleware.

## Capture a baseline

The drift monitor compares **live** embeddings against a frozen
**baseline** captured at training time. The audit recommends 5 000+
samples for a stable reference.

```python
from pathlib import Path
import tensorflow as tf

from deepvision.data.loader import load_cifar10
from deepvision.monitoring.baseline import (
    Baseline,
    extract_embeddings,
)

# 1. Load your trained model.
model = tf.keras.models.load_model("models/efficientnet_best.keras")

# 2. Pull 5 000 random training images for the baseline.
split = load_cifar10()
baseline_images = split.x_train[:5000]

# 3. Extract penultimate-layer activations.
embeddings = extract_embeddings(model, baseline_images)

# 4. Persist as a Baseline dataclass on disk.
Baseline(embeddings=embeddings, n_samples=5000).save(Path("models/baseline.npz"))
```

The resulting `models/baseline.npz` is small (~25 MB for EfficientNetB0's
1280-d embeddings on 5 000 samples) and is the only artefact the
drift exporter needs.

## Launch the drift exporter

```bash
python -m deepvision drift-monitor \
    --port 9091 \
    --interval 60 \
    --baseline ./models/baseline.npz \
    --ood-threshold -2.0
```

The exporter:

1. Loads the baseline at boot.
2. Every `--interval` seconds, simulates a batch of live inputs (in
   production this would be wired into your serving traffic) and
   computes the Wasserstein-1D distance per embedding dimension
   against the baseline.
3. Aggregates the per-dim distances into `mean`, `p95` and `max`,
   exposed as Prometheus gauges:

   ```
   deepvision_drift_score{aggregation="mean"} 0.42
   deepvision_drift_score{aggregation="p95"}  1.12
   deepvision_drift_score{aggregation="max"}  2.31
   ```

4. Computes the energy score on every live sample and surfaces the
   fraction above `--ood-threshold` as `deepvision_ood_ratio`.

Smoke run without a baseline (synthetic data, useful for CI):

```bash
python -m deepvision drift-monitor --port 9091 --interval 30
```

A warning is logged so nobody believes the readings.

## Prometheus + Grafana

The `docker-compose.yml` stack wires everything together. Bring it
up:

```bash
docker compose up -d prometheus grafana drift-monitor
```

Then:

- **Prometheus** -- <http://localhost:9090>. The rules in
  `monitoring/alerts.yml` give you eight pre-canned alerts (drift
  high/critical, OOD high/critical, API error rate, API latency
  p95/p99, drift exporter down).
- **Grafana** -- <http://localhost:3000> (login `admin` / `admin`,
  which it prompts you to change on first login). The
  `monitoring/grafana/dashboards/deepvision.json` dashboard is
  auto-provisioned and surfaces all six headline panels:

| Panel | Source | Why it matters |
|---|---|---|
| Drift score (mean / p95 / max) | `deepvision_drift_score` | Spot input-distribution shift |
| OOD ratio | `deepvision_ood_ratio` | Detect unknown classes / corrupt inputs |
| Predictions per second | `rate(deepvision_predictions_total[1m])` | Capacity planning |
| Latency p50 / p95 / p99 | `deepvision_inference_latency_seconds` | SLA monitoring |
| Active alerts | `ALERTS` | Single status row |
| Top-K predicted classes | `topk(5, sum by (class_name) (rate(deepvision_predictions_total[5m])))` | Sanity-check the model is actually predicting the right classes |

## Calibrating the OOD threshold

The default `--ood-threshold -2.0` was chosen on CIFAR-10 with
EfficientNetB0. For other models, **measure first**:

1. Pass the baseline images through your model to compute energy
   scores -- those define the "in-distribution" energy distribution.
2. Pass a sample of confidently OOD images (different dataset
   altogether: SVHN, ImageNet-O, ...) through the same model.
3. Pick a threshold that separates the two distributions with
   reasonable precision / recall on your validation cut.

The
[`deepvision.monitoring.ood.energy_score`][deepvision.monitoring.ood.energy_score]
function exposes the raw energy values so you can plot the histograms
yourself.

## Tuning the alerting rules

`monitoring/alerts.yml` defines eight rules. The ones you'll likely
want to revisit:

- `DeepVisionDriftHigh` -- fires when `mean` drift > 1.5 for 5
  minutes. Lower to 1.0 if you want to be alerted earlier, raise to
  2.0 if you get false positives.
- `DeepVisionDriftCritical` -- fires when `mean` drift > 3.0 for 1
  minute. Pages an oncall in our pretend setup.
- `DeepVisionApiLatencyP95High` -- fires when p95 latency exceeds
  500 ms for 5 minutes. Bump for batch endpoints.

Edit the file, then reload Prometheus:

```bash
docker compose restart prometheus
```

## Next steps

- Hardware-aware deployment: [Export & benchmark](export.md).
- Operator handbook: see [Contributing](../contributing.md) for the
  on-call workflow.
