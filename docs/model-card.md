# Model card -- EfficientNetB0 transfer-learning on CIFAR-10

This model card follows the
[Hugging Face Hub template](https://huggingface.co/docs/hub/model-cards)
so it slots cleanly into model registries and ML observability tools.

## Model details

- **Model name:** `efficientnet_b0_transfer`
- **Version:** `0.11.0`
- **Date:** 2026-05
- **Developed by:** Mohamed Lamine OULD BOUYA (
  [GitHub](https://github.com/Momo3972))
- **License:** [MIT](https://opensource.org/licenses/MIT)
- **Repository:** <https://github.com/Momo3972/deepvision-cifar10-classifier>
- **Architecture:** EfficientNetB0 backbone (ImageNet-pretrained,
  ~4M parameters) + GlobalAveragePooling2D + Dropout(0.2) + Dense(10,
  softmax).
- **Framework:** Keras 3.14 on TensorFlow 2.21 (CPU). Trained with
  the Adam optimiser.
- **Languages:** *not applicable* (vision model).
- **Related resources:** the audit document
  `Audit_DeepVision_CIFAR10.docx` at the repo root details the 13-phase
  industrial refactor roadmap of which this model is the deliverable
  of phases 3-4.

## Intended use

### Primary intended uses

- **Educational / demonstration.** Show what a production-grade
  pipeline looks like around a small computer-vision model: training,
  serving, monitoring, exporting, and CI/CD.
- **Benchmark target.** Compare runtime alternatives (Keras vs TF
  SavedModel vs ONNX Runtime vs TFLite) on a small reproducible
  workload.
- **Portfolio piece.** Showcase MLOps skills to recruiters and
  collaborators.

### Primary intended users

- ML engineers learning end-to-end production patterns.
- Hiring managers reviewing the codebase as part of a job application.
- Students of the audit's industrial-refactor curriculum.

### Out-of-scope use cases

- **Safety-critical decisions.** This is a small CIFAR-10 model
  trained for ~15 minutes of CPU time. Do not deploy it where a
  wrong prediction has real-world consequences (autonomous vehicles,
  medical diagnosis, content moderation at scale, ...).
- **Detecting objects outside the 10 CIFAR-10 classes** (airplane,
  automobile, bird, cat, deer, dog, frog, horse, ship, truck).
  The model will return one of those 10 labels for **any** input,
  including pure noise or unrelated objects -- use the bundled OOD
  detector to flag those (see
  [`deepvision.monitoring.ood`][deepvision.monitoring.ood]).
- **High-resolution images.** Inputs are resized to 32x32 before
  inference. Detail finer than that pixel grid is irrelevant by
  construction.

## Bias, risks, and limitations

- **Dataset bias.** CIFAR-10 is a *small, balanced* dataset of
  32x32 images. The cat / dog / horse classes contain mostly
  European-style pets and farm animals; expect lower accuracy on
  visually distinct populations.
- **Spurious correlations.** Background colour is a strong cue in
  CIFAR-10 (boats appear on blue water, frogs on green, ...). The
  Grad-CAM bundled with the project occasionally highlights
  background pixels for high-confidence predictions, which is a
  classic CIFAR-10 footgun rather than a defect of this model.
- **Resolution limit.** At 32x32, fine-grained distinctions (cat vs
  dog breed, ship vs boat, automobile vs truck variant) are not
  recoverable. Expect ~5 % of test errors in those confusion
  categories.
- **Adversarial robustness.** The model has *no* adversarial
  training. A 4/255-FGSM attack on the test set drops accuracy from
  ~0.89 to ~0.30 (see
  [`deepvision.evaluation.robustness`][deepvision.evaluation.robustness]).

### Recommendations

- Always pair the model with the bundled drift exporter and OOD
  detector when deploying. The combination catches input-distribution
  shift early enough to avoid surprises in production.
- For real CIFAR-10-like decisions, retrain on your own data, not
  on this artefact.

## How to get started

```python
import tensorflow as tf
from deepvision.serving.inference import InferenceEngine
from pathlib import Path

engine = InferenceEngine(model_path=Path("models/efficientnet_best.keras"))
engine.load()

image_batch = ...  # shape (1, 32, 32, 3), dtype float32, range [0, 255]
predictions, latency_ms = engine.predict(image_batch, top_k=3)

for class_index, class_name, probability in predictions:
    print(f"{class_name}: {probability:.2%}")
```

For a REST API instead of in-process inference, see the
[serving tutorial](tutorials/serving.md).

## Training details

### Training data

- **Source:** the canonical CIFAR-10 dataset
  ([Krizhevsky 2009](https://www.cs.toronto.edu/~kriz/cifar.html)).
- **Split:** 60 000 images concatenated then split **80 / 20** with
  `train_test_split(random_state=42, stratify=y)` -- the same
  split is shared by every model family for fair comparison.
  - Train: 48 000 images
  - Test: 12 000 images
- **Hash:** every training run logs a SHA-256 fingerprint of the
  `(images, labels)` pair to MLflow. Comparing two hashes confirms
  two runs saw the same data.
- **Preprocessing:** resize 32x32 -> 224x224 (EfficientNet input
  size), normalisation via the bundled
  `tf.keras.applications.efficientnet.preprocess_input`.

### Training procedure

- **Stage 1 -- frozen base:** 10 epochs, `lr=1e-3`, Adam, batch_size 64.
- **Stage 2 -- fine-tuning:** 5 epochs, last conv block unfrozen,
  `lr=1e-5`, Adam.
- **Augmentation:** random horizontal flip, ±10 % zoom, ±10 % shift,
  ±5° rotation -- all applied on the fly via Keras preprocessing
  layers so the model never sees the same augmented image twice.
- **Hardware:** Google Colab T4 GPU (free tier). Total wall-clock:
  ~12 minutes for the full 15 epochs.
- **Seed:** 42 (propagated to Python, NumPy, TensorFlow, Keras and
  the train/test split).

### Speeds, sizes, times

| Metric | Value |
|---|---|
| Parameters | 4,061,489 (4M) |
| FP32 model size | 16 MB (`.keras` format) |
| INT8 TFLite size | 4.2 MB (~4x smaller) |
| Training time (15 ep, T4 GPU) | ~12 min |
| Inference p95, CPU (Keras) | ~24 ms / image |
| Inference p95, CPU (ONNX Runtime) | ~10 ms / image |
| Inference p95, CPU (TFLite INT8) | ~5 ms / image |

## Evaluation

### Test data

The held-out 12 000-image test split described above.

### Metrics

| Metric | Value |
|---|---|
| **Accuracy** | 0.89 |
| **Macro F1** | 0.89 |
| **Weighted F1** | 0.89 |
| **Top-5 accuracy** | 0.99 |
| **Expected Calibration Error (ECE)** | 0.043 (before temperature scaling), 0.018 (after) |

Per-class accuracy stays within 0.85-0.93 -- no class is dramatically
under-represented in errors.

### Results breakdown

The model confuses these pairs most often (test set, 100 most likely
confusions per pair):

| Pair | Confusion rate |
|---|---|
| `cat` <-> `dog` | 4.2 % |
| `automobile` <-> `truck` | 3.8 % |
| `bird` <-> `frog` | 1.9 % |
| `deer` <-> `horse` | 1.4 % |

These pairs share visual cues at 32x32 resolution and are essentially
the model's accuracy ceiling.

## Environmental impact

| Metric | Value |
|---|---|
| Compute | Google Colab T4 GPU (free tier) |
| Training duration | ~12 minutes |
| Carbon emitted (estimated via [MLCO2 calculator](https://mlco2.github.io/impact/)) | ~0.03 kg CO2eq |

The carbon footprint of a single training run is roughly equivalent
to driving a passenger car for 150 metres. The CI pipeline runs full
training only on tagged releases (otherwise it stays on the
`--quick` smoke profile), so the total compute spend stays modest.

## Technical specifications

### Model architecture

```
Input(shape=(32, 32, 3))
-> Resizing(224, 224)
-> efficientnet.preprocess_input
-> EfficientNetB0(weights="imagenet", include_top=False)
-> GlobalAveragePooling2D()
-> Dropout(rate=0.2)
-> Dense(units=10, activation="softmax")
```

### Inputs and outputs

- **Input shape:** `(batch_size, 32, 32, 3)`, dtype `float32`,
  pixel range `[0, 255]` (preprocessing is internal to the model).
- **Output shape:** `(batch_size, 10)`, dtype `float32`, softmax
  probabilities summing to 1 per row.
- **Class order:** matches the `CLASS_NAMES_EN` tuple in
  [`deepvision.constants`][deepvision.constants] --
  `[airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]`.

### Hardware and software

- Trained on a single Google Colab T4 GPU (16 GB VRAM).
- Serves comfortably on a 2-vCPU / 4 GB RAM machine.
- Built with TensorFlow 2.21, Keras 3.14, Python 3.12.

## Citation

If you use this model or the surrounding pipeline in academic work,
please cite as:

```bibtex
@software{ouldbouya2026deepvision,
    author = {Ould Bouya, Mohamed Lamine},
    title = {DeepVision -- CIFAR-10: Industrial Computer Vision pipeline},
    year = {2026},
    version = {0.11.0},
    url = {https://github.com/Momo3972/deepvision-cifar10-classifier},
}
```

## Contact

For bug reports use the
[GitHub issue tracker](https://github.com/Momo3972/deepvision-cifar10-classifier/issues).
For other inquiries, reach out via
[LinkedIn](https://www.linkedin.com/in/mohamed-lamine-ould-bouya).
