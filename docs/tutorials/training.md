# Training

This tutorial covers the three available model families, the
hyperparameters that matter, and how to run a training on Colab when
you do not have a local GPU.

## The three model families

| Model | Architecture | Test accuracy* | Wall-clock (1 epoch, CPU) |
|---|---|---|---|
| `mlp` | 2 hidden layers, 512 units, dropout | ~0.50 | ~30 s |
| `cnn` | 3 conv blocks (BatchNorm + dropout) | ~0.78 | ~90 s |
| `efficientnet` | EfficientNetB0 frozen + dense head | ~0.86 (10 ep) | ~3 min |
| `efficientnet` + fine-tune | unfreeze last block, lr=1e-5 | ~0.89 (15 ep) | ~5 min |

\* on the canonical 80 / 20 stratified split of CIFAR-10 (60 000 images
merged then split with `random_state=42`).

The MLP exists as a sanity-check baseline -- if your CNN does not beat
it, something is wrong with the pipeline. The CNN is the lightweight
default for CPU experimentation. EfficientNet is the production target.

## Quick smoke training

```bash
python -m deepvision train --model efficientnet --quick
```

The `--quick` flag overrides the dataset to 1 000 images and forces
`--epochs 1`, so the whole run finishes in ~30 seconds even on a
modest CPU. Use it to verify the pipeline end-to-end, not to evaluate
the model.

## Full training

```bash
python -m deepvision train \
    --model efficientnet \
    --epochs 10 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --fine-tune-epochs 5 \
    --fine-tune-lr 1e-5 \
    --seed 42 \
    --experiment "phase-3-baseline"
```

Each argument maps to a field of
[`deepvision.training.train.TrainConfig`][deepvision.training.train.TrainConfig].
What every flag does:

- `--epochs` -- the first training stage with the convolutional base
  frozen. The dense head learns to read EfficientNet's features.
- `--fine-tune-epochs` -- the second stage. The last convolutional
  block is unfrozen and the whole network is trained with a *much*
  smaller learning rate. Skipping this stage costs ~3 accuracy points.
- `--seed` -- propagated to Python's `random`, NumPy, TensorFlow,
  Keras, and the CIFAR-10 stratified split. Two runs with the same
  seed produce bit-identical metrics on a given machine.
- `--experiment` -- MLflow experiment name. Defaults to
  `deepvision-cifar10`. Runs with different `--model` values can share
  the same experiment so you can compare them on a single page.

## Inspecting runs in MLflow

```bash
mlflow ui --backend-store-uri ./mlruns
# -> http://localhost:5000
```

For every run you will see:

- **Params** -- the exhaustive `TrainConfig`, plus the dataset hash,
  the active git revision (via `git rev-parse HEAD`), and the
  active Python version.
- **Metrics** -- training and validation `loss` / `accuracy` per epoch,
  plus the final test `accuracy`, `loss`, `macro_f1` and `weighted_f1`.
- **Artefacts** -- the `.keras` model, a confusion matrix PNG, a
  classification report TXT, and the full TensorFlow training history.

The dataset hash is the secret weapon: if anyone -- including you in
six months -- ever questions a result, you can verify they were
training on the same data by comparing the hash.

## Running on Google Colab

CIFAR-10 + EfficientNetB0 trains comfortably on a free Colab T4 GPU.
A minimal notebook cell that mirrors the local CLI:

```python
!git clone https://github.com/Momo3972/deepvision-cifar10-classifier.git
%cd deepvision-cifar10-classifier
!pip install -e . --quiet

!python -m deepvision train \
    --model efficientnet \
    --epochs 10 \
    --fine-tune-epochs 5 \
    --batch-size 128
```

The MLflow store ends up under `/content/.../mlruns`. To persist it,
mount Google Drive and pass `--experiment` pointing at a Drive path,
or rsync the folder after training.

## Common pitfalls

!!! warning "Don't use a different dataset split per model"
    The original notebook used the native Keras 50 000 / 10 000 split
    for EfficientNet but a custom 80 / 20 split for MLP and CNN. This
    leaked test samples between models and made the comparison table
    invalid. Phase 2 fixed this -- every model now sees the *exact
    same* `train_test_split(random_state=42, stratify=y)`.

!!! warning "Do not skip the fine-tuning stage"
    With the base frozen, EfficientNet's accuracy plateaus around 86 %.
    The last 3 points of accuracy come from fine-tuning the deepest
    convolutional block at `lr=1e-5`. Lower learning rates are
    critical -- the default `1e-3` will destroy the pre-trained
    weights.

!!! tip "Use `--quick` in CI"
    The smoke test workflow runs `deepvision train --quick` to catch
    pipeline regressions without burning real CI minutes on a 5-min
    training.

## Next steps

- Serve your trained model: [Serving tutorial](serving.md).
- Capture a drift baseline: [Monitoring tutorial](monitoring.md).
- Export to ONNX/TFLite: [Export tutorial](export.md).
