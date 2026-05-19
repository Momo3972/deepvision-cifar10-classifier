# Export & benchmark

Tutoriel pour la Phase 10 : conversion du `.keras` vers ONNX et
TFLite, puis benchmark de latence sur les quatre runtimes supportés.

!!! info "Disponible en anglais"
    Le tutoriel complet (modes de quantization détaillés, calibration,
    interprétation des résultats, pièges courants) est actuellement
    disponible uniquement en anglais. La traduction française
    intégrale est en cours.

[:material-arrow-right: Lire le tutoriel complet (anglais)](../export/){ .md-button .md-button--primary }

## Export ONNX

```bash
python -m deepvision export onnx \
    --model-path models/efficientnet_best.keras \
    --output models/exports/efficientnet.onnx \
    --opset 17
```

Validation forward automatique : `max_abs_diff < 1e-4` entre la sortie
Keras et la sortie ONNX Runtime sur des inputs aléatoires.

## Export TFLite

```bash
python -m deepvision export tflite \
    --model-path models/efficientnet_best.keras \
    --output models/exports/efficientnet_int8.tflite \
    --quantization int8 \
    --n-samples 200
```

Quatre modes : `dynamic`, `int8` (**défaut**), `int8_strict`, `fp16`.
Les modes Full INT8 calibrent automatiquement sur 200 images
CIFAR-10 du training set.

## Benchmark multi-runtime

```bash
python -m deepvision export benchmark \
    --keras-path models/efficientnet_best.keras \
    --onnx-path models/exports/efficientnet.onnx \
    --tflite-path models/exports/efficientnet_int8.tflite \
    --n-warmup 100 \
    --n-iter 1000 \
    --batch-sizes 1,8,32 \
    --output-csv reports/phase10_benchmark.csv
```

Reporte p50 / p90 / p95 / p99 / mean / std / throughput pour chaque
combinaison `(runtime, batch_size)`. Sur CPU 2-vCPU typique, ONNX
Runtime est ~2x plus rapide que TF SavedModel, et TFLite INT8 encore
2-3x plus rapide sur batch=1.
