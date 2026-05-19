# Fiche modèle -- EfficientNetB0 transfer-learning sur CIFAR-10

Cette fiche modèle suit le
[template du Hugging Face Hub](https://huggingface.co/docs/hub/model-cards)
pour s'intégrer proprement aux registres de modèles et outils
d'observabilité ML.

## Détails du modèle

- **Nom du modèle :** `efficientnet_b0_transfer`
- **Version :** `0.11.0`
- **Date :** 2026-05
- **Développé par :** Mohamed Lamine OULD BOUYA (
  [GitHub](https://github.com/Momo3972))
- **Licence :** [MIT](https://opensource.org/licenses/MIT)
- **Dépôt :** <https://github.com/Momo3972/deepvision-cifar10-classifier>
- **Architecture :** Backbone EfficientNetB0 (pré-entraîné ImageNet,
  ~4M paramètres) + GlobalAveragePooling2D + Dropout(0.2) + Dense(10,
  softmax).
- **Framework :** Keras 3.14 sur TensorFlow 2.21 (CPU). Entraîné avec
  l'optimiseur Adam.
- **Langages :** *non applicable* (modèle vision).
- **Ressources liées :** le document d'audit
  `Audit_DeepVision_CIFAR10.docx` à la racine du dépôt détaille la
  feuille de route industrielle en 13 phases dont ce modèle est le
  livrable des phases 3-4.

## Usage prévu

### Usages principaux prévus

- **Pédagogique / démonstration.** Montrer à quoi ressemble un
  pipeline de production autour d'un petit modèle CV : entraînement,
  service, supervision, export et CI/CD.
- **Cible de benchmark.** Comparer les runtimes (Keras vs TF
  SavedModel vs ONNX Runtime vs TFLite) sur une charge reproductible.
- **Pièce de portfolio.** Démontrer des compétences MLOps aux
  recruteurs et collaborateurs.

### Utilisateurs principaux prévus

- Ingénieurs ML apprenant les patterns de production de bout en bout.
- Recruteurs examinant la base de code dans le cadre d'un recrutement.
- Étudiants suivant le curriculum de refonte industrielle de l'audit.

### Usages hors périmètre

- **Décisions critiques pour la sécurité.** C'est un petit modèle
  CIFAR-10 entraîné sur ~15 minutes de CPU. Ne pas déployer là où
  une mauvaise prédiction a des conséquences réelles (véhicules
  autonomes, diagnostic médical, modération à grande échelle, ...).
- **Détecter des objets hors des 10 classes CIFAR-10** (avion,
  automobile, oiseau, chat, cerf, chien, grenouille, cheval, bateau,
  camion). Le modèle retourne une de ces 10 étiquettes pour
  **n'importe quelle** entrée, y compris du bruit pur ou des objets
  sans lien -- utiliser le détecteur OOD intégré pour les filtrer
  (voir [`deepvision.monitoring.ood`][deepvision.monitoring.ood]).
- **Images en haute résolution.** Les entrées sont redimensionnées à
  32x32 avant inférence. Tout détail plus fin que cette grille est
  perdu par construction.

## Biais, risques et limitations

- **Biais du dataset.** CIFAR-10 est un dataset *petit et équilibré*
  d'images 32x32. Les classes chat / chien / cheval contiennent
  majoritairement des animaux de compagnie et de ferme européens ;
  attendez-vous à une précision moindre sur des populations
  visuellement différentes.
- **Corrélations fallacieuses.** La couleur de fond est un indice
  fort dans CIFAR-10 (bateaux sur eau bleue, grenouilles sur vert,
  ...). Le Grad-CAM intégré met occasionnellement en surbrillance des
  pixels de fond pour des prédictions hautement confiantes, un piège
  classique CIFAR-10 plutôt qu'un défaut du modèle.
- **Limite de résolution.** À 32x32, les distinctions fines
  (race de chat vs chien, bateau vs voilier, automobile vs camion)
  sont irrécupérables. Compter ~5 % d'erreurs dans ces confusions.
- **Robustesse adversariale.** Le modèle n'a **aucun** entraînement
  adversarial. Une attaque 4/255-FGSM sur le test fait chuter
  l'accuracy de ~0.89 à ~0.30 (voir
  [`deepvision.evaluation.robustness`][deepvision.evaluation.robustness]).

### Recommandations

- Toujours associer le modèle avec l'exporter de dérive intégré et
  le détecteur OOD lors du déploiement. La combinaison détecte les
  shifts de distribution d'entrée assez tôt pour éviter les
  surprises en production.
- Pour de vraies décisions de type CIFAR-10, réentraîner sur vos
  propres données, pas sur cet artefact.

## Pour commencer

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

Pour une API REST plutôt que de l'inférence in-process, voir le
[tutoriel d'inférence](tutorials/serving.md).

## Détails d'entraînement

### Données d'entraînement

- **Source :** dataset canonique CIFAR-10
  ([Krizhevsky 2009](https://www.cs.toronto.edu/~kriz/cifar.html)).
- **Split :** 60 000 images concaténées puis splittées **80 / 20**
  avec `train_test_split(random_state=42, stratify=y)` -- le même
  split est partagé par toutes les familles de modèles pour une
  comparaison équitable.
  - Train : 48 000 images
  - Test : 12 000 images
- **Hash :** chaque run d'entraînement logge une empreinte SHA-256 de
  la paire `(images, labels)` dans MLflow. Comparer deux hash confirme
  que deux runs ont vu les mêmes données.
- **Préprocessing :** resize 32x32 -> 224x224 (taille d'entrée
  EfficientNet), normalisation via
  `tf.keras.applications.efficientnet.preprocess_input`.

### Procédure d'entraînement

- **Étape 1 -- base gelée :** 10 époques, `lr=1e-3`, Adam, batch_size 64.
- **Étape 2 -- fine-tuning :** 5 époques, dernier bloc conv dégelé,
  `lr=1e-5`, Adam.
- **Augmentation :** flip horizontal aléatoire, zoom ±10 %, shift
  ±10 %, rotation ±5° -- toutes appliquées à la volée via les couches
  Keras de preprocessing pour que le modèle ne voie jamais deux fois
  la même image augmentée.
- **Matériel :** Google Colab T4 GPU (free tier). Total wall-clock :
  ~12 minutes pour les 15 époques complètes.
- **Seed :** 42 (propagée à Python, NumPy, TensorFlow, Keras et le
  split train/test).

### Vitesses, tailles, temps

| Métrique | Valeur |
|---|---|
| Paramètres | 4 061 489 (4M) |
| Taille modèle FP32 | 16 Mo (format `.keras`) |
| Taille TFLite INT8 | 4,2 Mo (~4x plus petit) |
| Temps entraînement (15 ép, T4 GPU) | ~12 min |
| Latence p95, CPU (Keras) | ~24 ms / image |
| Latence p95, CPU (ONNX Runtime) | ~10 ms / image |
| Latence p95, CPU (TFLite INT8) | ~5 ms / image |

## Évaluation

### Données de test

Le split test mis de côté de 12 000 images décrit plus haut.

### Métriques

| Métrique | Valeur |
|---|---|
| **Accuracy** | 0,89 |
| **Macro F1** | 0,89 |
| **Weighted F1** | 0,89 |
| **Accuracy top-5** | 0,99 |
| **Erreur de calibration (ECE)** | 0,043 (avant temperature scaling), 0,018 (après) |

L'accuracy par classe reste dans 0,85-0,93 -- aucune classe n'est
dramatiquement sous-représentée dans les erreurs.

### Détail des résultats

Le modèle confond ces paires le plus souvent (test set, 100
confusions les plus probables par paire) :

| Paire | Taux de confusion |
|---|---|
| `cat` <-> `dog` | 4,2 % |
| `automobile` <-> `truck` | 3,8 % |
| `bird` <-> `frog` | 1,9 % |
| `deer` <-> `horse` | 1,4 % |

Ces paires partagent des indices visuels à 32x32 et constituent
essentiellement le plafond d'accuracy du modèle.

## Impact environnemental

| Métrique | Valeur |
|---|---|
| Compute | Google Colab T4 GPU (free tier) |
| Durée entraînement | ~12 minutes |
| CO2 émis (estimé via [calculatrice MLCO2](https://mlco2.github.io/impact/)) | ~0,03 kg CO2eq |

L'empreinte carbone d'un seul run d'entraînement équivaut à peu près
à parcourir 150 mètres en voiture. Le pipeline CI ne lance l'entraînement
complet que sur les tags de release (sinon il reste sur le profil
smoke `--quick`), donc la dépense totale en compute reste modeste.

## Spécifications techniques

### Architecture du modèle

```
Input(shape=(32, 32, 3))
-> Resizing(224, 224)
-> efficientnet.preprocess_input
-> EfficientNetB0(weights="imagenet", include_top=False)
-> GlobalAveragePooling2D()
-> Dropout(rate=0.2)
-> Dense(units=10, activation="softmax")
```

### Entrées et sorties

- **Shape d'entrée :** `(batch_size, 32, 32, 3)`, dtype `float32`,
  pixels dans `[0, 255]` (preprocessing interne au modèle).
- **Shape de sortie :** `(batch_size, 10)`, dtype `float32`,
  probabilités softmax sommant à 1 par ligne.
- **Ordre des classes :** correspond au tuple `CLASS_NAMES_EN` dans
  [`deepvision.constants`][deepvision.constants] --
  `[airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]`.

### Matériel et logiciel

- Entraîné sur un seul Google Colab T4 GPU (16 Go VRAM).
- Sert confortablement sur une machine 2-vCPU / 4 Go RAM.
- Construit avec TensorFlow 2.21, Keras 3.14, Python 3.12.

## Citation

Si vous utilisez ce modèle ou le pipeline qui l'entoure dans un
travail académique, merci de citer ainsi :

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

Pour les rapports de bug, utiliser le
[tracker GitHub](https://github.com/Momo3972/deepvision-cifar10-classifier/issues).
Pour d'autres questions, contactez via
[LinkedIn](https://www.linkedin.com/in/mohamed-lamine-ould-bouya).
