# Classification d’Images CIFAR-10 avec CNN et Transfer Learning (EfficientNetB0)

Ce projet explore la classification d’images issues du dataset CIFAR-10 en utilisant :

- un modèle **CNN construit from scratch** (baseline)
- un modèle **EfficientNetB0** pré-entraîné (Transfer Learning + Fine-Tuning)

L’objectif est de comparer les performances des deux approches et de montrer l’impact du Transfer Learning sur un problème réel de vision par ordinateur.

---

## Dataset : CIFAR-10

Le dataset contient **60 000 images couleur 32×32** réparties en **10 classes** :

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

- 50 000 images pour l’entraînement  
- 10 000 images pour le test  

---

## Structure du projet
deepvision-cifar10-classifier/  
├── .gitignore                          # Fichiers et dossiers ignorés par Git  
├── models/                             # Modèles entraînés sauvegardés  
│   ├── cnn_baseline_cifar10.h5         # Modèle CNN baseline (32×32, entraîné from scratch)  
│   └── efficientnetb0_tl_cifar10.h5     # Modèle EfficientNetB0 (transfer learning + fine-tuning)  
│
├── 01_cifar10_cnn.ipynb                # Notebook Jupyter complet :  
│                                       # - Exploration des données  
│                                       # - Modèle CNN baseline  
│                                       # - Modèle EfficientNetB0 (TL)  
│                                       # - Évaluations, courbes & matrices de confusion  
│
├── README.md                           # Présentation générale du projet (documentation principale)
└── (éventuels futurs fichiers : data/, src/, reports/, etc.)

---

## Modèles développés

### **1️⃣ CNN baseline (entraînement from scratch)**  
- 3 blocs Convolution + MaxPooling  
- Dense + Dropout  
- Accuracy test ≈ **0.70**

### **2️⃣ EfficientNetB0 (Transfer Learning + Fine-Tuning)**  
- Backbone gelé (phase 1)  
- Fine-tuning partiel (phase 2)  
- Accuracy test ≈ **0.95**

**➡️ Gain absolu d’accuracy : +0.24**

---

## Résultats principaux

| Modèle | Test Accuracy | Test Loss |
|--------|--------------|-----------|
| CNN baseline | ~0.70 | ~0.86 |
| EfficientNetB0 TL | ~0.95 | ~0.16 |

✔ Le Transfer Learning surpasse largement le CNN baseline  
✔ Stabilité accrue sur toutes les classes  
✔ Matrices de confusion beaucoup plus propres

---

## Organisation du projet

- `01_cifar10_cnn.ipynb` — Notebook complet contenant :
  - Préparation du dataset
  - Construction du CNN
  - Construction du modèle EfficientNetB0
  - Entraînement en 2 phases
  - Visualisation des courbes
  - Classification report
  - Matrices de confusion
  - Sauvegarde des modèles

- `cnn_baseline_cifar10.h5` — Modèle CNN sauvegardé  
- `efficientnetb0_tl_cifar10.h5` — Modèle TL sauvegardé  

---

## Conclusion

Le projet démontre clairement la puissance du Transfer Learning :

- convergence beaucoup plus rapide  
- accuracy très élevée  
- stabilité inter-classe  
- généralisation nettement améliorée  

Ce pipeline complet sert de base solide pour tout projet de vision basé sur la classification d’images.

---

## Auteur

Projet réalisé par **Mohamed Lamine OULD BOUYA**, Data Scientist
