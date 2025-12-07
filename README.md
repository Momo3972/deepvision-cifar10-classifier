# Classification d'Images CIFAR-10 : Benchmark Deep Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Transfer%20Learning-red?style=for-the-badge&logo=keras)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Terminé-green?style=for-the-badge)

Ce projet présente un pipeline complet de Vision par Ordinateur (Computer Vision) appliqué au dataset académique CIFAR-10. Il compare rigoureusement trois architectures de réseaux de neurones pour démontrer la supériorité des techniques modernes de Transfer Learning et de Data Augmentation.

---

## Objectifs Contexte

L'objectif est de classifier des images de 32x32 pixels en 10 catégories mutuellement exclusives (Avion, Voiture, Chat, Chien, etc.). La démarche suit une progression logique, du modèle le plus simple au plus complexe, pour illustrer les gains de performance.

### Architectures comparées
1.  **MLP (Baseline) :** Perceptron Multicouche pour établir un score de référence
2.  **CNN Custom (From Scratch) :** Architecture convolutive de type VGG (Conv2D + MaxPooling + BatchNormalization)
3.  **EfficientNetB0 (État de l'Art) :** Utilisation de Transfer Learning (poids ImageNet) combiné au Fine-Tuning et à la Data Augmentation

---

## Méthodologie & Stack Technique

Ce projet respecte les standards MLOps et les bonnes pratiques de Data Science :

* **Data Engineering :**
    * Split rigoureux **80% Train / 20% Test** avec stratification (respect de la distribution des classes)
    * Normalisation des données et encodage One-Hot des labels
    * Pipeline de **Data Augmentation** (Rotation, Zoom, Contraste, Flip) intégré au modèle final pour réduire le sur-apprentissage
* **Optimisation :**
    * Utilisation de l'optimiseur **Adam**
    * Callbacks pour le pilotage : `EarlyStopping` (arrêt précoce) et `ReduceLROnPlateau` (ajustement du taux d'apprentissage).
* **Évaluation :**
    * Analyse des courbes d'apprentissage (Loss/Accuracy)
    * Matrices de confusion et Rapports de classification (F1-Score)

---

## Organisation du Projet

Voici comment est structuré le dépôt :

```text
deepvision-cifar10-classifier/
│
├── models/
│   └── best_model_efficientnet_aug.h5   # Le modèle final entraîné (93% acc)
│
├── notebooks/
│   └── Projet_Vision_CIFAR10.ipynb      # Le code complet (EDA, Training, Eval)
│
├── app.py                               # Application de démonstration (Streamlit)
├── requirements.txt                     # Liste des dépendances Python
└── README.md                            # Documentation du projet

---

## Résultats et Analyse

Les performances ont été évaluées sur le jeu de test (données jamais vues durant l'entraînement) :

| Modèle         | Architecture              | Technique            | Accuracy (Test) | Notes |
|----------------|---------------------------|----------------------|-----------------|-------|
| **MLP**         | Dense (Fully Connected)   | Baseline             | ~48%            | — |
| **CNN Custom**  | From Scratch              | ~75%                 | ~75%            | Bonne détection des formes, mais limité par la taille du dataset. |
| **EfficientNetB0** | TL + Augmentation     | 93% 🚀               | 93%             | Performance État de l'Art. |

---

## Impact de la Data Augmentation

L'ajout de transformations aléatoires a réduit les confusions entre classes morphologiquement proches, améliorant nettement la robustesse du modèle

| Classe | F1-Score |
|--------|----------|
| Chat 🐱 | ↑ 0.87 |
| Chien 🐶 | ↑ 0.90 |
| Oiseau 🐦 | ↑ 0.93 |

**Note :** Une matrice de confusion détaillée est disponible à la fin du notebook pour visualiser ces résultats

---

## Installation et Utilisation

Vous pouvez tester le modèle directement sur votre machine via l'interface graphique

### 1. Cloner le projet
```bash
git clone https://github.com/VOTRE_NOM_UTILISATEUR/deepvision-cifar10-classifier.git
cd deepvision-cifar10-classifier

### 2. Installer les dépendances

Il est recommandé d'utiliser un environnement virtuel

pip install -r requirements.txt

### 3. Lancer l'application de démo

streamlit run app.py

Une page web s'ouvrira automatiquement. Vous pourrez y glisser n'importe quelle image (d'un avion, d'un chat, etc.) et voir l'IA la classifier en temps réel avec son indice de confiance.

👤 Auteur
Projet réalisé par : Mohamed Lamine OULD BOUYA
Data Scientist / Ingénieur Deep Learning
[Lien Portfolio] : https://github.com/Momo3972/Portfolio-Data-IA

Dernière mise à jour : Décembre 2025 - Propulsé par TensorFlow & Streamlit
