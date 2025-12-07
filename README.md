# Classification d’Images CIFAR-10 · Deep Learning & Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Transfer%20Learning-red?style=for-the-badge&logo=keras)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Terminé-green?style=for-the-badge)

Ce projet présente un pipeline complet de Vision par Ordinateur appliqué au dataset académique CIFAR‑10.  
Il compare trois architectures Deep Learning afin d’illustrer les gains obtenus grâce au Transfer Learning et à la Data Augmentation.

---

## Objectifs et contexte

L’objectif est de classifier des images de 32×32 pixels en 10 catégories (Avion, Voiture, Chat, Chien, etc.).  
Le projet suit une progression logique : d’un modèle simple vers un modèle avancé.

### Architectures comparées
1. **MLP (Baseline)** - Perceptron Multicouche pour établir un score de référence  
2. **CNN Custom (From Scratch)** - Architecture convolutive de type VGG (Conv2D + MaxPooling + BatchNormalization)  
3. **EfficientNetB0 (État de l’Art)** - Utilisation de Transfer Learning (poids ImageNet) combiné au Fine-Tuning et à la Data Augmentation

---

## Méthodologie et Stack cechnique

Ce projet respecte les standards MLOps et les bonnes pratiques de Data Science

### Data Engineering
- Split rigoureux 80% Train / 20% Test avec stratification (respect de la distribution des classes)
- Normalisation des données et encodage One-Hot des labels
- Pipeline de Data Augmentation (Rotation, Zoom, Contraste, Flip) intégré au modèle final pour réduire le sur-apprentissage


### Optimisation
- Utilisation de l'optimiseur Adam
- Callbacks pour le pilotage : EarlyStopping (arrêt précoce) et ReduceLROnPlateau (ajustement du taux d'apprentissage)


### Évaluation
- Analyse des courbes d'apprentissage (Loss/Accuracy)
- Matrices de confusion et Rapports de classification (F1-Score)


---

## Organisation du Projet

Voici la structure du dépôt :

```text
deepvision-cifar10-classifier/
│
├── models/
│   └── best_model_efficientnet_aug.h5        # Modèle final entraîné (93% acc)
│
├── notebooks/
│   └── Projet_Vision_CIFAR10.ipynb           # Notebook complet (EDA, Training, Eval)
│
├── app.py                                    # Application Streamlit
├── requirements.txt                           # Dépendances Python
└── README.md                                  # Documentation du projet
```

---

## Résultats et analyse

| Modèle              | Architecture              | Technique            | Accuracy Test | Observations |
|--------------------|---------------------------|----------------------|----------------|-------|
| **MLP**             | Fully Connected           | Baseline             | ~48%           | Limité par la perte de structure spatiale (Flatten) |
| **CNN Custom**      | From Scratch              | From Scratch                   | ~75%           | Bonne détection des formes, mais limité par la taille du dataset |
| **EfficientNetB0**  | TL + Data Augmentation    | Fine‑Tuning          | **93%**     | Performance excellente |

---

## Impact de la Data Augmentation

L'ajout de transformations aléatoires a réduit les confusions entre classes morphologiquement proches, améliorant nettement la robustesse du modèle

| Classe | F1‑Score |
|--------|----------|
| Chat 🐱 | ↑ 0.87 |
| Chien 🐶 | ↑ 0.90 |
| Oiseau 🐦 | ↑ 0.93 |

> Une matrice de confusion complète est disponible dans le notebook.

---

## Installation & Utilisation
Vous pouvez tester le modèle directement sur votre machine via l'interface graphique

### 1. Cloner le projet
```bash
git clone https://github.com/VOTRE_NOM_UTILISATEUR/deepvision-cifar10-classifier.git
cd deepvision-cifar10-classifier
```

### 2. Installer les dépendances
Il est recommandé d'utiliser un environnement virtuel

```bash
pip install -r requirements.txt
```

### 3. Lancer l’application Streamlit
```bash
streamlit run app.py
```

Une interface web s’ouvrira automatiquement.  
Vous pourrez y glisser une image (chat, avion, chien, etc.) pour obtenir une prédiction en temps réel.

---

## Auteur

Projet réalisé par **Mohamed Lamine OULD BOUYA**  
Data Scientist / Ingénieur Deep Learning  
Portfolio : https://github.com/Momo3972/Portfolio-Data-IA

**Dernière mise à jour : Décembre 2025 - Propulsé par TensorFlow & Streamlit**
