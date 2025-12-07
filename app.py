import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Configuration de la page
st.set_page_config(page_title="CIFAR-10 Vision", page_icon="👁️")
st.title("👁️ Reconnaissance d'Images (CIFAR-10)")
st.markdown("Ce modèle utilise **EfficientNetB0** (Transfer Learning) pour identifier 10 types d'objets avec ~93% de précision.")

# 2. Classes (en Français)
CLASS_NAMES = ["Avion", "Voiture", "Oiseau", "Chat", "Cerf",
               "Chien", "Grenouille", "Cheval", "Bateau", "Camion"]

# 3. Chargement du modèle
@st.cache_resource
def load_model():
    # Charge le modèle depuis le dossier 'models'
    return tf.keras.models.load_model('models/best_model_efficientnet_aug.h5')

try:
    with st.spinner('Chargement du modèle IA en cours...'):
        model = load_model()
    st.success("✅ Modèle chargé et prêt à l'emploi !")
except Exception as e:
    st.error(f"Erreur : Impossible de charger le modèle. Vérifiez que le fichier se trouve bien dans 'models/best_model_efficientnet_aug.h5'.\n\nDétail : {e}")

# 4. Interface Utilisateur
st.write("---")
uploaded_file = st.file_uploader("Glissez une image ici (Chat, Avion, etc.)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Affichage
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Image chargée', use_column_width=True)
    
    with col2:
        if st.button('🔍 Lancer la prédiction'):
            with st.spinner('Analyse des pixels...'):
                # Prétraitement
                # EfficientNet attend des pixels 0-255 (pas de division par 255 ici car le modèle le gère en interne via Rescaling)
                img = image.resize((32, 32)) # On remet en 32x32 car c'est l'entrée native de CIFAR
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)

                # Prédiction
                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                
                class_idx = np.argmax(predictions)
                confidence = np.max(predictions)

                # Résultat
                st.markdown(f"### Résultat : **{CLASS_NAMES[class_idx]}**")
                
                # Barre de confiance
                st.progress(int(confidence * 100))
                st.caption(f"Indice de confiance : {confidence*100:.2f}%")
                
                # Afficher les autres probabilités si ce n'est pas sûr à 100%
                if confidence < 0.99:
                    st.write("Autres possibilités :")
                    # On trie les 3 meilleures prédictions
                    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
                    for i in top_3_indices:
                        st.write(f"- {CLASS_NAMES[i]} : {predictions[0][i]*100:.1f}%")