import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from PIL import Image
import os
import hashlib

# Fonction pour hacher les mots de passe
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Fonction pour vérifier les mots de passe hachés
def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# Dictionnaire des utilisateurs (à remplacer par une base de données sécurisée en production)
users = {
    "user1": make_hashes("password1"),
    "user2": make_hashes("password2")
}

# Fonction de connexion
def login():
    st.subheader("Connexion")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type='password')
    if st.button("Se connecter"):
        if username in users:
            hashed_pswd = users[username]
            if check_hashes(password, hashed_pswd):
                st.success(f"Connecté en tant que {username}")
                st.session_state.logged_in = True
                st.session_state.username = username
            else:
                st.warning("Nom d'utilisateur/Mot de passe incorrect")
        else:
            st.warning("Utilisateur non reconnu")
    
    # Ajout du message pour contacter Djamel ELARIBI
    st.markdown("""
    <div style='font-size: 14px; color: #555; text-align: center; margin-top: 30px;'>
    To get your access, please contact Djamel ELARIBI at djamel_elaribi@hotmail.fr
    </div>
    """, unsafe_allow_html=True)

def get_smiley(probability):
    if probability <= 0.1666:
        return "😄", "green"   # Très heureux
    elif probability <= 0.3333:
        return "🙂", "lightgreen"  # Heureux
    elif probability <= 0.55:
        return "😐", "orange" # Neutre
    else:
        return "😟", "red"  # Inquiet

def main():
    # Vérifier si l'utilisateur est connecté
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login()
    else:
        # Charger le modèle calibré et les objets nécessaires (imputer et scaler)
        clf_isotonic = joblib.load('clf_isotonic.pkl')
        imputer = joblib.load('imputer.pkl')
        scaler = joblib.load('scaler.pkl')

        # Définir le style CSS pour l'application
        st.markdown(
            """
            <style>
            .main {
                background-color: #FFFFFF;
            }
            .stButton>button {
                color: white;
                background: #4CAF50;
                border-radius: 5px;
                padding: 8px 16px;
                border: none;
            }
            .stButton>button:hover {
                background: #45a049;
            }
            .stRadio>label, .stNumberInput>label {
                font-size: 16px;
                font-weight: bold;
            }
            .variables-container {
                background-color: #f0f0f0;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .centered {
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Titre de l'application (centré)
        st.markdown("<h1 style='text-align: center;'>Hemorrhage risk prediction in pediatric kidney transplant recipient</h1>", unsafe_allow_html=True)

        # Ajouter l'image sous le titre (centrée)
        try:
            image_path = 'images/kidney.jpg'
            image = Image.open(image_path)
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.image(image, width=200, caption='Save your Kidney (by DE-2024)', use_column_width=True)
        except FileNotFoundError:
            st.error(f"Le fichier image '{image_path}' est introuvable dans le répertoire 'images'.")
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'image : {e}")

        # Formulaire pour entrer les données du patient avec des boutons radio
        st.markdown('<div class="variables-container">', unsafe_allow_html=True)
        sexe = st.radio("Sexe", options=["Masculin", "Féminin"], index=0, format_func=lambda x: x.capitalize())
        anticoag = st.radio("Anticoagulation", options=["Non", "Oui"], index=0, format_func=lambda x: x.capitalize())
        donneur = st.radio("Type de donneur", options=["Décédé", "Vivant"], index=0, format_func=lambda x: x.capitalize())
        age = st.number_input("Âge", min_value=0, max_value=100, value=17, step=1)
        st.markdown('</div>', unsafe_allow_html=True)

        # Convertir les données d'entrée
        sexe = 1 if sexe == "Masculin" else 0
        anticoag = 1 if anticoag == "Oui" else 0
        donneur = 1 if donneur == "Vivant" else 0

        # Créer un DataFrame pour les nouvelles données du patient
        new_patient_data = {
            'sexe': [sexe],
            'anticoag': [anticoag],
            'donneur': [donneur],
            'age': [age]
        }
        new_patient_df = pd.DataFrame(new_patient_data)

        # Imputer les éventuelles valeurs manquantes et standardiser les données
        new_patient_imputed = imputer.transform(new_patient_df)
        new_patient_scaled = scaler.transform(new_patient_imputed)

        # Prédire avec le modèle calibré
        new_patient_pred_proba = clf_isotonic.predict_proba(new_patient_scaled)[:, 1]

        # Afficher le résultat de la prédiction
        st.subheader("Résultat de la prédiction")
        probability = new_patient_pred_proba[0]
        smiley, color = get_smiley(probability)
        st.markdown(f"Probabilité de complications hémorragiques : {probability*100:.1f}% {smiley}", unsafe_allow_html=True)

        # Utiliser le cutoff de 0.55 pour la prédiction
        cutoff = 0.55
        new_patient_pred = (new_patient_pred_proba >= cutoff).astype(int)
        pred_class = "Risque" if new_patient_pred[0] == 1 else "Pas de risque"

        st.write(f"Classe prédite avec un cutoff de {cutoff*100:.1f}% : {pred_class}")

        # Afficher un tachymètre avec Plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,  # Convertir en pourcentage
            number={'suffix': "%", 'valueformat': '.1f'},  # Ajouter le suffixe % et limiter à 1 décimale
            title={'text': "Risque de complications", 'font': {'size': 28}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},  # Changer la plage à 0-100
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 16.66], 'color': 'green'},
                    {'range': [16.66, 33.33], 'color': 'yellow'},
                    {'range': [33.33, 55], 'color': 'orange'},
                    {'range': [55, 100], 'color': 'red'}
                ]
            }
        ))

        fig.update_layout(
            title_text=f"Risque de complications <span style='font-size:48px;'>{smiley}</span>",
            title_x=0.5,
            height=500,
            width=600,
            paper_bgcolor="white",
            font={'color': "darkblue", 'family': "Arial"}
        )

        st.plotly_chart(fig, use_container_width=True)

        # Ajouter le disclaimer en bas de la page
        st.markdown("""
        <div style='font-size: 12px; color: gray; text-align: center; margin-top: 50px;'>
        <hr>
        Disclaimer: This ML algorithm was developed using data from 175 kidney transplant patients. 
        It was trained using a VotingClassifier including Logistic Regression, Linear Discriminant Analysis, and Bagging, 
        fine-tuned to obtain the best hyperparameters, and calibrated.(Djamel ELARIBI)
        </div>
        """, unsafe_allow_html=True)

        # Ajouter un bouton de déconnexion
        if st.button("Déconnexion"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.experimental_rerun()

if __name__ == "__main__":
    main()