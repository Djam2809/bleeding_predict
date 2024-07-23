import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from PIL import Image
import os

def main():
    # Charger le modèle calibré et les objets nécessaires (imputer et scaler)
    clf_isotonic = joblib.load('clf_isotonic.pkl')
    imputer = joblib.load('imputer.pkl')
    scaler = joblib.load('scaler.pkl')

    # Définir le style CSS pour l'application
    st.markdown(
        """
        <style>
        .main {
            background-color: #F0F8FF;
            padding: 20px;
        }
        .stButton>button {
            color: white;
            background: #4CAF50;
            border-radius: 10px;
            padding: 10px 24px;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: #45a049;
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        }
        .stRadio>label, .stNumberInput>label {
            font-size: 18px;
            font-weight: bold;
            color: #2C3E50;
        }
        .content-container {
            margin-left: 170px;
            background-color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .variable-container {
            background-color: #F8F9FA;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 1px solid #E9ECEF;
        }
        .result-container {
            background-color: #E9ECEF;
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
            border: 2px solid #4CAF50;
        }
        h1 {
            color: #2C3E50;
            text-align: center;
            margin-bottom: 30px;
        }
        h3 {
            color: #34495E;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Titre de l'application
    st.title("Prédiction du risque d'hémorragie post-transplantation rénale")

    # Ajouter une photo réduite à gauche
    try:
        image_path = 'images/kidney.jpg'
        image = Image.open(image_path)
        st.image(image, use_column_width=True, caption='Save your Kidney (by DE-2024)')
    except FileNotFoundError:
        st.error(f"Le fichier image '{image_path}' est introuvable dans le répertoire 'images'.")
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'image : {e}")

    # Créer une div pour le contenu principal avec une marge à gauche
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    
    # Formulaire pour entrer les données du patient avec des boutons radio
    with st.container():
        st.markdown('<div class="variable-container">', unsafe_allow_html=True)
        sexe = st.radio("Sexe", options=["Masculin", "Féminin"], index=0, format_func=lambda x: x.capitalize())
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="variable-container">', unsafe_allow_html=True)
        anticoag = st.radio("Anticoagulation", options=["Non", "Oui"], index=0, format_func=lambda x: x.capitalize())
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="variable-container">', unsafe_allow_html=True)
        donneur = st.radio("Type de donneur", options=["Décédé", "Vivant"], index=0, format_func=lambda x: x.capitalize())
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="variable-container">', unsafe_allow_html=True)
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
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.subheader("Résultat de la prédiction")
    probability = new_patient_pred_proba[0]
    st.write(f"Probabilité de complications hémorragiques : {probability:.4f}")

    # Utiliser le cutoff de 0.55 pour la prédiction
    cutoff = 0.55
    new_patient_pred = (new_patient_pred_proba >= cutoff).astype(int)
    pred_class = "Risque" if new_patient_pred[0] == 1 else "Pas de risque"

    st.write(f"Classe prédite avec un cutoff de {cutoff} : {pred_class}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Afficher un tachymètre avec Plotly
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.1666], 'color': 'green'},
                {'range': [0.1666, 0.3333], 'color': 'lightgreen'},
                {'range': [0.3333, 0.55], 'color': 'orange'},
                {'range': [0.55, 1], 'color': 'red'}
            ]
        }
    ))

    fig.update_layout(
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )

    st.plotly_chart(fig)
    
    # Fermer la div contenant le contenu principal
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()