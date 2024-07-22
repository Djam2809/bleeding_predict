import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from PIL import Image

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
            background-color: #f0f0f5;
        }
        .stButton>button {
            color: white;
            background: #4CAF50;
            border-radius: 10px;
            padding: 10px 24px;
            border: none;
        }
        .stButton>button:hover {
            background: #45a049;
        }
        .stRadio>label {
            font-size: 20px;
            font-weight: bold;
        }
        .stNumberInput>label {
            font-size: 20px;
            font-weight: bold;
        }
        .image-container {
            display: flex;
            align-items: flex-start;
            justify-content: flex-start;
            padding: 10px;
        }
        .image-container img {
            width: 150px;  /* Ajuster la taille de l'image */
            height: auto;
        }
        .content-container {
            margin-left: 170px; /* Ajustez cette valeur pour laisser de l'espace pour l'image */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Titre de l'application
    st.title("Prédiction du risque d'hémorragie post-transplantation rénale")

    # Ajouter une photo réduite à gauche
    st.markdown('<div class="image-container"><img src="images/kidney.jpg" alt="Kidney Image"></div>', unsafe_allow_html=True)

    # Créer une div pour le contenu principal avec une marge à gauche
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    
    # Formulaire pour entrer les données du patient avec des boutons radio
    sexe = st.radio("Sexe", options=["Masculin", "Féminin"], index=0, format_func=lambda x: x.capitalize())
    anticoag = st.radio("Anticoagulation", options=["Non", "Oui"], index=0, format_func=lambda x: x.capitalize())
    donneur = st.radio("Type de donneur", options=["Décédé", "Vivant"], index=0, format_func=lambda x: x.capitalize())
    age = st.number_input("Âge", min_value=0, max_value=100, value=30, step=1)

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
    st.write(f"Probabilité de complications hémorragiques : {probability:.4f}")

    # Utiliser le cutoff de 0.55 pour la prédiction
    cutoff = 0.55
    new_patient_pred = (new_patient_pred_proba >= cutoff).astype(int)
    pred_class = "Risque" if new_patient_pred[0] == 1 else "Pas de risque"

    st.write(f"Classe prédite avec un cutoff de {cutoff} : {pred_class}")

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
                {'range': [0.3333, 0.5], 'color': 'orange'},
                {'range': [0.5, 1], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': cutoff
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="lavender",
        font={'color': "darkblue", 'family': "Arial"}
    )

    st.plotly_chart(fig)
    
    # Fermer la div contenant le contenu principal
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
