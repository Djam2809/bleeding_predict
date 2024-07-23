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
    st.markdown("<h1 style='text-align: center;'>Prédiction du risque d'hémorragie post-transplantation rénale</h1>", unsafe_allow_html=True)

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
    # Ajouter le disclaimer en bas de la page
    st.markdown("""
    <div style='font-size: 10px; color: gray; text-align: center; margin-top: 50px;'>
    <hr>
    Disclaimer: This ML algorithm was developed using data from 275 transplant patients. 
    It was trained using a VotingClassifier including Logistic Regression, Linear Discriminant Analysis, and Bagging, 
    fine-tuned to obtain the best hyperparameters, and calibrated.(Djamel ELARIBI)
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()