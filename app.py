import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from PIL import Image
import os

def get_smiley(probability):
    if probability <= 0.1666:
        return "😄"  # Très heureux
    elif probability <= 0.3333:
        return "🙂"  # Heureux
    elif probability <= 0.55:
        return "😐"  # Neutre
    else:
        return "😟"  # Inquiet

def main():
    # ... (le reste du code reste inchangé jusqu'à la partie de prédiction)

    # Afficher le résultat de la prédiction
    st.subheader("Résultat de la prédiction")
    probability = new_patient_pred_proba[0]
    smiley = get_smiley(probability)
    st.write(f"Probabilité de complications hémorragiques : {probability:.4f} {smiley}")

    # Utiliser le cutoff de 0.55 pour la prédiction
    cutoff = 0.55
    new_patient_pred = (new_patient_pred_proba >= cutoff).astype(int)
    pred_class = "Risque" if new_patient_pred[0] == 1 else "Pas de risque"

    st.write(f"Classe prédite avec un cutoff de {cutoff} : {pred_class}")

    # Afficher un tachymètre avec Plotly
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': f"Risque de complications {smiley}", 'font': {'size': 24}},
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

    # ... (le reste du code reste inchangé)

if __name__ == "__main__":
    main()