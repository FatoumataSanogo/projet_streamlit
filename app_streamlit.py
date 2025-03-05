import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Charger le modèle
model = joblib.load("model.joblib")

# Ajouter une image
st.image("credit_risk_image.jpg", caption="Prédiction du Risque de Crédit")

# Titre de l'application
st.title("Prédiction du Risque de Crédit")

# Sous-titre avec le nom
st.subheader("Développé par Fatoumata SANOGO")

# Entrée utilisateur
age = st.number_input("Âge", min_value=18, max_value=100, value=30)
revenu = st.number_input("Revenu annuel", min_value=1000, max_value=200000, value=50000)
statut_logement = st.selectbox("Statut du logement", ["RENT", "OWN", "MORTGAGE", "OTHER"])
anciennete_emploi = st.number_input("Ancienneté de l'emploi (en années)", min_value=0, max_value=50, value=5)
objet_pret = st.selectbox("Objet du prêt", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
note_credit = st.selectbox("Note de crédit", ["A", "B", "C", "D", "E", "F", "G"])
montant_pret = st.number_input("Montant emprunté", min_value=500, max_value=50000, value=10000)
taux_interet = st.number_input("Taux d'intérêt du prêt", min_value=0.0, max_value=50.0, value=10.0)
pourcentage_revenu_pret = st.slider("Pourcentage du revenu alloué au prêt", 0.0, 1.0, 0.2)
defaut_historique = st.selectbox("Historique de défauts de paiement", ["N", "Y"])
longueur_historique_credit = st.number_input("Longueur de l'historique de crédit (en années)", min_value=1, max_value=50, value=5)

# Encodage des variables catégorielles
mapping_statut_logement = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
mapping_objet_pret = {"PERSONAL": 0, "EDUCATION": 1, "MEDICAL": 2, "VENTURE": 3, "HOMEIMPROVEMENT": 4, "DEBTCONSOLIDATION": 5}
mapping_note_credit = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
mapping_defaut = {"N": 0, "Y": 1}

statut_logement_enc = mapping_statut_logement[statut_logement]
objet_pret_enc = mapping_objet_pret[objet_pret]
note_credit_enc = mapping_note_credit[note_credit]
defaut_historique_enc = mapping_defaut[defaut_historique]

# Préparation des données pour la prédiction
features = np.array([[age, revenu, statut_logement_enc, anciennete_emploi, objet_pret_enc, note_credit_enc, montant_pret, taux_interet, pourcentage_revenu_pret, defaut_historique_enc, longueur_historique_credit]])
prediction = model.predict(features)[0]

# Affichage du résultat
st.write("## Résultat de la prédiction")
if prediction == 1:
    st.error("Ce client est considéré comme risqué !")
else:
    st.success("Ce client n'est pas risqué.")
