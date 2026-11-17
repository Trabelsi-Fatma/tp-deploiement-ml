import pandas as pd
import requests
import streamlit as st
API_URL = "http://127.0.0.1:8000"
NIVEAUX = [
 "some high school",
 "high school",
 "some college",
 "associate's degree",
 "bachelor's degree",
 "master's degree",
]
st.set_page_config(page_title="Réussite en maths")
st.title("Réussite en maths")
st.write("Saisissez le profil d'un étudiant : l'API prédit s'il réussira l'examen de maths.")
col1, col2 = st.columns(2)
with col1:
 reading_score = st.number_input("Note de lecture", min_value=0, max_value=100, value=85)
 writing_score = st.number_input("Note d'écriture", min_value=0, max_value=100, value=82)
 gender = st.selectbox("Genre", ["female", "male"])
 race_ethnicity = st.selectbox(
 "Groupe", ["group A", "group B", "group C", "group D", "group E"], index=3
 )
with col2:
 parental_level_of_education = st.selectbox(
 "Niveau d'études des parents", NIVEAUX, index=4
 )
 test_preparation_course = st.selectbox(
 "Cours de préparation", ["none", "completed"], index=1
 )
 lunch = st.selectbox("Repas", ["standard", "free/reduced"])
if st.button("Prédire"):
 payload = {
 "reading_score": reading_score,
 "writing_score": writing_score,
 "parental_level_of_education": parental_level_of_education,
 "test_preparation_course": test_preparation_course,
 "gender": gender,
 "race_ethnicity": race_ethnicity,
 "lunch": lunch,
 }
 try:
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
 except requests.exceptions.RequestException:
    st.error("Impossible de joindre l'API. Le serveur est-il lancé ?")
 else:
    if response.status_code == 200:
        result = response.json()
        st.success(f"Résultat prédit : {result['class_name']}")
        st.bar_chart(pd.Series(result["probabilities"]))
    else:
        st.error(f"Erreur {response.status_code} : {response.json().get('detail')}")