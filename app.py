import streamlit as st
import pandas as pd
import joblib

# Charger le modèle
model = joblib.load("model_artifacts/stroke_model.pkl")
scaler = joblib.load("model_artifacts/scaler.pkl")
features = joblib.load("model_artifacts/feature_names.pkl")

st.title("🧠 Prédiction du risque de Stroke")

age = st.slider("Âge", 1, 100, 30)
glucose = st.number_input("Glucose", 50, 300, 100)
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Maladie cardiaque", [0, 1])

if st.button("Prédire"):
    age_bmi_ratio = age / (bmi + 1e-5)
    glucose_age_interaction = glucose * age
    high_risk_flag = int(hypertension == 1 or heart_disease == 1 or glucose > 140)

    if age <= 20:
        age_group = 0
    elif age <= 40:
        age_group = 1
    elif age <= 60:
        age_group = 2
    else:
        age_group = 3

    data = pd.DataFrame([{
        'gender': 1,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': 1,
        'work_type': 2,
        'Residence_type': 1,
        'avg_glucose_level': glucose,
        'bmi': bmi,
        'smoking_status': 1,
        'age_bmi_ratio': age_bmi_ratio,
        'glucose_age_interaction': glucose_age_interaction,
        'high_risk_flag': high_risk_flag,
        'age_group': age_group
    }])

    data = data.reindex(columns=features, fill_value=0)
    data_scaled = scaler.transform(data)

    proba = model.predict_proba(data_scaled)[0][1]

    if proba >= 0.7:
        st.error(f"🔴 Risque élevé ({proba*100:.1f}%)")
    elif proba >= 0.4:
        st.warning(f"🟡 Risque modéré ({proba*100:.1f}%)")
    else:
        st.success(f"🟢 Risque faible ({proba*100:.1f}%)")
    
