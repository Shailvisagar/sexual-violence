# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
MODEL_PATH = "models/rf_high_burden_classifier.joblib"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="SVAW Prediction", page_icon=":female_sign:", layout="centered")

st.title("🔍 Sexual Violence Against Women (SVAW) Prediction")
st.write("Predict whether a State/UT is in the **high-burden category** based on victim age-distribution.")

# User input form
with st.form("svaw_form"):
    st.subheader("Enter Victim Age Distribution Data")
    age_0_10 = st.number_input("Victims age 0–10", min_value=0, step=1)
    age_11_14 = st.number_input("Victims age 11–14", min_value=0, step=1)
    age_15_18 = st.number_input("Victims age 15–18", min_value=0, step=1)
    age_18_30 = st.number_input("Victims age 18–30", min_value=0, step=1)
    age_30_45 = st.number_input("Victims age 30–45", min_value=0, step=1)
    age_45_60 = st.number_input("Victims age 45–60", min_value=0, step=1)
    age_above_60 = st.number_input("Victims age above 60", min_value=0, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Create dataframe from input
    X_input = pd.DataFrame([[
        age_0_10, age_11_14, age_15_18,
        age_18_30, age_30_45, age_45_60,
        age_above_60
    ]], columns=[
        "Victims_0_10", "Victims_11_14", "Victims_15_18",
        "Victims_18_30", "Victims_30_45", "Victims_45_60",
        "Victims_above_60"
    ])

    # Prediction
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    if pred == 1:
        st.error(f"⚠️ High-Burden State/UT (Probability: {proba:.2%})")
    else:
        st.success(f"✅ Low-Burden State/UT (Probability: {proba:.2%})")
