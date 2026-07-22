"""
Student Placement Predictor - Streamlit App
--------------------------------------------
Run locally with:
    pip install streamlit scikit-learn pandas numpy joblib
    streamlit run app.py

Expects best_model.joblib and scaler.joblib in the SAME folder as this file.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="centered")

# ------------------------------------------------------------------
# Load the trained model + scaler (fit during training, reused here)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "Could not find best_model.joblib / scaler.joblib next to app.py. "
        "Make sure both files are in the same folder as this script."
    )
    st.stop()

# Feature order must match exactly what the model was trained on
FEATURE_COLS = [
    "CGPA", "10th_Percentage", "12th_Percentage", "Internships",
    "Coding_Score", "Aptitude_Score", "Communication_Encoded",
    "Projects", "Backlogs", "Academic_Score", "Experience_Score", "Skill_Index"
]

COMM_MAP = {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3}

st.title("🎓 Student Placement Predictor")
st.write(
    "Enter a student's profile below. The app applies the same feature "
    "engineering, encoding, and scaling used during training, then returns "
    "a placement prediction from the saved model."
)

st.divider()

# ------------------------------------------------------------------
# Input form — mirrors the raw columns from the original dataset
# ------------------------------------------------------------------
with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.slider("CGPA", min_value=5.0, max_value=10.0, value=7.0, step=0.01)
        tenth = st.slider("10th Percentage", min_value=45.0, max_value=100.0, value=75.0)
        twelfth = st.slider("12th Percentage", min_value=45.0, max_value=100.0, value=73.0)
        internships = st.selectbox("Internships completed", options=[0, 1, 2, 3, 4], index=1)
        coding_score = st.slider("Coding Score", min_value=30.0, max_value=100.0, value=65.0)

    with col2:
        aptitude_score = st.slider("Aptitude Score", min_value=30.0, max_value=100.0, value=65.0)
        communication = st.selectbox("Communication Level", options=["Poor", "Average", "Good", "Excellent"], index=2)
        projects = st.selectbox("Projects completed", options=list(range(0, 9)), index=2)
        backlogs = st.selectbox("Current Backlogs", options=list(range(0, 9)), index=0)

    submitted = st.form_submit_button("Predict Placement", use_container_width=True)

# ------------------------------------------------------------------
# On submit: replicate feature engineering -> encoding -> scaling -> predict
# ------------------------------------------------------------------
if submitted:
    # Feature engineering (must match training exactly)
    academic_score = (cgpa * 10 + tenth + twelfth) / 3
    experience_score = internships * 2 + projects
    skill_index = (coding_score + aptitude_score) / 2 - backlogs * 3
    communication_encoded = COMM_MAP[communication]

    row = pd.DataFrame([{
        "CGPA": cgpa,
        "10th_Percentage": tenth,
        "12th_Percentage": twelfth,
        "Internships": internships,
        "Coding_Score": coding_score,
        "Aptitude_Score": aptitude_score,
        "Communication_Encoded": communication_encoded,
        "Projects": projects,
        "Backlogs": backlogs,
        "Academic_Score": academic_score,
        "Experience_Score": experience_score,
        "Skill_Index": skill_index,
    }])[FEATURE_COLS]  # enforce exact column order used in training

    scaled_row = scaler.transform(row)
    prediction = model.predict(scaled_row)[0]
    probability = model.predict_proba(scaled_row)[0][1]  # P(Placed)

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.success(f"✅ Likely to be **Placed** — predicted probability: {probability:.1%}")
    else:
        st.warning(f"⚠️ Likely to be **Not Placed** — predicted probability of placement: {probability:.1%}")

    st.progress(min(max(probability, 0.0), 1.0))

    with st.expander("See engineered feature values used for this prediction"):
        st.dataframe(row.T.rename(columns={0: "Value"}))

    st.caption(
        "Note: this model was trained on a synthetic dataset with ~65-68% test accuracy. "
        "Treat predictions as directional guidance, not a certainty."
    )
