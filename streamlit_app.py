# ============================================================
# STREAMLIT DEPLOYMENT APP – Diabetes Risk Predictor
# Run: streamlit run streamlit_app.py
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Load model artifacts ──────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load("model_xgb.pkl")
    scaler        = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Diabetes Risk Predictor")
st.markdown("""
This app uses an **XGBoost** machine learning model trained on the
*Pima Indians Diabetes Dataset* to predict the likelihood of diabetes
based on medical measurements.

> ⚠️ This tool is for **educational purposes only** and does not constitute medical advice.
""")

st.divider()

# ── Sidebar inputs ────────────────────────────────────────────
st.sidebar.header("📋 Enter Patient Details")

pregnancies    = st.sidebar.number_input("Pregnancies",          min_value=0,   max_value=20,   value=1)
glucose        = st.sidebar.slider("Glucose (mg/dL)",            min_value=50,  max_value=250,  value=120)
blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)",     min_value=30,  max_value=130,  value=70)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)",        min_value=5,   max_value=100,  value=20)
insulin        = st.sidebar.slider("Insulin (μU/mL)",            min_value=10,  max_value=900,  value=80)
bmi            = st.sidebar.slider("BMI",                        min_value=10.0,max_value=70.0, value=25.0, step=0.1)
dpf            = st.sidebar.slider("Diabetes Pedigree Function", min_value=0.0, max_value=2.5,  value=0.5,  step=0.01)
age            = st.sidebar.slider("Age (years)",                min_value=18,  max_value=100,  value=30)

# ── Feature engineering (must match training pipeline) ────────
def engineer_features(preg, gluc, bp, skin, ins, bmi_val, dpf_val, age_val):
    glucose_bmi     = gluc * bmi_val
    age_preg        = age_val * preg
    insulin_glucose = ins / (gluc + 1)

    bmi_cat = 0
    if bmi_val < 18.5:   bmi_cat = 0
    elif bmi_val < 25.0: bmi_cat = 1
    elif bmi_val < 30.0: bmi_cat = 2
    else:                bmi_cat = 3

    age_grp = 0
    if age_val <= 30:    age_grp = 0
    elif age_val <= 45:  age_grp = 1
    elif age_val <= 60:  age_grp = 2
    else:                age_grp = 3

    return [preg, gluc, bp, skin, ins, bmi_val, dpf_val, age_val,
            glucose_bmi, age_preg, insulin_glucose, bmi_cat, age_grp]

# ── Prediction ────────────────────────────────────────────────
if st.sidebar.button("🔍 Predict"):
    features = engineer_features(
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    )
    input_df = pd.DataFrame([features], columns=feature_names)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("🔎 Prediction Result")

    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error("⚠️ **High Risk: Diabetes Detected**")
        else:
            st.success("✅ **Low Risk: No Diabetes Detected**")
    with col2:
        st.metric("Diabetes Probability", f"{probability:.1%}")

    # Risk bar
    st.progress(float(probability))

    st.divider()

    # Input summary
    st.subheader("📊 Input Summary")
    summary = pd.DataFrame({
        "Feature": ["Pregnancies", "Glucose", "Blood Pressure",
                    "Skin Thickness", "Insulin", "BMI",
                    "Diabetes Pedigree Function", "Age"],
        "Value":   [pregnancies, glucose, blood_pressure,
                    skin_thickness, insulin, bmi, dpf, age]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.info("""
    **Interpretation guide:**
    - Probability < 30% → Low risk
    - Probability 30–60% → Moderate risk (consult a doctor)
    - Probability > 60% → High risk (seek medical attention)
    """)
else:
    st.info("👈 Enter patient details in the sidebar and click **Predict**.")

st.divider()
st.caption("Built with Streamlit & XGBoost | AI Mini-Project | Pima Indians Diabetes Dataset")
