import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


st.set_page_config(page_title="Prescriptive Analytics", layout="wide")
st.sidebar.success("Select a tab above.")
st.sidebar.image("./assets/Colorectal Cancer Logo.png")

st.markdown("""
<h1 style="font-size:60px; font-weight:700; text-align:center; color:#1261B5;">
Prescriptive Analytics
</h1>
""", unsafe_allow_html=True)


try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Shim for old scikit-learn pipelines"""
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception as e:
    st.warning(f"Shim injection warning: {e}")


MODEL_PATH = "jupyter-notebooks/assets/final_knn_k3_pipeline.joblib"
if not os.path.exists(MODEL_PATH):
    st.error(f"Pipeline not found at {MODEL_PATH}")
    st.stop()

try:
    pipeline = joblib.load(MODEL_PATH)
    st.success("Pipeline loaded successfully.")
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    st.stop()


st.subheader("Input patient data to see predictions & explanations:")

age = st.number_input("Age", min_value=0, max_value=120, value=50)
obesity_bmi = st.selectbox("Obesity BMI", ["Normal", "Overweight", "Obese"])
family_history = st.selectbox("Family History", ["Yes", "No"])
alcohol_consumption = st.selectbox("Alcohol Consumption", ["Yes", "No"])
diet_risk = st.selectbox("Diet Risk", ["Low", "Moderate", "High"])
screening_history = st.selectbox("Screening History", ["Never", "Irregular", "Regular"])

input_df = pd.DataFrame([{
    "Age": age,
    "Obesity_BMI": obesity_bmi,
    "Family_History": family_history,
    "Alcohol_Consumption": alcohol_consumption,
    "Diet_Risk": diet_risk,
    "Screening_History": screening_history
}])

for c in ["Obesity_BMI", "Diet_Risk", "Screening_History", "Family_History", "Alcohol_Consumption"]:
    input_df[c] = input_df[c].astype(str).str.strip().str.capitalize()

st.write("### Input data preview")
st.dataframe(input_df)

try:
    pred = pipeline.predict(input_df)
    proba = pipeline.predict_proba(input_df)[:, 1] if hasattr(pipeline, "predict_proba") else None

    if int(pred[0]) == 1:
        st.success(f"🟩 Predicted: Survive 5 years" + (f" — Probability: {proba[0]:.2f}" if proba is not None else ""))
    else:
        st.warning(f"🟥 Predicted: Not survive 5 years" + (f" — Probability: {proba[0]:.2f}" if proba is not None else ""))
    
    if proba is not None:
        st.progress(float(proba[0]))
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()


