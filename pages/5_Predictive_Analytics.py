import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Predictive Analytics", layout="wide")
st.sidebar.success("Select a tab above.")
st.sidebar.image("./assets/Colorectal Cancer Logo.png")

st.markdown("""
<h1 style="font-size:60px; font-weight:700; text-align:center; color:#1261B5;">
Predictive Analytics
</h1>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    model_path = "jupyter-notebooks/assets/trained_model.joblib"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.stop()
    return joblib.load(model_path)

pipeline = load_pipeline()

st.subheader("Enter patient data to predict 5-year survival:")

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

for c in input_df.columns:
    input_df[c] = input_df[c].astype(str).str.strip().str.capitalize()

cols_order = ["Age", "Obesity_BMI", "Family_History", "Alcohol_Consumption", "Diet_Risk", "Screening_History"]
input_df = input_df[cols_order]

if st.button("Predict 5-Year Survival"):
    try:
        pred = pipeline.predict(input_df)
        proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(input_df)[:, 1]
            except Exception:
                proba = None

        if int(pred[0]) == 1:
            st.success(
                f"🟩 Predicted: Survive 5 years — Probability: {proba[0]:.2f}"
                if proba is not None else "🟩 Predicted: Survive 5 years"
            )
        else:
            st.warning(
                f"🟥 Predicted: Not survive 5 years — Probability: {proba[0]:.2f}"
                if proba is not None else "🟥 Predicted: Not survive 5 years"
            )

        if proba is not None:
            st.progress(float(proba[0]))

    except Exception as e:
        st.error(f"❌ Error generating prediction: {e}")
