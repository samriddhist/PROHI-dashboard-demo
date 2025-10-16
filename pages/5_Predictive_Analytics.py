import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Predictive Analytics", layout="wide")
st.sidebar.success("Select a tab above.")
st.sidebar.image("./assets/Colorectal Cancer Logo.png")

st.markdown("""
<h1 style="font-size:60px; font-weight:700; text-align:center; color:#1261B5;">
Predictive Analytics
</h1>
""", unsafe_allow_html=True)


# -------------------------------
# ✅ Load the full KNN pipeline
# -------------------------------
@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load("jupyter-notebooks/final_knn_k3_pipeline.joblib")
        st.success("✅ Full KNN pipeline loaded successfully.")
        return pipeline
    except Exception as e:
        st.error(f"❌ Failed to load model pipeline: {e}")
        return None

pipeline = load_pipeline()

if pipeline is None:
    st.stop()


# -------------------------------
# 🧩 Collect user input
# -------------------------------
st.subheader("Select patient data and see the predicted survivability of 5 years:")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    obesity_bmi = st.selectbox("Obesity BMI", ["Normal", "Overweight", "Obese"])
    family_history = st.selectbox("Family History", ["Yes", "No"])

with col2:
    alcohol_consumption = st.selectbox("Alcohol Consumption", ["Yes", "No"])
    diet_risk = st.selectbox("Diet Risk", ["Low", "Moderate", "High"])
    screening_history = st.selectbox("Screening History", ["Never", "Irregular", "Regular"])

# Prepare input dataframe
input_df = pd.DataFrame([{
    "Age": age,
    "Obesity_BMI": obesity_bmi,
    "Family_History": family_history,
    "Alcohol_Consumption": alcohol_consumption,
    "Diet_Risk": diet_risk,
    "Screening_History": screening_history
}])


# -------------------------------
# 🧠 Make prediction
# -------------------------------
if st.button("Predict 5-Year Survival"):
    try:
        pred = pipeline.predict(input_df)
        proba = pipeline.predict_proba(input_df)[:, 1] if hasattr(pipeline, "predict_proba") else None

        if int(pred[0]) == 1:
            color = "#1DB954"
            text = f"🟩 Predicted: Survive 5 years"
        else:
            color = "#FF4B4B"
            text = f"🟥 Predicted: Not survive 5 years"

        st.markdown(f"""
        <div style="text-align:center; padding:15px; border-radius:12px; background-color:#f5f5f5;">
            <h4 style="color:{color};">{text}</h4>
            <p><b>Model Confidence:</b> {proba[0]:.2%}</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(proba[0]))

    except Exception as e:
        st.error(f"❌ Error generating prediction: {e}")
