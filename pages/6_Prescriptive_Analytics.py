import pickle as pickle  # instead of pickle/joblib
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Prescriptive Analytics", layout="wide")

# Load model safely
try:
    with open("jupyter-notebooks/trained_model.pickle", "rb") as f:
        model = pickle.load(f)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Example prediction
input_data = pd.DataFrame({
    "Age": [50],
    "Obesity_BMI": ["Normal"],
    "Family_History": ["Yes"],
    "Alcohol_Consumption": ["No"],
    "Diet_Risk": ["Low"],
    "Screening_History": ["Regular"]
})

try:
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[:, 1][0] if hasattr(model, "predict_proba") else 0.5
    st.write("Prediction:", prediction[0])
    st.write("Probability:", proba)
except Exception as e:
    st.error(f"Prediction error: {e}")
