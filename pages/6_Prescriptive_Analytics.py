# ...existing code...
import pickle as pickle  
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Prescriptive Analytics", layout="wide")

model = None
preprocessor = None
model_path = "jupyter-notebooks/trained_model.pickle"
preproc_paths = [
    os.path.join("jupyter-notebooks", "preprocessor.joblib"),
    os.path.join("jupyter-notebooks", "preprocessor.pickle"),
    os.path.join("jupyter-notebooks", "preprocessor.pkl"),
]

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# try to load standalone preprocessor if available
for p in preproc_paths:
    if os.path.exists(p):
        try:
            preprocessor = joblib.load(p)
            st.info(f"Loaded preprocessor from {p}")
            break
        except Exception:
            try:
                with open(p, "rb") as f:
                    preprocessor = pickle.load(f)
                    st.info(f"Loaded preprocessor from {p}")
                    break
            except Exception:
                st.warning(f"Failed to load preprocessor at {p}")

# example input (you can expose inputs via Streamlit widgets if desired)
input_data = pd.DataFrame({
    "Age": [50],
    "Obesity_BMI": ["Normal"],
    "Family_History": ["Yes"],
    "Alcohol_Consumption": ["No"],
    "Diet_Risk": ["Low"],
    "Screening_History": ["Regular"]
})

# ensure consistent string formatting
for c in ["Obesity_BMI", "Diet_Risk", "Screening_History", "Family_History", "Alcohol_Consumption"]:
    input_data[c] = input_data[c].astype(str).str.strip().str.capitalize()

try:
    X = None
    if preprocessor is not None:
        try:
            X = preprocessor.transform(input_data)
        except Exception as e:
            st.warning(f"Preprocessor exists but failed to transform input: {e}")
            X = None

    if X is None:
        # manual mapping fallback: must match training encoding exactly
        mapping = {
            "Obesity_BMI": {"Normal": 0, "Overweight": 1, "Obese": 2},
            "Family_History": {"No": 0, "Yes": 1},
            "Alcohol_Consumption": {"No": 0, "Yes": 1},
            "Diet_Risk": {"Low": 0, "Moderate": 1, "High": 2},
            "Screening_History": {"Never": 0, "Irregular": 1, "Regular": 2}
        }
        X_manual = input_data.copy()
        for col, m in mapping.items():
            X_manual[col] = X_manual[col].map(m)
        X = X_manual.astype(float).to_numpy()

    # predict
    prediction = model.predict(X)
    proba = model.predict_proba(X)[:, 1][0] if hasattr(model, "predict_proba") else None

    st.write("Prediction:", prediction[0])
    if proba is not None:
        st.write("Probability:", float(proba))
except Exception as e:
    st.error(f"Prediction error: {e}")
# ...existing code...