import streamlit as st
import pandas as pd
import joblib
import pickle
import os
import numpy as np

# ------------------ Page Setup ------------------ #
st.set_page_config(page_title="Predictive Analytics", layout="wide")
st.sidebar.success("Select a tab above.")
st.sidebar.image("./assets/Colorectal Cancer Logo.png")

st.markdown("""
<h1 style="font-size:60px; font-weight:700; text-align:center; color:#1261B5;">
Predictive Analytics
</h1>
""", unsafe_allow_html=True)

# ------------------ Load Models ------------------ #
@st.cache_resource
def load_artifacts():
    pipe_path = os.path.join("jupyter-notebooks", "final_knn_k3_pipeline.joblib")
    clf_path = os.path.join("jupyter-notebooks", "trained_model.pickle")

    pipe = None
    clf = None

    # Load full pipeline
    if os.path.exists(pipe_path):
        try:
            pipe = joblib.load(pipe_path)
        except Exception as e:
            st.warning(f"❌ Failed to load pipeline: {e}")
    else:
        st.warning(f"❌ Pipeline file not found at {pipe_path}")

    # Load classifier only
    if os.path.exists(clf_path):
        try:
            clf = pickle.load(open(clf_path, "rb"))
        except Exception as e:
            st.warning(f"❌ Failed to load classifier: {e}")
    else:
        st.warning(f"❌ Classifier file not found at {clf_path}")

    return pipe, clf

pipeline, classifier = load_artifacts()

if pipeline is None and classifier is None:
    st.error(
        "No model artifacts loaded. Make sure the following files exist in your GitHub repo:\n"
        "- `jupyter-notebooks/final_knn_k3_pipeline.joblib` (recommended full pipeline)\n"
        "- `jupyter-notebooks/trained_model.pickle` (classifier only)"
    )
    st.stop()

# ------------------ Input Form ------------------ #
st.subheader("Select patient data and see the predicted survivability of 5 years:")

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

# Capitalize & clean categorical inputs
for c in ["Obesity_BMI", "Diet_Risk", "Screening_History", "Family_History", "Alcohol_Consumption"]:
    input_df[c] = input_df[c].astype(str).str.strip().str.capitalize()

# Reorder columns
cols_order = ["Age", "Obesity_BMI", "Family_History", "Alcohol_Consumption", "Diet_Risk", "Screening_History"]
input_df = input_df[cols_order]

# ------------------ Prediction ------------------ #
if st.button("Predict 5-Year Survival"):
    try:
        # Full pipeline available
        if pipeline is not None:
            pred = pipeline.predict(input_df)
            proba = None
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(input_df)[:, 1]

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
            st.progress(float(proba[0]) if proba is not None else 0)
            st.stop()

        # Only classifier loaded
        if classifier is not None:
            st.error(
                "❌ Only classifier loaded (`trained_model.pickle`). This file expects numeric input, "
                "but no preprocessor is available.\n\n"
                "✅ Best fix: save and load the full pipeline (preprocessing + classifier) as a single artifact."
            )
            st.stop()

    except Exception as e:
        st.error(f"❌ Error generating prediction: {e}")
