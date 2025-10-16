import streamlit as st
import pandas as pd
import joblib
import pickle
import os
import numpy as np
import importlib

st.set_page_config(page_title="Predictive Analytics", layout="wide")
st.sidebar.success("Select a tab above.")
st.sidebar.image("./assets/Colorectal Cancer Logo.png")

st.markdown("""
<h1 style="font-size:60px; font-weight:700; text-align:center; color:#1261B5;">
Predictive Analytics
</h1>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    pipe_path = os.path.join("jupyter-notebooks", "final_knn_k3_pipeline.joblib")
    preproc_paths = [
        os.path.join("jupyter-notebooks", "preprocessor.joblib"),
        os.path.join("jupyter-notebooks", "preprocessor.pickle"),
        os.path.join("jupyter-notebooks", "preprocessor.pkl"),
    ]
    clf_path = os.path.join("jupyter-notebooks", "trained_model.pickle")

    pipe = None
    preprocessor = None
    clf = None

    try:
        ct_mod = importlib.import_module("sklearn.compose._column_transformer")
        if not hasattr(ct_mod, "_RemainderColsList"):
            class _RemainderColsList:
                def __init__(self, *args, **kwargs):
                    pass
            setattr(ct_mod, "_RemainderColsList", _RemainderColsList)
    except Exception:
        pass

    if os.path.exists(pipe_path):
        try:
            pipe = joblib.load(pipe_path)
        except Exception as e:
            st.warning(f"❌ Failed to load pipeline: {e}")
    else:
        st.info(f"Pipeline file not found at {pipe_path} (this is OK if you have separate preprocessor + classifier)")

    for p in preproc_paths:
        if os.path.exists(p):
            try:
                preprocessor = joblib.load(p)
                break
            except Exception:
                try:
                    with open(p, "rb") as f:
                        preprocessor = pickle.load(f)
                        break
                except Exception as e:
                    st.warning(f"❌ Failed to load preprocessor at {p}: {e}")

    if preprocessor is None:
        st.info("No standalone preprocessor artifact found. Will attempt manual preprocessing fallback when needed.")

    if os.path.exists(clf_path):
        try:
            clf = pickle.load(open(clf_path, "rb"))
        except Exception as e:
            st.warning(f"❌ Failed to load classifier: {e}")
    else:
        st.info(f"Classifier file not found at {clf_path}")

    return pipe, preprocessor, clf

pipeline, preprocessor, classifier = load_artifacts()

if pipeline is None and classifier is None:
    st.error(
        "No model artifacts loaded. Make sure the following files exist in your GitHub repo:\n"
        "- `jupyter-notebooks/final_knn_k3_pipeline.joblib` (recommended full pipeline)\n"
        "- `jupyter-notebooks/trained_model.pickle` (classifier only)\n"
        "- optional: `jupyter-notebooks/preprocessor.joblib` (preprocessor only)"
    )
    st.stop()

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

for c in ["Obesity_BMI", "Diet_Risk", "Screening_History", "Family_History", "Alcohol_Consumption"]:
    input_df[c] = input_df[c].astype(str).str.strip().str.capitalize()

cols_order = ["Age", "Obesity_BMI", "Family_History", "Alcohol_Consumption", "Diet_Risk", "Screening_History"]
input_df = input_df[cols_order]

if st.button("Predict 5-Year Survival"):
    try:
        if pipeline is not None:
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
            st.stop()

        if classifier is not None:
            X_proc = None
            if preprocessor is not None:
                try:
                    X_proc = preprocessor.transform(input_df)
                except Exception as e:
                    st.warning(f"Preprocessor exists but failed to transform input: {e}")
                    X_proc = None

            if X_proc is None:
                mapping = {
                    "Obesity_BMI": {"Normal": 0, "Overweight": 1, "Obese": 2},
                    "Family_History": {"No": 0, "Yes": 1},
                    "Alcohol_Consumption": {"No": 0, "Yes": 1},
                    "Diet_Risk": {"Low": 0, "Moderate": 1, "High": 2},
                    "Screening_History": {"Never": 0, "Irregular": 1, "Regular": 2}
                }
                X_manual = input_df.copy()
                for col, m in mapping.items():
                    X_manual[col] = X_manual[col].map(m)
                try:
                    X_proc = X_manual.astype(float).to_numpy()
                except Exception as e:
                    st.error(f"Failed to apply manual preprocessing fallback: {e}")
                    st.stop()
            try:
                pred = classifier.predict(X_proc)
                proba = None
                if hasattr(classifier, "predict_proba"):
                    try:
                        proba = classifier.predict_proba(X_proc)[:, 1]
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
                st.stop()
            except Exception as e:
                st.error(f"❌ Error generating prediction with classifier: {e}")
                st.stop()

        st.error(
            "❌ Only classifier loaded (`trained_model.pickle`). This file expects numeric input, "
            "but no preprocessor is available.\n\n"
            "✅ Best fix: save and load the full pipeline (preprocessing + classifier) as a single artifact."
        )
        st.stop()

    except Exception as e:
        st.error(f"❌ Error generating prediction: {e}")
