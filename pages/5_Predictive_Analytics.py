import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np

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
    pipe = None
    clf = None
    try:
        pipe = joblib.load("jupyter-notebooks/final_knn_k3_pipeline.joblib")
    except Exception as e:
        pipe = None

    try:
        clf = pickle.load(open("jupyter-notebooks/trained_model.pickle", "rb"))
    except Exception as e:
        clf = None

    return pipe, clf

pipeline, classifier = load_artifacts()

if pipeline is None and classifier is None:
    st.error("No model artifacts loaded. Make sure one of the files exists:\n"
             "`jupyter-notebooks/final_knn_k3_pipeline.joblib` (recommended full pipeline) or\n"
             "`jupyter-notebooks/trained_model.pickle` (classifier).")
    st.stop()

st.subheader("Select patient data and see the predicted survivability of 5 years: ")

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
            if hasattr(pipeline, "predict"):
                pred = pipeline.predict(input_df)
                proba = None
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(input_df)[:, 1]
                # show results
                if int(pred[0]) == 1:
                    st.success(f"Predicted Survivability of 5 years — Probability: {proba[0]:.2f}" if proba is not None else "🟩 Predicted: Survive 5 years")
                else:
                    st.warning(f"Predicted: No Survivability of 5 years — Probability: {proba[0]:.2f}" if proba is not None else "🟥 Predicted: Not survive 5 years")
                raise StopIteration 

            if hasattr(pipeline, "transform"):
                X_trans = pipeline.transform(input_df)
                if classifier is not None:
                    pred = classifier.predict(X_trans)
                    proba = classifier.predict_proba(X_trans)[:, 1] if hasattr(classifier, "predict_proba") else None
                    if int(pred[0]) == 1:
                        st.success(f"🟩 Predicted: Survive 5 years — probability: {proba[0]:.2f}" if proba is not None else "🟩 Predicted: Survive 5 years")
                    else:
                        st.warning(f"🟥 Predicted: Not survive 5 years — probability: {proba[0]:.2f}" if proba is not None else "🟥 Predicted: Not survive 5 years")
                    raise StopIteration
                else:
                    st.error("Pipeline is a transformer (has transform) but no classifier was loaded. Provide the classifier pickle or use a full pipeline that ends with the classifier.")
                    raise StopIteration

        if classifier is not None:
            st.error("Only classifier file loaded (trained_model.pickle). This file likely expects numeric input but no preprocessor is available. "
                     "Best fix: save and load the full pipeline (preprocessing + classifier) as a single artifact (recommended).")
            st.stop()

    except StopIteration:
        pass
    except Exception as e:
        st.error(f"❌ Error generating prediction: {e}")
