import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prescriptive Analytics", layout="wide")
st.sidebar.success("Select a tab above.")
st.sidebar.image("./assets/Colorectal Cancer Logo.png")

st.markdown("""
<h1 style="font-size:60px; font-weight:700; text-align:center; color:#1261B5;">
Prescriptive Analytics
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding: 12px; background-color:#1261B5; border-radius:8px;">
  <p style="font-size:18px; color:white; margin:0;">
    This page includes the prescriptive analytics component. This section provides actionable recommendations to improve patient outcomes based on model predictions and explanations.
  </p>
</div>
""", unsafe_allow_html=True)

try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Shim for backward compatibility."""
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception as e:
    st.warning(f"Shim injection warning: {e}")

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          "..", "jupyter-notebooks", "assets",
                                          "final_knn_k3_pipeline.joblib"))
FIGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Figs"))
os.makedirs(FIGS_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: {MODEL_PATH}")
    st.stop()

try:
    obj = joblib.load(MODEL_PATH)
    if isinstance(obj, list):
        from sklearn.pipeline import Pipeline
        pipeline = next((x for x in obj if isinstance(x, Pipeline)), obj[0])
    else:
        pipeline = obj
    st.success("Pipeline loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading pipeline: {e}")
    st.stop()

# ---- Inputs ----
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

for col in ["Obesity_BMI", "Diet_Risk", "Screening_History", "Family_History", "Alcohol_Consumption"]:
    input_df[col] = input_df[col].astype(str).str.strip().str.capitalize()

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
    st.error(f"❌ Prediction failed: {e}")
    st.stop()

st.markdown("---")
st.markdown("## Prescriptive Insights")

shap_global = os.path.join(FIGS_DIR, "SHAP_global_kernel.png")
shap_local = os.path.join(FIGS_DIR, "SHAP_local_kernel_row0.png")

if not os.path.exists(shap_global) or not os.path.exists(shap_local):
    with st.spinner("Generating SHAP explanations..."):
        try:
            try:
                if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
                    preprocessor = pipeline.named_steps["preprocessor"]
                    X_transformed = preprocessor.transform(input_df)
                    feature_names = preprocessor.get_feature_names_out()
                else:
                    X_transformed = pipeline[:-1].transform(input_df)
                    feature_names = input_df.columns
            except Exception:
                X_transformed = pipeline.transform(input_df)
                feature_names = input_df.columns

            X_df = pd.DataFrame(np.asarray(X_transformed), columns=feature_names)

            def model_predict(X):
                clf = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]
                return clf.predict_proba(X)[:, 1]

            explainer = shap.Explainer(model_predict, X_df)
            shap_values = explainer(X_df)

            plt.figure()
            shap.summary_plot(shap_values.values, X_df, show=False)
            plt.title("Global SHAP Summary (Numeric Space)")
            plt.tight_layout()
            plt.savefig(shap_global, dpi=300)
            plt.close()

            plt.figure()
            shap.plots.waterfall(shap_values[0], show=False)
            plt.title("Local SHAP Explanation")
            plt.tight_layout()
            plt.savefig(shap_local, dpi=300)
            plt.close()
        except Exception as e:
            st.info(f"Could not generate SHAP automatically: {e}")

if os.path.exists(shap_global):
    st.subheader("Global Feature Importance (SHAP)")
    st.image(shap_global, caption="Features influencing survival across all patients", use_container_width=True)
else:
    st.info("Global SHAP visualization not found.")

if os.path.exists(shap_local):
    st.subheader("Local Explanation (for example patient)")
    st.image(shap_local, caption="Most influential features for an individual case", use_container_width=True)
else:
    st.info("Local SHAP visualization not found.")

