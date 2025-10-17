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

try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Shim for backward compatibility."""
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception as e:
    st.warning(f"Shim injection warning: {e}")

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jupyter-notebooks", "assets", "final_knn_k3_pipeline.joblib"))
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
perm_import = os.path.join(FIGS_DIR, "Permutation_Importance_Top20.png")
cf_csv = os.path.join(FIGS_DIR, "counterfactuals_row0.csv")

if not os.path.exists(shap_global):
    with st.spinner("Generating SHAP explanations... please wait"):
        try:
            import shap
            import numpy as np
            import matplotlib.pyplot as plt
            try:
                if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
                    preprocessor = pipeline.named_steps["preprocessor"]
                    X_transformed = preprocessor.transform(input_df)
                    feature_names = preprocessor.get_feature_names_out()
                else:
                    X_transformed = pipeline[:-1].transform(input_df)
                    feature_names = input_df.columns
            except Exception as e:
                st.warning(f"Could not access preprocessing step directly: {e}")
                X_transformed = pipeline.transform(input_df)
                feature_names = input_df.columns
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

            def model_predict(X):
                return pipeline.named_steps[list(pipeline.named_steps.keys())[-1]].predict_proba(X)[:, 1]

            explainer = shap.Explainer(model_predict, X_transformed)
            shap_values = explainer(X_transformed)

            shap.summary_plot(shap_values.values, X_transformed, show=False)
            plt.title("Global SHAP Summary (Numeric Space)")
            plt.tight_layout()
            plt.savefig(shap_global, dpi=300)
            plt.close()

            shap.plots.waterfall(shap_values[0], show=False)
            plt.title("Local SHAP Explanation")
            plt.tight_layout()
            plt.savefig(shap_local, dpi=300)
            plt.close()

            st.success("SHAP explanations generated successfully.")
        except Exception as e:
            st.warning(f"Could not generate SHAP automatically: {e}")


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

# ----------------------- #
# COUNTERFACTUALS via DiCE
# ----------------------- #
try:
    import dice_ml

    st.info("Generating counterfactual recommendation (DiCE)...")

    # Identify categorical & continuous features
    continuous_features = ["Age"]
    categorical_features = [
        "Obesity_BMI",
        "Family_History",
        "Alcohol_Consumption",
        "Diet_Risk",
        "Screening_History",
    ]

    # Create a dummy DataFrame with a target column
    df_for_dice = input_df.copy()
    df_for_dice["Survival_5_years"] = int(pred[0])

    # Build DiCE Data interface
    d = dice_ml.Data(
        dataframe=df_for_dice,
        continuous_features=continuous_features,
        outcome_name="Survival_5_years",
    )

    # Build DiCE model interface
    m = dice_ml.Model(model=pipeline, backend="sklearn", model_type="classifier")

    # Initialize DiCE explainer
    exp = dice_ml.Dice(d, m, method="genetic")  # 'random' also works but is less effective

    # Generate counterfactuals (try 5 examples)
    cf = exp.generate_counterfactuals(
        input_df,
        total_CFs=5,
        desired_class="opposite",
        features_to_vary=continuous_features + categorical_features,
    )

    # Convert and display results
    cf_df = cf.cf_examples_list[0].final_cfs_df_sparse

    if not cf_df.empty:
        st.success("✅ Counterfactuals generated successfully.")
        st.subheader("🧭 Counterfactual Recommendations (DiCE)")
        st.dataframe(cf_df)

        # Compare original vs first counterfactual
        original = input_df.iloc[0].to_dict()
        cf_row = cf_df.iloc[0].to_dict()
        diffs = {k: (original[k], cf_row[k]) for k in original.keys() if str(original[k]) != str(cf_row[k])}

        if diffs:
            st.markdown("### 🔁 Recommended Changes")
            for feat, (orig, new) in diffs.items():
                st.markdown(f"- **{feat}**: change from `{orig}` → `{new}`")
        else:
            st.info("No actionable changes detected.")
    else:
        st.warning("⚠️ DiCE could not generate meaningful counterfactuals for this case.")

except Exception as e:
    st.warning(f"Could not generate counterfactuals dynamically: {e}")



st.markdown("""
**Interpretation:**  
- Features at the top of the SHAP plots have the strongest influence on 5-year survival.  
- If the predicted survival probability is **low**, review the recommended feature changes above — these are minimal adjustments that could improve survival odds.  
- This integrates predictive, explainable, and prescriptive analytics for clinical insight.
""")
