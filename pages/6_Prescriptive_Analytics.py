import pickle
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="Prescriptive Analytics", layout="wide")
st.sidebar.success("Select a tab above.")
st.sidebar.image("./assets/Colorectal Cancer Logo.png")

st.markdown("""
<h1 style="font-size:60px; font-weight:700; text-align:center; color:#1261B5;">
Prescriptive Analytics
</h1>
""", unsafe_allow_html=True)

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

st.write("Input sample (you can modify below or use the manual numeric fallback):")
input_data = pd.DataFrame({
    "Age": [50],
    "Obesity_BMI": ["Normal"],
    "Family_History": ["Yes"],
    "Alcohol_Consumption": ["No"],
    "Diet_Risk": ["Low"],
    "Screening_History": ["Regular"]
})

for c in ["Obesity_BMI", "Diet_Risk", "Screening_History", "Family_History", "Alcohol_Consumption"]:
    input_data[c] = input_data[c].astype(str).str.strip().str.capitalize()

st.dataframe(input_data)

mapping = {
    "Obesity_BMI": {"Normal": 0, "Overweight": 1, "Obese": 2},
    "Family_History": {"No": 0, "Yes": 1},
    "Alcohol_Consumption": {"No": 0, "Yes": 1},
    "Diet_Risk": {"Low": 0, "Moderate": 1, "High": 2},
    "Screening_History": {"Never": 0, "Irregular": 1, "Regular": 2}
}

try:
    X = None
    if preprocessor is not None:
        try:
            X = preprocessor.transform(input_data)
        except Exception as e:
            st.warning(f"Preprocessor exists but failed to transform input: {e}")
            X = None

    if X is None:
        X_manual = input_data.copy()
        for col, m in mapping.items():
            if col in X_manual.columns:
                X_manual[col] = X_manual[col].map(m)
        X_manual = X_manual.apply(pd.to_numeric, errors="coerce")
        X = X_manual.to_numpy()

    X_np = X if isinstance(X, np.ndarray) else np.asarray(X)
    provided_features = X_np.shape[1]

    expected = None
    if hasattr(model, "n_features_in_"):
        expected = int(model.n_features_in_)
    else:
        try:
            expected = int(model.coef_.shape[1])
        except Exception:
            expected = None
    if expected is not None and expected != provided_features:
        st.error(
            f"Prediction error: feature count mismatch — model expects {expected} features, input has {provided_features}."
        )
        st.markdown("### Fallback: choose which input columns to provide to the model (for testing)")
        feat_names = getattr(model, "feature_names_in_", None)
        cols = list(input_data.columns)

        if feat_names is not None:
            st.write("Model feature_names_in_ detected:", list(feat_names))
            if all(fn in cols for fn in feat_names):
                chosen = list(feat_names)
                st.success("Using model.feature_names_in_ from the classifier.")
            else:
                st.warning("Model.feature_names_in_ contains names not present in the input. Please map manually below.")
                chosen = None
        else:
            chosen = None

        if chosen is None:
            chosen = []
            available = cols
            for i in range(expected if expected is not None else 2):
                sel = st.selectbox(f"Select feature #{i+1} to pass to model", options=available, index=min(i, len(available)-1), key=f"sel_feat_{i}")
                chosen.append(sel)

        st.write("Chosen columns to feed model:", chosen)

        vector = []
        for col in chosen:
            val = input_data.iloc[0][col]
            try:
                num = float(val)
                vector.append(num)
                continue
            except Exception:
                pass
            if col in mapping:
                mapped = mapping[col].get(str(val).capitalize(), None)
                if mapped is not None:
                    vector.append(float(mapped))
                    continue
            st.warning(f"Column '{col}' is categorical/unmapped. Provide numeric override:")
            override = st.number_input(f"Numeric value for {col}", value=0.0, key=f"ovr_{col}")
            vector.append(float(override))

        if len(vector) != expected:
            st.error(f"Constructed vector length {len(vector)} != expected {expected}. Adjust selections.")
            st.stop()

        arr = np.asarray(vector, dtype=float).reshape(1, -1)
        st.write("Vector passed to model:", arr)

        try:
            pred = model.predict(arr)
            proba = model.predict_proba(arr)[:, 1][0] if hasattr(model, "predict_proba") else None
            st.write("Prediction:", pred[0])
            if proba is not None:
                st.write("Probability:", float(proba))
        except Exception as e:
            st.error(f"Manual/mapped prediction failed: {e}")
        st.stop()

    prediction = model.predict(X_np)
    proba = model.predict_proba(X_np)[:, 1][0] if hasattr(model, "predict_proba") else None

    st.write("Prediction:", prediction[0])
    if proba is not None:
        st.write("Probability:", float(proba))

except Exception as e:
    st.error(f"Prediction error: {e}")
