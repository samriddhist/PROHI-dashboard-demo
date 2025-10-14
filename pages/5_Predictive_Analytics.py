import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------- PAGE CONFIG ---------------------- #
st.set_page_config(page_title="Predictive Analytics", layout="wide")

st.sidebar.image("./assets/Colorectal Cancer Logo.png")
st.sidebar.success("Select a Tab Above")

st.markdown("""
    <style>
        .hero {
            background: linear-gradient(90deg, #1261B5 0%, #5FA8D3 100%);
            padding: 40px 20px;
            border-radius: 16px;
            color: white;
            text-align: center;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        }
        .hero h1 {
            font-size: 50px;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .input-section, .result-section {
            background-color: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            margin-top: 30px;
        }
        .result-card {
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            font-size: 20px;
            font-weight: 600;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ---------------------- #
st.markdown("""
<div class="hero">
    <h1>Predictive Analytics</h1>
    <p>Given a patient’s demographic, lifestyle, and clinical features, predict whether they will survive at least 5 years after a colorectal cancer diagnosis.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------- LOAD MODEL ---------------------- #
@st.cache_resource
def load_model():
    try:
        model = joblib.load("jupyter-notebooks/final_knn_k3_pipeline.joblib")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ---------------------- PATIENT INPUTS ---------------------- #
st.markdown("### 🧍‍♂️ Patient Information", unsafe_allow_html=True)
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 20, 90, 55)
    obesity_bmi = st.selectbox("Obesity (BMI Category)", ["Normal", "Overweight", "Obese"])
with col2:
    family_history = st.radio("Family History of Colorectal Cancer?", ["Yes", "No"])
    alcohol = st.radio("Alcohol Consumption?", ["Yes", "No"])
with col3:
    diet_risk = st.selectbox("Dietary Risk", ["Low", "Moderate", "High"])
    screening_history = st.selectbox("Screening History", ["Never", "Irregular", "Regular"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- PREDICTION ---------------------- #
if st.button("🔍 Generate Prediction"):
    if model is not None:
        # Create DataFrame from inputs
        input_data = pd.DataFrame([{
            "Age": age,
            "Obesity_BMI": obesity_bmi,
            "Family_History": family_history,
            "Alcohol_Consumption": alcohol,
            "Diet_Risk": diet_risk,
            "Screening_History": screening_history
        }])

        # Predict
        try:
            pred_proba = model.predict_proba(input_data)[0][1]
            pred_class = model.predict(input_data)[0]
            
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            if pred_class == 1:
                st.markdown(
                    f"<div class='result-card' style='background-color:#2E8B57;'>✅ Likely to Survive 5 Years</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div class='result-card' style='background-color:#D9534F;'>❌ Unlikely to Survive 5 Years</div>",
                    unsafe_allow_html=True)

            st.markdown(f"<p style='text-align:center;font-size:18px;'>Predicted probability of survival: <b>{pred_proba:.2%}</b></p>", unsafe_allow_html=True)
            st.progress(int(pred_proba * 100))
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.warning("Model could not be loaded. Please check file path or format.")

# ---------------------- FOOTER ---------------------- #
st.markdown("""
<hr style="margin-top: 50px;">
<p style="text-align:center; color:gray;">Model: KNN (k=3) | Trained on cleaned colorectal cancer dataset</p>
""", unsafe_allow_html=True)
