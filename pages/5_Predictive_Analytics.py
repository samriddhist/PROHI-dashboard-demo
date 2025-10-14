import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Streamlit page setup
st.set_page_config(page_title="Predictive Analytics", layout="wide")

# Sidebar
st.sidebar.image("./assets/Colorectal Cancer Logo.png")
st.sidebar.success("Select a tab above.")

# Page header
st.markdown("""
    <h1 style="
        font-size: 60px;
        font-weight: 700;
        text-align: center;
        color: #1261B5;
    ">
        Predictive Analytics
    </h1>
""", unsafe_allow_html=True)

# Load model safely
model_path = "jupyter-notebooks/final_knn_k3_pipeline.joblib"
try:
    with open(model_path, "rb") as file:
        model = joblib.load(file)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# File upload or dataset
st.markdown("### Upload or use default dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
else:
    df = pd.read_csv("jupyter-notebooks/postprocessed_colorectal_cancer_dataset.csv", sep=";")

st.write("Sample of the data:")
st.dataframe(df.head())

# Prediction input section
st.markdown("### Predict Patient Survivability")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=18, max_value=120, value=60)
with col2:
    tumor_size = st.number_input("Tumor Size (mm)", min_value=0, max_value=200, value=50)
with col3:
    stage = st.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])

if st.button("Run Prediction", use_container_width=True):
    try:
        input_data = pd.DataFrame({
            "Age": [age],
            "Tumor_Size": [tumor_size],
            "Stage": [stage]
        })

        prediction = model.predict(input_data)
        st.success(f"🧠 Predicted Outcome: **{prediction[0]}**")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
