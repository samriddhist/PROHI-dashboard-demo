import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Predictive Analytics", layout="wide"
)

st.sidebar.success("Select a tab above.")
st.sidebar.image("./assets/Colorectal Cancer Logo.png")

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

# --- Load pre-fitted pipeline ---
try:
    with open("jupyter-notebooks/trained_model.pickle", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

st.subheader("Enter Patient Data for Prediction")

# --- UI elements for input ---
age = st.number_input("Age", min_value=0, max_value=120, value=50)

obesity_bmi = st.selectbox(
    "Obesity BMI",
    options=["Normal", "Overweight", "Obese"]
)

family_history = st.selectbox(
    "Family History of Colorectal Cancer",
    options=["Yes", "No"]
)

alcohol_consumption = st.selectbox(
    "Alcohol Consumption",
    options=["Yes", "No"]
)

diet_risk = st.selectbox(
    "Diet Risk",
    options=["Low", "Moderate", "High"]
)

screening_history = st.selectbox(
    "Screening History",
    options=["Never", "Irregular", "Regular"]
)

# --- Collect input into DataFrame ---
input_df = pd.DataFrame({
    "Age": [age],
    "Obesity_BMI": [obesity_bmi],
    "Family_History": [family_history],
    "Alcohol_Consumption": [alcohol_consumption],
    "Diet_Risk": [diet_risk],
    "Screening_History": [screening_history]
})

# Normalize strings (match pre-fitted categories)
for col in input_df.columns:
    if input_df[col].dtype == "object":
        input_df[col] = input_df[col].str.strip().str.capitalize()

# --- Make prediction ---
if st.button("Predict Survival"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown(f"**Predicted 5-Year Survival:** {'Yes' if prediction == 1 else 'No'}")
        st.markdown(f"**Probability of Survival:** {probability:.2%}")
    except Exception as e:
        st.error(f"Error generating prediction: {e}")
