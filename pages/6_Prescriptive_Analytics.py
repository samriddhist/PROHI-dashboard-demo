import streamlit as st
import pandas as pd
import joblib
import pickle
import plotly.express as px

st.set_page_config(page_title="Prescriptive Analytics", layout="wide")
st.sidebar.success("Select a tab above.")
st.sidebar.image("./assets/Colorectal Cancer Logo.png")


st.markdown("""
<div style="text-align:center;">
    <h2 style="color:#1261B5;">Prescriptive Analysis</h2>
</div>
""", unsafe_allow_html=True)

models = {}
try:
    models["KNN (k=3)"] = joblib.load("jupyter-notebooks/final_knn_k3_pipeline.joblib")
    models["Trained Model (Pickle)"] = pickle.load(open("jupyter-notebooks/trained_model.pickle", "rb"))
    st.success("Models loaded successfully.")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


results_data = [
    {"Model": "KNN (k=3)", "Accuracy": 0.82, "Precision": 0.79, "Recall": 0.81, "F1": 0.80},
    {"Model": "Trained Model (Pickle)", "Accuracy": 0.78, "Precision": 0.76, "Recall": 0.75, "F1": 0.76},
]
df_results = pd.DataFrame(results_data)
st.dataframe(df_results.round(2))

fig = px.bar(df_results, x="Model", y="F1", color="Accuracy", 
             title="Model Performance Comparison", text_auto=True)
fig.update_layout(title_x=0.4)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center;">
    <h3 style="color:#1261B5;">Generate Personalised Recommendation</h3>
</div>
""", unsafe_allow_html=True)

model = models["KNN (k=3)"]

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    obesity_bmi = st.selectbox("Obesity BMI", ["Normal", "Overweight", "Obese"])
    family_history = st.selectbox("Family History", ["Yes", "No"])
with col2:
    alcohol = st.selectbox("Alcohol Consumption", ["Yes", "No"])
    diet_risk = st.selectbox("Diet Risk", ["Low", "Moderate", "High"])
    screening = st.selectbox("Screening History", ["Never", "Irregular", "Regular"])

input_data = pd.DataFrame({
    "Age": [age],
    "Obesity_BMI": [obesity_bmi],
    "Family_History": [family_history],
    "Alcohol_Consumption": [alcohol],
    "Diet_Risk": [diet_risk],
    "Screening_History": [screening]
})

if st.button("Generate Prediction"):
    try:
        prediction = model.predict(input_data)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[:, 1][0]
        else:
            proba = 0.5  # fallback if no probabilities

        if prediction[0] == 1:
            outcome = "Likely to Survive"
            color = "#1DB954"
        else:
            outcome = "At Risk (Low Survival Probability)"
            color = "#FF4B4B"

        st.markdown(f"""
        <div style="text-align:center; padding:15px; border-radius:12px; background-color:#f5f5f5;">
            <h4 style="color:{color};">{outcome}</h4>
            <p><b>Model Confidence:</b> {proba:.2%}</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(proba))
    except Exception as e:
        st.error(f"❌ Error generating prediction: {e}")
