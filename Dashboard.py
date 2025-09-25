import streamlit as st

st.set_page_config(
    page_title="Predicting Colorectal Cancer Survivability Dashboard",layout="wide")

st.sidebar.success("Select a tab above.")

tab_about, tab_desc, tab_diag, tab_pred, tab_presc = st.tabs (["ℹ️ About", "📊 Descriptive Analytics", "🔍 Diagnostic Analytics", "🤖 Predictive Analytics", "🧭 Prescriptive Analytics"])

st.write("# Colorectal Cancer Global Dataset & Predictions")


