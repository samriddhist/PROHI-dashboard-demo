import streamlit as st

st.set_page_config(
    page_title="Predicting Colorectal Cancer Survivability Dashboard",layout="wide")

st.sidebar.success("Select a tab above.")

st.title("# Colorectal Cancer Global Dataset & Predictions")

with st.container():
    st.write("This is inside the container")


