import streamlit as st

st.set_page_config(
    page_title="Predicting Colorectal Cancer Survivability Dashboard",layout="wide")

st.sidebar.success("Select a tab above.")

st.sidebar.image("./assets/Colorectal Cancer Logo.png",)

st.markdown("""
    <h1 style="
        font-size: 60px;
        font-weight: 700;
        text-align: center;
        color: #1261B5;
    ">
    Colorectal Cancer Global Dataset & Predictions
    </h1>
""", unsafe_allow_html=True)

with st.container():
    st.write("What is colorectal cancer?")
    st.write("What is it important to investigate?")
    st.write("What can you find on this dashboard?")




