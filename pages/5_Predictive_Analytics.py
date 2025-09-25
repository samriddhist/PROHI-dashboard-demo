import streamlit as st

st.set_page_config(
    page_title="Predictive Analytics",layout="wide")

st.sidebar.success("Select a tab above.")

st.sidebar.image("./assets/Colorectal Cancer Logo.png",)

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

st.markdown(

        """
         Propose a pipeline where a user can interact with the UI to generate predictions on new data based on a pre-trained ML model. 
         Note that the training of the ML models should happen in a Jupyter notebook, and the dashboard only accesses a pre-trained model from a file (e.g., pickle) to generate predictions on the user’s input.
        """
    )