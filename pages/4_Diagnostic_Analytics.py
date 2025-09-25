import streamlit as st

st.set_page_config(
    page_title="Diagnostic Analytics",layout="wide")

st.sidebar.success("Select a tab above.")

st.sidebar.image("./assets/Colorectal Cancer Logo.png",)

st.markdown("""
    <h1 style="
        font-size: 60px;
        font-weight: 700;
        text-align: center;
        color: #1261B5;
    ">
        Diagnostic Analytics
    </h1>
""", unsafe_allow_html=True)

st.markdown(

        """
        Propose a pipeline where a user can interact with the UI to understand relationships between variables and get insights of diagnostic nature (e.g., correlations, statistical analysis, clustering). 
        The dashboard should include enough text to show the reasoning on the questions that are trying to be solved as well as the conclusions of the analysis.
        """
    )