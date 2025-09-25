import streamlit as st

st.set_page_config(
    page_title="Descriptive Analytics",layout="wide")

st.sidebar.success("Select a tab above.")

st.sidebar.image("./assets/Colorectal Cancer Logo.png",)

st.markdown("""
    <h1 style="
        font-size: 60px;
        font-weight: 700;
        text-align: center;
        color: #1261B5;
    ">
        Descriptive Analytics
    </h1>
""", unsafe_allow_html=True)

st.markdown(

        """
        Propose a pipeline where a user can interact with UI elements to get interesting insights about the dataset using analytical techniques of descriptive nature (e.g., summary, pivot tables, basic plots).
        The dashboard should be self-explanatory as it shows enough text to understand what the data is about and how to interact with it.
        """
    )