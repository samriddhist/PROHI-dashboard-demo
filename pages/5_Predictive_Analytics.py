import streamlit as st

st.header("Predictive Analytics")

st.markdown(

        """
         Propose a pipeline where a user can interact with the UI to generate predictions on new data based on a pre-trained ML model. 
         Note that the training of the ML models should happen in a Jupyter notebook, and the dashboard only accesses a pre-trained model from a file (e.g., pickle) to generate predictions on the user’s input.
        """
    )