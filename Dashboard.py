import streamlit as st

st.set_page_config(
    page_title="Predicting Colorectal Cancer Survivability Dashboard",layout="wide")

st.sidebar.success("Select a tab above.")

tab_about, tab_desc, tab_diag, tab_pred, tab_presc = st.tabs (["ℹ️ About", "📊 Descriptive Analytics", "🔍 Diagnostic Analytics", "🤖 Predictive Analytics", "🧭 Prescriptive Analytics"])

st.write("# Colorectal Cancer Global Dataset & Predictions")

with tab_about:
    st.header("About this Project")
    st.markdown(
        """
        This tab contains information about the dataset, references, group members and contact information.

        """
    )
with tab_desc:
    st.header("Descriptive Analytics")
    st.markdown(

        """
        Propose a pipeline where a user can interact with UI elements to get interesting insights about the dataset using analytical techniques of descriptive nature (e.g., summary, pivot tables, basic plots).
        The dashboard should be self-explanatory as it shows enough text to understand what the data is about and how to interact with it.
        """
    )

with tab_diag:
    st.header("Diagnostic Analytics")
    st.markdown(

        """
        Propose a pipeline where a user can interact with the UI to understand relationships between variables and get insights of diagnostic nature (e.g., correlations, statistical analysis, clustering). 
        The dashboard should include enough text to show the reasoning on the questions that are trying to be solved as well as the conclusions of the analysis.
        """
    )

with tab_pred:
    st.header("Diagnostic Analytics")
    st.markdown(

        """
        Propose a pipeline where a user can interact with the UI to understand relationships between variables and get insights of diagnostic nature (e.g., correlations, statistical analysis, clustering). 
        The dashboard should include enough text to show the reasoning on the questions that are trying to be solved as well as the conclusions of the analysis.
        """
    )
