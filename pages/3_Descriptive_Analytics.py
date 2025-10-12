import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

def load_data():
    df = pd.read_csv("colorectal_cancer_dataset.csv")
    return df

df = load_data()

question = st.selectbox(
    "Choose a question to explore:",
    [
        "Q1: What is the age and gender distribution of colorectal cancer in the dataset?",
        "Q2: What is the distribution of cancer stages among patients?",
        "Q3: How does survivability vary across cancer stages?",
        "Q4: Among patients who did not survive past 5 years, how many had a history of smoking?",
        "Q5: How does the average tumor size vary across cancer stages?"
    ]
)

if "Q1" in question:
    st.subheader("Q1: Age and Gender Distribution")
    st.write("Analyze the distribution of age and gender among colorectal cancer patients.")

    # Clean
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[(df["Age"] > 0) & (df["Age"] < 120)]
    df["Gender"] = df["Gender"].str.strip().str.title()

    col1, col2 = st.columns(2)

    with col1:
        fig_age = px.histogram(df, x="Age", nbins=20, color="Gender", marginal="box",
                               title="Age Distribution by Gender", opacity=0.7)
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        gender_counts = df["Gender"].value_counts()
        fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index,
                            title="Gender Distribution (%)", color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_gender, use_container_width=True)

    st.write("**Descriptive Stats:**")
    st.dataframe(df.groupby("Gender")["Age"].describe().round(2))


