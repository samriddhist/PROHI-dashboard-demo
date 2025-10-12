import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

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
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "..", "jupyter-notebooks", "colorectal_cancer_dataset.csv")
    df = pd.read_csv(file_path)
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

if "Q2" in question:
    st.subheader("Q2: Distribution of Cancer Stages")
    st.write("Explore how patients are distributed across different cancer stages.")

    df["Cancer_Stage"] = df["Cancer_Stage"].astype(str).str.strip().str.title()
    stage_counts = df["Cancer_Stage"].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        fig_bar = px.bar(
            x=stage_counts.index, y=stage_counts.values,
            labels={"x":"Cancer Stage", "y":"Number of Patients"},
            title="Distribution of Cancer Stages",
            color=stage_counts.index, color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            values=stage_counts.values, names=stage_counts.index,
            title="Cancer Stage Distribution (%)",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_pie, use_container_width=True)

if "Q3" in question:
    st.subheader("Q3: Survivability Across Cancer Stages")
    st.write("Compare 5-year survival rates across cancer stages.")

    df["Cancer_Stage"] = df["Cancer_Stage"].astype(str).str.strip().str.title()
    df["Survival_5_years"] = df["Survival_5_years"].str.strip().str.lower().map({"yes": 1, "no": 0})
    df_clean = df.dropna(subset=["Cancer_Stage", "Survival_5_years"])

    surv = df_clean.groupby("Cancer_Stage")["Survival_5_years"].mean().reset_index()
    surv["Survival_5_years"] *= 100

    fig = px.bar(surv, x="Cancer_Stage", y="Survival_5_years", color="Cancer_Stage",
                 title="5-Year Survival Rates by Cancer Stage",
                 labels={"Survival_5_years":"Survival Rate (%)"},
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(surv.rename(columns={"Survival_5_years":"Survival Rate (%)"}))

if "Q4" in question:
    st.subheader("Q4: Smoking Among Non-Survivors")
    st.write("Analyze how many patients who did not survive past 5 years had a history of smoking.")

    df["Survival_5_years"] = df["Survival_5_years"].astype(str).str.lower().map({"yes":1, "no":0})
    df["Smoking_History"] = df["Smoking_History"].astype(str).str.lower().map({"yes":1, "no":0})

    non_survivors = df[df["Survival_5_years"] == 0]
    counts = non_survivors["Smoking_History"].value_counts().rename({0:"Non-Smoker", 1:"Smoker"})

    fig = px.bar(x=counts.index, y=counts.values, text=counts.values,
                 title="Smoking History Among Non-Survivors",
                 labels={"x":"Smoking History", "y":"Number of Patients"},
                 color=counts.index, color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

if "Q5" in question:
    st.subheader("Q5: Tumor Size Variation Across Stages")
    st.write("Understand how average tumor size changes across different cancer stages.")

    df["Cancer_Stage"] = df["Cancer_Stage"].astype(str).str.strip().str.title()

    fig = px.box(df, x="Cancer_Stage", y="Tumor_Size_mm", color="Cancer_Stage",
                 title="Tumor Size Distribution by Cancer Stage",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Summary Statistics:**")
    st.dataframe(df.groupby("Cancer_Stage")["Tumor_Size_mm"].agg(["count","mean","median","std"]).round(2))


