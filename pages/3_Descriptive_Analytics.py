import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import seaborn as sns
import matplotlib.pyplot as plt


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

df = pd.read_csv ("jupyter-notebooks/colorectal_cancer_dataset.csv", sep=";")
                  
question = st.selectbox(
    "Select an analysis:",
    ["Age and Gender Distribution",
     "Distribution of Cancer Stages",
     "Survivability across Cancer Stages",
     "Smoking History among Non-survivors",
     "Tumor Size across Stages"]
)

if "Age and Gender Distribution" in question:
    st.subheader("Age and Gender Distribution")
    st.write("What is the age and gender distribution of colorectal cancer patients in the dataset?")

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[(df["Age"] > 0) & (df["Age"] < 120)]
    df["Gender"] = df["Gender"].str.strip().str.title()

    unique_genders = df["Gender"].dropna().unique().tolist()
    selected_genders = st.multiselect(
        "Select gender(s) to include:",
        options=unique_genders,
        default=unique_genders
    )

    min_age = int(df["Age"].min())
    max_age = int(df["Age"].max())
    age_range = st.slider(
        "Select age range:",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    )

    df = df[df["Gender"].isin(selected_genders)]
    df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]


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

if "Distribution of Cancer Stages" in question:
    st.subheader("Distribution of Cancer Stages")
    st.write("What is the distribution of cancer stages among patients?")

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

if "Survivability Across Cancer Stages" in question:
    st.subheader("Survivability Across Cancer Stages")
    st.write("How does survivability vary across cancer stages?")

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

if "Smoking Among Non-Survivors" in question:
    st.subheader("Smoking Among Non-Survivors")
    st.write("Among patients who did not survive past 5 years, how many had a history of smoking?")

    df["Survival_5_years"] = df["Survival_5_years"].astype(str).str.lower().map({"yes":1, "no":0})
    df["Smoking_History"] = df["Smoking_History"].astype(str).str.lower().map({"yes":1, "no":0})

    non_survivors = df[df["Survival_5_years"] == 0]
    counts = non_survivors["Smoking_History"].value_counts().rename({0:"Non-Smoker", 1:"Smoker"})

    fig = px.bar(x=counts.index, y=counts.values, text=counts.values,
                 title="Smoking History Among Non-Survivors",
                 labels={"x":"Smoking History", "y":"Number of Patients"},
                 color=counts.index, color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

if "Tumor Size Variation Across Stages" in question:
    st.subheader("Tumor Size Variation Across Stages")
    st.write("How does the average tumor size vary across cancer stages?")

    df["Cancer_Stage"] = df["Cancer_Stage"].astype(str).str.strip().str.title()

    fig = px.box(df, x="Cancer_Stage", y="Tumor_Size_mm", color="Cancer_Stage",
                 title="Tumor Size Distribution by Cancer Stage",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Summary Statistics:**")
    st.dataframe(df.groupby("Cancer_Stage")["Tumor_Size_mm"].agg(["count","mean","median","std"]).round(2))



