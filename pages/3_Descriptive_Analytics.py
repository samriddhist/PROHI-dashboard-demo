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
    "Select an analytical question:",
    ["Q1: Age and Gender Distribution",
     "Q2: Distribution of Cancer Stages",
     "Q3: Survivability across Cancer Stages",
     "Q4: Smoking History among Non-survivors",
     "Q5: Tumor Size across Stages"]
)

if "Q1" in question:
    age_col = "Age"
    gender_col = "Gender"

    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df.loc[(df[age_col] < 0) | (df[age_col] > 120), age_col] = pd.NA

    def normalize_gender(x):
        if pd.isna(x): return "Unknown"
        s = str(x).strip().lower()
        if s in {"m","male","man"}: return "Male"
        if s in {"f","female","woman"}: return "Female"
        return "Other/Unknown"

    df[gender_col] = df[gender_col].apply(normalize_gender)

    sns.set(style="whitegrid", palette="Set2", font_scale=1.2)

    st.subheader("Age and Gender Distribution of Colorectal Cancer Patients")

    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.histplot(data=df, x=age_col, bins=20, kde=True, color="skyblue", ax=ax1)
    ax1.set_title("Age Distribution")
    ax1.set_xlabel("Age (years)")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(6,5))
    sns.countplot(data=df, x=gender_col, order=df[gender_col].value_counts().index, ax=ax2)
    ax2.set_title("Gender Distribution")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(7,5))
    sns.boxplot(data=df, x=gender_col, y=age_col, ax=ax3)
    ax3.set_title("Age Distribution by Gender")
    st.pyplot(fig3)

    st.write("### Gender Distribution Summary")
    gender_counts = df[gender_col].value_counts(dropna=False)
    gender_pct = df[gender_col].value_counts(normalize=True, dropna=False) * 100
    for g in gender_counts.index:
        st.write(f"**{g}:** {gender_counts[g]} patients ({gender_pct[g]:.1f}%)")

    st.write("### Age Descriptive Statistics (Overall)")
    st.dataframe(df[age_col].describe())

    st.write("### Age Descriptive Statistics by Gender")
    st.dataframe(df.groupby(gender_col)[age_col].describe())

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



