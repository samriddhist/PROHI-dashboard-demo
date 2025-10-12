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

    print(np.shape(df))
    df[Age] = pd.to_numeric(df[Age], errors="coerce")
    df.loc[(df[Age] < 0) | (df[Age] > 120), Age] = pd.NA

    def normalize_gender(x):
        if pd.isna(x): return "Unknown"
        s = str(x).strip().lower()
        if s in {"m","male","man"}: return "Male"
        if s in {"f","female","woman"}: return "Female"
        return "Other/Unknown"

    df[Gender] = df[Gender].apply(normalize_gender)

    sns.set(style="whitegrid", palette="Set2", font_scale=1.2)

    st.subheader("Age and Gender Distribution of Colorectal Cancer Patients")

    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.histplot(data=df, x=age_col, bins=20, kde=True, color="skyblue", ax=ax1)
    ax1.set_title("Age Distribution")
    ax1.set_xlabel("Age (years)")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(6,5))
    sns.countplot(data=df, x=Gender, order=df[Gender].value_counts().index, ax=ax2)
    ax2.set_title("Gender Distribution")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(7,5))
    sns.boxplot(data=df, x=Gender, y=Age, ax=ax3)
    ax3.set_title("Age Distribution by Gender")
    st.pyplot(fig3)

    st.write("### Gender Distribution Summary")
    gender_counts = df[Gender].value_counts(dropna=False)
    gender_pct = df[Gender].value_counts(normalize=True, dropna=False) * 100
    for g in gender_counts.index:
        st.write(f"**{g}:** {gender_counts[g]} patients ({gender_pct[g]:.1f}%)")

    st.write("### Age Descriptive Statistics (Overall)")
    st.dataframe(df[Age].describe())

    st.write("### Age Descriptive Statistics by Gender")
    st.dataframe(df.groupby(Gender)[Age].describe())
