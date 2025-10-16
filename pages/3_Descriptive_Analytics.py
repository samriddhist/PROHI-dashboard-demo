import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Descriptive Analytics", layout="wide")
st.sidebar.success("Select a tab above.")
st.sidebar.image("./assets/Colorectal Cancer Logo.png")

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

st.markdown("""
<div style="text-align:center; padding: 10px; background-color:#1261B5; border-radius:8px;">
    <p style="font-size:18px; color:white;">
        Explore key characteristics of the colorectal cancer dataset using interactive charts and tables.
        Use the dropdowns and sliders to filter data and gain insights into patient demographics, cancer stages, survivability, smoking history, and tumor size.
    </p>
</div>
""", unsafe_allow_html=True)


df = pd.read_csv("jupyter-notebooks/postprocessed_colorectal_cancer_dataset.csv", sep=";")

question = st.selectbox(
    "Select an analysis:",
    [
        "Age and Gender Distribution",
        "Distribution of Cancer Stages",
        "Survivability Across Cancer Stages",
        "Smoking History Among Non-Survivors",
        "Tumor Size Across Stages"
    ]
)

if question == "Age and Gender Distribution":
    st.subheader("Age and Gender Distribution")
    st.write("Understand the age distribution and gender composition of patients in the dataset. Use the slider to focus on specific age ranges.")

    min_age = int(df["Age"].min())
    max_age = int(df["Age"].max())
    age_range = st.slider("Select age range:", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    df_filtered = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

    col1, col2 = st.columns(2)
    with col1:
        fig_age = px.histogram(df_filtered, x="Age", nbins=20, color="Gender",
                               title="Age Distribution by Gender", opacity=0.7)
        st.plotly_chart(fig_age, use_container_width=True)
    with col2:
        gender_counts = df_filtered["Gender"].value_counts()
        fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index,
                            title="Gender Distribution (%)",
                            color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_gender, use_container_width=True)

    st.write("**Descriptive Stats:**")
    st.dataframe(df_filtered.groupby("Gender")["Age"].describe().round(2))


elif question == "Distribution of Cancer Stages":
    st.subheader("Distribution of Cancer Stages")
    st.write("See how patients are distributed across different cancer stages. Select a specific stage or view all patients together.")

    selected_stage = st.radio(
        "Select a Cancer Stage:",
        options=["All"] + sorted(df["Cancer_Stage"].unique().tolist()),
        horizontal=True
    )

    df_stage = df if selected_stage == "All" else df[df["Cancer_Stage"] == selected_stage]
    stage_counts = df_stage["Cancer_Stage"].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        fig_bar = px.bar(
            x=stage_counts.index, y=stage_counts.values,
            labels={"x": "Cancer Stage", "y": "Number of Patients"},
            title="Distribution of Cancer Stages",
            color=stage_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            values=stage_counts.values, names=stage_counts.index,
            title="Cancer Stage Distribution (%)",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_pie, use_container_width=True)


elif question == "Survivability Across Cancer Stages":
    st.subheader("Survivability Across Cancer Stages")
    st.write("Compare the 5-year survival rate across different cancer stages to understand prognosis trends.")

    df["Survival_5_years"] = df["Survival_5_years"].map({"Yes": 1, "No": 0})
    df_clean = df.dropna(subset=["Cancer_Stage", "Survival_5_years"])

    selected_stage = st.radio(
        "Select a Cancer Stage:",
        options=["All"] + sorted(df_clean["Cancer_Stage"].unique().tolist()),
        horizontal=True
    )

    df_filtered = df_clean if selected_stage == "All" else df_clean[df_clean["Cancer_Stage"] == selected_stage]
    surv = df_filtered.groupby("Cancer_Stage")["Survival_5_years"].mean().reset_index()
    surv["Survival_5_years"] *= 100

    fig = px.bar(
        surv, x="Cancer_Stage", y="Survival_5_years", color="Cancer_Stage",
        title="5-Year Survival Rates by Cancer Stage",
        labels={"Survival_5_years": "Survival Rate (%)"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(surv.rename(columns={"Survival_5_years": "Survival Rate (%)"}))


elif question == "Smoking History Among Non-Survivors":
    st.subheader("Smoking History Among Non-Survivors")
    st.write("Analyze the smoking habits of patients who did not survive past 5 years. Filter by gender and cancer stage.")

    df["Survival_5_years"] = df["Survival_5_years"].map({"Yes": 1, "No": 0})
    df["Smoking_History"] = df["Smoking_History"].map({"Yes": 1, "No": 0})

    selected_gender = st.radio(
        "Select Gender:",
        options=["All", "M", "F"],
        index=0,
        horizontal=True
    )
    selected_stage = st.radio(
        "Select Cancer Stage:",
        options=["All"] + sorted(df["Cancer_Stage"].unique().tolist()),
        horizontal=True
    )

    df_filtered = df.copy()
    if selected_gender != "All":
        df_filtered = df_filtered[df_filtered["Gender"] == selected_gender]
    if selected_stage != "All":
        df_filtered = df_filtered[df_filtered["Cancer_Stage"] == selected_stage]

    non_survivors = df_filtered[df_filtered["Survival_5_years"] == 0]
    counts = non_survivors["Smoking_History"].value_counts().rename({0: "Non-Smoker", 1: "Smoker"})

    fig = px.bar(
        x=counts.index, y=counts.values, text=counts.values,
        title="Smoking History Among Non-Survivors",
        labels={"x": "Smoking History", "y": "Number of Patients"},
        color=counts.index,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)


elif question == "Tumor Size Across Stages":
    st.subheader("Tumor Size Across Stages")
    st.write("Investigate how tumor size varies across cancer stages. Larger tumors may indicate more advanced disease.")

    selected_stage = st.radio(
        "Select Cancer Stage:",
        options=["All"] + sorted(df["Cancer_Stage"].unique().tolist()),
        horizontal=True
    )

    df_filtered = df if selected_stage == "All" else df[df["Cancer_Stage"] == selected_stage]

    fig = px.box(
        df_filtered, x="Cancer_Stage", y="Tumor_Size_mm", color="Cancer_Stage",
        title="Tumor Size Distribution by Cancer Stage",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Summary Statistics:**")
    st.dataframe(df_filtered.groupby("Cancer_Stage")["Tumor_Size_mm"].agg(["count", "mean", "median", "std"]).round(2))
