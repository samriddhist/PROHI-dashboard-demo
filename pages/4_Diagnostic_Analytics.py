import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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

df = pd.read_csv("jupyter-notebooks/postprocessed_colorectal_cancer_dataset.csv", sep=";")

analysis = st.selectbox(
    "Select a Diagnostic Analysis:",
    [
        "Correlation between Cancer Stage, Age, and Survival Prediction",
        "Patient Clustering by Risk Factors and Outcomes"
    ]
)

if "Correlation between Cancer Stage" in analysis:
    st.subheader("Correlation: Cancer Stage, Age, and Survival Prediction")

    method = st.selectbox(
        "Select Correlation Method:",
        ["Spearman (for ordinal data)", "Pearson", "Kendall"]
    )

    df['Cancer_Stage_Encoded'] = df['Cancer_Stage'].map({'Localized': 1, 'Regional': 2, 'Metastatic': 3})
    df['Survival_Prediction_Encoded'] = df['Survival_Prediction'].map({'No': 0, 'Yes': 1})

    corr_data = df[['Age', 'Cancer_Stage_Encoded', 'Survival_Prediction_Encoded']].dropna()
    method_map = {"Spearman (for ordinal data)": "spearman", "Pearson": "pearson", "Kendall": "kendall"}
    corr_matrix = corr_data.corr(method=method_map[method])

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Correlation Matrix")
        st.dataframe(corr_matrix.round(3))
    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title(f"{method} Correlation Heatmap")
        st.pyplot(fig)

elif "Patient Clustering" in analysis:
    st.subheader("Patient Clustering by Risk Factors and Outcomes")

    n_clusters = st.slider("Select number of clusters:", 2, 6, 3)

    features = [
        'Age','Cancer_Stage','Tumor_Size_mm','Family_History','Smoking_History',
        'Alcohol_Consumption','Obesity_BMI','Diet_Risk','Physical_Activity',
        'Diabetes','Genetic_Mutation','Screening_History','Early_Detection','Treatment_Type'
    ]

    df_cluster = df[features].dropna()
    num_cols = ['Age','Tumor_Size_mm']
    cat_cols = [c for c in features if c not in num_cols]

    preprocess = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ])

    pipe = Pipeline([
        ('prep', preprocess),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42, n_init='auto'))
    ])

    df['Cluster'] = pipe.fit_predict(df_cluster)

    st.write("### Cluster Profiles")
    cluster_profile = df.groupby('Cluster').agg({
        'Age': 'mean',
        'Tumor_Size_mm': 'mean',
        'Cancer_Stage': lambda x: x.value_counts().index[0],
        'Smoking_History': lambda x: x.value_counts().index[0],
        'Alcohol_Consumption': lambda x: x.value_counts().index[0],
        'Treatment_Type': lambda x: x.value_counts().index[0]
    })
    st.dataframe(cluster_profile.round(2))

    tab = pd.crosstab(df['Cluster'], df['Survival_5_years'], normalize='index')*100
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(tab, annot=True, fmt=".1f", cmap="Greens", ax=ax)
    ax.set_title("5-Year Survival Rate (%) per Cluster")
    st.pyplot(fig)

    X_trans = pipe.named_steps['prep'].transform(df_cluster)
    pca = PCA(n_components=2).fit_transform(X_trans)
    pca_df = pd.DataFrame(pca, columns=["PC1", "PC2"])
    pca_df["Cluster"] = df['Cluster']

    fig_pca = px.scatter(
        pca_df, x="PC1", y="PC2", color=pca_df["Cluster"].astype(str),
        title="Patient Clusters by Risk Factors (PCA 2D)",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_pca, use_container_width=True)


