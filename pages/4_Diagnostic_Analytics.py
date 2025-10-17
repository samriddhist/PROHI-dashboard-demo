import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Diagnostic Analytics", layout="wide")

st.sidebar.success("Select a tab above.")
st.sidebar.image("./assets/Colorectal Cancer Logo.png")

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

st.markdown("""
<div style="text-align:center; padding: 10px; background-color:#1261B5; border-radius:8px;">
    <p style="font-size:18px; color:white;">
      This page allows you to explore diagnostic insights from the colorectal cancer dataset. 
You can analyze relationships between variables (correlation) or group patients into clusters based on clinical and lifestyle features. 
Use the interactive options below to select features, filter data, and understand patterns in the dataset.
    </p>
</div>
""", unsafe_allow_html=True)


df = pd.read_csv("jupyter-notebooks/postprocessed_colorectal_cancer_dataset.csv", sep=";")

analysis = st.selectbox(
    "Select a Diagnostic Analysis:",
    [
        "Correlation between Key Features",
        "Patient Clustering by Risk Factors and Outcomes"
    ]
)

if "Correlation" in analysis:
    st.subheader("Feature Correlation Analysis")
    st.write("Examine how key patient features relate to each other. Correlation helps identify which variables tend to increase or decrease together.")

    df['Cancer_Stage_Encoded'] = df['Cancer_Stage'].map({'Localized': 1, 'Regional': 2, 'Metastatic': 3})
    df['Survival_Prediction_Encoded'] = df['Survival_Prediction'].map({'No': 0, 'Yes': 1})

    st.write("### Select features to correlate")
    available_features = [
        "Age", "Tumor_Size_mm", "Cancer_Stage_Encoded",
        "Survival_Prediction_Encoded"
    ]

    selected_features = st.multiselect(
        "Pick at least two features:",
        available_features,
        default=["Age", "Tumor_Size_mm"]
    )

    if len(selected_features) >= 2:
        corr_matrix = df[selected_features].corr(method="spearman")

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Correlation Matrix")
            st.dataframe(corr_matrix.round(3))
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Spearman Correlation Heatmap")
            st.pyplot(fig)

        st.write("""
         **Interpretation:** 
        Values close to 1 or -1 indicate strong positive or negative relationships respectively. 
        Use this insight to understand which features may influence outcomes such as survival or cancer progression.
        """)
    else:
        st.warning("Please select at least two features to view correlations.")


elif "Clustering" in analysis:
    st.subheader("Patient Clustering by Risk Factors and Outcomes")
    st.write("""
    Group patients into clusters based on selected clinical and lifestyle features. 
    Clustering can reveal patterns in risk factors and outcomes, helping identify subgroups of patients with similar characteristics.
    """)

    available_features = [
        'Age', 'Cancer_Stage', 'Tumor_Size_mm', 'Family_History', 'Smoking_History',
        'Alcohol_Consumption', 'Obesity_BMI', 'Diet_Risk', 'Physical_Activity',
        'Diabetes', 'Genetic_Mutation', 'Screening_History', 'Early_Detection', 'Treatment_Type'
    ]
    
    selected_features = st.multiselect(
        "Select features to include in clustering:",
        available_features,
        default=['Age', 'Cancer_Stage', 'Tumor_Size_mm', 'Smoking_History']
    )

    if len(selected_features) < 2:
        st.warning("Please select at least 2 features for clustering.")
    else:
        n_clusters = st.slider("Select number of clusters:", 2, 6, 3)

        df_cluster = df[selected_features].dropna()
        num_cols = [c for c in selected_features if df[c].dtype in [np.int64, np.float64]]
        cat_cols = [c for c in selected_features if c not in num_cols]

        preprocess = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first'), cat_cols)
        ])

        pipe = Pipeline([
            ('prep', preprocess),
            ('kmeans', KMeans(n_clusters=n_clusters, random_state=42, n_init='auto'))
        ])

        df['Cluster'] = pipe.fit_predict(df_cluster)

        st.markdown("### Cluster Profiles")
        agg_dict = {col: (np.mean if col in num_cols else lambda x: x.value_counts().index[0])
                    for col in selected_features}
        cluster_profile = df.groupby('Cluster').agg(agg_dict)
        st.dataframe(cluster_profile.round(2))

        st.markdown("<hr>", unsafe_allow_html=True)

        if 'Survival_5_years' in df.columns:
            tab = pd.crosstab(df['Cluster'], df['Survival_5_years'], normalize='index') * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(tab, annot=True, fmt=".1f", cmap="Greens", ax=ax)
            ax.set_title("5-Year Survival Rate (%) per Cluster")
            st.pyplot(fig)

        X_trans = pipe.named_steps['prep'].transform(df_cluster)
        pca = PCA(n_components=2).fit_transform(X_trans)
        pca_df = pd.DataFrame(pca, columns=["PC1", "PC2"])
        pca_df["Cluster"] = df['Cluster']

        fig_pca = px.scatter(
            pca_df, x="PC1", y="PC2",
            color=pca_df["Cluster"].astype(str),
            title="Patient Clusters by Risk Factors (PCA 2D)",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig_pca.update_layout(
            xaxis=dict(range=[-4, 4]),
            yaxis=dict(range=[-4, 4])
        )

        st.plotly_chart(fig_pca, use_container_width=True)

        st.write("""
        **Interpretation:** 
        Clusters group patients with similar risk profiles and outcomes. 
        Review the cluster profiles and PCA plot to identify high-risk groups and patterns in the dataset.
        """)
