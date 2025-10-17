# PROHI Dashboard Example

# [Cancer Survivability prediction - ColorectalCrew]

_You can modify this README file with all the information that your team consider relevant for a technical audience who would like to understand your project or to run it in the future._

_Note that this file is written in **MarkDown** language. A reference is available here: <https://www.markdownguide.org/basic-syntax/>_

Include the name, logo and images refering to your project



![Predicting Colorectal Cancer Survivability](./assets/project-logo.jpg)

### Introduction

[Predicting Colorectal Cancer Survivability by ColorectalCrew] is an interactive web dashboard that predicts **5-year survivability** for colorectal-cancer patients and explains each prediction with **SHAP** and **counterfactual “what-if”** analysis. The dashboard integrates descriptive, diagnostic, predictive, and prescriptive analytics to support exploration, risk stratification practice, and transparent model reasoning on an openly available dataset.

The problem we address is the lack of **accessible, data-driven tools** that combine demographic, lifestyle, and clinical variables into an individualized survivability estimate usable for planning treatment and discussing expectations.

Our proposed solution is valuable because it demonstrates an end-to-end, **reproducible** pipeline—from data cleaning and EDA to model training and explanation—delivered as a simple web app that others can run, inspect, and extend.

### Setup & excecution

  ##  Retrieve Our Code

To Clone this repository run: https://github.com/samriddhist/PROHI-dashboard-demo

### Dataset
  Visit: https://www.kaggle.com/datasets/ankushpanday2/colorectal-cancer-global-dataset-and-predictions?select=colorectal_cancer_dataset.csv


### System description
The dashboard implements a five-stage analytical pipeline:

| Stage | Description | Output / File |
|--------|-------------|----------------|
| **1 Exploratory Data Analysis (EDA)** | Technical inspection of dataset, datatype optimization, missing-value validation, and categorical conversion for modeling readiness. | `jupyter-notebooks/data-analysis-pipeline.ipynb` |
| **2 Descriptive Analytics** | Visualization of demographics (age, gender), cancer stage distribution, tumor sizes, and lifestyle factors (smoking, BMI, screening). | `pages/3_Descriptive_Analytics.py` |
| **3 Diagnostic Analytics** | Examines relationships and correlations — e.g., survivability vs stage, smoking, or screening; validates hypotheses statistically. | `pages/4_Diagnostic_Analytics.py` |
| **4 Predictive Analytics** | Implements a **K-Nearest Neighbors (KNN)** model (`final_knn_k3_pipeline.joblib`) to classify 5-year survivability. | `pages/5_Predictive_Analytics.py` |
| **5 Prescriptive Analytics** | Uses **SHAP** and **DiCE** to provide model explanations, feature importance plots, and counterfactual “what-if” scenarios. | `pages/6_Prescriptive_Analytics.py` |

All stages are integrated through an interactive **Streamlit** dashboard that allows users to explore, test, and interpret results.

###  Project Architecture
PROHI-Dashboard-Demo/
│
├── assets/
│ ├── Colorectal Cancer Logo.png
│ ├── example-image.jpg
│ ├── project-logo.jpg
│
├── jupyter-notebooks/
│ ├── assets/
│ │ ├── trained_model.joblib
│ │ └── trained_model.pickle
│ ├── Figs/
│ │ ├── counterfactuals_row0.csv
│ │ ├── permutation_importance_top20.csv
│ │ ├── Permutation_Importance_Top20.png
│ │ ├── SHAP_global_kernel.png
│ │ └── SHAP_local_kernel_row0.png
│ ├── colorectal_cancer_dataset.csv
│ ├── data-analysis-pipeline.ipynb
│ ├── final_knn_k3_pipeline.joblib
│ ├── postprocessed_colorectal_cancer_dataset.csv
│ └── trained_model.pickle
│
├── pages/
│ ├── 1_Start_Page.py
│ ├── 2_About.py
│ ├── 3_Descriptive_Analytics.py
│ ├── 4_Diagnostic_Analytics.py
│ ├── 5_Predictive_Analytics.py
│ └── 6_Prescriptive_Analytics.py
│
├── requirements.txt
├── README.md
└── .gitignore

  ##  Explanation

- **`assets/`** – Contains branding files and project images used in the dashboard.  
- **`jupyter-notebooks/`** – Contains the full data pipeline notebook, trained models, dataset files, and generated visual outputs.  
  - `assets/` → Pre-trained models (`.joblib` / `.pickle`)  
  - `Figs/` → SHAP plots, permutation importance graphs, and counterfactual results.  
- **`pages/`** – Contains all Streamlit modules, each representing one analytical phase:
  - `3_Descriptive_Analytics.py` → Demographic and stage visualization  
  - `4_Diagnostic_Analytics.py` → Correlation & statistical insights  
  - `5_Predictive_Analytics.py` → KNN model integration  
  - `6_Prescriptive_Analytics.py` → SHAP explainability & what-if analysis  
- **`requirements.txt`** – Lists dependencies for reproducibility.  
- **`README.md`** – Full project documentation and usage guide.  

###  Pre-Trained Model Note

If you encounter an error indicating that the model file does not exist, ensure that the pre-trained model  
(`trained_model.joblib` or `trained_model.pickle`) is available inside the `/jupyter-notebooks/assets/` directory.

If the model file is missing, open and execute the following notebook: jupyter-notebooks/data-analysis-pipeline.ipynb


### Dependencies

Tested on Python 3.12.7 with the following packages:
  - Jupyter v1.1.1
  - Streamlit v1.46.1
  - Seaborn v0.13.2
  - Plotly v6.2.0y
  - Scikit-Learn v1.7.0
  - shap v0.48.0


### Contributors

| Name | Role | Main Contributions |
|------|------|--------------------|
| **Amanuel** | Data Analysis & Preprocessing | Led **Exploratory Data Analysis (EDA)**, performed data cleaning, validation, and ensured dataset consistency and readiness for modeling. |
| **Jakob** | Data Science & Model Development | Conducted **Feature Engineering**, **Model Evaluation**, and contributed to **Descriptive** and **Predictive Analytics** phases. |
| **Samriddhi** | Dashboard Developer & Optimization Lead | Designed and optimized the **Streamlit Dashboard**, integrated analytical modules, and managed the overall project repository structure. |
| **Dharinisri** | Analytical Interpretation & Explainability | Led **Diagnostic Analytics** and **Prescriptive Analytics**, implemented **SHAP** and **Counterfactual Explainability** |
| **Carlos** | Usability Testing & Model Refinement | Conducted **usability testing**, gathered **user feedback**, and assisted in **model debugging and optimization** based on real-world test results. |

> *Team  collaborated across all stages — from data exploration and model development to explainability, testing, and final deployment — ensuring analytical robustness and end-user usability.*

### Installation

Run the commands below in a terminal to configure the project and install the package dependencies for the first time.

If you are using Mac, you may need to follow install Xcode. Check the official Streamlit documentation [here](https://docs.streamlit.io/get-started/installation/command-line#prerequisites). 

1. Create the environment with `python -m venv env`
2. Activate the virtual environment for Python
   - `source env/bin/activate` [in Linux/Mac]
   - `.\env\Scripts\activate.bat` [in Windows command prompt]
   - `.\env\Scripts\Activate.ps1` [in Windows PowerShell]
3. Make sure that your terminal is in the environment (`env`) not in the global Python installation
4. Install required packages `pip install -r ./requirements.txt`
5. Check that everything is ok running `streamlit hello`

### Execution

To run the dashboard execute the following command:

```
> streamlit run Dashboard.py
# If the command above fails, use:
> python -m streamlit run Dashboard.py
```


