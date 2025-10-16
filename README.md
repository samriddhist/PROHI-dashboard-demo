# PROHI Dashboard Example

# [Cancer Survivability prediction - ColorectalCrew]

_You can modify this README file with all the information that your team consider relevant for a technical audience who would like to understand your project or to run it in the future._

_Note that this file is written in **MarkDown** language. A reference is available here: <https://www.markdownguide.org/basic-syntax/>_

Include the name, logo and images refering to your project



![Predicting Colorectal Cancer Survivability](./assets/project-logo.jpg)

## Introduction

[Predicting Colorectal Cancer Survivability by ColorectalCrew] is an interactive web dashboard that predicts **5-year survivability** for colorectal-cancer patients and explains each prediction with **SHAP** and **counterfactual “what-if”** analysis. The dashboard integrates descriptive, diagnostic, predictive, and prescriptive analytics to support exploration, risk stratification practice, and transparent model reasoning on an openly available dataset.

The problem we address is the lack of **accessible, data-driven tools** that combine demographic, lifestyle, and clinical variables into an individualized survivability estimate usable for planning treatment and discussing expectations.

Our proposed solution is valuable because it demonstrates an end-to-end, **reproducible** pipeline—from data cleaning and EDA to model training and explanation—delivered as a simple web app that others can run, inspect, and extend.


## System description

### Dependencies

Tested on Python 3.12.7 with the following packages:
  - Jupyter v1.1.1
  - Streamlit v1.46.1
  - Seaborn v0.13.2
  - Plotly v6.2.0y
  - Scikit-Learn v1.7.0
  - shap v0.48.0

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


