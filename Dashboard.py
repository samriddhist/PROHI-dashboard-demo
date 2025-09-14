import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Page Setup ---
st.set_page_config(page_title="Amanuel's Dashboard", layout="wide")
st.title("Amanuel Dashboard")

# --- Sidebar ---
with st.sidebar:
    name = st.text_input("Enter your name")
    date = st.date_input("Pick a date")
    option = st.selectbox("Choose an option", ["Option 1", "Option 2", "Option 3"])

# --- Data ---
data = pd.DataFrame({"A": np.random.randint(1, 100, 10), "B": np.random.randn(10)})
st.subheader("Sample Data Table")
st.dataframe(data)

# --- Chart ---
st.subheader("Sample Chart")
st.line_chart(data.set_index("A"))  