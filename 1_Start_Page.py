import streamlit as st

st.set_page_config(
    page_title="Predicting Colorectal Cancer Survivability Dashboard",layout="wide")

st.sidebar.success("Select a Tab Above")

st.sidebar.image("./assets/Colorectal Cancer Logo.png",)

st.markdown("""
    <h1 style="
        font-size: 60px;
        font-weight: 700;
        text-align: center;
        color: #1261B5;
    ">
    Predicting Colorectal Cancer Survivability Dashboard
    </h1>
""", unsafe_allow_html=True)

with st.container():
    st.write("""**What is colorectal cancer?**

Colorectal cancer, also known as bowel cancer, colon cancer, or rectal cancer, 
is the development of cancer from the colon or rectum. 
It is the consequence of uncontrolled growth of colon cells that can invade/spread to other parts of the body.""")

    st.write("""**What is it important to investigate?**

Worldwide, colorectal cancer is the third most common cancer,
with over 1.9 million new cases diagnosed and more than 900,000 deaths annually in recent years.""")

    st.write("""**What can you find on this dashboard?**

This dashboard includes ...""")








