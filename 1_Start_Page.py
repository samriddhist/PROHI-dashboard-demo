import streamlit as st

st.sidebar.success("Select a Tab Above")

st.sidebar.image("./assets/Colorectal Cancer Logo.png",)

col1, col2, col3 = st.columns([1, 4, 1]) 

with col1:
    st.image("./assets/Colorectal Cancer Logo.png", width=80)  

with col2:
    st.markdown("""
        <h1 style="
            font-size: 50px;
            font-weight: 700;
            text-align: center;
            color: #1261B5;
        ">
        Predicting Colorectal Cancer Survivability Dashboard
        </h1>
    """, unsafe_allow_html=True)

with col3:
    st.image("./assets/Colorectal Cancer Logo.png", width=80)  


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








