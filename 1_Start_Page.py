import streamlit as st

st.set_page_config(page_title="Colorectal Cancer Dashboard", layout="wide")

st.sidebar.image("./assets/Colorectal Cancer Logo.png")
st.sidebar.success("Select a Tab Above")

st.markdown("""
    <style>
        .main {
            background-color: #F8FAFD;
        }
        .hero {
            background: linear-gradient(90deg, #1261B5 100%, #5FA8D3 0%);
            padding: 60px 20px;
            border-radius: 12px;
            color: white;
            text-align: center;
        }
        .hero h1 {
            font-size: 50px;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .hero p {
            font-size: 20px;
            font-weight: 300;
            margin-bottom: 0px;
        }
        .section {
            padding: 40px 20px;
            text-align: center;
        }
        .card {
            background-color: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            height: 100%;
            transition: transform 0.25s, box-shadow 0.25s;
            text-decoration: none;
            color: inherit;
            display: block;
            border: 2px solid transparent;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(18,97,181,0.2);
            border: 2px solid #1261B5;
        }
        .card h3 {
            color: #1261B5;
            margin: 10px 0 0 0;
        }
        .card p {
            color: #333;
            font-size: 15px;
            margin-top: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Predicting Colorectal Cancer Survivability</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section">
    <h2 style="color:#1261B5;">About Colorectal Cancer</h2>
    <p style="max-width:800px; margin:auto; font-size:17px; color:#1261B5;">
        Colorectal cancer arises from abnormal cell growth in the colon or rectum. 
        It is one of the most common cancers globally, with over <b>1.9 million new cases</b> and <b>900,000 deaths annually</b>. 
        Understanding the patterns behind survivability is crucial for prevention, early detection, and effective treatment.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section">
    <h2 style="color:#1261B5;">What You Can Explore</h2>
</div>
""", unsafe_allow_html=True)

top_row = st.columns(4)

with top_row[0]:
    st.markdown("""
     <a href="/Descriptive_Analytics" target="_self" class="card">
        <h3>📊 Descriptive Analytics</h3>
     </a>
    """, unsafe_allow_html=True)

with top_row[1]:
    st.markdown("""
     <a href="/Diagnostic_Analytics" target="_self" class="card">
        <h3>🧠 Diagnostic Analytics</h3>
     </a>
    """, unsafe_allow_html=True)

with top_row[2]:
    st.markdown("""
     <a href="/Predictive_Analytics" target="_self" class="card">
        <h3>📈 Predictive Analytics</h3>
     </a>
    """, unsafe_allow_html=True)

with top_row[3]:
    st.markdown("""
     <a href="/Prescriptive_Analytics" target="_self" class="card">
        <h3>💡 Prescriptive Analytics</h3>
     </a>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


