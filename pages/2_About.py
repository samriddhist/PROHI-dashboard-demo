import streamlit as st

st.set_page_config(page_title="About", layout="wide")

st.sidebar.image("./assets/Colorectal Cancer Logo.png")
st.sidebar.success("Select a tab above.")

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
            transition: transform 0.2s, box-shadow 0.2s;
            margin-bottom: 20px;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }
        .contact {
            font-size: 16px;
            color: #333;
        }
        a {
            color: #1261B5;
            text-decoration: none;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>About This Dashboard</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section">
    <h2 style="color:#1261B5;">Dataset Information</h2>
    <div class="card">
        <p style="color:black; font-size:16px;">
            This dashboard uses the <a href="https://www.kaggle.com/datasets/ankushpanday2/colorectal-cancer-global-dataset-and-predictions/data" target="_blank">Colorectal Cancer Global Dataset</a>.
            It contains <b>28 features</b> including demographic, medical, and lifestyle factors relevant to colorectal cancer.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="section">
    <h2 style="color:#1261B5;">Group Members & Contact</h2>
    <div class="card contact">
             <p style="color:black; font-size:16px;">
        <b>Amanuel Teklehaimanot</b><br>Email: <a href="mailto:amanit_kagts@yahoo.co.uk">amanit_kagts@yahoo.co.uk</a>
    </div>
    <div class="card contact">
             <p style="color:black; font-size:16px;">
        <b>Carlos Etornam Adabe</b><br>Email: <a href="mailto:cavitydrey@gmail.com">cavitydrey@gmail.com</a>
    </div>
    <div class="card contact">
             <p style="color:black; font-size:16px;">
        <b>Dharinisri Magudapathy Saravanakumar</b><br>Email: <a href="mailto:dharinisri.s@gmail.com">dharinisri.s@gmail.com</a>
    </div>
    <div class="card contact">
             <p style="color:black; font-size:16px;">
        <b>Jakob Lyckström</b><br>Email: <a href="mailto:jakob.lyckstrom@gmail.com">jakob.lyckstrom@gmail.com</a>
    </div>
    <div class="card contact">
             <p style="color:black; font-size:16px;">
        <b>Samriddhi Tripathi</b><br>Email: <a href="mailto:tripathisamriddhi@gmail.com">tripathisamriddhi@gmail.com</a>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section">
    <h2 style="color:#1261B5;">References</h2>
    <div class="card contact">
        <ul style="color:black; font-size:16px; text-align:left;">
            <li>Kaggle Dataset: <a href="https://www.kaggle.com/datasets/ankushpanday2/colorectal-cancer-global-dataset-and-predictions/data" target="_blank" style="color:black;">Colorectal Cancer Dataset</a></li>
            <li>Streamlit: <a href="https://docs.streamlit.io/" target="_blank" style="color:black;">Streamlit Documentation</a></li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)
