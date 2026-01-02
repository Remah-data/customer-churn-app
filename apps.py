import streamlit as st
import pandas as pd
import joblib

# =============================
# Page Config
# =============================
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# =============================
# Session State
# =============================
if "page" not in st.session_state:
    st.session_state.page = "main"

# =============================
# Background & Styles
# =============================
BG_IMAGE = "https://images.unsplash.com/photo-1601597111158-2fceff292cdc"

st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(rgba(0,0,0,0.82), rgba(0,0,0,0.82)),
    url("{BG_IMAGE}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

h1, h2, h3 {{
    color: white !important;
    text-align: center;
}}

p, label {{
    color: #e5e5e5 !important;
}}

.stButton>button {{
    background-color: #0f0f0f;
    color: #d4af37;
    border: 1px solid #d4af37;
    border-radius: 10px;
    padding: 0.6em 2em;
}}

.card {{
    background: rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 25px;
    border: 1px solid rgba(212,175,55,0.25);
    margin-top: 15px;
}}
</style>
""", unsafe_allow_html=True)

# =============================
# Load Model
# =============================
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

model = load_model()

# =====================================================
# ===================== ABOUT PAGE =====================
# =====================================================
if st.session_state.page == "about":

    # st.markdown("<h1>About Me</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="text-align:center;">
        <h3>Remah Ramadan</h3>
        <p>Data Analyst & Junior Data Scientist</p>
        <p>Passionate about data analysis, machine learning, and data-driven solutions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <a href="https://github.com/Remah-data" target="_blank">
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="48">
            </a>
            <p>GitHub</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <a href="https://www.kaggle.com/remahramadan10" target="_blank">
                <img src="https://www.vectorlogo.zone/logos/kaggle/kaggle-icon.svg" width="48">
            </a>
            <p>Kaggle</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <a href="https://bit.ly/4lDIiPy" target="_blank">
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="48">
            </a>
            <p>LinkedIn</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("⬅ Back to Home"):
        st.session_state.page = "main"
        st.rerun()

# =====================================================
# ===================== MAIN PAGE ======================
# =====================================================
if st.session_state.page == "main":

    st.markdown("<h1>Customer Churn Prediction</h1>", unsafe_allow_html=True)

    # -------- Customer Information Card --------
    st.markdown("""
    <div class="card">
        <h2>Customer Information</h2>
    </div>
    """, unsafe_allow_html=True)

    # -------- Input Form --------
    col1, col2 = st.columns(2)

    with col1:
        CreditScore = st.slider("Credit Score", 350, 850, 650)
        Age = st.slider("Age", 18, 92, 35)
        Tenure = st.slider("Tenure (Years)", 0, 10, 5)
        NumOfProducts = st.slider("Number of Products", 1, 4, 1)
        Balance = st.number_input("Balance", value=0.0)

    with col2:
        Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        HasCrCard = st.selectbox("Has Credit Card", [0, 1])
        IsActiveMember = st.selectbox("Is Active Member", [0, 1])
        EstimatedSalary = st.number_input("Estimated Salary", value=50000.0)

    # -------- Run Prediction Button (AS IS) --------
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    run_pred = st.button("Run Prediction")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------- Prepare Data --------
    gender = 1 if Gender == "Male" else 0
    geo_germany = 1 if Geography == "Germany" else 0
    geo_spain = 1 if Geography == "Spain" else 0

    input_df = pd.DataFrame([[
        CreditScore, gender, Age, Tenure, Balance,
        NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
        geo_germany, geo_spain
    ]], columns=[
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember',
        'EstimatedSalary', 'Geography_Germany', 'Geography_Spain'
    ])

    # -------- Prediction Result --------
    if run_pred:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        color = "#c0392b" if pred == 1 else "#27ae60"
        status = "Customer likely to churn" if pred == 1 else "Customer likely to stay"
        prob = proba[1] if pred == 1 else proba[0]

        st.markdown(f"""
        <div class="card" style="border-left:6px solid {color};">
            <h3>{status}</h3>
            <p>Probability: <b>{prob:.2%}</b></p>
        </div>
        """, unsafe_allow_html=True)

    # -------- About Me Button (BOTTOM) --------
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("About Me"):
            st.session_state.page = "about"
            st.rerun()
