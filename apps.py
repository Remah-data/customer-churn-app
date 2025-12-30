import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =============================
# Page Config
# =============================
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# =============================
# Background Image (Official Bank - Unsplash)
# =============================
BG_IMAGE = "https://images.unsplash.com/photo-1601597111158-2fceff292cdc"

# =============================
# Global Theme & Styles
# =============================
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
    font-weight: 800;
}}

p, label {{
    color: #e5e5e5 !important;
}}

[data-testid="stSidebar"] {{
    background: rgba(15,15,15,0.96);
    border-right: 2px solid #d4af37;
}}

[data-testid="stSidebar"] .stSlider > div > div > div > div {{
    background-color: #d4af37 !important;
}}

[data-testid="stSidebar"] .stSlider span {{
    color: #d4af37 !important;
}}

.stButton>button {{
    background-color: #0f0f0f;
    color: #d4af37;
    border: 1px solid #d4af37;
    border-radius: 10px;
    padding: 0.7em 2em;
    font-weight: 600;
}}

.stButton>button:hover {{
    background-color: #1c1c1c;
    color: #f1c40f;
    border-color: #f1c40f;
}}

.card {{
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 28px;
    border: 1px solid rgba(212,175,55,0.25);
    margin-top: 10px;
}}

hr {{
    border: 0.5px solid #2c2c2c;
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

# =============================
# Navigation
# =============================
page = st.sidebar.radio("Navigation", ["Prediction App", "About"])

# =============================
# Prediction App Page
# =============================
if page == "Prediction App":

    st.markdown("<h1>Customer Churn Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.sidebar.markdown("### Customer Information")

    CreditScore = st.sidebar.slider("Credit Score", 350, 850, 650)
    Geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    Age = st.sidebar.slider("Age", 18, 92, 35)
    Tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
    Balance = st.sidebar.number_input("Balance", value=0.0)
    NumOfProducts = st.sidebar.slider("Number of Products", 1, 4, 1)
    HasCrCard = st.sidebar.selectbox("Has Credit Card", [0, 1])
    IsActiveMember = st.sidebar.selectbox("Is Active Member", [0, 1])
    EstimatedSalary = st.sidebar.number_input("Estimated Salary", value=50000.0)

    gender_mapped = 1 if Gender == "Male" else 0
    geo_germany = 1 if Geography == "Germany" else 0
    geo_spain = 1 if Geography == "Spain" else 0

    input_df = pd.DataFrame([[
        CreditScore, gender_mapped, Age, Tenure, Balance,
        NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
        geo_germany, geo_spain
    ]], columns=[
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember',
        'EstimatedSalary', 'Geography_Germany', 'Geography_Spain'
    ])

    # =============================
    # Input Summary Card
    # =============================
    st.markdown(f"""
    <div class="card" style="text-align:center; color:white;">
        <h3>Input Summary</h3>
        <p><b>Credit Score:</b> {CreditScore}</p>
        <p><b>Age:</b> {Age} years</p>
        <p><b>Geography:</b> {Geography}</p>
        <p><b>Products:</b> {NumOfProducts}</p>
        <p><b>Active Member:</b> {"Yes" if IsActiveMember else "No"}</p>
    </div>
    """, unsafe_allow_html=True)

    # =============================
    # Run Prediction Button
    # =============================
    st.markdown('<div style="display:flex; justify-content:center; margin-top:20px;">', unsafe_allow_html=True)
    run_pred = st.button("Run Prediction")
    st.markdown('</div>', unsafe_allow_html=True)

    # =============================
    # Prediction Result
    # =============================
    if run_pred:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        if prediction == 1:
            st.markdown(f"""
            <div class="card" style="border-left:6px solid #c0392b;">
                <h3>Customer likely to churn</h3>
                <p>Predicted probability: <b>{proba[1]:.2%}</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card" style="border-left:6px solid #27ae60;">
                <h3>Customer likely to stay</h3>
                <p>Predicted probability: <b>{proba[0]:.2%}</b></p>
            </div>
            """, unsafe_allow_html=True)

    # =============================
    # Model Insights
    # =============================
    if st.checkbox("Show Model Insights"):
        st.markdown('<h4 style="color:white;">Feature Importance</h4>', unsafe_allow_html=True)

        imp_df = pd.DataFrame({
            "Feature": input_df.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0f0f0f")
        ax.set_facecolor("#0f0f0f")

        ax.barh(imp_df["Feature"], imp_df["Importance"], color="#d4af37")
        ax.set_xlabel("Importance", color="#e5e5e5")
        ax.set_ylabel("Feature", color="#e5e5e5")
        ax.tick_params(colors="#e5e5e5")

        for spine in ax.spines.values():
            spine.set_color("#2c2c2c")

        st.pyplot(fig)

# =============================
# About Page
# =============================
if page == "About":

    st.markdown('<h1 style="color:white; text-align:center;">About Me</h1>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<h3 style="text-align:center; color:white;">Remah Ramadan</h3>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align:center; color:white;">
        Data Analyst & Junior Data Scientist<br>
        Passionate about data analysis, machine learning, and building impactful data-driven solutions.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: rgba(0,0,0,0.5); border-radius: 12px; padding: 15px; text-align:center;">
            <a href="https://github.com/Remah-data" target="_blank">
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="48">
            </a>
            <p style="color:white;">GitHub</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: rgba(0,0,0,0.5); border-radius: 12px; padding: 15px; text-align:center;">
            <a href="https://www.kaggle.com/remahramadan10" target="_blank">
                <img src="https://www.vectorlogo.zone/logos/kaggle/kaggle-icon.svg" width="48">
            </a>
            <p style="color:white;">Kaggle</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: rgba(0,0,0,0.5); border-radius: 12px; padding: 15px; text-align:center;">
            <a href="https://bit.ly/4lDIiPy" target="_blank">
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="48">
            </a>
            <p style="color:white;">LinkedIn</p>
        </div>
        """, unsafe_allow_html=True)