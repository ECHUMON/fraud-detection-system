import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔐",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #ffffff;
}
.subtitle {
    font-size: 18px;
    color: #cfd8dc;
    margin-bottom: 25px;
}
.metric-card {
    background-color: #151f2e;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #263445;
    text-align: center;
}
.metric-value {
    font-size: 26px;
    font-weight: 700;
    color: #4ade80;
}
.metric-label {
    font-size: 14px;
    color: #cbd5e1;
}
.info-box {
    background-color: #102a43;
    padding: 18px;
    border-radius: 12px;
    color: #dbeafe;
    border-left: 5px solid #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# Load model and dataset
with open("models/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

data = pd.read_csv("data/fraud.csv")

# Header
st.markdown('<div class="main-title">🔐 Financial Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A machine learning-based system for detecting suspicious financial transactions using Random Forest.</div>',
    unsafe_allow_html=True
)

st.divider()

# Performance cards
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card"><div class="metric-value">99.95%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card"><div class="metric-value">96%</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card"><div class="metric-value">73%</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card"><div class="metric-value">83%</div><div class="metric-label">F1 Score</div></div>', unsafe_allow_html=True)

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("💳 Transaction Input")

    demo_mode = st.selectbox(
        "Choose input mode",
        ["Manual Input", "Real Legitimate Transaction", "Real Fraud Transaction"]
    )

    if demo_mode == "Manual Input":
        time = st.number_input("Transaction Time (seconds)", min_value=0.0, value=0.0)
        amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)

        input_data = np.zeros((1, 30))
        input_data[0][0] = time
        input_data[0][29] = amount

        st.markdown(
            '<div class="info-box">NOTE:- In manual mode im using only Time and Amount only(bcz it is manual). '
            'The anonymised PCA features V1–V28 are set to 0, so this mode is mainly for interface demonstration.</div>',
            unsafe_allow_html=True
        )

    elif demo_mode == "Real Legitimate Transaction":
        sample = data[data["Class"] == 0].sample(1, random_state=7)
        input_data = sample.drop("Class", axis=1).values

        st.markdown(
            '<div class="info-box">NOTE:- Using a real legitimate transaction from the dataset(Here i took data from the dataset itself), including Time, Amount, and V1–V28 PCA features.</div>',
            unsafe_allow_html=True
        )
        st.write("Actual class in dataset: **Legitimate (0)**")
        st.write("Transaction amount:", float(sample["Amount"].iloc[0]))

    else:
        sample = data[data["Class"] == 1].sample(1, random_state=7)
        input_data = sample.drop("Class", axis=1).values

        st.markdown(
            '<div class="info-box">NOTE:- Using a real fraudulent transaction from the dataset(I mainly did this for demonstration pupose .bcz  manual inputs cannot reliably replicate real fraud patterns in the dataset.... ), including Time, Amount, and V1–V28 PCA features.</div>',
            unsafe_allow_html=True
        )
        st.write("Actual class in dataset: **Fraudulent (1)**")
        st.write("Transaction amount:", float(sample["Amount"].iloc[0]))

    predict_button = st.button("🔍 Predict Transaction", use_container_width=True)

with right:
    st.subheader("System Information")
    st.write("""
    **Selected Model:** Random Forest  
    **Model Type:** Classification  
    **Dataset:** Credit Card Fraud Detection  
    **Output:** Legitimate or Fraudulent Transaction
    """)

    st.write("""
    The system compares transaction behaviour against patterns learned from historical financial transaction data.
    """)

    st.info(
    "NOTE:- For a more realistic demonstration, I use real transaction modes since they include all anonymised PCA features (because PCA features are not practical to enter manually)."
)

if predict_button:
    fraud_probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("🔎 Prediction Result")

    st.write(f"Fraud Probability: **{fraud_probability:.2%}**")
    st.progress(float(fraud_probability))

    if fraud_probability >= 0.30:
        st.error("🚨 Fraudulent Transaction Detected")
        st.write("This transaction shows suspicious patterns and should be reviewed.")
    else:
        st.success("✅ Legitimate Transaction")
        st.write("No major suspicious pattern detected based on the model.")