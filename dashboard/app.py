import os
import sys

# Add project root for src imports (VERY IMPORTANT)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import streamlit as st
import requests
import numpy as np
import time
import pandas as pd

from src.load_data import load_dataset

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Real-Time Credit Card Fraud Detection Dashboard")
st.markdown("---")


# -------------------------------------------------------
# Helper: API caller
# -------------------------------------------------------
def call_api(features):
    payload = {"features": features}
    res = requests.post(API_URL, json=payload)

    if res.status_code == 200:
        return res.json()

    return {"error": res.text}


# Sidebar mode switch
mode = st.sidebar.radio("Select Mode", ["Manual Prediction", "Live Simulation"])


# =======================================================
# MODE 1 — MANUAL
# =======================================================
if mode == "Manual Prediction":
    st.header("Manual Fraud Prediction")
    st.write("Enter 30 model features below:")

    cols = st.columns(3)
    inputs = []

    for i in range(30):
        with cols[i % 3]:
            value = st.number_input(f"Feature {i+1}", value=0.0, format="%.6f")
            inputs.append(value)

    if st.button("Predict"):
        res = call_api(inputs)

        if "error" in res:
            st.error(res["error"])
        else:
            st.success(f"Fraud Probability: {res['fraud_probability']:.4f}")
            st.write(f"Risk Category: **{res['risk_category']}**")

            st.subheader("Feature Values Used")
            df = pd.DataFrame({
                "Feature": [f"F{i+1}" for i in range(30)],
                "Value": res["features_used"]
            })
            st.dataframe(df)


# =======================================================
# MODE 2 — LIVE SIMULATION
# =======================================================
else:
    st.header("Live Transaction Simulation")

    interval = st.sidebar.slider("Transaction interval (seconds)", 0.1, 2.0, 0.5)
    run = st.sidebar.checkbox("Start Simulation")

    # Load dataset stats ONCE
    if "stats_loaded" not in st.session_state:
        df = load_dataset()
        X = df.drop(columns=["Class"])

        st.session_state.means = X.mean().values
        st.session_state.stds = X.std().values
        st.session_state.stats_loaded = True

    means = st.session_state.means
    stds = st.session_state.stds

    # Fraud injection slider
    fraud_rate = st.sidebar.slider("Fraud Injection Rate (%)", 0, 50, 5)

    # History initialization
    if "history" not in st.session_state:
        st.session_state.history = []

    graph_placeholder = st.empty()
    table_placeholder = st.empty()
    detail_placeholder = st.empty()

    # -------------------------------
    # Transaction generators
    # -------------------------------
    def generate_realistic_transaction():
        """Synthetic but realistic transaction based on PCA-feature distribution."""
        return np.random.normal(means, stds).round(6).tolist()

    def inject_fraud(features):
        """Moderate, realistic fraud perturbation on PCA features."""
        features = np.array(features)

        # Instead of x6 boost, use controlled distortion
        fraud_indices = [1, 3, 5, 7, 10, 12]
        for idx in fraud_indices:
            features[idx] += np.random.uniform(1.0, 3.0) * np.sign(features[idx])

        return features.round(6).tolist()

    # -------------------------------
    # LIVE LOOP
    # -------------------------------
    while run:
        features = generate_realistic_transaction()

        # Fraud injection
        if np.random.rand() < fraud_rate / 100:
            features = inject_fraud(features)

        # Call API
        res = call_api(features)

        if "error" not in res:
            st.session_state.history.append({
                "prob": res["fraud_probability"],
                "category": res["risk_category"],
                "features": features
            })

        # Prepare table
        rows = []
        for h in st.session_state.history[-40:]:
            row = {
                "Probability": h["prob"],
                "Category": h["category"]
            }
            # Add first 6 feature values for readability
            for i in range(6):
                row[f"F{i+1}"] = h["features"][i]
            rows.append(row)

        df = pd.DataFrame(rows)
        table_placeholder.dataframe(df, height=300)

        # Line graph
        if not df.empty:
            graph_placeholder.line_chart(df["Probability"])

        # Last transaction details
        if st.session_state.history:
            last = st.session_state.history[-1]
            detail_placeholder.markdown(f"""
            ### Latest Transaction
            **Fraud Probability:** {last['prob']:.4f}  
            **Category:** `{last['category']}`  
            """)

            with st.expander("Full Feature Vector"):
                st.json({f"Feature {i+1}": v for i, v in enumerate(last["features"])})

        time.sleep(interval)

    st.info("Simulation stopped.")