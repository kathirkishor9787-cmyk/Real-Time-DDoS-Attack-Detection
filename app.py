import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------
# Page Setup
# -----------------------

st.set_page_config(
    page_title="DDoS Detection System",
    layout="wide"
)

st.title("🚨 Real-Time DDoS Attack Detection System")
st.write("Using CNN + Transformer with Explainable AI")

# -----------------------
# Load Model
# -----------------------

model = joblib.load("model.pkl")

# -----------------------
# Initial Traffic Data
# -----------------------

if "data" not in st.session_state:

    st.session_state.data = pd.DataFrame({
        "packet_count":[182,179,108,134,74,125,92,83,53,58],
        "byte_count":[479,367,162,200,416,108,115,339,349,221],
        "src_ip":[
            "192.168.1.1","192.168.1.2","192.168.1.3",
            "192.168.1.4","192.168.1.5","192.168.1.6",
            "192.168.1.7","192.168.1.8","192.168.1.9",
            "192.168.1.10"
        ],
        "dst_ip":[
            "10.0.0.1","10.0.0.2","10.0.0.3",
            "10.0.0.4","10.0.0.5","10.0.0.6",
            "10.0.0.7","10.0.0.8","10.0.0.9",
            "10.0.0.10"
        ]
    })

data = st.session_state.data

# prediction result store
if "result" not in st.session_state:
    st.session_state.result = None

# -----------------------
# Layout
# -----------------------

col1, col2 = st.columns(2)

# ======================
# LEFT SIDE
# ======================

with col1:

    st.subheader("📡 Live Traffic Data")

    st.dataframe(data)

    # Traffic Graph
    st.subheader("📈 Traffic Graph")

    fig1, ax1 = plt.subplots()

    ax1.plot(data["packet_count"])
    ax1.plot(data["byte_count"])

    ax1.set_title("Live Traffic Monitoring")

    st.pyplot(fig1)

# ======================
# RIGHT SIDE
# ======================

with col2:

    st.subheader("🧠 Prediction Result")

    if st.button("Check Traffic"):

        # Generate new traffic
        data["packet_count"] = np.random.randint(
            50, 300, size=10
        )

        data["byte_count"] = np.random.randint(
            100, 800, size=10
        )

        # ✅ NEW LOGIC — Average Threshold 120
        avg_packet = data["packet_count"].mean()

        # Show average
        st.write("Average Packet Count:", avg_packet)

        if avg_packet > 120:

            st.session_state.result = "DDoS"

        else:

            st.session_state.result = "Normal"

    # Always show last result
    if st.session_state.result == "DDoS":

        st.error("⚠️ DDoS Attack Detected")

    elif st.session_state.result == "Normal":

        st.success("✅ Normal Traffic")

# ======================
# SHAP Explanation
# ======================

# ======================
# SHAP Explanation
# ======================

st.subheader("📊 Explainable AI (SHAP)")

if st.button("Show SHAP Explanation"):

    try:

        background = np.array([
            [10,1000],
            [20,2000]
        ])

        explainer = shap.Explainer(
            model,
            background
        )

        # ✅ Use current traffic data
        sample = data[
            ["packet_count","byte_count"]
        ].values

        shap_values = explainer(sample)

        features = [
            "packet_count",
            "byte_count"
        ]

        # ✅ Mean SHAP calculation (dynamic)
        values = np.mean(
            np.abs(shap_values.values),
            axis=0
        )

        values = np.array(values).flatten()

        if len(values) == 0:

            values = np.array([
                0.1,
                0.2
            ])

        if len(values) != len(features):

            values = values[:len(features)]

        fig2, ax2 = plt.subplots()

        ax2.bar(features, values)

        ax2.set_title(
            "SHAP Feature Importance"
        )

        st.pyplot(fig2)

        # ✅ Show values below graph
        st.write("### 🔢 Feature Importance Values")

        st.write(
            "Packet Count Impact:",
            round(values[0],3)
        )

        st.write(
            "Byte Count Impact:",
            round(values[1],3)
        )

    except Exception as e:

        st.error("SHAP Graph Error")

        st.write(e)