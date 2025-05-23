# main_app.py (Final working version with multi-RTD support, aligned X-Y, and visual stats)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import os
import json
from fpdf import FPDF
from io import BytesIO
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from ecg_loader import load_ecg_data
from utils import save_simulation_video
from reservoir import Reservoir

st.set_page_config(page_title="RTD-Reservoir ECG Simulator", layout="wide")
st.title("RTD-based Reservoir Computing for ECG Signal Simulation")

# --- Config loading ---
config_path = "best_config.json"
def load_config():
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {"reservoir_size": 200, "delay": 2, "use_mlp": False, "index": 10000, "lead": "I"}

def save_config(cfg):
    with open(config_path, "w") as f:
        json.dump(cfg, f)

cfg = load_config()

if st.button("Use Best Settings"):
    for key in cfg:
        st.session_state[key] = cfg[key]
    st.success("Best settings loaded!")

# --- Data upload ---
upload_option = st.radio("Choose Input Type", ["Upload CSV File", "Use PTB-XL Sample"])
ecq_file = None

if upload_option == "Upload CSV File":
    ecq_file = st.file_uploader("Upload ECG CSV File", type="csv")
else:
    st.subheader("PTB-XL Sample Loader")
    ptbxl_root = st.text_input("Enter path to PTB-XL dataset folder:", "./ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
    if os.path.exists(ptbxl_root):
        try:
            meta = pd.read_csv(f'{ptbxl_root}/ptbxl_database.csv')
            index = st.slider("Choose sample index", 0, len(meta)-1, cfg.get("index", 0), key="index")
            record_path = f"{ptbxl_root}/{meta.loc[index, 'filename_lr']}"
            record = wfdb.rdrecord(record_path)
            lead_names = record.sig_name
            selected_lead = st.selectbox("Select ECG Lead", lead_names, index=lead_names.index(cfg.get("lead", lead_names[0])), key="lead")
            lead_index = lead_names.index(selected_lead)
            signal = record.p_signal[:, lead_index]
            df = pd.DataFrame({'ecg': signal})
            st.line_chart(df['ecg'].values[:500])
            ecq_file = df
        except Exception as e:
            st.error(f"Error loading PTB-XL file: {e}")
    else:
        st.warning("PTB-XL folder path is invalid or does not exist.")

if ecq_file is not None:
    df = load_ecg_data(ecq_file) if not isinstance(ecq_file, pd.DataFrame) else ecq_file
    st.success("ECG data loaded.")

    # --- Sidebar controls ---
    st.sidebar.header("Simulation Settings")
    reservoir_size = st.sidebar.slider("Reservoir Size", 50, 1000, cfg.get("reservoir_size", 200))
    delay = st.sidebar.slider("Delay Steps", 1, 50, cfg.get("delay", 2))
    use_mlp = st.sidebar.radio("Readout Type", ["MLP (Neural Network)", "Ridge Regression"]) == "MLP (Neural Network)"
    warmup_percent = st.sidebar.slider("Warm-up % (ignored from start)", 0, 50, 20, step=5)

    if st.button("Save Current Settings as Best"):
        save_config({
            "reservoir_size": reservoir_size,
            "delay": delay,
            "use_mlp": use_mlp,
            "index": st.session_state.get("index", 0),
            "lead": st.session_state.get("lead", "I")
        })
        st.success("Settings saved as best config.")

    if st.button("Run Simulation"):
        ecg_series = df['ecg'].values
        ecg_series = (ecg_series - np.mean(ecg_series)) / (np.std(ecg_series) + 1e-8)

        num_units = 3
        delays = [delay + i for i in range(num_units)]
        X_blocks = []
        for d in delays:
            r = Reservoir(size=reservoir_size, delay=d, use_mlp=False)
            X_r, Y_r = r.generate_XY(ecg_series)
            X_blocks.append(X_r)

        min_len = min(len(x) for x in X_blocks + [Y_r])
        X = np.hstack([x[:min_len] for x in X_blocks])
        Y = Y_r[:min_len].reshape(-1, 1)

        if use_mlp:
            readout = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=0)
        else:
            readout = Ridge(alpha=1.0)

        readout.fit(X, Y.ravel())
        Y_pred = readout.predict(X)

        mae = np.mean(np.abs(Y.ravel() - Y_pred))
        rmse = np.sqrt(np.mean((Y.ravel() - Y_pred)**2))
        r2 = r2_score(Y.ravel(), Y_pred)

        signal_range = np.max(Y.ravel()) - np.min(Y.ravel()) or 1.0
        acc = (1 - (mae / signal_range)) * 100

        st.subheader("Evaluation Metrics")
        st.markdown(f"""
        * **Accuracy:** {acc:.2f}%
        * **R²:** {r2:.4f}
        * **MAE:** {mae:.6f}
        * **RMSE:** {rmse:.6f}
        """)

        result_df = pd.DataFrame({
            'True ECG': Y.ravel(),
            'Predicted ECG': Y_pred
        })
        result_df['Error'] = np.abs(result_df['True ECG'] - result_df['Predicted ECG'])

        threshold = st.slider("Anomaly threshold (error):", 0.0, 1.0, 0.2)
        highlighted = result_df[result_df['Error'] > threshold]
        st.markdown(f"**Anomalies Detected:** {len(highlighted)}")
        st.dataframe(result_df.head(100))

        fig, ax = plt.subplots()
        ax.plot(Y.ravel(), label='True ECG', alpha=0.7)
        ax.plot(Y_pred, label='Predicted ECG', alpha=0.7)
        ax.legend()
        st.pyplot(fig)

        fig_err, ax_err = plt.subplots(figsize=(10, 3))
        ax_err.plot(result_df['Error'], label='Absolute Error')
        ax_err.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        ax_err.legend()
        st.pyplot(fig_err)

        fig_anom, ax_anom = plt.subplots(figsize=(12, 3))
        ax_anom.plot(result_df['True ECG'].values, label='True ECG', alpha=0.6)
        ax_anom.plot(result_df['Predicted ECG'].values, label='Predicted ECG', alpha=0.6)
        ax_anom.scatter(highlighted.index, highlighted['True ECG'], color='red', label='Anomalies', s=10)
        ax_anom.legend()
        st.pyplot(fig_anom)

        st.download_button("📥 Download CSV", result_df.to_csv(index=False).encode(), "ecg_prediction_results.csv")

        if st.button("📄 Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="RTD-Reservoir ECG Report", ln=True, align='C')
            pdf.ln(10)
            pdf.multi_cell(0, 10, f"MSE: {rmse**2:.6f}\nMAE: {mae:.6f}\nAccuracy: {acc:.2f}%\nAnomalies: {len(highlighted)}")
            pdf_output = BytesIO()
            pdf.output(pdf_output)
            st.download_button("📥 Download PDF", data=pdf_output.getvalue(), file_name="simulation_report.pdf", mime="application/pdf")

        if st.button("🎥 Export Video"):
            video_path = save_simulation_video(Y.ravel(), Y_pred)
            with open(video_path, "rb") as file:
                st.download_button("Download Video", file, "simulation.mp4")
