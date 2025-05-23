# main_app.py - Fully Restored with Warm-up, Anomalies, Exports

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import os
import json
from fpdf import FPDF
from io import BytesIO
from itertools import product

from rtd_model import rtd_nonlinearity
from reservoir import Reservoir
from ecg_loader import load_ecg_data
from utils import plot_ecg_prediction, save_simulation_video

st.set_page_config(page_title="RTD-Reservoir ECG Simulator", layout="wide")
st.title("🫀 RTD-based Reservoir Computing for ECG Signal Simulation")

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

if st.button("📌 Use Best Settings"):
    st.session_state["reservoir_size"] = cfg["reservoir_size"]
    st.session_state["delay"] = cfg["delay"]
    st.session_state["use_mlp"] = cfg["use_mlp"]
    st.session_state["index"] = cfg["index"]
    st.session_state["lead"] = cfg["lead"]
    st.success("Best settings loaded!")

upload_option = st.radio("Choose Input Type", ["Upload CSV File", "Use PTB-XL Sample"])
ecq_file = None

if upload_option == "Upload CSV File":
    ecq_file = st.file_uploader("📁 Upload ECG CSV File", type="csv")
else:
    st.markdown("---")
    st.subheader("🔎 PTB-XL Sample Loader")
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
    st.success("✅ ECG data loaded.")
    df = load_ecg_data(ecq_file) if not isinstance(ecq_file, pd.DataFrame) else ecq_file

    st.sidebar.header("⚙️ Simulation Settings")
    reservoir_size = st.sidebar.slider("Reservoir Size", 50, 1000, cfg.get("reservoir_size", 200), key="reservoir_size")
    delay = st.sidebar.slider("Delay Steps", 1, 50, cfg.get("delay", 2), key="delay")
    use_mlp = st.sidebar.radio("Readout Type", ["MLP (Neural Network)", "Ridge Regression"], index=0 if cfg.get("use_mlp") else 1, key="use_mlp") == "MLP (Neural Network)"
    warmup_percent = st.sidebar.slider("Warm-up % (ignored from start)", 0, 50, 10, step=5)

    if st.button("💾 Save Current Settings as Best"):
        save_config({
            "reservoir_size": reservoir_size,
            "delay": delay,
            "use_mlp": use_mlp,
            "index": st.session_state.get("index", 0),
            "lead": st.session_state.get("lead", "I")
        })
        st.success("Settings saved as best config.")

    if st.button("🚀 Run Simulation"):
        ecg_series = df['ecg'].values
        model = Reservoir(size=reservoir_size, delay=delay, use_mlp=use_mlp, show_progress=True)
        predicted = model.fit_predict(ecg_series, dropout_rate=0.1, noise_std=0.01, warmup_percent=warmup_percent)

        metrics = model.get_metrics()

        st.subheader("📊 Evaluation Metrics")
        st.markdown(f"**Total MSE:** {metrics['mse_total']:.6f} | **MAE:** {metrics['mae_total']:.6f}")
        st.markdown(f"**Stable MSE (after warm-up):** {metrics['mse_stable']:.6f} | **MAE:** {metrics['mae_stable']:.6f}")

        result_df = pd.DataFrame({
            'True ECG': model.Y_true,
            'Predicted ECG': model.Y_pred
        })
        result_df['Error'] = np.abs(result_df['True ECG'] - result_df['Predicted ECG'])

        threshold = st.slider("Anomaly threshold (error):", 0.0, 1.0, 0.2)
        highlighted = result_df[result_df['Error'] > threshold]
        st.markdown(f"**⚠️ Anomalies Detected:** {len(highlighted)}")
        st.dataframe(result_df.head(100))

        fig, ax = plt.subplots()
        ax.plot(model.Y_true, label="True ECG", alpha=0.6)
        ax.plot(model.Y_pred, label="Predicted ECG", alpha=0.6)
        ax.axvline(model.warmup_threshold, color='red', linestyle='--', label=f'Warm-up {warmup_percent}%')
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

        if use_mlp:
            st.subheader("📈 MLP Training Loss Curve")
            loss_curve = model.get_loss_curve()
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(loss_curve, label='Loss')
            ax_loss.set_title("MLP Loss Curve")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            st.pyplot(fig_loss)

        st.download_button("📥 Download CSV", result_df.to_csv(index=False).encode(), file_name="ecg_prediction_results.csv")

        if st.button("📄 Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="RTD-Reservoir ECG Report", ln=True, align='C')
            pdf.ln(10)
            pdf.multi_cell(0, 10, f"MSE: {metrics['mse_total']:.6f}\nMAE: {metrics['mae_total']:.6f}\nStable MSE: {metrics['mse_stable']:.6f}\nStable MAE: {metrics['mae_stable']:.6f}\nAnomalies: {len(highlighted)}")
            pdf_output = BytesIO()
            pdf.output(pdf_output)
            st.download_button("📥 Download PDF", data=pdf_output.getvalue(), file_name="simulation_report.pdf", mime="application/pdf")

        if st.button("🎥 Export Video"):
            video_path = save_simulation_video(ecg_series, predicted)
            with open(video_path, "rb") as file:
                st.download_button("Download Video", file, "simulation.mp4")
else:
    st.info("👆 Please upload a CSV file or choose a PTB-XL sample to begin.")
