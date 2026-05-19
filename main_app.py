'''
## PATENT NOTICE
This repository contains novel inventions related to multi-RTD reservoir computing for ECG prediction.
Provisional patent application pending. All rights reserved.
Unauthorized commercial use or replication of the multi-RTD architecture (>2 units) is prohibited.
'''

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import os
import json
from io import BytesIO
from scipy.integrate import solve_ivp
from scipy.signal import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from fpdf2 import FPDF

from ecg_loader import load_ecg_data, list_mitbih_records
from reservoir import Reservoir
from bayesian_optimizer import BayesianRTDOptimizer
from utils import save_simulation_video, save_model, load_model

st.set_page_config(page_title="RTD-Reservoir ECG Simulator", layout="wide")
st.title("RTD-Based Reservoir Computing — ECG Signal Simulator")

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
CONFIG_PATH = "best_config.json"
OPT_PARAMS_PATH = "optimized_params.json"

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"reservoir_size": 200, "delay": 2, "use_mlp": False,
            "index": 10000, "lead": "I"}

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f)

def load_optimized_params():
    if os.path.exists(OPT_PARAMS_PATH):
        with open(OPT_PARAMS_PATH) as f:
            return json.load(f)
    return None

def save_optimized_params(params):
    with open(OPT_PARAMS_PATH, "w") as f:
        json.dump(params, f)

cfg = load_config()
optimized_params = load_optimized_params()

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_ptbxl_local(ptbxl_root):
    """Load a PTB-XL record from a local directory."""
    meta = pd.read_csv(f"{ptbxl_root}/ptbxl_database.csv")
    index = st.slider("Choose sample index", 0, len(meta) - 1,
                      cfg.get("index", 0), key="index")
    record_path = f"{ptbxl_root}/{meta.loc[index, 'filename_lr']}"
    record = wfdb.rdrecord(record_path)
    lead_names = record.sig_name
    selected_lead = st.selectbox(
        "Select ECG lead", lead_names,
        index=lead_names.index(cfg.get("lead", lead_names[0])), key="lead")
    signal = record.p_signal[:, lead_names.index(selected_lead)]
    return pd.DataFrame({"ecg": signal})


def _load_mitbih(mitbih_root):
    """Load a MIT-BIH record from a local directory (max 3000 samples)."""
    MAX_SAMPLES = 3000
    record_list = sorted({f.split(".")[0]
                          for f in os.listdir(mitbih_root) if f.endswith(".dat")})
    selected = st.selectbox("Select MIT-BIH record", record_list)
    record = wfdb.rdrecord(os.path.join(mitbih_root, selected))
    ecg_raw = record.p_signal[:, 0]
    ecg_resampled = resample(ecg_raw, int(len(ecg_raw) * 500 / record.fs))
    df = pd.DataFrame({"ecg": ecg_resampled}).iloc[:MAX_SAMPLES]
    st.success(f"Loaded MIT-BIH record: {selected}")
    return df


def _lorenz_signal():
    """Generate a normalised Lorenz x-component as a synthetic test signal."""
    def lorenz(t, s, sigma=10, beta=8/3, rho=28):
        x, y, z = s
        return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

    sol = solve_ivp(lorenz, (0, 40), [1., 1., 1.],
                    t_eval=np.linspace(0, 40, 4000))
    data = MinMaxScaler().fit_transform(sol.y.T)
    return pd.DataFrame({"ecg": data[:, 0]})

# ---------------------------------------------------------------------------
# Best settings button
# ---------------------------------------------------------------------------
if st.button("Use Best Settings"):
    for k, v in cfg.items():
        st.session_state[k] = v
    st.success("Best settings loaded!")

# ---------------------------------------------------------------------------
# Mode / data selection
# ---------------------------------------------------------------------------
mode = st.radio("Select Simulation Mode",
                ["ECG — Upload CSV", "ECG — PTB-XL (local)", "ECG — MIT-BIH (local)",
                 "Lorenz System (chaotic)"])

df = None

if mode == "ECG — Upload CSV":
    uploaded = st.file_uploader("Upload ECG CSV file", type="csv")
    if uploaded:
        df = load_ecg_data(uploaded)
        st.success("ECG data loaded.")

elif mode == "ECG — PTB-XL (local)":
    st.subheader("PTB-XL Local Loader")
    ptbxl_root = st.text_input(
        "Path to PTB-XL dataset folder:",
        "./ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
    if os.path.exists(ptbxl_root):
        try:
            df = _load_ptbxl_local(ptbxl_root)
            st.line_chart(df["ecg"].values[:500])
        except Exception as e:
            st.error(f"Error loading PTB-XL record: {e}")
    else:
        st.warning("PTB-XL folder not found. Check the path above.")

elif mode == "ECG — MIT-BIH (local)":
    st.subheader("MIT-BIH Local Loader")
    mitbih_root = st.text_input(
        "Path to MIT-BIH dataset folder:",
        "./mit-bih-arrhythmia-database-1.0.0")
    if os.path.exists(mitbih_root):
        try:
            df = _load_mitbih(mitbih_root)
            st.line_chart(df["ecg"].values[:500])
        except Exception as e:
            st.error(f"Error loading MIT-BIH record: {e}")
    else:
        st.warning("MIT-BIH folder not found. Check the path above.")

elif mode == "Lorenz System (chaotic)":
    df = _lorenz_signal()
    st.success("Lorenz attractor signal generated.")

# ---------------------------------------------------------------------------
# Simulation controls (only shown when data is ready)
# ---------------------------------------------------------------------------
if df is not None:
    st.sidebar.header("Simulation Settings")

    # Bayesian optimisation toggle
    st.sidebar.subheader("Bayesian Optimisation")
    run_optimization = st.sidebar.checkbox("Run Bayesian Optimisation", value=False)
    optimization_iters = st.sidebar.slider("Optimisation iterations", 10, 100, 30, 5)

    if optimized_params:
        st.sidebar.markdown("**Last optimised parameters:**")
        for k, v in optimized_params.items():
            st.sidebar.text(f"{k}: {v:.4f}")

    reservoir_size = st.sidebar.slider("Reservoir size", 50, 1000,
                                       cfg.get("reservoir_size", 200))
    delay = st.sidebar.slider("Delay steps", 1, 50, cfg.get("delay", 2))
    use_mlp = (st.sidebar.radio("Readout type",
                                ["Ridge Regression", "MLP (Neural Network)"])
               == "MLP (Neural Network)")
    warmup_percent = st.sidebar.slider("Warm-up %", 0, 50, 10, step=5)

    if st.button("Save current settings as best"):
        save_config({
            "reservoir_size": reservoir_size,
            "delay": delay,
            "use_mlp": use_mlp,
            "index": st.session_state.get("index", 0),
            "lead": st.session_state.get("lead", "I"),
        })
        st.success("Settings saved.")

    # Model persistence
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Persistence")
    model_path = st.sidebar.text_input("Model file path", "saved_model.pkl")
    if st.sidebar.button("Load saved model"):
        _readout, _scaler = load_model(model_path)
        if _readout is not None:
            st.session_state["loaded_readout"] = _readout
            st.session_state["loaded_scaler"]  = _scaler
            st.sidebar.success("Model loaded.")
        else:
            st.sidebar.warning(f"No model found at '{model_path}'.")

    # -----------------------------------------------------------------------
    # Run simulation
    # -----------------------------------------------------------------------
    if st.button("Run Simulation"):
        ecg_series = df["ecg"].values
        ecg_series = (ecg_series - np.mean(ecg_series)) / (np.std(ecg_series) + 1e-8)

        # --- Optional Bayesian optimisation ---
        if run_optimization:
            st.subheader("Bayesian Optimisation Progress")
            sample_size = min(1000, len(ecg_series))
            X_sample = ecg_series[:sample_size].reshape(-1, 1)
            y_sample = ecg_series[:sample_size]

            optimizer = BayesianRTDOptimizer(X_sample, y_sample, Reservoir,
                                             target="prediction")
            progress_bar = st.progress(0)
            counter = [0]
            total_iter = 10 + optimization_iters

            original_probe = optimizer.optimizer.probe
            def tracked_probe(params, *args, **kwargs):
                result = original_probe(params, *args, **kwargs)
                counter[0] += 1
                progress_bar.progress(min(1.0, counter[0] / total_iter))
                return result
            optimizer.optimizer.probe = tracked_probe

            best_params = optimizer.optimize(init_points=10, n_iter=optimization_iters)
            save_optimized_params(best_params)
            optimized_params = best_params
            reservoir_size = int(best_params["n_virtual"])
            st.dataframe(pd.DataFrame([best_params]).T.rename(columns={0: "Value"}))

        # --- Multi-RTD ensemble (3 parallel reservoirs with staggered delays) ---
        num_units = 3
        delays = [delay + i for i in range(num_units)]
        X_blocks, Y_r = [], None

        for d in delays:
            r_kwargs = dict(size=reservoir_size, delay=d, use_mlp=False)
            if optimized_params:
                r_kwargs.update(
                    input_scaling=optimized_params["input_scaling"],
                    feedback_scaling=optimized_params["feedback_scaling"],
                    v_bias=optimized_params["v_bias"],
                    leaky=optimized_params["leakage"],
                )
            r = Reservoir(**r_kwargs)
            X_r, Y_r = r.generate_XY(ecg_series)
            X_blocks.append(X_r)

        min_len = min(len(x) for x in X_blocks + [Y_r])
        X = np.hstack([x[:min_len] for x in X_blocks])
        Y = Y_r[:min_len]

        # --- Train/predict split (80 % train, 20 % test) ---
        split = int(0.8 * len(Y))
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]

        # Use a pre-loaded model if available, otherwise train from scratch
        if "loaded_readout" in st.session_state:
            readout = st.session_state["loaded_readout"]
            X_train_sc = st.session_state["loaded_scaler"].transform(X_train)
            X_test_sc  = st.session_state["loaded_scaler"].transform(X_test)
            st.info("Using loaded model — skipping training.")
        else:
            from sklearn.preprocessing import MinMaxScaler as _MMS
            _sc = _MMS()
            X_train_sc = _sc.fit_transform(X_train)
            X_test_sc  = _sc.transform(X_test)
            readout = (MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=0)
                       if use_mlp else Ridge(alpha=1.0))
            readout.fit(X_train_sc, Y_train)
            if st.button("Save trained model"):
                save_model(readout, _sc, model_path)
                st.success(f"Model saved to '{model_path}'.")

        Y_pred_test = readout.predict(X_test_sc)

        # --- Metrics on held-out test set ---
        mae  = float(np.mean(np.abs(Y_test - Y_pred_test)))
        rmse = float(np.sqrt(np.mean((Y_test - Y_pred_test) ** 2)))
        r2   = float(r2_score(Y_test, Y_pred_test))
        signal_range = float(np.max(Y_test) - np.min(Y_test)) or 1.0
        norm_err = (mae / signal_range) * 100  # Normalised Error Score (not accuracy)

        st.subheader("Evaluation Metrics (held-out test set — last 20 %)")
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| **R²** | {r2:.4f} |
| **MAE** | {mae:.6f} |
| **RMSE** | {rmse:.6f} |
| **Normalised Error** | {norm_err:.2f} % |
""")

        result_df = pd.DataFrame({"True ECG": Y_test, "Predicted ECG": Y_pred_test})
        result_df["Error"] = np.abs(result_df["True ECG"] - result_df["Predicted ECG"])

        threshold = st.slider("Anomaly threshold (absolute error):", 0.0, 1.0, 0.2)
        highlighted = result_df[result_df["Error"] > threshold]
        st.markdown(f"**Anomalies detected:** {len(highlighted)}")
        st.dataframe(result_df.head(100))

        # --- Plots ---
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(Y_test, label="True ECG", alpha=0.7)
        ax.plot(Y_pred_test, label="Predicted ECG", alpha=0.7)
        ax.legend()
        ax.set_title("ECG Prediction — Test Set")
        st.pyplot(fig)

        fig_err, ax_err = plt.subplots(figsize=(10, 2))
        ax_err.plot(result_df["Error"], label="Absolute Error")
        ax_err.axhline(threshold, color="r", linestyle="--", label="Threshold")
        ax_err.legend()
        st.pyplot(fig_err)

        fig_anom, ax_anom = plt.subplots(figsize=(12, 3))
        ax_anom.plot(result_df["True ECG"].values, label="True ECG", alpha=0.6)
        ax_anom.plot(result_df["Predicted ECG"].values, label="Predicted ECG", alpha=0.6)
        ax_anom.scatter(highlighted.index, highlighted["True ECG"],
                        color="red", label="Anomalies", s=10)
        ax_anom.legend()
        st.pyplot(fig_anom)

        # --- CSV download ---
        st.download_button("Download CSV",
                           result_df.to_csv(index=False).encode(),
                           "ecg_prediction_results.csv")

        # --- PDF report ---
        if st.button("Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            pdf.cell(0, 10, "RTD-Reservoir ECG Simulation Report", ln=True, align="C")
            pdf.ln(8)
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(0, 8,
                f"Evaluation on held-out test set (last 20% of signal)\n"
                f"R²:               {r2:.4f}\n"
                f"MAE:              {mae:.6f}\n"
                f"RMSE:             {rmse:.6f}\n"
                f"Normalised Error: {norm_err:.2f}%\n"
                f"Anomalies:        {len(highlighted)}\n")
            if optimized_params:
                pdf.ln(4)
                pdf.cell(0, 8, "Optimised Parameters:", ln=True)
                for k, v in optimized_params.items():
                    pdf.cell(0, 8, f"  {k}: {v:.4f}", ln=True)
            pdf_bytes = pdf.output()
            st.download_button("Download PDF", data=bytes(pdf_bytes),
                               file_name="simulation_report.pdf",
                               mime="application/pdf")

        # --- Video export ---
        if st.button("Export Simulation Video"):
            try:
                video_path = save_simulation_video(Y_test, Y_pred_test)
                with open(video_path, "rb") as f:
                    st.download_button("Download Video", f, "simulation.mp4")
            except Exception as e:
                st.error(f"Video export failed: {e}. Ensure ffmpeg is installed.")
