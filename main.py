import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import os
import json
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler
from fpdf import FPDF
from io import BytesIO
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from scipy.signal import resample
from ecg_loader import load_ecg_data
from utils import save_simulation_video
from reservoir import Reservoir
from bayesian_optimizer import BayesianRTDOptimizer  # New import

st.set_page_config(page_title="RTD-Reservoir ECG Simulator", layout="wide")
st.title("RTD-based Reservoir Computing Simulator")

# --- Config loading ---
config_path = "best_config.json"
optimized_params_path = "optimized_params.json"  # New config for optimized parameters

def load_config():
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {"reservoir_size": 200, "delay": 2, "use_mlp": False, "index": 10000, "lead": "I"}

def load_optimized_params():
    if os.path.exists(optimized_params_path):
        with open(optimized_params_path, "r") as f:
            return json.load(f)
    return None

def save_config(cfg):
    with open(config_path, "w") as f:
        json.dump(cfg, f)
        
def save_optimized_params(params):  # New function
    with open(optimized_params_path, "w") as f:
        json.dump(params, f)

cfg = load_config()
optimized_params = load_optimized_params()  # Load optimized parameters

if st.button("Use Best Settings"):
    for key in cfg:
        st.session_state[key] = cfg[key]
    st.success("Best settings loaded!")

# --- Mode Selection ---
mode = st.radio("Select Simulation Mode", ["ECG (real)","Use MIT-BIH Online", "Lorenz System (chaotic)"])

ecq_file = None

if mode == "ECG (real)":
    upload_option = st.radio("Choose Input Type", ["Upload CSV File", "Use PTB-XL Sample"])
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

elif mode == "Use MIT-BIH Online":
    MAX_SAMPLES = 3000
    st.subheader("MIT-BIH Loader (Local Folder or Online)")
    mitbih_root = st.text_input("Enter path to MIT-BIH folder:", "G:/My Drive/mit-bih-arrhythmia-database-1.0.0")
    try:
        record_list = [f.split('.')[0] for f in os.listdir(mitbih_root) if f.endswith('.dat')]
        selected_record = st.selectbox("Select MIT-BIH Record", sorted(set(record_list)))
        record_path = os.path.join(mitbih_root, selected_record)

        record = wfdb.rdrecord(record_path)
        ecg_raw = record.p_signal[:, 0]  # Default to channel 0

        ecg_resampled = resample(ecg_raw, int(len(ecg_raw) * 500 / record.fs))
        df = pd.DataFrame({'ecg': ecg_resampled})
        df = df.iloc[:MAX_SAMPLES]  
        st.line_chart(df['ecg'].values[:500])
        ecq_file = df
        st.success(f"Loaded record: {selected_record} from local folder.")
    except Exception as e:
        st.error(f"Error loading local record: {e}")

elif mode == "Lorenz System (chaotic)":
    st.subheader("Generating Lorenz System")
    def lorenz(t, state, sigma=10, beta=8/3, rho=28):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    t_span = (0, 40)
    t_eval = np.linspace(t_span[0], t_span[1], 4000)
    initial_state = [1.0, 1.0, 1.0]
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
    data = sol.y.T

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    df = pd.DataFrame({'ecg': data_scaled[:, 0]})
    st.success("Lorenz data generated.")

if ecq_file is not None:
    df = load_ecg_data(ecq_file) if not isinstance(ecq_file, pd.DataFrame) else ecq_file
    st.success("ECG data loaded and ready for simulation.")
    
if 'df' in locals():
    st.sidebar.header("Simulation Settings")
    
    # Add Bayesian optimization section
    st.sidebar.subheader("Bayesian Optimization")
    run_optimization = st.sidebar.checkbox("Run Bayesian Optimization", value=False)
    optimization_iters = st.sidebar.slider("Optimization Iterations", 10, 100, 30, 5)
    
    # Show optimized parameters if available
    if optimized_params:
        st.sidebar.markdown("**Optimized Parameters:**")
        for param, value in optimized_params.items():
            st.sidebar.text(f"{param}: {value:.4f}")
    
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
        
        # Bayesian optimization block
        # Bayesian optimization block
        if run_optimization:
            st.subheader("⚙️ Bayesian Optimization Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Use a subset for faster optimization
            sample_size = min(1000, len(ecg_series))
            X_sample = ecg_series[:sample_size].reshape(-1, 1)
            y_sample = ecg_series[:sample_size]
            
            # Initialize and run optimizer
            optimizer = BayesianRTDOptimizer(X_sample, y_sample, Reservoir, target='prediction')
            
            # Calculate total iterations
            init_points_val = 10
            n_iter_val = optimization_iters
            total_iter = init_points_val + n_iter_val
            
            # Create a wrapper function to track progress
            iteration_counter = [0]  # Use list to make it mutable in closure
            
            def optimization_progress():
                """Callback to track optimization progress"""
                iteration_counter[0] += 1
                progress = min(1.0, iteration_counter[0] / total_iter)
                progress_bar.progress(progress)
                status_text.text(f"Iteration {iteration_counter[0]}/{total_iter} - Exploring parameter space...")
                return True
            
            # Monkey-patch the optimizer to track progress
            original_probe = optimizer.optimizer.probe
            
            def tracked_probe(params, *args, **kwargs):
                result = original_probe(params, *args, **kwargs)
                optimization_progress()
                return result
            
            optimizer.optimizer.probe = tracked_probe
            
            # Run the optimization
            best_params = optimizer.optimize(init_points=init_points_val, n_iter=n_iter_val)
            
            # Save optimized parameters
            save_optimized_params(best_params)
            optimized_params = best_params
            
            # Update parameters from optimization
            reservoir_size = int(best_params['n_virtual'])
            delay = 2  # Reset to default delay
            
            # Show optimization results
            status_text.text("✅ Optimization complete! Using optimized parameters.")
            st.dataframe(pd.DataFrame([best_params]).T.rename(columns={0: 'Value'}))
            
            # Visualize optimization
            with st.expander("Optimization Visualization"):
                optimizer.visualize_optimization()
                st.pyplot(plt.gcf())
        
        num_units = 3
        delays = [delay + i for i in range(num_units)]
        X_blocks = []
        
        # Create reservoir with optimized parameters if available
        if optimized_params:
            reservoirs = []
            for d in delays:
                r = Reservoir(
                    size=int(optimized_params['n_virtual']),
                    delay=d,
                    use_mlp=False,
                    input_scaling=optimized_params['input_scaling'],
                    feedback_scaling=optimized_params['feedback_scaling'],
                    v_bias=optimized_params['v_bias'],
                    leaky=optimized_params['leakage']
                )
                X_r, Y_r = r.generate_XY(ecg_series)
                X_blocks.append(X_r)
                reservoirs.append(r)
        else:
            reservoirs = []
            for d in delays:
                r = Reservoir(size=reservoir_size, delay=d, use_mlp=False)
                X_r, Y_r = r.generate_XY(ecg_series)
                X_blocks.append(X_r)
                reservoirs.append(r)

        min_len = min(len(x) for x in X_blocks + [Y_r])
        X = np.hstack([x[:min_len] for x in X_blocks])
        Y = Y_r[:min_len].reshape(-1, 1)

        readout = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=0) if use_mlp else Ridge(alpha=1.0)
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
            
            # Add optimized parameters if available
            if optimized_params:
                pdf.ln(10)
                pdf.cell(0, 10, "Optimized Parameters:", ln=True)
                for param, value in optimized_params.items():
                    pdf.cell(0, 10, f"{param}: {value:.4f}", ln=True)
            
            pdf_output = BytesIO()
            pdf.output(pdf_output)
            st.download_button("📥 Download PDF", data=pdf_output.getvalue(), file_name="simulation_report.pdf", mime="application/pdf")

        if st.button("🎥 Export Video"):
            video_path = save_simulation_video(Y.ravel(), Y_pred)
            with open(video_path, "rb") as file:
                st.download_button("Download Video", file, "simulation.mp4")