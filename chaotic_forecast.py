'''
## PATENT NOTICE
This repository contains novel inventions related to multi-RTD reservoir computing for ECG prediction.
Provisional patent application pending. All rights reserved.
Unauthorized commercial use or replication of the multi-RTD architecture (>2 units) is prohibited.
'''

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from reservoir import Reservoir

st.set_page_config(page_title="RTD Chaotic / ECG Forecasting", layout="wide")
st.title("Chaotic & ECG Signal Forecasting with RTD-Reservoir Computing")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
mode = st.radio("Input source", ["Lorenz Chaotic System", "Synthetic ECG Signal"])
prediction_horizon = st.sidebar.slider("Prediction horizon (steps)", 1, 200, 50)
reservoir_size = st.sidebar.slider("Reservoir size", 50, 1000, 300)
delay = st.sidebar.slider("Delay steps", 1, 20, 2)
warmup_percent = st.sidebar.slider("Warm-up %", 0, 50, 10)
use_mlp = (st.sidebar.radio("Readout type",
                             ["Ridge Regression", "MLP (Neural Network)"])
           == "MLP (Neural Network)")

# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def _solve_lorenz():
    def lorenz(t, s, sigma=10, beta=8/3, rho=28):
        x, y, z = s
        return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]
    sol = solve_ivp(lorenz, [0, 40], [1., 1., 1.],
                    t_eval=np.linspace(0, 40, 4000))
    return sol.y.T  # shape (4000, 3)

if mode == "Lorenz Chaotic System":
    data = _solve_lorenz()

    st.subheader("Lorenz Attractor")
    fig3d = plt.figure(figsize=(6, 5))
    ax3d = fig3d.add_subplot(projection="3d")
    ax3d.plot(data[:, 0], data[:, 1], data[:, 2], lw=0.4)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    st.pyplot(fig3d)

    comp = st.selectbox("Component to forecast", ["x", "y", "z"])
    var_idx = {"x": 0, "y": 1, "z": 2}[comp]
    raw_signal = data[:, var_idx]

else:  # Synthetic ECG-like
    t = np.linspace(0, 10, 4000)
    raw_signal = 0.6 * np.sin(2 * np.pi * 5 * t) + 0.2 * np.sin(2 * np.pi * 50 * t)
    st.info("Synthetic ECG-like waveform (sum of two sinusoids) loaded.")

# ---------------------------------------------------------------------------
# Normalise with a per-variable scaler so inverse-transform is unambiguous
# ---------------------------------------------------------------------------
scaler = MinMaxScaler()
signal = scaler.fit_transform(raw_signal.reshape(-1, 1)).flatten()

split_idx = int(len(signal) * 0.8)
train_signal = signal[:split_idx]
test_signal  = signal[split_idx:]

# ---------------------------------------------------------------------------
# Reservoir training
# ---------------------------------------------------------------------------
r = Reservoir(size=reservoir_size, delay=delay, use_mlp=use_mlp)
X_train, Y_train = r.generate_XY(train_signal)
r.scaler.fit(X_train)

readout = (MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
           if use_mlp else Ridge(alpha=1.0))
readout.fit(X_train, Y_train)

# ---------------------------------------------------------------------------
# Autoregressive multi-step forecast
# Reservoir state carries over from training (correct: it reflects the last
# known system state at the train/test boundary).
# ---------------------------------------------------------------------------
current_input = train_signal[-max(delay, 1):].tolist()
predicted_scaled = []

for _ in range(prediction_horizon):
    r.states = ((1 - r.leaky) * r.states
                + r.leaky * np.roll(r.states, 1))
    r.states[0] = r._nonlinear(current_input[-delay])
    X_step = r.scaler.transform(r.states.reshape(1, -1))
    next_val = readout.predict(X_step)[0]
    predicted_scaled.append(next_val)
    current_input.append(next_val)
    current_input.pop(0)

# Inverse-transform predictions and ground truth using the same per-variable scaler
predicted_array = scaler.inverse_transform(
    np.array(predicted_scaled).reshape(-1, 1)).flatten()
true_future = scaler.inverse_transform(
    test_signal[:prediction_horizon].reshape(-1, 1)).flatten()

# ---------------------------------------------------------------------------
# Evaluation — on the held-out future window only
# ---------------------------------------------------------------------------
n_eval = min(len(true_future), len(predicted_array))
y_true_eval = true_future[:n_eval]
y_pred_eval = predicted_array[:n_eval]

mse  = float(mean_squared_error(y_true_eval, y_pred_eval))
mae  = float(mean_absolute_error(y_true_eval, y_pred_eval))
rmse = float(np.sqrt(mse))
r2   = float(r2_score(y_true_eval, y_pred_eval))
signal_range = float(np.max(y_true_eval) - np.min(y_true_eval)) or 1.0
norm_err = (mae / signal_range) * 100  # Normalised Error Score

st.subheader("Forecast Evaluation")
st.markdown(f"""
| Metric | Value |
|--------|-------|
| **R²** | {r2:.4f} |
| **MAE** | {mae:.6f} |
| **RMSE** | {rmse:.6f} |
| **Normalised Error** | {norm_err:.2f} % |
""")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
st.subheader("True vs Predicted")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(y_true_eval, label="True", alpha=0.8)
ax.plot(y_pred_eval, label="Predicted (RTD-RC)", linestyle="--", alpha=0.8)
ax.set_xlabel("Steps ahead")
ax.legend()
st.pyplot(fig)

st.subheader("Absolute Prediction Error")
fig_err, ax_err = plt.subplots(figsize=(11, 2))
ax_err.plot(np.abs(y_true_eval - y_pred_eval), color="orange")
ax_err.set_xlabel("Steps ahead")
ax_err.set_ylabel("Absolute error")
st.pyplot(fig_err)

st.caption("RTD-Reservoir autoregressive forecasting. "
           "Metrics are computed on the held-out future window only.")
