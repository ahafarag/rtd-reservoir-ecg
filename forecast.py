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
from scipy.integrate import solve_ivp
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from reservoir import Reservoir

st.set_page_config(page_title="Unified Forecasting App", layout="wide")
st.title("RTD-Reservoir Future Forecasting (ECG + Chaotic)")

# --- Mode Selection ---
mode = st.radio("Select Input Source", ["Lorenz Chaotic System", "Synthetic ECG Signal"])
prediction_horizon = st.sidebar.slider("Prediction Horizon (steps)", 1, 100, 25)
reservoir_size = st.sidebar.slider("Reservoir Size", 50, 1000, 300)
delay = st.sidebar.slider("Delay Steps", 1, 20, 2)
warmup_percent = st.sidebar.slider("Warm-up %", 0, 50, 10)
use_mlp = st.sidebar.radio("Readout Type", ["Ridge Regression", "MLP (Neural Network)"]) == "MLP (Neural Network)"
inference_only = st.sidebar.checkbox("Inference Only (no training)", value=False)

# --- Load and prepare signal ---
if mode == "Lorenz Chaotic System":
    def lorenz(t, state, sigma=10, beta=8/3, rho=28):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    t_eval = np.linspace(0, 40, 4000)
    sol = solve_ivp(lorenz, [0, 40], [1.0, 1.0, 1.0], t_eval=t_eval)
    data = sol.y.T
    comp = st.selectbox("Forecast Lorenz Component", ["x", "y", "z"])
    index = {"x": 0, "y": 1, "z": 2}[comp]
    raw_signal = data[:, index]
else:
    t = np.linspace(0, 10, 4000)
    raw_signal = 0.6 * np.sin(2 * np.pi * 5 * t) + 0.2 * np.sin(2 * np.pi * 50 * t)
    st.info("Synthetic ECG-like waveform loaded")

scaler = MinMaxScaler()
signal = scaler.fit_transform(raw_signal.reshape(-1, 1)).flatten()
split_idx = int(len(signal) * 0.75)
train_signal = signal[:split_idx]

# --- Reservoir Training ---
r = Reservoir(size=reservoir_size, delay=delay, use_mlp=use_mlp)
X_train, Y_train = r.generate_XY(train_signal)
r.scaler.fit(X_train)

readout = None
if not inference_only:
    readout = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42) if use_mlp else Ridge(alpha=1.0)
    readout.fit(X_train, Y_train)

# --- Multi-step Future Forecast ---
last_input = train_signal[-delay:].tolist()
predicted = []
for _ in range(prediction_horizon):
    r.states = (1 - r.leaky) * r.states + r.leaky * np.roll(r.states, 1)
    r.states[0] = r._nonlinear(last_input[-delay])
    X_pred = r.scaler.transform(r.states.reshape(1, -1))
    next_val = readout.predict(X_pred)[0] if readout else 0.0
    predicted.append(next_val)
    last_input.append(next_val)
    last_input.pop(0)

predicted_array = scaler.inverse_transform(np.array(predicted).reshape(-1, 1)).flatten()
true_future = scaler.inverse_transform(signal[split_idx:split_idx + prediction_horizon].reshape(-1, 1)).flatten()

# --- Evaluation ---
mse = mean_squared_error(true_future, predicted_array)
mae = mean_absolute_error(true_future, predicted_array)
rmse = np.sqrt(mse)
r2 = r2_score(true_future, predicted_array)
accuracy = (1 - (mae / (np.max(true_future) - np.min(true_future)))) * 100

st.subheader("Forecast Evaluation")
st.markdown(f"""
- **Accuracy:** {accuracy:.2f}%
- **R²:** {r2:.4f}
- **MAE:** {mae:.6f}
- **RMSE:** {rmse:.6f}
""")

# --- Visualization ---
st.subheader("True vs Predicted")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(true_future, label="True Future", alpha=0.7)
ax.plot(predicted_array, label="Predicted", linestyle="--")
ax.legend()
st.pyplot(fig)

st.subheader("Prediction Error")
fig_err, ax_err = plt.subplots(figsize=(10, 2))
ax_err.plot(np.abs(true_future - predicted_array), label="Absolute Error", color='orange')
ax_err.legend()
st.pyplot(fig_err)

st.caption("RTD-based reservoir computing forecasting for both chaotic and ECG-like signals.")
