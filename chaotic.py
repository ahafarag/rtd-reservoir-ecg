import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from reservoir import Reservoir

st.set_page_config(page_title="Lorenz Reservoir Simulator", layout="wide")
st.title("Lorenz System Forecasting with RTD-Reservoir Computing")

# --- Settings ---
reservoir_size = st.sidebar.slider("Reservoir Size", 50, 1000, 300)
delay = st.sidebar.slider("Delay Steps", 1, 20, 2)
warmup_percent = st.sidebar.slider("Warm-up %", 0, 50, 10)
use_mlp = st.sidebar.radio("Readout Type", ["Ridge Regression", "MLP (Neural Network)"]) == "MLP (Neural Network)"

# --- Lorenz system ---
st.header("Lorenz Chaotic System Generation")

def lorenz(t, state, sigma=10, beta=8/3, rho=28):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Solve Lorenz system
t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[1], 4000)
sol = solve_ivp(lorenz, t_span, [1.0, 1.0, 1.0], t_eval=t_eval)
data = sol.y.T
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Select variable to forecast
component = st.selectbox("Select variable to forecast", ["x", "y", "z"])
var_index = {"x": 0, "y": 1, "z": 2}[component]
signal = data_scaled[:, var_index]

# --- Run reservoir ---
r = Reservoir(size=reservoir_size, delay=delay, use_mlp=use_mlp)
Y_pred = r.fit_predict(signal, warmup_percent=warmup_percent)
Y_true = r.Y_true

# --- Inverse transform for better interpretability ---
true_full = data[:, var_index][-len(Y_true):]
pred_full = scaler.inverse_transform(data_scaled[-len(Y_pred):])[:, var_index]

# --- Metrics ---
mse = mean_squared_error(true_full, pred_full)
mae = mean_absolute_error(true_full, pred_full)
rmse = np.sqrt(mse)
r2 = r2_score(true_full, pred_full)
accuracy = (1 - (mae / (np.max(true_full) - np.min(true_full)))) * 100

st.subheader("Evaluation Metrics")
st.markdown(f"""
- **Accuracy:** {accuracy:.2f}%
- **R²:** {r2:.4f}
- **MAE:** {mae:.6f}
- **RMSE:** {rmse:.6f}
""")

# --- Plots ---
st.subheader("True vs Predicted")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(true_full, label='True', alpha=0.7)
ax.plot(pred_full, label='Predicted', linestyle='--')
ax.legend()
st.pyplot(fig)

st.subheader("Prediction Error")
errors = np.abs(true_full - pred_full)
fig_err, ax_err = plt.subplots(figsize=(10, 2))
ax_err.plot(errors, label='Absolute Error', color='orange')
ax_err.legend()
st.pyplot(fig_err)

st.caption("This simulation uses the x/y/z projection of the Lorenz attractor with a reservoir computing model and leaky tanh dynamics.")
