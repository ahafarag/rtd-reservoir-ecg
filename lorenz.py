import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from reservoir import Reservoir

st.set_page_config(page_title="Lorenz Forecasting with RTD-Reservoir", layout="wide")
st.title("Lorenz System Forecasting with Reservoir Computing")

# Settings
prediction_horizon = st.sidebar.slider("Prediction Horizon (steps)", 1, 100, 25)
reservoir_size = st.sidebar.slider("Reservoir Size", 50, 1000, 300)
delay = st.sidebar.slider("Delay Steps", 1, 20, 2)
use_mlp = st.sidebar.radio("Readout Type", ["Ridge Regression", "MLP (Neural Network)"]) == "MLP (Neural Network)"

# Lorenz system generation
def lorenz(t, state, sigma=10, beta=8/3, rho=28):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

t_eval = np.linspace(0, 40, 4000)
sol = solve_ivp(lorenz, [0, 40], [1.0, 1.0, 1.0], t_eval=t_eval)
data = sol.y.T

# Show Lorenz attractor plot
st.subheader("Lorenz Attractor")
fig_attractor = plt.figure(figsize=(6, 5))
ax = fig_attractor.add_subplot(projection='3d')
ax.plot(data[:, 0], data[:, 1], data[:, 2], lw=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
st.pyplot(fig_attractor)

# Select Lorenz component
comp = st.selectbox("Select Component to Forecast", ["x", "y", "z"])
index = {"x": 0, "y": 1, "z": 2}[comp]
raw_signal = data[:, index]

scaler = MinMaxScaler()
signal = scaler.fit_transform(raw_signal.reshape(-1, 1)).flatten()
split_idx = int(len(signal) * 0.8)
train_signal = signal[:split_idx]
test_signal = signal[split_idx:]

# Reservoir
r = Reservoir(size=reservoir_size, delay=delay, use_mlp=use_mlp)
X_train, Y_train = r.generate_XY(train_signal)
r.scaler.fit(X_train)

readout = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42) if use_mlp else Ridge(alpha=1.0)
readout.fit(X_train, Y_train)

# Forecast the entire test set recursively
predicted = []
true_values = []
current_input = train_signal[-delay:].tolist()

for i in range(len(test_signal)):
    r.states = (1 - r.leaky) * r.states + r.leaky * np.roll(r.states, 1)
    r.states[0] = r._nonlinear(current_input[-delay])
    X_pred = r.scaler.transform(r.states.reshape(1, -1))
    next_val = readout.predict(X_pred)[0]
    predicted.append(next_val)
    true_values.append(test_signal[i])
    current_input.append(next_val)
    current_input.pop(0)

predicted_array = scaler.inverse_transform(np.array(predicted).reshape(-1, 1)).flatten()
true_array = scaler.inverse_transform(np.array(true_values).reshape(-1, 1)).flatten()

# Evaluation
mse = mean_squared_error(true_array, predicted_array)
mae = mean_absolute_error(true_array, predicted_array)
rmse = np.sqrt(mse)
r2 = r2_score(true_array, predicted_array)
accuracy = (1 - (mae / (np.max(true_array) - np.min(true_array)))) * 100

st.subheader("Forecast Evaluation")
st.markdown(f"""
- **Accuracy:** {accuracy:.2f}%
- **R²:** {r2:.4f}
- **MAE:** {mae:.6f}
- **RMSE:** {rmse:.6f}
""")

# Prediction plots
st.subheader("True vs Predicted")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(true_array, label="True Future", alpha=0.7)
ax.plot(predicted_array, label="Predicted", linestyle="--")
ax.legend()
st.pyplot(fig)

st.subheader("Prediction Error")
fig_err, ax_err = plt.subplots(figsize=(10, 2))
ax_err.plot(np.abs(true_array - predicted_array), label="Absolute Error", color='orange')
ax_err.legend()
st.pyplot(fig_err)

st.caption("Lorenz system forecast using RTD-Reservoir computing with recursive multi-step prediction.")
