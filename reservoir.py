# reservoir.py (updated with tanh + leaky integration support)
'''
## PATENT NOTICE
This repository contains novel inventions related to multi-RTD reservoir computing for ECG prediction. 
Provisional patent application pending. All rights reserved. 
Unauthorized commercial use or replication of the multi-RTD architecture (>2 units) is prohibited.
'''

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

class Reservoir:
    def __init__(self, size=200, delay=5, use_mlp=True, show_progress=False, leaky=1.0, activation="tanh"):
        self.size = size
        self.delay = delay
        self.use_mlp = use_mlp
        self.show_progress = show_progress
        self.leaky = leaky
        self.activation = activation
        self.scaler = MinMaxScaler()
        self.states = np.zeros((self.size,))
        self.loss_curve = []

    def _nonlinear(self, x):
        if self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "custom":
            return x - x**3 + np.exp(-x)
        else:
            return x
    def generate_XY(self, signal, dropout_rate=0.0, noise_std=0.0):
        X, Y = self._generate_states(signal, dropout_rate=dropout_rate, noise_std=noise_std, dynamic_warmup=False)
        return X, Y
    
    def _generate_states(self, signal, dropout_rate=0.0, noise_std=0.0, dynamic_warmup=True):
        X, Y = [], []
        warmup_threshold = int(0.1 * len(signal)) if dynamic_warmup else 200
        for i in range(self.delay, len(signal)):
            if i < warmup_threshold:
                # Allow reservoir to warm up without collecting outputs
                noisy_input = signal[i - self.delay] + np.random.normal(0, noise_std)
                self.states = (1 - self.leaky) * self.states + self.leaky * np.roll(self.states, 1)
                self.states[0] = self._nonlinear(noisy_input)
                continue
            noisy_input = signal[i - self.delay] + np.random.normal(0, noise_std)
            self.states = (1 - self.leaky) * self.states + self.leaky * np.roll(self.states, 1)
            input_val = noisy_input
            self.states[0] = self._nonlinear(input_val)
            if dropout_rate > 0.0:
                mask = np.random.binomial(1, 1 - dropout_rate, self.states.shape)
                self.states *= mask
            X.append(self.states.copy())
            Y.append(signal[i])
        return np.array(X), np.array(Y)
    def fit_transform(self, signal, warmup_percent=10, dropout_rate=0.0, noise_std=0.0):
        """Return raw reservoir states without prediction, for multi-RTD use."""
        X, _ = self._generate_states(signal, dropout_rate=dropout_rate, noise_std=noise_std, dynamic_warmup=False)
        return X    
    
    def fit_predict(self, signal, dropout_rate=0.0, noise_std=0.0, warmup_percent=10):
        X, Y = self._generate_states(signal, dropout_rate=dropout_rate, noise_std=noise_std, dynamic_warmup=False)
        warmup_threshold = int(warmup_percent / 100 * len(signal))
        X_scaled = self.scaler.fit_transform(X)
        if self.use_mlp:
            with st.spinner('Training MLP...'):
                self.readout = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42, verbose=self.show_progress)
                self.readout.fit(X_scaled, Y)
                self.loss_curve = self.readout.loss_curve_
        else:
            self.readout = Ridge()
            self.readout.fit(X_scaled, Y)
        Y_pred = self.readout.predict(X_scaled)
        self.Y_true = np.array(Y)
        self.Y_pred = np.array(Y_pred)
        self.warmup_threshold = warmup_threshold
        return Y_pred

    def get_metrics(self):
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse_total = mean_squared_error(self.Y_true, self.Y_pred)
        mae_total = mean_absolute_error(self.Y_true, self.Y_pred)
        stable_true = self.Y_true[self.warmup_threshold:]
        stable_pred = self.Y_pred[self.warmup_threshold:]
        mse_stable = mean_squared_error(stable_true, stable_pred)
        mae_stable = mean_absolute_error(stable_true, stable_pred)
        return {
            'mse_total': mse_total,
            'mae_total': mae_total,
            'mse_stable': mse_stable,
            'mae_stable': mae_stable
        }
    


    def get_loss_curve(self):
        return self.loss_curve
