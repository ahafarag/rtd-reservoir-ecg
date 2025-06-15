# reservoir.py (updated for Bayesian optimization)
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
    def __init__(self, size=200, delay=5, use_mlp=True, show_progress=False, 
                 leaky=1.0, activation="tanh", input_scaling=1.0, 
                 feedback_scaling=0.0, v_bias=0.5):
        """
        Enhanced RTD Reservoir with parameters for Bayesian optimization
        
        Args:
            size: Number of virtual nodes (n_virtual)
            leaky: Leakage rate (α)
            input_scaling: Input weight scaling (γ_in)
            feedback_scaling: Feedback weight scaling (γ_fb)
            v_bias: RTD bias voltage (normalized 0-1)
        """
        self.size = size
        self.delay = delay
        self.use_mlp = use_mlp
        self.show_progress = show_progress
        self.leaky = leaky
        self.activation = activation
        self.input_scaling = input_scaling
        self.feedback_scaling = feedback_scaling
        self.v_bias = v_bias  # Mapped to actual voltage in physical implementation
        
        self.scaler = MinMaxScaler()
        self.states = np.zeros((self.size,))
        self.loss_curve = []

    def _nonlinear(self, x):
        """RTD-inspired nonlinear activation function"""
        # v_bias controls operating point in NDR region
        bias_effect = np.tanh(5 * (self.v_bias - 0.5))
        
        if self.activation == "tanh":
            return np.tanh(x) + 0.1 * bias_effect
        elif self.activation == "custom":
            return x - 0.7*x**3 + 0.2*np.exp(-x) + 0.1 * bias_effect
        else:
            return x + 0.1 * bias_effect

    def generate_XY(self, signal, dropout_rate=0.0, noise_std=0.0):
        X, Y = self._generate_states(signal, dropout_rate=dropout_rate, 
                                    noise_std=noise_std, dynamic_warmup=False)
        return X, Y
    
    def _generate_states(self, signal, dropout_rate=0.0, noise_std=0.0, dynamic_warmup=True):
        X, Y = [], []
        warmup_threshold = int(0.1 * len(signal)) if dynamic_warmup else 200
        
        for i in range(self.delay, len(signal)):
            if i < warmup_threshold:
                # Allow reservoir to warm up without collecting outputs
                noisy_input = signal[i - self.delay] + np.random.normal(0, noise_std)
                self.states = (1 - self.leaky) * self.states + self.leaky * np.roll(self.states, 1)
                
                # Apply input scaling and feedback
                input_val = self.input_scaling * noisy_input
                if self.size > 0:
                    input_val += self.feedback_scaling * self.states[-1]
                    
                self.states[0] = self._nonlinear(input_val)
                continue
                
            noisy_input = signal[i - self.delay] + np.random.normal(0, noise_std)
            self.states = (1 - self.leaky) * self.states + self.leaky * np.roll(self.states, 1)
            
            # Apply input scaling and feedback
            input_val = self.input_scaling * noisy_input
            if self.size > 0:
                input_val += self.feedback_scaling * self.states[-1]
                
            self.states[0] = self._nonlinear(input_val)
            
            if dropout_rate > 0.0:
                mask = np.random.binomial(1, 1 - dropout_rate, self.states.shape)
                self.states *= mask
                
            X.append(self.states.copy())
            Y.append(signal[i])
            
        return np.array(X), np.array(Y)
        
    def fit_transform(self, signal, warmup_percent=10, dropout_rate=0.0, noise_std=0.0):
        """Return reservoir states for Bayesian optimization"""
        X, _ = self._generate_states(signal, dropout_rate=dropout_rate, 
                                    noise_std=noise_std, dynamic_warmup=False)
        return X    
    
    def fit_predict(self, signal, dropout_rate=0.0, noise_std=0.0, warmup_percent=10):
        X, Y = self._generate_states(signal, dropout_rate=dropout_rate, 
                                    noise_std=noise_std, dynamic_warmup=False)
        warmup_threshold = int(warmup_percent / 100 * len(signal))
        
        X_scaled = self.scaler.fit_transform(X)
        if self.use_mlp:
            with st.spinner('Training MLP...'):
                self.readout = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, 
                                           random_state=42, verbose=self.show_progress)
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

    # ... (keep existing get_metrics and get_loss_curve methods) ...

    def set_params(self, input_scaling, feedback_scaling, v_bias, leakage, n_virtual):
        """For Bayesian optimization interface"""
        self.input_scaling = input_scaling
        self.feedback_scaling = feedback_scaling
        self.v_bias = v_bias
        self.leaky = leakage
        self.size = int(n_virtual)
        self.states = np.zeros((self.size,))  # Reset state

    def generate_states(self, X):
        """Interface for Bayesian optimizer"""
        return self.fit_transform(X.flatten())