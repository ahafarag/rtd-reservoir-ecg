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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rtd_model import rtd_nonlinearity


class Reservoir:
    def __init__(self, size=200, delay=5, use_mlp=True, show_progress=False,
                 leaky=1.0, activation="tanh", input_scaling=1.0,
                 feedback_scaling=0.0, v_bias=0.5):
        """
        RTD-inspired delay-based reservoir (single-node time-multiplexing scheme).

        Args:
            size:             Number of virtual nodes (n_virtual)
            delay:            Input delay in samples
            use_mlp:          True → MLPRegressor readout, False → Ridge readout
            show_progress:    Print training progress to stdout
            leaky:            Leakage rate α ∈ (0, 1]
            activation:       'tanh' | 'rtd' | 'linear'
            input_scaling:    Input weight scaling γ_in
            feedback_scaling: Output feedback weight γ_fb
            v_bias:           Normalised RTD bias voltage ∈ [0, 1]
        """
        self.size = size
        self.delay = delay
        self.use_mlp = use_mlp
        self.show_progress = show_progress
        self.leaky = leaky
        self.activation = activation
        self.input_scaling = input_scaling
        self.feedback_scaling = feedback_scaling
        self.v_bias = v_bias

        self.scaler = MinMaxScaler()
        self.states = np.zeros((self.size,))
        self.loss_curve = []

        self.Y_true = None
        self.Y_pred = None
        self.warmup_threshold = 0

    # ------------------------------------------------------------------
    # Nonlinearity
    # ------------------------------------------------------------------

    def _nonlinear(self, x):
        """RTD-inspired nonlinear activation.

        Activation modes:
          'tanh'  — standard leaky-ESN activation (fast, well-understood baseline)
          'rtd'   — Lorentzian NDR curve from rtd_model.rtd_nonlinearity();
                    physically motivated by double-barrier tunnel junction I-V
          'linear'— pass-through (for ablation studies)
        """
        if self.activation == "tanh":
            bias_effect = np.tanh(5 * (self.v_bias - 0.5))
            return np.tanh(x) + 0.1 * bias_effect
        elif self.activation == "rtd":
            return rtd_nonlinearity(np.atleast_1d(x), v_bias=self.v_bias).squeeze()
        else:
            return x

    # ------------------------------------------------------------------
    # State generation
    # ------------------------------------------------------------------

    def _generate_states(self, signal, dropout_rate=0.0, noise_std=0.0,
                         warmup_steps=200):
        """Evolve reservoir and collect (X, Y) pairs after warmup."""
        # Clamp warmup so it never consumes the whole signal
        warmup_steps = min(warmup_steps, max(0, len(signal) // 5))

        X, Y = [], []
        for i in range(self.delay, len(signal)):
            noisy_input = signal[i - self.delay] + np.random.normal(0, noise_std)

            self.states = (1 - self.leaky) * self.states + self.leaky * np.roll(self.states, 1)
            input_val = self.input_scaling * noisy_input
            if self.size > 0:
                input_val += self.feedback_scaling * self.states[-1]
            self.states[0] = self._nonlinear(input_val)

            if i < warmup_steps:
                continue  # spin up the reservoir, don't collect

            if dropout_rate > 0.0:
                mask = np.random.binomial(1, 1 - dropout_rate, self.states.shape)
                self.states = self.states * mask

            X.append(self.states.copy())
            Y.append(signal[i])

        return np.array(X), np.array(Y)

    def generate_XY(self, signal, dropout_rate=0.0, noise_std=0.0):
        """Public interface used by the multi-RTD ensemble in main_app.py."""
        warmup = min(200, len(signal) // 5)
        return self._generate_states(signal, dropout_rate=dropout_rate,
                                     noise_std=noise_std, warmup_steps=warmup)

    def fit_transform(self, signal, dropout_rate=0.0, noise_std=0.0):
        """Return reservoir states only (used by Bayesian optimizer)."""
        warmup = min(200, len(signal) // 5)
        X, _ = self._generate_states(signal, dropout_rate=dropout_rate,
                                     noise_std=noise_std, warmup_steps=warmup)
        return X

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit_predict(self, signal, dropout_rate=0.0, noise_std=0.0,
                    warmup_percent=10):
        """Train the readout and return predictions on training data.

        warmup_percent controls both:
          1. How many samples the reservoir evolves before collecting states.
          2. The index stored in self.warmup_threshold for plot annotation.
        """
        warmup_steps = max(1, int(warmup_percent / 100 * len(signal)))
        warmup_steps = min(warmup_steps, len(signal) // 5)

        X, Y = self._generate_states(signal, dropout_rate=dropout_rate,
                                     noise_std=noise_std,
                                     warmup_steps=warmup_steps)

        X_scaled = self.scaler.fit_transform(X)

        if self.use_mlp:
            if self.show_progress:
                print("Training MLP readout...")
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
        self.warmup_threshold = warmup_steps
        return Y_pred

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def get_metrics(self):
        """Return a dict of evaluation metrics split by warmup boundary."""
        if self.Y_true is None or self.Y_pred is None:
            raise RuntimeError("Call fit_predict() before get_metrics().")

        w = self.warmup_threshold

        def _metrics(yt, yp):
            mse = float(mean_squared_error(yt, yp))
            mae = float(mean_absolute_error(yt, yp))
            return {"mse": mse, "mae": mae}

        total = _metrics(self.Y_true, self.Y_pred)
        stable = _metrics(self.Y_true[w:], self.Y_pred[w:]) if w < len(self.Y_true) else total

        return {
            "mse_total":  total["mse"],
            "mae_total":  total["mae"],
            "mse_stable": stable["mse"],
            "mae_stable": stable["mae"],
        }

    def get_loss_curve(self):
        """Return MLP training loss curve (empty list for Ridge)."""
        return self.loss_curve

    # ------------------------------------------------------------------
    # Bayesian optimiser interface
    # ------------------------------------------------------------------

    def set_params(self, input_scaling, feedback_scaling, v_bias, leakage, n_virtual):
        self.input_scaling = input_scaling
        self.feedback_scaling = feedback_scaling
        self.v_bias = v_bias
        self.leaky = leakage
        self.size = int(n_virtual)
        self.states = np.zeros((self.size,))

    def generate_states(self, X):
        return self.fit_transform(X.flatten())
