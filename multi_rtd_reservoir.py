'''
## PATENT NOTICE
This repository contains novel inventions related to multi-RTD reservoir computing for ECG prediction.
Provisional patent application pending. All rights reserved.
Unauthorized commercial use or replication of the multi-RTD architecture (>2 units) is prohibited.
'''

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from reservoir import Reservoir


class MultiRTDReservoir:
    """Parallel ensemble of RTD reservoirs with staggered delays.

    Architecture (matches README diagram):

        Input signal
            │
            ├──► RTD Reservoir 1 (delay = d)       ──► states_1
            ├──► RTD Reservoir 2 (delay = d+1)     ──► states_2
            │         ...
            └──► RTD Reservoir N (delay = d+N-1)   ──► states_N
                                                         │
                                              [concatenate horizontally]
                                                         │
                                               Trainable readout layer
                                                         │
                                                    Prediction ŷ

    Each unit runs independently (no cross-coupling between units at the
    hardware level; coupling is achieved implicitly through the shared
    readout that sees all states simultaneously).  This matches the
    delay-diversity architecture proposed in the thesis and aligns with
    the patent-pending multi-RTD design.
    """

    def __init__(self, n_units: int = 3, size: int = 200, base_delay: int = 2,
                 use_mlp: bool = False, activation: str = "rtd",
                 leaky: float = 0.3, input_scaling: float = 0.5,
                 feedback_scaling: float = 0.0, v_bias: float = 0.5):
        """
        Args:
            n_units:          Number of parallel RTD units (≥ 1).
            size:             Virtual nodes per unit (n_virtual).
            base_delay:       Delay for the first unit; each subsequent unit
                              receives base_delay + unit_index.
            use_mlp:          True → MLPRegressor readout; False → Ridge.
            activation:       Nonlinearity for every unit ('rtd' | 'tanh').
            leaky:            Leakage rate α shared across all units.
            input_scaling:    Input weight γ_in shared across all units.
            feedback_scaling: Feedback weight γ_fb shared across all units.
            v_bias:           RTD bias voltage (normalised) shared across units.
        """
        self.n_units = n_units
        self.size = size
        self.base_delay = base_delay
        self.use_mlp = use_mlp
        self.activation = activation
        self.leaky = leaky
        self.input_scaling = input_scaling
        self.feedback_scaling = feedback_scaling
        self.v_bias = v_bias

        self.units: list[Reservoir] = []
        self.readout = None
        self.scaler = StandardScaler()  # robust to zero-variance features

        self.Y_true = None
        self.Y_pred = None

        self._build_units()

    def _build_units(self):
        self.units = [
            Reservoir(
                size=self.size,
                delay=self.base_delay + i,
                use_mlp=False,           # readout is handled at ensemble level
                activation=self.activation,
                leaky=self.leaky,
                input_scaling=self.input_scaling,
                feedback_scaling=self.feedback_scaling,
                v_bias=self.v_bias,
            )
            for i in range(self.n_units)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_states(self, signal: np.ndarray) -> tuple:
        """Run all units and return concatenated states + aligned targets.

        Each unit's state is reset to zero before processing so that records
        are processed independently — critical for classification where each
        record must produce features based solely on its own signal content.

        Returns:
            X: (n_samples, n_units * size)  concatenated reservoir states
            Y: (n_samples,)                 target values (aligned to shortest)
        """
        blocks, Y_ref = [], None
        for unit in self.units:
            unit.states = np.zeros(unit.size)   # reset before each record
            X_r, Y_r = unit.generate_XY(signal)
            blocks.append(X_r)
            Y_ref = Y_r

        min_len = min(len(b) for b in blocks + [Y_ref])
        X = np.hstack([b[:min_len] for b in blocks])
        Y = Y_ref[:min_len]
        return X, Y

    def fit(self, signal: np.ndarray, train_ratio: float = 0.8):
        """Train the ensemble readout on the training portion of the signal.

        Args:
            signal:      Normalised input ECG/chaotic signal (1-D array).
            train_ratio: Fraction of samples used for training (chronological).

        Returns:
            Y_pred_test: Predictions on the held-out test set.
        """
        X, Y = self.generate_states(signal)

        split = int(train_ratio * len(Y))
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]

        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc  = self.scaler.transform(X_test)

        n_features = X_train_sc.shape[1]
        n_samples  = X_train_sc.shape[0]

        if n_features >= n_samples:
            import warnings
            warnings.warn(
                f"MultiRTDReservoir: feature count ({n_features}) >= training "
                f"samples ({n_samples}). Increasing Ridge alpha automatically. "
                f"Consider reducing --reservoir-size or --n-units.",
                UserWarning, stacklevel=2
            )

        if self.use_mlp:
            self.readout = MLPRegressor(hidden_layer_sizes=(100,),
                                        max_iter=1000, random_state=42)
        else:
            # Scale alpha with feature/sample ratio to prevent underdetermined collapse
            alpha = max(1.0, n_features / max(n_samples, 1) * 10.0)
            self.readout = Ridge(alpha=alpha)

        self.readout.fit(X_train_sc, Y_train)

        self.Y_true = Y_test
        self.Y_pred = self.readout.predict(X_test_sc)
        return self.Y_pred

    def get_metrics(self) -> dict:
        """Return evaluation metrics on the held-out test set."""
        if self.Y_true is None:
            raise RuntimeError("Call fit() before get_metrics().")
        yt, yp = self.Y_true, self.Y_pred
        mae  = float(mean_absolute_error(yt, yp))
        mse  = float(mean_squared_error(yt, yp))
        r2   = float(r2_score(yt, yp))
        rng  = float(np.max(yt) - np.min(yt)) or 1.0
        return {
            "mae":           mae,
            "rmse":          float(np.sqrt(mse)),
            "r2":            r2,
            "norm_error_pct": (mae / rng) * 100,
        }

    def set_params(self, **kwargs):
        """Update hyperparameters and rebuild all units (used by optimiser)."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._build_units()
