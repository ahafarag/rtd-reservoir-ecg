'''
## PATENT NOTICE
This repository contains novel inventions related to multi-RTD reservoir computing for ECG prediction.
Provisional patent application pending. All rights reserved.
Unauthorized commercial use or replication of the multi-RTD architecture (>2 units) is prohibited.
'''

import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


class BayesianRTDOptimizer:
    """Bayesian hyperparameter optimiser for the RTD Reservoir Computing system."""

    def __init__(self, X, y, reservoir_class, target='prediction'):
        """
        Args:
            X:               Input ECG signal as a column vector (samples, 1).
            y:               Target values (samples,) — same signal, same order.
            reservoir_class: The Reservoir class to instantiate per trial.
            target:          'prediction' only (anomaly_detection removed — needs
                             labelled ground truth which is not available here).
        """
        self.X = X
        self.y = y
        self.Reservoir = reservoir_class
        self.target = target

        # Chronological 80/20 split — no shuffling for time-series data
        n = len(y)
        split = int(0.8 * n)
        self.X_train = X[:split].flatten()
        self.X_val   = X[split:].flatten()
        self.y_train = y[:split]
        self.y_val   = y[split:]

        self.pbounds = {
            'input_scaling':   (0.1, 1.0),
            'feedback_scaling':(0.0, 0.5),
            'v_bias':          (0.3, 0.7),
            'leakage':         (0.01, 0.5),
            'n_virtual':       (30, 100),
        }

        self.optimizer = BayesianOptimization(
            f=self._evaluate_config,
            pbounds=self.pbounds,
            random_state=42,
            verbose=0,
        )

        self.performance_history = []
        self.parameter_history = []

    def _evaluate_config(self, input_scaling, feedback_scaling, v_bias,
                         leakage, n_virtual):
        n_virtual = int(n_virtual)
        params = dict(input_scaling=input_scaling,
                      feedback_scaling=feedback_scaling,
                      v_bias=v_bias, leakage=leakage, n_virtual=n_virtual)

        # RTD stability constraints
        if feedback_scaling >= 0.4 or not (0.3 <= v_bias <= 0.7):
            return -10.0

        try:
            reservoir = self.Reservoir(
                size=n_virtual, leaky=leakage,
                input_scaling=input_scaling,
                feedback_scaling=feedback_scaling,
                v_bias=v_bias,
            )

            # Train states — reservoir evolves forward through training data
            X_train_states = reservoir.fit_transform(self.X_train,
                                                     noise_std=0.01)

            # Reset state before validation so val states are independent
            reservoir.states = np.zeros(reservoir.size)
            X_val_states = reservoir.fit_transform(self.X_val,
                                                   noise_std=0.01)

            if X_train_states.shape[0] == 0 or X_val_states.shape[0] == 0:
                return -10.0

            # Align targets with states (states start after warmup)
            n_tr = X_train_states.shape[0]
            n_vl = X_val_states.shape[0]
            y_tr = self.y_train[-n_tr:]
            y_vl = self.y_val[-n_vl:]

            readout = Ridge(alpha=1e-4)
            readout.fit(X_train_states, y_tr)
            y_pred = readout.predict(X_val_states)

            score = -float(mean_squared_error(y_vl, y_pred))  # maximise → minimise MSE

            self.performance_history.append(score)
            self.parameter_history.append(params)
            return score

        except Exception as e:
            print(f"Trial failed: {e}")
            return -10.0

    def optimize(self, init_points=10, n_iter=50):
        print("Starting Bayesian optimisation for RTD reservoir...")
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)

        best_params = self.optimizer.max['params']
        best_params['n_virtual'] = int(best_params['n_virtual'])

        print("\nOptimisation complete. Best parameters:")
        for k, v in best_params.items():
            print(f"  {k:>20}: {v:.4f}")

        return best_params

    def visualize_optimization(self):
        if not self.performance_history:
            print("No optimisation history to visualise.")
            return

        param_names = list(self.pbounds.keys())
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        axes[0, 0].plot(np.maximum.accumulate(self.performance_history), 'o-', ms=4)
        axes[0, 0].set_title('Best score over iterations')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Score (−MSE)')
        axes[0, 0].grid(True)

        importance = [abs(np.corrcoef(
            [p[n] for p in self.parameter_history],
            self.performance_history)[0, 1])
            for n in param_names]
        axes[0, 1].barh(param_names, importance)
        axes[0, 1].set_title('Parameter importance (|correlation|)')
        axes[0, 1].grid(True)

        for ax, param in zip(axes.flat[2:], param_names[:2]):
            values = [p[param] for p in self.parameter_history]
            ax.scatter(values, self.performance_history, alpha=0.6)
            ax.set_title(f'{param} vs score')
            ax.set_xlabel(param)
            ax.set_ylabel('Score')
            ax.grid(True)

        plt.tight_layout()
        plt.savefig('rtd_optimization_results.png', dpi=150)
        plt.show()

    def get_optimal_reservoir(self):
        best = self.optimizer.max['params']
        return self.Reservoir(
            size=int(best['n_virtual']), leaky=best['leakage'],
            input_scaling=best['input_scaling'],
            feedback_scaling=best['feedback_scaling'],
            v_bias=best['v_bias'],
        )
