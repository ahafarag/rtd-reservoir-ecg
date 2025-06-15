# bayesian_optimizer.py
'''
## PATENT NOTICE
This repository contains novel inventions related to multi-RTD reservoir computing for ECG prediction. 
Provisional patent application pending. All rights reserved. 
Unauthorized commercial use or replication of the multi-RTD architecture (>2 units) is prohibited.
'''

import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from tqdm import tqdm

class BayesianRTDOptimizer:
    """
    Advanced Bayesian optimizer for RTD Reservoir Computing hyperparameters
    with specialized features for ECG prediction systems
    """
    def __init__(self, X, y, reservoir_class, target='prediction'):
        """
        Initialize optimizer with ECG data and reservoir class
        
        Args:
            X: Input ECG data (shape: [samples, timesteps])
            y: Target values (shape: [samples])
            reservoir_class: The Reservoir class to optimize
            target: Optimization target ('prediction' or 'anomaly_detection')
        """
        self.X = X
        self.y = y
        self.Reservoir = reservoir_class
        self.target = target
        
        # Split data for validation during optimization
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Parameter bounds (optimized for RTD characteristics)
        self.pbounds = {
            'input_scaling': (0.1, 1.0),      # Input weight scaling
            'feedback_scaling': (0.0, 0.5),   # Feedback weight scaling (conservative)
            'v_bias': (0.3, 0.7),             # RTD bias voltage (normalized to NDR region)
            'leakage': (0.01, 0.5),           # Leakage rate
            'n_virtual': (30, 100)            # Number of virtual nodes
        }
        
        # RTD-specific constraints
        self.rtd_constraints = [
            {'name': 'stability', 'condition': lambda params: params['feedback_scaling'] < 0.4},
            {'name': 'bias_range', 'condition': lambda params: 0.3 <= params['v_bias'] <= 0.7}
        ]
        
        self.optimizer = BayesianOptimization(
            f=self._evaluate_config,
            pbounds=self.pbounds,
            random_state=42,
            verbose=0
        )
        
        # Optimization history
        self.performance_history = []
        self.parameter_history = []

    def _evaluate_config(self, input_scaling, feedback_scaling, v_bias, leakage, n_virtual):
        """Evaluate a hyperparameter configuration with RTD-specific constraints"""
        # Convert to integer parameters
        n_virtual = int(n_virtual)
        
        # Check RTD-specific constraints
        params = {
            'input_scaling': input_scaling,
            'feedback_scaling': feedback_scaling,
            'v_bias': v_bias,
            'leakage': leakage,
            'n_virtual': n_virtual
        }
        
        # Apply constraints
        for constraint in self.rtd_constraints:
            if not constraint['condition'](params):
                return -10  # Strong penalty for invalid configurations
                
        try:
            # Initialize reservoir with current parameters
            reservoir = self.Reservoir(
                size=n_virtual,
                leaky=leakage,
                input_scaling=input_scaling,
                feedback_scaling=feedback_scaling,
                v_bias=v_bias
            )
            
            # Generate reservoir states
            X_train_states = reservoir.fit_transform(
                self.X_train.flatten(),
                dropout_rate=0.0,
                noise_std=0.01
            )
            X_val_states = reservoir.fit_transform(
                self.X_val.flatten(),
                dropout_rate=0.0,
                noise_std=0.01
            )
            
            # Train readout layer
            readout = Ridge(alpha=1e-6)
            readout.fit(X_train_states, self.y_train)
            
            # Predict and evaluate
            if self.target == 'prediction':
                y_pred = readout.predict(X_val_states)
                score = -mean_squared_error(self.y_val, y_pred)  # Negative MSE to maximize
            else:  # Anomaly detection
                y_pred = readout.predict(X_val_states)
                y_binary = (y_pred > 0.5).astype(int)
                score = accuracy_score(self.y_val, y_binary)
                
            # Store performance for analysis
            self.performance_history.append(score)
            self.parameter_history.append(params)
                
            return score
            
        except Exception as e:
            print(f"Configuration failed: {e}")
            return -10  # Return low score for unstable configurations

    def optimize(self, init_points=10, n_iter=50):
        """Run Bayesian optimization with progress tracking"""
        print("⚙️ Starting Bayesian optimization for RTD reservoir...")
        self.optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )
        
        # Get best configuration
        best_params = self.optimizer.max['params']
        best_params['n_virtual'] = int(best_params['n_virtual'])
        
        print("\n✅ Optimization complete! Optimal parameters:")
        for param, value in best_params.items():
            print(f"{param:>20}: {value:.4f}")
            
        return best_params

    def visualize_optimization(self):
        """Visualize optimization progress and parameter relationships"""
        if not self.performance_history:
            print("No optimization data to visualize")
            return
            
        try:
            plt.figure(figsize=(15, 10))
            
            # Optimization progress
            plt.subplot(2, 2, 1)
            best_scores = np.maximum.accumulate(self.performance_history)
            plt.plot(best_scores, 'o-', markersize=4)
            plt.title('Optimization Progress')
            plt.xlabel('Iteration')
            plt.ylabel('Best Score')
            plt.grid(True)
            
            # Parameter importance
            param_names = list(self.pbounds.keys())
            importance_scores = []
            
            for param in param_names:
                values = [p[param] for p in self.parameter_history]
                corr = np.corrcoef(values, self.performance_history)[0, 1]
                importance_scores.append(abs(corr))
            
            plt.subplot(2, 2, 2)
            plt.barh(param_names, importance_scores)
            plt.title('Parameter Importance')
            plt.xlabel('Absolute Correlation with Score')
            plt.grid(True)
            
            # Parameter distributions
            for i, param in enumerate(param_names[:2]):
                plt.subplot(2, 2, 3+i)
                values = [p[param] for p in self.parameter_history]
                plt.scatter(values, self.performance_history, alpha=0.6)
                plt.title(f'{param} vs Performance')
                plt.xlabel(param)
                plt.ylabel('Score')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('rtd_optimization_results.png', dpi=300)
            plt.show()
            
        except Exception as e:
            print(f"Visualization failed: {e}")

    def get_optimal_reservoir(self):
        """Return a reservoir instance with optimal parameters"""
        best_params = self.optimizer.max['params']
        return self.Reservoir(
            size=int(best_params['n_virtual']),
            leaky=best_params['leakage'],
            input_scaling=best_params['input_scaling'],
            feedback_scaling=best_params['feedback_scaling'],
            v_bias=best_params['v_bias']
        )