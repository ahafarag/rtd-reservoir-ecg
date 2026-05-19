"""
run.py - Local CLI runner for the RTD Reservoir Computing ECG project.

No Streamlit required. All experiments run end-to-end from the terminal.

Usage examples
--------------
# Quick smoke test (no dataset needed):
    python run.py --mode lorenz

# ECG forecasting on PTB-XL:
    python run.py --mode forecast --ptbxl-path ./ptb-xl-1.0.1 --index 0 --lead II

# Arrhythmia classification on PTB-XL:
    python run.py --mode classify --ptbxl-path ./ptb-xl-1.0.1 --max-records 200

# Bayesian optimisation then forecast:
    python run.py --mode forecast --ptbxl-path ./ptb-xl-1.0.1 --bayesian

All outputs (metrics JSON, results CSV, PNG plots) are saved to ./output/
"""

import argparse
import json
import os
import sys
import ast

# Force UTF-8 output so Unicode characters render on all terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display required
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (r2_score, mean_absolute_error,
                              mean_squared_error, accuracy_score,
                              classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler

from ecg_loader import (load_ecg_data, load_ptbxl_record,
                        load_ptbxl_metadata, load_mitbih_record,
                        list_mitbih_records, bandpass_filter, normalise)
from multi_rtd_reservoir import MultiRTDReservoir
from bayesian_optimizer import BayesianRTDOptimizer
from reservoir import Reservoir
from utils import save_model, load_model

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stamp(name: str, ext: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{name}_{TIMESTAMP}.{ext}")


def _save_metrics(metrics: dict, label: str):
    path = _stamp(f"metrics_{label}", "json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {path}")


def _save_csv(df: pd.DataFrame, label: str):
    path = _stamp(f"results_{label}", "csv")
    df.to_csv(path, index=False)
    print(f"  CSV saved     → {path}")


def _save_plot(fig: plt.Figure, label: str):
    path = _stamp(f"plot_{label}", "png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved    → {path}")


def _print_metrics(metrics: dict, title: str = ""):
    width = 44
    print("\n" + "-" * width)
    if title:
        print(f"  {title}")
        print("-" * width)
    for k, v in metrics.items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:<28} {val}")
    print("-" * width)


def _build_reservoir(args, opt_params=None) -> MultiRTDReservoir:
    kwargs = dict(
        n_units=args.n_units,
        size=args.reservoir_size,
        base_delay=args.delay,
        activation=args.activation,
        leaky=args.leaky,
        use_mlp=args.use_mlp,
    )
    if opt_params:
        kwargs.update(
            size=int(opt_params["n_virtual"]),
            leaky=opt_params["leakage"],
        )
    return MultiRTDReservoir(**kwargs)


# ---------------------------------------------------------------------------
# Mode: Lorenz (smoke test, no dataset)
# ---------------------------------------------------------------------------

def run_lorenz(args):
    print("\n[MODE] Lorenz chaotic system forecast")
    from scipy.integrate import solve_ivp

    def lorenz(t, s, sigma=10, beta=8/3, rho=28):
        x, y, z = s
        return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

    sol = solve_ivp(lorenz, [0, 40], [1., 1., 1.],
                    t_eval=np.linspace(0, 40, 4000))
    raw = sol.y.T[:, 0]                          # x-component

    # Single normalisation only — z-score keeps signal in a stable range
    # for the reservoir without distorting the inverse transform.
    mean_, std_ = raw.mean(), raw.std() + 1e-8
    signal = (raw - mean_) / std_

    reservoir = _build_reservoir(args)
    Y_pred = reservoir.fit(signal, train_ratio=0.8)
    Y_true = reservoir.Y_true

    # Inverse transform to original Lorenz units for interpretable metrics
    Y_pred_orig = Y_pred * std_ + mean_
    Y_true_orig = Y_true * std_ + mean_

    metrics = {
        "r2":             float(r2_score(Y_true_orig, Y_pred_orig)),
        "mae":            float(mean_absolute_error(Y_true_orig, Y_pred_orig)),
        "rmse":           float(np.sqrt(mean_squared_error(Y_true_orig, Y_pred_orig))),
        "norm_error_pct": reservoir.get_metrics()["norm_error_pct"],
    }
    _print_metrics(metrics, "Lorenz Forecast — Test Set (last 20%)")
    _save_metrics(metrics, "lorenz")

    result_df = pd.DataFrame({"True": Y_true_orig, "Predicted": Y_pred_orig})
    result_df["AbsError"] = np.abs(result_df["True"] - result_df["Predicted"])
    _save_csv(result_df, "lorenz")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(Y_true_orig, label="True", alpha=0.8)
    axes[0].plot(Y_pred_orig, label="Predicted", linestyle="--", alpha=0.8)
    axes[0].set_title(f"Lorenz Forecast  |  R² = {metrics['r2']:.4f}")
    axes[0].legend()
    axes[1].plot(result_df["AbsError"], color="orange", label="Absolute Error")
    axes[1].set_xlabel("Samples (test set)")
    axes[1].legend()
    fig.tight_layout()
    _save_plot(fig, "lorenz")


# ---------------------------------------------------------------------------
# Mode: ECG Forecasting (PTB-XL)
# ---------------------------------------------------------------------------

def run_forecast(args):
    res = "500 Hz HR" if args.hr else "100 Hz LR"
    print(f"\n[MODE] ECG Forecasting — PTB-XL  (index={args.index}, lead={args.lead}, {res})")
    if not args.ptbxl_path or not os.path.exists(args.ptbxl_path):
        sys.exit("ERROR: --ptbxl-path is required for forecast mode and must exist.")

    print("  Loading ECG record...")
    df = load_ptbxl_record(args.ptbxl_path, index=args.index,
                            lead=args.lead, apply_filter=True, use_hr=args.hr)
    signal = normalise(df["ecg"].values)
    print(f"  Signal length: {len(signal)} samples  |  fs={df.attrs.get('fs', '?')} Hz")

    opt_params = None
    if args.bayesian:
        print("  Running Bayesian optimisation (this may take a few minutes)...")
        # Use up to 3000 samples so the optimizer sees enough of the signal.
        # More samples = better generalisation for the chosen hyperparameters.
        sample_size = min(3000, len(signal))
        sample = signal[:sample_size]
        print(f"  Optimisation sample: {sample_size} samples")
        optimizer = BayesianRTDOptimizer(
            sample.reshape(-1, 1), sample, Reservoir, target="prediction")
        opt_params = optimizer.optimize(init_points=5, n_iter=args.bayes_iters)
        _print_metrics(opt_params, "Optimised Parameters")
        with open(_stamp("optimized_params", "json"), "w") as f:
            json.dump(opt_params, f, indent=2)

    reservoir = _build_reservoir(args, opt_params)
    print(f"  Running {reservoir.n_units}-unit RTD reservoir  "
          f"(size={reservoir.size}, activation={reservoir.activation})...")
    Y_pred = reservoir.fit(signal, train_ratio=0.8)
    Y_true = reservoir.Y_true

    metrics = reservoir.get_metrics()
    metrics["r2"] = float(r2_score(Y_true, Y_pred))
    _print_metrics(metrics, f"ECG Forecast — Test Set  |  lead {args.lead}")
    _save_metrics(metrics, "forecast")

    result_df = pd.DataFrame({"True_ECG": Y_true, "Predicted_ECG": Y_pred})
    result_df["AbsError"] = np.abs(Y_true - Y_pred)
    _save_csv(result_df, "forecast")

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(Y_true,  label="True ECG",      alpha=0.8)
    axes[0].plot(Y_pred,  label="Predicted ECG", alpha=0.8, linestyle="--")
    axes[0].set_title(f"ECG Waveform Prediction  |  "
                       f"R² = {metrics['r2']:.4f}  |  MAE = {metrics['mae']:.5f}")
    axes[0].legend()

    axes[1].plot(result_df["AbsError"], color="orange", label="Absolute Error")
    threshold = result_df["AbsError"].mean() + 2 * result_df["AbsError"].std()
    axes[1].axhline(threshold, color="red", linestyle="--", label=f"Anomaly threshold (μ+2σ)")
    axes[1].legend()

    anomalies = result_df[result_df["AbsError"] > threshold]
    axes[2].plot(Y_true, alpha=0.6, label="True ECG")
    axes[2].scatter(anomalies.index, anomalies["True_ECG"],
                    color="red", s=15, label=f"Anomalies ({len(anomalies)})")
    axes[2].legend()
    axes[2].set_xlabel("Samples (test set)")

    fig.tight_layout()
    _save_plot(fig, "forecast")
    print(f"\n  Anomalies detected: {len(anomalies)}")


# ---------------------------------------------------------------------------
# Mode: Arrhythmia Classification (PTB-XL)
# ---------------------------------------------------------------------------

def _parse_scp(scp_str):
    try:
        return ast.literal_eval(scp_str)
    except Exception:
        return {}


def _is_normal(scp_dict):
    return int("NORM" in scp_dict and scp_dict["NORM"] > 0)


def run_classify(args):
    print(f"\n[MODE] Arrhythmia Classification — PTB-XL  "
          f"(max_records={args.max_records}, lead={args.lead})")
    if not args.ptbxl_path or not os.path.exists(args.ptbxl_path):
        sys.exit("ERROR: --ptbxl-path is required for classify mode and must exist.")

    meta = load_ptbxl_metadata(args.ptbxl_path)
    meta["label"] = meta["scp_codes"].apply(_parse_scp).apply(_is_normal)

    train_meta = meta[meta["strat_fold"] <= 9].reset_index(drop=True)
    test_meta  = meta[meta["strat_fold"] == 10].reset_index(drop=True)

    n_per_class = args.max_records // 2
    train_meta = pd.concat([
        train_meta[train_meta["label"] == 1].head(n_per_class),
        train_meta[train_meta["label"] == 0].head(n_per_class),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    test_meta = pd.concat([
        test_meta[test_meta["label"] == 1].head(max(1, n_per_class // 5)),
        test_meta[test_meta["label"] == 0].head(max(1, n_per_class // 5)),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  Train: {len(train_meta)} records  |  Test: {len(test_meta)} records")

    reservoir = _build_reservoir(args)

    def extract(subset, label):
        features, labels = [], []
        for i, (_, row) in enumerate(subset.iterrows()):
            sys.stdout.write(f"\r  Loading {label}: {i+1}/{len(subset)}")
            sys.stdout.flush()
            try:
                df = load_ptbxl_record(args.ptbxl_path,
                                       index=int(row.name),
                                       lead=args.lead,
                                       apply_filter=True,
                                       use_hr=args.hr)
                sig = normalise(df["ecg"].values)
                X_r, _ = reservoir.generate_states(sig)
                if X_r.shape[0] == 0:
                    continue
                feat = np.concatenate([
                    X_r.mean(axis=0), X_r.std(axis=0),
                    X_r.max(axis=0),  X_r.min(axis=0),
                ])
                features.append(feat)
                labels.append(int(row["label"]))
            except Exception:
                pass
        print()
        return np.array(features), np.array(labels)

    X_train, y_train = extract(train_meta, "train")
    X_test,  y_test  = extract(test_meta,  "test ")

    if len(X_train) == 0 or len(X_test) == 0:
        sys.exit("ERROR: Could not extract features. Check dataset path and lead name.")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    clf = (MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
           if args.use_mlp else RidgeClassifier(alpha=1.0))
    print(f"  Training {type(clf).__name__}...")
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)

    report = classification_report(y_test, y_pred,
                                   target_names=["Abnormal", "Normal"],
                                   output_dict=True)
    metrics = {
        "accuracy":    float(accuracy_score(y_test, y_pred)),
        "sensitivity": float(report["Normal"]["recall"]),
        "specificity": float(report["Abnormal"]["recall"]),
        "f1_normal":   float(report["Normal"]["f1-score"]),
        "f1_abnormal": float(report["Abnormal"]["f1-score"]),
        "n_train":     int(len(y_train)),
        "n_test":      int(len(y_test)),
    }
    _print_metrics(metrics, "Classification — strat_fold 10 (patient-independent test)")
    _save_metrics(metrics, "classify")

    result_df = pd.DataFrame({"True": y_test, "Predicted": y_pred})
    _save_csv(result_df, "classify")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Abnormal", "Normal"])
    ax.set_yticklabels(["Abnormal", "Normal"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix  |  Accuracy = {metrics['accuracy']*100:.1f}%")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_plot(fig, "classify_confusion")


# ---------------------------------------------------------------------------
# Mode: Anomaly-based Classification (one-class RC approach)
# ---------------------------------------------------------------------------

def run_anomaly_classify(args):
    """One-class reservoir classification via reconstruction error.

    Train the multi-RTD ensemble as a one-step predictor on NORM records only.
    At test time, compute the prediction MAE for every record.
    NORM records produce low error (the reservoir knows their dynamics).
    ABNORMAL records produce high error (unseen dynamics).
    A threshold on the MAE separates the two classes.
    """
    print(f"\n[MODE] Anomaly-based Classification — PTB-XL  "
          f"(lead={args.lead}, {'500Hz HR' if args.hr else '100Hz LR'})")
    if not args.ptbxl_path or not os.path.exists(args.ptbxl_path):
        sys.exit("ERROR: --ptbxl-path is required.")

    meta = load_ptbxl_metadata(args.ptbxl_path)
    meta["label"] = meta["scp_codes"].apply(_parse_scp).apply(_is_normal)

    # Training: NORM records from folds 1-9 only
    train_norm = meta[(meta["strat_fold"] <= 9) & (meta["label"] == 1)].head(
        args.max_records // 2).reset_index(drop=True)
    # Test: balanced NORM + ABNORMAL from fold 10
    test_norm   = meta[(meta["strat_fold"] == 10) & (meta["label"] == 1)].head(
        args.max_records // 10).reset_index(drop=True)
    test_abnorm = meta[(meta["strat_fold"] == 10) & (meta["label"] == 0)].head(
        args.max_records // 10).reset_index(drop=True)

    print(f"  Train (NORM only): {len(train_norm)} records")
    print(f"  Test  (NORM):      {len(test_norm)} records")
    print(f"  Test  (ABNORMAL):  {len(test_abnorm)} records")

    reservoir = _build_reservoir(args)

    def _load_signal(row_name):
        df = load_ptbxl_record(args.ptbxl_path, index=int(row_name),
                               lead=args.lead, apply_filter=True,
                               use_hr=args.hr)
        return normalise(df["ecg"].values)

    # --- Train readout on NORM records ---
    print("\n  Training readout on NORM records...")
    X_all, Y_all = [], []
    for i, (_, row) in enumerate(train_norm.iterrows()):
        sys.stdout.write(f"\r  Training: {i+1}/{len(train_norm)}")
        sys.stdout.flush()
        try:
            sig = _load_signal(row.name)
            X_r, Y_r = reservoir.generate_states(sig)
            if X_r.shape[0] > 0:
                X_all.append(X_r)
                Y_all.append(Y_r)
        except Exception:
            pass
    print()

    if not X_all:
        sys.exit("ERROR: Could not load any training records.")

    X_train = np.vstack(X_all)
    Y_train = np.concatenate(Y_all)

    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    from sklearn.linear_model import Ridge
    readout = Ridge(alpha=1.0)
    readout.fit(X_train_sc, Y_train)
    print(f"  Readout trained on {len(X_train):,} state vectors from {len(X_all)} records.")

    # --- Compute reconstruction MAE for every test record ---
    def _record_mae(row_name):
        sig = _load_signal(row_name)
        X_r, Y_r = reservoir.generate_states(sig)
        if X_r.shape[0] == 0:
            return np.nan
        X_sc   = scaler.transform(X_r)
        Y_pred = readout.predict(X_sc)
        return float(np.mean(np.abs(Y_r - Y_pred)))

    print("\n  Computing reconstruction error on test records...")
    norm_errors, abnorm_errors = [], []

    for i, (_, row) in enumerate(test_norm.iterrows()):
        sys.stdout.write(f"\r  NORM test:     {i+1}/{len(test_norm)}")
        sys.stdout.flush()
        try:
            norm_errors.append(_record_mae(row.name))
        except Exception:
            pass
    print()

    for i, (_, row) in enumerate(test_abnorm.iterrows()):
        sys.stdout.write(f"\r  ABNORMAL test: {i+1}/{len(test_abnorm)}")
        sys.stdout.flush()
        try:
            abnorm_errors.append(_record_mae(row.name))
        except Exception:
            pass
    print()

    norm_errors   = np.array([e for e in norm_errors   if not np.isnan(e)])
    abnorm_errors = np.array([e for e in abnorm_errors if not np.isnan(e)])

    print(f"\n  Mean MAE  NORM:     {norm_errors.mean():.6f}  (std={norm_errors.std():.6f})")
    print(f"  Mean MAE  ABNORMAL: {abnorm_errors.mean():.6f}  (std={abnorm_errors.std():.6f})")

    # --- Find best threshold via ROC ---
    from sklearn.metrics import roc_auc_score, roc_curve
    y_true   = np.concatenate([np.ones(len(norm_errors)),
                                np.zeros(len(abnorm_errors))])
    y_scores = np.concatenate([-norm_errors, -abnorm_errors])  # lower MAE = more normal

    auc = float(roc_auc_score(y_true, y_scores))
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Youden J = sensitivity + specificity - 1; pick threshold that maximises it
    j_scores = tpr - fpr
    best_idx  = np.argmax(j_scores)
    best_thr  = -thresholds[best_idx]   # convert back to MAE threshold

    y_pred = (np.concatenate([norm_errors, abnorm_errors]) <= best_thr).astype(int)

    acc  = float(accuracy_score(y_true, y_pred))
    sens = float(tpr[best_idx])
    spec = float(1 - fpr[best_idx])

    metrics = {
        "accuracy":         acc,
        "sensitivity":      sens,
        "specificity":      spec,
        "auc_roc":          auc,
        "mae_threshold":    float(best_thr),
        "mean_mae_norm":    float(norm_errors.mean()),
        "mean_mae_abnormal":float(abnorm_errors.mean()),
        "n_train_norm":     int(len(X_all)),
        "n_test_norm":      int(len(norm_errors)),
        "n_test_abnormal":  int(len(abnorm_errors)),
    }
    _print_metrics(metrics, "Anomaly Classification — strat_fold 10 test")
    _save_metrics(metrics, "anomaly_classify")

    # --- Plots ---
    # MAE distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(norm_errors,   bins=20, alpha=0.7, label="NORM",     color="royalblue")
    axes[0].hist(abnorm_errors, bins=20, alpha=0.7, label="ABNORMAL", color="tomato")
    axes[0].axvline(best_thr, color="black", linestyle="--",
                    label=f"Threshold={best_thr:.4f}")
    axes[0].set_xlabel("Reconstruction MAE")
    axes[0].set_ylabel("Count")
    axes[0].set_title("MAE Distribution: NORM vs ABNORMAL")
    axes[0].legend()

    # ROC curve
    axes[1].plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC (AUC = {auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].scatter(fpr[best_idx], tpr[best_idx], color="red", s=80, zorder=5,
                    label=f"Best threshold (J={j_scores[best_idx]:.3f})")
    axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[1].set_ylabel("True Positive Rate (Sensitivity)")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    fig.tight_layout()
    _save_plot(fig, "anomaly_classify")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RTD Reservoir Computing — local experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    p.add_argument("--mode", choices=["lorenz", "forecast", "classify", "anomaly"],
                   default="lorenz",
                   help="Experiment to run  (anomaly = one-class RC classification)")

    # Dataset paths
    p.add_argument("--ptbxl-path", default="",
                   help="Local path to PTB-XL dataset root folder")
    p.add_argument("--index", type=int, default=0,
                   help="PTB-XL record index (forecast mode)")
    p.add_argument("--lead", default="II",
                   help="ECG lead name, e.g. I, II, V1 … (forecast & classify)")
    p.add_argument("--hr", action="store_true",
                   help="Use high-resolution 500 Hz records (records500/) instead of 100 Hz")
    p.add_argument("--max-records", type=int, default=100,
                   help="Max records to load for classification (per class × 2)")

    # Reservoir hyperparameters
    p.add_argument("--n-units", type=int, default=3,
                   help="Number of parallel RTD units")
    p.add_argument("--reservoir-size", type=int, default=100,
                   help="Virtual nodes per RTD unit")
    p.add_argument("--delay", type=int, default=2,
                   help="Base input delay (steps)")
    p.add_argument("--activation", choices=["rtd", "tanh"], default="rtd",
                   help="Nonlinearity: rtd (Lorentzian NDR) or tanh")
    p.add_argument("--leaky", type=float, default=0.3,
                   help="Leakage rate α")
    p.add_argument("--use-mlp", action="store_true",
                   help="Use MLP readout instead of Ridge Regression")

    # Bayesian optimisation
    p.add_argument("--bayesian", action="store_true",
                   help="Run Bayesian hyperparameter optimisation before training")
    p.add_argument("--bayes-iters", type=int, default=20,
                   help="Number of Bayesian optimisation iterations")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = build_parser().parse_args()

    print("=" * 44)
    print("  RTD Reservoir Computing — ECG Project")
    print(f"  Mode : {args.mode}")
    print(f"  Output dir : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 44)

    if args.mode == "lorenz":
        run_lorenz(args)
    elif args.mode == "forecast":
        run_forecast(args)
    elif args.mode == "classify":
        run_classify(args)
    elif args.mode == "anomaly":
        run_anomaly_classify(args)

    print("\nDone.")
