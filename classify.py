'''
## PATENT NOTICE
This repository contains novel inventions related to multi-RTD reservoir computing for ECG prediction.
Provisional patent application pending. All rights reserved.
Unauthorized commercial use or replication of the multi-RTD architecture (>2 units) is prohibited.
'''

import streamlit as st
import numpy as np
import pandas as pd
import ast
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

from ecg_loader import load_ptbxl_record, load_ptbxl_metadata, bandpass_filter
from multi_rtd_reservoir import MultiRTDReservoir

st.set_page_config(page_title="RTD Arrhythmia Classification", layout="wide")
st.title("Arrhythmia Classification — RTD Reservoir + PTB-XL")

st.markdown("""
This page implements **true binary arrhythmia classification** using the
PTB-XL diagnostic labels (`scp_codes`).

- **Train set**: PTB-XL `strat_fold` 1–9
- **Test set**:  PTB-XL `strat_fold` 10  (patient-independent holdout)
- **Label**:     NORM (normal sinus rhythm) vs. any abnormal condition
- **Classifier**: RTD reservoir states → Ridge or MLP readout
""")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
ptbxl_root = st.sidebar.text_input(
    "PTB-XL dataset path",
    "./ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
lead = st.sidebar.selectbox("ECG Lead", ["I","II","III","aVR","aVL","aVF",
                                          "V1","V2","V3","V4","V5","V6"], index=1)
n_units       = st.sidebar.slider("RTD units (N)", 1, 6, 3)
reservoir_size = st.sidebar.slider("Virtual nodes per unit", 20, 200, 50)
base_delay    = st.sidebar.slider("Base delay", 1, 20, 2)
activation    = st.sidebar.selectbox("Activation", ["rtd", "tanh"])
max_records   = st.sidebar.slider("Max records to load (speed vs. accuracy)",
                                   50, 500, 100, step=50)
classifier_type = st.sidebar.radio("Classifier", ["Ridge", "MLP"])

# ---------------------------------------------------------------------------
# Label extraction helpers
# ---------------------------------------------------------------------------

def _parse_scp(scp_str: str) -> dict:
    """Parse the scp_codes string column into a dict."""
    try:
        return ast.literal_eval(scp_str)
    except Exception:
        return {}

def _is_normal(scp_dict: dict) -> int:
    """Return 1 if record is NORM, 0 if any abnormality is present."""
    return int("NORM" in scp_dict and scp_dict["NORM"] > 0)

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
if not os.path.exists(ptbxl_root):
    st.warning("PTB-XL folder not found. Enter the correct path in the sidebar.")
    st.stop()

if st.button("Run Classification"):
    meta = load_ptbxl_metadata(ptbxl_root)

    # Encode labels
    meta["label"] = meta["scp_codes"].apply(_parse_scp).apply(_is_normal)

    # Train / test split using predefined strat_fold (patient-independent)
    train_meta = meta[meta["strat_fold"] <= 9].reset_index(drop=True)
    test_meta  = meta[meta["strat_fold"] == 10].reset_index(drop=True)

    # Subsample for speed (balanced: equal number of NORM / abnormal)
    n_per_class = max_records // 2
    train_meta = pd.concat([
        train_meta[train_meta["label"] == 1].head(n_per_class),
        train_meta[train_meta["label"] == 0].head(n_per_class),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    test_meta = pd.concat([
        test_meta[test_meta["label"] == 1].head(n_per_class // 5),
        test_meta[test_meta["label"] == 0].head(n_per_class // 5),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    st.info(f"Training on {len(train_meta)} records, "
            f"testing on {len(test_meta)} records.")

    # Build reservoir ensemble (weights shared across records)
    reservoir = MultiRTDReservoir(
        n_units=n_units, size=reservoir_size, base_delay=base_delay,
        activation=activation, use_mlp=False,
    )

    def _extract_features(subset_meta: pd.DataFrame,
                          label: str) -> tuple[np.ndarray, np.ndarray]:
        """Load records, run through reservoir, return (X_features, y_labels)."""
        features, labels = [], []
        bar = st.progress(0.0, text=f"Loading {label} records…")
        n = len(subset_meta)
        for i, (_, row) in enumerate(subset_meta.iterrows()):
            try:
                df = load_ptbxl_record(ptbxl_root,
                                       index=int(row.name),
                                       lead=lead,
                                       apply_filter=True)
                signal = df["ecg"].values
                signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

                # Each record → one feature vector (mean of reservoir states)
                X_r, _ = reservoir.generate_states(signal)
                if X_r.shape[0] == 0:
                    continue
                features.append(X_r.mean(axis=0))   # shape: (n_units * size,)
                labels.append(int(row["label"]))
            except Exception:
                pass  # skip unreadable records silently
            bar.progress((i + 1) / n, text=f"Loading {label} records… {i+1}/{n}")
        bar.empty()
        return np.array(features), np.array(labels)

    X_train, y_train = _extract_features(train_meta, "train")
    X_test,  y_test  = _extract_features(test_meta,  "test")

    if len(X_train) == 0 or len(X_test) == 0:
        st.error("Could not extract features. Check dataset path and lead name.")
        st.stop()

    # Standardise features (important for Ridge and MLP)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train classifier
    if classifier_type == "MLP":
        clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500,
                            random_state=42)
    else:
        clf = RidgeClassifier(alpha=1.0)

    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------
    acc  = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=["Abnormal", "Normal"],
                                   output_dict=True)

    st.subheader("Classification Results (patient-independent test set)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy",    f"{acc * 100:.2f} %")
    col2.metric("Sensitivity", f"{report['Normal']['recall'] * 100:.2f} %")
    col3.metric("Specificity", f"{report['Abnormal']['recall'] * 100:.2f} %")

    st.markdown("#### Per-class report")
    st.dataframe(pd.DataFrame(report).T.round(3))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Abnormal", "Normal"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set (strat_fold 10)")
    st.pyplot(fig)

    # Class distribution
    st.markdown("#### Label distribution in test set")
    counts = pd.Series(y_test).value_counts().rename({0: "Abnormal", 1: "Normal"})
    st.bar_chart(counts)
