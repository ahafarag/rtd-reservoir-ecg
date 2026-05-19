'''
## PATENT NOTICE
This repository contains novel inventions related to multi-RTD reservoir computing for ECG prediction.
Provisional patent application pending. All rights reserved.
Unauthorized commercial use or replication of the multi-RTD architecture (>2 units) is prohibited.
'''

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
import wfdb
import os


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def bandpass_filter(signal: np.ndarray, fs: float = 500.0,
                    low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """4th-order Butterworth bandpass filter (0.5–40 Hz default).

    Zero-phase (filtfilt) to avoid phase distortion. Matches the preprocessing
    described in the thesis methodology and index.html.
    """
    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def normalise(signal: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation with division-by-zero guard."""
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


# ---------------------------------------------------------------------------
# CSV loader (original interface — backward compatible)
# ---------------------------------------------------------------------------

def load_ecg_data(file, apply_filter: bool = True,
                  fs: float = 500.0) -> pd.DataFrame:
    """Load an ECG CSV file.

    Expects a single column or a column named 'ecg'.
    Applies bandpass filter when apply_filter=True (default).
    """
    if isinstance(file, pd.DataFrame):
        df = file.copy()
    else:
        df = pd.read_csv(file)

    if "ecg" not in df.columns:
        df.columns = ["ecg"]

    if apply_filter and len(df) > 20:
        df["ecg"] = bandpass_filter(df["ecg"].values, fs=fs)

    return df


# ---------------------------------------------------------------------------
# PTB-XL loader
# ---------------------------------------------------------------------------

def load_ptbxl_record(ptbxl_root: str, index: int,
                      lead: str = "I",
                      apply_filter: bool = True) -> pd.DataFrame:
    """Load one PTB-XL record by its database index.

    Args:
        ptbxl_root:   Local path to the PTB-XL dataset root directory.
        index:        Row index in ptbxl_database.csv (0-based).
        lead:         Lead name, e.g. 'I', 'II', 'V1'.
        apply_filter: Apply bandpass filter (0.5-40 Hz) after loading.

    Returns:
        DataFrame with column 'ecg'. attrs['fs'] carries the sampling frequency.
    """
    meta_path = os.path.join(ptbxl_root, "ptbxl_database.csv")
    meta = pd.read_csv(meta_path)
    record_path = os.path.join(ptbxl_root, meta.loc[index, "filename_lr"])
    record = wfdb.rdrecord(record_path)

    lead_names = record.sig_name
    if lead not in lead_names:
        raise ValueError(f"Lead '{lead}' not found. Available: {lead_names}")

    signal = record.p_signal[:, lead_names.index(lead)]
    fs = float(record.fs)

    if apply_filter:
        signal = bandpass_filter(signal, fs=fs)

    df = pd.DataFrame({"ecg": signal})
    df.attrs["fs"] = fs
    df.attrs["lead"] = lead
    return df


def load_ptbxl_metadata(ptbxl_root: str) -> pd.DataFrame:
    """Return the full PTB-XL metadata CSV (includes strat_fold for splits)."""
    return pd.read_csv(os.path.join(ptbxl_root, "ptbxl_database.csv"))


# ---------------------------------------------------------------------------
# MIT-BIH loader
# ---------------------------------------------------------------------------

def load_mitbih_record(mitbih_root: str, record_name: str,
                       channel: int = 0,
                       target_fs: float = 500.0,
                       max_samples: int = 3000,
                       apply_filter: bool = True) -> pd.DataFrame:
    """Load one MIT-BIH record, resample to target_fs, and trim to max_samples.

    Args:
        mitbih_root:  Local path to the MIT-BIH dataset directory.
        record_name:  Record identifier string, e.g. '100', '101'.
        channel:      Signal channel index (0 = first channel).
        target_fs:    Target sampling frequency in Hz (default 500).
        max_samples:  Truncate to this many samples after resampling.
        apply_filter: Apply bandpass filter (0.5-40 Hz) after resampling.

    Returns:
        DataFrame with column 'ecg'. attrs['fs'] = target_fs.
    """
    record = wfdb.rdrecord(os.path.join(mitbih_root, record_name))
    ecg_raw = record.p_signal[:, channel]
    ecg_resampled = resample(ecg_raw,
                             int(len(ecg_raw) * target_fs / record.fs))

    if apply_filter:
        ecg_resampled = bandpass_filter(ecg_resampled, fs=target_fs)

    df = pd.DataFrame({"ecg": ecg_resampled[:max_samples]})
    df.attrs["fs"] = target_fs
    df.attrs["record"] = record_name
    return df


def list_mitbih_records(mitbih_root: str) -> list:
    """Return sorted list of available MIT-BIH record names in a directory."""
    return sorted({f.split(".")[0]
                   for f in os.listdir(mitbih_root) if f.endswith(".dat")})
