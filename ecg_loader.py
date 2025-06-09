# ecg_loader.py
import pandas as pd
import wfdb


def load_ecg_data(file):
    df = pd.read_csv(file)
    if 'ecg' not in df.columns:
        df.columns = ['ecg']  # fallback for single column file
    return df


def load_mitbih_record(record_id="100", lead=0):
    """Download a record from the MIT-BIH arrhythmia dataset via PhysioNet."""
    record = wfdb.rdrecord(record_id, pn_dir="mitdb")
    signal = record.p_signal[:, lead]
    return pd.DataFrame({"ecg": signal})
