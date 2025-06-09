import os
import pandas as pd
import wfdb


def load_ptbxl_sample(ptbxl_root: str, index: int = 0, lead: str = "I") -> pd.DataFrame:
    """Load a single sample from the PTB-XL dataset."""
    meta = pd.read_csv(os.path.join(ptbxl_root, "ptbxl_database.csv"))
    record_path = os.path.join(ptbxl_root, meta.loc[index, "filename_lr"])
    record = wfdb.rdrecord(record_path)
    lead_names = record.sig_name
    lead_index = lead_names.index(lead)
    signal = record.p_signal[:, lead_index]
    return pd.DataFrame({"ecg": signal})


def load_mitbih_sample(mitbih_root: str, record_name: str = "100", lead_index: int = 0) -> pd.DataFrame:
    """Load a sample from the MIT-BIH Arrhythmia Database."""
    record_path = os.path.join(mitbih_root, record_name)
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, lead_index]
    return pd.DataFrame({"ecg": signal})
