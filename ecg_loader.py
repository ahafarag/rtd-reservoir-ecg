# ecg_loader.py
import pandas as pd


def load_ecg_data(file):
    df = pd.read_csv(file)
    if 'ecg' not in df.columns:
        df.columns = ['ecg']  # fallback for single column file
    return df
