# ecg_loader.py
'''
## PATENT NOTICE
This repository contains novel inventions related to multi-RTD reservoir computing for ECG prediction. 
Provisional patent application pending. All rights reserved. 
Unauthorized commercial use or replication of the multi-RTD architecture (>2 units) is prohibited.
'''

import pandas as pd


def load_ecg_data(file):
    df = pd.read_csv(file)
    if 'ecg' not in df.columns:
        df.columns = ['ecg']  # fallback for single column file
    return df