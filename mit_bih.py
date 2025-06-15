'''
## PATENT NOTICE
This repository contains novel inventions related to multi-RTD reservoir computing for ECG prediction. 
Provisional patent application pending. All rights reserved. 
Unauthorized commercial use or replication of the multi-RTD architecture (>2 units) is prohibited.
'''

import wfdb
import numpy as np
import pandas as pd
from scipy.signal import resample
import streamlit as st

st.subheader("MIT-BIH Online Loader (via PhysioNet API)")
online_records = ['100', '101', '102', '103', '104', '105', '106', '107', '108']
selected_record = st.selectbox("Select MIT-BIH Record", online_records, index=0)

try:
    # Use wfdb's built-in PhysioNet remote access
    record = wfdb.rdrecord(record_name=selected_record, pn_dir="mitdb")
    ecg_raw = record.p_signal[:, 0]
    ecg_resampled = resample(ecg_raw, int(len(ecg_raw) * 500 / record.fs))
    df = pd.DataFrame({'ecg': ecg_resampled})
    st.line_chart(df['ecg'].values[:500])
    ecq_file = df
    st.success(f"Loaded online record: {selected_record} from PhysioNet")
except Exception as e:
    st.error(f"Failed to load online record {selected_record}: {e}")
