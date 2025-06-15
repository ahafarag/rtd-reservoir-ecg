# rtd_model.py
'''
## PATENT NOTICE
This repository contains novel inventions related to multi-RTD reservoir computing for ECG prediction. 
Provisional patent application pending. All rights reserved. 
Unauthorized commercial use or replication of the multi-RTD architecture (>2 units) is prohibited.
'''
import numpy as np

def rtd_nonlinearity(v, a=1.0, b=0.5, c=0.3):
    return a * v - b * v**3 + c * np.exp(-v)