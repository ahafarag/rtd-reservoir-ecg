# rtd_model.py
import numpy as np


def rtd_nonlinearity(v, a=1.0, b=0.5, c=0.3):
    return a * v - b * v**3 + c * np.exp(-v)