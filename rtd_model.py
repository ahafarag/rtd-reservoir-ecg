'''
## PATENT NOTICE
This repository contains novel inventions related to multi-RTD reservoir computing for ECG prediction.
Provisional patent application pending. All rights reserved.
Unauthorized commercial use or replication of the multi-RTD architecture (>2 units) is prohibited.
'''

import numpy as np


# ---------------------------------------------------------------------------
# Physically-grounded RTD I-V characteristic
# ---------------------------------------------------------------------------
#
# Model basis: Lorentzian resonant-tunnelling current (Schulman et al. 1996).
#
# The current through a double-barrier RTD is approximated as:
#
#   I(V) = I_p * (V / V_p) / (1 + ((V - V_p) / W)^2)   [peak-tunnelling term]
#          + G_v * V                                       [valley / leakage term]
#
# where:
#   I_p  — peak current magnitude (normalised to 1.0 here)
#   V_p  — peak voltage (position of NDR onset)
#   W    — resonance half-width (controls sharpness of the NDR region)
#   G_v  — valley conductance (controls depth of the valley)
#
# The resulting curve has:
#   • a rising region (positive differential resistance)  for V < V_p
#   • a peak at V ≈ V_p
#   • an NDR (negative differential resistance) region    for V_p < V < V_valley
#   • a recovery region                                   for large V
#
# In the reservoir computing context V is the scaled input/state variable,
# and v_bias shifts the operating point into/out of the NDR region.
# ---------------------------------------------------------------------------


def rtd_iv(v: np.ndarray,
           v_peak: float = 0.5,
           width: float = 0.4,
           g_valley: float = 0.1) -> np.ndarray:
    """Lorentzian RTD I-V characteristic (normalised, dimensionless).

    Args:
        v:        Input voltage (scalar or array).
        v_peak:   Normalised peak voltage V_p  (default 0.5).
        width:    Resonance half-width W        (default 0.4).
        g_valley: Valley conductance G_v        (default 0.1).

    Returns:
        Current I(V) with the same shape as v.
    """
    peak_term   = (v / (v_peak + 1e-9)) / (1.0 + ((v - v_peak) / (width + 1e-9)) ** 2)
    valley_term = g_valley * v
    return peak_term + valley_term


def rtd_nonlinearity(v: np.ndarray,
                     v_bias: float = 0.5,
                     v_peak: float = 0.5,
                     width: float = 0.4,
                     g_valley: float = 0.1) -> np.ndarray:
    """RTD activation function for use in the reservoir state update.

    v_bias shifts the effective operating point:
      - v_bias > 0.5 pushes into the NDR region (more nonlinear)
      - v_bias < 0.5 operates in the linear rising region
      - v_bias = 0.5 sits at the peak (maximum sensitivity)

    The output is centred (zero-mean) to avoid reservoir state drift.
    """
    bias_shift = v_bias - 0.5          # signed shift from neutral
    v_eff = v + bias_shift             # shift operating point
    raw = rtd_iv(v_eff, v_peak=v_peak, width=width, g_valley=g_valley)
    # Centre: subtract the DC offset introduced by the bias shift
    dc_offset = rtd_iv(np.array([bias_shift]), v_peak=v_peak,
                       width=width, g_valley=g_valley)[0]
    return raw - dc_offset
