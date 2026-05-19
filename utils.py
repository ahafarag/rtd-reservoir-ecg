'''
## PATENT NOTICE
This repository contains novel inventions related to multi-RTD reservoir computing for ECG prediction.
Provisional patent application pending. All rights reserved.
Unauthorized commercial use or replication of the multi-RTD architecture (>2 units) is prohibited.
'''

import os
import io
import numpy as np
import joblib
from matplotlib import pyplot as plt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ecg_prediction(true_vals, predicted_vals):
    """Return a matplotlib Figure comparing true and predicted ECG."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(true_vals[:len(predicted_vals)], label="True ECG", alpha=0.6)
    ax.plot(predicted_vals, label="Predicted ECG", alpha=0.7)
    ax.legend()
    ax.set_title("ECG Signal Prediction")
    return fig


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(readout, scaler, path: str = "saved_model.pkl"):
    """Persist the trained readout and its input scaler to disk."""
    joblib.dump({"readout": readout, "scaler": scaler}, path)
    return path


def load_model(path: str = "saved_model.pkl") -> tuple:
    """Load a previously saved readout and scaler.

    Returns:
        (readout, scaler) or (None, None) if the file does not exist.
    """
    if not os.path.exists(path):
        return None, None
    bundle = joblib.load(path)
    return bundle["readout"], bundle["scaler"]


# ---------------------------------------------------------------------------
# Video / GIF export  (no ffmpeg required — uses imageio + Pillow)
# ---------------------------------------------------------------------------

def save_simulation_video(true_vals: np.ndarray,
                          predicted_vals: np.ndarray,
                          filename: str = "output/simulation.gif",
                          fps: int = 30,
                          max_frames: int = 300) -> str:
    """Render true vs. predicted ECG as an animated GIF.

    Uses imageio + Pillow — no external ffmpeg binary required.
    Frames are downsampled to max_frames for manageable file size.

    Args:
        true_vals:       Ground-truth signal array.
        predicted_vals:  Model prediction array.
        filename:        Output path (should end in .gif).
        fps:             Playback speed in frames per second.
        max_frames:      Maximum number of animation frames.

    Returns:
        Path to the saved GIF file.
    """
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for GIF export. "
            "Install it with: pip install imageio[pillow]"
        ) from exc

    n = min(len(predicted_vals), max_frames)
    step = max(1, len(predicted_vals) // max_frames)
    indices = list(range(0, len(predicted_vals), step))[:n]

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    frames = []
    fig, ax = plt.subplots(figsize=(8, 3))
    for i in indices:
        ax.clear()
        ax.plot(true_vals[:i+1], color="royalblue", lw=1.5, label="True")
        ax.plot(predicted_vals[:i+1], color="tomato", lw=1.5,
                linestyle="--", label="Predicted")
        ax.set_xlim(0, len(predicted_vals))
        ax.set_ylim(min(true_vals.min(), predicted_vals.min()) - 0.1,
                    max(true_vals.max(), predicted_vals.max()) + 0.1)
        if i == 0:
            ax.legend(loc="upper right")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()

    plt.close(fig)
    imageio.mimsave(filename, frames, fps=fps)
    return filename
