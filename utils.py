# utils.py
'''
## PATENT NOTICE
This repository contains novel inventions related to multi-RTD reservoir computing for ECG prediction. 
Provisional patent application pending. All rights reserved. 
Unauthorized commercial use or replication of the multi-RTD architecture (>2 units) is prohibited.
'''

from matplotlib import pyplot as plt
import os


def plot_ecg_prediction(true_vals, predicted_vals):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(true_vals[:len(predicted_vals)], label='True ECG', alpha=0.6)
    ax.plot(predicted_vals, label='Predicted ECG', alpha=0.7)
    ax.legend()
    ax.set_title("ECG Signal Prediction")
    return fig

def save_simulation_video(true_vals, predicted_vals, filename="output/simulation.mp4"):
    import matplotlib.animation as animation
    fig, ax = plt.subplots()
    ax.set_xlim(0, len(predicted_vals))
    ax.set_ylim(min(true_vals), max(true_vals))
    line1, = ax.plot([], [], lw=2, label="True")
    line2, = ax.plot([], [], lw=2, label="Predicted")
    ax.legend()

    def update(i):
        line1.set_data(range(i), true_vals[:i])
        line2.set_data(range(i), predicted_vals[:i])
        return line1, line2

    ani = animation.FuncAnimation(fig, update, frames=len(predicted_vals), blit=True)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    ani.save(filename, fps=30, extra_args=["-vcodec", "libx264"])
    return filename