import numpy as np
import matplotlib.pyplot as plt

def plot(t, y_data):
    t = t.astype(float, copy=False)
    height_per_plot = 2.2
    n_sig = len(y_data)
    fig_h = max(1.8, height_per_plot * n_sig)
    fig, axes = plt.subplots(n_sig, 1, sharex=True, figsize=(8, fig_h))    
    curr_plot = 0
    for key, y in y_data.items():
        print(y)
        # Safe truncate to common length
        n = min(t.size, y.size)
        if y.size != t.size:
            print("length differs (t={t.size}, y={y.size}); truncated to {n}.")

        tt = t[:n].astype(float, copy=False)
        yy = y[:n].astype(float, copy=False)

        #ax = plt.subplot(n_sig, 1 , curr_plot + 1)
        axes[curr_plot].plot(tt, yy,linewidth=2)        
        axes[curr_plot].grid(True, linestyle="--", alpha=0.4)
        axes[curr_plot].set_title(key)
        curr_plot += 1
    plt.show()
    return fig, axes
