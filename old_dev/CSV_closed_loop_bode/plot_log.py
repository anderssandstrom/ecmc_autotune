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
        #axes[curr_plot].plot(yy,linewidth=2)
        axes[curr_plot].plot(tt, yy,linewidth=2)      

        axes[curr_plot].grid(True, linestyle="--", alpha=0.4)
        axes[curr_plot].set_title(key)
        curr_plot += 1
    plt.show()
    return fig, axes

def plot2(t, y1, y2, segments=None,fs=1000, ax=None):
    t = t.astype(float, copy=False)    
    # Safe truncate to common length
    n = min(t.size, y1.size, y2.size)
    if y1.size != t.size:
        print("length differs (t={t.size}, y={y.size}); truncated to {n}.")
    tt = t[:n].astype(float, copy=False)
    yy1 = y1[:n].astype(float, copy=False)
    yy2 = y2[:n].astype(float, copy=False)
    showFig=0
    # just plot
    if ax is None:
        fig, ax = plt.subplots(1,1)
        showFig=1  # only show if ax is not passed as arg
    ax.plot(tt, yy1,linewidth=2)
    ax.plot(tt, yy2,linewidth=2)

    if segments is not None:
        axy2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
        axy2.set_ylabel('Freq. [Hz]')
        for seg in segments:            
            axy2.plot([float(seg[0])/fs,float(seg[1]/fs)] ,[seg[2],seg[2]] ,'bx--',linewidth=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    if showFig:
        plt.show()
    return
