import numpy as np
import matplotlib.pyplot as plt

def plot_log(
    log,
    signals,
    labels=None,
    suptitle=None,
    linewidth=1.2,
    height_per_plot=2.2,
    show=True,
):
    """
    Uniform log expected:
      log["t"] : 1-D time vector (uniform grid)
      log[s]   : 1-D array per signal s in `signals`
    Plots each signal in its own subplot (shared x-axis).
    Safely truncates if lengths differ and drops non-finite samples.
    """
    if "t" not in log:
        raise KeyError("Uniform log must contain a global time vector at log['t'].")

    t = np.ravel(np.asarray(log["t"]))
    if t.size == 0:
        raise ValueError("log['t'] is empty.")

    labels = labels or {}
    n_sig = len(signals)
    if n_sig == 0:
        raise ValueError("No signals provided.")

    fig_h = max(1.8, height_per_plot * n_sig)
    fig, axes = plt.subplots(n_sig, 1, sharex=True, figsize=(8, fig_h))
    if n_sig == 1:
        axes = [axes]  # normalize to list

    notes = []

    for ax, s in zip(axes, signals):
        if s not in log:
            ax.text(0.02, 0.5, f"Signal '{s}' not in log.", transform=ax.transAxes,
                    va="center", ha="left", fontsize=9, alpha=0.8)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_ylabel(labels.get(s, s))
            continue

        y = np.ravel(np.asarray(log[s]))
        if y.size == 0:
            ax.text(0.02, 0.5, f"Signal '{s}' has no samples.", transform=ax.transAxes,
                    va="center", ha="left", fontsize=9, alpha=0.8)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_ylabel(labels.get(s, s))
            continue

        # Safe truncate to common length
        n = min(t.size, y.size)
        if y.size != t.size:
            notes.append(f"'{s}' length differs (t={t.size}, y={y.size}); truncated to {n}.")

        tt = t[:n].astype(float, copy=False)
        yy = y[:n].astype(float, copy=False)

        # Drop non-finite samples (robustness)
        m = np.isfinite(tt) & np.isfinite(yy)
        if not np.any(m):
            ax.text(0.02, 0.5, f"Signal '{s}' has no finite samples.", transform=ax.transAxes,
                    va="center", ha="left", fontsize=9, alpha=0.8)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_ylabel(labels.get(s, s))
            continue

        ax.plot(tt[m], yy[m], linewidth=linewidth)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylabel(labels.get(s, s))

    axes[-1].set_xlabel("Time [s]")

    if suptitle:
        fig.suptitle(suptitle)

    # Surface any truncation notes below the last subplot
    if notes:
        axes[-1].text(0.01, -0.25, "\n".join(notes),
                      transform=axes[-1].transAxes, va="top", ha="left",
                      fontsize=8, alpha=0.75)

    fig.tight_layout(rect=[0, 0, 1, 0.97] if suptitle else None)

    if show:
        plt.show()
    return fig, axes