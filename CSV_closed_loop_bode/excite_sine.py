import numpy as np
import matplotlib.pyplot as plt

def generate(
    fs, f_start, f_stop, n_points, amp,
    n_settle, n_meas,
    transition_min_s=1.0,     # ≥ this many seconds of zero-gap between tones
    transition_frac=0.0,      # or fraction of tone length (0 = ignore)
    edge_taper_cycles=1.0,    # small ramp (cycles) applied INSIDE the settle part only
    return_masks=True
):
    """
    Stepped-sine generator with explicit zero transitions between tones.
    - Tones have FULL amplitude (no long taper).
    - Gaps ('transitions') are zeros, length = max(transition_min_s, transition_frac * tone_len).
    - A tiny edge taper (default 1 cycle) is applied at tone start/end but fits inside the 'settle' cycles.

    Returns:
        t           (N,)
        u           (N,)
        freqs       (n_points,)
        f_array     (N,) frequency per sample, 0 during transitions
        meas_mask   (N,) True on measured cycles of each tone (optional)
        settle_mask (N,) True on settle cycles of each tone (optional)
    """
    freqs = np.logspace(np.log10(f_start), np.log10(f_stop), n_points)

    u_segs, f_segs = [], []
    meas_mask_segs, settle_mask_segs = [], []

    for f in freqs:
        T = 1.0 / f
        n_set = int(round(n_settle * fs * T))   # samples in settle part
        n_mea = int(round(n_meas   * fs * T))   # samples in measured part
        n_tone = n_set + n_mea

        # --- tone (full amplitude) ---
        t_tone = np.arange(n_tone) / fs
        tone = amp * np.sin(2*np.pi*f*t_tone)

        # --- small edge taper confined to settle cycles ---
        n_taper = int(round(edge_taper_cycles * fs * T))
        n_taper = max(0, min(n_taper, n_set//2))  # keep taper within settle only
        if n_taper > 0:
            # fade-in over first n_taper samples
            fade_in = 0.5*(1 - np.cos(np.pi * (np.arange(n_taper)+1) / n_taper))
            tone[:n_taper] *= fade_in
            # fade-out over last n_taper samples (at the end of settle+meas)
            fade_out = fade_in[::-1]
            tone[-n_taper:] *= fade_out

        # frequency labels (nonzero only during tone)
        f_tone = np.full(n_tone, f, dtype=float)

        # masks
        settle_mask = np.zeros(n_tone, dtype=bool)
        meas_mask   = np.zeros(n_tone, dtype=bool)
        settle_mask[:n_set] = True
        meas_mask[n_set:n_set+n_mea] = True

        # add tone
        u_segs.append(tone)
        f_segs.append(f_tone)
        settle_mask_segs.append(settle_mask)
        meas_mask_segs.append(meas_mask)

        # --- transition gap (zeros) ---
        trans_len = int(round(
            max(transition_min_s, transition_frac * (n_tone / fs)) * fs
        ))
        if trans_len > 0:
            u_segs.append(np.zeros(trans_len, dtype=float))
            f_segs.append(np.zeros(trans_len, dtype=float))
            settle_mask_segs.append(np.zeros(trans_len, dtype=bool))
            meas_mask_segs.append(np.zeros(trans_len, dtype=bool))

    # concatenate
    u = np.concatenate(u_segs)
    f_array = np.concatenate(f_segs)
    settle_mask = np.concatenate(settle_mask_segs)
    meas_mask = np.concatenate(meas_mask_segs)
    t = np.arange(len(u)) / fs

    # preview
    plt.figure(figsize=(10,4))
    plt.plot(t, u, label="excitation")
    # show freq (scaled) just for visualization
    if np.max(freqs) > 0:
        plt.plot(t, (f_array/np.max(freqs))*amp, 'r--', label="freq (scaled)")
    plt.xlabel("Time [s]"); plt.ylabel("Amplitude / scaled freq")
    plt.title("Stepped-sine with explicit zero transitions")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    if return_masks:
        return t, u, freqs, f_array, meas_mask, settle_mask
    else:
        return t, u, freqs, f_array

# =========================
# Example usage
# =========================
if __name__ == "__main__":    
    # --- Parameters ---
    fs = 1000.0             # Sampling frequency [Hz]
    f_start, f_stop = 1, 200
    n_points = 40           # Number of frequencies (log spaced)
    amp = 30.0              # Command amplitude (+/-)
    n_settle = 10           # Settle cycles before measuring
    n_meas = 20             # Measured cycles per frequency
    taper = 0.05            # fraction of cosine fade-in/out per step
    t, u, freqs = generate(fs,f_start, f_stop,amp, n_settle, n_meas, taper)

