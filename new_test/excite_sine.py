import numpy as np
import matplotlib.pyplot as plt


def generate(fs,f_start, f_stop,n_points, amp, n_settle, n_meas,taper):
    # --- Generate frequency list (log spaced) ---
    freqs = np.logspace(np.log10(f_start), np.log10(f_stop), n_points)
    
    # --- Generate stepped-sine signal ---
    segments = []
    for f in freqs:
        T = 1.0 / f
        t_total = (n_settle + n_meas) * T
        t = np.arange(0, t_total, 1/fs)
    
        # fade-in/out window (cosine taper)
        win = np.ones_like(t)
        n_taper = int(np.floor(taper * len(t)))
        if n_taper > 0:
            fade = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))
            win[:n_taper] = fade
            win[-n_taper:] = fade[::-1]
    
        segment = amp * win * np.sin(2 * np.pi * f * t)
        segments.append(segment)
    
    # Concatenate all frequencies
    u = np.concatenate(segments)
    t = np.arange(len(u)) / fs
    
    print(f"Generated stepped-sine with {len(freqs)} frequencies, total length {len(u)/fs:.1f} s")
    
    # --- Plot preview ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, u)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Stepped-sine Excitation Sequence")
    plt.grid(True)
    plt.show()
    
    # --- Save to file ---
    #np.savez("stepped_sine_excitation.npz", t=t, u=u, freqs=freqs)

    #print("Saved to 'stepped_sine_excitation.npz'")
    return t, u, freqs

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

