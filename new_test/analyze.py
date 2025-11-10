import numpy as np
import matplotlib.pyplot as plt

# =========================
# User settings / constants
# =========================
fs = 1000.0          # sampling rate [Hz]
tau_ms = 2.5         # (optional) known delay to compensate in phase (ms)
block_len_s = 0.50   # analysis block for frequency detection (seconds)
overlap = 0.50       # block overlap fraction [0..0.9]
fmin, fmax = 0.5, 500.0     # search band for dominant frequency [Hz]
freq_tolerance = 0.02       # allow ±2% change within the same tone
settle_frac = 0.30          # discard first 30% of each detected tone as settle
r2_min = 0.95               # reject fits with R^2 below this

# If you have a planned frequency list (optional), set it here for comparison:
planned_freqs = None  # e.g., np.logspace(np.log10(1), np.log10(200), 40)

# =========================
# Load/prepare your signals
# =========================
# Replace these with your data loading:
# Example: npz recorded by you that contains t, u, y
# data = np.load("your_recording.npz")
# t = data["t"].astype(float); u = data["u"].astype(float); y = data["y"].astype(float)

# For demo purposes, I assume you've already loaded:
# t, u, y = ...

# -------------- helpers --------------

def hann(n):
    # Periodic Hann (good for FFT of one window)
    return 0.5 - 0.5 * np.cos(2*np.pi*np.arange(n)/n)

def estimate_delay_samples(u, y, fs, max_ms=10.0, probe_s=3.0):
    """Estimate constant delay (y lags u) using normalized cross-correlation on a mid slice."""
    N = len(u)
    if N < int(fs*1.0):
        raise ValueError("Signals too short for delay estimation")
    i0 = max(0, N//2 - int(probe_s*fs/2))
    i1 = min(N, i0 + int(probe_s*fs))
    x = u[i0:i1].astype(float) - np.mean(u[i0:i1])
    z = y[i0:i1].astype(float) - np.mean(y[i0:i1])
    K = int(round(max_ms*1e-3*fs))
    lags = np.arange(-K, K+1)
    best = 0.0; best_k = 0
    for k in lags:
        if k >= 0:
            xa, za = x[:-k or None], z[k:]
        else:
            xa, za = x[-k:], z[:k or None]
        num = float(np.dot(xa - xa.mean(), za - za.mean()))
        den = float(np.linalg.norm(xa - xa.mean()) * np.linalg.norm(za - za.mean()))
        c = 0.0 if den == 0 else num/den
        if c > best:
            best = c; best_k = k
    return best_k  # samples (positive if y lags u)

def dominant_freq(block, fs, fmin, fmax):
    """Return dominant frequency in a block (Hz), ignoring DC and out-of-band."""
    n = len(block)
    w = hann(n)
    X = np.fft.rfft((block - block.mean()) * w)
    f = np.fft.rfftfreq(n, d=1.0/fs)
    # mask band and ignore DC
    band = (f >= max(fmin, f[1])) & (f <= fmax)
    if not np.any(band):
        return np.nan
    mag = np.abs(X)[band]
    if mag.size == 0 or np.all(mag == 0):
        return np.nan
    k = np.argmax(mag)
    return float(f[band][k])

def segment_by_frequency(u, fs, block_len_s=0.5, overlap=0.5, fmin=0.5, fmax=500.0, tol=0.02):
    """Scan the record, track dominant frequency per block, group contiguous runs with stable freq."""
    n = len(u)
    blen = int(round(block_len_s * fs))
    hop = max(1, int(round(blen * (1.0 - overlap))))
    starts = np.arange(0, max(1, n - blen + 1), hop, dtype=int)
    freqs = []
    for s in starts:
        f = dominant_freq(u[s:s+blen], fs, fmin, fmax)
        freqs.append(f)
    freqs = np.array(freqs)

    segments = []
    i = 0
    while i < len(starts):
        f0 = freqs[i]
        if not np.isfinite(f0):
            i += 1
            continue
        # grow while frequency stays within tolerance
        j = i + 1
        while j < len(starts):
            f1 = freqs[j]
            if not np.isfinite(f1):
                break
            if abs(f1 - f0) / max(f0, 1e-9) <= tol:
                j += 1
            else:
                break
        # segment sample indices (span from start of i to end of j-1 block)
        s0 = starts[i]
        s1 = min(starts[j-1] + blen, n)
        f_est = np.nanmedian(freqs[i:j])
        segments.append((s0, s1, f_est))
        i = j
    return segments

def sine_fit_complex(tseg, vseg, f):
    """
    Fit v(t) ~ a*sin(2πft) + b*cos(2πft) + c
    Return complex phasor V = b - j*a (cosine reference), and R^2.
    """
    w = 2*np.pi*f
    s = np.sin(w*tseg)
    c = np.cos(w*tseg)
    X = np.column_stack([s, c, np.ones_like(tseg)])
    beta, *_ = np.linalg.lstsq(X, vseg, rcond=None)
    a, b, _ = beta
    V = b - 1j*a
    vhat = X @ beta
    ss_res = np.sum((vseg - vhat)**2)
    ss_tot = np.sum((vseg - np.mean(vseg))**2) + 1e-12
    R2 = 1.0 - ss_res/ss_tot
    return V, R2

def frf_from_segments(t, u, y, fs, segments, settle_frac=0.3, r2_min=0.95, delay_samp=0):
    H, F, R2u, R2y = [], [], [], []
    # Delay-comp y to align with u (advance y by delay_samp)
    if delay_samp > 0:
        y_al = np.concatenate([y[delay_samp:], np.zeros(delay_samp)])
    elif delay_samp < 0:
        y_al = np.concatenate([np.zeros(-delay_samp), y[:len(y)+delay_samp]])
    else:
        y_al = y.copy()

    for (s0, s1, f) in segments:
        if not np.isfinite(f) or s1 - s0 < int(5 * fs / max(f, 1e-6)):  # need a few cycles
            continue
        # settle/meas split
        n = s1 - s0
        s_settle = s0 + int(round(settle_frac * n))
        # measurement slice
        ti = t[s_settle:s1]
        ui = u[s_settle:s1] - np.mean(u[s_settle:s1])
        yi = y_al[s_settle:s1] - np.mean(y_al[s_settle:s1])
        if len(ti) < int(3 * fs / max(f,1e-6)):
            continue
        U, r2u = sine_fit_complex(ti, ui, f)
        Y, r2y = sine_fit_complex(ti, yi, f)
        if np.abs(U) < 1e-12 or (r2u < r2_min) or (r2y < r2_min):
            continue
        H.append(Y / U)
        F.append(f)
        R2u.append(r2u)
        R2y.append(r2y)
    return np.array(F), np.array(H), np.array(R2u), np.array(R2y)

# ========================
# Run the robust pipeline
# ========================
def analyze_stepped_sine(t, u, y, fs, tau_ms=0.0,
                         block_len_s=0.5, overlap=0.5, fmin=0.5, fmax=500.0,
                         freq_tolerance=0.02, settle_frac=0.3, r2_min=0.95):
    # 1) Estimate constant I/O delay (once)
    delay_samp = estimate_delay_samples(u, y, fs, max_ms=10.0, probe_s=3.0)
    delay_ms = 1000.0 * delay_samp / fs

    # 2) Detect tone segments (drift-resistant)
    segments = segment_by_frequency(u, fs, block_len_s, overlap, fmin, fmax, freq_tolerance)

    # 3) Compute FRF per detected tone
    F, H, R2u, R2y = frf_from_segments(t, u, y, fs, segments, settle_frac, r2_min, delay_samp)

    # 4) Phase delay compensation (your known τ plus the estimated discrete delay)
    tau = max(0.0, tau_ms) * 1e-3
    Hc = H * np.exp(1j * 2*np.pi * F * tau)

    # 5) Magnitude & phase
    mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-16))
    phase_deg = np.degrees(np.unwrap(np.angle(H)))
    phase_c_deg = np.degrees(np.unwrap(np.angle(Hc)))

    results = {
        "F": F, "H": H, "mag_db": mag_db,
        "phase_deg": phase_deg, "phase_comp_deg": phase_c_deg,
        "delay_samples": delay_samp, "delay_ms_est": delay_ms,
        "R2_u": R2u, "R2_y": R2y, "segments": segments
    }
    return results

# ======================
# Example call & plotting
# ======================
# results = analyze_stepped_sine(t, u, y, fs,
#                                tau_ms=tau_ms,
#                                block_len_s=block_len_s, overlap=overlap,
#                                fmin=fmin, fmax=fmax,
#                                freq_tolerance=freq_tolerance,
#                                settle_frac=settle_frac, r2_min=r2_min)

# F = results["F"]; mag_db = results["mag_db"]
# phase = results["phase_deg"]; phase_c = results["phase_comp_deg"]

# # Quality mask
# mask = (results["R2_u"] > r2_min) & (results["R2_y"] > r2_min)
# Fm, Mm, Ph, Pc = F[mask], mag_db[mask], phase[mask], phase_c[mask]

# # Plot
# plt.figure(figsize=(9,5))
# plt.semilogx(Fm, Mm, marker='o')
# plt.grid(True, which='both')
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude (dB)")
# plt.title("Bode Magnitude (Stepped-sine, drift-robust)")
# plt.tight_layout()

# plt.figure(figsize=(9,5))
# plt.semilogx(Fm, Ph, marker='o', label="Raw phase")
# plt.semilogx(Fm, Pc, marker='o', label=f"Phase (+{tau_ms:.2f} ms)")
# plt.grid(True, which='both')
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Phase (deg)")
# plt.title("Bode Phase (Stepped-sine)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# print(f"Estimated I/O delay from data: {results['delay_ms_est']:.2f} ms  (integer-sample)")
# print("Median R^2 (input, output):", np.median(results["R2_u"][mask]), np.median(results["R2_y"][mask]))
