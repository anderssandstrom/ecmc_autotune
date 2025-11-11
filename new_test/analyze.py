import numpy as np
import matplotlib.pyplot as plt

# =========================
# User settings / constants
# =========================

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
class bode:
    def __init__(self,t, u, y, fs, tau_ms=2.5,
                 block_len_s=0.5, overlap=0.5, fmin=0.5, fmax=500.0,
                 freq_tolerance=0.02, settle_frac=0.3, r2_min=0.95):
        self.t = t
        self.u = u
        self.y = y 
        self.fs = fs                          # sampling rate [Hz]
        self.tau_ms = 2.5                     # (optional) known delay to compensate in phase (ms)
        self.block_len_s = block_len_s        # analysis block for frequency detection (seconds)
        self.overlap = overlap                # block overlap fraction [0..0.9]
        self.fmin, self.fmax = fmin, fmax     # search band for dominant frequency [Hz]
        self.freq_tolerance = freq_tolerance  # allow ±2% change within the same tone
        self.settle_frac = settle_frac        # discard first 30% of each detected tone as settle
        self.r2_min = r2_min                  # reject fits with R^2 below this

    def hann(self, n):
        # Periodic Hann (good for FFT of one window)
        return 0.5 - 0.5 * np.cos(2*np.pi*np.arange(n)/n)
    
    def estimate_delay_samples(self, u, y, fs, max_ms=10.0, probe_s=3.0):
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
    
    def dominant_freq(self,block, fs, fmin, fmax):
        """Return dominant frequency in a block (Hz), ignoring DC and out-of-band."""
        n = len(block)
        w = self.hann(n)
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
    
    def segment_by_frequency(self, u, fs, block_len_s=0.5, overlap=0.5, fmin=0.5, fmax=500.0, tol=0.02):
        """Scan the record, track dominant frequency per block, group contiguous runs with stable freq."""
        n = len(u)
        blen = int(round(block_len_s * fs))
        hop = max(1, int(round(blen * (1.0 - overlap))))
        starts = np.arange(0, max(1, n - blen + 1), hop, dtype=int)
        freqs = []
        for s in starts:
            f = self.dominant_freq(u[s:s+blen], fs, fmin, fmax)
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
    
    def sine_fit_complex(self,tseg, vseg, f):
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

    def _taper(self,n, frac=0.05):
        """Symmetric cosine taper of length n with 'frac' on each end."""
        m = max(1, int(round(frac*n)))
        w = np.ones(n, float)
        if m > 0:
            k = np.arange(m)
            ramp = 0.5*(1 - np.cos(np.pi*(k+1)/m))
            w[:m] = ramp
            w[-m:] = ramp[::-1]
        return w
    
    def _lockin_phasor(self,t, v, f):
        """
        Complex lock-in estimator at frequency f:
        V = <v(t)*e^{-jωt}> / <e^{-jωt}*e^{jωt}>
        Equivalent (up to scaling) to the LS sin/cos fit but more numerically stable.
        Also returns R^2 of a 3-term LS fit [sin, cos, 1].
        """
        w = 2*np.pi*f
        # LS for R^2 (with DC):
        s = np.sin(w*t); c = np.cos(w*t)
        X = np.column_stack([s, c, np.ones_like(t)])
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        vhat = X @ beta
        ss_res = np.sum((v - vhat)**2)
        ss_tot = np.sum((v - np.mean(v))**2) + 1e-12
        R2 = 1.0 - ss_res/ss_tot
    
        # Complex phasor by lock-in (mean = projection):
        ej = np.exp(-1j*w*t)
        V = np.vdot(ej, v) / np.vdot(ej, ej)  # vdot does conj on first arg
        return V, R2
    
    def _refine_freq_lockin(self,t, v, f0, pct=0.08, ngrid=31):
        """Scan a small grid around f0 and pick the f with the highest input R^2."""
        if f0 <= 0: return f0
        fmin, fmax = f0*(1-pct), f0*(1+pct)
        grid = np.linspace(fmin, fmax, ngrid)
        best = ( -np.inf, f0 )
        for f in grid:
            _, R2 = self._lockin_phasor(t, v, f)
            if R2 > best[0]:
                best = (R2, f)
        return best[1]
    
    def _periodogram_peak_freq(self,t, x, fs, fmin=0.5, fmax=500.0, oversample=8):
        # Assume ~uniform fs; use FFT on tapered slice for speed/robustness
        N = len(x)
        if N < 64:
            return np.nan
        w = np.hanning(N)
        X = np.fft.rfft((x - x.mean()) * w, n=int(2**np.ceil(np.log2(N))*oversample))
        f = np.fft.rfftfreq(len(X), d=1.0/fs)
        band = (f >= fmin) & (f <= fmax)
        if not np.any(band):
            return np.nan
        P = np.abs(X)**2
        # ignore DC bin
        band_idx = np.where(band)[0]
        band_idx = band_idx[band_idx > 0]
        if band_idx.size == 0:
            return np.nan
        k = band_idx[np.argmax(P[band_idx])]
        # quadratic sub-bin interpolation
        fhat = self._quad_peak(f, P, k) if 0 < k < len(P)-1 else f[k]
        return float(fhat)

    def _taper(self,n, frac=0.05):
        m = max(1, int(round(frac*n)))
        w = np.ones(n, float)
        if m > 0:
            k = np.arange(m)
            ramp = 0.5*(1 - np.cos(np.pi*(k+1)/m))
            w[:m] = ramp
            w[-m:] = ramp[::-1]
        return w
    
    def _quad_peak(self,f, P, k):
        # Quadratic interpolation around bin k (1 <= k < len(P)-1)
        k0 = np.clip(k, 1, len(P)-2)
        y1, y2, y3 = P[k0-1], P[k0], P[k0+1]
        denom = (y1 - 2*y2 + y3)
        if denom == 0:
            return f[k0]
        delta = 0.5*(y1 - y3)/denom
        return f[k0] + delta*(f[k0+1]-f[k0])
    
    def frf_from_segments_autofreq(self,
        t, u, y, fs, segments,
        settle_frac=0.20,
        min_meas_cycles=4.0, min_meas_points=400,
        r2_min=0.90,
        delay_samp=0,
        f_search_min=0.5, f_search_max=200.0,
        oversample=8,
        taper_frac=0.05,
        verbose=True
    ):
        # Global integer alignment (drop, no wrap)
        tt, uu, yy = t.copy(), u.copy(), y.copy()
        delay_samp=int(delay_samp)
        if delay_samp > 0:
            if delay_samp >= len(yy): raise ValueError("delay_samp too large")
            yy = yy[delay_samp:]; uu = uu[:-delay_samp]; tt = tt[:-delay_samp]
        elif delay_samp < 0:
            k = -delay_samp
            if k >= len(uu): raise ValueError("delay_samp too negative")
            uu = uu[k:]; tt = tt[k:]; yy = yy[:-k]
        N = min(len(tt), len(uu), len(yy))
        tt, uu, yy = tt[:N], uu[:N], yy[:N]
    
        F, H, R2u, R2y = [], [], [], []
        reasons = []
    
        for i, seg in enumerate(segments):
            try:
                s0, s1, f0 = int(seg[0]), int(seg[1]), float(seg[2])
            except Exception:
                reasons.append((i, seg, "bad segment tuple")); continue
    
            s0 = max(0, min(s0, N-1)); s1 = max(0, min(s1, N))
            if s1 - s0 < 10:
                reasons.append((i,(s0,s1,f0),"segment too short")); continue
    
            n = s1 - s0
            s_set = s0 + int(round(settle_frac*n))
            if s_set >= s1:
                reasons.append((i,(s0,s1,f0),"no meas after settle")); continue
    
            ti = tt[s_set:s1]
            ui = uu[s_set:s1].astype(float)
            yi = yy[s_set:s1].astype(float)
    
            # zero-mean and shift time origin
            ti = ti - ti[0]
            ui = ui - np.mean(ui)
            yi = yi - np.mean(yi)
    
            # basic length checks
            meas_dur = len(ti)/fs
            # use nominal f0 for cycles check, but ensure >= 1 Hz to avoid divide issues
            f_chk = max(f0, 1.0)
            meas_cyc = meas_dur * f_chk
            if meas_cyc < min_meas_cycles:
                reasons.append((i,(s0,s1,f0), f"few cycles ({meas_cyc:.2f}<{min_meas_cycles})")); continue
            if len(ti) < min_meas_points:
                reasons.append((i,(s0,s1,f0), f"few points ({len(ti)}<{min_meas_points})")); continue
    
            # detect the true tone from the INPUT slice
            fhat = self._periodogram_peak_freq(ti, ui, fs, f_search_min, f_search_max, oversample=oversample)
            if not np.isfinite(fhat):
                reasons.append((i,(s0,s1,f0), "no peak freq detected")); continue
    
            # taper (reduces leakage) then lock-in at fhat
            w = self._taper(len(ti), frac=taper_frac)
            U, R2_ui = self._lockin_phasor(ti, ui*w, fhat)
            Y, R2_yi = self._lockin_phasor(ti, yi*w, fhat)
    
            # adaptive |U| floor: must be at least 1% of input std
            u_std = float(np.std(ui*w)) + 1e-12
            if np.abs(U) < 0.01 * u_std:
                reasons.append((i,(s0,s1,fhat), f"|U| too small ({abs(U):.3e} < {0.01*u_std:.3e})")); continue
    
            if (R2_ui < r2_min) or (R2_yi < r2_min):
                reasons.append((i,(s0,s1,fhat), f"poor R2 (u={R2_ui:.3f}, y={R2_yi:.3f})")); continue
    
            F.append(fhat)
            H.append(Y / U)
            R2u.append(R2_ui)
            R2y.append(R2_yi)
    
        F = np.array(F); H = np.array(H)
        R2u = np.array(R2u); R2y = np.array(R2y)
    
        if verbose:
            print(f"[frf_from_segments_autofreq] kept {len(F)} / {len(segments)} points")
            show = min(20, len(segments) - len(F))
            for r in reasons[:show]:
                print(" drop:", r)
            if len(reasons) > show:
                print(f" ... and {len(reasons)-show} more drops.")
    
        return F, H, R2u, R2y, reasons

    def frf_from_segments(self,t, u, y, fs, segments, settle_frac=0.3, r2_min=0.95, delay_samp=0):
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
            print('segment: ' + str(s0) +', ' + str(s1) +', ' +str(f))
            # settle/meas split
            n = s1 - s0
            s_settle = s0 + int(round(settle_frac * n))
            # measurement slice
            ti = t[s_settle:s1]
            ui = u[s_settle:s1] - np.mean(u[s_settle:s1])
            yi = y_al[s_settle:s1] - np.mean(y_al[s_settle:s1])
            if len(ti) < int(3 * fs / max(f,1e-6)):
                print("hepp1")
                continue
            U, r2u = self.sine_fit_complex(ti, ui, f)
            Y, r2y = self.sine_fit_complex(ti, yi, f)
            if np.abs(U) < 1e-12 or (r2u < r2_min) or (r2y < r2_min):
                print('U:' +str(U))
                print('r2u:' +str(r2u))
                print('r2y:' +str(r2y))
                print("hepp2")
                continue
            H.append(Y / U)
            F.append(f)
            R2u.append(r2u)
            R2y.append(r2y)
        print(F)
        print(H)
        print(R2u)
        print(R2y)
        return np.array(F), np.array(H), np.array(R2u), np.array(R2y)
    def check_timebase(self,t, nominal_fs=1000.0):
        dt = np.median(np.diff(t))
        fs_est = 1.0 / dt
        print(f"[timebase] median dt = {dt:.9f} s  →  fs ≈ {fs_est:.3f} Hz")
        if fs_est < nominal_fs*0.5 or fs_est > nominal_fs*1.5:
            print("  ⚠️ timestamps may not be in seconds (e.g., ms/us/ns). Convert to seconds first.")
    
    def _quad_peak(self,f, P, k):
        k = int(np.clip(k, 1, len(P)-2))
        y1, y2, y3 = P[k-1], P[k], P[k+1]
        den = (y1 - 2*y2 + y3)
        if den == 0: return f[k]
        delta = 0.5*(y1 - y3)/den
        return f[k] + delta*(f[k+1] - f[k])
    
    def _block_dominant_freq(self,x, fs, fmin=1.0, fmax=200.0, oversample=8):
        N = len(x)
        if N < 64: return np.nan
        w = np.hanning(N)
        nfft = int(2**np.ceil(np.log2(N)) * oversample)
        X = np.fft.rfft((x - x.mean()) * w, n=nfft)
        f = np.fft.rfftfreq(nfft, d=1.0/fs)
        band = (f >= fmin) & (f <= fmax)
        if not np.any(band): return np.nan
        P = (np.abs(X)**2)
        idx = np.where(band)[0]
        idx = idx[idx > 0]  # ignore DC
        if idx.size == 0: return np.nan
        k = idx[np.argmax(P[idx])]
        return float(self._quad_peak(f, P, k) if 0 < k < len(P)-1 else f[k])
    
    def detect_segments_by_dips(self,
        u, 
        fs,
        rms_win_s=0.25,          # RMS window (s) — set near your taper length
        thresh_frac=0.1,        # envelope threshold as fraction of envelope max
        min_gap_s=0.50,          # merge dips closer than this (s)
        trim_s=0.20,             # trim this many seconds off each side of every boundary
        min_len_s=0.40,          # drop segments shorter than this
        estimate_freq=True,      # optionally estimate freq per segment (oversampled FFT)
        fmin=0.5, fmax=1000.0,   # freq search band for estimation
        oversample=8
    ):
        """
        Return: segments = [(s0, s1)] or [(s0, s1, fhat)] if estimate_freq=True
        All indices are sample indices into u.
        """
        u = np.asarray(u, float)
        N = len(u)
        if N < 10:
            return []
    
        # --- Envelope via moving RMS (no SciPy needed) ---
        win = max(1, int(round(rms_win_s * fs)))
        w = np.ones(win, float) / win
        env = np.sqrt(np.convolve(u**2, w, mode='same'))
    
        # --- Threshold to find "dip" regions ---
        thr = float(thresh_frac) * (env.max() if np.isfinite(env).any() else 0.0)
        low = env < thr
    
        # --- Collapse contiguous low regions into dip centers ---
        dips = []
        in_region = False
        start = 0
        for i, flag in enumerate(low):
            if flag and not in_region:
                in_region = True
                start = i
            elif not flag and in_region:
                in_region = False
                stop = i
                center = (start + stop) // 2
                dips.append(center)
        if in_region:
            center = (start + N) // 2
            dips.append(center)
    
        if not dips:
            # No dips found: treat the whole record as one segment with edge trims
            s0 = int(round(trim_s * fs))
            s1 = int(round(N - trim_s * fs))
            if s1 > s0 and (s1 - s0) >= int(min_len_s * fs):
                if estimate_freq:
                    return [(*_estimate_segment_with_freq(u, fs, s0, s1, fmin, fmax, oversample),)]
                else:
                    return [(s0, s1)]
            return []
    
        # --- Merge dips that are very close ---
        min_gap = int(round(min_gap_s * fs))
        dips = np.array(sorted(dips), int)
        merged = [dips[0]]
        for d in dips[1:]:
            if d - merged[-1] < min_gap:
                merged[-1] = (merged[-1] + d) // 2  # keep the mid-point
            else:
                merged.append(d)
        dips = np.array(merged, int)
    
        # --- Build segments between dips, applying trims ---
        segments = []
        trim = int(round(trim_s * fs))
    
        # Prepend a virtual dip at start and append at end to form outer segments
        edges = np.concatenate([[0], dips, [N]])
        for a, b in zip(edges[:-1], edges[1:]):
            s0 = max(a + trim, 0)
            s1 = min(b - trim, N)
            if s1 - s0 >= int(min_len_s * fs):
                segments.append((s0, s1))
    
        if not estimate_freq:
            return segments
    
        # --- Optionally estimate frequency for each segment (oversampled FFT) ---
        segs_with_f = []
        for (s0, s1) in segments:
            fhat = self._estimate_freq_fft(u[s0:s1], fs, fmin=fmin, fmax=fmax, oversample=oversample)
            segs_with_f.append((s0, s1, fhat))
        return segs_with_f

    def _estimate_freq_fft(self,x, fs, fmin=0.5, fmax=1000.0, oversample=8):
        """Dominant frequency via oversampled periodogram with quadratic peak interpolation."""
        x = np.asarray(x, float)
        N = len(x)
        if N < 64:
            return np.nan
        # Hann window + oversampled rFFT
        w = np.hanning(N)
        nfft = int(2 ** np.ceil(np.log2(N)) * oversample)
        X = np.fft.rfft((x - x.mean()) * w, n=nfft)
        f = np.fft.rfftfreq(nfft, d=1.0 / fs)
        band = (f >= fmin) & (f <= fmax)
        if not np.any(band): return np.nan
        P = np.abs(X) ** 2
        idx = np.where(band)[0]
        idx = idx[idx > 0]  # skip DC
        if idx.size == 0: return np.nan
        k = idx[np.argmax(P[idx])]
        return self._quad_interp(f, P, k)

    def _quad_interp(self,f, P, k):
        """Quadratic peak interpolation around bin k."""
        k = int(np.clip(k, 1, len(P) - 2))
        y1, y2, y3 = P[k - 1], P[k], P[k + 1]
        den = (y1 - 2 * y2 + y3)
        if den == 0:
            return float(f[k])
        delta = 0.5 * (y1 - y3) / den
        return float(f[k] + delta * (f[k + 1] - f[k]))
    
    def segment_by_frequency_nonoverlap(self,u, fs,
                                        fmin=1.0, fmax=200.0,
                                        block_len_s=0.5, overlap=0.5,
                                        tol_lo=0.05, tol_hi=0.02, f_split=10.0):
        """
        Returns NON-OVERLAPPING segments [(s0, s1, f_hat), ...]
        built from short blocks with run-length merging.
    
        - block_len_s: analysis window (short so high-f steps don't mix tones)
        - overlap: 0..0.9 (e.g., 0.5 for 50%)
        - tol_lo: relative tol (±) when f < f_split
        - tol_hi: relative tol (±) when f >= f_split
        """
        n = len(u)
        blen = int(round(block_len_s * fs))
        hop = max(1, int(round(blen * (1.0 - overlap))))
        starts = np.arange(0, max(1, n - blen + 1), hop, dtype=int)
    
        fhat = np.full(len(starts), np.nan, float)
        for i, s in enumerate(starts):
            fhat[i] = self._block_dominant_freq(u[s:s+blen], fs, fmin=fmin, fmax=fmax, oversample=8)
    
        segs = []
        i = 0
        while i < len(starts):
            f0 = fhat[i]
            s0 = starts[i]
            e0 = s0 + blen
            if not np.isfinite(f0):
                i += 1
                continue
    
            j = i + 1
            while j < len(starts):
                fj = fhat[j]
                if not np.isfinite(fj):
                    break
                tol = tol_lo if min(f0, fj) < f_split else tol_hi
                # same-frequency run?
                if abs(fj - f0) / max(f0, 1e-12) <= tol:
                    f0 = 0.5*(f0 + fj)  # smooth
                    j += 1
                else:
                    break
    
            # segment span: from starts[i] to the END of the last block in the run
            s1 = starts[j-1] + blen
            segs.append((s0, s1, float(f0)))
            i = j
    
        # clip to signal length and drop trivially short segments
        segs2 = []
        for s0, s1, f in segs:
            s0 = int(max(0, min(s0, n-1))); s1 = int(max(s0+1, min(s1, n)))
            if s1 - s0 >= int(0.1*fs):  # keep at least 0.1 s
                segs2.append((s0, s1, f))
    
        return segs2


    def getSegments(self):
        return self.segments

    # ========================
    # Run the robust pipeline
    # ========================
    def analyze_stepped_sine(self,t, u, y, fs, tau_ms=0.0,
                             block_len_s=0.5, overlap=0.5, fmin=0.5, fmax=500.0,
                             freq_tolerance=0.02, settle_frac=0.3, r2_min=0.95):
        # 1) Estimate constant I/O delay (once)
        delay_samp = self.estimate_delay_samples(u, y, fs, max_ms=10.0, probe_s=3.0)
        delay_ms = 1000.0 * delay_samp / fs
    
        # 2) Detect tone segments (drift-resistant)
        #segments = self.segment_by_frequency_nonoverlap(u, fs, fmin=fmin, fmax=fmax, block_len_s=block_len_s, overlap=overlap, tol_lo=0.05, tol_hi=0.02, f_split=10.0)
        self.segments = self.detect_segments_by_dips(u, fs,fmin=fmin,fmax=fmax)

        print("segments (first 10):", self.segments[:10])

        #segments = self.merge_segments_adaptive(segments, self.fs)
        F, H, R2u, R2y = self.frf_from_segments( t, u, y, fs, self.segments, settle_frac=0.2,r2_min=r2_min)
        # 3) Compute FRF per detected tone
        #F, H, R2u, R2y, reasons = self.frf_from_segments_autofreq( t, u, y, fs,segments,      # or your segments
        #    settle_frac=0.20,
        #    min_meas_cycles=4.0,
        #    min_meas_points=400,
        #    r2_min=0.90,
        #    delay_samp=delay_ms,     # 0 if unsure
        #    f_search_min=1.0, f_search_max=200.0,
        #    oversample=8, taper_frac=0.05,
        #    verbose=True)
        
        # 4) Phase delay compensation (your known τ plus the estimated discrete delay)
        tau = max(0.0, tau_ms) * 1e-3
        Hc = H * np.exp(1j * 2*np.pi * F * tau)
    
        # 5) Magnitude & phase
        mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-16))
        phase_deg = np.degrees(np.unwrap(np.angle(H)))
        phase_c_deg = np.degrees(np.unwrap(np.angle(Hc)))
    
        self.results = {
            "F": F, "H": H, "mag_db": mag_db,
            "phase_deg": phase_deg, "phase_comp_deg": phase_c_deg,
            "delay_samples": delay_samp, "delay_ms_est": delay_ms,
            "R2_u": R2u, "R2_y": R2y, "segments": self.segments
        }
        

        return self.results

    def same_freq(self,fa, fb):
        tol = 0.05 if min(fa, fb) < 10.0 else 0.02
        return abs(fa - fb) / max(fa, 1e-12) <= tol

    def cycles(self,s0, s1, f): 
        return (s1 - s0) / self.fs * max(f, 1e-9)

    def merge_segments_adaptive(self,segments, fs, min_cycles=8, grow_gaps_s=0.5):
        """
        Merge adjacent/overlapping same-frequency segments and drop tiny ones.
        Handles list, tuple, or numpy.array inputs gracefully.
        """
        # --- normalize input ---
        if segments is None:
            return []
        if isinstance(segments, np.ndarray):
            # convert to list of tuples
            segments = [tuple(x) for x in segments.tolist()]
        elif isinstance(segments, tuple):
            # a single (s0, s1, f)
            segments = [segments]
        elif not isinstance(segments, list):
            raise TypeError(f"Unsupported type for 'segments': {type(segments)}")
    
        # Ensure it's a list of 3-element tuples/lists
        segments = [tuple(s[:3]) for s in segments if len(s) >= 3]
    
        if len(segments) == 0:
            return []
    
        # --- main loop ---
        segments = sorted(segments, key=lambda x: x[0])  # sort by start index
        out = []
    
        for s0, s1, f in segments:
            if self.cycles(s0, s1, f) < min_cycles:
                continue
            if not out:
                out.append([s0, s1, f])
                continue
    
            S0, S1, F = out[-1]
            small_gap = (s0 - S1) / fs <= grow_gaps_s
    
            if self.same_freq(F, f) and (s0 <= S1 or small_gap):
                out[-1][1] = max(S1, s1)
                out[-1][2] = 0.5 * (F + f)  # smooth freq estimate
            else:
                out.append([s0, s1, f])
    
        # --- filter short segments ---
        merged = []
        for s0, s1, f in out:
            if self.cycles(s0, s1, f) >= min_cycles:
                merged.append((s0, s1, f))
    
        return merged
        
    def plotBode(self):
        results = self.analyze_stepped_sine(self.t, self.u, self.y, self.fs,                
                                            block_len_s=self.block_len_s, overlap=self.overlap,
                                            fmin=self.fmin, fmax=self.fmax,
                                            freq_tolerance=self.freq_tolerance,
                                            settle_frac=self.settle_frac, r2_min=self.r2_min)

        F = results["F"]; mag_db = results["mag_db"]
        phase = results["phase_deg"]; phase_c = results["phase_comp_deg"]
        
        # Quality mask
        mask = (results["R2_u"] > self.r2_min) & (results["R2_y"] > self.r2_min)
        Fm, Mm, Ph, Pc = F[mask], mag_db[mask], phase[mask], phase_c[mask]
        # Plot

        plt.figure(figsize=(9,5))
        plt.subplots(2,1)
        plt.subplot(2,1,1)
        plt.semilogx(Fm, Mm, marker='o')
        plt.plot([min(Fm),max(Fm)],[0,0],'k--')
        plt.grid(True, which='both')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Bode Magnitude (Stepped-sine, drift-robust)")
        plt.tight_layout()
        #plt.figure(figsize=(9,5))
        plt.subplot(2,1,2)
        plt.semilogx(Fm, Ph, marker='o', label="Raw phase")
        plt.semilogx(Fm, Pc, marker='o', label=f"Phase (+{self.tau_ms:.2f} ms)")
        plt.plot([min(Fm),max(Fm)],[-180,-180],'k--')
        plt.grid(True, which='both')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (deg)")
        plt.title("Bode Phase (Stepped-sine)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        print(f"Estimated I/O delay from data: {results['delay_ms_est']:.2f} ms  (integer-sample)")
        print("Median R^2 (input, output):", np.median(results["R2_u"][mask]), np.median(results["R2_y"][mask]))


# 0..33000
#36000..63000
#67000..93000
#95000..116000
#119000..138000
#141000..158000
#160000..174000
#177000..190000
