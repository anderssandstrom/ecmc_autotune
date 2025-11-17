import numpy as np
import matplotlib.pyplot as plt
import plot_log

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
        return np.array(F), np.array(H), np.array(R2u), np.array(R2y)
   
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
        self.segments = self.detect_segments_by_dips(u, fs,fmin=fmin,fmax=fmax)

        print("segments (first 10):", self.segments[:10])
        F, H, R2u, R2y = self.frf_from_segments( t, u, y, fs, self.segments, settle_frac=0.2,r2_min=r2_min)
        
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
        
        # Plot amplitude        
        fig, axes = plt.subplots(3,1, constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.2, hspace=0.1, wspace=0.1)

        plt.subplot(3,1,1)
        
        plt.semilogx(Fm, Mm, marker='o')
        plt.plot([min(Fm),max(Fm)],[0,0],'k--')
        plt.grid(True, which='both')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Bode Magnitude (Stepped-sine)")
        
        
        # Plot phase
        plt.subplot(3,1,2)
        plt.semilogx(Fm, Ph, marker='o', label="Raw phase")
        plt.semilogx(Fm, Pc, marker='o', label=f"Phase (+{self.tau_ms:.2f} ms)")
        plt.plot([min(Fm),max(Fm)],[-180,-180],'k--')
        plt.grid(True, which='both')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (deg)")
        plt.title("Bode Phase (Stepped-sine)")
        plt.legend()
        
    
        # Plot raw data
        ax=plt.subplot(3,1,3)
        plot_log.plot2(self.t,self.u, self.y,segments=self.segments,fs=self.fs,ax=ax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Raw")
        ax.set_title("Raw values")
        ax.legend(['Command []', 'Actual []'])        
        plt.show()

        print(f"Estimated I/O delay from data: {results['delay_ms_est']:.2f} ms  (integer-sample)")
        print("Median R^2 (input, output):", np.median(results["R2_u"][mask]), np.median(results["R2_y"][mask]))
