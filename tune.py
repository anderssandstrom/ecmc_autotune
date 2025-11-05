# epics_autotune.py

 
# c6025a-08:m1s000-Drv01-TrgDS402Ena
# c6025a-08:m1s000-Drv01-Cmd-RB
# c6025a-08:m1s000-Drv01-Trq-RB
# c6025a-08:m1s000-Enc01-PosAct
# c6025a-08:m1s000-Drv01-VelAct
# c6025a-08:m1s000-Drv01-TrqAct
# c6025a-08:m1s000-Drv01-VolAct
# c6025a-08:m1s000-Drv02-TmpAct
# c6025a-08:m1s000-NxtObjId
# c6025a-08:m1s000-PrvObjId
# c6025a-08:m1s000-Drv01-Cmd
# c6025a-08:m1s000-Drv01-Trq
# c6025a-08:m1s000-Drv01-WrnAlrm
# c6025a-08:m1s000-Drv01-ErrAlrm
# c6025a-08:m1s000-Tp01-BI01
# c6025a-08:m1s000-Tp01-BI02
# c6025a-08:m1s000-Enc01-NotValid
# c6025a-08:m1s000-Online
# c6025a-08:m1s000-Operational
# c6025a-08:m1s000-Alstate-Init
# c6025a-08:m1s000-Alstate-Preop
# c6025a-08:m1s000-Alstate-Safeop
# c6025a-08:m1s000-Alstate-Op
# c6025a-08:m1s000-EntryCntr
# c6025a-08:m1s000-Stat
# c6025a-08:m1s000-One
# c6025a-08:m1s000-Zero
# c6025a-08:m1s000-Drv01-Stat
# c6025a-08:m1s000-Tp01-Stat
# c6025a-08:m1s000-Enc01-Stat
# c6025a-08:m1s000-Drv01-TrgDS402Ena
# c6025a-08:m1s000-Drv01-TrgDS402Dis


import time
import threading
import numpy as np
from epics import PV
import matplotlib.pyplot as plt

# =========================
# >>> USER PV CONFIG <<<
# Define your PVs here
# =========================

# 31bits=8000Hz=8000*2*pi rad/s

velScale = 8000*2*np.pi/(2**31)

prefix = "c6025a-08:m1s000-"

# Note TrqAct woks much better for fiting since in same timebase as teh other values
PVS = {
    # Setpoint you will WRITE (switch between torque or velocity target)
    "SP": prefix + "Drv01-Trq",            # e.g. CST torque PV OR CSV velocity PV
    # Readback of setpoint (the drive's seen/latched target)
    "SP_RBV": prefix + "Drv01-TrqAct",    # optional but recommended
    # Actual signals to MONITOR
    "VEL_ACT": prefix + "Drv01-VelAct",      # 0x606C equivalent
    "POS_ACT": prefix + "Enc01-PosAct",      # 0x6064 equivalent
    "TRQ_ACT": prefix + "Drv01-TrqAct",      # 0x6077 equivalent (optional)
}

def align(u, y, max_lag=50):
    """Align input u and output y by cross-correlation (max_lag in samples)."""
    import numpy as np
    c = np.correlate(u - u.mean(), y - y.mean(), mode='full')
    lag = np.arange(-len(u)+1, len(u))[np.argmax(c)]
    lag = int(np.clip(lag, -max_lag, max_lag))
    if lag > 0:
        u = u[lag:]
        y = y[:-lag]
    elif lag < 0:
        u = u[:lag]
        y = y[-lag:]
    return u, y, lag

def plot_log(log, title="Excitation Test Log"):
    """
    Plot recorded data from EthercatAutoTuner or EPICS run.
    Expects dict with at least keys:
        log["t"], log["cmd"], log["vel"], log["pos"], log["torque"]
    Missing channels are ignored gracefully.
    """
    t = log.get("t", None)
    if t is None:
        raise ValueError("log must contain a time vector 't'")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    fig.suptitle(title)

    # --- Command / Setpoint
    if "cmd" in log:
        axes[0].plot(t, log["cmd"], lw=1.2, label="Command / Setpoint")
        axes[0].set_ylabel("Command")
        axes[0].legend()
        axes[0].grid(True, ls="--", alpha=0.4)

    # --- Velocity
    if "vel" in log:
        axes[1].plot(t, log["vel"], lw=1.0, label="Velocity actual")
        axes[1].set_ylabel("Velocity [unit/s]")
        axes[1].legend()
        axes[1].grid(True, ls="--", alpha=0.4)

    # --- Position or Torque
    if "pos" in log and np.any(log["pos"]):
        axes[2].plot(t, log["pos"], lw=1.0, label="Position actual")
        axes[2].set_ylabel("Position [unit]")
    elif "torque" in log and np.any(log["torque"]):
        axes[2].plot(t, log["torque"], lw=1.0, label="Torque actual")
        axes[2].set_ylabel("Torque [Nm]")

    axes[2].set_xlabel("Time [s]")
    axes[2].legend()
    axes[2].grid(True, ls="--", alpha=0.4)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

class EthercatAutoTunerEPICS:
    def __init__(self, Ts=1e-3, pvs=PVS):
        self.Ts = float(Ts)
        self.pvs = {k: PV(v) for k, v in pvs.items()}

        for pv in self.pvs.values():
          if pv is not None:
            pv.prec = 4
            pv.convert = True


        # Monitor buffers
        self._buf = {
            "SP_RBV": [],   # (t, value)
            "VEL_ACT": [],
            "POS_ACT": [],
            "TRQ_ACT": [],
        }
        self._lock = threading.Lock()
        self._cids = {}  # callback IDs

    def cleanup(self):
        sp_pv = self.pvs["SP"]
        if sp_pv is None:
          return
        sp_pv.put(0, wait=False)

    # -----------------------------
    # Excitation generators
    # -----------------------------
    def prbs_bits(self, n=10):
        taps = {10: (10, 7), 11: (11, 9)}
        if n not in taps:
            raise ValueError("Supported n: 10 or 11.")
        t = taps[n]
        state = (1 << n) - 1
        bits = []
        for _ in range((1 << n) - 1):
            bits.append(state & 1)
            fb = ((state >> (t[0]-1)) ^ (state >> (t[1]-1))) & 1
            state = (state >> 1) | (fb << (n-1))
        return np.array(bits, dtype=np.uint8)

    def prbs_waveform(self, A=0.15, Tc=0.008, n=10):
        chips = int(round(Tc / self.Ts))
        if chips < 1:
            raise ValueError("Tc/Ts must be ≥ 1 sample.")
        bits = self.prbs_bits(n)
        return (np.repeat(bits.astype(float) * 2 - 1.0, chips)) * float(A)

    def chirp_waveform(self, A=0.15, f1=1.0, f2=120.0, T=12.0):
        t = np.arange(0.0, T, self.Ts)
        k = np.log(f2 / f1) / T
        phase = 2 * np.pi * f1 * (np.exp(k * t) - 1) / k
        return A * np.sin(phase)

    # -----------------------------
    # Monitors
    # -----------------------------
    def _cb_factory(self, key):
        def _cb(pvname=None, value=None, **kw):
            ts = kw.get("timestamp", time.time())
            with self._lock:
                if key=="VEL_ACT":
                    value = float(value) / 9.5367e-7 * velScale
                self._buf[key].append((ts, float(value)))
                print(key + "  " + str(float(value)) )
        return _cb

    def start_monitors(self):
        # Create monitors for SP_RBV (if defined) and actuals
        for key in ["SP_RBV", "VEL_ACT", "POS_ACT", "TRQ_ACT"]:
            pv = self.pvs.get(key)
            if pv is not None and pv.pvname:
                self._buf[key].clear()
                self._cids[key] = pv.add_callback(self._cb_factory(key))

    def stop_monitors(self):
        for key, cid in list(self._cids.items()):
            pv = self.pvs.get(key)
            try:
                if pv is not None:
                    pv.remove_callback(cid)
            finally:
                self._cids.pop(key, None)

    # -----------------------------
    # Run excitation using EPICS put() and monitors
    # -----------------------------
    def run_epics_test(self, setpoints, flush_ms=100):
        """
        Writes setpoints to SP PV using put(), monitors collect RBV and actuals.
        After streaming, waits flush_ms so late monitor events arrive.
        Returns raw monitor buffers and a 'sent' timeline for reference.
        """
        sp_pv = self.pvs["SP"]
        if sp_pv is None:
            raise RuntimeError("SP PV not configured.")

        self.start_monitors()
        sent_t = np.zeros(len(setpoints))
        sent_u = np.zeros(len(setpoints))

        t0 = time.monotonic()
        for k, u in enumerate(setpoints):
            sp_pv.put(float(u), wait=False)
            sent_t[k] = time.monotonic() - t0
            sent_u[k] = u
            # pacing: best-effort; your IOC timing may dominate anyway
            time.sleep(self.Ts)

        sp_pv.put(0, wait=False)
        # allow monitors to catch up
        time.sleep(flush_ms / 1000.0)
        self.stop_monitors()

        # Copy buffers safely
        with self._lock:
            buf = {k: np.array(v, dtype=float).reshape(-1, 2) if len(v) else
                   np.zeros((0, 2), dtype=float)
                   for k, v in self._buf.items()}

        # Also attach the 'sent' arrays (relative monotonic time)
        buf["SENT"] = np.column_stack([sent_t, sent_u])
        return buf

    # -----------------------------
    # Resampling utilities
    # -----------------------------
    def _resample(self, tv_pairs, t_grid):
        """Linear resample series defined by (t, v) onto t_grid."""
        if tv_pairs.shape[0] == 0:
            return np.zeros_like(t_grid)
        t = tv_pairs[:, 0]
        v = tv_pairs[:, 1]
        # if timestamps are absolute (Unix), normalize them
        if t.max() > 1e6:
            t = t - t.min()
        # Ensure strictly increasing for interp
        order = np.argsort(t)
        t = t[order]; v = v[order]
        # Guard against repeated timestamps
        mask = np.concatenate([[True], np.diff(t) > 1e-9])
        t = t[mask]; v = v[mask]
        # Extrapolate with edges
        return np.interp(t_grid, t, v, left=v[0], right=v[-1])

    def build_uniform_log(self, buf, t_end=None):
        """
        Create a uniform-time log (dict of arrays) at Ts from monitor buffers.
        If t_end not given, uses max(SENT time).
        """
        sent = buf["SENT"]
        if sent.shape[0] == 0:
            raise RuntimeError("No SENT data captured.")
        tmax = float(sent[:, 0].max()) if t_end is None else float(t_end)
        t = np.arange(0.0, tmax, self.Ts)

        log = {"t": t}
        # Choose command source: prefer SP_RBV monitor; else SENT values
        if buf["SP_RBV"].shape[0] > 0:            
            log["cmd"] = self._resample(buf["SP_RBV"], t)            
        else:
            log["cmd"] = np.interp(t, sent[:, 0], sent[:, 1], left=sent[0, 1], right=sent[-1, 1])

        for key, outkey in [("VEL_ACT", "vel"), ("POS_ACT", "pos"), ("TRQ_ACT", "torque")]:
            log[outkey] = self._resample(buf[key], t)

        return log

    # -----------------------------
    # Signal processing helpers
    # -----------------------------
    def moving_avg(self, x, win):
        w = max(1, int(win))
        if w == 1:
            return x.copy()
        pad = w // 2
        xpad = np.pad(x, (pad, pad), mode="edge")
        ker = np.ones(w) / w
        return np.convolve(xpad, ker, mode="valid")

    def diff_robust(self, x, smooth_win=11):
        x_s = self.moving_avg(x, smooth_win)
        dx = np.empty_like(x_s)
        dx[1:-1] = (x_s[2:] - x_s[:-2]) / (2 * self.Ts)
        dx[0] = (x_s[1] - x_s[0]) / self.Ts
        dx[-1] = (x_s[-1] - x_s[-2]) / self.Ts
        return dx

    # -----------------------------
    # Identification
    # -----------------------------
    def identify_cst(self, log, smooth_win=11):
        """CST (torque→velocity):  τ ≈ J·dω + B·ω + τc·sgn(ω)."""
        tau = np.asarray(log["cmd"])
        w = np.asarray(log["vel"])
        w_s = self.moving_avg(w, smooth_win)
        dw = self.diff_robust(w_s, smooth_win)
        X = np.column_stack([dw, w_s, np.sign(w_s)])
        theta, *_ = np.linalg.lstsq(X, tau, rcond=None)
        resid = tau - X @ theta
        return {
            "J": float(theta[0]),
            "B": float(theta[1]),
            "tau_c": float(theta[2]),
            "resid_rms": float(np.sqrt(np.mean(resid**2))),
        }

    def identify_csv(self, log, smooth_win=11):
        """CSV (velocity_cmd→velocity):  v = -T_v·dv + K_v·v_cmd."""
        vcmd = np.asarray(log["cmd"])
        v = np.asarray(log["vel"])
        v_s = self.moving_avg(v, smooth_win)
        dv = self.diff_robust(v_s, smooth_win)
        X = np.column_stack([-dv, vcmd])
        beta, *_ = np.linalg.lstsq(X, v_s, rcond=None)
        Tv, Kv = float(beta[0]), float(beta[1])
        resid = v_s - (-Tv * dv + Kv * vcmd)
        return {
            "Kv": Kv,
            "Tv": Tv,
            "resid_rms": float(np.sqrt(np.mean(resid**2))),
        }

    # -----------------------------
    # Gain suggestions
    # -----------------------------
    def velocity_pi_from_JB(self, J, B, f_bw=100.0, zeta=1.0):
        alpha = 2 * np.pi * float(f_bw)
        Kp = (2 * zeta * alpha) * J - B
        Ki = J * alpha * alpha
        return max(float(Kp), 1e-9), max(float(Ki), 0.0)

    def position_pi_from_Kv(self, Kv, f_bw=20.0, zeta=1.0):
        alpha = 2 * np.pi * float(f_bw)
        Kp = (2 * zeta * alpha) / Kv
        Ki = (alpha * alpha) / Kv
        return max(float(Kp), 1e-9), max(float(Ki), 0.0)

# =========================
# Example usage
# =========================
if __name__ == "__main__":    
    print("1")
    tuner = EthercatAutoTunerEPICS(Ts=0.001, pvs=PVS)
    print("2")
    # --- Build excitation (pick ONE)
    setpoints = tuner.prbs_waveform(A=10, Tc=0.008, n=10)   # ~8.2 s PRBS
    # setpoints = tuner.chirp_waveform(A=0.15, f1=1, f2=120, T=12)
    
    for val in setpoints:
       print(val)
    # --- Stream via EPICS and collect monitors
    
    try:
      buf = tuner.run_epics_test(setpoints, flush_ms=150)
    except KeyboardInterrupt:
      tuner.cleanup()
      quit()

    print("4")
    # --- Resample monitors to uniform grid


    log = tuner.build_uniform_log(buf)
    plot_log(log)

    #tau, w, lag = align(log["cmd"], log["vel"])
    #print(f"Alignment shift: {lag} samples")
    # Replace the aligned signals back into log
    # tempraryu rescale velocity to rad/s    
    #log["cmd"], log["vel"] = tau, w
    #plot_log(log)
    print("5")
    # --- Identify (pick CST or CSV path)
    # CST path (torque setpoints in SP): use velocity actual for ID
    id_cst = tuner.identify_cst(log)
    print(f"[CST] J={id_cst['J']:.4e}, B={id_cst['B']:.4e}, tau_c={id_cst['tau_c']:.4e}, resid={id_cst['resid_rms']:.3e}")
    Kp_v, Ki_v = tuner.velocity_pi_from_JB(id_cst["J"], id_cst["B"], f_bw=100)
    print(f"Velocity PI: Kp={Kp_v:.3g}, Ki={Ki_v:.3g}")

    # CSV path (velocity setpoints in SP): use velocity_cmd→velocity identification
    # id_csv = tuner.identify_csv(log)
    # print(f"[CSV] Kv={id_csv['Kv']:.3f}, Tv={id_csv['Tv']:.3e}, resid={id_csv['resid_rms']:.3e}")
    # Kp_p, Ki_p = tuner.position_pi_from_Kv(id_csv["Kv"], f_bw=20)
    # print(f"Position PI: Kp={Kp_p:.3g}, Ki={Ki_p:.3g}")



#
#[CST] J=-1.0626e-05, B=-3.2935e-02, tau_c=2.7626e-01, resid=1.940e+00
#Velocity PI: Kp=0.0196, Ki=0



#[CST] J=2.7981e-02, B=2.2396e-01, tau_c=5.9351e-01, resid=5.383e+00
#Velocity PI: Kp=34.9, Ki=1.1e+04
#[CST] J=2.7885e-02, B=2.0322e-01, tau_c=7.3048e-01, resid=5.335e+00
#Velocity PI: Kp=34.8, Ki=1.1e+04

