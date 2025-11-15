import time
from epics import PV
import threading
import numpy as np
from pathlib import Path
import pickle

PREFIX = "c6025a-08:m1s000-"
# Note TrqAct woks much better for fiting since in same timebase as teh other values
PVS = {
    # Setpoint you will WRITE (switch between torque or velocity target)
    "SP": PREFIX + "Drv01-Spd",              # e.g. CST torque PV OR CSV velocity PV
    # Readback of setpoint (the drive's seen/latched target)
    "SP_RBV": PREFIX + "Drv01-Spd-RB",       # optional but recommended, works better than 
    # Actual signals to MONITOR
    "VEL_ACT": PREFIX + "Drv01-VelAct",      # 0x606C equivalent
    "POS_ACT": PREFIX + "Enc01-PosAct",      # 0x6064 equivalent
    "TRQ_ACT": PREFIX + "Drv01-TrqAct",      # 0x6077 equivalent (optional)
}

class logger:
    def __init__(self, Ts=1e-3, pvs=PVS):
        self.Ts = float(Ts)
        self.pvs = {k: PV(v) for k, v in pvs.items()}
        self._buf = {}
        self.metadata = {}
        for pv in self.pvs.values():
          if pv is not None:
            pv.prec = 4
            pv.convert = True
        for name in self.pvs.keys():
            self._buf[name] = [] # (t, value)

        self._lock = threading.Lock()
        self._cids = {}  # callback IDs

    # -----------------------------
    # Monitors
    # -----------------------------
    def _cb_factory(self, key):        
        def _cb(pvname=None, value=None, **kw):
            ts = kw.get("timestamp", time.time())
            if self.fistDataPoint:
               self.time_offset=ts
            self.fistDataPoint=0
            #print(ts-self.time_offset)
            with self._lock:
                self._buf[key].append((ts-self.time_offset, float(value)))
                #print(key + "  " + str(float(value)) )
        return _cb

    def start_monitors(self):
        # Create monitors for SP_RBV (if defined) and actuals
        self.fistDataPoint=1        
        for key in self.pvs: #["SP_RBV", "VEL_ACT", "POS_ACT", "TRQ_ACT"]:
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

    def save_log(self, filename, metadata=None):
        """
        Save a log dict to compressed .npz file, plus optional JSON metadata.
    
        Example:
            save_log(log, "logs/test_2025-11-07_run1")
        """
        filename = Path(filename)
        if filename.suffix == "":
            filename = filename.with_suffix(".pkl")

        payload = {"data": self._buf, "metadata": metadata or self.metadata or {}}
        with open(filename, "wb") as f:
            pickle.dump(payload, f)
        
        print(f"[saved] {filename}")

    def load_log(self, filename):
        """
        Load a log saved by save_log(). Returns a plain dict with numpy arrays.
        """
        filename = Path(filename)
        if filename.suffix == "":
            filename = filename.with_suffix(".pkl")
        with open(filename, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "data" in payload and "metadata" in payload:
            self._buf = payload.get("data", {})
            self.metadata = payload.get("metadata", {})
        else:
            self._buf = payload
            self.metadata = {}
        print(f"[loaded] {filename}")        

    def get_data(self):
        return self._buf

    def get_data(self,key):
        return self._buf[key]

    # All PVs are sampled with its own timebase 
    def resample_to_common_time_base(
        self,
        dt=None,                 # seconds per sample (if None and fs is None, inferred)
        fs=None,                 # Hz (alternative to dt)
        t_min=None,              # optional start time override (seconds)
        t_max=None,              # optional end time override (seconds)
        time_range="union",      # "union" or "intersection"
        method="linear",         # "linear" or "zoh"
        fill="nan",              # for linear: "nan" or "extrapolate"
        return_matrix=True       # also return stacked (T x M) matrix
    ):
        """
        Resample EPICS PV monitor logs onto a common timebase.
    
        Parameters
        ----------
        data :
            One of:
              - dict: {pv_name: np.array shape (N,2) [timestamp, value]}
              - list of dicts (merged by keys)
              - list of (pv_name, np.array) pairs
        dt : float, optional
            Time step in seconds. If None and fs is None, inferred from the densest PV.
        fs : float, optional
            Sampling frequency in Hz (alternative to dt). If both dt and fs are given, dt wins.
        t_min, t_max : float, optional
            Override the output time range (seconds, same epoch as inputs).
        time_range : {"union","intersection"}
            "union": from min(start_i) to max(end_i);
            "intersection": from max(start_i) to min(end_i).
        method : {"linear","zoh"}
            Interpolation method. "zoh" is zero-order hold (previous sample).
        fill : {"nan","extrapolate"}
            For linear only: outside each PV’s span, use NaN or edge extrapolation.
            For "zoh", times before first sample -> NaN; after last sample -> hold last.
        return_matrix : bool
            If True, also return a stacked matrix values[T, M] ordered by 'pv_names'.
    
        Returns
        -------
        t : np.ndarray (T,)
        values_by_pv : dict {name: np.ndarray (T,)}
        X : np.ndarray (T, M) or None
        pv_names : list of str
        """

        data = self._buf
        # ---- Normalize input to dict {name: Nx2 array} ----
        if isinstance(data, dict):
            pv_dict = {str(k): np.asarray(v) for k, v in data.items()}
        elif isinstance(data, list):
            pv_dict = {}
            for item in data:
                if isinstance(item, dict):
                    for k, v in item.items():
                        pv_dict[str(k)] = np.asarray(v)
                else:
                    name, arr = item
                    pv_dict[str(name)] = np.asarray(arr)
        else:
            raise TypeError("Unsupported 'data' type. Use dict, list[dict], or list[(name, array)].")
    
        if not pv_dict:
            raise ValueError("No PVs found in 'data'.")
    
        # ---- Clean & sort each PV ----
        cleaned = {}
        spans = []
        for name, arr in pv_dict.items():
            arr = np.asarray(arr)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("PV '%s' must be array of shape (N,2): [timestamp, value]." % name)
    
            t_i = np.asarray(arr[:, 0], dtype=float)
            v_i = np.asarray(arr[:, 1], dtype=float)
    
            mask = np.isfinite(t_i)
            t_i = t_i[mask]; v_i = v_i[mask]
            if t_i.size == 0:
                continue
    
            order = np.argsort(t_i)
            t_i = t_i[order]; v_i = v_i[order]
    
            # keep last occurrence for duplicate timestamps
            _, last_idx = np.unique(t_i[::-1], return_index=True)
            keep_idx = (len(t_i) - 1) - last_idx
            keep_idx.sort()
            t_i = t_i[keep_idx]; v_i = v_i[keep_idx]
    
            good = np.isfinite(v_i)
            t_i = t_i[good]; v_i = v_i[good]
            if t_i.size < 2:
                continue
    
            cleaned[name] = (t_i, v_i)
            spans.append((t_i[0], t_i[-1]))
    
        if not cleaned:
            raise ValueError("All PVs empty after cleaning/sorting.")
    
        # ---- Determine output time range ----
        starts = np.array([s for s, _ in spans])
        ends = np.array([e for _, e in spans])
    
        if time_range == "union":
            start = starts.min()
            end = ends.max()
        elif time_range == "intersection":
            start = starts.max()
            end = ends.min()
            if not (end > start):
                raise ValueError("Empty intersection of PV time spans.")
        else:
            raise ValueError("time_range must be 'union' or 'intersection'.")
    
        if t_min is not None:
            start = max(start, float(t_min))
        if t_max is not None:
            end = min(end, float(t_max))
        if not (end > start):
            raise ValueError("Invalid t_min/t_max after applying range constraints.")
    
        # ---- Choose dt ----
        if dt is None:
            if fs is not None:
                dt = 1.0 / float(fs)
            else:
                med_dts = []
                for t_i, _ in cleaned.values():
                    diffs = np.diff(t_i)
                    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                    if diffs.size:
                        med_dts.append(np.median(diffs))
                if not med_dts:
                    raise ValueError("Cannot infer dt; provide dt or fs explicitly.")
                dt = float(np.min(med_dts))  # densest PV sets dt
        else:
            dt = float(dt)
        if dt <= 0:
            raise ValueError("dt must be positive.")
    
        # ---- Build uniform time grid ----
        eps = dt * 1e-9
        t = np.arange(start, end + eps, dt)
        T = t.size
    
        # ---- Interpolate each PV ----
        values_by_pv = {}
        pv_names = sorted(cleaned.keys())
    
        for name in pv_names:
            ti, vi = cleaned[name]
    
            if method == "linear":
                y = np.interp(t, ti, vi)
                if fill == "nan":
                    y[t < ti[0]] = np.nan
                    y[t > ti[-1]] = np.nan
                elif fill != "extrapolate":
                    raise ValueError("fill must be 'nan' or 'extrapolate' for linear method.")
    
            elif method == "zoh":
                idx = np.searchsorted(ti, t, side="right") - 1
                y = np.full(T, np.nan, dtype=float)
                valid = idx >= 0
                y[valid] = vi[idx[valid]]
            else:
                raise ValueError("method must be 'linear' or 'zoh'.")
    
            values_by_pv[name] = y
    
        X = None
        if return_matrix:
            X = np.column_stack([values_by_pv[name] for name in pv_names])
    
        return t, values_by_pv, X, pv_names

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

    t, u, freqs = generate(fs,f_start, f_stop, n_points, amp, n_settle, n_meas, taper)
    
