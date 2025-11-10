from epics import PV

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

        for pv in self.pvs.values():
          if pv is not None:
            pv.prec = 4
            pv.convert = True
         
            self._buf[pv] = [] # (t, value)

        # Monitor buffers
        #self._buf = {
        #    "SP_RBV": [],   
        #    "VEL_ACT": [],
        #    "POS_ACT": [],
        #    "TRQ_ACT": [],
        #}
        self._lock = threading.Lock()
        self._cids = {}  # callback IDs

    def cleanup(self):
        sp_pv = self.pvs["SP"]
        if sp_pv is None:
          return
        sp_pv.put(0, wait=False)

    # -----------------------------
    # Monitors
    # -----------------------------
    def _cb_factory(self, key):
        def _cb(pvname=None, value=None, **kw):
            ts = kw.get("timestamp", time.time())
            with self._lock:
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

    def get_data(self):
        return self._buf


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
    

