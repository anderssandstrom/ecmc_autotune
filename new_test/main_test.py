# bode.py
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from epics import PV
import excite_sine as excite
import epics_logger as logger
import plot_log
import analyze

PREFIX = "c6025a-08:m1s000-"
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

# vel to PV scaling [rad/s] 31bits=8000Hz=8000*2*pi rad/s for PVs 1/42722.83, the "9.5367e-7" is the strange scale of the PV
velScaleInput  = 1 #[rad/s]
velScaleOutput = 1 #[rad/s]

# Motor rated trq [Nm]
motorRatedTrq = 0.5
# trqSetpoint in 1% of rated 
trqScaleInput  = 100 / motorRatedTrq
trqScaleOutput = 1 / trqScaleInput

if __name__ == "__main__":    
    
    #-  Generate setpoint signal
    fs = 1000.0             # Sampling frequency [Hz]
    f_start, f_stop = 1, 100
    n_points = 40           # Number of frequencies (log spaced)
    amp = 30.0              # Command amplitude (+/-)
    n_settle = 10           # Settle cycles before measuring
    n_meas = 20             # Measured cycles per frequency
    taper = 0.05            # fraction of cosine fade-in/out per step


    #fs = 1000.0             # Sampling frequency [Hz]
    #f_start, f_stop = 1, 50
    #n_points = 10           # Number of frequencies (log spaced)
    #amp = 10.0              # Command amplitude (+/-)
    #n_settle = 5           # Settle cycles before measuring
    #n_meas = 5              # Measured cycles per frequency
    #taper = 0.05            # fraction of cosine fade-in/out per step

    mylogger = logger.logger(Ts=1/fs, pvs=PVS)
    mylogger.load_log("data.pkl")
    
    # Itersection is important (use only values frokm both (ignore old startup value for cmd))
    t, vals_by_pv, X, pv_names = mylogger.resample_to_common_time_base(fill="extrapolate", time_range ="intersection")
    #print(pv_names)
    #print(t[1:100]) 
    #print(vals_by_pv["SP"])
    #print(vals_by_pv["VEL_ACT"])
    plot_log.plot(t,vals_by_pv)
    
    # now do bode for SP_RBV and VEL_ACT
    my_bode=analyze.bode(t, vals_by_pv["SP_RBV"], vals_by_pv["VEL_ACT"], fs, tau_ms=2.5,
                         block_len_s=1.5, overlap=0.8, fmin=f_start, fmax=f_stop,
                         freq_tolerance=0.05, settle_frac=0.3, r2_min=0.95)
    my_bode.plotBode()
 
