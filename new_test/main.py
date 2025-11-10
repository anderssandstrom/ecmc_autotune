# bode.py
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import excite_sine as excite
import epics_logger as logger

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
    f_start, f_stop = 1, 200
    n_points = 40           # Number of frequencies (log spaced)
    amp = 30.0              # Command amplitude (+/-)
    n_settle = 10           # Settle cycles before measuring
    n_meas = 20             # Measured cycles per frequency
    taper = 0.05            # fraction of cosine fade-in/out per step
    t, u, freqs = excite.generate(fs,f_start, f_stop,n_points, amp, n_settle, n_meas, taper)
    mylogger = logger.logger(Ts=1/fs, pvs=PVS)
    mylogger.start_monitors()
    
    sleep(1)

    mylogger.stop_monitors()

    mylogger