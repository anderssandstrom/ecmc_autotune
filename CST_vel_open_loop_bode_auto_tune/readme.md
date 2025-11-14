# General

Tunings:
* step 1: Velocity loop (terminal in CST)
* step 2: Position loop (terminal in CSV)

WARNING: Things will move and limit switches is not supervised!!!!

* SCALINGS need to be adjusted for all logged parametres, 
  * Nm for troque, 
  * rad/s for velo,... The velo scaling in PV is strange.
* Stuff will move
* No ecmc axis linked to hardware. Data direct from PVS
* Data direct from PVs (increase PV rate to max needed)
```
${SCRIPTEXEC} ${ecmccfg_DIR}setRecordUpdateRate.cmd "RATE_MS=1"
${SCRIPTEXEC} ${ecmccfg_DIR}addSlave.cmd,      "HW_DESC=EP7211-0034_CST_STD"
#- Limit torque to 50% of motor rated torque.  Rated current = 2710mA, set to half I_MAX_MA=1355
${SCRIPTEXEC} ${ecmccfg_DIR}applyComponent.cmd "COMP=Motor-Beckhoff-AM8111-XFX0, MACROS='I_MAX_MA=1355'"
${SCRIPTEXEC} ${ecmccfg_DIR}restoreRecordUpdateRate.cmd
```

# Semi successful tune for velo loop in CST
**Note TrqAct woks much better for fiting since in same timebase as the other values**

** Scalings are still wrong. For instance Torque should be in Nm and not in 0.1% of rated torque.**

This cfg for CST seems to work (but wrong scalings)
```
in main:
setpoints = tuner.prbs_waveform(A=10, Tc=0.008, n=10)   # ~8.2 s PRBS

PVS = {
    # Setpoint you will WRITE (switch between torque or velocity target)
    "SP": prefix + "Drv01-Trq",              # e.g. CST torque PV OR CSV velocity PV
    # Readback of setpoint (the drive's seen/latched target)
    "SP_RBV": prefix + "Drv01-TrqAct",       # optional but recommended, works better than 
    # Actual signals to MONITOR
    "VEL_ACT": prefix + "Drv01-VelAct",      # 0x606C equivalent
    "POS_ACT": prefix + "Enc01-PosAct",      # 0x6064 equivalent
    "TRQ_ACT": prefix + "Drv01-TrqAct",      # 0x6077 equivalent (optional)
}
```

# CST

1. Identify mech system
2. Calculate control params

## 1. Identify CST system

Sequence:
1. The system is excited by a prbs (Pseudo random binary sequence) of torqus (+- a set torque) and samples data
2. The velocity signal w is filtered and smoothed and then differntiated to calculate teh acceleration `dw/dt`.
3. The data is fitted to the mechanical model (torque balance):
```
tau=J*dw/dt*+B*w+tc*sign(w)

* Tau or T torque [N/m]
* w: velocity [rad/s]
* J: inertia [Nms²/rad] or [kgm²] (the total reflected inertia seen by the motor shaft (motor + load))
* B: viscous damping [Nms/rad]
* tc: Coulomb (static) friction [Nm]
```
4. Solve least squares and finds best fit of J, B, Tc and calculates the residual error

** NOTE: Here the fit match much better if the Actual torqe is used instead of the setpont torqe** 

Now we have a mechanical model of the system and can estimate control paarmeters

## Caluclate control params

We assume:
`tau=J*dw/dt+B*w+tc*sign(w)`

Design a second order PI with target bandwindth f_bw  (100Hz) and damping ζ = 1 (criticall)
```
Kp= (2*ζ*alpha)*J-B
Ki = J*alpha²
alpha = 2*pi*f_bw

* ζ damping, critcally damed ζ=1
* alpha: target target angular bandwitdh = 2*pi*f_bw
*f_bw:  target bandwidth (100Hz for velocity loop)
```

# CSV 
1. Identify CSV loop first order system
2. Calculate control params

## 1. Identify CSV system

1. Get data w_cmd, w_act
2. The velocity signal w is filtered and smoothed and then differntiated to calculate the acceleration `dw/dt`.
3. The velocity loop behaves like a first order system with model
```
w*Tv*dw/dt=Kv*w_cmd

or w_act(s)/w_cmd(s)=Kv/(1-Tv)

* w: velocity [rad/s]
* Tv: Time constant [s]
* Kv: velocity gain [] (ideally 1 if properlly tuned)
```
4. Solve least squares and finds best Kv, Tv and calculates the residual error

If Ki is needed it can be converted:

Ki=Kv/Tv

## 2. Claculate control params
The identified Kv and Tv (or Ki) are used to estimate a good parameters.

This step assumes that the velocity loop is stable and fast
`theta(s) -> PI -> w_cmd(s)-> inner velo PI loop -> v(s)`

For a critically damped 2nd order design with target bandwidth f_bw (10-20Hz) the PI gains are given by:
```
Kp = 2*ζ*alpha / Kv
Ki =  alpha²/Kv
alpha = 2*pi*f_bw

* ζ damping, critcally damed ζ=1
* alpha: target target angular bandwitdh = 2*pi*f_bw
*f_bw:  target bandwidth (10-20hz for position loop)
```

# More aggressive or less
Tuned parameters are moderate-conservative — tuned for a critically damped (ζ = 1) response at the chosen bandwidth (e.g. 100 Hz).
If you want it more aggressive, raise the target bandwidth (f_bw) or reduce ζ (≈ 0.7);
for smoother/stabler, lower f_bw (50–70 Hz) in te follwoing code:
```
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
```

# Units of K_p and K_i for Velocity loop (CST mode)
  
For the velocity loop (CST outer loop):

`tau(t) = K_p * (omega_ref - omega) + K_i *integral(omega_ref - omega)dt`
 
Given:
* tau = torque command [N·m]
* omega = angular velocity [rad/s]
* K_p proportional gain [N·m / (rad/s)]
* K_i integral gain [N·m / (rad) = (N·m · s) / rad]

Or more intuitively: 
K_p gives how many N·m per rad/s of speed error,
K_i gives how many N·m per integrated rad of speed error.
 
If your torque command is scaled differently (e.g., percentage, drive units), multiply/divide accordingly.
 

# Units of K_p and K_i for Position loop (CSV mode)
 
For the position loop (CSV outer loop), where command is velocity [rad/s] and output is position [rad]:
```
e(t)=theta_ref()t-theta(t)
Vcmd(t) = K_p*e(t) + K_i * integral(e(t))dt +k_d*derivate(e(t)/dt)

* Vcmd velocity command [rad/s]
* e(t) position error (theta_ref(t)-theta(t))
* theta position [rad]
* K_p proportional gain [(rad/s) / rad = 1/s]
* K_i integral gain [(rad/s) / (rad·s) = 1/s²]
```
# Conversion between Ki and Ti (for both CSV andd CST mode )
 T_i = K_p/K_i

where T_i is the integral time constant (in seconds).
Smaller T_i → faster integration (more aggressive); larger T_i → slower.

#### EL72xx PVs
CST:
```
c6025a-08:m1s000-Drv01-TrgDS402Ena
c6025a-08:m1s000-Drv01-Cmd-RB
c6025a-08:m1s000-Drv01-Trq-RB
c6025a-08:m1s000-Enc01-PosAct
c6025a-08:m1s000-Drv01-VelAct
c6025a-08:m1s000-Drv01-TrqAct
c6025a-08:m1s000-Drv01-VolAct
c6025a-08:m1s000-Drv02-TmpAct
c6025a-08:m1s000-NxtObjId
c6025a-08:m1s000-PrvObjId
c6025a-08:m1s000-Drv01-Cmd
c6025a-08:m1s000-Drv01-Trq
c6025a-08:m1s000-Drv01-WrnAlrm
c6025a-08:m1s000-Drv01-ErrAlrm
c6025a-08:m1s000-Tp01-BI01
c6025a-08:m1s000-Tp01-BI02
c6025a-08:m1s000-Enc01-NotValid
c6025a-08:m1s000-Online
c6025a-08:m1s000-Operational
c6025a-08:m1s000-Alstate-Init
c6025a-08:m1s000-Alstate-Preop
c6025a-08:m1s000-Alstate-Safeop
c6025a-08:m1s000-Alstate-Op
c6025a-08:m1s000-EntryCntr
c6025a-08:m1s000-Stat
c6025a-08:m1s000-One
c6025a-08:m1s000-Zero
c6025a-08:m1s000-Drv01-Stat
c6025a-08:m1s000-Tp01-Stat
c6025a-08:m1s000-Enc01-Stat
c6025a-08:m1s000-Drv01-TrgDS402Ena
c6025a-08:m1s000-Drv01-TrgDS402Dis
```
CSV:
```
c6025a-08:m1s000-Drv01-Cmd-RB
c6025a-08:m1s000-Drv01-Spd-RB
c6025a-08:m1s000-Enc01-PosAct
c6025a-08:m1s000-Drv01-VelAct
c6025a-08:m1s000-Drv01-TrqAct
c6025a-08:m1s000-Drv01-VolAct
c6025a-08:m1s000-Drv01-TrqOff-RB
c6025a-08:m1s000-NxtObjId
c6025a-08:m1s000-PrvObjId
c6025a-08:m1s000-Drv01-Cmd
c6025a-08:m1s000-Drv01-Spd
c6025a-08:m1s000-Drv01-TrqOff
c6025a-08:m1s000-Drv01-WrnAlrm
c6025a-08:m1s000-Drv01-ErrAlrm
c6025a-08:m1s000-Drv01-BI01
c6025a-08:m1s000-Drv01-BI02
c6025a-08:m1s000-Drv01-STO01
c6025a-08:m1s000-Tp01-BI01
c6025a-08:m1s000-Tp01-BI02
c6025a-08:m1s000-Online
c6025a-08:m1s000-Operational
c6025a-08:m1s000-Alstate-Init
c6025a-08:m1s000-Alstate-Preop
c6025a-08:m1s000-Alstate-Safeop
c6025a-08:m1s000-Alstate-Op
c6025a-08:m1s000-EntryCntr
c6025a-08:m1s000-Stat
c6025a-08:m1s000-One
c6025a-08:m1s000-Zero
c6025a-08:m1s000-Drv01-Stat
c6025a-08:m1s000-Drv01-InfoData2
c6025a-08:m1s000-Tp01-Stat
c6025a-08:m1s000-Stat_
c6025a-08:m1s000-Drv01-TrgDS402Ena
c6025a-08:m1s000-Drv01-TrgDS402Dis
```
