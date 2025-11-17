# Test Ex72xx
```
# Defaults EP7211 AM8121 with the alu inertia:
0.5Nm 0.5A at 4000mA
0x8010:14, rwrwrw, uint32, 32 bit, "Velocity loop integral time" 150 mA/(rad/s)
0x8010:15, rwrwrw, uint32, 32 bit, "Velocity loop proportional gain" 37 0.1s


# Kp
150mA=150/4000*0.5=0.02Nm
150 mA/(rad/s) -> 0.02Nm/(rad/s)

# Ti
37 0.1s =3.7s

Mechanical fit using TRQ_ACT:
  J  = 0.0001067 [N·m·s²/rad]
  B  = 0.001409 [N·m·s/rad]
  Tc = 0.001068 [N·m]
  residual RMS = 0.0246068 (96.0% variance explained)

Suggested CST velocity PI (f_bw=100 Hz, ζ=1):

Kp = 0.1326 [N·m/(rad/s)] (=1000mA/(rad/s))
Ki = 42.11 [N·m/rad]
Ti = 0.00315s  = 0.031 (0.1s)
```
ethercat sdos -m1 -p0 
ethercat sdos -m1 -p0 upload 0x8000  0x8010:14, rwrwrw, uint32, 32 bit, "Velocity loop integral time"
ethercat sdos -m1 -p0 upload  0x8010 0x14
ethercat -m1 -p0 upload  0x8010 0x14
ethercat -m1 -p0 upload  0x8010 0x15
# test auto tune params but less aggressive Ti
ethercat -m1 -p0 download  0x8010 0x14 1280
ethercat -m1 -p0 download  0x8010 0x15 10