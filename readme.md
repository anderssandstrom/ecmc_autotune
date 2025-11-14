# ECMC Autotune GUI

This repository contains the PyQt-based GUI and helper scripts for collecting stepped-sine data, logging EPICS PVs, analyzing bode responses, and recommending tuning parameters for the CST velocity loop.

## Requirements

- Python 3.7+
- EPICS CA client libraries accessible via PyEPICS (configure `EPICS_CA_*` environment variables as usual).
- Python packages:
  - numpy
  - matplotlib
  - PyQt5
  - pyepics

## Initial setup

1. Ensure your terminal environment exports the correct EPICS variables (`EPICS_CA_ADDR_LIST`, `EPICS_CA_AUTO_ADDR_LIST`, etc.) so the GUI can connect to your PVs.
2. Install the Python dependencies, e.g.:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install numpy matplotlib pyqt5 pyepics
   ```
3. From the repo root, launch the GUI:
   ```bash
   python3 autotune/main.py
   ```

## Using the GUI

### PV Settings tab
- Set the common prefix or leave empty to use fully qualified PV names.
- Configure the setpoint (`SP`), readback (`SP_RBV`), and the actual signal (`ACT`) used for bode fitting.
- Use the suggestion list on the right to drag and drop common PV suffixes into the fields, or double-click to insert.
- Additional PVs can be listed (one per line) in the "Extra log PVs" box; each entry will be logged for later analysis.

### Excitation tab
- Configure the stepped-sine sweep: sample rate, start/stop frequency, amplitude, and settle/measurement cycles.
- Before a measurement starts you can review the generated excitation sequence in a preview dialog.

### Analysis tab
- Tune bode analysis parameters such as delay compensation, block length, frequency range, and required `R2` thresholds.

### Mechanical tab
- Configure mechanical model fitting options and the target bandwidth / damping used for PI suggestions.

## Running a measurement
1. Fill out the desired settings on all tabs.
2. Click **Run Measurement**.
   - The GUI verifies PV connectivity, shows you the excitation preview for approval, then logs all configured PVs while sending the command sequence.
   - Progress is displayed in the progress bar; you can abort mid-run using the **Abort** button, which restores the original setpoint.
3. After capture, the GUI resamples the data, computes the bode response, fits the mechanical model, and displays:
   - Inline Bode magnitude/phase plots (with a button to pop out a full window).
   - Command vs. actual time traces (also expandable).
   - Mechanical fit results and suggested PI gains in the log pane.
4. Logs are saved to the path shown at the top; change it to store different runs.

## Reanalyzing a log
- Set the log path, adjust analysis/mechanical parameters as needed, and click **Reanalyze Log** to re-process existing `.pkl` files without repeating the excitation.

## Scripted usage
The GUI uses the modules under `autotune/` (`pipeline.py`, `epics_logger.py`, `analyze.py`, etc.). You can import `autotune.pipeline` in your own scripts and call:
```python
from autotune import pipeline

pv = pipeline.PVSettings(prefix="c6025a-08:m1s000-", sp="Drv01-Trq", act="Drv01-VelAct")
exc = pipeline.ExcitationSettings(...)
analysis = pipeline.AnalysisSettings(...)
mech = pipeline.MechanicalSettings(...)
result = pipeline.run_measurement(pv, exc, analysis, mech, log_filename="data.pkl")
```

## Notes
- The measurement sends commands directly to the drive SP PV; ensure the motor is free to move and safety interlocks are handled externally.
- Setpoint value is restored after each run (even after an abort).
- The GUI operates entirely client-side; no PV writes are performed until you approve the excitation preview.

