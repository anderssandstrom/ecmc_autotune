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

### Flow tab
- Select the operating mode (CST velocity loop tuning, CSV closed-loop bode, CSV position loop tune, **CSP closed loop bode**, or logger-only).
- A flowchart illustrates the capture/analysis path and updates to match the chosen mode.
- Inline text plus a “mode info” button summarize what each mode logs, how scaling is applied, and whether mechanical identification/PID suggestions are available.

### PV Settings tab
- Configure the two-part PV prefix (`P` + `R`) or leave blank to use absolute PV names.
- Set the command (`SP`), readback (`SP_RBV`), response (`ACT`), and any extra PVs to sample. Extra entries are logged for plotting and can be selected for the “Extra PVs” plots.
- Specify torque, velocity, and position scales (plus motor rated torque). Commands are divided by their scale; logged readbacks are multiplied so you can work in Nm, rad/s, or rad. CST uses torque/velocity, CSV* uses velocity, and CSP/CSV position modes also use position scaling. Defaults of `1.0` let you work in native drive units.

### Excitation tab
- Configure the stepped-sine sweep: sampling rate, frequency span, amplitude, settle/measurement cycles, and transition rules.
- A preview button displays the generated command and instantaneous frequency (dual y-axes) before you start a run.

### Analysis tab
- Adjust bode-processing settings (delay compensation, block length, frequency window, `R²` limits, etc.) across two columns for easier scanning.
- The right-most “Mechanical*” group controls smoothing/derivative filters, velocity deadband, and target PI bandwidth/zeta. It is automatically greyed out when the active mode does not perform mechanical fitting.

### Results tab
- Enter the target closed-loop bandwidth and damping used when generating velocity PI (CST) or position PID (CSV position tune) suggestions (controls appear in the Analysis tab’s mechanical group).
- The “Latest suggestions” table lists Kp/Ki/Kd/Ti plus J/B/Tc/residual values; the **Clear results** button resets the history.
- The integrated bode/command plots sit above a compact log window (shortened to leave more room for the graphics).
- The “Extra PV plotting” panel lets you select additional PVs (including derived segment masks/frequency tracks) and open them in a separate window using the **Plot extra PVs** button. Segments and frequency traces always plot on the right y-axis for clarity.

### PV / Signals tab
- Choose which logged PVs populate the time-domain plots. Extra PVs (including identified segment boundaries and detected frequency tracks) can be graphed in separate windows without disturbing the embedded signal plot.

### File tab
- Manage log destinations independently of capture mode. The GUI stores every measurement—along with PV selections, excitation, analysis, and PID settings—in the `.pkl` metadata. Loading a log restores all tabs (including mode) to the exact configuration that produced it. The **Reanalyze Log** button lives here.

### Docs tab
- Contains detailed descriptions of every mode at all times. The flow-tab info button opens the relevant excerpt in a pop-up for quick reference.

## Running a measurement
1. Fill out the desired settings on all tabs.
2. Click **Run Measurement**.
   - The GUI verifies PV connectivity, shows you the excitation preview for approval, then logs all configured PVs while sending the command sequence.
   - Progress is displayed in the progress bar; you can abort mid-run using the **Abort** button, which restores the original setpoint.
3. After capture, the GUI resamples the data, computes the bode response, fits the mechanical model, and displays:
   - Inline Bode magnitude/phase plots (with a button to pop out a full window).
   - Command vs. actual time traces (also expandable).
   - Mechanical fit results and suggested PI gains in the log pane.
4. Logs (including metadata) are saved to the path shown on the File tab; change it to store different runs. Logger mode captures PVs without sending commands.

## Reanalyzing a log
- Set the log path, adjust analysis/mechanical/PID parameters as needed, and click **Reanalyze Log** to re-process existing `.pkl` files without repeating the excitation. The restored PV selections ensure reanalysis uses the same signals that were captured originally.

## Scripted usage
The GUI uses the modules under `autotune/` (`pipeline.py`, `epics_logger.py`, `analyze.py`, etc.). You can import `autotune.pipeline` in your own scripts and call:
```python
from autotune import pipeline

pv = pipeline.PVSettings(prefix_p="c6025a-08:", prefix_r="m1s000-", sp="Drv01-Trq", act="Drv01-VelAct")
exc = pipeline.ExcitationSettings(...)
analysis = pipeline.AnalysisSettings(...)
mech = pipeline.MechanicalSettings(...)
result = pipeline.run_measurement(pv, exc, analysis, mech, log_filename="data.pkl")
```

## Notes
- The measurement sends commands directly to the drive SP PV; ensure the motor is free to move and safety interlocks are handled externally.
- Setpoint value is restored after each run (even after an abort).
- The GUI operates entirely client-side; no PV writes are performed until you approve the excitation preview.
