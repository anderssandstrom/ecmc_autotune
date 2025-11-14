import time
from pathlib import Path

import numpy as np

try:  # Allow running as script from the autotune directory
    from . import analyze, epics_logger, excite_sine
except ImportError:  # pragma: no cover - fallback for direct execution
    import analyze  # type: ignore
    import epics_logger  # type: ignore
    import excite_sine  # type: ignore


class PVSettings(object):
    """Definition of the PVs used during excitation/logging."""

    def __init__(
        self,
        prefix="c6025a-08:m1s000-",
        sp="Drv01-Trq",
        sp_rbv="Drv01-TrqAct",
        vel_act="Drv01-VelAct",
        pos_act="Enc01-PosAct",
        trq_act="Drv01-TrqAct",
    ):
        self.prefix = prefix
        self.sp = sp
        self.sp_rbv = sp_rbv
        self.vel_act = vel_act
        self.pos_act = pos_act
        self.trq_act = trq_act

    def _normalize(self, value):
        value = (value or "").strip()
        if not value:
            return None
        if value.startswith(self.prefix) or ":" in value:
            return value
        return f"{self.prefix}{value}"

    def to_pv_map(self):
        mapping = {}
        for key, attr in (
            ("SP", "sp"),
            ("SP_RBV", "sp_rbv"),
            ("VEL_ACT", "vel_act"),
            ("POS_ACT", "pos_act"),
            ("TRQ_ACT", "trq_act"),
        ):
            pv_name = self._normalize(getattr(self, attr, ""))
            if pv_name:
                mapping[key] = pv_name
        return mapping


class ExcitationSettings(object):
    def __init__(
        self,
        fs=1000.0,
        f_start=10.0,
        f_stop=400.0,
        n_points=20,
        amp=0.01,
        n_settle=10,
        n_meas=50,
        transition_min_s=1.0,
        transition_frac=0.0,
        edge_taper_cycles=1.0,
    ):
        self.fs = fs
        self.f_start = f_start
        self.f_stop = f_stop
        self.n_points = n_points
        self.amp = amp
        self.n_settle = n_settle
        self.n_meas = n_meas
        self.transition_min_s = transition_min_s
        self.transition_frac = transition_frac
        self.edge_taper_cycles = edge_taper_cycles


class AnalysisSettings(object):
    def __init__(
        self,
        tau_ms=2.5,
        block_len_s=0.5,
        overlap=0.5,
        fmin=0.5,
        fmax=500.0,
        freq_tolerance=0.02,
        settle_frac=0.3,
        r2_min=0.05,
        sample_hz=None,
    ):
        self.tau_ms = tau_ms
        self.block_len_s = block_len_s
        self.overlap = overlap
        self.fmin = fmin
        self.fmax = fmax
        self.freq_tolerance = freq_tolerance
        self.settle_frac = settle_frac
        self.r2_min = r2_min
        self.sample_hz = sample_hz


class MechanicalSettings(object):
    def __init__(
        self,
        motor_rated_trq=0.5,
        torque_scale=None,
        velocity_scale=1.0,
        smooth_hz=150.0,
        deriv_hz=120.0,
        vel_deadband=1e-3,
        pi_bandwidth=100.0,
        pi_zeta=1.0,
    ):
        self.motor_rated_trq = motor_rated_trq
        self.torque_scale = torque_scale
        self.velocity_scale = velocity_scale
        self.smooth_hz = smooth_hz
        self.deriv_hz = deriv_hz
        self.vel_deadband = vel_deadband
        self.pi_bandwidth = pi_bandwidth
        self.pi_zeta = pi_zeta

    def torque_multiplier(self):
        if self.torque_scale is not None:
            return float(self.torque_scale)
        return float(self.motor_rated_trq) / 100.0


class RunResult(object):
    def __init__(
        self,
        t,
        values_by_pv,
        bode,
        command_key,
        response_key,
        log_file=None,
        segments=None,
        mechanical=None,
    ):
        self.t = np.asarray(t)
        self.values_by_pv = {k: np.asarray(v) for k, v in values_by_pv.items()}
        self.bode = bode
        self.command_key = command_key
        self.response_key = response_key
        self.log_file = log_file
        self.segments = segments
        self.mechanical = mechanical


def run_measurement(
    pv,
    excitation,
    analysis,
    mechanical,
    log_filename=None,
    log_fn=None,
    progress_fn=None,
):
    pvs = pv.to_pv_map()
    if "SP" not in pvs:
        raise ValueError("SP PV must be defined")
    _log(log_fn, "Using PV map: %s" % ", ".join(f"{k}={v}" for k, v in sorted(pvs.items())))

    _log(log_fn, "Generating stepped-sine command sequence…")
    seq = excite_sine.generate(
        excitation.fs,
        excitation.f_start,
        excitation.f_stop,
        excitation.n_points,
        excitation.amp,
        excitation.n_settle,
        excitation.n_meas,
        transition_min_s=excitation.transition_min_s,
        transition_frac=excitation.transition_frac,
        edge_taper_cycles=excitation.edge_taper_cycles,
        preview=False,
    )
    _, command, *_ = seq
    duration = len(command) / float(excitation.fs)
    _log(log_fn, f"Command length: {duration:.1f} s ({len(command)} samples)")

    mylogger = epics_logger.logger(Ts=1.0 / float(excitation.fs), pvs=pvs)
    sp_pv = mylogger.pvs.get("SP")
    _log(log_fn, str(sp_pv))
    
    if sp_pv is None:
        raise RuntimeError("SP PV handle is not available")

    init_val = sp_pv.get(timeout=2.0)
    if init_val is None:
        raise RuntimeError("Failed to read the setpoint PV (%s)" % pvs["SP"])
    sp_pv.put(init_val, wait=True)

    _log(log_fn, "Starting EPICS monitors…")
    mylogger.start_monitors()
    dt = 1.0 / float(excitation.fs)
    total = len(command)
    t_next = time.monotonic()
    for idx, value in enumerate(command):
        sp_pv.put(float(value), wait=False)
        t_next += dt
        if progress_fn:
            progress_fn(idx / total)
        delay = t_next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
    time.sleep(1.0)
    _log(log_fn, "Excitation finished, stopping monitors…")
    mylogger.stop_monitors()
    if progress_fn:
        progress_fn(1.0)

    log_path = None
    if log_filename:
        log_path = Path(log_filename)
        if log_path.suffix == "":
            log_path = log_path.with_suffix(".pkl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        mylogger.save_log(str(log_path))
        _log(log_fn, f"Log saved to {log_path}")

    _log(log_fn, "Running analysis on captured data…")
    t, vals_by_pv, *_ = mylogger.resample_to_common_time_base(fill="extrapolate", time_range="intersection")
    result = _analyze(t, vals_by_pv, excitation.fs, analysis, mechanical, log_path)
    if log_path is None:
        return result
    return RunResult(
        t=result.t,
        values_by_pv=result.values_by_pv,
        bode=result.bode,
        command_key=result.command_key,
        response_key=result.response_key,
        log_file=str(log_path),
        segments=result.segments,
        mechanical=result.mechanical,
    )


def reanalyze_log(
    log_filename,
    analysis,
    mechanical,
    log_fn=None,
):
    if not log_filename:
        raise ValueError("Log file must be provided for reanalysis")
    log_path = Path(log_filename)
    if log_path.suffix == "":
        log_path = log_path.with_suffix(".pkl")
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    _log(log_fn, f"Loading log {log_path}…")
    mylogger = epics_logger.logger()
    mylogger.load_log(str(log_path))
    t, vals_by_pv, *_ = mylogger.resample_to_common_time_base(fill="extrapolate", time_range="intersection")
    fs = analysis.sample_hz or _infer_sample_rate(t)
    _log(log_fn, f"Using sample frequency {fs:.2f} Hz for analysis")
    return _analyze(t, vals_by_pv, fs, analysis, mechanical, log_path)


def _analyze(
    t,
    vals_by_pv,
    fs,
    analysis,
    mechanical,
    log_path,
):
    cmd_key = "SP_RBV" if "SP_RBV" in vals_by_pv else ("SP" if "SP" in vals_by_pv else None)
    resp_key = "VEL_ACT" if "VEL_ACT" in vals_by_pv else None
    if cmd_key is None or resp_key is None:
        raise RuntimeError("Both command (SP / SP_RBV) and response (VEL_ACT) signals are required")

    bode_obj = analyze.bode(
        t,
        vals_by_pv[cmd_key],
        vals_by_pv[resp_key],
        fs,
        tau_ms=analysis.tau_ms,
        block_len_s=analysis.block_len_s,
        overlap=analysis.overlap,
        fmin=analysis.fmin,
        fmax=analysis.fmax,
        freq_tolerance=analysis.freq_tolerance,
        settle_frac=analysis.settle_frac,
        r2_min=analysis.r2_min,
    )
    bode_results = bode_obj.analyze_stepped_sine(
        t,
        vals_by_pv[cmd_key],
        vals_by_pv[resp_key],
        fs,
        tau_ms=analysis.tau_ms,
        block_len_s=analysis.block_len_s,
        overlap=analysis.overlap,
        fmin=analysis.fmin,
        fmax=analysis.fmax,
        freq_tolerance=analysis.freq_tolerance,
        settle_frac=analysis.settle_frac,
        r2_min=analysis.r2_min,
    )
    segments = bode_obj.getSegments()

    mechanical_result = None
    trq_key = "TRQ_ACT" if "TRQ_ACT" in vals_by_pv else cmd_key
    try:
        mechanical_result = analyze.fit_mechanical_model(
            t,
            vals_by_pv[trq_key],
            vals_by_pv[resp_key],
            fs=fs,
            torque_scale=mechanical.torque_multiplier(),
            vel_scale=mechanical.velocity_scale,
            smooth_hz=mechanical.smooth_hz,
            deriv_hz=mechanical.deriv_hz,
            vel_deadband=mechanical.vel_deadband,
        )
        kp, ki = analyze.velocity_pi_from_JB(
            mechanical_result["J"],
            mechanical_result["B"],
            f_bw=mechanical.pi_bandwidth,
            zeta=mechanical.pi_zeta,
        )
        mechanical_result["kp"] = kp
        mechanical_result["ki"] = ki
        mechanical_result["ti"] = (kp / ki) if ki > 1e-12 else float("inf")
    except Exception:
        mechanical_result = None

    bode_payload = {
        "freq": bode_results.get("F", np.array([])),
        "mag_db": bode_results.get("mag_db", np.array([])),
        "phase": bode_results.get("phase_deg", np.array([])),
        "phase_comp": bode_results.get("phase_comp_deg", np.array([])),
        "r2_u": bode_results.get("R2_u", np.array([])),
        "r2_y": bode_results.get("R2_y", np.array([])),
    }

    return RunResult(
        t=np.asarray(t),
        values_by_pv={k: np.asarray(v) for k, v in vals_by_pv.items()},
        bode=bode_payload,
        command_key=cmd_key,
        response_key=resp_key,
        log_file=str(log_path) if log_path else None,
        segments=segments,
        mechanical=mechanical_result,
    )


def _infer_sample_rate(t):
    dt = np.diff(np.asarray(t, float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Unable to infer sampling rate from time vector")
    return 1.0 / float(np.median(dt))


def _log(log_fn, message):
    if log_fn:
        log_fn(message)
