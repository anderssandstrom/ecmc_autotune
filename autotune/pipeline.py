import time
from pathlib import Path

import numpy as np

try:  # Allow running as script from the autotune directory
    from . import analyze, epics_logger, excite_sine
except ImportError:  # pragma: no cover - fallback for direct execution
    import analyze  # type: ignore
    import epics_logger  # type: ignore
    import excite_sine  # type: ignore


DEFAULT_MODE = "cst_velocity"
MEASUREMENT_MODES = {
    "cst_velocity": {"label": "CST velocity loop tuning", "supports_mechanical": True},
    "csv_velocity_bode": {"label": "CSV closed loop bode", "supports_mechanical": False},
    "csv_position_tune": {"label": "CSV closed loop position tune", "supports_mechanical": False},
    "generic": {"label": "Generic", "supports_mechanical": False},
    "logger": {"label": "Logger", "supports_mechanical": False},
}


class PVSettings(object):
    """Definition of the PVs used during excitation/logging."""

    def __init__(
        self,
        prefix=None,
        sp="Drv01-Trq",
        sp_rbv="Drv01-TrqAct",
        act="Drv01-VelAct",
        extra_logs=None,
        prefix_p="c6025a-08:",
        prefix_r="m1s000-",
    ):
        if prefix is not None:
            prefix_p = prefix or ""
            prefix_r = ""
        self.prefix_p = prefix_p or ""
        self.prefix_r = prefix_r or ""
        self.prefix = f"{self.prefix_p}{self.prefix_r}"
        self.sp = sp
        self.sp_rbv = sp_rbv
        self.act = act
        self.extra_logs = extra_logs or {}

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
            ("ACT", "act"),
        ):
            pv_name = self._normalize(getattr(self, attr, ""))
            if pv_name:
                mapping[key] = pv_name
        for key, value in self.extra_logs.items():
            norm = self._normalize(value)
            key = (key or "").strip()
            if not norm or not key or key in mapping:
                continue
            mapping[key] = norm
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
        mode=DEFAULT_MODE,
        position_pid=None,
    ):
        self.t = np.asarray(t)
        self.values_by_pv = {k: np.asarray(v) for k, v in values_by_pv.items()}
        self.bode = bode
        self.command_key = command_key
        self.response_key = response_key
        self.log_file = log_file
        self.segments = segments
        self.mechanical = mechanical
        self.mode = _normalize_mode(mode)
        self.position_pid = position_pid


def run_measurement(
    pv,
    excitation,
    analysis,
    mechanical,
    log_filename=None,
    log_fn=None,
    progress_fn=None,
    should_abort=None,
    mode=DEFAULT_MODE,
):
    mode = _normalize_mode(mode)
    pvs = pv.to_pv_map()
    if "SP" not in pvs:
        raise ValueError("SP PV must be defined")
    if "ACT" not in pvs:
        raise ValueError("ACT PV must be defined for analysis")
    _log(log_fn, "Using PV map: %s" % ", ".join(f"{k}={v}" for k, v in sorted(pvs.items())))

    if should_abort and should_abort():
        raise RuntimeError("Measurement aborted before start.")

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
    _verify_pv_connections(mylogger, log_fn)
    sp_pv = mylogger.pvs.get("SP")
    init_val = None
    if not log_only:
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
    aborted = False
    try:
        for idx, value in enumerate(command):
            if should_abort and should_abort():
                aborted = True
                _log(log_fn, "Abort requested; stopping command stream...")
                break
            if not log_only and sp_pv is not None:
                sp_pv.put(float(value), wait=False)
            t_next += dt
            if progress_fn:
                progress_fn(idx / total)
            delay = t_next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        if not aborted:
            time.sleep(1.0)
    finally:
        _log(log_fn, "Excitation finished, stopping monitors…")
        mylogger.stop_monitors()
        if progress_fn:
            progress_fn(1.0)
        if not log_only and sp_pv is not None and init_val is not None:
            try:
                sp_pv.put(float(init_val), wait=False)
                _log(log_fn, "Restored SP to initial value.")
            except Exception as exc:
                _log(log_fn, f"Warning: failed to restore SP PV: {exc}")
    if aborted:
        raise RuntimeError("Measurement aborted by user.")

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
    result = _analyze_with_fallback(t, vals_by_pv, excitation.fs, analysis, mechanical, log_path, mode)
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
        mode=mode,
        position_pid=result.position_pid,
    )


def reanalyze_log(
    log_filename,
    analysis,
    mechanical,
    log_fn=None,
    mode=DEFAULT_MODE,
):
    mode = _normalize_mode(mode)
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
    return _analyze_with_fallback(t, vals_by_pv, fs, analysis, mechanical, log_path, mode)


def _analyze(
    t,
    vals_by_pv,
    fs,
    analysis,
    mechanical,
    log_path,
    mode,
):
    mode = _normalize_mode(mode)
    cmd_key = "SP_RBV" if "SP_RBV" in vals_by_pv else ("SP" if "SP" in vals_by_pv else None)
    resp_key = None
    for candidate in ("ACT", "VEL_ACT", "POS_ACT", "TRQ_ACT"):
        if candidate in vals_by_pv:
            resp_key = candidate
            break
    if cmd_key is None or resp_key is None:
        raise RuntimeError("Both command (SP / SP_RBV) and response (ACT) signals are required")

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

    bode_payload = {
        "freq": bode_results.get("F", np.array([])),
        "mag_db": bode_results.get("mag_db", np.array([])),
        "phase": bode_results.get("phase_deg", np.array([])),
        "phase_comp": bode_results.get("phase_comp_deg", np.array([])),
        "r2_u": bode_results.get("R2_u", np.array([])),
        "r2_y": bode_results.get("R2_y", np.array([])),
    }

    mechanical_result = None
    if mode in ("cst_velocity", "csv_position_tune"):
        mechanical_result = _fit_mechanical(t, vals_by_pv, cmd_key, resp_key, fs, mechanical)
    position_pid = None
    if mode == "csv_position_tune":
        position_pid = _position_pid_from_bode(bode_payload, mechanical)

    return RunResult(
        t=np.asarray(t),
        values_by_pv={k: np.asarray(v) for k, v in vals_by_pv.items()},
        bode=bode_payload,
        command_key=cmd_key,
        response_key=resp_key,
        log_file=str(log_path) if log_path else None,
        segments=segments,
        mechanical=mechanical_result,
        mode=mode,
        position_pid=position_pid,
    )


def _analyze_with_fallback(t, vals_by_pv, fs, analysis, mechanical, log_path, mode):
    mode = _normalize_mode(mode)
    try:
        return _analyze(t, vals_by_pv, fs, analysis, mechanical, log_path, mode)
    except RuntimeError as exc:
        if mode != "logger":
            raise
        return _basic_run_result(t, vals_by_pv, log_path, mode)


def _basic_run_result(t, vals_by_pv, log_path, mode):
    if not vals_by_pv:
        raise RuntimeError("No PV data available for logging")
    keys = list(vals_by_pv.keys())
    cmd_key = keys[0]
    resp_key = keys[1] if len(keys) > 1 else keys[0]
    bode_payload = {
        "freq": np.array([]),
        "mag_db": np.array([]),
        "phase": np.array([]),
        "phase_comp": np.array([]),
        "r2_u": np.array([]),
        "r2_y": np.array([]),
    }
    return RunResult(
        t=np.asarray(t),
        values_by_pv={k: np.asarray(v) for k, v in vals_by_pv.items()},
        bode=bode_payload,
        command_key=cmd_key,
        response_key=resp_key,
        log_file=str(log_path) if log_path else None,
        segments=None,
        mechanical=None,
        mode=mode,
        position_pid=None,
    )


def _fit_mechanical(t, vals_by_pv, cmd_key, resp_key, fs, mechanical):
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
    return mechanical_result


def _position_pid_from_bode(bode_payload, mechanical):
    if mechanical is None:
        return None
    try:
        bw = float(mechanical.pi_bandwidth)
        zeta = float(mechanical.pi_zeta)
    except Exception:
        return None
    if not np.isfinite(bw) or bw <= 0.0:
        return None
    if not np.isfinite(zeta) or zeta <= 0.0:
        zeta = 1.0
    freq = np.asarray(bode_payload.get("freq", np.array([])), float)
    mag_db = np.asarray(bode_payload.get("mag_db", np.array([])), float)
    mask = (freq > 0) & np.isfinite(freq) & np.isfinite(mag_db)
    freq = freq[mask]
    mag_db = mag_db[mask]
    if freq.size == 0:
        return _position_pid_from_targets(bw, zeta)
    mag = 10.0 ** (mag_db / 20.0)
    w = 2.0 * np.pi * freq
    points = min(5, mag.size)
    k_est = float(np.median(mag[:points] * w[:points]))
    if not np.isfinite(k_est) or k_est <= 0.0:
        return _position_pid_from_targets(bw, zeta)
    wn = 2.0 * np.pi * bw
    kp = (2.0 * zeta * wn) / k_est
    ki = (wn * wn) / k_est
    kd = 0.0
    return {
        "kp": float(max(kp, 0.0)),
        "ki": float(max(ki, 0.0)),
        "kd": float(max(kd, 0.0)),
        "ti": (kp / ki) if ki > 1e-12 else float("inf"),
        "target_bw_hz": bw,
        "zeta": zeta,
        "gain_from_bode": True,
        "plant_gain": k_est,
    }


def _position_pid_from_targets(bw, zeta):
    wn = 2.0 * np.pi * bw
    kp = 2.0 * zeta * wn
    ki = wn * wn
    kd = 0.0
    return {
        "kp": float(max(kp, 0.0)),
        "ki": float(max(ki, 0.0)),
        "kd": float(max(kd, 0.0)),
        "ti": (kp / ki) if ki > 1e-12 else float("inf"),
        "target_bw_hz": bw,
        "zeta": zeta,
        "gain_from_bode": False,
        "plant_gain": None,
    }


def _infer_sample_rate(t):
    dt = np.diff(np.asarray(t, float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Unable to infer sampling rate from time vector")
    return 1.0 / float(np.median(dt))


def _log(log_fn, message):
    if log_fn:
        log_fn(message)


def _verify_pv_connections(mylogger, log_fn, timeout=2.0):
    missing = []
    for name, pv in mylogger.pvs.items():
        if pv is None:
            missing.append(name)
            continue
        try:
            ok = pv.wait_for_connection(timeout=timeout)
        except Exception:
            ok = False
        if not ok:
            missing.append(name)
    if missing:
        raise RuntimeError("Failed to connect to PVs: %s" % ", ".join(sorted(missing)))
    _log(log_fn, "PV connectivity check passed.")


def available_modes():
    return dict(MEASUREMENT_MODES)


def mode_supports_mechanical(mode):
    return _mode_supports_mechanical(_normalize_mode(mode))


def _normalize_mode(mode):
    if isinstance(mode, str):
        key = mode.strip().lower()
    else:
        key = getattr(mode, "value", None)
        if key is None and mode is not None:
            key = str(mode).strip().lower()
    if key in MEASUREMENT_MODES:
        return key
    return DEFAULT_MODE if key is None else "generic"


def _mode_supports_mechanical(mode):
    info = MEASUREMENT_MODES.get(mode)
    if not info:
        return False
    return bool(info.get("supports_mechanical"))
