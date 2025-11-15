import os
import sys
import traceback
from datetime import datetime

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

try:
    from . import pipeline, excite_sine
except ImportError:  # pragma: no cover
    import pipeline  # type: ignore
    import excite_sine  # type: ignore


PV_SUGGESTIONS = [
    "Enc01-PosAct",
    "Drv01-VelAct",
    "Drv01-TrqAct",
    "Drv01-Spd",
    "Drv01-Spd-RB",
]


class Worker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    log = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(float)

    def __init__(self, mode, kwargs):
        super().__init__()
        self.mode = mode
        self.kwargs = kwargs
        self._abort = False

    @QtCore.pyqtSlot()
    def run(self):
        try:
            if self.mode == "measure":
                result = pipeline.run_measurement(
                    progress_fn=self.progress.emit,
                    log_fn=self.log.emit,
                    should_abort=self.should_abort,
                    **self.kwargs,
                )
            elif self.mode == "reanalyze":
                result = pipeline.reanalyze_log(log_fn=self.log.emit, **self.kwargs)
            else:
                raise ValueError(f"Unsupported worker mode {self.mode}")
        except Exception as exc:
            message = str(exc)
            if isinstance(exc, RuntimeError) and "aborted" in message.lower():
                self.failed.emit(message)
            else:
                self.failed.emit(traceback.format_exc())
        else:
            self.finished.emit(result)

    def request_abort(self):
        self._abort = True

    def should_abort(self):
        thread = QtCore.QThread.currentThread()
        return self._abort or (thread.isInterruptionRequested() if thread else False)


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, rows=1, cols=1, **fig_kwargs):
        self.figure = Figure(**fig_kwargs)
        super().__init__(self.figure)
        self.axes = self.figure.subplots(rows, cols, squeeze=False)

    def clear(self):
        for row in self.axes:
            for ax in row:
                ax.cla()
        self.draw_idle()


class PVLineEdit(QtWidgets.QLineEdit):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasText():
            self.setText(event.mimeData().text().strip())
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class PVPlainTextEdit(QtWidgets.QPlainTextEdit):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasText():
            addition = event.mimeData().text().strip()
            if addition:
                text = self.toPlainText().strip()
                if text:
                    text += "\n" + addition
                else:
                    text = addition
                self.setPlainText(text)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class PVSuggestionList(QtWidgets.QListWidget):
    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.addItems(items)
        self.setDragEnabled(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

    def startDrag(self, supported_actions):
        item = self.currentItem()
        if item is None:
            return
        drag = QtGui.QDrag(self)
        mime = QtCore.QMimeData()
        mime.setText(item.text())
        drag.setMimeData(mime)
        drag.exec_(QtCore.Qt.CopyAction)


class AutotuneWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECMC Autotune")
        self.resize(1200, 800)
        self.worker_thread = None
        self.worker = None
        self.latest_result = None
        self.last_bode_data = None
        self.last_time_data = None
        self.last_analysis_settings = pipeline.AnalysisSettings()
        self._build_ui()

    # -----------------------
    # UI construction
    # -----------------------
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_pv_tab(), "PV Settings")
        tabs.addTab(self._build_excitation_tab(), "Excitation")
        tabs.addTab(self._build_analysis_tab(), "Analysis")
        tabs.addTab(self._build_mechanical_tab(), "Mechanical")
        layout.addWidget(tabs)

        log_path_row = QtWidgets.QHBoxLayout()
        self.log_path_edit = QtWidgets.QLineEdit("autotune/logs/latest.pkl")
        self.log_path_edit.setClearButtonEnabled(True)
        self._set_tooltip(self.log_path_edit, "Destination for captured logs or the file to reopen for analysis.")
        log_path_row.addWidget(QtWidgets.QLabel("Log file"))
        log_path_row.addWidget(self.log_path_edit)
        save_btn = QtWidgets.QPushButton("Browse Save…")
        save_btn.clicked.connect(lambda: self._browse_log(True))
        load_btn = QtWidgets.QPushButton("Browse Load…")
        load_btn.clicked.connect(lambda: self._browse_log(False))
        self._set_tooltip(save_btn, "Choose where to write the next acquisition log.")
        self._set_tooltip(load_btn, "Pick an existing log to reanalyze.")
        log_path_row.addWidget(save_btn)
        log_path_row.addWidget(load_btn)
        layout.addLayout(log_path_row)

        button_row = QtWidgets.QHBoxLayout()
        self.measure_btn = QtWidgets.QPushButton("Run Measurement")
        self.reanalyze_btn = QtWidgets.QPushButton("Reanalyze Log")
        self.measure_btn.clicked.connect(self._start_measurement)
        self.reanalyze_btn.clicked.connect(self._start_reanalysis)
        self._set_tooltip(self.measure_btn, "Excite the axis with the configured stepped-sine sequence and capture fresh data.")
        self._set_tooltip(self.reanalyze_btn, "Process a previously recorded log with the current analysis settings.")
        button_row.addWidget(self.measure_btn)
        button_row.addWidget(self.reanalyze_btn)
        self.abort_btn = QtWidgets.QPushButton("Abort")
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self._abort_worker)
        self._set_tooltip(self.abort_btn, "Stop the active measurement or analysis as soon as possible.")
        button_row.addWidget(self.abort_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self._set_tooltip(self.progress_bar, "Shows completion of the running measurement or analysis step.")
        layout.addWidget(self.progress_bar)

        plots_row = QtWidgets.QHBoxLayout()
        self.bode_group = QtWidgets.QGroupBox("Bode")
        bode_layout = QtWidgets.QVBoxLayout()
        self.bode_canvas = PlotCanvas(rows=2, cols=1, figsize=(5, 4), constrained_layout=True)
        bode_layout.addWidget(self.bode_canvas)
        self.bode_popup_btn = QtWidgets.QPushButton("Open Window")
        self.bode_popup_btn.clicked.connect(self._show_bode_popup)
        self._set_tooltip(self.bode_popup_btn, "Pop out the Bode plot into its own resizable window.")
        bode_layout.addWidget(self.bode_popup_btn)
        self.bode_group.setLayout(bode_layout)

        self.time_group = QtWidgets.QGroupBox("Signals")
        time_layout = QtWidgets.QVBoxLayout()
        self.time_canvas = PlotCanvas(rows=1, cols=1, figsize=(5, 4), constrained_layout=True)
        time_layout.addWidget(self.time_canvas)
        self.time_popup_btn = QtWidgets.QPushButton("Open Window")
        self.time_popup_btn.clicked.connect(self._show_time_popup)
        self._set_tooltip(self.time_popup_btn, "Pop out the captured time-domain signals into a separate window.")
        time_layout.addWidget(self.time_popup_btn)
        self.time_group.setLayout(time_layout)

        plots_row.addWidget(self.bode_group, 1)
        plots_row.addWidget(self.time_group, 1)
        layout.addLayout(plots_row, 1)

        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output, 1)

    def _build_pv_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        form_container = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(form_container)
        self.pv_prefix = self._line_edit("c6025a-08:m1s000-")
        self._set_tooltip(self.pv_prefix, "Prefix prepended to PV names when they do not include a full record (leave blank for absolute PVs).")
        self.pv_sp = self._line_edit("Drv01-Trq")
        self._set_tooltip(self.pv_sp, "Command PV written during excitation (e.g. torque setpoint).")
        self.pv_sp_rbv = self._line_edit("Drv01-TrqAct")
        self._set_tooltip(self.pv_sp_rbv, "Optional readback PV for the command; falls back to SP when blank.")
        self.pv_act = self._line_edit("Drv01-VelAct")
        self._set_tooltip(self.pv_act, "Measured plant response PV, typically velocity.")
        self.pv_extra = PVPlainTextEdit("")
        self.pv_extra.setPlaceholderText("One PV per line (optional NAME=PV). Empty line removes logging.")
        self._set_tooltip(self.pv_extra, "Add extra PVs to log during excitation. Enter NAME=PV pairs or raw PV names, one per line.")
        form.addRow("Prefix", self.pv_prefix)
        form.addRow("SP", self.pv_sp)
        form.addRow("SP_RBV", self.pv_sp_rbv)
        form.addRow("ACT PV", self.pv_act)
        form.addRow("Extra log PVs", self.pv_extra)
        layout.addWidget(form_container, 2)

        suggestion_box = QtWidgets.QGroupBox("Available PVs")
        suggestion_layout = QtWidgets.QVBoxLayout(suggestion_box)
        hint = QtWidgets.QLabel("Double-click to insert into focused field or drag text.")
        hint.setWordWrap(True)
        suggestion_layout.addWidget(hint)
        self.pv_list = PVSuggestionList(PV_SUGGESTIONS)
        self.pv_list.itemDoubleClicked.connect(lambda item: self._insert_pv_text(item.text()))
        self._set_tooltip(self.pv_list, "Drag or double-click to copy common PV names into the fields.")
        suggestion_layout.addWidget(self.pv_list)
        layout.addWidget(suggestion_box, 1)
        return widget

    def _build_excitation_tab(self):
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)
        self.ex_fs = self._line_edit("1000")
        self._set_tooltip(self.ex_fs, "Logger sampling frequency and excitation update rate in Hz.")
        self.ex_f_start = self._line_edit("10")
        self._set_tooltip(self.ex_f_start, "Lowest stepped-sine frequency to command.")
        self.ex_f_stop = self._line_edit("400")
        self._set_tooltip(self.ex_f_stop, "Highest stepped-sine frequency to command.")
        self.ex_points = self._line_edit("20")
        self._set_tooltip(self.ex_points, "Number of logarithmically spaced excitation points.")
        self.ex_amp = self._line_edit("0.01")
        self._set_tooltip(self.ex_amp, "Command amplitude in SP units.")
        self.ex_settle = self._line_edit("10")
        self._set_tooltip(self.ex_settle, "Cycles per frequency used to settle before logging.")
        self.ex_meas = self._line_edit("50")
        self._set_tooltip(self.ex_meas, "Cycles per frequency captured for analysis after settling.")
        self.ex_trans_min = self._line_edit("1.0")
        self._set_tooltip(self.ex_trans_min, "Minimum seconds to transition between frequencies.")
        self.ex_trans_frac = self._line_edit("0.0")
        self._set_tooltip(self.ex_trans_frac, "Fraction of settle time spent in transition ramps.")
        self.ex_taper = self._line_edit("1.0")
        self._set_tooltip(self.ex_taper, "Number of cycles used to taper the edges of the excitation waveform.")
        form.addRow("fs [Hz]", self.ex_fs)
        form.addRow("f start [Hz]", self.ex_f_start)
        form.addRow("f stop [Hz]", self.ex_f_stop)
        form.addRow("Points", self.ex_points)
        form.addRow("Amplitude", self.ex_amp)
        form.addRow("Settle cycles", self.ex_settle)
        form.addRow("Measure cycles", self.ex_meas)
        form.addRow("Transition min [s]", self.ex_trans_min)
        form.addRow("Transition frac", self.ex_trans_frac)
        form.addRow("Edge taper cycles", self.ex_taper)
        return widget

    def _build_analysis_tab(self):
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)
        self.an_tau = self._line_edit("2.5")
        self._set_tooltip(self.an_tau, "Exponential smoothing constant for bode estimator in milliseconds.")
        self.an_block = self._line_edit("0.5")
        self._set_tooltip(self.an_block, "FFT block length for the PSD estimate in seconds.")
        self.an_overlap = self._line_edit("0.5")
        self._set_tooltip(self.an_overlap, "Fractional overlap between analysis blocks.")
        self.an_fmin = self._line_edit("0.5")
        self._set_tooltip(self.an_fmin, "Lower analysis frequency bound in Hz.")
        self.an_fmax = self._line_edit("500")
        self._set_tooltip(self.an_fmax, "Upper analysis frequency bound in Hz.")
        self.an_freq_tol = self._line_edit("0.02")
        self._set_tooltip(self.an_freq_tol, "Allowed deviation between requested and detected stepped-sine frequencies.")
        self.an_settle = self._line_edit("0.3")
        self._set_tooltip(self.an_settle, "Portion of each step treated as settling (ignored) before measurement.")
        self.an_r2_min = self._line_edit("0.05")
        self._set_tooltip(self.an_r2_min, "Minimum fit quality (R²) required to keep a data point.")
        self.an_fs = self._line_edit("")
        self.an_fs.setPlaceholderText("auto")
        self._set_tooltip(self.an_fs, "Override sampling rate for offline reanalysis; leave blank to infer from log.")
        form.addRow("Tau [ms]", self.an_tau)
        form.addRow("Block length [s]", self.an_block)
        form.addRow("Overlap", self.an_overlap)
        form.addRow("f min [Hz]", self.an_fmin)
        form.addRow("f max [Hz]", self.an_fmax)
        form.addRow("Freq tolerance", self.an_freq_tol)
        form.addRow("Settle frac", self.an_settle)
        form.addRow("R2 min", self.an_r2_min)
        form.addRow("Sample rate [Hz]", self.an_fs)
        return widget

    def _build_mechanical_tab(self):
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)
        self.me_motor = self._line_edit("0.5")
        self._set_tooltip(self.me_motor, "Motor rated torque used to derive the default torque scaling (Nm).")
        self.me_torque_scale = self._line_edit("")
        self.me_torque_scale.setPlaceholderText("motor/100 if blank")
        self._set_tooltip(self.me_torque_scale, "Manual Nm-per-command scaling; defaults to rated torque / 100.")
        self.me_vel_scale = self._line_edit("1.0")
        self._set_tooltip(self.me_vel_scale, "Scaling factor to convert measured velocity units to physical units.")
        self.me_smooth = self._line_edit("150")
        self._set_tooltip(self.me_smooth, "Low-pass filter cutoff used before fitting the mechanical model.")
        self.me_deriv = self._line_edit("120")
        self._set_tooltip(self.me_deriv, "Cutoff for derivative filter when estimating velocity/acceleration.")
        self.me_deadband = self._line_edit("0.001")
        self._set_tooltip(self.me_deadband, "Deadband applied to small velocity values to reduce noise.")
        self.me_pi_bw = self._line_edit("100")
        self._set_tooltip(self.me_pi_bw, "Target closed-loop bandwidth for the suggested PI gains.")
        self.me_pi_zeta = self._line_edit("1.0")
        self._set_tooltip(self.me_pi_zeta, "Desired damping ratio for the suggested PI gains.")
        form.addRow("Motor rated torque [Nm]", self.me_motor)
        form.addRow("Torque scale [Nm/unit]", self.me_torque_scale)
        form.addRow("Velocity scale", self.me_vel_scale)
        form.addRow("Smooth cutoff [Hz]", self.me_smooth)
        form.addRow("Derivative cutoff [Hz]", self.me_deriv)
        form.addRow("Velocity deadband", self.me_deadband)
        form.addRow("PI bandwidth [Hz]", self.me_pi_bw)
        form.addRow("PI zeta", self.me_pi_zeta)
        return widget

    def _line_edit(self, default):
        edit = PVLineEdit(default)
        edit.setClearButtonEnabled(True)
        return edit

    def _set_tooltip(self, widget, text):
        if widget is not None:
            widget.setToolTip(text.strip())
        return widget

    # -----------------------
    # Actions
    # -----------------------
    def _start_measurement(self):
        try:
            pv_cfg, ex_cfg, an_cfg, mech_cfg = self._collect_settings()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Invalid input", str(exc))
            return
        if not self._confirm_excitation(ex_cfg):
            self.append_log("Measurement cancelled by user before excitation")
            return
        log_path = self.log_path_edit.text().strip()
        self.last_analysis_settings = an_cfg
        self._run_worker("measure", dict(pv=pv_cfg, excitation=ex_cfg, analysis=an_cfg, mechanical=mech_cfg, log_filename=log_path))

    def _start_reanalysis(self):
        try:
            _, _, an_cfg, mech_cfg = self._collect_settings()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Invalid input", str(exc))
            return
        log_path = self.log_path_edit.text().strip()
        if not log_path:
            QtWidgets.QMessageBox.warning(self, "Missing log", "Select a log file to reanalyze.")
            return
        self.last_analysis_settings = an_cfg
        self._run_worker("reanalyze", dict(log_filename=log_path, analysis=an_cfg, mechanical=mech_cfg))

    def _browse_log(self, save):
        initial = self.log_path_edit.text().strip() or os.path.join(os.getcwd(), "autotune_log.pkl")
        if save:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select log destination", initial, "Pickle (*.pkl)")
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select log file", initial, "Pickle (*.pkl)")
        if path:
            self.log_path_edit.setText(path)

    def _collect_settings(self):
        pv_cfg = pipeline.PVSettings(
            prefix=self.pv_prefix.text().strip() or "",
            sp=self.pv_sp.text().strip(),
            sp_rbv=self.pv_sp_rbv.text().strip(),
            act=self.pv_act.text().strip(),
            extra_logs=self._parse_extra_pvs(self.pv_extra.toPlainText()),
        )
        ex_cfg = pipeline.ExcitationSettings(
            fs=self._float(self.ex_fs, "Sampling frequency"),
            f_start=self._float(self.ex_f_start, "Start frequency"),
            f_stop=self._float(self.ex_f_stop, "Stop frequency"),
            n_points=self._int(self.ex_points, "Points"),
            amp=self._float(self.ex_amp, "Amplitude"),
            n_settle=self._int(self.ex_settle, "Settle cycles"),
            n_meas=self._int(self.ex_meas, "Measure cycles"),
            transition_min_s=self._float(self.ex_trans_min, "Transition min"),
            transition_frac=self._float(self.ex_trans_frac, "Transition frac"),
            edge_taper_cycles=self._float(self.ex_taper, "Edge taper"),
        )
        an_cfg = pipeline.AnalysisSettings(
            tau_ms=self._float(self.an_tau, "Tau"),
            block_len_s=self._float(self.an_block, "Block length"),
            overlap=self._float(self.an_overlap, "Overlap"),
            fmin=self._float(self.an_fmin, "f min"),
            fmax=self._float(self.an_fmax, "f max"),
            freq_tolerance=self._float(self.an_freq_tol, "Freq tolerance"),
            settle_frac=self._float(self.an_settle, "Settle frac"),
            r2_min=self._float(self.an_r2_min, "R2 min"),
            sample_hz=self._optional_float(self.an_fs),
        )
        mech_cfg = pipeline.MechanicalSettings(
            motor_rated_trq=self._float(self.me_motor, "Motor torque"),
            torque_scale=self._optional_float(self.me_torque_scale),
            velocity_scale=self._float(self.me_vel_scale, "Velocity scale"),
            smooth_hz=self._float(self.me_smooth, "Smooth cutoff"),
            deriv_hz=self._float(self.me_deriv, "Derivative cutoff"),
            vel_deadband=self._float(self.me_deadband, "Velocity deadband"),
            pi_bandwidth=self._float(self.me_pi_bw, "PI bandwidth"),
            pi_zeta=self._float(self.me_pi_zeta, "PI zeta"),
        )
        return pv_cfg, ex_cfg, an_cfg, mech_cfg

    def _parse_extra_pvs(self, text):
        extra = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
            else:
                key = line
                value = line
            if not key or not value:
                continue
            extra[key] = value
        return extra

    def _insert_pv_text(self, value):
        widget = QtWidgets.QApplication.focusWidget()
        if isinstance(widget, QtWidgets.QLineEdit):
            widget.setText(value)
        elif isinstance(widget, QtWidgets.QPlainTextEdit):
            text = widget.toPlainText().strip()
            if text:
                text += "\n" + value
            else:
                text = value
            widget.setPlainText(text)

    def _float(self, edit, label):
        text = edit.text().strip()
        if not text:
            raise ValueError(f"{label} is required")
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number") from exc

    def _optional_float(self, edit):
        text = edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{edit.placeholderText() or 'value'} must be numeric") from exc

    def _int(self, edit, label):
        value = self._float(edit, label)
        return int(round(value))

    def _run_worker(self, mode, kwargs):
        if self.worker_thread is not None:
            return
        self.append_log(f"[{mode}] started")
        self.progress_bar.setValue(0)
        self.measure_btn.setEnabled(False)
        self.reanalyze_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self.worker_thread = QtCore.QThread(self)
        self.worker = Worker(mode, kwargs)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self._update_progress)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.failed.connect(self._cleanup_worker)
        self.worker_thread.start()

    def _cleanup_worker(self):
        if self.worker_thread:
            try:
                self.worker_thread.requestInterruption()
            except Exception:
                pass
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None
            self.measure_btn.setEnabled(True)
            self.reanalyze_btn.setEnabled(True)
            self.abort_btn.setEnabled(False)

    def _on_worker_finished(self, result):
        self.append_log("Worker finished")
        self.latest_result = result
        self._update_plots(result)
        self._report_result(result)
        self.progress_bar.setValue(100)
        self.abort_btn.setEnabled(False)

    def _on_worker_failed(self, trace):
        self.append_log(trace)
        if trace.lower().startswith("measurement aborted"):
            QtWidgets.QMessageBox.information(self, "Measurement aborted", trace)
        else:
            QtWidgets.QMessageBox.critical(self, "Worker failed", trace)
        self.progress_bar.setValue(0)
        self.abort_btn.setEnabled(False)

    def _abort_worker(self):
        if self.worker and self.worker_thread:
            self.append_log("Abort requested by user")
            self.worker.request_abort()
            self.worker_thread.requestInterruption()
            QtWidgets.QMessageBox.information(self, "Abort", "Stop requested. Allow current PV puts to finish.")

    def _update_progress(self, value):
        self.progress_bar.setValue(int(max(0.0, min(1.0, value)) * 100))

    def append_log(self, message):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{stamp}] {message}")

    def _update_plots(self, result):
        if result is None:
            self.bode_canvas.clear()
            self.time_canvas.clear()
            self.last_bode_data = None
            self.last_time_data = None
            return
        freq = result.bode.get("freq", np.array([]))
        mag = result.bode.get("mag_db", np.array([]))
        phase = result.bode.get("phase", np.array([]))
        phase_comp = result.bode.get("phase_comp", np.array([]))
        r2_u = result.bode.get("r2_u", np.array([]))
        r2_y = result.bode.get("r2_y", np.array([]))
        mask = np.ones_like(freq, dtype=bool)
        if freq.size and r2_u.size and r2_y.size:
            threshold = getattr(self.last_analysis_settings, "r2_min", 0.0)
            mask = (r2_u >= threshold) & (r2_y >= threshold)
        f_plot = freq[mask]
        mag_plot = mag[mask]
        phase_plot = phase[mask]
        phase_c_plot = phase_comp[mask]
        self.last_bode_data = (f_plot, mag_plot, phase_plot, phase_c_plot)
        ax_mag, ax_phase = self.bode_canvas.axes[0][0], self.bode_canvas.axes[1][0]
        ax_mag.cla()
        ax_phase.cla()
        if f_plot.size:
            ax_mag.semilogx(f_plot, mag_plot, marker="o")
            ax_mag.axhline(0, color="k", linestyle="--", linewidth=0.8)
            ax_mag.set_ylabel("Mag [dB]")
            ax_mag.grid(True, which="both", linestyle="--", alpha=0.4)

            ax_phase.semilogx(f_plot, phase_plot, marker="o", label="Phase")
            ax_phase.semilogx(f_plot, phase_c_plot, marker="o", label="Phase (comp)")
            ax_phase.axhline(-180, color="k", linestyle="--", linewidth=0.8)
            ax_phase.set_ylabel("Phase [deg]")
            ax_phase.set_xlabel("Frequency [Hz]")
            ax_phase.grid(True, which="both", linestyle="--", alpha=0.4)
            ax_phase.legend()
        else:
            ax_mag.text(0.5, 0.5, "No data", ha="center", va="center")
            ax_phase.text(0.5, 0.5, "No data", ha="center", va="center")
        self.bode_canvas.draw_idle()

        ax_sig = self.time_canvas.axes[0][0]
        ax_sig.cla()
        if result.t.size:
            t = result.t
            cmd = result.values_by_pv.get(result.command_key)
            resp = result.values_by_pv.get(result.response_key)
            if cmd is not None:
                ax_sig.plot(t[: len(cmd)], cmd, label=result.command_key)
            if resp is not None:
                ax_sig.plot(t[: len(resp)], resp, label=result.response_key)
            ax_sig.set_xlabel("Time [s]")
            ax_sig.set_ylabel("Signal")
            ax_sig.grid(True, linestyle="--", alpha=0.4)
            ax_sig.legend()
            self.last_time_data = (t, cmd, resp, result.command_key, result.response_key)
        else:
            ax_sig.text(0.5, 0.5, "No data", ha="center", va="center")
            self.last_time_data = None
        self.time_canvas.draw_idle()

    def _report_result(self, result):
        if result.log_file:
            self.append_log(f"Log: {result.log_file}")
        if result.mechanical:
            mech = result.mechanical
            self.append_log(
                "Mechanical fit -> "
                f"J={mech['J']:.4g}, B={mech['B']:.4g}, Tc={mech['Tc']:.4g}, residual={mech['residual_rms']:.4g}" )
            if all(k in mech for k in ("kp", "ki", "ti")):
                self.append_log(
                    f"Suggested PI: Kp={mech['kp']:.4g}, Ki={mech['ki']:.4g}, Ti={mech['ti']:.4g}"
                )

    def _show_bode_popup(self):
        if not self.last_bode_data:
            QtWidgets.QMessageBox.information(self, "No data", "Run a measurement/reanalysis first.")
            return
        f_plot, mag_plot, phase_plot, phase_c_plot = self.last_bode_data
        if f_plot.size == 0:
            QtWidgets.QMessageBox.information(self, "No data", "No valid Bode points to show.")
            return
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        ax_mag.semilogx(f_plot, mag_plot, marker="o")
        ax_mag.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax_mag.set_ylabel("Mag [dB]")
        ax_mag.grid(True, which="both", linestyle="--", alpha=0.4)
        ax_mag.set_title("Bode Magnitude")
        ax_phase.semilogx(f_plot, phase_plot, marker="o", label="Phase")
        ax_phase.semilogx(f_plot, phase_c_plot, marker="o", label="Phase (comp)")
        ax_phase.axhline(-180, color="k", linestyle="--", linewidth=0.8)
        ax_phase.set_ylabel("Phase [deg]")
        ax_phase.set_xlabel("Frequency [Hz]")
        ax_phase.grid(True, which="both", linestyle="--", alpha=0.4)
        ax_phase.legend()
        fig.tight_layout()
        fig.show()

    def _show_time_popup(self):
        if not self.last_time_data:
            QtWidgets.QMessageBox.information(self, "No data", "Run a measurement/reanalysis first.")
            return
        t, cmd, resp, cmd_label, resp_label = self.last_time_data
        if t is None or ((cmd is None) and (resp is None)):
            QtWidgets.QMessageBox.information(self, "No data", "No signals available to plot.")
            return
        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
        if cmd is not None:
            ax.plot(t[: len(cmd)], cmd, label=cmd_label)
        if resp is not None:
            ax.plot(t[: len(resp)], resp, label=resp_label)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Signal")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        ax.set_title("Command vs Response")
        fig.tight_layout()
        fig.show()

    def _confirm_excitation(self, ex_cfg):
        try:
            t, u, _, f_array, _, _ = excite_sine.generate(
                ex_cfg.fs,
                ex_cfg.f_start,
                ex_cfg.f_stop,
                ex_cfg.n_points,
                ex_cfg.amp,
                ex_cfg.n_settle,
                ex_cfg.n_meas,
                transition_min_s=ex_cfg.transition_min_s,
                transition_frac=ex_cfg.transition_frac,
                edge_taper_cycles=ex_cfg.edge_taper_cycles,
                return_masks=True,
                preview=False,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Excitation error", str(exc))
            return False
        dialog = ExcitationPreviewDialog(self, t, u, f_array)
        return dialog.exec_() == QtWidgets.QDialog.Accepted


class ExcitationPreviewDialog(QtWidgets.QDialog):
    def __init__(self, parent, t, u, f_array):
        super().__init__(parent)
        self.setWindowTitle("Review Excitation Sequence")
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel("Inspect the stepped-sine command below. Click OK to proceed.")
        layout.addWidget(label)
        canvas = PlotCanvas(rows=1, cols=1, figsize=(7, 3), constrained_layout=True)
        layout.addWidget(canvas)
        ax = canvas.axes[0][0]
        ax.plot(t, u, label="Command")
        if np.max(f_array) > 0:
            ax2 = ax.twinx()
            ax2.plot(t, f_array, "r--", alpha=0.6, label="Freq [Hz]")
            ax2.set_ylabel("Freq [Hz]")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Command")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title("Excitation preview")
        canvas.draw_idle()
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok,
            QtCore.Qt.Horizontal,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = AutotuneWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
