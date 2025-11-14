import os
import sys
import traceback
from datetime import datetime

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

try:
    from . import pipeline, excite_sine
except ImportError:  # pragma: no cover
    import pipeline  # type: ignore
    import excite_sine  # type: ignore


class Worker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    log = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(float)

    def __init__(self, mode, kwargs):
        super().__init__()
        self.mode = mode
        self.kwargs = kwargs

    @QtCore.pyqtSlot()
    def run(self):
        try:
            if self.mode == "measure":
                result = pipeline.run_measurement(progress_fn=self.progress.emit, log_fn=self.log.emit, **self.kwargs)
            elif self.mode == "reanalyze":
                result = pipeline.reanalyze_log(log_fn=self.log.emit, **self.kwargs)
            else:
                raise ValueError(f"Unsupported worker mode {self.mode}")
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.finished.emit(result)


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
        log_path_row.addWidget(QtWidgets.QLabel("Log file"))
        log_path_row.addWidget(self.log_path_edit)
        save_btn = QtWidgets.QPushButton("Browse Save…")
        save_btn.clicked.connect(lambda: self._browse_log(True))
        load_btn = QtWidgets.QPushButton("Browse Load…")
        load_btn.clicked.connect(lambda: self._browse_log(False))
        log_path_row.addWidget(save_btn)
        log_path_row.addWidget(load_btn)
        layout.addLayout(log_path_row)

        button_row = QtWidgets.QHBoxLayout()
        self.measure_btn = QtWidgets.QPushButton("Run Measurement")
        self.reanalyze_btn = QtWidgets.QPushButton("Reanalyze Log")
        self.measure_btn.clicked.connect(self._start_measurement)
        self.reanalyze_btn.clicked.connect(self._start_reanalysis)
        button_row.addWidget(self.measure_btn)
        button_row.addWidget(self.reanalyze_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        plots_row = QtWidgets.QHBoxLayout()
        self.bode_group = QtWidgets.QGroupBox("Bode")
        bode_layout = QtWidgets.QVBoxLayout()
        self.bode_canvas = PlotCanvas(rows=2, cols=1, figsize=(5, 4), constrained_layout=True)
        bode_layout.addWidget(self.bode_canvas)
        self.bode_popup_btn = QtWidgets.QPushButton("Open Window")
        self.bode_popup_btn.clicked.connect(self._show_bode_popup)
        bode_layout.addWidget(self.bode_popup_btn)
        self.bode_group.setLayout(bode_layout)

        self.time_group = QtWidgets.QGroupBox("Signals")
        time_layout = QtWidgets.QVBoxLayout()
        self.time_canvas = PlotCanvas(rows=1, cols=1, figsize=(5, 4), constrained_layout=True)
        time_layout.addWidget(self.time_canvas)
        self.time_popup_btn = QtWidgets.QPushButton("Open Window")
        self.time_popup_btn.clicked.connect(self._show_time_popup)
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
        form = QtWidgets.QFormLayout(widget)
        self.pv_prefix = self._line_edit("c6025a-08:m1s000-")
        self.pv_sp = self._line_edit("Drv01-Trq")
        self.pv_sp_rbv = self._line_edit("Drv01-TrqAct")
        self.pv_vel = self._line_edit("Drv01-VelAct")
        self.pv_pos = self._line_edit("Enc01-PosAct")
        self.pv_trq = self._line_edit("Drv01-TrqAct")
        form.addRow("Prefix", self.pv_prefix)
        form.addRow("SP", self.pv_sp)
        form.addRow("SP_RBV", self.pv_sp_rbv)
        form.addRow("VEL_ACT", self.pv_vel)
        form.addRow("POS_ACT", self.pv_pos)
        form.addRow("TRQ_ACT", self.pv_trq)
        return widget

    def _build_excitation_tab(self):
        widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(widget)
        self.ex_fs = self._line_edit("1000")
        self.ex_f_start = self._line_edit("10")
        self.ex_f_stop = self._line_edit("400")
        self.ex_points = self._line_edit("20")
        self.ex_amp = self._line_edit("0.01")
        self.ex_settle = self._line_edit("10")
        self.ex_meas = self._line_edit("50")
        self.ex_trans_min = self._line_edit("1.0")
        self.ex_trans_frac = self._line_edit("0.0")
        self.ex_taper = self._line_edit("1.0")
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
        self.an_block = self._line_edit("0.5")
        self.an_overlap = self._line_edit("0.5")
        self.an_fmin = self._line_edit("0.5")
        self.an_fmax = self._line_edit("500")
        self.an_freq_tol = self._line_edit("0.02")
        self.an_settle = self._line_edit("0.3")
        self.an_r2_min = self._line_edit("0.05")
        self.an_fs = self._line_edit("")
        self.an_fs.setPlaceholderText("auto")
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
        self.me_torque_scale = self._line_edit("")
        self.me_torque_scale.setPlaceholderText("motor/100 if blank")
        self.me_vel_scale = self._line_edit("1.0")
        self.me_smooth = self._line_edit("150")
        self.me_deriv = self._line_edit("120")
        self.me_deadband = self._line_edit("0.001")
        self.me_pi_bw = self._line_edit("100")
        self.me_pi_zeta = self._line_edit("1.0")
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
        edit = QtWidgets.QLineEdit(default)
        edit.setClearButtonEnabled(True)
        return edit

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
            vel_act=self.pv_vel.text().strip(),
            pos_act=self.pv_pos.text().strip(),
            trq_act=self.pv_trq.text().strip(),
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
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None
            self.measure_btn.setEnabled(True)
            self.reanalyze_btn.setEnabled(True)

    def _on_worker_finished(self, result):
        self.append_log("Worker finished")
        self.latest_result = result
        self._update_plots(result)
        self._report_result(result)
        self.progress_bar.setValue(100)

    def _on_worker_failed(self, trace):
        self.append_log(trace)
        QtWidgets.QMessageBox.critical(self, "Worker failed", trace)
        self.progress_bar.setValue(0)

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


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = AutotuneWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
