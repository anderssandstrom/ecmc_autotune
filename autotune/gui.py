import os
import sys
import traceback
import pickle
from datetime import datetime
from pathlib import Path

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

MODE_ORDER = [
    pipeline.DEFAULT_MODE,
    "csv_velocity_bode",
    "csv_position_tune",
    "generic",
    "logger",
]
DEFAULT_FLOW_STEPS = [
    ("PV Setup", "Define prefixes and PVs to log/control."),
    ("Excitation", "Configure stepped-sine sweep."),
    ("Acquire", "Drive the axis and capture PV data."),
    ("Analyze", "Compute Bode/plant fits from the log."),
    ("Suggest", "Derive PI/PID gains from analysis."),
]


MODE_DEFINITIONS = {
    pipeline.DEFAULT_MODE: {
        "label": "CST velocity loop tune",
        "description": "Command torque (Trq) and observe velocity (VelAct). Mechanical model fitting and PI suggestions are enabled.",
        "supports_mechanical": True,
        "flow_steps": DEFAULT_FLOW_STEPS,
        "pv_defaults": {
            "prefix_p": "c6025a-08:",
            "prefix_r": "m1s000-",
            "sp": "Drv01-Trq",
            "sp_rbv": "Drv01-TrqAct",
            "act": "Drv01-VelAct",
            "motor_rated_trq": "1.0",
            "torque_scale": "",
            "velocity_scale": "1.0",
        },
        "pv_labels": {
            "prefix_p": "Prefix P",
            "prefix_r": "Prefix R",
            "sp": "Torque command PV (SP) [%]",
            "sp_rbv": "Torque readback PV (SP_RBV) [%]",
            "act": "Velocity response PV (ACT)",
            "extra": "Extra log PVs",
        },
        "pv_tooltips": {
            "prefix_p": "First prefix segment (e.g. IOC name). Leave blank for absolute PVs.",
            "prefix_r": "Second prefix segment appended after P (e.g. axis identifier).",
            "sp": "EPICS PV that accepts the torque demand (e.g. Drv01-Trq). Values interpreted as percent torque before scaling.",
            "sp_rbv": "Readback of the commanded torque (e.g. Drv01-TrqAct). Used as the command trace for analysis.",
            "act": "Measured velocity feedback PV (e.g. Drv01-VelAct).",
            "extra": "Optional extra PVs to log (NAME=PV or PV). One per line.",
        },
        "time_plot": {
            "command_label": "Torque command",
            "command_units": "%",
            "response_label": "Velocity feedback",
            "response_units": "user units",
            "ylabel": "Torque [%] / Velocity",
        },
        "mechanical_hint": "",
    },
    "csv_velocity_bode": {
        "label": "CSV closed loop bode",
        "description": "Closed-loop velocity bode plot using speed demand (Spd) and velocity feedback (VelAct) in rad/s.",
        "supports_mechanical": False,
        "flow_steps": DEFAULT_FLOW_STEPS,
        "pv_defaults": {
            "prefix_p": "c6025a-08:",
            "prefix_r": "m1s000-",
            "sp": "Drv01-Spd",
            "sp_rbv": "",
            "act": "Drv01-VelAct",
            "motor_rated_trq": "1.0",
            "torque_scale": "",
            "velocity_scale": "1.0",
        },
        "pv_labels": {
            "prefix_p": "Prefix P",
            "prefix_r": "Prefix R",
            "sp": "Speed command PV (SP) [rad/s]",
            "sp_rbv": "Speed readback PV (optional)",
            "act": "Velocity response PV (ACT) [rad/s]",
            "extra": "Extra log PVs",
        },
        "pv_tooltips": {
            "prefix_p": "First prefix segment applied before PV names (optional).",
            "prefix_r": "Second prefix segment appended after P.",
            "sp": "Speed demand PV to drive during the bode sweep (e.g. Drv01-Spd).",
            "sp_rbv": "Optional readback PV for the commanded speed. Leave blank to use SP.",
            "act": "Velocity feedback PV (VelAct) expressed in rad/s.",
            "extra": "Optional extra PVs to log (NAME=PV or PV). One per line.",
        },
        "time_plot": {
            "command_label": "Speed command",
            "command_units": "rad/s",
            "response_label": "Velocity feedback",
            "response_units": "rad/s",
            "ylabel": "Speed [rad/s]",
        },
        "mechanical_hint": "Mechanical identification is disabled in CSV bode mode.",
    },
    "csv_position_tune": {
        "label": "CSV position loop tune",
        "description": "Use speed demand (Spd), velocity readback (VelAct) as command feedback, and position response (PosAct).",
        "supports_mechanical": True,
        "flow_steps": [
            ("PV Setup", "Select speed, velocity, and position PVs."),
            ("Excitation", "Configure the stepped-sine outer-loop drive."),
            ("Acquire", "Drive the position loop while logging PVs."),
            ("Analyze", "Bode + plant fit for the position loop."),
            ("PID Suggest", "Compute position PID from bode fit."),
        ],
        "pv_defaults": {
            "prefix_p": "c6025a-08:",
            "prefix_r": "m1s000-",
            "sp": "Drv01-Spd",
            "sp_rbv": "Drv01-VelAct",
            "act": "Enc01-PosAct",
            "motor_rated_trq": "1.0",
            "torque_scale": "",
            "velocity_scale": "1.0",
        },
        "pv_labels": {
            "prefix_p": "Prefix P",
            "prefix_r": "Prefix R",
            "sp": "Speed command PV (SP) [rad/s]",
            "sp_rbv": "Velocity inner-loop PV (SP_RBV) [rad/s]",
            "act": "Position response PV (ACT)",
            "extra": "Extra log PVs",
        },
        "pv_tooltips": {
            "prefix_p": "First prefix segment for PV shortcuts (optional).",
            "prefix_r": "Second prefix segment appended after P.",
            "sp": "Outer-loop speed demand PV (Spd) to excite the position loop.",
            "sp_rbv": "Velocity feedback PV (VelAct) used as the command trace.",
            "act": "Position feedback PV (PosAct) used as the response.",
            "extra": "Optional extra PVs to log (NAME=PV or PV). One per line.",
        },
        "time_plot": {
            "command_label": "Speed command",
            "command_units": "rad/s",
            "response_label": "Position feedback",
            "response_units": "user units",
            "ylabel": "Speed / Position",
        },
        "mechanical_hint": "",
    },
    "generic": {
        "label": "Generic mode",
        "description": "Fully manual configuration. Select and scale PVs as needed. Mechanical fitting is disabled.",
        "supports_mechanical": False,
        "flow_steps": DEFAULT_FLOW_STEPS,
        "pv_defaults": {
            "prefix_p": "",
            "prefix_r": "",
            "sp": "",
            "sp_rbv": "",
            "act": "",
            "motor_rated_trq": "1.0",
            "torque_scale": "",
            "velocity_scale": "1.0",
        },
        "pv_labels": {
            "prefix_p": "Prefix P",
            "prefix_r": "Prefix R",
            "sp": "Command PV (SP)",
            "sp_rbv": "Command readback PV (SP_RBV)",
            "act": "Response PV (ACT)",
            "extra": "Extra log PVs",
        },
        "pv_tooltips": {
            "prefix_p": "Optional first prefix segment prepended to PVs.",
            "prefix_r": "Optional second prefix segment appended after P.",
            "sp": "EPICS PV to write during the excitation.",
            "sp_rbv": "Optional readback PV for the commanded signal.",
            "act": "Measured response PV used for analysis.",
            "extra": "Optional extra PVs to log (NAME=PV or PV). One per line.",
        },
        "time_plot": {
            "command_label": "Command",
            "command_units": "",
            "response_label": "Response",
            "response_units": "",
            "ylabel": "Command / Response",
        },
        "mechanical_hint": "Mechanical identification is disabled in generic mode.",
    },
    "logger": {
        "label": "Logger",
        "description": "Record PVs over time without applying stepped-sine excitation.",
        "supports_mechanical": False,
        "preserve_prefix": True,
        "flow_steps": [
            ("PV Setup", "Choose PVs to monitor; no commands are written."),
            ("Log", "Start acquisition and record all selected PVs."),
            ("Review", "Visualize signals or export logs."),
        ],
        "pv_defaults": {
            "prefix_p": "",
            "prefix_r": "",
            "sp": "",
            "sp_rbv": "",
            "act": "",
            "motor_rated_trq": "1.0",
            "torque_scale": "",
            "velocity_scale": "1.0",
        },
        "pv_labels": {
            "prefix_p": "Prefix P",
            "prefix_r": "Prefix R",
            "sp": "Primary PV (optional)",
            "sp_rbv": "Secondary PV (optional)",
            "act": "Tertiary PV (optional)",
            "extra": "Extra log PVs",
        },
        "pv_tooltips": {
            "prefix_p": "Optional first prefix, applied to all PV names.",
            "prefix_r": "Optional second prefix segment.",
            "sp": "PV to log as 'SP' (no writes performed).",
            "sp_rbv": "PV to log as 'SP_RBV'.",
            "act": "PV to log as 'ACT'.",
            "extra": "Additional PVs to log (NAME=PV or PV).",
        },
        "time_plot": {
            "command_label": "SP",
            "command_units": "",
            "response_label": "ACT",
            "response_units": "",
            "ylabel": "Signals",
        },
        "mechanical_hint": "",
    },
}


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
        self.last_extra_series = {}
        self.derived_extra_series = {}
        self.last_analysis_settings = pipeline.AnalysisSettings()
        self.current_mode_key = pipeline.DEFAULT_MODE
        self._last_auto_values = {}
        self.segment_overlay_cb = None
        self.mechanical_hint_label = QtWidgets.QLabel("")
        self.mechanical_hint_label.setWordWrap(True)
        self.mechanical_hint_label.setStyleSheet("color: #a00;")
        self._build_ui()

    # -----------------------
    # UI construction
    # -----------------------
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        settings_widget = QtWidgets.QWidget()
        settings_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        settings_widget.setMaximumHeight(420)
        settings_layout = QtWidgets.QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(6)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.tabs.setMaximumHeight(320)
        self.flow_tab = self._build_flow_tab()
        self.pv_tab = self._build_pv_tab()
        self.exc_tab = self._build_excitation_tab()
        self.analysis_tab = self._build_analysis_tab()
        self.pid_tab = self._build_pid_tab()
        self.file_tab = self._build_file_tab()
        self.docs_tab = self._build_docs_tab()
        self.tabs.addTab(self.flow_tab, "Flow")
        self.tabs.addTab(self.pv_tab, "PV Settings")
        self.tabs.addTab(self.exc_tab, "Excitation")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.pid_tab, "PID Tune")
        self.tabs.addTab(self.file_tab, "File")
        self.tabs.addTab(self.docs_tab, "Docs")
        settings_layout.addWidget(self.tabs)

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
        settings_layout.addLayout(button_row)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self._set_tooltip(self.progress_bar, "Shows completion of the running measurement or analysis step.")
        settings_layout.addWidget(self.progress_bar)
        layout.addWidget(settings_widget, 0)

        results_widget = QtWidgets.QWidget()
        results_layout = QtWidgets.QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(6)

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
        extra_label = QtWidgets.QLabel("Extra PVs (select to plot)")
        extra_label.setWordWrap(True)
        time_layout.addWidget(extra_label)
        self.extra_pv_list = QtWidgets.QListWidget()
        self.extra_pv_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.extra_pv_list.itemSelectionChanged.connect(self._refresh_time_plot)
        self._set_tooltip(self.extra_pv_list, "Select extra logged PVs to overlay on the time plot.")
        time_layout.addWidget(self.extra_pv_list)
        button_row = QtWidgets.QHBoxLayout()
        self.time_popup_btn = QtWidgets.QPushButton("Open Signals Plot")
        self.time_popup_btn.clicked.connect(self._show_time_popup)
        self._set_tooltip(self.time_popup_btn, "Pop out the primary command/response signals into a separate window.")
        button_row.addWidget(self.time_popup_btn)
        self.extra_popup_btn = QtWidgets.QPushButton("Open Selected PVs")
        self.extra_popup_btn.clicked.connect(self._show_selected_pv_popup)
        self._set_tooltip(self.extra_popup_btn, "Plot the currently selected extra PVs in a separate window.")
        button_row.addWidget(self.extra_popup_btn)
        button_row.addStretch(1)
        time_layout.addLayout(button_row)
        self.time_group.setLayout(time_layout)

        plots_row.addWidget(self.bode_group, 1)
        plots_row.addWidget(self.time_group, 1)
        results_layout.addLayout(plots_row, 1)

        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        results_layout.addWidget(self.log_output, 1)

        layout.addWidget(results_widget, 1)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        initial_mode = self.mode_combo.currentData()
        self._apply_mode_settings(initial_mode, force_defaults=True)

    def _build_pv_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        form_container = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(form_container)
        self.pv_prefix_p = self._line_edit("")
        self._set_tooltip(self.pv_prefix_p, "Prefix part P (optional).")
        self.pv_prefix_r = self._line_edit("")
        self._set_tooltip(self.pv_prefix_r, "Prefix part R (optional).")
        self.pv_sp = self._line_edit("")
        self.pv_sp_rbv = self._line_edit("")
        self.pv_act = self._line_edit("")
        self.pv_motor_torque = self._line_edit("1.0")
        self._set_tooltip(self.pv_motor_torque, "Motor rated torque used for torque scaling (Nm).")
        self.pv_torque_scale = self._line_edit("")
        self.pv_torque_scale.setPlaceholderText("motor/100 if blank")
        self._set_tooltip(self.pv_torque_scale, "Manual Nm-per-command scaling; defaults to rated torque / 100.")
        self.pv_velocity_scale = self._line_edit("1.0")
        self._set_tooltip(self.pv_velocity_scale, "Scaling factor to convert velocity PV units to physical units.")
        form_container_right = QtWidgets.QWidget()
        form_right = QtWidgets.QVBoxLayout(form_container_right)
        self.pv_extra = PVPlainTextEdit("")
        self.pv_extra.setPlaceholderText("One PV per line (optional NAME=PV). Empty line removes logging.")
        self.pv_prefix_label_p = QtWidgets.QLabel("Prefix P")
        self.pv_prefix_label_r = QtWidgets.QLabel("Prefix R")
        self.pv_sp_label = QtWidgets.QLabel("SP")
        self.pv_sp_rbv_label = QtWidgets.QLabel("SP_RBV")
        self.pv_act_label = QtWidgets.QLabel("ACT PV")
        self.pv_extra_label = QtWidgets.QLabel("Extra log PVs")
        form.addRow(self.pv_prefix_label_p, self.pv_prefix_p)
        form.addRow(self.pv_prefix_label_r, self.pv_prefix_r)
        form.addRow(self.pv_sp_label, self.pv_sp)
        form.addRow(self.pv_sp_rbv_label, self.pv_sp_rbv)
        form.addRow(self.pv_act_label, self.pv_act)
        form.addRow("Motor rated torque [Nm]", self.pv_motor_torque)
        form.addRow("Torque scale [Nm/unit]", self.pv_torque_scale)
        form.addRow("Velocity scale", self.pv_velocity_scale)
        layout.addWidget(form_container, 2)
        self.torque_fields = [
            self.pv_motor_torque,
            self.pv_torque_scale,
        ]
        self.velocity_fields = [self.pv_velocity_scale]
        self._pv_fields = {
            "prefix_p": self.pv_prefix_p,
            "prefix_r": self.pv_prefix_r,
            "sp": self.pv_sp,
            "sp_rbv": self.pv_sp_rbv,
            "act": self.pv_act,
            "motor_rated_trq": self.pv_motor_torque,
            "torque_scale": self.pv_torque_scale,
            "velocity_scale": self.pv_velocity_scale,
        }
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

        layout.addWidget(form_container_right, 1)
        form_right.addWidget(self.pv_extra_label)
        form_right.addWidget(self.pv_extra)
        return widget

    def _build_excitation_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        left_form = QtWidgets.QFormLayout()
        right_form = QtWidgets.QFormLayout()
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
        left_form.addRow("fs [Hz]", self.ex_fs)
        left_form.addRow("f start [Hz]", self.ex_f_start)
        left_form.addRow("f stop [Hz]", self.ex_f_stop)
        left_form.addRow("Points", self.ex_points)
        left_form.addRow("Amplitude", self.ex_amp)
        right_form.addRow("Settle cycles", self.ex_settle)
        right_form.addRow("Measure cycles", self.ex_meas)
        right_form.addRow("Transition min [s]", self.ex_trans_min)
        right_form.addRow("Transition frac", self.ex_trans_frac)
        right_form.addRow("Edge taper cycles", self.ex_taper)
        layout.addLayout(left_form, 1)
        layout.addLayout(right_form, 1)
        preview_btn = QtWidgets.QPushButton("Preview excitation")
        preview_btn.clicked.connect(self._preview_excitation_signal)
        self._set_tooltip(preview_btn, "Generate and plot the current excitation waveform.")
        layout.addWidget(preview_btn)
        return widget

    def _build_analysis_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        form_container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(form_container)
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
        labels = [
            ("Tau [ms]", self.an_tau),
            ("Block length [s]", self.an_block),
            ("Overlap", self.an_overlap),
            ("f min [Hz]", self.an_fmin),
            ("f max [Hz]", self.an_fmax),
            ("Freq tolerance", self.an_freq_tol),
            ("Settle frac", self.an_settle),
            ("R2 min", self.an_r2_min),
            ("Sample rate [Hz]", self.an_fs),
        ]
        cols = 2
        for idx, (label, widget_field) in enumerate(labels):
            row = idx // cols
            col = col_offset = col = idx % cols
            row_pos = row
            grid.addWidget(QtWidgets.QLabel(label), row_pos, col * 2)
            grid.addWidget(widget_field, row_pos, col * 2 + 1)
        layout.addWidget(form_container, 2)
        self.mech_filter_group = QtWidgets.QGroupBox("Mechanical fitting filters")
        mech_form = QtWidgets.QFormLayout(self.mech_filter_group)
        self.me_smooth = self._line_edit("150")
        self._set_tooltip(self.me_smooth, "Low-pass filter cutoff used before fitting the mechanical model.")
        self.me_deriv = self._line_edit("120")
        self._set_tooltip(self.me_deriv, "Cutoff for derivative filter when estimating velocity/acceleration.")
        self.me_deadband = self._line_edit("0.001")
        self._set_tooltip(self.me_deadband, "Deadband applied to small velocity values to reduce noise.")
        mech_form.addRow("Smooth cutoff [Hz]", self.me_smooth)
        mech_form.addRow("Derivative cutoff [Hz]", self.me_deriv)
        mech_form.addRow("Velocity deadband", self.me_deadband)
        layout.addWidget(self.mech_filter_group, 1)
        return widget

    def _build_pid_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        form = QtWidgets.QFormLayout()
        self.pid_bw = self._line_edit("100")
        self._set_tooltip(self.pid_bw, "Target closed-loop bandwidth used for PI/PID suggestions.")
        self.pid_zeta = self._line_edit("1.0")
        self._set_tooltip(self.pid_zeta, "Desired damping ratio for the suggested controllers.")
        form.addRow("Target bandwidth [Hz]", self.pid_bw)
        form.addRow("Target zeta", self.pid_zeta)
        layout.addLayout(form)
        note = QtWidgets.QLabel("Velocity PI (CST) and position PID (CSV position tune) both use these targets.")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addWidget(QtWidgets.QLabel("Latest suggestions"))
        self.pid_result_model = QtGui.QStandardItemModel(0, 5, widget)
        self.pid_result_model.setHorizontalHeaderLabels([
            "Mode",
            "Kp [unit/command]",
            "Ki [unit/command/s]",
            "Kd [unit/command·s]",
            "Ti [s]",
        ])
        self.pid_result_view = QtWidgets.QTableView()
        self.pid_result_view.setModel(self.pid_result_model)
        self.pid_result_view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        results_row = QtWidgets.QHBoxLayout()
        results_row.addWidget(self.pid_result_view, 1)
        clear_btn = QtWidgets.QPushButton("Clear results")
        clear_btn.setMaximumWidth(120)
        clear_btn.clicked.connect(self._clear_pid_results)
        button_column = QtWidgets.QVBoxLayout()
        button_column.addWidget(clear_btn)
        button_column.addStretch(1)
        results_row.addLayout(button_column)
        layout.addLayout(results_row)
        layout.addStretch(1)
        return widget

    def _build_file_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        path_row = QtWidgets.QHBoxLayout()
        path_label = QtWidgets.QLabel("Log file")
        self.log_path_edit = QtWidgets.QLineEdit("autotune/logs/latest.pkl")
        self.log_path_edit.setClearButtonEnabled(True)
        self._set_tooltip(self.log_path_edit, "Destination for captured logs or the file to reopen for analysis.")
        path_row.addWidget(path_label)
        path_row.addWidget(self.log_path_edit)
        layout.addLayout(path_row)
        button_row = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Browse Save…")
        save_btn.clicked.connect(lambda: self._browse_log(True))
        load_btn = QtWidgets.QPushButton("Browse Load…")
        load_btn.clicked.connect(lambda: self._browse_log(False))
        self._set_tooltip(save_btn, "Choose where to write the next acquisition log.")
        self._set_tooltip(load_btn, "Pick an existing log to reanalyze and restore settings.")
        button_row.addWidget(save_btn)
        button_row.addWidget(load_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        self.log_meta_label = QtWidgets.QLabel("No log loaded.")
        self.log_meta_label.setWordWrap(True)
        layout.addWidget(self.log_meta_label)
        layout.addStretch(1)
        return widget

    def _build_docs_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.addWidget(QtWidgets.QLabel("Mode overview"))
        self.docs_area = QtWidgets.QTextEdit()
        self.docs_area.setReadOnly(True)
        self.docs_area.setMinimumHeight(200)
        layout.addWidget(self.docs_area, 1)
        layout.addStretch(1)
        self._update_docs_tab()
        return widget

    def _build_flow_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        mode_row = QtWidgets.QHBoxLayout()
        mode_label = QtWidgets.QLabel("Mode")
        self.mode_combo = QtWidgets.QComboBox()
        for key in MODE_ORDER:
            config = MODE_DEFINITIONS.get(key)
            if not config:
                continue
            self.mode_combo.addItem(config["label"], key)
        self._set_tooltip(self.mode_combo, "Select the measurement template. You can still edit PVs after choosing a mode.")
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.mode_combo, 1)
        docs_btn = QtWidgets.QPushButton("Mode info")
        docs_btn.setMaximumWidth(120)
        docs_btn.clicked.connect(self._show_mode_info_popup)
        mode_row.addWidget(docs_btn)
        layout.addLayout(mode_row)
        self.mode_description = QtWidgets.QLabel("")
        self.mode_description.setWordWrap(True)
        self.mode_description.setStyleSheet("color: #555;")
        layout.addWidget(self.mode_description)
        intro = QtWidgets.QLabel("Flow steps update based on the selected mode.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.flow_steps_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(self.flow_steps_layout)
        layout.addStretch(1)
        return widget

    def _show_mode_info_popup(self):
        mode_key = self.mode_combo.currentData()
        config = MODE_DEFINITIONS.get(mode_key, {})
        text = []
        text.append(f"Mode: {config.get('label', mode_key)}")
        if config.get("description"):
            text.append(config["description"])
        pv_defaults = config.get("pv_defaults", {})
        text.append("")
        text.append("PV defaults:")
        for field in ("prefix_p", "prefix_r", "sp", "sp_rbv", "act"):
            text.append(f"  {field}: {pv_defaults.get(field, '')}")
        mechanics = config.get("supports_mechanical", False)
        text.append(f"\nMechanical fit: {'enabled' if mechanics else 'disabled'}")
        hint = config.get("mechanical_hint")
        if hint:
            text.append(hint)
        QtWidgets.QMessageBox.information(self, "Mode info", "\n".join(text))

    def _line_edit(self, default):
        edit = PVLineEdit(default)
        edit.setClearButtonEnabled(True)
        return edit

    def _set_tooltip(self, widget, text):
        if widget is not None:
            widget.setToolTip(text.strip())
        return widget

    def _mode_config(self, key=None):
        key = key or self.current_mode_key or pipeline.DEFAULT_MODE
        return MODE_DEFINITIONS.get(key, MODE_DEFINITIONS[pipeline.DEFAULT_MODE])

    def _apply_mode_settings(self, key, force_defaults=False):
        resolved_key = key if key in MODE_DEFINITIONS else pipeline.DEFAULT_MODE
        config = self._mode_config(resolved_key)
        self.current_mode_key = resolved_key
        label = config.get("label", "Mode")
        desc = config.get("description", "")
        self.mode_description.setText(f"{label}: {desc}")
        pv_labels = config.get("pv_labels", {})
        self.pv_prefix_label_p.setText(pv_labels.get("prefix_p", "Prefix P"))
        self.pv_prefix_label_r.setText(pv_labels.get("prefix_r", "Prefix R"))
        self.pv_sp_label.setText(pv_labels.get("sp", "SP"))
        self.pv_sp_rbv_label.setText(pv_labels.get("sp_rbv", "SP_RBV"))
        self.pv_act_label.setText(pv_labels.get("act", "ACT PV"))
        self.pv_extra_label.setText(pv_labels.get("extra", "Extra log PVs"))

        pv_tooltips = config.get("pv_tooltips", {})
        self._set_tooltip(self.pv_prefix_p, pv_tooltips.get("prefix_p", self.pv_prefix_p.toolTip() or ""))
        self._set_tooltip(self.pv_prefix_r, pv_tooltips.get("prefix_r", self.pv_prefix_r.toolTip() or ""))
        self._set_tooltip(self.pv_sp, pv_tooltips.get("sp", self.pv_sp.toolTip() or ""))
        self._set_tooltip(self.pv_sp_rbv, pv_tooltips.get("sp_rbv", self.pv_sp_rbv.toolTip() or ""))
        self._set_tooltip(self.pv_act, pv_tooltips.get("act", self.pv_act.toolTip() or ""))
        self._set_tooltip(self.pv_extra, pv_tooltips.get("extra", self.pv_extra.toolTip() or ""))

        defaults = config.get("pv_defaults", {})
        for field_name, widget in getattr(self, "_pv_fields", {}).items():
            preserve = config.get("preserve_prefix", False) and field_name in {"prefix_p", "prefix_r"}
            self._update_field_default(field_name, widget, defaults.get(field_name, ""), force_defaults, preserve)

        self._set_tooltip(
            self.measure_btn,
            f"Run {label} using the current excitation and analysis settings.",
        )
        self._set_tooltip(
            self.reanalyze_btn,
            f"Reprocess a saved log using the {label} signal mapping.",
        )

        mech_hint = config.get("mechanical_hint", "")
        supports_mech = bool(config.get("supports_mechanical"))
        hint_text = mech_hint if supports_mech else ""
        self.mechanical_hint_label.setText(hint_text)
        self.mechanical_hint_label.setVisible(bool(hint_text))
        if hasattr(self, "mech_filter_group") and self.mech_filter_group is not None:
            self.mech_filter_group.setEnabled(supports_mech)
        torque_enabled = resolved_key == "cst_velocity"
        for widget in getattr(self, "torque_fields", []):
            widget.setEnabled(torque_enabled)
        velocity_modes = {"cst_velocity", "csv_velocity_bode", "csv_position_tune"}
        vel_enabled = resolved_key in velocity_modes
        for widget in getattr(self, "velocity_fields", []):
            widget.setEnabled(vel_enabled)
        self._update_flow_diagram(config.get("flow_steps"))
        self._update_docs_text(config)

    def _update_field_default(self, name, widget, value, force=False, preserve=False):
        if widget is None:
            return
        value = value or ""
        current = widget.text().strip() if isinstance(widget, QtWidgets.QLineEdit) else widget.toPlainText().strip()
        last_auto = self._last_auto_values.get(name)
        if preserve and current:
            return
        if force or not current or (last_auto is not None and current == last_auto):
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(value)
            else:
                widget.setPlainText(value)
            self._last_auto_values[name] = value

    def _format_signal_label(self, base, units, pv_name):
        label = base or pv_name or ""
        if units:
            label = f"{label} [{units}]" if label else f"[{units}]"
        if pv_name and pv_name not in label:
            label = f"{label} ({pv_name})" if label else pv_name
        return label or (pv_name or "Signal")

    def _update_flow_diagram(self, steps):
        if not hasattr(self, "flow_steps_layout"):
            return
        layout = self.flow_steps_layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        steps = steps or DEFAULT_FLOW_STEPS
        for idx, (title, desc) in enumerate(steps):
            card = QtWidgets.QGroupBox(title)
            card_layout = QtWidgets.QVBoxLayout(card)
            label = QtWidgets.QLabel(desc)
            label.setWordWrap(True)
            card_layout.addWidget(label)
            layout.addWidget(card, 1)
            if idx < len(steps) - 1:
                arrow = QtWidgets.QLabel("→")
                arrow.setAlignment(QtCore.Qt.AlignCenter)
                arrow.setFixedWidth(24)
                layout.addWidget(arrow)
        layout.addStretch(1)

    def _update_docs_text(self, config):
        self._update_docs_tab()

    def _update_docs_tab(self):
        if not hasattr(self, "docs_area") or self.docs_area is None:
            return
        lines = []
        for key in MODE_ORDER:
            info = MODE_DEFINITIONS.get(key)
            if not info:
                continue
            lines.append(f"Mode: {info.get('label', key)}")
            desc = info.get("description")
            if desc:
                lines.append(desc)
            pv_defaults = info.get("pv_defaults", {})
            lines.append("PV defaults:")
            for field in ("prefix_p", "prefix_r", "sp", "sp_rbv", "act"):
                lines.append(f"  {field}: {pv_defaults.get(field, '')}")
            mechanics = info.get("supports_mechanical", False)
            lines.append(f"Mechanical fit: {'enabled' if mechanics else 'disabled'}")
            hint = info.get("mechanical_hint")
            if hint:
                lines.append(hint)
            lines.append("")
        self.docs_area.setText("\n".join(lines).strip())

    def _on_mode_changed(self, index):
        key = self.mode_combo.itemData(index)
        self._apply_mode_settings(key)

    # -----------------------
    # Actions
    # -----------------------
    def _start_measurement(self):
        try:
            pv_cfg, ex_cfg, an_cfg, mech_cfg = self._collect_settings()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Invalid input", str(exc))
            return
        mode_key = self.current_mode_key or pipeline.DEFAULT_MODE
        if mode_key != "logger" and not self._confirm_excitation(ex_cfg):
            self.append_log("Measurement cancelled by user before excitation")
            return
        log_path = self.log_path_edit.text().strip()
        self.last_analysis_settings = an_cfg
        self._run_worker(
            "measure",
            dict(pv=pv_cfg, excitation=ex_cfg, analysis=an_cfg, mechanical=mech_cfg, log_filename=log_path, mode=mode_key),
        )

    def _preview_excitation_signal(self):
        try:
            _, ex_cfg, _, _ = self._collect_settings()
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Invalid excitation", str(exc))
            return
        try:
            t, u, _, freq, _, _ = excite_sine.generate(
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
                preview=False,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Excitation error", str(exc))
            return
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.plot(t, u)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Command")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title("Excitation preview")
        if freq is not None:
            ax2 = ax.twinx()
            ax2.plot(t[: len(freq)], freq, "r--", alpha=0.6)
            ax2.set_ylabel("Frequency [Hz]")
        fig.tight_layout()
        fig.show()

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
        mode_key = self.current_mode_key or pipeline.DEFAULT_MODE
        self._run_worker("reanalyze", dict(log_filename=log_path, analysis=an_cfg, mechanical=mech_cfg, mode=mode_key))

    def _browse_log(self, save):
        initial = self.log_path_edit.text().strip() or os.path.join(os.getcwd(), "autotune_log.pkl")
        if save:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select log destination", initial, "Pickle (*.pkl)")
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select log file", initial, "Pickle (*.pkl)")
        if path:
            self.log_path_edit.setText(path)
            if not save:
                self._load_pv_settings_from_log(path)

    def _load_pv_settings_from_log(self, path):
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:
            self.append_log(f"Failed to load log metadata: {exc}")
            return
        meta = {}
        if isinstance(payload, dict):
            if "metadata" in payload:
                meta = payload.get("metadata") or {}
            elif "pv_settings" in payload:
                meta = payload
        settings = meta.get("pv_settings") if isinstance(meta, dict) else None
        if settings:
            self._restore_pv_settings_from_meta(settings)
        summary = []
        if isinstance(meta, dict):
            mode = meta.get("mode")
            if mode:
                summary.append(f"Mode: {mode}")
            if "excitation" in meta:
                ex = meta["excitation"] or {}
                summary.append(
                    f"Excitation fs={ex.get('fs', '?')} Hz, range {ex.get('f_start', '?')}–{ex.get('f_stop', '?')} Hz"
                )
            if "analysis" in meta:
                an = meta["analysis"] or {}
                summary.append(
                    f"Analysis block={an.get('block_len_s', '?')} s, overlap={an.get('overlap', '?')}"
                )
            if "pid_targets" in meta:
                pid = meta["pid_targets"] or {}
                summary.append(
                    f"PID targets bw={pid.get('pi_bandwidth', '?')} Hz, zeta={pid.get('pi_zeta', '?')}"
                )
        if hasattr(self, "log_meta_label"):
            text = f"Loaded {Path(path).name}\n" + ("\n".join(summary) if summary else "")
            self.log_meta_label.setText(text.strip() or "No metadata available.")
        self._update_docs_tab()

    def _restore_pv_settings_from_meta(self, settings):
        def set_line(widget, name, value):
            value = value or ""
            widget.setText(value)
            self._last_auto_values[name] = value

        if not isinstance(settings, dict):
            return
        set_line(self.pv_prefix_p, "prefix_p", settings.get("prefix_p", ""))
        set_line(self.pv_prefix_r, "prefix_r", settings.get("prefix_r", ""))
        set_line(self.pv_sp, "sp", settings.get("sp", ""))
        set_line(self.pv_sp_rbv, "sp_rbv", settings.get("sp_rbv", ""))
        set_line(self.pv_act, "act", settings.get("act", ""))
        extra_logs = settings.get("extra_logs") or {}
        lines = []
        for key, value in extra_logs.items():
            key = key or value
            if not key:
                continue
            if value and value != key:
                lines.append(f"{key}={value}")
            else:
                lines.append(key)
        text = "\n".join(lines)
        self.pv_extra.setPlainText(text)
        self._last_auto_values["extra_logs"] = text

    def _collect_settings(self):
        pv_cfg = pipeline.PVSettings(
            prefix_p=self.pv_prefix_p.text().strip(),
            prefix_r=self.pv_prefix_r.text().strip(),
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
            motor_rated_trq=self._optional_float(self.pv_motor_torque) or 0.5,
            torque_scale=self._optional_float(self.pv_torque_scale),
            velocity_scale=self._optional_float(self.pv_velocity_scale) or 1.0,
            smooth_hz=self._float(self.me_smooth, "Smooth cutoff"),
            deriv_hz=self._float(self.me_deriv, "Derivative cutoff"),
            vel_deadband=self._float(self.me_deadband, "Velocity deadband"),
            pi_bandwidth=self._float(self.pid_bw, "Target bandwidth"),
            pi_zeta=self._float(self.pid_zeta, "Target zeta"),
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
            self.last_extra_series = {}
            self._update_extra_pv_list({})
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

        mode_config = self._mode_config(getattr(result, "mode", None))
        time_cfg = mode_config.get("time_plot", {})
        cmd_label = self._format_signal_label(time_cfg.get("command_label"), time_cfg.get("command_units"), result.command_key)
        resp_label = self._format_signal_label(time_cfg.get("response_label"), time_cfg.get("response_units"), result.response_key)
        ylabel = time_cfg.get("ylabel") or "Signal"
        t = result.t if result.t.size else None
        cmd = result.values_by_pv.get(result.command_key)
        resp = result.values_by_pv.get(result.response_key)
        self.last_time_data = dict(
            t=t,
            cmd=cmd,
            resp=resp,
            cmd_label=cmd_label,
            resp_label=resp_label,
            ylabel=ylabel,
        )
        extras = {
            name: np.asarray(values)
            for name, values in result.values_by_pv.items()
            if name not in (result.command_key, result.response_key)
        }
        self.last_extra_series = extras
        self.derived_extra_series = self._build_derived_series(result)
        self._update_extra_pv_list(extras, self.derived_extra_series)
        self._refresh_time_plot()

    def _update_extra_pv_list(self, extras, derived=None):
        if self.extra_pv_list is None:
            return
        current = set(self._selected_extra_pvs())
        self.extra_pv_list.blockSignals(True)
        self.extra_pv_list.clear()
        for name in sorted(extras.keys()):
            item = QtWidgets.QListWidgetItem(name)
            if name in current:
                item.setSelected(True)
            self.extra_pv_list.addItem(item)
        if derived:
            self.extra_pv_list.addItem("----------------")
            for name in sorted(derived.keys()):
                item = QtWidgets.QListWidgetItem(name)
                if name in current:
                    item.setSelected(True)
                self.extra_pv_list.addItem(item)
        self.extra_pv_list.blockSignals(False)

    def _selected_extra_pvs(self):
        if self.extra_pv_list is None:
            return []
        return [item.text() for item in self.extra_pv_list.selectedItems() if item.text().strip("-")]

    def _series_for_name(self, name):
        if name in self.last_extra_series:
            return self.last_extra_series.get(name)
        if name in self.derived_extra_series:
            return self.derived_extra_series.get(name)
        return None

    def _build_derived_series(self, result):
        derived = {}
        if result is None or result.t.size == 0:
            return derived
        t_len = len(result.t)
        segments = getattr(result, "segments", None) or []
        if segments:
            mask = np.zeros(t_len, dtype=float)
            freq = np.full(t_len, np.nan, dtype=float)
            for seg in segments:
                if not isinstance(seg, (list, tuple)) or len(seg) < 2:
                    continue
                start = max(int(seg[0]), 0)
                end = min(int(seg[1]), t_len)
                if end <= start:
                    continue
                mask[start:end] = 1.0
                if len(seg) >= 3 and np.isfinite(seg[2]):
                    freq[start:end] = float(seg[2])
            if np.any(mask > 0):
                derived["Segments mask"] = mask
            if np.isfinite(freq).any():
                derived["Segment freq [Hz]"] = freq
        return derived

    def _refresh_time_plot(self):
        ax_sig = self.time_canvas.axes[0][0]
        ax_sig.cla()
        data = self.last_time_data
        if not data or data["t"] is None:
            ax_sig.text(0.5, 0.5, "No data", ha="center", va="center")
            self.time_canvas.draw_idle()
            return
        t = data["t"]
        cmd = data["cmd"]
        resp = data["resp"]
        cmd_label = data["cmd_label"]
        resp_label = data["resp_label"]
        ylabel = data["ylabel"]
        if cmd is not None:
            ax_sig.plot(t[: len(cmd)], cmd, label=cmd_label)
        if resp is not None:
            ax_sig.plot(t[: len(resp)], resp, label=resp_label)
        if (
            self.segment_overlay_cb
            and self.segment_overlay_cb.isChecked()
            and self.latest_result
            and getattr(self.latest_result, "segments", None)
        ):
            fs = 1.0 / float(self.last_analysis_settings.sample_hz or self.latest_result.t[1] - self.latest_result.t[0] or 1.0)
            for seg in getattr(self.latest_result, "segments", []) or []:
                if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                    start = seg[0] / fs
                    end = seg[1] / fs
                    ax_sig.axvspan(start, end, color="yellow", alpha=0.15)
        ax_sig.set_xlabel("Time [s]")
        ax_sig.set_ylabel(ylabel)
        ax_sig.grid(True, linestyle="--", alpha=0.4)
        ax_sig.legend()
        self.time_canvas.draw_idle()

    def _append_pid_result(self, label, kp, ki, kd, ti=None):
        if not hasattr(self, "pid_result_model") or self.pid_result_model is None:
            return
        items = [
            QtGui.QStandardItem(f"{label}"),
            QtGui.QStandardItem(f"{kp:.4g}"),
            QtGui.QStandardItem(f"{ki:.4g}"),
            QtGui.QStandardItem(f"{kd:.4g}"),
            QtGui.QStandardItem(f"{ti:.4g}" if ti is not None and np.isfinite(ti) else "∞"),
        ]
        for item in items:
            item.setEditable(False)
        self.pid_result_model.appendRow(items)
        print("[PID] table rows:", self.pid_result_model.rowCount())

    def _clear_pid_results(self):
        if not hasattr(self, "pid_result_model") or self.pid_result_model is None:
            return
        self.pid_result_model.removeRows(0, self.pid_result_model.rowCount())

    def _report_result(self, result):
        print(f"[PID] _report_result called; mechanical={'yes' if bool(getattr(result,'mechanical',None)) else 'no'} mode={getattr(result,'mode',None)}")
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
                print("[PID] appending velocity PI row", mech.get("kp"), mech.get("ki"), mech.get("ti"))
                self._append_pid_result("Velocity PI", mech['kp'], mech['ki'], 0.0, mech.get('ti'))
        if result.position_pid:
            pid = result.position_pid
            source = "bode fit" if pid.get("gain_from_bode") else "targets"
            self.append_log(
                "Position PID -> "
                f"Kp={pid['kp']:.4g}, Ki={pid['ki']:.4g}, Kd={pid['kd']:.4g}, Ti={pid['ti']:.4g} "
                f"(target {pid['target_bw_hz']:.4g} Hz, zeta={pid['zeta']:.3g}, source={source})"
            )
            self._append_pid_result("Position PID", pid['kp'], pid['ki'], pid['kd'], pid.get('ti'))

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
        data = self.last_time_data
        if not data or data["t"] is None:
            QtWidgets.QMessageBox.information(self, "No data", "Run a measurement/reanalysis first.")
            return
        t = data["t"]
        cmd = data["cmd"]
        resp = data["resp"]
        cmd_label = data["cmd_label"]
        resp_label = data["resp_label"]
        ylabel = data["ylabel"]
        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
        if cmd is not None:
            ax.plot(t[: len(cmd)], cmd, label=cmd_label)
        if resp is not None:
            ax.plot(t[: len(resp)], resp, label=resp_label)
        for name in self._selected_extra_pvs():
            series = self.last_extra_series.get(name)
            if series is None:
                continue
            ax.plot(t[: len(series)], series, label=name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        ax.set_title("Command vs Response")
        fig.tight_layout()
        fig.show()

    def _show_selected_pv_popup(self):
        if not self.last_time_data:
            QtWidgets.QMessageBox.information(self, "No data", "Run a measurement and select extra PVs to plot.")
            return
        selected = self._selected_extra_pvs()
        if not selected:
            QtWidgets.QMessageBox.information(self, "No selection", "Select one or more extra PVs from the list.")
            return
        t = self.last_time_data.get("t")
        if t is None:
            QtWidgets.QMessageBox.information(self, "No data", "No time base available to plot.")
            return
        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
        plotted = False
        for name in selected:
            series = self._series_for_name(name)
            if series is None:
                continue
            ax.plot(t[: len(series)], series, label=name)
            plotted = True
        if not plotted:
            QtWidgets.QMessageBox.information(self, "No data", "Selected PVs have no samples to plot.")
            plt.close(fig)
            return
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Selected PVs")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        ax.set_title("Extra PVs")
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
