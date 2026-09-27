"""Neutron/gamma pulse-shape discrimination (PSD) on the API tab.

Runs taken with a PSD-capable detector store a short PIXIE trace with every API
event (the parquet ``Trace`` column). The API controller turns them into a
:class:`wara.neutron_psd.TracePSD` at load; ticking **Neutron run** computes a
PSD value per event and adds a PSD-vs-energy panel whose rubber band cuts the
events to neutrons or gammas like any other API cut.

This module holds the pieces that don't need the controller's state: the
detector-name lookup, the PSD panel drawing, and :class:`PsdGatesDialog`, the
pop-out where the trigger-relative gates are tuned on a sample of aligned
traces.
"""
import json

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from wara.neutron_psd import (
    DEFAULT_FF_THRESHOLD, DEFAULT_PRE_NS, DEFAULT_PROMPT_NS,
)

from . import theme as T
from .api_common import API_PLOT_BG, GRAY_CMAP
from .widgets import SpinBox, header, hsep, labeled_row

#: Bins on both axes of the PSD-vs-energy panel ("PSD bins" box).
DEFAULT_PBINS = 300
PSD_CMAP = "jet"


def detector_name(date, runnr, ch):
    """Name of the detector on channel *ch*, from the run's ``metadata.json``
    (e.g. ``"EJ250"``), or ``None`` when the file or the entry is missing."""
    from wara import helper_api

    try:
        meta = json.loads(
            (helper_api.find_data_path(date, runnr) / "metadata.json").read_text())
    except Exception:  # noqa: BLE001 -- metadata is optional
        return None
    for name, det in (meta.get("detectors") or {}).items():
        if isinstance(det, dict) and det.get("channel") == ch:
            return str(name)
    return None


def psd_view(psd, pad=0.05):
    """PSD-axis view: the 1–99 % band of the finite values plus a margin, so a
    few far outliers (noisy pulses with Q_total near 0) can't squash the
    bands. ``(0, 1)`` when there is nothing to go on."""
    vals = np.asarray(psd, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(vals, [1.0, 99.0])
    margin = pad * (hi - lo) if hi > lo else 0.05
    return float(lo - margin), float(hi + margin)


def draw_psd_panel(ax, energy, psd, bins, erange, view, xlabel, ghost=None):
    """2-D energy-vs-PSD histogram (log colour) on *ax*, opened on the PSD
    *view*. Every finite pulse is counted: PSD outliers land in overflow bins
    past the view (the Neutrons tab's binning) so the in-view resolution stays
    *bins* across the band.

    *ghost* -- ``(energy, psd)`` of the uncut run while a cut is applied -- is
    drawn first as a faint grey density on the same bins, so the kept events
    (in colour, empty bins left transparent) read against the whole run, like
    the uncut outline behind the spectra."""
    from .neutrons import NeutronsPage

    def finite(e, p):
        e = np.asarray(e, dtype=float); p = np.asarray(p, dtype=float)
        ok = np.isfinite(e) & np.isfinite(p)
        return e[ok], p[ok]

    energy, psd = finite(energy, psd)
    if ghost is not None:
        ghost = finite(*ghost)
        if ghost[0].size == 0:
            ghost = None
    ax.set_facecolor(API_PLOT_BG)
    # Shared bins: the ghost is the superset, so its PSD extremes set the
    # overflow edges when it is drawn.
    edges_from = ghost[1] if ghost is not None else psd
    edges = (NeutronsPage._psd_edges(edges_from, view, bins)
             if edges_from.size else None)
    if ghost is not None:
        ax.hist2d(ghost[0], ghost[1], bins=[bins, edges],
                  range=[list(erange), None], norm=LogNorm(), cmap=GRAY_CMAP,
                  cmin=1, zorder=1)
    if energy.size:
        ax.hist2d(energy, psd, bins=[bins, edges], range=[list(erange), None],
                  norm=LogNorm(), cmap=PSD_CMAP, cmin=1, zorder=2)
    else:
        ax.text(0.5, 0.5, "No pulses with a valid PSD", transform=ax.transAxes,
                ha="center", va="center", color=T.TEXT_DIM, fontsize=12)
    ax.set_xlim(*erange)
    ax.set_ylim(*view)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PSD = 1 - Q_prompt / Q_total")


def gates_summary(pre_ns, prompt_ns, tail_ns, threshold, detector=None):
    """One-line description of the gates for the options panel / run log."""
    tail = "end of trace" if tail_ns is None else f"+{tail_ns:g} ns"
    head = f"{detector} · " if detector else ""
    return (f"{head}gate −{pre_ns:g} ns → prompt +{prompt_ns:g} ns → "
            f"tail {tail} · trigger {threshold:g} ADC")


class PsdGatesDialog(QDialog):
    """Tune the PSD gates on a sample of trigger-aligned traces.

    The x-axis is time from each pulse's fast-filter trigger, so the three
    dashed markers read directly as the gate offsets: drag them (or type the
    values) and press **Apply** to recompute the PSD of the whole run. The
    traces are coloured by their current PSD, so neutron-like and gamma-like
    pulses stand apart. The sample stays fixed until **New random sample**.
    """

    MARKERS = (("pre", "gate start", T.ACCENT_CYAN),
               ("prompt", "prompt end", T.ACCENT_AMBER),
               ("tail", "tail end", T.ACCENT_GREEN))

    def __init__(self, ctrl, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self.setWindowTitle("PSD gates")
        self.setStyleSheet(T.STYLESHEET)
        self.resize(1000, 620)
        self._lines = {}
        self._drag = None
        self._sample = None

        root = QHBoxLayout(self)
        self.fig = Figure(figsize=(7, 5), constrained_layout=True,
                          facecolor=API_PLOT_BG)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111)
        plot_col = QVBoxLayout(); plot_col.setContentsMargins(0, 0, 0, 0)
        self.toolbar = NavToolbar(self.canvas, self)
        self.toolbar.setObjectName("plot_toolbar")
        self.toolbar.setIconSize(QSize(22, 22))
        T.recolor_toolbar_icons(self.toolbar, T.TEXT_PRIMARY)
        plot_col.addWidget(self.toolbar, 0)
        plot_col.addWidget(self.canvas, 1)
        root.addLayout(plot_col, 1)

        side = QWidget(); side.setFixedWidth(270)
        lay = QVBoxLayout(side); lay.setContentsMargins(6, 6, 6, 6); lay.setSpacing(6)
        root.addWidget(side, 0)

        self.lbl_det = QLabel(""); self.lbl_det.setWordWrap(True)
        self.lbl_det.setObjectName("stat_key")
        lay.addWidget(header("GATES (ns FROM TRIGGER)"))
        lay.addWidget(self.lbl_det)
        self.ed_pre = QLineEdit(); self.ed_prompt = QLineEdit(); self.ed_tail = QLineEdit()
        for lbl, ed, tip in (
                ("Gate start −", self.ed_pre,
                 "How long before the trigger the gate opens (ns). The baseline "
                 "is the mean of the samples before it."),
                ("Prompt end +", self.ed_prompt,
                 "When the prompt window closes, after the trigger (ns)"),
                ("Tail end +", self.ed_tail,
                 "When the tail window closes, after the trigger (ns). Ignored "
                 "while 'Tail to end of trace' is ticked.")):
            ed.setFixedWidth(80); ed.setToolTip(tip)
            ed.returnPressed.connect(self._fields_changed)
            row, _ = labeled_row(lbl, ed); row.setToolTip(tip)
            lay.addWidget(row)
        self.cb_to_end = QCheckBox("Tail to end of trace")
        self.cb_to_end.setToolTip("Integrate the tail all the way to the last "
                                  "sample of each trace")
        self.cb_to_end.toggled.connect(self._fields_changed)
        lay.addWidget(self.cb_to_end)

        lay.addWidget(hsep()); lay.addWidget(header("TRIGGER"))
        self.ed_thresh = QLineEdit(); self.ed_thresh.setFixedWidth(80)
        self.ed_thresh.setToolTip(
            "Fast-filter trigger threshold in ADC units: each trace is timed "
            "where its trapezoidal fast filter (rise 5 / gap 2 samples) first "
            "rises through it. Changing it re-times every trace on Apply.")
        row, _ = labeled_row("Threshold (ADC)", self.ed_thresh)
        row.setToolTip(self.ed_thresh.toolTip())
        lay.addWidget(row)

        lay.addWidget(hsep()); lay.addWidget(header("DISPLAY"))
        self.spin_n = SpinBox(); self.spin_n.setRange(10, 2000)
        self.spin_n.setSingleStep(10); self.spin_n.setValue(150)
        self.spin_n.setMaximumWidth(84)
        self.spin_n.setToolTip("How many traces to draw (a random sample of the "
                               "events that pass the current cuts)")
        row, _ = labeled_row("Traces", self.spin_n); lay.addWidget(row)
        self.spin_n.valueChanged.connect(lambda *_: self.redraw())
        self.cb_log = QCheckBox("Log Y")
        self.cb_log.setToolTip("Logarithmic amplitude axis: shows the tail")
        self.cb_log.toggled.connect(lambda *_: self.redraw())
        lay.addWidget(self.cb_log)
        self.btn_resample = QPushButton("New random sample")
        self.btn_resample.setObjectName("primary_btn")
        self.btn_resample.setCursor(Qt.PointingHandCursor)
        self.btn_resample.setToolTip("Draw a different random sample of traces")
        self.btn_resample.clicked.connect(self.resample)
        lay.addWidget(self.btn_resample)

        lay.addWidget(hsep())
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setObjectName("open_btn")
        self.btn_apply.setCursor(Qt.PointingHandCursor)
        self.btn_apply.setToolTip(
            "Recompute the PSD of every event with these gates and redraw the "
            "API panels. An active PSD cut is dropped (it was drawn against the "
            "old PSD values), which clears the other cuts too.")
        self.btn_apply.clicked.connect(self._apply)
        self.btn_defaults = QPushButton("Defaults")
        self.btn_defaults.setObjectName("mini_btn")
        self.btn_defaults.setCursor(Qt.PointingHandCursor)
        self.btn_defaults.setToolTip(
            f"Gate −{DEFAULT_PRE_NS:g} ns, prompt +{DEFAULT_PROMPT_NS:g} ns, "
            f"tail to the end of the trace, trigger {DEFAULT_FF_THRESHOLD:g} ADC")
        self.btn_defaults.clicked.connect(self._defaults)
        for b in (self.btn_resample, self.btn_apply, self.btn_defaults):
            b.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        lay.addWidget(self.btn_apply)
        lay.addWidget(self.btn_defaults)
        self.lbl_status = QLabel(""); self.lbl_status.setWordWrap(True)
        self.lbl_status.setObjectName("stat_key")
        lay.addWidget(self.lbl_status)
        lay.addStretch(1)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.load_from_psd()

    # -- state <-> fields -------------------------------------------------------
    @property
    def tpsd(self):
        return self.ctrl._tpsd

    def load_from_psd(self):
        """Fill the fields from the controller's current TracePSD and redraw."""
        p = self.tpsd
        self.lbl_det.setText(self.ctrl._psd_detector_label())
        self.ed_pre.setText(f"{p.pre_ns:g}")
        self.ed_prompt.setText(f"{p.prompt_ns:g}")
        self.cb_to_end.blockSignals(True)
        self.cb_to_end.setChecked(p.tail_ns is None)
        self.cb_to_end.blockSignals(False)
        self.ed_tail.setText("" if p.tail_ns is None else f"{p.tail_ns:g}")
        self.ed_tail.setEnabled(p.tail_ns is not None)
        self.ed_thresh.setText(f"{p.threshold:g}")
        self._sample = None
        self.redraw()

    def gates(self):
        """``(pre_ns, prompt_ns, tail_ns or None, threshold)`` from the fields,
        or None (with a status message) when a field is not a valid number."""
        try:
            pre = float(self.ed_pre.text())
            prompt = float(self.ed_prompt.text())
            thresh = float(self.ed_thresh.text())
            tail = None if self.cb_to_end.isChecked() else float(self.ed_tail.text())
        except ValueError:
            self.lbl_status.setText("Every gate and the threshold must be a number")
            return None
        if pre <= 0 or prompt <= 0 or thresh <= 0:
            self.lbl_status.setText("Gate start, prompt end and threshold must "
                                    "be positive")
            return None
        if tail is not None and tail <= prompt:
            self.lbl_status.setText("The tail must end after the prompt window")
            return None
        return pre, prompt, tail, thresh

    def _fields_changed(self, *_):
        self.ed_tail.setEnabled(not self.cb_to_end.isChecked())
        if self.cb_to_end.isChecked():
            self.ed_tail.setText("")
        elif not self.ed_tail.text().strip():
            self.ed_tail.setText(f"{self._record_end():g}")
        self._place_markers()
        self.canvas.draw_idle()

    def _defaults(self):
        self.ed_pre.setText(f"{DEFAULT_PRE_NS:g}")
        self.ed_prompt.setText(f"{DEFAULT_PROMPT_NS:g}")
        self.ed_thresh.setText(f"{DEFAULT_FF_THRESHOLD:g}")
        self.cb_to_end.setChecked(True)
        self._fields_changed()

    def _apply(self):
        g = self.gates()
        if g is None:
            return
        self.ctrl._flash_button(self.btn_apply)
        self.lbl_status.setText("Recomputing PSD…")
        self.lbl_status.repaint()
        self.ctrl.apply_psd_gates(*g)
        p = self.tpsd
        self.lbl_status.setText(
            f"{int(p.valid.sum()):,} of {p.n_events:,} events have a valid PSD")
        self.redraw()

    # -- drawing ---------------------------------------------------------------
    def _record_end(self):
        """Time of the last sample, relative to the common trigger (ns)."""
        p = self.tpsd
        return float(p.time_ns[-1] - p.trigger_ns)

    def resample(self):
        self.ctrl._flash_button(self.btn_resample)
        self._sample = None
        self.redraw(reshuffle=True)

    def redraw(self, reshuffle=False):
        p = self.tpsd
        n = self.spin_n.value()
        if self._sample is None or len(self._sample) != n or reshuffle:
            self._sample = self.ctrl.psd_trace_sample(n, reshuffle=reshuffle)
        idx = self._sample
        ax = self.ax
        ax.clear()
        ax.set_facecolor(API_PLOT_BG)
        t = p.time_ns - p.trigger_ns
        if idx.size:
            traces = p.aligned(idx)
            vals = p.psd[idx]
            # Colour over the 5-95 % PSD band so the gamma and neutron bands
            # take clearly different colours.
            lo, hi = np.percentile(p.psd[p.valid], [5.0, 95.0])                 if p.valid.any() else (0.0, 1.0)
            norm = Normalize(vmin=lo, vmax=hi)
            cmap = colormaps["plasma"]
            for i in np.argsort(np.nan_to_num(vals, nan=lo)):
                c = cmap(norm(vals[i])) if np.isfinite(vals[i]) else T.TEXT_DIM
                ax.plot(t, traces[i], lw=0.6, alpha=0.6, color=c)
            ax.set_title(f"{idx.size} traces aligned on the fast-filter trigger "
                         "· coloured by PSD", color=T.TEXT_PRIMARY, fontsize=12)
        else:
            ax.text(0.5, 0.5, "No events with a trace in the current cut",
                    transform=ax.transAxes, ha="center", va="center",
                    color=T.TEXT_DIM, fontsize=12)
        ax.axvline(0.0, color=T.TEXT_DIM, lw=0.8, ls=":")
        self._lines = {}
        for key, label, color in self.MARKERS:
            self._lines[key] = ax.axvline(0.0, color=color, ls="--", lw=1.5,
                                          label=label, zorder=6)
        self._place_markers()
        ax.set_xlim(-3 * max(p.pre_ns, 10.0), float(t[-1]))
        if self.cb_log.isChecked():
            ax.set_yscale("log")
            # Baseline noise dips to ~0 after subtraction; floor at 1 ADC so
            # the pulse tails fill the axis instead of decades of noise.
            ax.set_ylim(bottom=1.0)
        ax.set_xlabel("Time from trigger (ns)")
        ax.set_ylabel("Amplitude (ADC, baseline-subtracted)")
        leg = ax.legend(loc="upper right", fontsize=12, facecolor=API_PLOT_BG,
                        edgecolor=T.BORDER)
        for txt in leg.get_texts():
            txt.set_color(T.TEXT_PRIMARY)
        ax.tick_params(colors=T.TEXT_DIM)
        ax.xaxis.label.set_color(T.TEXT_DIM); ax.yaxis.label.set_color(T.TEXT_DIM)
        for sp in ax.spines.values():
            sp.set_color(T.BORDER)
        self.canvas.draw_idle()

    def _place_markers(self):
        """Move the three markers to the gate values in the fields."""
        if not self._lines:
            return
        pos = {}
        for key, ed, sign in (("pre", self.ed_pre, -1.0),
                              ("prompt", self.ed_prompt, 1.0)):
            try:
                pos[key] = sign * float(ed.text())
            except ValueError:
                pass
        if self.cb_to_end.isChecked():
            pos["tail"] = self._record_end()
        else:
            try:
                pos["tail"] = float(self.ed_tail.text())
            except ValueError:
                pass
        for key, x in pos.items():
            self._lines[key].set_xdata([x, x])

    # -- marker dragging -------------------------------------------------------
    def _on_press(self, event):
        if (event.inaxes is not self.ax or event.button != 1
                or event.xdata is None or self.toolbar.mode):
            return                          # zoom/pan owns the mouse
        # Grab the nearest marker within ~8 px.
        best, best_d = None, 8.0
        for key, line in self._lines.items():
            x = line.get_xdata()[0]
            px = self.ax.transData.transform((x, 0))[0]
            d = abs(px - event.x)
            if d < best_d:
                best, best_d = key, d
        self._drag = best

    def _on_motion(self, event):
        if self._drag is None or event.inaxes is not self.ax or event.xdata is None:
            return
        x = float(event.xdata)
        self._lines[self._drag].set_xdata([x, x])
        self._write_field(self._drag, x)
        self.canvas.draw_idle()

    def _on_release(self, event):
        if self._drag is not None:
            self._fields_changed()
        self._drag = None

    def _write_field(self, key, x):
        """Mirror a dragged marker into its field (0.5 ns resolution)."""
        x = round(x * 2) / 2
        if key == "pre":
            self.ed_pre.setText(f"{max(-x, 0.5):g}")
        elif key == "prompt":
            self.ed_prompt.setText(f"{max(x, 0.5):g}")
        else:
            self.cb_to_end.blockSignals(True)
            self.cb_to_end.setChecked(False)
            self.cb_to_end.blockSignals(False)
            self.ed_tail.setEnabled(True)
            self.ed_tail.setText(f"{x:g}")
