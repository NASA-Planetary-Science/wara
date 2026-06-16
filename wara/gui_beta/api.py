"""API tab: explore Associated-Particle-Imaging parquet runs.

Port of the legacy ``ApiMixin`` main tab (``wara/gui/_mixins/api.py``), restyled
for the dark beta GUI and split into the Options / Page / Controller trio used by
the other beta tabs.

Workflow:

* Enter a run (date, run number, channel) and pick the data type — experimental
  *raw*, experimental *calibrated/time-shifted*, or *simulated* — then Load. The
  parquet file is read via :mod:`wara.read_parquet_api`; raw/calibrated data get
  reconstructed X-Y positions through :func:`wara.apicalc.calc_own_pos`.
* The figure shows three linked panels: the energy spectrum (top-left), the dt
  time histogram (bottom-left), and the X-Y hit map as a hexbin (right).
* Dragging a span on the energy or time panel, or a rectangle on the X-Y map,
  filters the event list live; the other panels redraw from the filtered data.
  The same filters can be entered by hand from the Filters dialog.
* "Send energy spectrum → Spectrum tab" hands the current energy histogram to the
  Spectrum tab for the full analysis workflow. An uncalibrated spectrum is sent
  with its *real* channel values (the bin centres) as the channel axis — not
  0..nbins indices — so a calibration built on it round-trips back here.
* "3D view" pops out a Plotly volume render (:class:`Api3DDialog`) of the
  reconstructed X-Y-Z hit cloud for the current (filtered) events.
* "Retrieve calibration" pulls the Calibration tab's current calibration curve
  (or equation) and applies that channel→energy polynomial to the dataframe —
  it is applied to the *real* per-event channel values (which run past the bin
  count, e.g. 0–65535). Afterwards every panel and the selections read in
  calibrated energy; "Clear calibration" reverts to the original raw channels.
* Energy *selections*: arm "Add selection", drag a band on the energy spectrum,
  then label it and pick a colour (:class:`EnergySelectionDialog`). Each
  selection stores the energy-cut dataframe. "Plot selections" overlays them in
  their colours: the energy band is shaded, dt shows a coloured histogram per
  selection over a gray base, and the X-Y map overlays a translucent coloured
  hexbin per selection over the gray density hexbin. The 3D view renders one
  coloured volume per selection.

The X-Y map has no colorbar; instead, hovering a hexagon reports its intensity
(count) in amber in the cursor readout under the figures, alongside the X/Y
position.

Bug fixes carried over from the legacy port are flagged inline (mutually
exclusive data-type radios; guarded numeric inputs; the X-Y panel now redraws
from the *filtered* data after an X-Y selection).
"""
import traceback
import time

import numpy as np

from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Patch
from matplotlib.widgets import SpanSelector, RectangleSelector
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLineEdit, QRadioButton, QButtonGroup, QCheckBox, QSizePolicy, QDialog,
    QGridLayout, QMessageBox, QColorDialog, QDialogButtonBox, QTabWidget,
    QComboBox,
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, QSize, QTimer, QUrl

from wara import read_parquet_api, apicalc
from wara import spectrum as sp

from . import theme as T
from .widgets import hsep, header, labeled_row

API_PLOT_BG = T.BG_PLOT
COLORMAP = "plasma"
# Faint gray density map for the *base* X-Y hexbin when colored energy
# selections are overlaid on top: low counts fade into the plot background, high
# counts brighten to a dim gray so the selections stay the focus.
GRAY_CMAP = LinearSegmentedColormap.from_list("api_gray", [API_PLOT_BG, T.TEXT_DIM])

# Data-type choices (mutually exclusive — see ApiOptions radios).
TYPE_RAW, TYPE_CAL, TYPE_SIMS = "raw", "calibrated", "sims"

# Send-to-Spectrum button: resting label vs the green "Sent" confirmation. Kept
# short (and the same length) so the button fits the original-width options panel
# without forcing it wider; the green styling is the "sent" signal.
SEND_DEFAULT_TEXT = "Send to spectrum"
SEND_SENT_TEXT = "Sent to spectrum"


class ApiFilterDialog(QDialog):
    """Manual min/max entry for the X-Y, time and energy filters — the typed
    equivalent of dragging a selector on the plot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API filters")
        self.setStyleSheet(T.STYLESHEET)
        self.setMinimumWidth(320)
        lay = QVBoxLayout(self)
        lay.addWidget(header("MANUAL FILTERS"))
        lay.addWidget(QLabel("Leave a pair blank to skip that filter."))

        grid = QGridLayout()
        grid.addWidget(QLabel("min"), 0, 1)
        grid.addWidget(QLabel("max"), 0, 2)
        self.fields = {}
        for r, (key, label) in enumerate(
                [("x", "X"), ("y", "Y"), ("t", "dt"), ("e", "Energy")], start=1):
            grid.addWidget(QLabel(label), r, 0)
            lo, hi = QLineEdit(), QLineEdit()
            for e in (lo, hi):
                e.setFixedWidth(90)
            grid.addWidget(lo, r, 1)
            grid.addWidget(hi, r, 2)
            self.fields[key] = (lo, hi)
        lay.addLayout(grid)

        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_apply = QPushButton("Apply"); self.btn_apply.setObjectName("open_btn")
        self.btn_apply.setCursor(Qt.PointingHandCursor)
        row.addWidget(self.btn_apply)
        lay.addLayout(row)

    def pair(self, key):
        """Return (min, max) floats for *key*, or None if either box is empty."""
        lo, hi = self.fields[key]
        lo_t, hi_t = lo.text().strip(), hi.text().strip()
        if lo_t == "" or hi_t == "":
            return None
        try:
            return float(lo_t), float(hi_t)
        except ValueError:
            return None


class EnergySelectionDialog(QDialog):
    """Label + colour entry for a new energy selection, shown after the user
    drags a band on the energy spectrum with 'Add selection' armed."""

    def __init__(self, emin, emax, color, n_events, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label energy selection")
        self.setStyleSheet(T.STYLESHEET)
        self.setMinimumWidth(300)
        self._color = color

        lay = QVBoxLayout(self)
        lay.addWidget(header("ENERGY SELECTION"))
        info = QLabel(f"Range  {emin:g} – {emax:g}\n{n_events:,} events")
        info.setObjectName("stat_key")
        lay.addWidget(info)

        self.ed_label = QLineEdit()
        self.ed_label.setPlaceholderText("e.g. Fe, Si, 511 keV")
        self.ed_label.setFixedWidth(150)
        row, _ = labeled_row("Label", self.ed_label)
        lay.addWidget(row)

        self.btn_color = QPushButton("Pick colour")
        self.btn_color.setCursor(Qt.PointingHandCursor)
        self.btn_color.clicked.connect(self._pick_color)
        crow, _ = labeled_row("Colour", self.btn_color)
        lay.addWidget(crow)
        self._apply_swatch()

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _pick_color(self):
        col = QColorDialog.getColor(QColor(self._color), self, "Selection colour")
        if col.isValid():
            self._color = col.name()
            self._apply_swatch()

    def _apply_swatch(self):
        # Paint the button in the chosen colour so it doubles as a live swatch.
        self.btn_color.setStyleSheet(
            f"background-color:{self._color}; color:{T.BG_DARK}; "
            f"border:2px solid {self._color}; border-radius:5px; "
            f"padding:6px 12px; font-weight:800;")

    def label(self):
        return self.ed_label.text().strip()

    def color(self):
        return self._color


class Api3DDialog(QDialog):
    """Pop-out Plotly volume render of the reconstructed X-Y-Z hit cloud.

    Port of the legacy ``WindowAPI3D`` / ``create_plot_api3D``. The controller
    fills :attr:`browser` (a ``QWebEngineView``) with a Plotly ``Volume`` figure
    built from the *current* (filtered) event dataframe; the controls here only
    set the histogram resolution and isosurface look. Like the legacy window the
    gamma-detector position is ignored (``api_xyz(use_det=False)``), so no
    detector-position fields are exposed."""

    # (label, attribute, default) for the numeric controls.
    FIELDS = [
        ("No. of bins", "no_bins", "50"),
        ("Iso min", "isomin", "0.1"),
        ("Iso max", "isomax", "0.8"),
        ("Opacity", "opacity", "0.1"),
        ("Surface count", "surfcount", "20"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API 3D volume")
        self.setStyleSheet(T.STYLESHEET)
        # Maximise/minimise help when orbiting the volume.
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.resize(1040, 700)

        # QtWebEngine is imported lazily (as the legacy GUI does via its .ui) so
        # the offscreen test suite never has to load it. AA_ShareOpenGLContexts
        # is already set in app.main() before the QApplication is created.
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        from PyQt5.QtGui import QColor

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(8)

        self.browser = QWebEngineView()
        self.browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.browser.setMinimumWidth(560)
        # The QWebEngineView renders white by default — paint the page background
        # dark so the blank initial page, the load flash, and any margin around
        # the Plotly figure all match the dark theme instead of flashing white.
        self.browser.page().setBackgroundColor(QColor(API_PLOT_BG))
        self.browser.setHtml(
            f"<html><body style='margin:0;background:{API_PLOT_BG}'></body></html>")
        lay.addWidget(self.browser, 1)

        side = QVBoxLayout(); side.setSpacing(6)
        side.addWidget(header("3D VOLUME"))
        note = QLabel("Volume render of the X-Y-Z hit cloud reconstructed from "
                      "the current (filtered) events.")
        note.setObjectName("stat_key"); note.setWordWrap(True)
        side.addWidget(note)

        self.fields = {}
        for label, attr, default in self.FIELDS:
            ed = QLineEdit(default); ed.setFixedWidth(70)
            row, _ = labeled_row(label, ed)
            side.addWidget(row)
            self.fields[attr] = ed

        self.btn_plot = QPushButton("Plot"); self.btn_plot.setObjectName("open_btn")
        self.btn_plot.setCursor(Qt.PointingHandCursor)
        side.addWidget(self.btn_plot)

        self.status = QLabel(""); self.status.setObjectName("stat_key")
        self.status.setWordWrap(True)
        side.addWidget(self.status)
        side.addStretch(1)

        holder = QWidget(); holder.setFixedWidth(230); holder.setLayout(side)
        lay.addWidget(holder, 0)

    def value(self, attr, cast):
        """Read a control as *cast* (int/float), or None if it isn't valid."""
        try:
            return cast(self.fields[attr].text().strip())
        except (ValueError, KeyError):
            return None


class ShiftsDialog(QDialog):
    """Drift-correction window: gain-shift the energy axis and shift/align the
    time axis, independently. Two tabs (Energy, Time), each showing the raw
    segment overlay above the aligned overlay (Preview), with controls to apply
    the correction to the API dataframe via the controller."""

    def __init__(self, controller):
        super().__init__(controller.app)
        self.c = controller
        self.setWindowTitle("Drift correction — shifts")
        self.setStyleSheet(T.STYLESHEET)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.resize(1000, 700)

        # Per-tab widget handles, keyed by "energy"/"time".
        self.ax_raw = {}
        self.ax_aligned = {}
        self.canvas = {}
        self.toolbar = {}
        self.fig = {}
        self.f_nseg = {}
        self.f_method = {}
        self.f_xmin = {}
        self.f_xmax = {}
        self.lbl_state = {}
        self.btn_clear = {}

        lay = QVBoxLayout(self)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)
        self.tabs.addTab(self._build_tab("energy"), "Energy")
        self.tabs.addTab(self._build_tab("time"), "Time")

    # ── construction ──────────────────────────────────────────────────────────
    def _build_tab(self, kind):
        tab = QWidget()
        row = QHBoxLayout(tab); row.setContentsMargins(8, 8, 8, 8); row.setSpacing(8)

        fig = Figure(figsize=(6, 6), facecolor=API_PLOT_BG)
        self.fig[kind] = fig
        self.ax_raw[kind] = fig.add_subplot(211)
        # Share the x-axis so zooming/panning the raw overlay moves the aligned
        # one in lockstep (they are the same axis, before vs after correction).
        self.ax_aligned[kind] = fig.add_subplot(212, sharex=self.ax_raw[kind])
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.setMinimumWidth(540)
        self.canvas[kind] = canvas
        # Pan/zoom toolbar above the plot — styled to match the API page's.
        toolbar = NavToolbar(canvas, self)
        toolbar.setObjectName("plot_toolbar")
        toolbar.setIconSize(QSize(22, 22))
        T.recolor_toolbar_icons(toolbar, T.TEXT_PRIMARY)
        self.toolbar[kind] = toolbar
        plot_col = QVBoxLayout(); plot_col.setContentsMargins(0, 0, 0, 0); plot_col.setSpacing(2)
        plot_col.addWidget(toolbar)
        plot_col.addWidget(canvas, 1)
        plot_w = QWidget(); plot_w.setLayout(plot_col)
        row.addWidget(plot_w, 1)

        side = QVBoxLayout(); side.setSpacing(6)
        axis = "energy channels" if kind == "energy" else "time (dt)"
        side.addWidget(header(f"{kind.upper()} DRIFT"))
        note = QLabel(f"Split the run into time segments and align each segment's "
                      f"{axis} onto the first (earliest) one.")
        note.setObjectName("stat_key"); note.setWordWrap(True)
        side.addWidget(note)

        nseg = QLineEdit("20"); nseg.setFixedWidth(70)
        r, _ = labeled_row("Segments", nseg); side.addWidget(r)
        self.f_nseg[kind] = nseg

        method = QComboBox(); method.addItems(["shift", "linear"])
        method.setToolTip("shift: additive slide (robust, single peak ok)\n"
                          "linear: slope + offset (true gain drift; needs ≥2 peaks)")
        r, _ = labeled_row("Method", method); side.addWidget(r)
        self.f_method[kind] = method

        xmin = QLineEdit(); xmin.setPlaceholderText("min"); xmin.setFixedWidth(70)
        xmax = QLineEdit(); xmax.setPlaceholderText("max"); xmax.setFixedWidth(70)
        xr = QHBoxLayout(); xr.setContentsMargins(0, 0, 0, 0)
        xr.addWidget(QLabel("Window")); xr.addWidget(xmin); xr.addWidget(xmax)
        xrw = QWidget(); xrw.setLayout(xr); side.addWidget(xrw)
        xmin.setToolTip("Optional: restrict the alignment to this axis window")
        self.f_xmin[kind] = xmin; self.f_xmax[kind] = xmax

        btn_prev = QPushButton("Preview"); btn_prev.setObjectName("action_btn")
        btn_prev.setCursor(Qt.PointingHandCursor)
        btn_prev.clicked.connect(lambda _=False, k=kind: self._preview(k))
        side.addWidget(btn_prev)

        btn_apply = QPushButton("Apply alignment"); btn_apply.setObjectName("open_btn")
        btn_apply.setCursor(Qt.PointingHandCursor)
        btn_apply.clicked.connect(lambda _=False, k=kind: self._apply_segments(k))
        side.addWidget(btn_apply)

        if kind == "time":
            # The Time axis additionally keeps the simple manual constant shift.
            side.addWidget(hsep())
            side.addWidget(header("CONSTANT SHIFT"))
            const = QLineEdit(); const.setPlaceholderText("Δt (ns)"); const.setFixedWidth(70)
            r, b = labeled_row("Δt (ns)", const, apply_btn=True)
            const.setToolTip("Add a constant to every dt value (negative shifts left)")
            b.clicked.connect(self._apply_constant)
            const.returnPressed.connect(self._apply_constant)
            self.f_const = const
            side.addWidget(r)

        side.addWidget(hsep())
        clear = QPushButton("Clear correction"); clear.setObjectName("mini_btn")
        clear.setCursor(Qt.PointingHandCursor); clear.setEnabled(False)
        clear.clicked.connect(lambda _=False, k=kind: self._clear(k))
        self.btn_clear[kind] = clear
        side.addWidget(clear)

        state = QLabel("Not applied"); state.setObjectName("stat_key")
        state.setWordWrap(True); self.lbl_state[kind] = state
        side.addWidget(state)
        side.addStretch(1)

        holder = QWidget(); holder.setFixedWidth(250); holder.setLayout(side)
        row.addWidget(holder, 0)
        return tab

    # ── parameter reading ──────────────────────────────────────────────────────
    def _params(self, kind):
        """Return (n_segments, method, xrange) from the tab controls."""
        try:
            nseg = max(2, int(float(self.f_nseg[kind].text().strip())))
        except ValueError:
            nseg = 20
        method = self.f_method[kind].currentText()
        try:
            xr = (float(self.f_xmin[kind].text()), float(self.f_xmax[kind].text()))
            if xr[0] >= xr[1]:
                xr = None
        except ValueError:
            xr = None
        return nseg, method, xr

    def _axis_cfg(self, kind):
        """(base column, out column, bins, base erange) for the tab's axis."""
        c = self.c
        if kind == "energy":
            return (c._chan_base or "energy_orig", "energy_drift",
                    c.ebins, c._chan_erange())
        return ("dt", "dt_cal", c.tbins, c._dt_erange())

    def _segment_spectra(self, col, n_segments, bins, erange):
        """Per-segment histograms of *col* (no alignment, nothing committed)."""
        gs = apicalc.GainShift(self.c.df_api, n_segments=n_segments, bins=bins,
                               erange=erange, col=col, out_col="_preview_tmp")
        return gs._spectra

    # ── actions ─────────────────────────────────────────────────────────────────
    def _preview(self, kind):
        """Redraw the tab: raw segments on top; on the bottom the *committed*
        correction (constant or segment) when one is applied, otherwise a
        prospective segment alignment for the current controls."""
        c = self.c
        if c.df_api is None:
            return
        nseg, method, xr = self._params(kind)
        base, out_col, bins, base_erange = self._axis_cfg(kind)
        applied = c._dt_corrected if kind == "time" else c._egain_applied
        try:
            raw_spectra = self._segment_spectra(base, nseg, bins, base_erange)
            if applied and out_col in c.df_api.columns:
                vals = c.df_api[out_col].astype(float)
                erange = (float(vals.min()), float(vals.max()))
                bottom = self._segment_spectra(out_col, nseg, bins, erange)
                bottom_title = "Corrected"
            else:
                gs = apicalc.GainShift(c.df_api, n_segments=nseg, bins=bins,
                                       erange=base_erange, col=base, out_col=out_col)
                gs.align(method=method, xrange=xr)
                bottom = gs._aligned_spectra
                bottom_title = f"Aligned ({method})"
        except Exception as exc:  # noqa: BLE001 — surface to the state label
            self.lbl_state[kind].setText(f"Preview failed: {exc}")
            return
        xlabel = "Raw channel" if kind == "energy" else "dt (ns)"
        yscale = "log" if kind == "energy" else "linear"
        self._draw_overlay(self.ax_raw[kind], raw_spectra, nseg,
                           f"Raw segments ({nseg})", xlabel, yscale)
        self._draw_overlay(self.ax_aligned[kind], bottom, nseg,
                           bottom_title, xlabel, yscale)
        self.fig[kind].tight_layout()
        self.canvas[kind].draw_idle()
        # Reset the pan/zoom history so "Home" returns to this fresh view.
        self.toolbar[kind].update()

    def _draw_overlay(self, ax, spectra, n_seg, title, xlabel, yscale="log"):
        ax.clear()
        colors = cm.viridis(np.linspace(0, 1, n_seg))
        for i, spe in enumerate(spectra):
            ax.step(spe.energies, spe.counts, where="mid",
                    color=colors[i], lw=0.8, alpha=0.8)
        ax.set_yscale(yscale)
        ax.set_title(title, color=T.TEXT_PRIMARY, fontsize=10)
        ax.set_xlabel(xlabel, color=T.TEXT_DIM, fontsize=9)
        ax.set_facecolor(API_PLOT_BG)
        ax.tick_params(colors=T.TEXT_DIM, which="both", length=3, labelsize=8)
        for sp_ in ax.spines.values():
            sp_.set_color(T.BORDER)

    def _apply_segments(self, kind):
        nseg, method, xr = self._params(kind)
        if kind == "energy":
            self.c.apply_energy_gainshift(nseg, method=method, xrange=xr)
        else:
            self.c.apply_dt_gainshift(nseg, method=method, xrange=xr)
        self._preview(kind)

    def _apply_constant(self):
        txt = self.f_const.text().strip()
        try:
            shift = float(txt)
        except ValueError:
            self.lbl_state["time"].setText("Constant shift must be a number (ns)")
            return
        self.c.apply_dt_shift(shift)
        self._preview("time")

    def _clear(self, kind):
        if kind == "energy":
            self.c.clear_energy_gainshift()
        else:
            self.c._clear_dt_shift()
        self._preview(kind)

    # ── state sync (called by the controller) ───────────────────────────────────
    def set_energy_state(self, applied, label):
        self.lbl_state["energy"].setText(label or "Not applied")
        self.btn_clear["energy"].setEnabled(applied)

    def set_time_state(self, applied, label):
        self.lbl_state["time"].setText(label or "Not applied")
        self.btn_clear["time"].setEnabled(applied)

    def refresh_previews(self):
        if self.c.df_api is None:
            return
        for kind in ("energy", "time"):
            self._preview(kind)


# ── Plot column ───────────────────────────────────────────────────────────────
class ApiPage(QWidget):
    """Plot area for the API tab: the three-panel API figure plus a cursor
    readout. The controller draws onto the axes built here."""

    def __init__(self):
        super().__init__()
        self.setObjectName("content")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(9, 6), constrained_layout=True, facecolor=API_PLOT_BG)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.toolbar = NavToolbar(self.canvas, self)
        self.toolbar.setObjectName("plot_toolbar")
        self.toolbar.setIconSize(QSize(22, 22))
        T.recolor_toolbar_icons(self.toolbar, T.TEXT_PRIMARY)
        lay.addWidget(self.toolbar, 0)
        lay.addWidget(self.canvas, 1)

        self.readout = QLabel("")
        self.readout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.readout.setStyleSheet(
            f"background:{API_PLOT_BG}; font-size:15px; font-weight:700; padding:3px 8px;")
        lay.addWidget(self.readout, 0)

        self.ax_spe = self.ax_dt = self.ax_xy = None
        # Hexbin lookup data for the cursor readout (replaces the colorbar): the
        # per-hexagon centres/counts of the current X-Y map, the log-scale flag,
        # and the squared pick radius (≈ one hex spacing) used to ignore the
        # cursor when it sits away from any drawn hexagon.
        self.xy_offsets = self.xy_values = None
        self.xy_log = False
        self.xy_pick_r2 = 0.0
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("axes_leave_event", lambda *_: self.readout.setText(""))
        self.show_empty()

    def show_empty(self, msg="Enter a run and press Load to explore an API file"):
        self.fig.clf()
        self._clear_xy_data()
        self.readout.setText("")
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color=T.BORDER, fontweight="bold", wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        self.ax_spe = self.ax_dt = self.ax_xy = None
        self._style(ax)
        self.canvas.draw_idle()

    def build_axes(self, flood_field=False):
        """(Re)build the panel layout. Flood-field runs show only the X-Y map."""
        self.fig.clf()
        self._clear_xy_data()
        if flood_field:
            self.ax_xy = self.fig.add_subplot(111)
            self.ax_spe = self.ax_dt = None
            self._style(self.ax_xy, grid=False)
            return
        gs = self.fig.add_gridspec(2, 2, width_ratios=[0.5, 0.5], height_ratios=[1, 1])
        self.ax_spe = self.fig.add_subplot(gs[0, 0])
        self.ax_dt = self.fig.add_subplot(gs[1, 0])
        self.ax_xy = self.fig.add_subplot(gs[:, 1])
        self._style(self.ax_spe)
        self._style(self.ax_dt)
        self._style(self.ax_xy, grid=False)

    def reset_nav(self):
        """Drop the navigation toolbar's zoom/pan history. After a filter redraws
        a panel with a new axis range, the toolbar's 'Home' still holds the
        pre-filter limits; pressing it would restore that stale (wider) view and
        squash the filtered data. Clearing the stack makes the freshly drawn view
        the new home."""
        try:
            self.toolbar.update()
        except Exception:  # noqa: BLE001
            pass

    def _clear_xy_data(self):
        """Forget the previous X-Y hexbin lookup data so a stale map can't feed
        the cursor readout before the next redraw."""
        self.xy_offsets = self.xy_values = None
        self.xy_log = False
        self.xy_pick_r2 = 0.0

    def set_xy_lookup(self, mappable, log, pick_r2):
        """Cache the drawn hexbin's centres/counts for the cursor readout. With
        no colorbar, this is how the user reads off a hexagon's intensity."""
        if mappable is None:
            self._clear_xy_data()
            return
        self.xy_offsets = np.asarray(mappable.get_offsets())
        self.xy_values = np.asarray(mappable.get_array(), dtype=float)
        self.xy_log = bool(log)
        self.xy_pick_r2 = float(pick_r2)

    def xy_count_at(self, x, y):
        """Return the count of the hexagon nearest the cursor, or None when the
        cursor is away from any drawn hexagon (or no map is loaded). Under a log
        color scale the stored array is log10(count), so undo it for display."""
        if self.xy_offsets is None or len(self.xy_offsets) == 0:
            return None
        d2 = ((self.xy_offsets[:, 0] - x) ** 2
              + (self.xy_offsets[:, 1] - y) ** 2)
        i = int(np.argmin(d2))
        if self.xy_pick_r2 and d2[i] > self.xy_pick_r2:
            return None
        val = self.xy_values[i]
        if self.xy_log:
            val = 10 ** val
        return val

    def _style(self, ax, grid=False):
        ax.set_facecolor(API_PLOT_BG)
        ax.tick_params(colors=T.TEXT_DIM, which="both", length=3)
        ax.xaxis.label.set_color(T.TEXT_DIM)
        ax.yaxis.label.set_color(T.TEXT_DIM)
        if ax.get_title():
            ax.title.set_color(T.TEXT_PRIMARY)
        for sp_ in ax.spines.values():
            sp_.set_color(T.BORDER)
        # All three panels read cleaner without grid lines (the hexbin, and the
        # energy/time spectra). NB: ax.grid(False, **style) would force the grid
        # back ON (matplotlib treats supplied line kwargs as "enable the grid"),
        # so pass False alone.
        if grid:
            ax.grid(True, color="#3c3c66", linewidth=0.6, alpha=0.7)
        else:
            ax.grid(False)

    def restyle_all(self):
        for ax in self.fig.axes:
            self._style(ax)

    def _on_motion(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            self.readout.setText("")
            return
        xlab = ax.get_xlabel() or "x"
        ylab = ax.get_ylabel() or "y"
        parts = [
            f"<span style='color:{T.ACCENT_CYAN}'>{xlab}: {event.xdata:.4g}</span>",
            f"<span style='color:{T.ACCENT_GREEN}'>{ylab}: {event.ydata:.4g}</span>",
        ]
        # On the X-Y map, report the hovered hexagon's intensity in amber — this
        # stands in for the colorbar the panel no longer carries.
        if ax is self.ax_xy:
            cnt = self.xy_count_at(event.xdata, event.ydata)
            if cnt is not None:
                parts.append(
                    f"<span style='color:{T.ACCENT_AMBER}'>Counts: {cnt:,.0f}</span>")
        self.readout.setText("&nbsp;&nbsp;&nbsp;&nbsp;".join(parts))


# ── Options column ────────────────────────────────────────────────────────────
class ApiOptions(QScrollArea):
    """Scrollable options for the API tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        # Slightly tighter side margins so the controls fit the original 270px
        # options width (shared with the Spectrum tab) without clipping.
        lay.setContentsMargins(6, 12, 6, 12); lay.setSpacing(6)

        title = QLabel("API"); title.setObjectName("opt_title")
        lay.addWidget(title); lay.addWidget(hsep())

        # ── Run selection ────────────────────────────────────────────
        lay.addWidget(header("RUN"))
        self.ed_date = QLineEdit(); self.ed_date.setPlaceholderText("YYYY-MM-DD")
        self.ed_run = QLineEdit(); self.ed_run.setPlaceholderText("e.g. 91")
        self.ed_ch = QLineEdit(); self.ed_ch.setPlaceholderText("e.g. 5")
        for lbl, ed in [("Date", self.ed_date), ("Run", self.ed_run), ("Channel", self.ed_ch)]:
            row, _ = labeled_row(lbl, ed); ed.setFixedWidth(104)
            lay.addWidget(row)
        self.ed_path = QLineEdit()
        self.ed_path.setPlaceholderText("optional data path (else data-path.txt)")
        lay.addWidget(self.ed_path)

        # ── Data type (mutually exclusive) ───────────────────────────
        lay.addWidget(hsep()); lay.addWidget(header("DATA TYPE"))
        self.type_group = QButtonGroup(self); self.type_group.setExclusive(True)
        self.rb_raw = QRadioButton("Raw")
        self.rb_cal = QRadioButton("Calibrated")
        self.rb_sims = QRadioButton("Simulated")
        self.rb_raw.setToolTip("Experimental raw data")
        self.rb_cal.setToolTip("Calibrated and time-shifted experimental data")
        self.rb_sims.setToolTip("Simulated data")
        self.rb_raw.setChecked(True)
        for rb in (self.rb_raw, self.rb_cal, self.rb_sims):
            self.type_group.addButton(rb); lay.addWidget(rb)
        note = QLabel("Raw / Calibrated are experimental data.")
        note.setObjectName("stat_key"); note.setWordWrap(True)
        lay.addWidget(note)

        self.btn_load = QPushButton("Load API file"); self.btn_load.setObjectName("open_btn")
        self.btn_load.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_load)

        # ── Run info readout ─────────────────────────────────────────
        lay.addWidget(hsep()); lay.addWidget(header("RUN INFO"))
        self.lbl_info = QLabel("—")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setTextFormat(Qt.RichText)
        self.lbl_info.setStyleSheet(
            f"color:{T.TEXT_PRIMARY}; font-family:{T.MONO_FAMILY}; font-size:13px;")
        lay.addWidget(self.lbl_info)

        # ── Display controls ─────────────────────────────────────────
        lay.addWidget(hsep()); lay.addWidget(header("DISPLAY"))
        self.cb_spe_log = QCheckBox("Energy Log Y")
        self.cb_spe_log.setToolTip("Logarithmic y-axis on the energy spectrum")
        lay.addWidget(self.cb_spe_log)
        self.cb_xy_log = QCheckBox("X-Y Log color")
        self.cb_xy_log.setToolTip("Logarithmic color scale on the X-Y map")
        lay.addWidget(self.cb_xy_log)
        self.ed_vmax = QLineEdit(); self.ed_vmax.setPlaceholderText("vmax")
        vrow, self.btn_vmax = labeled_row("vmax", self.ed_vmax, apply_btn=True)
        self.ed_vmax.setFixedWidth(70)
        self.ed_vmax.setToolTip("Cap the X-Y color scale at this count (forces linear scale)")
        lay.addWidget(vrow)
        self.btn_filters = QPushButton("Filters…"); self.btn_filters.setObjectName("action_btn")
        self.btn_filters.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_filters)

        # ── Energy calibration ───────────────────────────────────────
        lay.addWidget(hsep()); lay.addWidget(header("ENERGY CALIBRATION"))
        cal_note = QLabel(
            "Send the spectrum to the Spectrum tab, build a calibration on the "
            "Calibration tab, then retrieve it here to convert the dataframe to "
            "energy.")
        cal_note.setObjectName("stat_key"); cal_note.setWordWrap(True)
        lay.addWidget(cal_note)
        self.btn_retrieve_cal = QPushButton("Retrieve calibration")
        self.btn_retrieve_cal.setObjectName("action_btn")
        self.btn_retrieve_cal.setCursor(Qt.PointingHandCursor)
        self.btn_retrieve_cal.setToolTip(
            "Apply the Calibration tab's current calibration curve/equation to "
            "the API dataframe so the panels read in energy")
        lay.addWidget(self.btn_retrieve_cal)
        self.btn_clear_cal = QPushButton("Clear calibration")
        self.btn_clear_cal.setObjectName("mini_btn")
        self.btn_clear_cal.setCursor(Qt.PointingHandCursor)
        self.btn_clear_cal.setToolTip(
            "Remove the calibration and revert the panels to the original raw channels")
        self.btn_clear_cal.setEnabled(False)
        lay.addWidget(self.btn_clear_cal)
        self.lbl_cal = QLabel("Uncalibrated (channels)")
        self.lbl_cal.setObjectName("stat_key"); self.lbl_cal.setWordWrap(True)
        lay.addWidget(self.lbl_cal)

        # ── Drift correction (time + energy shifts) ──────────────────
        lay.addWidget(hsep()); lay.addWidget(header("DRIFT CORRECTION"))
        shift_note = QLabel(
            "Correct gain/time drift over the run: split into time segments and "
            "align them, or shift dt by a constant — independently for the "
            "energy and time axes.")
        shift_note.setObjectName("stat_key"); shift_note.setWordWrap(True)
        lay.addWidget(shift_note)
        self.btn_shifts = QPushButton("Shifts…")
        self.btn_shifts.setObjectName("action_btn")
        self.btn_shifts.setCursor(Qt.PointingHandCursor)
        self.btn_shifts.setToolTip(
            "Open the energy gain-shift and time-shift correction window")
        lay.addWidget(self.btn_shifts)

        # ── Energy selections ────────────────────────────────────────
        lay.addWidget(hsep()); lay.addWidget(header("ENERGY SELECTIONS"))
        snote = QLabel("Arm “Add selection”, then drag a band on the "
                       "energy spectrum to tag a line. “Plot” overlays "
                       "each selection in its colour.")
        snote.setObjectName("stat_key"); snote.setWordWrap(True)
        lay.addWidget(snote)
        self.btn_add_sel = QPushButton("Add selection")
        self.btn_add_sel.setObjectName("action_btn")
        self.btn_add_sel.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_add_sel)
        # Controller rebuilds this layout into one row per selection.
        self.sel_box = QVBoxLayout(); self.sel_box.setSpacing(3)
        self.sel_box.setContentsMargins(0, 0, 0, 0)
        sel_holder = QWidget(); sel_holder.setLayout(self.sel_box)
        lay.addWidget(sel_holder)
        srow = QHBoxLayout(); srow.setContentsMargins(0, 0, 0, 0); srow.setSpacing(6)
        self.btn_plot_sel = QPushButton("Plot selections")
        self.btn_plot_sel.setObjectName("open_btn")
        self.btn_plot_sel.setCursor(Qt.PointingHandCursor)
        self.btn_clear_sel = QPushButton("Clear")
        self.btn_clear_sel.setObjectName("danger_btn")
        self.btn_clear_sel.setCursor(Qt.PointingHandCursor)
        srow.addWidget(self.btn_plot_sel, 1); srow.addWidget(self.btn_clear_sel, 0)
        sroww = QWidget(); sroww.setLayout(srow)
        lay.addWidget(sroww)

        # ── Actions ──────────────────────────────────────────────────
        lay.addWidget(hsep()); lay.addWidget(header("ACTIONS"))
        self.btn_send = QPushButton(SEND_DEFAULT_TEXT)
        self.btn_send.setObjectName("open_btn"); self.btn_send.setCursor(Qt.PointingHandCursor)
        self.btn_send.setToolTip("Hand the current energy histogram to the Spectrum tab")
        lay.addWidget(self.btn_send)
        self.btn_3d = QPushButton("3D view…"); self.btn_3d.setObjectName("action_btn")
        self.btn_3d.setCursor(Qt.PointingHandCursor)
        self.btn_3d.setToolTip("Plotly volume render of the reconstructed X-Y-Z hit cloud")
        lay.addWidget(self.btn_3d)
        self.btn_reset = QPushButton("Reset"); self.btn_reset.setObjectName("danger_btn")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_reset)

        lay.addStretch(1)
        self.setWidget(inner)


# ── Controller ────────────────────────────────────────────────────────────────
class ApiController:
    """Wires an ApiOptions panel and ApiPage to the main app."""

    def __init__(self, app, options: ApiOptions, page: ApiPage):
        self.app = app
        self.opts = options
        self.page = page
        self.df_api = None
        self.df_current = None
        self.df_previous = None
        # Filter "used before" flags, mirroring the legacy previous/current scheme.
        self.en_flag = self.dt_flag = self.xy_flag = 0
        self.xkey = self.ykey = self.ekey = None
        self.xyplane = (-0.9, 0.9, -0.9, 0.9)
        self.erange = [0, 1]
        self.ebins = 2 ** 12
        self.tbins = 512
        self.hexbins = 80
        self.vmax = None
        self.flood_field = False
        # Static run-settings rows: list of (label, value, value_color).
        self._settings_rows = []
        self._settings_error = ""
        # Energy histogram (kept for "send to Spectrum").
        self.gam = self.gam_x = None
        self._filter_dlg = None
        self._api3d_dlg = None
        self._api3d_tmp = None      # last temp HTML file, removed on the next plot
        self._selectors = []
        # Energy selections: list of dict(label, color, emin, emax, df). Each
        # snapshots the energy-cut dataframe at creation time.
        self.selections = []
        self._arming_selection = False
        self._sel_color_idx = 0
        # Energy calibration: the channel column the histogram is binned from
        # (set in _configure_keys), the polynomial coeffs retrieved from the
        # Calibration tab, and the energy units once calibrated (None ⇒ raw
        # channels).
        self._chan_key = None
        self._chan_base = None      # pristine raw-channel column for the data type
        self._cal_coeffs = None
        self.e_units = None
        # Gain-shift drift correction (Shifts… window): when applied, the
        # corrected channels live on the df_api "energy_drift" column and
        # _chan_key points at it (so it layers under the polynomial calibration).
        self._egain_applied = False
        self._egain_label = ""
        # Time correction (Shifts… window): a constant added to ``dt`` or a
        # segment alignment, both writing the ``dt_cal`` column. _dt_shift holds
        # the constant (float) when that mode is used; _dt_segments_applied marks
        # the segment-alignment mode. Either makes _dt_key switch to "dt_cal".
        self._dt_shift = None
        self._dt_segments_applied = False
        self._dt_label = "No time shift"
        self._dt_key = "dt"
        self._shifts_dlg = None
        self._wire()
        self._refresh_sel_list()

    @property
    def _dt_corrected(self):
        """True when any time correction (constant or segment) is active."""
        return self._dt_shift is not None or self._dt_segments_applied

    def _wire(self):
        o = self.opts
        o.btn_load.clicked.connect(self._load)
        o.btn_reset.clicked.connect(self._reset)
        o.btn_send.clicked.connect(self._send_to_spectrum)
        o.btn_3d.clicked.connect(self._open_3d)
        o.btn_filters.clicked.connect(self._open_filters)
        o.btn_add_sel.clicked.connect(self._arm_selection)
        o.btn_plot_sel.clicked.connect(self._plot_selections)
        o.btn_clear_sel.clicked.connect(self._clear_selections)
        o.btn_retrieve_cal.clicked.connect(self._retrieve_calibration)
        o.btn_clear_cal.clicked.connect(self._clear_calibration)
        o.btn_shifts.clicked.connect(self._open_shifts)
        o.cb_spe_log.toggled.connect(self._toggle_spe_log)
        o.cb_xy_log.toggled.connect(lambda *_: self._replot_xy())
        o.btn_vmax.clicked.connect(self._apply_vmax)
        o.ed_vmax.returnPressed.connect(self._apply_vmax)

    def _status(self, msg):
        self.app.statusBar().showMessage(f"  {msg}")

    def _data_type(self):
        if self.opts.rb_sims.isChecked():
            return TYPE_SIMS
        if self.opts.rb_cal.isChecked():
            return TYPE_CAL
        return TYPE_RAW

    # -- loading ---------------------------------------------------------------
    def _load(self):
        # Brighten the Load button so the click registers before the (blocking)
        # file read freezes the UI — purple to match its open_btn identity.
        self._flash_button(self.opts.btn_load, bg=T.SNR_PURPLE, fg=T.BG_DARK)
        try:
            self._load_file()
        except Exception as exc:  # noqa: BLE001 — surface load errors to the user
            traceback.print_exc()
            self._status(f"Could not load API file: {exc}")

    def _load_file(self):
        date = self.opts.ed_date.text().strip()
        run_txt = self.opts.ed_run.text().strip()
        ch_txt = self.opts.ed_ch.text().strip()
        if not date or not run_txt:
            self._status("Enter at least a date and a run number")
            return
        try:
            runnr = int(run_txt)
        except ValueError:
            self._status("Run number must be an integer")
            return
        data_path = self.opts.ed_path.text().strip() or None
        dtype = self._data_type()

        # Energy binning: the high-resolution channels use more bins.
        self.ebins = 2 ** 14 if ch_txt in ("6", "7", "10", "11") else 2 ** 12

        self.flood_field = ch_txt == "9"
        if self.flood_field:
            ch = 9
        else:
            try:
                ch = int(ch_txt)
            except ValueError:
                self._status("Channel must be an integer")
                return

        self._status(f"Loading run {date}-{runnr} ch {ch}…")
        df = read_parquet_api.read_parquet_file(
            date=date, runnr=runnr, ch=ch,
            flood_field=self.flood_field, data_path_txt=data_path)
        if df is None:
            QMessageBox.critical(
                self.app, "Error while opening parquet data",
                f"No parquet file available for run {date}-{runnr}.")
            self._status("No parquet file for that run")
            return
        if not self.flood_field:
            df = df.copy()
            df["dt"] *= 1e9  # s → ns

        self.df_api = df
        self.df_current = df.copy()
        self.df_previous = df.copy()
        self.en_flag = self.dt_flag = self.xy_flag = 0
        # A new run invalidates selections snapshotted from the old data.
        self._reset_selections()
        # …and any calibration: the fresh dataframe has no energy_cal column.
        self._cal_coeffs = None
        self.e_units = None
        self.opts.lbl_cal.setText("Uncalibrated (channels)")
        self.opts.btn_clear_cal.setEnabled(False)
        # …and any drift corrections: the fresh dataframe has no derived columns.
        self._dt_shift = None
        self._dt_segments_applied = False
        self._dt_label = "No time shift"
        self._dt_key = "dt"
        self._egain_applied = False
        self._egain_label = ""
        self._refresh_shift_labels()

        self._configure_keys(dtype)
        self._initialize_plots()
        self._load_settings(date, runnr, ch, data_path)
        n = self.df_current.shape[0]
        self._status(f"Loaded run {date}-{runnr} ch {ch}  ·  {n:,} events")

    def _configure_keys(self, dtype):
        """Set the X/Y/energy column keys and plot ranges for the data type.

        Channel axis layering (low → high):
        ``_chan_base`` is the pristine raw channel column for the data type
        ("energy_orig"/"energy"). When a gain-shift drift correction is applied,
        ``_chan_key`` switches to the corrected ``energy_drift`` column so the
        histogram, send-to-spectrum and the polynomial calibration all read the
        drift-corrected channels. ``ekey`` is then ``energy_cal`` once a
        polynomial calibration is applied, else ``_chan_key``.
        """
        df = self.df_current
        if dtype == TYPE_SIMS:
            self.xkey, self.ykey, self._chan_base = "X", "Y", "energy"
            self.xyplane = (-0.2, 0.2, -0.2, 0.2)
            chan_range = [0, float(df["energy"].max())]
        elif dtype == TYPE_CAL:
            self.df_current = apicalc.calc_own_pos(self.df_current)
            self.xkey, self.ykey, self._chan_base = "X2", "Y2", "energy"
            self.xyplane = (-0.9, 0.9, -0.9, 0.9)
            chan_range = [0, float(self.df_current["energy"].max())]
        else:  # raw
            self.df_current = apicalc.calc_own_pos(self.df_current)
            self.xkey, self.ykey, self._chan_base = "X2", "Y2", "energy_orig"
            self.xyplane = (-0.9, 0.9, -0.9, 0.9)
            chan_range = [0, 2 ** 16]
        # Gain-shift drift correction layers under everything: the corrected
        # channels become the working channel axis when present.
        if self._egain_applied and "energy_drift" in self.df_current.columns:
            self._chan_key = "energy_drift"
        else:
            self._chan_key = self._chan_base
        # Fit the X-Y extent to where the hits actually land instead of the
        # detector's full physical plane. Most runs illuminate only a central
        # patch, so the fixed plane left the map floating in empty space (and the
        # cursor reporting coordinates out there). The hardcoded plane above stays
        # as the fallback for empty/degenerate data.
        self.xyplane = self._fit_xyplane(self.df_current, self.xkey, self.ykey,
                                         fallback=self.xyplane)
        if self.e_units is not None and "energy_cal" in self.df_current.columns:
            self.ekey = "energy_cal"
            self.erange = [0.0, float(self.df_current["energy_cal"].max())]
        else:
            self.ekey = self._chan_key
            self.erange = chan_range
        # Time axis: read the corrected column when a time correction (constant
        # shift or segment alignment) is active and present. Mirrors the energy
        # path so it survives Reset (which keeps dt_cal on the rebuilt frame).
        if self._dt_corrected and "dt_cal" in self.df_current.columns:
            self._dt_key = "dt_cal"
        else:
            self._dt_key = "dt"

    @staticmethod
    def _fit_xyplane(df, xkey, ykey, fallback, pad=0.06):
        """Square X-Y extent that snugly bounds the hit positions (plus a small
        margin) so the map fills its panel. Bounds come from the 0.2–99.8
        percentiles, so a handful of stray edge events from position
        reconstruction can't blow the view back out to the full plane. Returns
        *fallback* when the data is empty or has no spread."""
        if df is None or df.shape[0] == 0 or xkey not in df or ykey not in df:
            return fallback
        x = df[xkey].to_numpy(dtype=float); y = df[ykey].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size == 0:
            return fallback
        xlo, xhi = np.percentile(x, [0.2, 99.8])
        ylo, yhi = np.percentile(y, [0.2, 99.8])
        cx, cy = (xlo + xhi) / 2.0, (ylo + yhi) / 2.0
        half = max(xhi - xlo, yhi - ylo) / 2.0
        if half <= 0:
            return fallback
        half *= (1.0 + pad)
        return (cx - half, cx + half, cy - half, cy + half)

    def _initialize_plots(self):
        self.page.build_axes(flood_field=self.flood_field)
        self._detach_selectors()
        if self.flood_field:
            self._replot_xy()
        else:
            self._plot_energy(self.df_current)
            self._plot_time(self.df_current)
            self._replot_xy()
            self._attach_selectors()
        self.page.reset_nav()
        self.page.canvas.draw_idle()

    def _load_settings(self, date, runnr, ch, data_path):
        try:
            tot_time = apicalc.get_total_time(date, runnr, ch, data_path)
            tot_alphas = apicalc.get_total_counts(date, runnr, ch=9, data_path=data_path)
            nyield = apicalc.calculate_neutron_yield(date, runnr, ch=9, data_path=data_path)
            self._settings_rows = [
                ("Total alphas", f"{tot_alphas:.3E}", T.ACCENT_AMBER),
                ("Live time (HH:MM:SS)",
                 time.strftime("%H:%M:%S", time.gmtime(tot_time)), T.ACCENT_CYAN),
                ("Neutron yield (n/s)", f"{nyield:.3E}", T.ACCENT_GREEN),
            ]
            self._settings_error = ""
        except Exception:  # noqa: BLE001 — settings are best-effort
            traceback.print_exc()
            self._settings_rows = []
            self._settings_error = "Run-settings file not found"
        self._update_info()

    def _update_info(self):
        """Compose the RUN INFO readout: the static run settings plus the live
        processed-event count (this replaces the old X-Y plot title). Labels are
        bold; each value gets its own accent colour to stand out."""
        rows = list(self._settings_rows)
        if self.df_current is not None:
            rows.append(("Processed counts",
                         f"{self.df_current.shape[0]:,}", T.SNR_PURPLE))
        if not rows and not self._settings_error:
            self.opts.lbl_info.setText("—")
            return
        lines = [f"<b>{lbl}:</b> "
                 f"<span style='color:{color}; font-weight:700'>{val}</span>"
                 for lbl, val, color in rows]
        if self._settings_error:
            lines.insert(0, f"<i style='color:{T.TEXT_DIM}'>{self._settings_error}</i>")
        self.opts.lbl_info.setText("<br>".join(lines))

    # -- panel drawing ---------------------------------------------------------
    def _compute_energy_hist(self, df):
        """Bin *df* into the energy histogram (self.gam / self.gam_x).

        Kept separate from drawing so "Send to spectrum" never depends on the
        energy panel having been drawn first — otherwise the first click can
        find self.gam still None (no draw yet) and silently do nothing.
        """
        self.gam, edg = np.histogram(df[self.ekey], bins=self.ebins, range=self.erange)
        self.gam_x = (edg[1:] + edg[:-1]) / 2

    def _plot_energy(self, df):
        # Always (re)compute the histogram, even if the axis isn't ready yet, so
        # Send has data to work with regardless of draw timing.
        self._compute_energy_hist(df)
        # The energy histogram just changed — clear any "✓ Sent" confirmation so
        # the button reflects that this spectrum hasn't been sent yet.
        self._reset_send_button()
        ax = self.page.ax_spe
        if ax is None:
            return
        ax.plot(self.gam_x, self.gam, color=T.LOGO_GREEN, linewidth=0.9)
        ax.set_yscale("log" if self.opts.cb_spe_log.isChecked() else "linear")
        ax.set_xlabel(self._energy_xlabel())
        ax.set_ylabel("Counts")
        self.page._style(ax)

    def _energy_xlabel(self):
        """Energy-panel x-label: the calibrated units once a calibration is
        applied, MeV for simulated/calibrated data, else raw channels."""
        if self.e_units:
            return f"Energy ({self.e_units})"
        return "Energy (MeV)" if self.ekey == "energy" else "Channels"

    def _plot_time(self, df):
        ax = self.page.ax_dt
        if ax is None or df.shape[0] == 0:
            return
        low, high = np.percentile(df[self._dt_key], [0.2, 99.5])
        corrected = self._dt_corrected and self._dt_key != "dt" and "dt" in df.columns
        if corrected:
            # Show the raw dt faintly behind the corrected spectrum, over a
            # combined range, so a constant offset (which otherwise just re-ranges
            # the panel) is clearly visible as the peak moving off the raw one.
            rlow, rhigh = np.percentile(df["dt"], [0.2, 99.5])
            low, high = min(low, rlow), max(high, rhigh)
            ax.hist(df["dt"], bins=self.tbins, range=(low, high),
                    color=T.TEXT_DIM, alpha=0.35, label="raw")
        # Dark bin edges (as in the legacy plot) separate the bars on the cyan fill.
        ax.hist(df[self._dt_key], bins=self.tbins, range=(low, high),
                color=T.ACCENT_CYAN, alpha=0.8, edgecolor=API_PLOT_BG, linewidth=0.4,
                label="corrected" if corrected else None)
        ax.set_xlabel("dt (ns, shifted)" if self._dt_corrected else "dt (ns)")
        ax.set_ylabel("Counts")
        if corrected:
            ax.legend(loc="upper right", fontsize=8, facecolor=API_PLOT_BG,
                      edgecolor=T.BORDER, labelcolor=T.TEXT_PRIMARY)
        self.page._style(ax)

    def _replot_xy(self):
        """Redraw the X-Y hexbin from df_current with the current scale/vmax."""
        ax = self.page.ax_xy
        if ax is None or self.df_current is None:
            return
        # The processed-count readout (formerly the plot title) lives in RUN INFO.
        self._update_info()
        ax.clear()
        log = self.opts.cb_xy_log.isChecked()
        kwargs = {}
        if log:
            kwargs["bins"] = "log"
        if self.vmax is not None:
            kwargs["vmax"] = self.vmax
        df = self.df_current
        if df.shape[0] == 0:
            self.page.set_xy_lookup(None, log, 0.0)
            self.page._style(ax, grid=False)
            self.page.canvas.draw_idle()
            return
        df.plot.hexbin(
            x=self.xkey, y=self.ykey, gridsize=self.hexbins, cmap=COLORMAP,
            ax=ax, colorbar=False, extent=self.xyplane, **kwargs)
        # pandas leaves the PolyCollection as the last collection on the axes.
        # Cache it (centres + counts) so the cursor readout can report a
        # hexagon's intensity in place of the removed colorbar; the pick radius
        # is one hex spacing across the X-Y extent.
        mappable = ax.collections[-1] if ax.collections else None
        spacing = (self.xyplane[1] - self.xyplane[0]) / self.hexbins
        self.page.set_xy_lookup(mappable, log, spacing ** 2)
        # Pin the axes to the hexbin extent. Otherwise hexbin autoscales with a
        # 5% margin, so the axis range runs wider than the drawn hexagons and a
        # black band (no hexagon → cursor reads no intensity) rings the map.
        ax.set_xlim(self.xyplane[0], self.xyplane[1])
        ax.set_ylim(self.xyplane[2], self.xyplane[3])
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        self.page._style(ax, grid=False)
        self.page.canvas.draw_idle()

    # -- selectors -------------------------------------------------------------
    def _attach_selectors(self):
        self._detach_selectors()
        if self.page.ax_spe is not None:
            self._selectors.append(SpanSelector(
                self.page.ax_spe, self._on_energy_span, "horizontal",
                useblit=True, interactive=True,
                props=dict(alpha=0.3, facecolor=T.ACCENT_AMBER)))
        if self.page.ax_dt is not None:
            self._selectors.append(SpanSelector(
                self.page.ax_dt, self._on_time_span, "horizontal",
                useblit=True, interactive=True,
                props=dict(alpha=0.3, facecolor=T.ACCENT_AMBER)))
        if self.page.ax_xy is not None:
            self._selectors.append(RectangleSelector(
                self.page.ax_xy, self._on_xy_select, useblit=True,
                button=[1, 3], minspanx=0, minspany=0, spancoords="pixels",
                interactive=True,
                props=dict(facecolor=T.TEXT_PRIMARY, edgecolor=T.TEXT_PRIMARY,
                           alpha=0.15, fill=True)))

    def _detach_selectors(self):
        for s in self._selectors:
            try:
                s.disconnect_events()
                s.set_visible(False)
            except Exception:  # noqa: BLE001
                pass
        self._selectors = []

    def _on_energy_span(self, xmin, xmax):
        if xmax <= xmin:
            return
        # When "Add selection" is armed, an energy drag tags a labeled band
        # instead of filtering the data.
        if self._arming_selection:
            self._create_selection(round(xmin, 4), round(xmax, 4))
        else:
            self.apply_energy_filter(round(xmin, 4), round(xmax, 4))

    def _on_time_span(self, tmin, tmax):
        if tmax > tmin:
            self.apply_t_filter(tmin, tmax)

    def _on_xy_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        self.apply_xy_filter(x1, x2, y1, y2)

    # -- filters ---------------------------------------------------------------
    def apply_energy_filter(self, xmin, xmax):
        if self.df_current is None:
            return
        if self.en_flag == 0:
            mask = (self.df_current[self.ekey] > xmin) & (self.df_current[self.ekey] < xmax)
            self.df_previous = self.df_current.copy()
            self.df_current = self.df_current[mask]
            self.en_flag = 1
        else:
            mask = (self.df_previous[self.ekey] > xmin) & (self.df_previous[self.ekey] < xmax)
            self.df_current = self.df_previous[mask]
        self.df_current = self.df_current.reset_index(drop=True)
        if self.page.ax_dt is not None:
            self.page.ax_dt.clear()
            self._plot_time(self.df_current)
        self._replot_xy()
        self.page.reset_nav()
        self.page.canvas.draw_idle()
        self._status(f"Energy filter [{xmin:g}, {xmax:g}]  ·  {self.df_current.shape[0]:,} events")

    def apply_t_filter(self, tmin, tmax):
        if self.df_current is None:
            return
        if self.dt_flag == 0:
            mask = (self.df_current[self._dt_key] > tmin) & (self.df_current[self._dt_key] < tmax)
            self.df_previous = self.df_current.copy()
            self.df_current = self.df_current[mask]
            self.dt_flag = 1
        else:
            mask = (self.df_previous[self._dt_key] > tmin) & (self.df_previous[self._dt_key] < tmax)
            self.df_current = self.df_previous[mask]
        self.df_current = self.df_current.reset_index(drop=True)
        if self.page.ax_spe is not None:
            self.page.ax_spe.clear()
            self._plot_energy(self.df_current)
        self._replot_xy()
        self.page.reset_nav()
        self.page.canvas.draw_idle()
        self._status(f"Time filter [{tmin:g}, {tmax:g}] ns  ·  {self.df_current.shape[0]:,} events")

    def apply_xy_filter(self, x1, x2, y1, y2):
        if self.df_current is None:
            return
        xlo, xhi = sorted((x1, x2))
        ylo, yhi = sorted((y1, y2))
        if self.xy_flag == 0:
            base = self.df_current
            self.df_previous = self.df_current.copy()
            self.xy_flag = 1
        else:
            base = self.df_previous
        mask = ((base[self.xkey] > xlo) & (base[self.xkey] < xhi)
                & (base[self.ykey] > ylo) & (base[self.ykey] < yhi))
        self.df_current = base[mask].reset_index(drop=True)
        if self.page.ax_spe is not None:
            self.page.ax_spe.clear()
            self._plot_energy(self.df_current)
        if self.page.ax_dt is not None:
            self.page.ax_dt.clear()
            self._plot_time(self.df_current)
        # Bug fix vs legacy: redraw the X-Y map from the *filtered* data so the
        # panel reflects the selection (legacy replotted df_previous, the full map).
        self._replot_xy()
        self.page.reset_nav()
        self.page.canvas.draw_idle()
        self._status(f"X-Y filter  ·  {self.df_current.shape[0]:,} events")

    def _open_filters(self):
        if self.df_current is None:
            self._status("Load an API file first")
            return
        if self._filter_dlg is None:
            self._filter_dlg = ApiFilterDialog(self.app)
            self._filter_dlg.btn_apply.clicked.connect(self._apply_manual_filters)
        self._filter_dlg.show()
        self._filter_dlg.raise_()

    def _apply_manual_filters(self):
        dlg = self._filter_dlg
        xy_x = dlg.pair("x")
        xy_y = dlg.pair("y")
        if xy_x is not None and xy_y is not None:
            self.apply_xy_filter(xy_x[0], xy_x[1], xy_y[0], xy_y[1])
        t = dlg.pair("t")
        if t is not None:
            self.apply_t_filter(t[0], t[1])
        e = dlg.pair("e")
        if e is not None:
            self.apply_energy_filter(e[0], e[1])

    # -- energy selections -----------------------------------------------------
    def _arm_selection(self):
        if self.df_current is None or self.page.ax_spe is None:
            self._status("Load an API file with an energy panel first")
            return
        self._arming_selection = True
        self._set_add_armed(True)
        self._status("Drag a band on the energy spectrum to add a selection…")

    def _set_add_armed(self, armed):
        """Highlight the Add-selection button while a selection drag is armed."""
        b = self.opts.btn_add_sel
        if armed:
            b.setText("Drag energy band…")
            b.setStyleSheet(
                f"background-color:{T.ACCENT_AMBER}; color:{T.BG_DARK}; "
                f"border:2px solid {T.ACCENT_AMBER}; border-radius:5px; "
                f"padding:8px 13px; font-weight:800;")
        else:
            b.setText("Add selection")
            b.setStyleSheet("")

    def _create_selection(self, emin, emax):
        self._arming_selection = False
        self._set_add_armed(False)
        if self.df_current is None:
            return
        mask = (self.df_current[self.ekey] > emin) & (self.df_current[self.ekey] < emax)
        sub = self.df_current[mask].copy()
        if sub.shape[0] == 0:
            self._status(f"No events in [{emin:g}, {emax:g}] — selection not added")
            return
        color = T.OVERLAY_COLORS[self._sel_color_idx % len(T.OVERLAY_COLORS)]
        dlg = EnergySelectionDialog(emin, emax, color, sub.shape[0], self.app)
        if dlg.exec_() != QDialog.Accepted:
            self._status("Selection cancelled")
            return
        label = dlg.label() or f"sel {len(self.selections) + 1}"
        self.selections.append(dict(
            label=label, color=dlg.color(), emin=emin, emax=emax, df=sub))
        self._sel_color_idx += 1
        self._refresh_sel_list()
        self._status(f"Added '{label}'  ·  {sub.shape[0]:,} events "
                     f"[{emin:g}, {emax:g}]")

    def _refresh_sel_list(self):
        """Rebuild the options-panel list: one colored row per selection."""
        box = self.opts.sel_box
        while box.count():
            item = box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if not self.selections:
            empty = QLabel("No selections yet."); empty.setObjectName("stat_key")
            box.addWidget(empty)
            return
        for sel in self.selections:
            row = QWidget()
            rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(6)
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{sel['color']}; font-size:15px;")
            name = QLabel(f"{sel['label']}  [{sel['emin']:g}–{sel['emax']:g}]")
            name.setStyleSheet(f"color:{T.TEXT_PRIMARY}; font-size:12px;")
            btn = QPushButton("✕"); btn.setObjectName("mini_btn")
            btn.setFixedWidth(26); btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip("Remove this selection")
            btn.clicked.connect(lambda _=False, s=sel: self._remove_selection(s))
            rl.addWidget(dot); rl.addWidget(name, 1); rl.addWidget(btn, 0)
            box.addWidget(row)

    def _remove_selection(self, sel):
        try:
            self.selections.remove(sel)
        except ValueError:
            return
        self._refresh_sel_list()
        self._status(f"Removed selection '{sel['label']}'")

    def _clear_selections(self):
        if not self.selections:
            return
        self.selections = []
        self._refresh_sel_list()
        self._status("Cleared all selections")

    def _reset_selections(self):
        """Drop all selections (e.g. when a new run is loaded) without a status
        message — the caller reports the higher-level action."""
        self.selections = []
        self._sel_color_idx = 0
        self._arming_selection = False
        self._set_add_armed(False)
        self._refresh_sel_list()

    def _plot_selections(self):
        if self.df_current is None:
            self._status("Load an API file first")
            return
        if not self.selections:
            self._status("No selections to plot — arm “Add selection” and drag a band")
            return
        self._flash_button(self.opts.btn_plot_sel, bg=T.SNR_PURPLE, fg=T.BG_DARK)
        self._plot_energy_overlays()
        self._plot_time_overlays()
        self._plot_xy_overlays()
        self.page.reset_nav()
        self.page.canvas.draw_idle()
        self._status(f"Plotted {len(self.selections)} selection(s)")

    def _plot_energy_overlays(self):
        self._compute_energy_hist(self.df_current)
        self._reset_send_button()
        ax = self.page.ax_spe
        if ax is None:
            return
        ax.clear()
        # Full spectrum dim/gray, each selection's band filled in its colour.
        ax.plot(self.gam_x, self.gam, color=T.TEXT_DIM, linewidth=0.9, alpha=0.7)
        for sel in self.selections:
            band = (self.gam_x >= sel["emin"]) & (self.gam_x <= sel["emax"])
            ax.fill_between(self.gam_x, self.gam, where=band,
                            color=sel["color"], alpha=0.55, label=sel["label"])
        ax.set_yscale("log" if self.opts.cb_spe_log.isChecked() else "linear")
        ax.set_xlabel(self._energy_xlabel())
        ax.set_ylabel("Counts")
        ax.legend(loc="upper right", fontsize=8, facecolor=API_PLOT_BG,
                  edgecolor=T.BORDER, labelcolor=T.TEXT_PRIMARY)
        self.page._style(ax)

    def _plot_time_overlays(self):
        ax = self.page.ax_dt
        if ax is None:
            return
        ax.clear()
        base = self.df_current
        if base.shape[0] == 0:
            self.page._style(ax)
            return
        low, high = np.percentile(base[self._dt_key], [0.2, 99.5])
        # Gray base histogram, each selection overlaid as a coloured outline.
        ax.hist(base[self._dt_key], bins=self.tbins, range=(low, high),
                color=T.TEXT_DIM, alpha=0.30, edgecolor=API_PLOT_BG, linewidth=0.3)
        for sel in self.selections:
            d = sel["df"][self._dt_key]
            if len(d) == 0:
                continue
            ax.hist(d, bins=self.tbins, range=(low, high), histtype="step",
                    color=sel["color"], linewidth=1.3)
        ax.set_xlabel("dt (ns)"); ax.set_ylabel("Counts")
        self.page._style(ax)

    def _plot_xy_overlays(self):
        ax = self.page.ax_xy
        if ax is None:
            return
        self._update_info()
        ax.clear()
        base = self.df_current
        if base.shape[0]:
            base.plot.hexbin(
                x=self.xkey, y=self.ykey, gridsize=self.hexbins, cmap=GRAY_CMAP,
                ax=ax, colorbar=False, extent=self.xyplane)
            mappable = ax.collections[-1] if ax.collections else None
            spacing = (self.xyplane[1] - self.xyplane[0]) / self.hexbins
            self.page.set_xy_lookup(mappable, False, spacing ** 2)
        # One translucent single-hue hexbin layer per selection: each cell fades
        # from transparent (low count) to the selection's solid colour (high
        # count), and empty cells are dropped (mincnt=1) so layers don't occlude.
        for sel in self.selections:
            d = sel["df"]
            if d.shape[0] == 0:
                continue
            r, g, b, _ = to_rgba(sel["color"])
            cmap = LinearSegmentedColormap.from_list(
                f"sel_{sel['label']}", [(r, g, b, 0.0), (r, g, b, 1.0)])
            d.plot.hexbin(
                x=self.xkey, y=self.ykey, gridsize=self.hexbins, cmap=cmap,
                ax=ax, colorbar=False, extent=self.xyplane, mincnt=1)
        handles = [Patch(facecolor=s["color"], edgecolor="none", label=s["label"])
                   for s in self.selections]
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=8,
                      facecolor=API_PLOT_BG, edgecolor=T.BORDER,
                      labelcolor=T.TEXT_PRIMARY)
        ax.set_xlim(self.xyplane[0], self.xyplane[1])
        ax.set_ylim(self.xyplane[2], self.xyplane[3])
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        self.page._style(ax, grid=False)

    # -- energy calibration ----------------------------------------------------
    def _retrieve_calibration(self):
        """Pull the Calibration tab's current calibration and apply it here.

        The calibration there is E = f(channel); because the API spectrum is sent
        to the Spectrum tab carrying its *real* channel values (see
        _send_to_spectrum), that polynomial maps the same channel axis as the
        dataframe's raw energy column — so it applies directly."""
        if self.df_current is None:
            self._status("Load an API file first")
            return
        cal = getattr(self.app, "calibration", None)
        res = cal.current_calibration() if cal is not None else None
        if res is None:
            self._status("No calibration in the Calibration tab — build one there "
                         "first (send this spectrum over, then calibrate)")
            return
        coeffs, units = res
        self.apply_calibration(coeffs, units)

    def apply_calibration(self, coeffs, units="keV"):
        """Calibrate the list-mode dataframe with a channel→energy polynomial:
        add an ``energy_cal`` column E = Σ cᵢ·chⁱ (ch = the raw channel column),
        then rebuild the view in energy."""
        if self.df_api is None or self._chan_key is None:
            return
        coeffs = [float(c) for c in coeffs]
        base_col = self._chan_key

        def add_cal(df):
            ch = df[base_col].astype(float).to_numpy()
            e = np.zeros_like(ch)
            for i, c in enumerate(coeffs):
                e = e + c * ch ** i
            df = df.copy()
            df["energy_cal"] = e
            return df

        # Calibrate the master, then re-derive the working frames (reset-style).
        self.df_api = add_cal(self.df_api)
        self._cal_coeffs = coeffs
        self.e_units = units
        self.df_current = self.df_api.copy()
        self.df_previous = self.df_api.copy()
        self.en_flag = self.dt_flag = self.xy_flag = 0
        self.vmax = None
        self.opts.ed_vmax.clear()
        # Existing selections were cut in the old (channel) axis — drop them.
        self._reset_selections()
        self._configure_keys(self._data_type())
        self._initialize_plots()
        self.opts.btn_clear_cal.setEnabled(True)
        deg = len(coeffs) - 1
        self.opts.lbl_cal.setText(f"Calibrated → energy ({units}), degree {deg}")
        self._status(f"Retrieved calibration from the Calibration tab "
                     f"(degree {deg}, {units})")

    def _clear_calibration(self):
        """Drop the calibration and revert the panels to the original raw
        channels (rebuilds the view from the master frame, like reset)."""
        if self.e_units is None:
            self._status("Already uncalibrated")
            return
        if self.df_api is not None and "energy_cal" in self.df_api.columns:
            self.df_api = self.df_api.drop(columns=["energy_cal"])
        self._cal_coeffs = None
        self.e_units = None
        self.df_current = self.df_api.copy()
        self.df_previous = self.df_api.copy()
        self.en_flag = self.dt_flag = self.xy_flag = 0
        self.vmax = None
        self.opts.ed_vmax.clear()
        self._reset_selections()
        self._configure_keys(self._data_type())
        self._initialize_plots()
        self.opts.btn_clear_cal.setEnabled(False)
        self.opts.lbl_cal.setText("Uncalibrated (channels)")
        self._status("Calibration cleared — back to raw channels")

    # -- drift correction (Shifts… window) -------------------------------------
    def _rebuild_from_master(self):
        """Re-derive the working frames from df_api and redraw (reset-style)."""
        self.df_current = self.df_api.copy()
        self.df_previous = self.df_api.copy()
        self.en_flag = self.dt_flag = self.xy_flag = 0
        self.vmax = None
        self.opts.ed_vmax.clear()
        self._reset_selections()
        self._configure_keys(self._data_type())
        self._initialize_plots()

    def _chan_erange(self):
        """Histogram range for the raw channel axis of the current data type."""
        base = self._chan_base or "energy_orig"
        if base == "energy_orig":
            return (0.0, float(2 ** 16))
        return (0.0, float(self.df_api[base].max()))

    def _dt_erange(self):
        dt = self.df_api["dt"].astype(float)
        return (float(dt.min()), float(dt.max()))

    # energy gain shift ........................................................
    def apply_energy_gainshift(self, n_segments, method="shift", xrange=None):
        """Drift-correct the energy channels: split into time segments and align
        each onto the first, writing corrected channels to ``energy_drift``.
        ``_chan_key`` then reads that column, so the histogram, send-to-spectrum
        and any polynomial calibration all layer on the corrected channels. The
        column lives on df_api, so it survives Reset."""
        if self.df_api is None:
            self._status("Load an API file first")
            return
        base = self._chan_base or "energy_orig"
        gs = apicalc.GainShift(self.df_api, n_segments=int(n_segments),
                               bins=self.ebins, erange=self._chan_erange(),
                               col=base, out_col="energy_drift")
        gs.align(method=method, xrange=xrange)
        self.df_api = gs.df
        self._egain_applied = True
        self._egain_label = f"Gain-shift: {int(n_segments)} seg · {method}"
        self._rebuild_from_master()
        # Keep any polynomial calibration consistent with the corrected channels.
        if self._cal_coeffs is not None:
            self.apply_calibration(self._cal_coeffs, self.e_units)
        self._refresh_shift_labels()
        self._status(f"Applied energy gain-shift ({int(n_segments)} seg, {method})")

    def clear_energy_gainshift(self):
        if not self._egain_applied:
            self._status("No energy gain-shift applied")
            return
        if self.df_api is not None and "energy_drift" in self.df_api.columns:
            self.df_api = self.df_api.drop(columns=["energy_drift"])
        self._egain_applied = False
        self._egain_label = ""
        self._rebuild_from_master()
        if self._cal_coeffs is not None:
            self.apply_calibration(self._cal_coeffs, self.e_units)
        self._refresh_shift_labels()
        self._status("Energy gain-shift cleared")

    # time correction ..........................................................
    def apply_dt_shift(self, shift):
        """Constant time shift: dt_cal = dt + shift (from the original dt, so
        re-applying is not cumulative). Replaces any segment alignment. The
        column lives on df_api so it survives Reset; independent of the energy
        correction."""
        if self.df_api is None:
            return
        shift = float(shift)
        df = self.df_api.copy()
        df["dt_cal"] = df["dt"].astype(float) + shift
        self.df_api = df
        self._dt_shift = shift
        self._dt_segments_applied = False
        self._dt_label = f"Time shifted by {shift:+g} ns"
        self._rebuild_from_master()
        self._refresh_shift_labels()
        self._status(f"Applied time shift {shift:+g} ns")

    def apply_dt_gainshift(self, n_segments, method="shift", xrange=None):
        """Segment-based time alignment: split dt into time segments and align
        each onto the first, writing ``dt_cal``. Replaces any constant shift."""
        if self.df_api is None:
            self._status("Load an API file first")
            return
        gs = apicalc.GainShift(self.df_api, n_segments=int(n_segments),
                               bins=self.tbins, erange=self._dt_erange(),
                               col="dt", out_col="dt_cal")
        gs.align(method=method, xrange=xrange)
        self.df_api = gs.df
        self._dt_shift = None
        self._dt_segments_applied = True
        self._dt_label = f"Time aligned: {int(n_segments)} seg · {method}"
        self._rebuild_from_master()
        self._refresh_shift_labels()
        self._status(f"Applied time alignment ({int(n_segments)} seg, {method})")

    def _clear_dt_shift(self):
        """Drop the time correction (constant or segment) and revert to raw dt."""
        if not self._dt_corrected:
            self._status("No time correction applied")
            return
        if self.df_api is not None and "dt_cal" in self.df_api.columns:
            self.df_api = self.df_api.drop(columns=["dt_cal"])
        self._dt_shift = None
        self._dt_segments_applied = False
        self._dt_label = "No time shift"
        self._rebuild_from_master()
        self._refresh_shift_labels()
        self._status("Time correction cleared — back to raw dt")

    def _refresh_shift_labels(self):
        """Push the current correction state into the Shifts dialog if it's open."""
        dlg = self._shifts_dlg
        if dlg is None:
            return
        dlg.set_energy_state(self._egain_applied, self._egain_label)
        dlg.set_time_state(self._dt_corrected, self._dt_label)

    def _open_shifts(self):
        if self.df_api is None:
            self._status("Load an API file first")
            return
        if self._shifts_dlg is None:
            self._shifts_dlg = ShiftsDialog(self)
        self._refresh_shift_labels()
        self._shifts_dlg.refresh_previews()
        self._shifts_dlg.show()
        self._shifts_dlg.raise_()

    # -- 3D volume -------------------------------------------------------------
    def _open_3d(self):
        if self.df_current is None:
            self._status("Load an API file first")
            return
        if self._api3d_dlg is None:
            try:
                self._api3d_dlg = Api3DDialog(self.app)
            except Exception as exc:  # noqa: BLE001 — QtWebEngine may be missing
                traceback.print_exc()
                self._status(f"Could not open the 3D view: {exc}")
                return
            self._api3d_dlg.btn_plot.clicked.connect(self._create_plot_3d)
        self._api3d_dlg.show()
        self._api3d_dlg.raise_()
        self._api3d_dlg.activateWindow()

    def _xyz(self, df=None):
        """Reconstructed (X, Y, Z) for *df* (default: the current view). Sims
        carry X/Y/Z directly; experimental data is reconstructed via
        apicalc.api_xyz (the gamma-detector position is ignored, as in the
        legacy 3D plot)."""
        df = self.df_current if df is None else df
        if self._data_type() == TYPE_SIMS:
            return df["X"], df["Y"], df["Z"]
        return apicalc.api_xyz(df=df, use_det=False)

    def _volume_trace(self, go, df, num_bins, iso_min, iso_max, opacity,
                      surfcount, color=None, name=None):
        """Build a Plotly Volume from *df*'s reconstructed X-Y-Z cloud, or None
        if the histogram is empty/flat. A *color* gives a single-hue colorscale
        (one selection); otherwise the default Plasma density scale is used."""
        from scipy import ndimage

        X, Y, Z = self._xyz(df)
        hist, edges = np.histogramdd((X, Y, Z), bins=num_bins)
        centers = [0.5 * (e[1:] + e[:-1]) for e in edges]
        xc, yc, zc = np.meshgrid(*centers, indexing="ij")
        values = hist.flatten()
        span = values.max() - values.min()
        if span == 0:
            return None
        # Normalise → Gaussian-smooth → renormalise (matches the legacy plot).
        norm = (values - values.min()) / span
        vol = ndimage.gaussian_filter(norm, 4)
        vmax = vol.max()
        if vmax > 0:
            vol = vol / vmax
        colorscale = [[0, color], [1, color]] if color else "Plasma"
        return go.Volume(
            x=xc.flatten(), y=yc.flatten(), z=zc.flatten(), value=vol,
            isomin=iso_min, isomax=iso_max, opacity=opacity,
            surface_count=surfcount, colorscale=colorscale,
            showscale=(color is None), name=name or "")

    def _create_plot_3d(self):
        # With selections we render one coloured volume each (from their stored
        # cut dataframes); without, the whole current view as a single volume.
        if not self.selections and (self.df_current is None
                                    or self.df_current.shape[0] == 0):
            self._set_3d_status("No events to plot — load or relax the filters.")
            return
        dlg = self._api3d_dlg
        num_bins = dlg.value("no_bins", int)
        iso_min = dlg.value("isomin", float)
        iso_max = dlg.value("isomax", float)
        opacity = dlg.value("opacity", float)
        surfcount = dlg.value("surfcount", int)
        if None in (num_bins, iso_min, iso_max, opacity, surfcount):
            self._set_3d_status("Check the plot inputs — they must be numbers.")
            return
        if num_bins < 2 or surfcount < 1:
            self._set_3d_status("Need ≥2 bins and ≥1 surface.")
            return

        self._set_3d_status("Building volume…")
        try:
            import tempfile
            import plotly.graph_objs as go

            traces = []
            if self.selections:
                for sel in self.selections:
                    t = self._volume_trace(
                        go, sel["df"], num_bins, iso_min, iso_max, opacity,
                        surfcount, color=sel["color"], name=sel["label"])
                    if t is not None:
                        traces.append(t)
                if not traces:
                    self._set_3d_status("Selections have no events to render.")
                    return
                rendered = f"{len(traces)} selection(s)"
            else:
                t = self._volume_trace(go, self.df_current, num_bins, iso_min,
                                       iso_max, opacity, surfcount)
                if t is None:
                    self._set_3d_status("Histogram is empty/flat — nothing to render.")
                    return
                traces.append(t)
                rendered = f"{self.df_current.shape[0]:,} events"

            fig = go.Figure(data=traces)
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=API_PLOT_BG,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=bool(self.selections),
                scene=dict(
                    dragmode="orbit",
                    xaxis=dict(title="x", showticklabels=False),
                    yaxis=dict(title="y", showticklabels=False),
                    zaxis=dict(title="z", showticklabels=False),
                ),
            )

            # Inline plotly.js so the render works offline; load via a temp file
            # (setHtml has a payload-size limit that a full volume can exceed).
            # Force a dark page body so there's no white margin around the figure.
            html = fig.to_html(include_plotlyjs="inline", full_html=True)
            html = html.replace(
                "<body>", f"<body style='margin:0;background:{API_PLOT_BG}'>", 1)
            with tempfile.NamedTemporaryFile(
                    "w", delete=False, suffix=".html", encoding="utf-8") as f:
                f.write(html)
                path = f.name
            dlg.browser.setUrl(QUrl.fromLocalFile(path))
            self._cleanup_3d_tmp()
            self._api3d_tmp = path
            self._set_3d_status(f"Rendered {rendered}.")
        except Exception as exc:  # noqa: BLE001 — surface render errors to the dialog
            traceback.print_exc()
            self._set_3d_status(f"Could not build the 3D plot: {exc}")

    def _set_3d_status(self, msg):
        if self._api3d_dlg is not None:
            self._api3d_dlg.status.setText(msg)
        self._status(msg)

    def _cleanup_3d_tmp(self):
        if self._api3d_tmp:
            import os
            try:
                os.unlink(self._api3d_tmp)
            except OSError:
                pass
            self._api3d_tmp = None

    # -- display toggles -------------------------------------------------------
    def _toggle_spe_log(self):
        ax = self.page.ax_spe
        if ax is None:
            return
        ax.set_yscale("log" if self.opts.cb_spe_log.isChecked() else "linear")
        self.page.canvas.draw_idle()

    def _apply_vmax(self):
        txt = self.opts.ed_vmax.text().strip()
        if txt == "":
            self.vmax = None
        else:
            try:
                self.vmax = float(txt)
            except ValueError:
                self._status("vmax must be a number")
                return
        # vmax forces a linear color scale (matches the legacy behaviour).
        self.opts.cb_xy_log.blockSignals(True)
        self.opts.cb_xy_log.setChecked(False)
        self.opts.cb_xy_log.blockSignals(False)
        self._replot_xy()

    # -- reset / send ----------------------------------------------------------
    def _reset(self):
        self._flash_button(self.opts.btn_reset)
        if self.df_api is None:
            self.page.show_empty()
            return
        # Retrieve the original (loaded) dataframe — no re-read of the file, same
        # as the legacy GUI's reset_button_api. df_api is the untouched load; the
        # position columns are re-derived from it below.
        self.df_current = self.df_api.copy()
        self.df_previous = self.df_api.copy()
        self.en_flag = self.dt_flag = self.xy_flag = 0
        self.vmax = None
        self.opts.ed_vmax.clear()
        self._reset_selections()
        self._configure_keys(self._data_type())
        self._initialize_plots()
        self._status("API view reset")

    def _flash_button(self, button, bg=T.ACCENT_RED, fg="#ffffff"):
        """Briefly brighten a button so the user sees the click registered, then
        revert to its themed style. ``repaint()`` paints the bright state *now*,
        before the (blocking) load/reset work freezes the event loop — otherwise
        the flash would never reach the screen."""
        button.setStyleSheet(
            f"background-color:{bg}; color:{fg}; "
            f"border:2px solid {bg}; border-radius:5px; "
            f"padding:8px 13px; font-size:14px; font-weight:800;")
        button.repaint()
        QTimer.singleShot(220, lambda: button.setStyleSheet(""))

    def _send_to_spectrum(self):
        if self.gam is None:
            # Normally populated when the energy panel draws; if no draw has
            # happened yet, bin it on demand so the first click sends instead of
            # silently doing nothing (and needing a second press). Flood-field
            # runs have no energy spectrum, so there is genuinely nothing to send.
            if self.df_current is None or self.ekey is None or self.flood_field:
                self._status("Nothing to send — load an API file first")
                return
            self._compute_energy_hist(self.df_current)
        dtype = self._data_type()
        # Prefer an energy axis when one exists: a smart calibration we applied,
        # else the native MeV of simulated/calibrated data; raw stays in channels.
        if self.e_units is not None:
            units = self.e_units
        elif dtype in (TYPE_SIMS, TYPE_CAL):
            units = "MeV"
        else:
            units = None
        try:
            if units is not None:
                spect = sp.Spectrum(counts=self.gam, energies=self.gam_x, e_units=units)
            else:
                # Uncalibrated: carry the *real* channel values (gam_x bin centres)
                # on adc_channels (a pure coordinate), leaving `channels` as 0..N
                # indices for the peak-finding/fitting machinery. A calibration
                # built here fits against spect.cal_channels — i.e. the real
                # channel space of the API dataframe's raw energy column — so its
                # coefficients apply to that column directly and "Retrieve
                # calibration" round-trips correctly.
                spect = sp.Spectrum(counts=self.gam)
                spect.adc_channels = np.asarray(self.gam_x, dtype=float)
        except Exception as exc:  # noqa: BLE001
            self._status(f"Could not build spectrum: {exc}")
            return
        # Load it as the active spectrum but stay on the API tab; the button turns
        # green to confirm instead of yanking the user away.
        self.app.load_external_spectrum(spect, "API spectrum", switch_tab=False)
        self._mark_send_sent()
        self._status("Sent energy spectrum to the Spectrum tab")

    def _mark_send_sent(self):
        b = self.opts.btn_send
        b.setText(SEND_SENT_TEXT); b.setObjectName("sent_btn")
        b.style().unpolish(b); b.style().polish(b)

    def _reset_send_button(self):
        b = self.opts.btn_send
        if b.objectName() != "open_btn":
            b.setObjectName("open_btn"); b.setText(SEND_DEFAULT_TEXT)
            b.style().unpolish(b); b.style().polish(b)
