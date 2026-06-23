"""API tab: explore Associated-Particle-Imaging parquet runs.

Port of the legacy ``ApiMixin`` main tab (``wara/gui/_mixins/api.py``), restyled
for the dark beta GUI and split into the Options / Page / Controller trio used by
the other beta tabs.

Workflow:

* Enter a run (date, run number, channel) and pick the data type  -- experimental
  *raw*, experimental *calibrated/time-shifted*, or *simulated*  -- then Load. The
  parquet file is read via :mod:`wara.read_parquet_api`; raw/calibrated data get
  reconstructed X-Y positions through :func:`wara.apicalc.calc_own_pos`.
* The figure shows three linked panels: the energy spectrum (top-left), the dt
  time histogram (bottom-left), and the X-Y hit map as a hexbin (right).
* Dragging a span on the energy or time panel, or a rectangle on the X-Y map,
  filters the event list live; the other panels redraw from the filtered data.
  The same filters can be entered by hand from the Filters dialog.
* "Send energy spectrum → Spectrum tab" hands the current energy histogram to the
  Spectrum tab for the full analysis workflow. An uncalibrated spectrum is sent
  with its *real* channel values (the bin centres) as the channel axis  -- not
  0..nbins indices  -- so a calibration built on it round-trips back here.
* "3D view" pops out a Plotly volume render (:class:`Api3DDialog`) of the
  reconstructed X-Y-Z hit cloud for the current (filtered) events.
* "Retrieve calibration" pulls the Calibration tab's current calibration curve
  (or equation) and applies that channel→energy polynomial to the dataframe  --
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
import shutil
from datetime import datetime

import numpy as np

from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.widgets import SpanSelector, RectangleSelector
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLineEdit, QCheckBox, QSizePolicy, QDialog,
    QGridLayout, QMessageBox, QColorDialog, QDialogButtonBox, QTabWidget,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import Qt, QSize, QTimer, QUrl

from wara import read_parquet_api, apicalc, combine_runs as cr
from wara import spectrum as sp
from wara import peaksearch as ps

from . import theme as T
from .widgets import hsep, header, labeled_row
from .slicefit import (
    plot_slice_spectra, plot_slice_offset, plot_slice_waterfall,
    SliceFitWindow, band_snr, ratio_to_ref, MAX_SLICES,
    TECHNIQUE_LABELS, TECHNIQUE_FROM_LABEL, YLABELS,
    TECH_FIT, TECH_SNR,
)
from . import xyprofile as xyp

API_PLOT_BG = T.BG_PLOT
COLORMAP = "plasma"
# Default histogram bin counts (energy spectrum, dt histogram, X-Y hexbin grid).
# Shown pre-filled in the Display "bins" boxes; high-resolution channels bump the
# energy bins at load (see _load_file).
DEFAULT_EBINS = 2 ** 12
DEFAULT_TBINS = 512
DEFAULT_HEXBINS = 80
# Faint gray density map for the *base* X-Y hexbin when colored energy
# selections are overlaid on top: low counts fade into the plot background, high
# counts brighten to a dim gray so the selections stay the focus.
GRAY_CMAP = LinearSegmentedColormap.from_list("api_gray", [API_PLOT_BG, T.TEXT_DIM])

# Send-to-Spectrum button label. The click is confirmed with a brief green blink
# (see ApiController._flash_button), not a persistent label change.
SEND_DEFAULT_TEXT = "Send to spectrum"


def _combo_row(label_text, widget):
    """A 'label  [widget]' row where the widget expands to fill the width  --
    unlike labeled_row's fixed-width inputs, so a long combo entry isn't cut
    off."""
    row = QWidget()
    rl = QHBoxLayout(row)
    rl.setContentsMargins(0, 0, 0, 0)
    rl.setSpacing(6)
    rl.addWidget(QLabel(label_text))
    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    rl.addWidget(widget, 1)
    return row


class ApiFilterDialog(QDialog):
    """Manual min/max entry for the X-Y, time and energy filters  -- the typed
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
        # The QWebEngineView renders white by default  -- paint the page background
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


class ApplyToDataDialog(QDialog):
    """Ask for the destination run when baking the active calibration/time-shift
    into a new on-disk run. The date defaults to the loaded run's date (editable)
    and the run number must be entered by the user  -- writing to a *new* run keeps
    the source run untouched and avoids the loader concatenating duplicates."""

    def __init__(self, date, runnr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Apply to data  -- save calibrated run")
        self.setStyleSheet(T.STYLESHEET)
        self.setMinimumWidth(340)

        lay = QVBoxLayout(self)
        lay.addWidget(header("SAVE CALIBRATED RUN"))
        note = QLabel(
            "Combine all of the source run's parquet files and bake the active "
            "energy calibration and/or time shift into energy_cal / dt_cal, then "
            "save as a new run. Pick a new run number so the original run is left "
            "untouched.")
        note.setObjectName("stat_key"); note.setWordWrap(True)
        lay.addWidget(note)

        self.ed_date = QLineEdit(str(date or "")); self.ed_date.setFixedWidth(150)
        drow, _ = labeled_row("Date", self.ed_date)
        lay.addWidget(drow)
        self.ed_run = QLineEdit(); self.ed_run.setFixedWidth(150)
        self.ed_run.setPlaceholderText("new run number")
        rrow, _ = labeled_row("Run", self.ed_run)
        lay.addWidget(rrow)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def values(self):
        """Return (date_str, runnr_int). runnr is None if it isn't an integer."""
        date = self.ed_date.text().strip()
        try:
            runnr = int(self.ed_run.text().strip())
        except ValueError:
            runnr = None
        return date, runnr


class ShiftsDialog(QDialog):
    """Drift-correction window: gain-shift the energy axis and shift/align the
    time axis, independently. Two tabs (Energy, Time), each showing the raw
    segment overlay above the aligned overlay (Preview), with controls to apply
    the correction to the API dataframe via the controller."""

    def __init__(self, controller):
        super().__init__(controller.app)
        self.c = controller
        self.setWindowTitle("Drift correction  -- shifts")
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
        self.f_bins = {}
        self.f_xmin = {}
        self.f_xmax = {}
        self.lbl_state = {}
        self.btn_clear = {}
        self.btn_apply = {}
        # When the user previews a time-segment alignment, cache the per-event
        # aligned dt (and its description) so a constant Δt can be applied
        # straight on top of the *previewed* data -- no re-alignment, no separate
        # "Apply alignment" step.
        self._time_preview_pending = False
        self._previewed_aligned = None
        self._previewed_seg_desc = ""

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
        self.ax_aligned[kind] = fig.add_subplot(212, sharex=self.ax_raw[kind],
                                                       sharey=self.ax_raw[kind])
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.setMinimumWidth(540)
        self.canvas[kind] = canvas
        # Pan/zoom toolbar above the plot  -- styled to match the API page's.
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

        default_bins = self.c.ebins if kind == "energy" else self.c.tbins
        bins = QLineEdit(str(default_bins)); bins.setFixedWidth(70)
        bins.setToolTip("Number of histogram bins used to align the segments"
                        + (" (time range focuses on the 1%–99% dt span)"
                           if kind == "time" else ""))
        r, _ = labeled_row("Bins", bins); side.addWidget(r)
        self.f_bins[kind] = bins

        xmin = QLineEdit(); xmin.setPlaceholderText("min"); xmin.setFixedWidth(70)
        xmax = QLineEdit(); xmax.setPlaceholderText("max"); xmax.setFixedWidth(70)
        xr = QHBoxLayout(); xr.setContentsMargins(0, 0, 0, 0)
        xr.addWidget(QLabel("Window")); xr.addWidget(xmin); xr.addWidget(xmax)
        xrw = QWidget(); xrw.setLayout(xr); side.addWidget(xrw)
        xmin.setToolTip("Optional: restrict the alignment to this axis window")
        self.f_xmin[kind] = xmin; self.f_xmax[kind] = xmax

        btn_prev = QPushButton("Preview"); btn_prev.setObjectName("action_btn")
        btn_prev.setCursor(Qt.PointingHandCursor)
        btn_prev.clicked.connect(lambda _=False, k=kind: self._preview(k, prospective=True))
        side.addWidget(btn_prev)

        btn_apply = QPushButton("Apply alignment")
        # Distinct colour per axis (green = energy panel, cyan = dt panel). Idle =
        # outlined (transparent fill); pressing fills it solid, and it returns to
        # the outline once the alignment is applied  -- so the user sees exactly
        # when the action takes place.
        btn_apply._color = T.ACCENT_GREEN if kind == "energy" else T.ACCENT_CYAN
        btn_apply._base_css = self._apply_css(btn_apply._color, solid=False)
        btn_apply.setStyleSheet(btn_apply._base_css)
        btn_apply.setCursor(Qt.PointingHandCursor)
        btn_apply.clicked.connect(lambda _=False, k=kind: self._apply_segments(k))
        self.btn_apply[kind] = btn_apply
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

    def _bins(self, kind):
        """User-defined histogram bin count for the tab (falls back to the main
        view's energy/time bins on a bad entry)."""
        default = self.c.ebins if kind == "energy" else self.c.tbins
        try:
            return max(2, int(float(self.f_bins[kind].text().strip())))
        except ValueError:
            return default

    def _axis_cfg(self, kind):
        """(base column, out column, bins, base erange) for the tab's axis."""
        c = self.c
        if kind == "energy":
            return (c._chan_base or "energy_orig", "energy_drift",
                    self._bins("energy"), c._chan_erange())
        return ("dt", "dt_cal", self._bins("time"), c._dt_erange())

    def _segment_spectra(self, col, n_segments, bins, erange):
        """Per-segment histograms of *col* (no alignment, nothing committed)."""
        gs = apicalc.GainShift(self.c.df_api, n_segments=n_segments, bins=bins,
                               erange=erange, col=col, out_col="_preview_tmp")
        return gs._spectra

    # ── actions ─────────────────────────────────────────────────────────────────
    def _preview(self, kind, prospective=False):
        """Redraw the tab: raw segments on top; on the bottom the *committed*
        correction (constant or segment) when one is applied. The prospective
        segment alignment for the current controls is computed only when the user
        asks for it (``prospective=True``, the Preview button) -- opening the
        window must not run any shift computation, just show the raw segments."""
        c = self.c
        if c.df_api is None:
            return
        nseg, method, xr = self._params(kind)
        base, out_col, bins, base_erange = self._axis_cfg(kind)
        applied = c._dt_corrected if kind == "time" else c._egain_applied
        bottom = None
        try:
            raw_spectra = self._segment_spectra(base, nseg, bins, base_erange)
            if applied and out_col in c.df_api.columns:
                # Histogram the corrected column over the SAME kind of range as the
                # raw panel (time: 0.2%–99.5% percentile; energy: [0, max]) so the
                # shared-x view keeps the percentile zoom instead of snapping back
                # to the full dt span after applying.
                vals = c.df_api[out_col].astype(float).to_numpy()
                if kind == "time":
                    lo, hi = np.percentile(vals, [0.2, 99.5])
                    erange = ((float(lo), float(hi)) if hi > lo
                              else (float(vals.min()), float(vals.max())))
                else:
                    erange = (0.0, float(vals.max()))
                bottom = self._segment_spectra(out_col, nseg, bins, erange)
                bottom_title = "Corrected"
            elif prospective:
                gs = apicalc.GainShift(c.df_api, n_segments=nseg, bins=bins,
                                       erange=base_erange, col=base, out_col=out_col)
                gs.align(method=method, xrange=xr)
                bottom = gs._aligned_spectra
                bottom_title = f"Aligned ({method})"
                # Cache the per-event aligned dt so a constant Δt can be applied
                # directly on top of this preview (see _apply_constant).
                if kind == "time":
                    self._time_preview_pending = True
                    self._previewed_aligned = gs.df[out_col].astype(float).to_numpy()
                    self._previewed_seg_desc = f"{nseg} seg · {method}"
            else:
                bottom_title = "Press Preview to compute the alignment"
        except Exception as exc:  # noqa: BLE001  -- surface to the state label
            self.lbl_state[kind].setText(f"Preview failed: {exc}")
            return
        xlabel = "Raw channel" if kind == "energy" else "dt (ns)"
        yscale = "log" if kind == "energy" else "linear"
        # Draw the bottom panel first, then the raw panel last: the axes share y,
        # so the raw overlay's autoscale sets the final (positive) limits.
        if bottom is not None:
            self._draw_overlay(self.ax_aligned[kind], bottom, nseg,
                               bottom_title, xlabel, yscale)
        else:
            self._draw_placeholder(self.ax_aligned[kind], bottom_title, xlabel, yscale)
        self._draw_overlay(self.ax_raw[kind], raw_spectra, nseg,
                           f"Raw segments ({nseg})", xlabel, yscale)
        self.fig[kind].tight_layout()
        self.canvas[kind].draw_idle()
        # Reset the pan/zoom history so "Home" returns to this fresh view.
        self.toolbar[kind].update()

    def _draw_placeholder(self, ax, message, xlabel, yscale="log"):
        """Empty aligned panel with a hint, shown until the user runs a preview."""
        ax.clear()
        ax.set_ylim(0.1, 1.0)          # positive, so a log scale doesn't warn
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, color=T.TEXT_DIM, fontsize=12)
        ax.set_facecolor(API_PLOT_BG)
        ax.tick_params(colors=T.TEXT_DIM, which="both", length=3, labelsize=11)
        for sp_ in ax.spines.values():
            sp_.set_color(T.BORDER)
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
                color=T.TEXT_DIM, fontsize=12)

    def _draw_overlay(self, ax, spectra, n_seg, title, xlabel, yscale="log"):
        ax.clear()
        colors = cm.viridis(np.linspace(0, 1, n_seg))
        for i, spe in enumerate(spectra):
            ax.step(spe.energies, spe.counts, where="mid",
                    color=colors[i], lw=0.8, alpha=0.8)
        ax.set_yscale(yscale)
        ax.set_title(title, color=T.TEXT_PRIMARY, fontsize=12)
        ax.set_xlabel(xlabel, color=T.TEXT_DIM, fontsize=12)
        ax.set_facecolor(API_PLOT_BG)
        ax.tick_params(colors=T.TEXT_DIM, which="both", length=3, labelsize=11)
        for sp_ in ax.spines.values():
            sp_.set_color(T.BORDER)
        handles = [
            Line2D([0], [0], color=cm.viridis(0.0), lw=1.5, label="earlier"),
            Line2D([0], [0], color=cm.viridis(1.0), lw=1.5, label="later"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=11,
                  facecolor=API_PLOT_BG, edgecolor=T.BORDER,
                  labelcolor=T.TEXT_DIM, framealpha=0.7)

    @staticmethod
    def _apply_css(color, solid):
        """Stylesheet for an Apply button. ``solid`` fills it with *color* (the
        pressed/active state); otherwise it is an outline with a transparent
        fill (the idle state)."""
        bg = color if solid else "transparent"
        fg = T.BG_DARK if solid else color
        weight = 800 if solid else 700
        return (f"background-color:{bg}; color:{fg}; "
                f"border:2px solid {color}; border-radius:5px; "
                f"padding:8px 13px; font-size:14px; font-weight:{weight};")

    def _apply_segments(self, kind):
        # Fill the button solid for the (blocking) alignment so the press is
        # visible, then revert to the outline once it is applied.
        b = self.btn_apply[kind]
        b.setStyleSheet(self._apply_css(b._color, solid=True))
        b.repaint()
        nseg, method, xr = self._params(kind)
        recal = False
        if kind == "energy":
            recal = self.c.apply_energy_gainshift(nseg, method=method, xrange=xr,
                                                  bins=self._bins("energy"))
        else:
            self.c.apply_dt_gainshift(nseg, method=method, xrange=xr,
                                      bins=self._bins("time"))
            self._time_preview_pending = False
            self._previewed_aligned = None
        self._preview(kind)
        QTimer.singleShot(220, lambda: b.setStyleSheet(b._base_css))
        # Confirm the action with a clear message (the controller already pushed
        # the descriptive label into lbl_state via _refresh_shift_labels).
        label = self.c._egain_label if kind == "energy" else self.c._dt_label
        axis = "Energy" if kind == "energy" else "Time"
        self.lbl_state[kind].setText(f"✓ {axis} alignment applied  --  {label}")
        # The energy shift recomputed the energy axis, dropping a file-provided
        # calibration  -- tell the user it must be redone.
        if recal:
            QMessageBox.information(
                self, "Re-calibrate after shifting",
                "The energy gain-shift recomputed the energy axis from the raw "
                "channels, so the run's previous energy calibration was "
                "removed.\n\nRe-calibrate the energy before any quantitative "
                "analysis.")

    def _apply_constant(self):
        txt = self.f_const.text().strip()
        try:
            shift = float(txt)
        except ValueError:
            self.lbl_state["time"].setText("Constant shift must be a number (ns)")
            return
        # If a segment alignment was previewed (and not yet committed), apply the
        # constant directly on top of that previewed data -- using the exact
        # aligned values the user saw, not a fresh re-alignment.
        prev = self._previewed_aligned
        if (self._time_preview_pending and prev is not None
                and not self.c._dt_segments_applied
                and self.c.df_api is not None and len(prev) == len(self.c.df_api)):
            self.c.apply_dt_preview_shift(prev, shift, self._previewed_seg_desc)
        else:
            self.c.apply_dt_shift(shift)
        self._preview("time")
        self.lbl_state["time"].setText(
            f"✓ Time correction applied  --  {self.c._dt_label}")

    def _clear(self, kind):
        if kind == "energy":
            self.c.clear_energy_gainshift()
        else:
            self.c._clear_dt_shift()
            self._time_preview_pending = False
            self._previewed_aligned = None
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


class CombineRunsDialog(QDialog):
    """Visualize and stitch several API runs (any dates) into one.

    The Energy and Time tabs overlay the per-run spectra for the selected channel
    so you can compare the runs before combining; "Combine & save" simply
    concatenates the runs (in table order) and writes the result as a new run
    (settings copied from the first run + provenance README). No drift correction
    is applied here  -- if the combined run needs gain/time alignment, do it
    afterwards with the single-run Shifts window (combine first, then shift)."""

    def __init__(self, controller):
        super().__init__(controller.app)
        self.c = controller
        self.setWindowTitle("Combine multiple runs")
        self.setStyleSheet(T.STYLESHEET)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.resize(1100, 720)

        self.runs_data = None        # list of (date, runnr, df) once loaded
        self.channels = []           # channels common to all loaded runs
        self._data_path = None       # data path of the first seeded/loaded run

        self.ax_raw = {}
        self.canvas = {}
        self.toolbar = {}
        self.fig = {}

        root = QHBoxLayout(self); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)

        # Plot side: Energy / Time tabs, each raw-over-aligned.
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_plot_tab("energy"), "Energy")
        self.tabs.addTab(self._build_plot_tab("time"), "Time")
        root.addWidget(self.tabs, 1)

        # Control side.
        root.addWidget(self._build_controls(), 0)

    # ── construction ──────────────────────────────────────────────────────────
    def _build_plot_tab(self, kind):
        tab = QWidget()
        col = QVBoxLayout(tab); col.setContentsMargins(0, 0, 0, 0); col.setSpacing(2)
        fig = Figure(figsize=(6, 6), facecolor=API_PLOT_BG)
        self.fig[kind] = fig
        self.ax_raw[kind] = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.setMinimumWidth(540)
        self.canvas[kind] = canvas
        toolbar = NavToolbar(canvas, self)
        toolbar.setObjectName("plot_toolbar")
        toolbar.setIconSize(QSize(22, 22))
        T.recolor_toolbar_icons(toolbar, T.TEXT_PRIMARY)
        self.toolbar[kind] = toolbar
        col.addWidget(toolbar)
        col.addWidget(canvas, 1)
        self._draw_placeholder(self.ax_raw[kind], "Add runs and press Load runs",
                               self._xlabel(kind), self._yscale(kind))
        return tab

    def _build_controls(self):
        side = QVBoxLayout(); side.setSpacing(6)
        side.addWidget(header("RUNS TO COMBINE"))
        note = QLabel("Runs are concatenated in table order (dates may differ). "
                      "No shifting is applied  -- align the combined run later "
                      "with the Shifts window if needed.")
        note.setObjectName("stat_key"); note.setWordWrap(True)
        side.addWidget(note)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Date", "Run"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(150)
        side.addWidget(self.table)

        rowbtns = QHBoxLayout(); rowbtns.setContentsMargins(0, 0, 0, 0)
        btn_add = QPushButton("+ Add row"); btn_add.setObjectName("mini_btn")
        btn_add.setCursor(Qt.PointingHandCursor)
        btn_add.clicked.connect(lambda: self._add_row())
        btn_del = QPushButton("Remove selected"); btn_del.setObjectName("mini_btn")
        btn_del.setCursor(Qt.PointingHandCursor)
        btn_del.clicked.connect(self._remove_selected)
        rowbtns.addWidget(btn_add); rowbtns.addWidget(btn_del)
        rw = QWidget(); rw.setLayout(rowbtns); side.addWidget(rw)

        btn_load = QPushButton("Load runs"); btn_load.setObjectName("primary_btn")
        btn_load.setCursor(Qt.PointingHandCursor)
        btn_load.clicked.connect(self._load_runs)
        side.addWidget(btn_load)

        side.addWidget(hsep()); side.addWidget(header("VISUALIZATION"))
        self.cb_channel = QComboBox()
        self.cb_channel.setToolTip("Channel shown in the overlay plots")
        self.cb_channel.currentIndexChanged.connect(lambda *_: self._preview_all())
        r, _ = labeled_row("Preview ch", self.cb_channel); side.addWidget(r)

        self.ed_ebins = QLineEdit(str(DEFAULT_EBINS)); self.ed_ebins.setFixedWidth(70)
        self.ed_ebins.setToolTip("Number of energy bins in the overlay plot")
        r, _ = labeled_row("Energy bins", self.ed_ebins); side.addWidget(r)
        self.ed_tbins = QLineEdit(str(DEFAULT_TBINS)); self.ed_tbins.setFixedWidth(70)
        self.ed_tbins.setToolTip("Number of time (dt) bins in the overlay plot")
        r, _ = labeled_row("Time bins", self.ed_tbins); side.addWidget(r)

        btn_prev = QPushButton("Preview"); btn_prev.setObjectName("action_btn")
        btn_prev.setCursor(Qt.PointingHandCursor)
        btn_prev.clicked.connect(self._preview_all)
        side.addWidget(btn_prev)

        side.addWidget(hsep()); side.addWidget(header("SAVE COMBINED RUN"))
        self.ed_date = QLineEdit(); self.ed_date.setFixedWidth(150)
        r, _ = labeled_row("Date", self.ed_date); side.addWidget(r)
        self.ed_run = QLineEdit(); self.ed_run.setFixedWidth(150)
        self.ed_run.setPlaceholderText("new run number")
        r, _ = labeled_row("Run", self.ed_run); side.addWidget(r)

        btn_save = QPushButton("Combine && save"); btn_save.setObjectName("open_btn")
        btn_save.setCursor(Qt.PointingHandCursor)
        btn_save.clicked.connect(self._combine_save)
        side.addWidget(btn_save)

        self.lbl_state = QLabel(""); self.lbl_state.setObjectName("stat_key")
        self.lbl_state.setWordWrap(True); side.addWidget(self.lbl_state)
        side.addStretch(1)

        holder = QWidget(); holder.setFixedWidth(300); holder.setLayout(side)
        return holder

    # ── runs table ──────────────────────────────────────────────────────────
    def _add_row(self, date="", runnr=""):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(str(date)))
        self.table.setItem(r, 1, QTableWidgetItem(str(runnr)))

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _table_runs(self):
        """Parse the table into a list of (date, runnr); skips blank rows and
        returns None on a bad run number (with a status message)."""
        runs = []
        for r in range(self.table.rowCount()):
            d_item = self.table.item(r, 0)
            n_item = self.table.item(r, 1)
            date = d_item.text().strip() if d_item else ""
            ntxt = n_item.text().strip() if n_item else ""
            if not date and not ntxt:
                continue
            try:
                runnr = int(ntxt)
            except ValueError:
                self.lbl_state.setText(f"Row {r + 1}: run number must be an integer")
                return None
            runs.append((date, runnr))
        return runs

    def seed(self, date, runnr, data_path):
        """Pre-fill the first run (the one currently open in the API tab) when the
        table is still empty, so the common 'this run plus a few more' flow starts
        ready to go."""
        self._data_path = data_path
        if self.table.rowCount() == 0:
            self._add_row(date, runnr)
            self._add_row()  # one blank row ready for the next run
        if not self.ed_date.text().strip():
            self.ed_date.setText(str(date or ""))

    # ── load / preview ────────────────────────────────────────────────────────
    def _load_runs(self):
        runs = self._table_runs()
        if runs is None:
            return
        if len(runs) < 2:
            self.lbl_state.setText("Add at least two runs to combine")
            return
        self.lbl_state.setText("Loading runs...")
        try:
            self.runs_data = cr.read_runs(runs, data_path=self._data_path)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.lbl_state.setText(f"Could not load runs: {exc}")
            self.runs_data = None
            return
        self.channels = cr.channels_in_common(self.runs_data)
        self.cb_channel.blockSignals(True)
        self.cb_channel.clear()
        self.cb_channel.addItems([str(ch) for ch in self.channels])
        self.cb_channel.blockSignals(False)
        n = sum(len(df) for _, _, df in self.runs_data)
        chan_txt = (', '.join(str(c) for c in self.channels)
                    if self.channels else "none shared (preview off)")
        self.lbl_state.setText(
            f"Loaded {len(self.runs_data)} runs · channels {chan_txt} "
            f"· {n:,} events")
        self._preview_all()

    def _preview_channel(self):
        try:
            return self.channels[self.cb_channel.currentIndex()]
        except (IndexError, ValueError):
            return None

    def _bins(self, kind):
        ed = self.ed_ebins if kind == "energy" else self.ed_tbins
        default = DEFAULT_EBINS if kind == "energy" else DEFAULT_TBINS
        try:
            return max(2, int(float(ed.text().strip())))
        except ValueError:
            return default

    def _preview_all(self):
        if self.runs_data is None:
            return
        for kind in ("energy", "time"):
            self._preview(kind)

    def _preview(self, kind):
        ch = self._preview_channel()
        if self.runs_data is None or ch is None:
            return
        try:
            spectra = cr.run_spectra(self.runs_data, ch, kind,
                                     bins=self._bins(kind))
        except Exception as exc:  # noqa: BLE001
            self.lbl_state.setText(f"Preview failed: {exc}")
            return
        labels = [f"{d}-{r}" for d, r, _ in self.runs_data]
        self._draw_overlay(self.ax_raw[kind], spectra, labels,
                           f"Runs overlay  -- ch {ch}",
                           self._xlabel(kind), self._yscale(kind))
        self.fig[kind].tight_layout()
        self.canvas[kind].draw_idle()
        self.toolbar[kind].update()

    # ── drawing ───────────────────────────────────────────────────────────────
    @staticmethod
    def _xlabel(kind):
        return "Raw channel" if kind == "energy" else "dt (ns)"

    @staticmethod
    def _yscale(kind):
        return "log" if kind == "energy" else "linear"

    def _style_axis(self, ax, xlabel, yscale):
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, color=T.TEXT_DIM, fontsize=12)
        ax.set_facecolor(API_PLOT_BG)
        ax.tick_params(colors=T.TEXT_DIM, which="both", length=3, labelsize=11)
        for sp_ in ax.spines.values():
            sp_.set_color(T.BORDER)

    def _draw_placeholder(self, ax, message, xlabel, yscale="log"):
        ax.clear()
        ax.set_ylim(0.1, 1.0)
        self._style_axis(ax, xlabel, yscale)
        if message:
            ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center",
                    va="center", color=T.TEXT_DIM, fontsize=12)

    def _draw_overlay(self, ax, spectra, labels, title, xlabel, yscale="log"):
        ax.clear()
        # Start above viridis's dark-blue end so the first (reference) run reads
        # as a light teal that stands out against the dark plot background.
        n = max(len(spectra), 1)
        colors = cm.viridis(np.linspace(0.35, 1.0, n))
        for i, spe in enumerate(spectra):
            ax.step(spe.energies, spe.counts, where="mid", color=colors[i],
                    lw=0.9, alpha=0.85, label=labels[i] if i < len(labels) else None)
        ax.set_title(title, color=T.TEXT_PRIMARY, fontsize=12)
        self._style_axis(ax, xlabel, yscale)
        ax.legend(loc="upper right", fontsize=9, facecolor=API_PLOT_BG,
                  edgecolor=T.BORDER, labelcolor=T.TEXT_DIM, framealpha=0.7,
                  ncol=2 if len(spectra) > 4 else 1)

    # ── combine & save ──────────────────────────────────────────────────────
    def _combine_save(self):
        if self.runs_data is None:
            self.lbl_state.setText("Load the runs first")
            return
        new_date = self.ed_date.text().strip()
        if not new_date:
            self.lbl_state.setText("Enter a date for the combined run")
            return
        try:
            new_runnr = int(self.ed_run.text().strip())
        except ValueError:
            self.lbl_state.setText("New run number must be an integer")
            return

        try:
            run_dir, _, _ = read_parquet_api.run_parquet_path(
                new_date, new_runnr, self._data_path)
        except Exception as exc:  # noqa: BLE001
            self.lbl_state.setText(f"Bad destination date/run: {exc}")
            return
        if run_dir.exists():
            resp = QMessageBox.question(
                self, "Overwrite run?",
                f"Run {new_date}-{new_runnr} already exists at:\n{run_dir}\n\n"
                "Overwrite its combined parquet data?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if resp != QMessageBox.Yes:
                self.lbl_state.setText("Combine cancelled")
                return

        self.lbl_state.setText("Combining...")
        try:
            combined, info = cr.combine_runs(self.runs_data)
            out = read_parquet_api.save_combined_run(
                combined, new_date, new_runnr, self._data_path, overwrite=True)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.lbl_state.setText(f"Could not combine/save: {exc}")
            return

        extra = self._write_metadata(run_dir, new_date, new_runnr, info)
        msg = (f"Saved combined run {new_date}-{new_runnr} "
               f"({info['n_events']:,} events from {len(info['sources'])} runs) "
               f"→ {out}")
        if extra:
            msg += f"  ·  {extra}"
        self.lbl_state.setText(msg)
        self.c._status(msg)

        # Combining invalidates any per-run calibration; warn that the dropped
        # columns mean the combined run must be re-calibrated before analysis.
        if info["dropped_cal"]:
            QMessageBox.information(
                self, "Combined run is uncalibrated",
                f"Saved combined run {new_date}-{new_runnr}.\n\n"
                f"The source runs carried calibration columns "
                f"({', '.join(info['dropped_cal'])}) which were removed  -- "
                "combining runs invalidates them.\n\n"
                "Re-calibrate the energy (and re-align the time, if needed) on "
                "the combined run before any quantitative analysis.")

    def _write_metadata(self, dst_run_dir, new_date, new_runnr, info):
        """Copy the first run's settings folder as a starting point and write a
        provenance README listing every source run. Best-effort; returns a short
        note. Combined live-time is *not* summed automatically  -- the README
        flags that the settings come from the first run only."""
        notes = []
        first_date, first_runnr, _ = self.runs_data[0]
        try:
            src_run_dir, _, _ = read_parquet_api.run_parquet_path(
                first_date, first_runnr, self._data_path)
        except Exception:  # noqa: BLE001
            src_run_dir = None

        if src_run_dir is not None:
            src_settings = src_run_dir / "settings"
            if src_settings.is_dir():
                try:
                    shutil.copytree(src_settings, dst_run_dir / "settings",
                                    dirs_exist_ok=True)
                    notes.append("settings copied (from first run)")
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
                    notes.append("settings copy failed")

        try:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                "This run was combined (concatenated) from multiple source runs "
                "by the wara API tab. No drift correction was applied.",
                "",
                f"Combined on : {stamp}",
                f"New run     : {new_date}-{new_runnr}",
                f"Total events: {info['n_events']:,}",
                "",
                "Source runs (in combine order):",
            ]
            for d, r, n in info["sources"]:
                lines.append(f"  - {d}-{r}  ({n:,} events)")
            if info["dropped_cal"]:
                lines += [
                    "",
                    f"Calibration columns removed: {', '.join(info['dropped_cal'])}"
                    "  -- combining invalidates per-run calibration. "
                    "RE-CALIBRATE the energy (and re-align time, if needed) on "
                    "this combined run before analysis.",
                ]
            lines += [
                "",
                "NOTE: the settings/ folder was copied from the first run only; "
                "live-time and count totals across the combined runs are NOT "
                "summed automatically and may need manual adjustment.",
            ]
            (dst_run_dir / "README.txt").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")
            notes.append("README written")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        return "; ".join(notes)


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

    # Minimum readable font sizes shared across the API plots.
    LABEL_FS = 12
    TICK_FS = 11

    def _style(self, ax, grid=False):
        ax.set_facecolor(API_PLOT_BG)
        ax.tick_params(colors=T.TEXT_DIM, which="both", length=3,
                       labelsize=self.TICK_FS)
        ax.xaxis.label.set_color(T.TEXT_DIM)
        ax.yaxis.label.set_color(T.TEXT_DIM)
        ax.xaxis.label.set_fontsize(self.LABEL_FS)
        ax.yaxis.label.set_fontsize(self.LABEL_FS)
        if ax.get_title():
            ax.title.set_color(T.TEXT_PRIMARY)
            ax.title.set_fontsize(self.LABEL_FS)
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
        # On the X-Y map, report the hovered hexagon's intensity in amber  -- this
        # stands in for the colorbar the panel no longer carries.
        if ax is self.ax_xy:
            cnt = self.xy_count_at(event.xdata, event.ydata)
            if cnt is not None:
                parts.append(
                    f"<span style='color:{T.ACCENT_AMBER}'>Counts: {cnt:,.0f}</span>")
        self.readout.setText("&nbsp;&nbsp;&nbsp;&nbsp;".join(parts))


# ── Selections dialog ─────────────────────────────────────────────────────────
class SelectionsDialog(QDialog):
    """Non-modal window for managing energy selections and plotting S/B vs dt.

    The dialog holds the full Add / Remove / Clear selection workflow that used
    to live inline in the options panel, plus the time-slice-fits controls
    (pick a selection, dt slice width and technique to profile a line vs dt).
    Designed to grow: future tabs will add dt and X-Y selection management here.
    """

    def __init__(self, controller):
        super().__init__(controller.app)
        self.c = controller
        self.setWindowTitle("E / X-Y Selections")
        self.setStyleSheet(T.STYLESHEET)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.resize(1180, 620)

        # Three tabs: Energy (implemented), dt and X-Y (placeholders for now).
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        self.tabs = QTabWidget()
        self._fit_tabbar(self.tabs)
        root.addWidget(self.tabs)

        # ── Energy selections tab ─────────────────────────────────────
        energy_tab = QWidget()
        outer = QHBoxLayout(energy_tab)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(12)

        # Left: a compact, fixed-width controls column so the buttons and boxes
        # stay nicely sized instead of stretching across the wide window.
        left_w = QWidget()
        left_w.setFixedWidth(320)
        lay = QVBoxLayout(left_w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        outer.addWidget(left_w, 0)

        # ── Selection list ────────────────────────────────────────────
        lay.addWidget(header("ENERGY SELECTIONS"))
        self.sel_box = QVBoxLayout()
        self.sel_box.setSpacing(3)
        self.sel_box.setContentsMargins(0, 0, 0, 0)
        sel_holder = QWidget()
        sel_holder.setLayout(self.sel_box)
        lay.addWidget(sel_holder)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)
        self.btn_add = QPushButton("Add selection")
        self.btn_add.setObjectName("yellow_btn")
        self.btn_add.setCursor(Qt.PointingHandCursor)
        self.btn_add.setToolTip(
            "Arm: then drag a band on the energy spectrum to tag a line")
        self.btn_clear_sel = QPushButton("Clear all")
        self.btn_clear_sel.setObjectName("danger_btn")
        self.btn_clear_sel.setCursor(Qt.PointingHandCursor)
        btn_row.addWidget(self.btn_add, 1)
        btn_row.addWidget(self.btn_clear_sel, 0)
        lay.addLayout(btn_row)

        self.btn_plot_sel = QPushButton("Plot selections")
        self.btn_plot_sel.setObjectName("open_btn")
        self.btn_plot_sel.setCursor(Qt.PointingHandCursor)
        self.btn_plot_sel.setToolTip(
            "Overlay each selection on the energy, dt and X-Y panels")
        lay.addWidget(self.btn_plot_sel)

        self.btn_clear_plots = QPushButton("Clear plots")
        self.btn_clear_plots.setObjectName("mini_btn")
        self.btn_clear_plots.setCursor(Qt.PointingHandCursor)
        self.btn_clear_plots.setToolTip(
            "Remove selection overlays from the energy, dt and X-Y panels")
        lay.addWidget(self.btn_clear_plots)

        # ── Time-slice fits vs dt ─────────────────────────────────────
        lay.addWidget(hsep())
        lay.addWidget(header("TIME-SLICE FITS vs dt"))
        note = QLabel(
            "Split dt into slices and measure one selection's line in each, "
            "then overlay the value vs dt on the dt panel.")
        note.setObjectName("stat_key")
        note.setWordWrap(True)
        lay.addWidget(note)

        self.cmb_sel = QComboBox()
        self.cmb_sel.setToolTip("Which selection (energy line) to profile vs dt")
        lay.addWidget(_combo_row("Selection", self.cmb_sel))

        self.cmb_tech = QComboBox()
        self.cmb_tech.addItems([TECHNIQUE_LABELS[k]
                                for k in (TECH_FIT, TECH_SNR)])
        self.cmb_tech.setToolTip(
            "Fit: open the interactive slice-fit window; its Method selector "
            "chooses peak fit vs net area − linear bkg per slice.\n"
            "SNR: peak signal-to-noise from the kernel search (no fitting).")
        lay.addWidget(_combo_row("Technique", self.cmb_tech))

        self.ed_dt_slice = QLineEdit("2")
        self.ed_dt_slice.setFixedWidth(80)
        self.ed_dt_slice.setPlaceholderText("auto")
        self.ed_dt_slice.setToolTip(
            "dt slice width (ns).  Leave blank for ~10 slices over the dt range.")
        drow, _ = labeled_row("dt slice (ns)", self.ed_dt_slice)
        lay.addWidget(drow)

        self.ed_min_snr = QLineEdit("3")
        self.ed_min_snr.setFixedWidth(80)
        self.ed_min_snr.setToolTip(
            "Minimum SNR for the per-slice peak search.  Lower finds weaker "
            "peaks (more candidates in the fit window); higher is stricter.\n"
            "Change it and click Slice & fit again to re-find peaks.")
        snrrow, _ = labeled_row("Min SNR", self.ed_min_snr)
        lay.addWidget(snrrow)

        sf_btn_row = QHBoxLayout()
        sf_btn_row.setContentsMargins(0, 0, 0, 0)
        sf_btn_row.setSpacing(6)
        self.btn_slice_fit = QPushButton("Slice && fit")
        self.btn_slice_fit.setObjectName("primary_btn")
        self.btn_slice_fit.setCursor(Qt.PointingHandCursor)
        self.btn_slice_fit.setToolTip(
            "Build the per-slice spectra, open the spectra figure and "
            "(for fit techniques) the interactive slice-fit window")
        self.btn_clear_slice = QPushButton("Clear")
        self.btn_clear_slice.setObjectName("mini_btn")
        self.btn_clear_slice.setCursor(Qt.PointingHandCursor)
        self.btn_clear_slice.setToolTip("Remove the vs-dt overlay and stored results")
        sf_btn_row.addWidget(self.btn_slice_fit, 1)
        sf_btn_row.addWidget(self.btn_clear_slice, 0)
        lay.addLayout(sf_btn_row)

        # Normalize the vs-dt overlay by a reference selection (e.g. Mg/Si).
        # Enabled once at least two selections have a stored vs-dt curve.
        self.cmb_ratio_ref = QComboBox()
        self.cmb_ratio_ref.setEnabled(False)
        self.cmb_ratio_ref.setToolTip(
            "Plot each selection's vs-dt curve divided by this reference "
            "selection (e.g. Mg/Si, Fe/Si).  Needs at least two plotted "
            "selections; choose “(absolute)” for the raw values.")
        self.cmb_ratio_ref.currentIndexChanged.connect(self._on_ratio_ref_changed)
        lay.addWidget(_combo_row("Ratio to", self.cmb_ratio_ref))
        lay.addStretch(1)   # keep the controls packed at the top

        # Right: the per-slice energy spectra take the rest of the wide window
        # (embedded canvas + navigation toolbar for zoom/pan).
        right_w = QWidget()
        rlay = QVBoxLayout(right_w)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(6)
        # Header row with a view toggle: overlaid spectra vs waterfall (offset).
        hdr_row = QHBoxLayout()
        hdr_row.setContentsMargins(0, 0, 0, 0)
        hdr_row.addWidget(header("PER-SLICE SPECTRA"))
        hdr_row.addStretch(1)
        view_lbl = QLabel("View:"); view_lbl.setObjectName("stat_key")
        hdr_row.addWidget(view_lbl)
        self.cmb_spectra_view = QComboBox()
        self.cmb_spectra_view.addItems(["Overlay", "Offset", "Waterfall"])
        self.cmb_spectra_view.setToolTip(
            "Overlay: all per-slice spectra on one log axis.\n"
            "Offset: spectra offset vertically per dt slice (earliest at the "
            "bottom), coloured by dt.\n"
            "Waterfall: 2-D heatmap of energy × dt, colour = counts (log).")
        self.cmb_spectra_view.currentIndexChanged.connect(
            self._on_spectra_view_changed)
        hdr_row.addWidget(self.cmb_spectra_view)
        rlay.addLayout(hdr_row)
        # Remembered args so the view toggle can redraw without re-slicing.
        self._spectra_args = None
        self.spectra_fig = Figure(facecolor=API_PLOT_BG)
        self.spectra_canvas = FigureCanvas(self.spectra_fig)
        self.spectra_canvas.setMinimumHeight(300)
        self.spectra_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.spectra_toolbar = NavToolbar(self.spectra_canvas, self)
        self.spectra_toolbar.setObjectName("plot_toolbar")
        self.spectra_toolbar.setIconSize(QSize(22, 22))
        T.recolor_toolbar_icons(self.spectra_toolbar, T.TEXT_PRIMARY)
        rlay.addWidget(self.spectra_toolbar)
        rlay.addWidget(self.spectra_canvas, 1)
        outer.addWidget(right_w, 1)

        self.tabs.addTab(energy_tab, "Energy selections")
        self.tabs.addTab(self._build_xy_tab(), "X-Y selections")

        # ── Wire internal buttons to the controller ───────────────────
        self.btn_add.clicked.connect(self.c._arm_selection)
        self.btn_clear_sel.clicked.connect(self.c._clear_selections)
        self.btn_plot_sel.clicked.connect(self.c._plot_selections)
        self.btn_clear_plots.clicked.connect(self.c._clear_selection_plots)
        self.btn_slice_fit.clicked.connect(self._on_slice_fit)
        self.btn_clear_slice.clicked.connect(self.c._clear_slice_overlay)

        # X-Y tab wiring.
        self.xy_btn_build.clicked.connect(self.c._xy_build_tiles)
        self.xy_btn_all.clicked.connect(self.c._xy_select_all)
        self.xy_btn_clear_tiles.clicked.connect(self.c._xy_clear_tiles)
        self.xy_btn_add_band.clicked.connect(self.c._xy_arm_band)
        self.xy_btn_clear_bands.clicked.connect(self.c._xy_clear_bands)
        self.xy_btn_fit.clicked.connect(self.c._xy_fit_tiles)
        self.xy_btn_clear_area.clicked.connect(self.c._xy_clear_area)

    @staticmethod
    def _fit_tabbar(tabw):
        """Stop a QTabWidget clipping its labels.

        The global stylesheet paints the tabs bold 14px, but the tab bar sizes
        itself with the *default* font, so it ends up too narrow and the text is
        chopped on both sides.  Give the bar the matching bold font (so the size
        hint is right), disable eliding, and add a little width so the labels
        sit comfortably."""
        bar = tabw.tabBar()
        f = bar.font()
        f.setPixelSize(14)
        f.setBold(True)
        bar.setFont(f)
        bar.setElideMode(Qt.ElideNone)
        bar.setExpanding(False)
        bar.setUsesScrollButtons(False)
        # Merges with the app stylesheet (keeps the themed colours); only widens.
        tabw.setStyleSheet("QTabBar::tab { padding: 7px 24px; min-width: 96px; }")

    @staticmethod
    def _placeholder_tab(message):
        """A simple centred 'coming soon' tab for not-yet-implemented sections."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lbl = QLabel(message)
        lbl.setObjectName("stat_key")
        lbl.setAlignment(Qt.AlignCenter)
        lay.addStretch(1)
        lay.addWidget(lbl)
        lay.addStretch(1)
        return w

    def _build_xy_tab(self):
        """Build the X-Y tile-selections tab.

        Left: tile the X-Y plane, manage energy bands and fit the tiles. Right:
        three sub-tabs — the X-Y map (click tiles to select), the selected
        tiles' overlaid spectra (drag energy bands), and the net-area-vs-X / vs-Y
        plots.
        """
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(12)

        left_w = QWidget()
        left_w.setFixedWidth(320)
        lay = QVBoxLayout(left_w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        outer.addWidget(left_w, 0)

        # ── Tiling ────────────────────────────────────────────────────
        lay.addWidget(header("TILE THE X-Y PLANE"))
        note = QLabel(
            "Tile the current X-Y map into uniform rectangles, then click "
            "tiles to overlay their energy spectra. Larger tiles pool more "
            "counts; smaller tiles resolve position.")
        note.setObjectName("stat_key")
        note.setWordWrap(True)
        lay.addWidget(note)

        self.xy_ed_w = QLineEdit("0.3")
        self.xy_ed_w.setFixedWidth(80)
        self.xy_ed_w.setToolTip("Tile width along X, in plane units")
        wrow, _ = labeled_row("Tile width (X)", self.xy_ed_w)
        lay.addWidget(wrow)
        self.xy_ed_h = QLineEdit("0.3")
        self.xy_ed_h.setFixedWidth(80)
        self.xy_ed_h.setToolTip("Tile length along Y, in plane units")
        hrow, _ = labeled_row("Tile length (Y)", self.xy_ed_h)
        lay.addWidget(hrow)

        self.xy_btn_build = QPushButton("Build tiles")
        self.xy_btn_build.setObjectName("primary_btn")
        self.xy_btn_build.setCursor(Qt.PointingHandCursor)
        self.xy_btn_build.setToolTip(
            "Lay the tile grid over the current X-Y map and snapshot the events")
        lay.addWidget(self.xy_btn_build)
        self.xy_lbl_grid = QLabel("No grid")
        self.xy_lbl_grid.setObjectName("stat_key")
        lay.addWidget(self.xy_lbl_grid)

        tile_btns = QHBoxLayout()
        tile_btns.setContentsMargins(0, 0, 0, 0)
        tile_btns.setSpacing(6)
        self.xy_btn_all = QPushButton("Select all")
        self.xy_btn_all.setObjectName("mini_btn")
        self.xy_btn_all.setCursor(Qt.PointingHandCursor)
        self.xy_btn_clear_tiles = QPushButton("Clear tiles")
        self.xy_btn_clear_tiles.setObjectName("mini_btn")
        self.xy_btn_clear_tiles.setCursor(Qt.PointingHandCursor)
        tile_btns.addWidget(self.xy_btn_all)
        tile_btns.addWidget(self.xy_btn_clear_tiles)
        lay.addLayout(tile_btns)

        # ── Energy bands ──────────────────────────────────────────────
        lay.addWidget(hsep())
        lay.addWidget(header("ENERGY BANDS"))
        bnote = QLabel(
            "Arm Add band, then drag a band on the Tile spectra sub-tab to tag "
            "a line. Each band seeds the per-tile fit ROI.")
        bnote.setObjectName("stat_key")
        bnote.setWordWrap(True)
        lay.addWidget(bnote)

        self.xy_band_box = QVBoxLayout()
        self.xy_band_box.setSpacing(3)
        self.xy_band_box.setContentsMargins(0, 0, 0, 0)
        band_holder = QWidget()
        band_holder.setLayout(self.xy_band_box)
        lay.addWidget(band_holder)

        band_btns = QHBoxLayout()
        band_btns.setContentsMargins(0, 0, 0, 0)
        band_btns.setSpacing(6)
        self.xy_btn_add_band = QPushButton("Add band")
        self.xy_btn_add_band.setObjectName("yellow_btn")
        self.xy_btn_add_band.setCursor(Qt.PointingHandCursor)
        self.xy_btn_clear_bands = QPushButton("Clear all")
        self.xy_btn_clear_bands.setObjectName("danger_btn")
        self.xy_btn_clear_bands.setCursor(Qt.PointingHandCursor)
        band_btns.addWidget(self.xy_btn_add_band, 1)
        band_btns.addWidget(self.xy_btn_clear_bands, 0)
        lay.addLayout(band_btns)

        # ── Net area vs position ──────────────────────────────────────
        lay.addWidget(hsep())
        lay.addWidget(header("NET AREA vs POSITION"))
        fnote = QLabel(
            "Pick a band and fit each selected tile interactively; Plot vs X/Y "
            "lands one point per tile on the position plots.")
        fnote.setObjectName("stat_key")
        fnote.setWordWrap(True)
        lay.addWidget(fnote)

        self.xy_cmb_band = QComboBox()
        self.xy_cmb_band.setToolTip("Which energy band to fit across the tiles")
        lay.addWidget(_combo_row("Band", self.xy_cmb_band))

        self.xy_cmb_axis = QComboBox()
        self.xy_cmb_axis.addItems(
            ["Per tile — vs X & Y", "Combine → vs X", "Combine → vs Y"])
        self.xy_cmb_axis.setToolTip(
            "Per tile: one point per selected tile, plotted vs X and vs Y.\n"
            "Combine → vs X: sum the selected tiles sharing each column into one "
            "spectrum, one point per X (Y collapsed).\n"
            "Combine → vs Y: sum tiles sharing each row, one point per Y.")
        lay.addWidget(_combo_row("Plot", self.xy_cmb_axis))

        fit_btns = QHBoxLayout()
        fit_btns.setContentsMargins(0, 0, 0, 0)
        fit_btns.setSpacing(6)
        self.xy_btn_fit = QPushButton("Fit tiles")
        self.xy_btn_fit.setObjectName("primary_btn")
        self.xy_btn_fit.setCursor(Qt.PointingHandCursor)
        self.xy_btn_fit.setToolTip(
            "Open the interactive stepping fit window over the selected tiles")
        self.xy_btn_clear_area = QPushButton("Clear")
        self.xy_btn_clear_area.setObjectName("mini_btn")
        self.xy_btn_clear_area.setCursor(Qt.PointingHandCursor)
        fit_btns.addWidget(self.xy_btn_fit, 1)
        fit_btns.addWidget(self.xy_btn_clear_area, 0)
        lay.addLayout(fit_btns)

        # Already-fitted bands: toggle each one's curve on the overlay.
        flbl = QLabel("Fitted bands (toggle visibility):")
        flbl.setObjectName("stat_key")
        flbl.setWordWrap(True)
        lay.addWidget(flbl)
        self.xy_area_box = QVBoxLayout()
        self.xy_area_box.setSpacing(3)
        self.xy_area_box.setContentsMargins(0, 0, 0, 0)
        area_holder = QWidget()
        area_holder.setLayout(self.xy_area_box)
        lay.addWidget(area_holder)
        lay.addStretch(1)

        # ── Right: two sub-tabs (spectra / profile). Tiling and tile
        # selection happen on the main window's X-Y map, not here. ─────
        self.xy_inner = QTabWidget()
        self._fit_tabbar(self.xy_inner)
        outer.addWidget(self.xy_inner, 1)
        (self.xy_spec_fig, self.xy_spec_canvas, self.xy_spec_toolbar,
         self.xy_spec_ax) = self._xy_canvas_tab("Tile spectra")
        (self.xy_area_fig, self.xy_area_canvas, self.xy_area_toolbar,
         _) = self._xy_canvas_tab("Area vs X / Y", with_axes=False)
        return tab

    def _xy_canvas_tab(self, title, with_axes=True):
        """Add a titled sub-tab with a Matplotlib canvas + toolbar to the X-Y
        inner tab widget; return ``(fig, canvas, toolbar, ax)``."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        fig = Figure(facecolor=API_PLOT_BG)
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(300)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavToolbar(canvas, self)
        toolbar.setObjectName("plot_toolbar")
        toolbar.setIconSize(QSize(22, 22))
        T.recolor_toolbar_icons(toolbar, T.TEXT_PRIMARY)
        v.addWidget(toolbar)
        v.addWidget(canvas, 1)
        self.xy_inner.addTab(w, title)
        ax = None
        if with_axes:
            ax = fig.add_subplot(111)
            ax.set_facecolor(API_PLOT_BG)
        return fig, canvas, toolbar, ax

    def refresh_xy_bands(self):
        """Rebuild the X-Y band rows and Band picker from the controller list."""
        cmb = self.xy_cmb_band
        prev = cmb.currentText()
        cmb.blockSignals(True)
        cmb.clear()
        for b in self.c._xy_bands:
            cmb.addItem(f"{b['label']}  [{b['emin']:g}–{b['emax']:g}]")
        idx = cmb.findText(prev)
        if idx >= 0:
            cmb.setCurrentIndex(idx)
        cmb.blockSignals(False)
        box = self.xy_band_box
        while box.count():
            item = box.takeAt(0)
            wdg = item.widget()
            if wdg is not None:
                wdg.deleteLater()
        if not self.c._xy_bands:
            empty = QLabel("No bands yet.")
            empty.setObjectName("stat_key")
            box.addWidget(empty)
            return
        for b in self.c._xy_bands:
            row_w = QWidget()
            rl = QHBoxLayout(row_w)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(6)
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{b['color']}; font-size:15px;")
            name = QLabel(f"{b['label']}  [{b['emin']:g}–{b['emax']:g}]")
            name.setStyleSheet(f"color:{T.TEXT_PRIMARY}; font-size:12px;")
            btn = QPushButton("✕")
            btn.setObjectName("mini_btn")
            btn.setFixedWidth(26)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip("Remove this band")
            btn.clicked.connect(lambda _=False, bb=b: self.c._xy_remove_band(bb))
            rl.addWidget(dot)
            rl.addWidget(name, 1)
            rl.addWidget(btn, 0)
            box.addWidget(row_w)

    def refresh_xy_area_list(self):
        """Rebuild the fitted-band visibility toggles from the controller's
        area results."""
        box = self.xy_area_box
        while box.count():
            item = box.takeAt(0)
            wdg = item.widget()
            if wdg is not None:
                wdg.deleteLater()
        results = self.c._xy_area_results
        if not results:
            empty = QLabel("No fitted bands yet.")
            empty.setObjectName("stat_key")
            box.addWidget(empty)
            return
        axis_hint = {"tile": "X & Y", "x": "X", "y": "Y"}
        for r in results:
            row_w = QWidget()
            rl = QHBoxLayout(row_w)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(6)
            cb = QCheckBox()
            cb.setChecked(r.get("visible", True))
            cb.setCursor(Qt.PointingHandCursor)
            cb.setToolTip("Show this band's curve on the overlay")
            cb.toggled.connect(
                lambda ch, rr=r: self.c._xy_toggle_area(rr, ch))
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{r.get('color', T.TEXT_PRIMARY)}; "
                              "font-size:15px;")
            hint = axis_hint.get(r.get("mode", "tile"), "")
            name = QLabel(f"{r.get('label', '')}  ({hint})")
            name.setStyleSheet(f"color:{T.TEXT_PRIMARY}; font-size:12px;")
            rl.addWidget(cb)
            rl.addWidget(dot)
            rl.addWidget(name, 1)
            box.addWidget(row_w)

    def set_xy_band_armed(self, armed):
        """Reflect the armed/disarmed state on the X-Y Add-band button."""
        b = self.xy_btn_add_band
        if armed:
            b.setText("Drag band on spectra...")
            b.setStyleSheet(
                f"background-color:{T.ACCENT_AMBER}; color:{T.BG_DARK}; "
                f"border:2px solid {T.ACCENT_AMBER}; border-radius:5px; "
                f"padding:8px 13px; font-weight:800;")
        else:
            b.setText("Add band")
            b.setStyleSheet("")

    # ── public API (called by controller) ─────────────────────────────────────
    def refresh_list(self, select_latest=False):
        """Rebuild the selection rows from the controller's current list."""
        self._refresh_sel_combo(select_latest=select_latest)
        self.refresh_ratio_combo()
        self.refresh_xy_bands()
        self.refresh_xy_area_list()
        box = self.sel_box
        while box.count():
            item = box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if not self.c.selections:
            empty = QLabel("No selections yet.")
            empty.setObjectName("stat_key")
            box.addWidget(empty)
            return
        for sel in self.c.selections:
            row_w = QWidget()
            rl = QHBoxLayout(row_w)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(6)
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{sel['color']}; font-size:15px;")
            name = QLabel(f"{sel['label']}  [{sel['emin']:g}–{sel['emax']:g}]")
            name.setStyleSheet(f"color:{T.TEXT_PRIMARY}; font-size:12px;")
            btn = QPushButton("✕")
            btn.setObjectName("mini_btn")
            btn.setFixedWidth(26)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip("Remove this selection")
            btn.clicked.connect(lambda _=False, s=sel: self.c._remove_selection(s))
            rl.addWidget(dot)
            rl.addWidget(name, 1)
            rl.addWidget(btn, 0)
            box.addWidget(row_w)

    def set_add_armed(self, armed):
        """Reflect the armed/disarmed state on the Add button."""
        b = self.btn_add
        if armed:
            b.setText("Drag energy band...")
            b.setStyleSheet(
                f"background-color:{T.ACCENT_AMBER}; color:{T.BG_DARK}; "
                f"border:2px solid {T.ACCENT_AMBER}; border-radius:5px; "
                f"padding:8px 13px; font-weight:800;")
        else:
            b.setText("Add selection")
            b.setStyleSheet("")

    # ── internal ──────────────────────────────────────────────────────────────
    def _refresh_sel_combo(self, select_latest=False):
        """Keep the selection picker in sync with the controller's list,
        preserving the current choice by label where possible."""
        cmb = self.cmb_sel
        prev = cmb.currentText()
        cmb.blockSignals(True)
        cmb.clear()
        for sel in self.c.selections:
            cmb.addItem(f"{sel['label']}  [{sel['emin']:g}–{sel['emax']:g}]")
        if select_latest and cmb.count():
            cmb.setCurrentIndex(cmb.count() - 1)
        else:
            idx = cmb.findText(prev)
            if idx >= 0:
                cmb.setCurrentIndex(idx)
        cmb.blockSignals(False)

    def refresh_ratio_combo(self):
        """Sync the reference picker with the stored vs-dt results; enable it
        only when at least two selections have a curve to ratio."""
        cmb = self.cmb_ratio_ref
        prev = cmb.currentText()
        cmb.blockSignals(True)
        cmb.clear()
        cmb.addItem("(absolute)")
        for label in self.c._slice_results:
            cmb.addItem(label)
        idx = cmb.findText(prev)
        cmb.setCurrentIndex(idx if idx >= 0 else 0)
        cmb.setEnabled(len(self.c._slice_results) >= 2)
        cmb.blockSignals(False)

    def _on_ratio_ref_changed(self, _idx):
        txt = self.cmb_ratio_ref.currentText()
        ref = None if (not txt or txt == "(absolute)") else txt
        self.c._set_slice_ratio_ref(ref)

    def _on_slice_fit(self):
        idx = self.cmb_sel.currentIndex()
        if idx < 0 or idx >= len(self.c.selections):
            self.c._status("Add and select an energy selection first")
            return
        sel = self.c.selections[idx]
        technique = TECHNIQUE_FROM_LABEL.get(self.cmb_tech.currentText(), TECH_FIT)
        dt_slice_txt = self.ed_dt_slice.text().strip()
        dt_slice_w = None   # None → ~10 slices over the dt range
        if dt_slice_txt:
            try:
                dt_slice_w = float(dt_slice_txt)
                if dt_slice_w <= 0:
                    raise ValueError
            except ValueError:
                self.c._status("dt slice width must be a positive number")
                return
        try:
            min_snr = float(self.ed_min_snr.text().strip())
            if min_snr <= 0:
                raise ValueError
        except ValueError:
            self.c._status("Min SNR must be a positive number")
            return
        self.c._open_slice_fits(sel, dt_slice_w, technique, min_snr)

    def show_slice_spectra(self, slices, band, x_label):
        """Render the per-slice energy spectra into the embedded canvas, in the
        currently selected view (overlay or waterfall)."""
        self._spectra_args = (slices, band, x_label)
        self._draw_spectra()

    def _draw_spectra(self):
        """(Re)draw the embedded spectra in the current view mode."""
        if self._spectra_args is None:
            return
        slices, band, x_label = self._spectra_args
        view = self.cmb_spectra_view.currentText()
        if view == "Offset":
            plot_slice_offset(self.spectra_fig, slices, band, x_label)
        elif view == "Waterfall":
            plot_slice_waterfall(self.spectra_fig, slices, band, x_label)
        else:
            plot_slice_spectra(self.spectra_fig, slices, band, x_label)
        self.spectra_canvas.draw_idle()

    def _on_spectra_view_changed(self, _idx):
        self._draw_spectra()


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

        lay.addWidget(hsep())
        self.btn_load = QPushButton("Load API file"); self.btn_load.setObjectName("open_btn")
        self.btn_load.setCursor(Qt.PointingHandCursor)
        self.btn_load.setToolTip(
            "Loads the API file for the selected run.\n"
            "Make sure your data path is set up correctly in data-path.txt.")
        lay.addWidget(self.btn_load)

        self.btn_interactive = QPushButton("Interactive cuts")
        self.btn_interactive.setObjectName("yellow_btn")
        self.btn_interactive.setCheckable(True)
        self.btn_interactive.setCursor(Qt.PointingHandCursor)
        self.btn_interactive.setToolTip(
            "Enable dragging spans/rectangles on the plots to cut events interactively")
        self.btn_undo = QPushButton("← Back"); self.btn_undo.setObjectName("primary_btn")
        self.btn_undo.setCursor(Qt.PointingHandCursor)
        self.btn_undo.setToolTip("Restore the dataframe to the state before the last cut")
        self.btn_undo.setEnabled(False)
        self.btn_reset = QPushButton("Reset"); self.btn_reset.setObjectName("danger_btn")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        qr = QHBoxLayout(); qr.setContentsMargins(0, 0, 0, 0); qr.setSpacing(6)
        qr.addWidget(self.btn_undo, 1)
        qr.addWidget(self.btn_reset, 1)
        qrw = QWidget(); qrw.setLayout(qr)
        lay.addWidget(self.btn_interactive)
        lay.addWidget(qrw)

        # ── Run info readout ─────────────────────────────────────────
        lay.addWidget(hsep()); lay.addWidget(header("RUN INFO"))
        self.lbl_info = QLabel(" --")
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
        vrow, _ = labeled_row("vmax", self.ed_vmax)
        self.ed_vmax.setFixedWidth(70)
        self.ed_vmax.setToolTip("Cap the X-Y color scale at this count (forces "
                                "linear scale)  -- press Enter to apply")
        lay.addWidget(vrow)

        # ── Histogram bins ───────────────────────────────────────────
        # Pre-populated with the controller's defaults (see ApiController).
        self.ed_ebins = QLineEdit(str(DEFAULT_EBINS))
        self.ed_tbins = QLineEdit(str(DEFAULT_TBINS))
        self.ed_xybins = QLineEdit(str(DEFAULT_HEXBINS))
        self.ed_ebins.setToolTip("Number of bins on the energy spectrum")
        self.ed_tbins.setToolTip("Number of bins on the dt (time) histogram")
        self.ed_xybins.setToolTip("Hexbin grid size on the X-Y map")
        for lbl, ed in [("Energy bins", self.ed_ebins),
                        ("dt bins", self.ed_tbins),
                        ("X-Y bins", self.ed_xybins)]:
            ed.setFixedWidth(70)
            ed.setToolTip(f"{ed.toolTip()}  -- press Enter to apply")
            row, _ = labeled_row(lbl, ed)
            lay.addWidget(row)

        self.btn_filters = QPushButton("Filters..."); self.btn_filters.setObjectName("action_btn")
        self.btn_filters.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_filters)

        # ── Combine multiple runs ────────────────────────────────────
        # First step in the pipeline: stitch several runs  -- possibly on different
        # dates  -- into one by concatenating them, then correct drift and
        # calibrate the *combined* run below.
        lay.addWidget(hsep()); lay.addWidget(header("COMBINE RUNS"))
        self.btn_combine = QPushButton("Combine multiple...")
        self.btn_combine.setObjectName("yellow_btn")
        self.btn_combine.setCursor(Qt.PointingHandCursor)
        self.btn_combine.setToolTip(
            "Open the multi-run combine window: visualize and stitch several runs "
            "into one new run.\n"
            "Combines runs from any dates by concatenation; calibration columns are "
            "dropped — re-shift and re-calibrate the combined run afterwards.")
        lay.addWidget(self.btn_combine)

        # ── Drift correction (time + energy shifts) ──────────────────
        # Comes before calibration: drift is corrected on the raw channels first,
        # then the calibration maps the corrected channels to energy.
        lay.addWidget(hsep()); lay.addWidget(header("DRIFT CORRECTION"))
        self.btn_shifts = QPushButton("Shifts...")
        self.btn_shifts.setObjectName("yellow_btn")
        self.btn_shifts.setCursor(Qt.PointingHandCursor)
        self.btn_shifts.setToolTip(
            "Open the energy gain-shift and time-shift correction window.\n"
            "Corrects gain/time drift over the run: split into time segments and "
            "align them, or shift dt by a constant — independently for the "
            "energy and time axes.")
        lay.addWidget(self.btn_shifts)

        # ── Energy calibration ───────────────────────────────────────
        lay.addWidget(hsep()); lay.addWidget(header("ENERGY CALIBRATION"))
        self.btn_retrieve_cal = QPushButton("Retrieve calibration")
        self.btn_retrieve_cal.setObjectName("primary_btn")
        self.btn_retrieve_cal.setCursor(Qt.PointingHandCursor)
        self.btn_retrieve_cal.setToolTip(
            "Apply the Calibration tab's current calibration curve/equation to "
            "the API dataframe so the panels read in energy.\n"
            "Workflow: send the spectrum to the Spectrum tab, build a calibration "
            "on the Calibration tab, then click here to apply it.")
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

        # ── E / dt / X-Y selections ──────────────────────────────────
        lay.addWidget(hsep()); lay.addWidget(header("E/XY SELECTIONS"))
        self.btn_selections = QPushButton("Selections...")
        self.btn_selections.setObjectName("open_btn")
        self.btn_selections.setCursor(Qt.PointingHandCursor)
        self.btn_selections.setToolTip(
            "Manage energy / X-Y selections and run time-slice fits vs dt")
        lay.addWidget(self.btn_selections)
        self.lbl_sel_count = QLabel("No selections")
        self.lbl_sel_count.setObjectName("stat_key")
        lay.addWidget(self.lbl_sel_count)

        # ── Actions ──────────────────────────────────────────────────
        lay.addWidget(hsep()); lay.addWidget(header("ACTIONS"))
        self.btn_send = QPushButton(SEND_DEFAULT_TEXT)
        self.btn_send.setObjectName("open_btn"); self.btn_send.setCursor(Qt.PointingHandCursor)
        self.btn_send.setToolTip("Hand the current energy histogram to the Spectrum tab")
        lay.addWidget(self.btn_send)
        self.btn_3d = QPushButton("3D view..."); self.btn_3d.setObjectName("find_btn")
        self.btn_3d.setCursor(Qt.PointingHandCursor)
        self.btn_3d.setToolTip("Plotly volume render of the reconstructed X-Y-Z hit cloud")
        lay.addWidget(self.btn_3d)
        self.btn_apply_data = QPushButton("Apply to data")
        self.btn_apply_data.setObjectName("yellow_btn")
        self.btn_apply_data.setCursor(Qt.PointingHandCursor)
        self.btn_apply_data.setToolTip(
            "Combine all of the source run's parquet files, bake the active "
            "energy calibration and/or time shift into energy_cal / dt_cal, and "
            "save it as a new run (you choose the run number)")
        lay.addWidget(self.btn_apply_data)

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
        self._undo_state = None     # snapshot: (df_current, df_previous, en_flag, dt_flag, xy_flag)
        # Source run identity (set on load); used by "Apply to data".
        self._src_date = self._src_runnr = self._src_ch = None
        self._src_data_path = None
        # Filter "used before" flags, mirroring the legacy previous/current scheme.
        self.en_flag = self.dt_flag = self.xy_flag = 0
        self.xkey = self.ykey = self.ekey = None
        self.xyplane = (-0.9, 0.9, -0.9, 0.9)
        self.erange = [0, 1]
        self.ebins = DEFAULT_EBINS
        self.tbins = DEFAULT_TBINS
        self.hexbins = DEFAULT_HEXBINS
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
        # Persistent markers for the last applied interactive cut, kept on the
        # panels after the span/rectangle selector is removed so the active cut
        # stays visible. (emin, emax) / (tmin, tmax) / (xlo, xhi, ylo, yhi), or
        # None when no cut of that kind is active.
        self._cut_energy = None
        self._cut_time = None
        self._cut_xy = None
        self._cut_artists = []      # matplotlib artists for the markers above
        # Energy selections: list of dict(label, color, emin, emax, df). Each
        # snapshots the energy-cut dataframe at creation time.
        self.selections = []
        self._arming_selection = False
        self._arm_temp_selector = False
        self._sel_color_idx = 0
        self._sel_dlg = None        # SelectionsDialog (created lazily)
        self._sig_active = False    # dt-panel twin overlay currently shown
        self._ax_sig = None         # the twinx axes carrying the vs-dt curves
        # Time-slice fits vs dt: stored per-selection results (label -> payload)
        # and the open slice-fit windows.
        self._slice_results = {}
        self._slice_fit_wins = set()
        # When set to a selection label, the dt overlay shows every other
        # selection normalized by this reference (e.g. Mg/Si); None ⇒ absolute.
        self._slice_ratio_ref = None
        self._selections_plotted = False   # selection band overlays on screen
        # X-Y tile selections (see xyprofile): a snapshot of the working frame
        # taken on "Build tiles", the uniform tile grid laid over it, the
        # per-tile energy spectra, the set of clicked (col, row) tiles, the
        # energy bands dragged on the overlay, and the open per-tile fit windows.
        self._xy_df = None
        self._xy_keys = None        # dict(xkey, ykey, ekey, ebins, erange, ...)
        self._xy_plane = None       # (xlo, xhi, ylo, yhi) at snapshot time
        self._xy_tiles = None       # xyp.TileGrid
        self._xy_grid_centers = None
        self._xy_grid_counts = None
        self._xy_selected = set()   # selected (col, row) tile indices
        self._xy_overlay_artists = []  # grid/selection artists on the X-Y map
        self._xy_bands = []         # list of dict(label, color, emin, emax)
        self._xy_band_color_idx = 0
        self._xy_arming_band = False
        self._xy_band_selector = None
        self._xy_fit_wins = set()
        # Fitted area-vs-position results, one per band label, each a
        # TileFitWindow payload plus a 'visible' flag so already-fitted bands
        # can be kept on the overlay and toggled.
        self._xy_area_results = []
        # Energy calibration: the channel column the histogram is binned from
        # (set in _configure_keys), the polynomial coeffs retrieved from the
        # Calibration tab, and the energy units once calibrated (None ⇒ raw
        # channels).
        self._chan_key = None
        self._chan_base = None      # pristine raw-channel column for the data type
        # Whether the detected channel base is already in physical units: "MeV"
        # when the data carries an "energy" axis (calibrated/simulated), None when
        # it carries raw "energy_orig" channels. Drives the send-to-Spectrum units.
        self._native_units = None
        self._cal_coeffs = None
        self.e_units = None
        # Gain-shift drift correction (Shifts... window): when applied, the
        # corrected channels live on the df_api "energy_drift" column and
        # _chan_key points at it (so it layers under the polynomial calibration).
        self._egain_applied = False
        self._egain_label = ""
        # Time correction (Shifts... window): a segment alignment and/or a
        # constant offset, composed together. The segment alignment is written
        # to ``dt_aligned``; the final ``dt_cal`` is ``dt_aligned`` (or raw
        # ``dt`` when no alignment) plus the constant. _dt_shift holds the
        # constant (float) when applied; _dt_segments_applied marks the
        # alignment; _dt_seg_desc is its descriptive text. Either makes _dt_key
        # switch to "dt_cal".
        self._dt_shift = None
        self._dt_segments_applied = False
        self._dt_seg_desc = ""
        self._dt_label = "No time shift"
        self._dt_key = "dt"
        self._shifts_dlg = None
        self._combine_dlg = None
        self._wire()
        # Tile selection happens on the main X-Y panel: a click toggles a tile
        # while the Selections dialog's tiling is active (see _xy_on_map_click).
        self.page.canvas.mpl_connect("button_press_event", self._xy_on_map_click)
        self._refresh_sel_list()

    @property
    def _dt_corrected(self):
        """True when any time correction (constant or segment) is active."""
        return self._dt_shift is not None or self._dt_segments_applied

    def _wire(self):
        o = self.opts
        o.btn_load.clicked.connect(self._load)
        o.btn_reset.clicked.connect(self._reset)
        o.btn_undo.clicked.connect(self._undo)
        o.btn_send.clicked.connect(self._send_to_spectrum)
        o.btn_3d.clicked.connect(self._open_3d)
        o.btn_apply_data.clicked.connect(self._apply_to_data)
        o.btn_interactive.toggled.connect(self._toggle_interactive)
        o.btn_filters.clicked.connect(self._open_filters)
        o.btn_selections.clicked.connect(self._open_selections)
        o.btn_retrieve_cal.clicked.connect(self._retrieve_calibration)
        o.btn_clear_cal.clicked.connect(self._clear_calibration)
        o.btn_shifts.clicked.connect(self._open_shifts)
        o.btn_combine.clicked.connect(self._open_combine)
        o.cb_spe_log.toggled.connect(self._toggle_spe_log)
        o.cb_xy_log.toggled.connect(lambda *_: self._replot_xy())
        o.ed_vmax.returnPressed.connect(self._apply_vmax)
        for ed in (o.ed_ebins, o.ed_tbins, o.ed_xybins):
            ed.returnPressed.connect(self._apply_bins)

    def _save_undo_snapshot(self):
        """Capture the full filter state before a cut so ← Back can restore it."""
        self._undo_state = (
            self.df_current.copy(),
            self.df_previous.copy(),
            self.en_flag, self.dt_flag, self.xy_flag,
            self._cut_energy, self._cut_time, self._cut_xy,
        )
        self.opts.btn_undo.setEnabled(True)

    def _undo(self):
        """Restore the dataframe to the state before the last interactive cut."""
        if self._undo_state is None:
            return
        self._clear_significance()
        (df_cur, df_prev, en_flag, dt_flag, xy_flag,
         cut_e, cut_t, cut_xy) = self._undo_state
        self.df_current = df_cur
        self.df_previous = df_prev
        self.en_flag, self.dt_flag, self.xy_flag = en_flag, dt_flag, xy_flag
        self._cut_energy, self._cut_time, self._cut_xy = cut_e, cut_t, cut_xy
        self._undo_state = None
        self.opts.btn_undo.setEnabled(False)
        if self.page.ax_spe is not None:
            self.page.ax_spe.clear()
            self._plot_energy(self.df_current)
        if self.page.ax_dt is not None:
            self.page.ax_dt.clear()
            self._plot_time(self.df_current)
        self._replot_xy()
        self.page.reset_nav()
        self.page.canvas.draw_idle()
        self._status(f"Restored previous state  ·  {self.df_current.shape[0]:,} events")

    def _toggle_interactive(self, checked):
        b = self.opts.btn_interactive
        if checked:
            b.setStyleSheet(
                f"background-color:{T.ACCENT_AMBER}; color:{T.BG_DARK}; "
                f"border:2px solid {T.ACCENT_AMBER}; border-radius:5px; "
                f"padding:8px 13px; font-size:14px; font-weight:800;")
            self._attach_selectors()
            self._status("Interactive cuts enabled  -- drag on any panel to filter")
        else:
            b.setStyleSheet("")
            self._detach_selectors()
            # Force a redraw so the (now-hidden) span/rectangle selectors
            # actually disappear from the panels instead of lingering.
            self.page.canvas.draw_idle()
            self._status("Interactive cuts disabled")

    def _status(self, msg):
        self.app.statusBar().showMessage(f"  {msg}")

    # -- loading ---------------------------------------------------------------
    def _load(self):
        # Brighten the Load button so the click registers before the (blocking)
        # file read freezes the UI  -- purple to match its open_btn identity.
        self._flash_button(self.opts.btn_load)
        try:
            self._load_file()
        except Exception as exc:  # noqa: BLE001  -- surface load errors to the user
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
        data_path = None  # resolved from data-path.txt

        # Energy binning: the high-resolution channels use more bins. Reflect the
        # chosen value in the Display box so "Apply bins" starts from the truth.
        self.ebins = 2 ** 14 if ch_txt in ("6", "7", "10", "11") else DEFAULT_EBINS
        self.opts.ed_ebins.setText(str(self.ebins))

        self.flood_field = ch_txt == "9"
        if self.flood_field:
            ch = 9
        else:
            try:
                ch = int(ch_txt)
            except ValueError:
                self._status("Channel must be an integer")
                return

        self._status(f"Loading run {date}-{runnr} ch {ch}...")
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
            # Time priority: dt_cal is already in ns; raw dt is in seconds and
            # needs converting. dt_cal (when present) becomes the time axis in
            # _configure_keys.
            df["dt"] *= 1e9  # s → ns

        self.df_api = df
        self.df_current = df.copy()
        self.df_previous = df.copy()
        # Remember the source run so "Apply to data" can re-read every channel
        # and write the calibrated result to a new run.
        self._src_date = date
        self._src_runnr = runnr
        self._src_ch = ch
        self._src_data_path = data_path
        self.en_flag = self.dt_flag = self.xy_flag = 0
        self._cut_energy = self._cut_time = self._cut_xy = None
        self._cut_artists = []
        self._undo_state = None
        self.opts.btn_undo.setEnabled(False)
        # A new run invalidates selections snapshotted from the old data.
        self._reset_selections()
        # ...and any calibration we applied. Energy priority: a file-provided
        # energy_cal column is already a physical-energy axis (assume MeV); a
        # GUI calibration would re-populate this later.
        self._cal_coeffs = None
        if "energy_cal" in df.columns:
            self.e_units = "keV"
            self.opts.lbl_cal.setText("Calibrated → energy (keV)")
            self.opts.btn_clear_cal.setEnabled(True)
        else:
            self.e_units = None
            self.opts.lbl_cal.setText("Uncalibrated (channels)")
            self.opts.btn_clear_cal.setEnabled(False)
        # ...and any drift corrections: the fresh dataframe has no derived columns.
        self._dt_shift = None
        self._dt_segments_applied = False
        self._dt_seg_desc = ""
        self._dt_label = "No time shift"
        self._dt_key = "dt"
        self._egain_applied = False
        self._egain_label = ""
        self._refresh_shift_labels()

        self._configure_keys()
        self._initialize_plots()
        self._load_settings(date, runnr, ch, data_path)
        n = self.df_current.shape[0]
        self._status(f"Loaded run {date}-{runnr} ch {ch}  ·  {n:,} events")

    def _configure_keys(self):
        """Auto-detect the X/Y/energy/time column keys by priority and set the
        plot ranges. No explicit data-type choice: the columns present in the
        dataframe decide everything.

        Priorities:
          position : (X2, Y2)  →  (X, Y)
          energy   : energy_cal  →  energy_orig (raw channels)  →  energy (MeV)
          time     : dt_cal (already ns)  →  dt
        The channel/energy range is always ``[0, max]``.

        Channel axis layering (low → high): ``_chan_base`` is the pristine raw
        channel column ("energy_orig", else "energy"). When a gain-shift drift
        correction is applied, ``_chan_key`` switches to the corrected
        ``energy_drift`` column so the histogram, send-to-spectrum and the
        polynomial calibration all read the drift-corrected channels. ``ekey``
        is then ``energy_cal`` once a calibration is present, else ``_chan_key``.
        """
        # Position: reconstruct X2/Y2 from the A/B/C/D quadrants when they are
        # available, then prefer (X2, Y2) over (X, Y).
        if ("X2" not in self.df_current.columns
                and {"A", "B", "C", "D"}.issubset(self.df_current.columns)):
            self.df_current = apicalc.calc_own_pos(self.df_current)
        df = self.df_current
        if {"X2", "Y2"}.issubset(df.columns):
            self.xkey, self.ykey = "X2", "Y2"
        else:
            self.xkey, self.ykey = "X", "Y"

        # Channel base: raw channels (energy_orig) take priority over an already
        # physical energy axis (energy). _native_units flags which it is.
        if "energy_orig" in df.columns:
            self._chan_base, self._native_units = "energy_orig", None
        elif "energy" in df.columns:
            self._chan_base, self._native_units = "energy", "MeV"
        else:
            self._chan_base, self._native_units = "energy", None
        chan_range = [0.0, float(df[self._chan_base].max())]

        # Gain-shift drift correction layers under everything: the corrected
        # channels become the working channel axis when present.
        if self._egain_applied and "energy_drift" in df.columns:
            self._chan_key = "energy_drift"
        else:
            self._chan_key = self._chan_base
        # Fit the X-Y extent to where the hits actually land instead of the
        # detector's full physical plane. Most runs illuminate only a central
        # patch, so the fixed plane left the map floating in empty space (and the
        # cursor reporting coordinates out there). A generic plane is the
        # fallback for empty/degenerate data.
        self.xyplane = self._fit_xyplane(df, self.xkey, self.ykey,
                                         fallback=(-0.9, 0.9, -0.9, 0.9))
        # Energy axis: a calibration (energy_cal) wins over the raw channel axis.
        if self.e_units is not None and "energy_cal" in df.columns:
            self.ekey = "energy_cal"
            self.erange = [0.0, float(df["energy_cal"].max())]
        else:
            self.ekey = self._chan_key
            self.erange = chan_range
        # Time axis: a calibrated time column (already in ns) wins over raw dt.
        # Covers both a file-provided dt_cal and one written by a GUI time
        # correction (constant shift or segment alignment).
        if "dt_cal" in df.columns:
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
        # Axes are about to be rebuilt  -- clear secondary-axis state so the
        # twinx references don't keep a stale handle to the old axes.
        self._sig_active = False
        self._ax_sig = None
        self.page.build_axes(flood_field=self.flood_field)
        self._detach_selectors()
        if self.flood_field:
            self._replot_xy()
        else:
            self._plot_energy(self.df_current)
            self._plot_time(self.df_current)
            self._replot_xy()
            if self.opts.btn_interactive.isChecked():
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
        except Exception:  # noqa: BLE001  -- settings are best-effort
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
            self.opts.lbl_info.setText(" --")
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
        energy panel having been drawn first  -- otherwise the first click can
        find self.gam still None (no draw yet) and silently do nothing.
        """
        self.gam, edg = np.histogram(df[self.ekey], bins=self.ebins, range=self.erange)
        self.gam_x = (edg[1:] + edg[:-1]) / 2

    def _plot_energy(self, df):
        # Always (re)compute the histogram, even if the axis isn't ready yet, so
        # Send has data to work with regardless of draw timing.
        self._compute_energy_hist(df)
        ax = self.page.ax_spe
        if ax is None:
            return
        ax.plot(self.gam_x, self.gam, color=T.LOGO_GREEN, linewidth=0.9)
        ax.set_yscale("log" if self.opts.cb_spe_log.isChecked() else "linear")
        ax.set_xlabel(self._energy_xlabel())
        ax.set_ylabel("Counts")
        self.page._style(ax)
        self._draw_cut_markers()

    def _axis_units(self):
        """Units of the energy axis *as currently shown*, or None for a channel
        axis.  Single source of truth shared by the panel x-label, the per-slice
        spectra and send-to-spectrum so they can never disagree.

        For API dataframes only an applied calibration (the ``energy_cal``
        column, which sets ``e_units``) counts as a real energy axis; every
        other column — ``energy``, ``energy_orig``, the drift-corrected
        ``energy_drift`` — is treated and labelled as raw channels."""
        if self.e_units and self.ekey == "energy_cal":
            return self.e_units
        return None

    def _energy_xlabel(self):
        """Energy-panel x-label, derived from :meth:`_axis_units`."""
        units = self._axis_units()
        return f"Energy ({units})" if units else "Channels"

    def _plot_time(self, df):
        ax = self.page.ax_dt
        if ax is None or df.shape[0] == 0:
            return
        self._clear_dt_twin()
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
            ax.legend(loc="upper right", fontsize=11, facecolor=API_PLOT_BG,
                      edgecolor=T.BORDER, labelcolor=T.TEXT_PRIMARY)
        self.page._style(ax)
        self._draw_cut_markers()

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
            self._draw_cut_markers()
            self._xy_reattach_overlay()
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
        self._draw_cut_markers()
        self._xy_reattach_overlay()
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

    def _draw_cut_markers(self):
        """Re-draw the persistent markers for the last applied interactive cut:
        red dotted vertical lines bounding the energy / dt cuts and a red dashed
        rectangle around the X-Y cut.  Called at the end of every panel redraw so
        the markers survive across replots until a new cut or a reset.

        Any previously drawn marker artists are removed first so repeated cuts
        (which don't always clear their panel) don't accumulate stale lines."""
        for art in self._cut_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass  # already gone with an ax.clear()
        self._cut_artists = []
        if self._cut_energy is not None and self.page.ax_spe is not None:
            for x in self._cut_energy:
                self._cut_artists.append(self.page.ax_spe.axvline(
                    x, color=T.ACCENT_RED, linestyle=":", linewidth=1.3, zorder=6))
        if self._cut_time is not None and self.page.ax_dt is not None:
            for x in self._cut_time:
                self._cut_artists.append(self.page.ax_dt.axvline(
                    x, color=T.ACCENT_RED, linestyle=":", linewidth=1.3, zorder=6))
        if self._cut_xy is not None and self.page.ax_xy is not None:
            xlo, xhi, ylo, yhi = self._cut_xy
            rect = Rectangle(
                (xlo, ylo), xhi - xlo, yhi - ylo, fill=False,
                edgecolor=T.ACCENT_RED, linestyle="--", linewidth=1.3, zorder=6)
            self.page.ax_xy.add_patch(rect)
            self._cut_artists.append(rect)

    def _on_energy_span(self, xmin, xmax):
        if xmax <= xmin:
            return
        if self._arming_selection:
            self._create_selection(round(xmin, 4), round(xmax, 4))
        elif self.opts.btn_interactive.isChecked():
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
        self._clear_significance()
        self._save_undo_snapshot()
        self._cut_energy = (xmin, xmax)
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
        self._clear_significance()
        self._save_undo_snapshot()
        self._cut_time = (tmin, tmax)
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
        self._clear_significance()
        self._save_undo_snapshot()
        xlo, xhi = sorted((x1, x2))
        ylo, yhi = sorted((y1, y2))
        self._cut_xy = (xlo, xhi, ylo, yhi)
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
        if self._arming_selection:
            self._arming_selection = False
            self._set_add_armed(False)
            if self._arm_temp_selector:
                self._arm_temp_selector = False
                self._detach_selectors()
            self._status("Selection arming cancelled")
            return
        if self.df_current is None or self.page.ax_spe is None:
            self._status("Load an API file with an energy panel first")
            return
        if not self.opts.btn_interactive.isChecked():
            self._detach_selectors()
            self._selectors.append(SpanSelector(
                self.page.ax_spe, self._on_energy_span, "horizontal",
                useblit=True, interactive=True,
                props=dict(alpha=0.3, facecolor=T.ACCENT_AMBER)))
            self._arm_temp_selector = True
        self._arming_selection = True
        self._set_add_armed(True)
        self._status("Drag a band on the energy spectrum to add a selection...")

    def _set_add_armed(self, armed):
        """Reflect the armed/disarmed state on the dialog's Add button (if open)."""
        if self._sel_dlg is not None:
            self._sel_dlg.set_add_armed(armed)

    def _create_selection(self, emin, emax):
        self._arming_selection = False
        self._set_add_armed(False)
        if self._arm_temp_selector:
            self._arm_temp_selector = False
            self._detach_selectors()
        if self.df_current is None:
            return
        mask = (self.df_current[self.ekey] > emin) & (self.df_current[self.ekey] < emax)
        sub = self.df_current[mask].copy()
        if sub.shape[0] == 0:
            self._status(f"No events in [{emin:g}, {emax:g}]  -- selection not added")
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
        self._refresh_sel_list(select_latest=True)
        self._plot_energy_overlays()
        self.page.canvas.draw_idle()
        self._status(f"Added '{label}'  ·  {sub.shape[0]:,} events "
                     f"[{emin:g}, {emax:g}]")

    def _refresh_sel_list(self, select_latest=False):
        """Update the count label in the options panel and the dialog list if open."""
        n = len(self.selections)
        self.opts.lbl_sel_count.setText(
            f"{n} selection{'s' if n != 1 else ''}" if n else "No selections")
        if self._sel_dlg is not None:
            self._sel_dlg.refresh_list(select_latest=select_latest)

    def _remove_selection(self, sel):
        try:
            self.selections.remove(sel)
        except ValueError:
            return
        # Forget this selection's stored time-slice result so its curve doesn't
        # linger on the dt overlay after the selection is gone.
        self._slice_results.pop(sel["label"], None)
        if self._slice_ratio_ref == sel["label"]:
            self._slice_ratio_ref = None       # the reference itself is gone
        if self._sel_dlg is not None:
            self._sel_dlg.refresh_ratio_combo()
        self._refresh_sel_list()
        self._redraw_after_removal()
        self._status(f"Removed selection '{sel['label']}'")

    def _redraw_after_removal(self):
        """Refresh the panels after a selection is removed: its overlays
        disappear, while any remaining selections / vs-dt curves stay."""
        if self.df_current is None:
            return
        # Energy: always reflect the current selections list.
        if self.selections:
            self._plot_energy_overlays()
        else:
            if self.page.ax_spe is not None:
                self.page.ax_spe.clear()
                self._plot_energy(self.df_current)
        # X-Y overlay only if "Plot selections" is active.
        if self._selections_plotted:
            if self.selections:
                self._plot_xy_overlays()
            else:
                self._selections_plotted = False
                self._replot_xy()
        # dt panel: the slice-fit vs-dt overlay wins; else the selection dt
        # overlay (if plotted); else the plain histogram (drawn by the empty
        # overlay path).
        if self._slice_results or not (self._selections_plotted and self.selections):
            self._draw_slice_overlay()
        else:
            self._plot_time_overlays()
        self.page.reset_nav()
        self.page.canvas.draw_idle()

    def _open_selections(self):
        """Open (or raise) the SelectionsDialog.

        Interactive cuts are turned off first so its span/rectangle selectors
        (and any leftover span on the dt spectrum) don't linger while the user
        works with selections."""
        if self.opts.btn_interactive.isChecked():
            # Unchecking fires _toggle_interactive(False), which detaches the
            # selectors and redraws the panels.
            self.opts.btn_interactive.setChecked(False)
        else:
            self._detach_selectors()
            self.page.canvas.draw_idle()
        if self._sel_dlg is None:
            self._sel_dlg = SelectionsDialog(self)
            self._sel_dlg.refresh_list()
            # Drop the tile overlay from the main X-Y map when the dialog closes.
            self._sel_dlg.finished.connect(
                lambda *_: self._xy_clear_map_overlay())
        self._xy_redraw_area()
        self._sel_dlg.show()
        self._sel_dlg.raise_()
        self._sel_dlg.activateWindow()
        # Lay the tile overlay back over the main X-Y map (if tiles are built).
        self._xy_draw_map_overlay()

    def _clear_selections(self):
        if not self.selections:
            return
        self.selections = []
        # Drop the stored time-slice results too, otherwise the cleared
        # selections' curves reappear the next time the dt overlay is redrawn.
        self._slice_results.clear()
        self._slice_ratio_ref = None
        self._selections_plotted = False
        if self._sel_dlg is not None:
            self._sel_dlg.refresh_ratio_combo()
        self._clear_significance()
        self._refresh_sel_list()
        if self.df_current is not None and self.page.ax_spe is not None:
            self.page.ax_spe.clear()
            self._plot_energy(self.df_current)
            self.page.canvas.draw_idle()
        self._status("Cleared all selections")

    # -- X-Y tile selections ---------------------------------------------------
    def _xy_overlay_active(self):
        """True when the tile grid should be shown on the main X-Y panel: tiles
        are built and the Selections dialog is open."""
        return (self._xy_tiles is not None and self._sel_dlg is not None
                and self._sel_dlg.isVisible())

    def _xy_draw_map_overlay(self):
        """(Re)draw the tile grid + selected-tile overlay on the *main* X-Y
        panel, leaving the hexbin underneath untouched so clicks stay snappy."""
        ax = self.page.ax_xy
        if ax is None:
            return
        for art in self._xy_overlay_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        self._xy_overlay_artists = []
        if self._xy_overlay_active():
            self._xy_overlay_artists = xyp.draw_grid_overlay(
                ax, self._xy_tiles, self._xy_selected)
        self.page.canvas.draw_idle()

    def _xy_reattach_overlay(self):
        """Re-add the tile overlay after the main X-Y map redraws (its ``clear``
        dropped the artists). No canvas draw — the caller's redraw handles it."""
        self._xy_overlay_artists = []
        if self._xy_overlay_active() and self.page.ax_xy is not None:
            self._xy_overlay_artists = xyp.draw_grid_overlay(
                self.page.ax_xy, self._xy_tiles, self._xy_selected)

    def _xy_clear_map_overlay(self):
        """Remove the tile overlay from the main X-Y panel (dialog closed)."""
        if not self._xy_overlay_artists:
            return
        for art in self._xy_overlay_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        self._xy_overlay_artists = []
        if self.page.ax_xy is not None:
            self.page.canvas.draw_idle()

    def _xy_build_tiles(self):
        """Snapshot the current frame, tile the plane and build per-tile spectra."""
        dlg = self._sel_dlg
        if dlg is None:
            return
        if self.df_current is None or self.df_current.shape[0] == 0:
            self._status("Load an API file with events first")
            return
        if self.ekey is None or self.xkey is None:
            self._status("This run has no X-Y / energy columns to tile")
            return
        if self.page.ax_xy is None:
            self._status("The X-Y map isn't available for this run")
            return
        try:
            tw = float(dlg.xy_ed_w.text().strip())
            th = float(dlg.xy_ed_h.text().strip())
        except ValueError:
            self._status("Tile width and length must be numbers")
            return
        self._xy_df = self.df_current.copy()
        self._xy_plane = tuple(self.xyplane)
        units = self._axis_units()
        self._xy_keys = dict(
            xkey=self.xkey, ykey=self.ykey, ekey=self.ekey,
            ebins=self.ebins, erange=tuple(self.erange), e_units=units)
        try:
            tiles = xyp.make_tiles(self._xy_plane, tw, th)
        except ValueError as exc:
            self._status(str(exc))
            return
        k = self._xy_keys
        centers, counts = xyp.tile_spectra(
            self._xy_df, k["xkey"], k["ykey"], k["ekey"], tiles,
            k["ebins"], k["erange"])
        self._xy_tiles = tiles
        self._xy_grid_centers, self._xy_grid_counts = centers, counts
        self._xy_selected = set()
        dlg.xy_lbl_grid.setText(f"{tiles.nx}×{tiles.ny} = {tiles.n_tiles} tiles")
        self._xy_draw_map_overlay()
        self._xy_redraw_overlay()
        self._status(f"Built {tiles.nx}×{tiles.ny} tiles  ·  "
                     "click tiles on the X-Y map (main window) to select them")

    def _xy_on_map_click(self, event):
        """Toggle the clicked tile on the main X-Y panel into/out of the set."""
        if not self._xy_overlay_active():
            return
        if self.page.toolbar.mode:            # zoom/pan active
            return
        if event.inaxes is not self.page.ax_xy or event.xdata is None:
            return
        col, row, inside = self._xy_tiles.indices(
            np.array([event.xdata]), np.array([event.ydata]))
        if not inside[0]:
            return
        tile = (int(col[0]), int(row[0]))
        if tile in self._xy_selected:
            self._xy_selected.discard(tile)
        else:
            self._xy_selected.add(tile)
        self._xy_draw_map_overlay()
        self._xy_redraw_overlay()

    def _xy_select_all(self):
        if self._xy_tiles is None:
            self._status("Build tiles first")
            return
        t = self._xy_tiles
        self._xy_selected = {(c, r) for c in range(t.nx) for r in range(t.ny)}
        self._xy_draw_map_overlay()
        self._xy_redraw_overlay()

    def _xy_clear_tiles(self):
        if self._xy_tiles is None:
            return
        self._xy_selected = set()
        self._xy_draw_map_overlay()
        self._xy_redraw_overlay()

    # -- tile spectra + energy bands -------------------------------------------
    def _xy_redraw_overlay(self):
        """Redraw the selected tiles' overlaid spectra and re-arm the band
        SpanSelector on the (cleared) axes."""
        dlg = self._sel_dlg
        if dlg is None or self._xy_tiles is None:
            return
        xyp.plot_tile_overlay(
            dlg.xy_spec_ax, self._xy_grid_centers, self._xy_grid_counts,
            self._xy_selected, self._xy_tiles, self._xy_bands,
            self._energy_xlabel())
        self._xy_attach_band_selector()
        dlg.xy_spec_canvas.draw_idle()

    def _xy_attach_band_selector(self):
        dlg = self._sel_dlg
        if dlg is None:
            return
        if self._xy_band_selector is not None:
            try:
                self._xy_band_selector.disconnect_events()
            except Exception:  # noqa: BLE001
                pass
        self._xy_band_selector = SpanSelector(
            dlg.xy_spec_ax, self._xy_on_band_span, "horizontal",
            useblit=True, interactive=True,
            props=dict(alpha=0.3, facecolor=T.ACCENT_AMBER))

    def _xy_arm_band(self):
        dlg = self._sel_dlg
        if dlg is None:
            return
        if self._xy_arming_band:
            self._xy_arming_band = False
            dlg.set_xy_band_armed(False)
            self._status("Band arming cancelled")
            return
        if self._xy_tiles is None or not self._xy_selected:
            self._status("Build tiles and select at least one tile first")
            return
        self._xy_arming_band = True
        dlg.set_xy_band_armed(True)
        dlg.xy_inner.setCurrentIndex(0)   # show the spectra to drag on
        self._status("Drag a band on the Tile spectra overlay...")

    def _xy_band_event_count(self, lo, hi):
        """Events inside [lo, hi] pooled over the selected tiles (for the band
        dialog's count readout)."""
        if self._xy_grid_centers is None:
            return 0
        mask = (self._xy_grid_centers >= lo) & (self._xy_grid_centers <= hi)
        total = 0
        for (c, r) in self._xy_selected:
            cc = self._xy_grid_counts[c, r]
            if cc is not None:
                total += int(cc[mask].sum())
        return total

    def _xy_on_band_span(self, emin, emax):
        if not self._xy_arming_band or emax <= emin:
            return
        dlg = self._sel_dlg
        self._xy_arming_band = False
        if dlg is not None:
            dlg.set_xy_band_armed(False)
        emin, emax = round(emin, 4), round(emax, 4)
        color = T.OVERLAY_COLORS[self._xy_band_color_idx % len(T.OVERLAY_COLORS)]
        # Same name + colour picker as the energy selections.
        bdlg = EnergySelectionDialog(
            emin, emax, color, self._xy_band_event_count(emin, emax), self.app)
        if bdlg.exec_() != QDialog.Accepted:
            self._status("Band cancelled")
            return
        label = bdlg.label() or f"band {len(self._xy_bands) + 1}"
        self._xy_band_color_idx += 1
        self._xy_bands.append(dict(
            label=label, color=bdlg.color(), emin=emin, emax=emax))
        if dlg is not None:
            dlg.refresh_xy_bands()
        self._xy_redraw_overlay()
        # Stay on the Tile spectra tab (drawing the band can otherwise leave the
        # view on the Area tab); the user is still working with the spectra.
        if dlg is not None:
            dlg.xy_inner.setCurrentIndex(0)
        self._status(f"Added band '{label}'  ·  [{emin:g}, {emax:g}]")

    def _xy_remove_band(self, band):
        try:
            self._xy_bands.remove(band)
        except ValueError:
            return
        if self._sel_dlg is not None:
            self._sel_dlg.refresh_xy_bands()
        self._xy_redraw_overlay()

    def _xy_clear_bands(self):
        if not self._xy_bands:
            return
        self._xy_bands = []
        if self._sel_dlg is not None:
            self._sel_dlg.refresh_xy_bands()
        self._xy_redraw_overlay()
        self._status("Cleared all bands")

    # -- net area vs position --------------------------------------------------
    def _xy_fit_tiles(self):
        """Open the interactive per-tile fit window for the picked band."""
        dlg = self._sel_dlg
        if dlg is None:
            return
        if self._xy_tiles is None:
            self._status("Build tiles first")
            return
        if not self._xy_selected:
            self._status("Click at least one tile to select it")
            return
        idx = dlg.xy_cmb_band.currentIndex()
        if not (0 <= idx < len(self._xy_bands)):
            self._status("Add an energy band (drag on the Tile spectra) first")
            return
        band = self._xy_bands[idx]
        mode = {0: "tile", 1: "x", 2: "y"}.get(dlg.xy_cmb_axis.currentIndex(),
                                               "tile")
        k = self._xy_keys
        slices = xyp.tile_slices(
            self._xy_grid_centers, self._xy_grid_counts, self._xy_selected,
            self._xy_tiles, k["ebins"], k["e_units"], mode=mode)
        if not slices:
            self._status("No selected tiles to fit")
            return
        win = xyp.TileFitWindow(
            self.app, slices, (band["emin"], band["emax"]), band["label"],
            band["color"], mode=mode)
        win.results_ready.connect(self._xy_receive_area)
        win.finished.connect(lambda *_: self._xy_fit_wins.discard(win))
        self._xy_fit_wins.add(win)
        win.show()
        win.raise_()
        what = {"tile": "tiles", "x": "columns", "y": "rows"}[mode]
        self._status(f"Fit {len(slices)} {what} for '{band['label']}' — step "
                     "through them, then Plot")

    def _xy_receive_area(self, payload):
        """Store a band's fit result (replacing any prior fit of the same band)
        and redraw the overlay with every visible band."""
        dlg = self._sel_dlg
        if dlg is None:
            return
        payload["visible"] = True
        prev = next((r for r in self._xy_area_results
                     if r.get("label") == payload.get("label")), None)
        if prev is not None:
            self._xy_area_results[self._xy_area_results.index(prev)] = payload
        else:
            self._xy_area_results.append(payload)
        self._xy_redraw_area()
        dlg.refresh_xy_area_list()
        dlg.xy_inner.setCurrentIndex(1)
        self._status(f"Plotted net area vs position for '{payload['label']}'")

    def _xy_redraw_area(self):
        dlg = self._sel_dlg
        if dlg is None:
            return
        visible = [r for r in self._xy_area_results if r.get("visible", True)]
        xyp.plot_area_overlays(dlg.xy_area_fig, visible)
        dlg.xy_area_canvas.draw_idle()

    def _xy_toggle_area(self, result, visible):
        result["visible"] = bool(visible)
        self._xy_redraw_area()

    def _xy_clear_area(self):
        dlg = self._sel_dlg
        if dlg is None:
            return
        self._xy_area_results = []
        self._xy_redraw_area()
        dlg.refresh_xy_area_list()

    # -- time-slice fits vs dt -----------------------------------------------------
    def _build_slices(self, dt_slice_w, min_snr=3.0):
        """Split the current dt range into slices and build one energy
        Spectrum + PeakSearch per slice (shared energy binning/range).

        Returns a list of dicts ``{idx, t0, t1, tc, spe, search}``; an empty
        list if the data isn't ready.  ``dt_slice_w`` of None gives ~10 slices.
        ``min_snr`` sets the per-slice peak-search threshold.
        """
        if self.df_current is None or self.ekey is None:
            return []
        dt_col = self._dt_key
        dt = self.df_current[dt_col].to_numpy(dtype=float)
        e = self.df_current[self.ekey].to_numpy(dtype=float)
        dt_lo, dt_hi = np.percentile(dt, [0.2, 99.5])
        if not np.isfinite(dt_hi - dt_lo) or dt_hi <= dt_lo:
            return []
        if dt_slice_w is None or dt_slice_w <= 0:
            dt_slice_w = (dt_hi - dt_lo) / 10.0
        n = max(1, int(round((dt_hi - dt_lo) / dt_slice_w)))
        if n > MAX_SLICES:
            n = MAX_SLICES
            self._status(f"Too many slices for that width; capped at {MAX_SLICES}")
        edges = np.linspace(dt_lo, dt_hi, n + 1)

        # PeakSearch reference (channels) scaled to the current binning, matching
        # the example file (420 ch / 12 ch FWHM tuned at 2**11 bins).
        ref_x = max(1.0, 420.0 * self.ebins / 2 ** 11)
        ref_fwhm = max(1.0, 12.0 * self.ebins / 2 ** 11)

        # Carry the energy units onto each slice Spectrum so the fit window
        # labels them like the panel (see _axis_units): only an applied
        # calibration is energy; everything else is raw channels (no units).
        e_units = self._axis_units()

        slices = []
        for i in range(n):
            m = (dt >= edges[i]) & (dt < edges[i + 1])
            cts, edg = np.histogram(e[m], bins=self.ebins, range=self.erange)
            centers = (edg[1:] + edg[:-1]) / 2
            spe = sp.Spectrum(counts=cts, energies=centers, e_units=e_units,
                              label=f"t=[{edges[i]:.1f},{edges[i + 1]:.1f}]")
            search = ps.PeakSearch(spe, ref_x, ref_fwhm, fwhm_at_0=1.0,
                                   min_snr=min_snr)
            slices.append(dict(idx=i, t0=float(edges[i]), t1=float(edges[i + 1]),
                               tc=0.5 * (edges[i] + edges[i + 1]),
                               spe=spe, search=search))
        return slices

    def _open_slice_fits(self, sel, dt_slice_w, technique, min_snr=3.0):
        """Build the per-slice spectra, show the spectra figure, and (for the
        fit techniques) open the interactive slice-fit window."""
        if self.df_current is None or self.page.ax_dt is None:
            self._status("Load an API file first")
            return
        band = (float(sel["emin"]), float(sel["emax"]))
        if band[1] <= band[0]:
            self._status("This selection has an empty energy band")
            return
        slices = self._build_slices(dt_slice_w, min_snr)
        if not slices:
            self._status("Could not build dt slices from the current data")
            return

        # Spectra-per-dt figure (example figure 1), embedded in the Selections
        # dialog right below the Slice & fit button.  Skip if already drawn.
        if self._sel_dlg is not None and self._sel_dlg._spectra_args is None:
            self._sel_dlg.show_slice_spectra(slices, band, self._energy_xlabel())

        if technique == TECH_SNR:
            # SNR needs no fitting — compute per slice and overlay directly.
            centers = [s["tc"] for s in slices]
            vals = [band_snr(s["search"], band) for s in slices]
            errs = [0.0] * len(slices)
            self._receive_slice_results(dict(
                label=sel["label"], color=sel["color"], technique=TECH_SNR,
                dt_centers=centers, vals=vals, errs=errs,
                ylabel=YLABELS[TECH_SNR]))
            return

        # Fit → interactive, slice-stepping fit window (its Method selector
        # picks peak fit vs net area − linear bkg per slice).
        win = SliceFitWindow(self.app, slices, band, sel["label"], sel["color"])
        win.results_ready.connect(self._receive_slice_results)
        win.finished.connect(lambda *_: self._slice_fit_wins.discard(win))
        self._slice_fit_wins.add(win)
        win.show()
        win.raise_()
        self._status(
            f"Slice-fit '{sel['label']}' — {len(slices)} slices · "
            f"{TECHNIQUE_LABELS[technique]}")

    def _receive_slice_results(self, payload):
        """Store one selection's vs-dt curve and (re)draw the dt overlay."""
        self._slice_results[payload["label"]] = payload
        if self._sel_dlg is not None:
            self._sel_dlg.refresh_ratio_combo()
        self._draw_slice_overlay()
        self._status(
            f"{payload['label']} ({TECHNIQUE_LABELS[payload['technique']]}) vs dt "
            f"overlaid on the dt panel")

    def _set_slice_ratio_ref(self, ref):
        """Switch the dt overlay between absolute values and ratios to *ref*."""
        self._slice_ratio_ref = ref
        self._draw_slice_overlay()
        self._status(f"Plotting ratios to {ref}" if ref
                     else "Plotting absolute slice values")

    def _ratio_active(self):
        """True when a valid reference with ≥2 results is selected."""
        return (self._slice_ratio_ref in self._slice_results
                and len(self._slice_results) >= 2)

    def _slice_overlay_ylabel(self):
        """y-label for the overlay: ratio label, shared technique, else generic."""
        if self._ratio_active():
            return f"Ratio to {self._slice_ratio_ref}"
        techs = {r["technique"] for r in self._slice_results.values()}
        if len(techs) == 1:
            return YLABELS[next(iter(techs))]
        return "Slice value"

    def _draw_slice_overlay(self):
        """Redraw the dt panel with a gray base histogram and every stored
        selection's value-vs-dt curve (or ratio to a reference) on a shared
        secondary y-axis."""
        ax = self.page.ax_dt
        if ax is None or self.df_current is None:
            return
        # Reuse the significance overlay slots (_ax_sig / _sig_active) so the
        # existing clear/redraw plumbing keeps working unchanged.
        self._clear_dt_twin()
        dt_col = self._dt_key
        dt_lo, dt_hi = np.percentile(self.df_current[dt_col], [0.2, 99.5])
        ax.clear()
        ax.hist(self.df_current[dt_col], bins=self.tbins, range=(dt_lo, dt_hi),
                color=T.TEXT_DIM, alpha=0.35, edgecolor=API_PLOT_BG, linewidth=0.3)
        ax.set_xlabel("dt (ns, shifted)" if self._dt_corrected else "dt (ns)")
        ax.set_ylabel("Counts", color=T.TEXT_DIM)
        self.page._style(ax)

        if not self._slice_results:
            self.page.canvas.draw_idle()
            return

        ax_val = ax.twinx()
        ax_val.set_ylabel(self._slice_overlay_ylabel(), color=T.TEXT_DIM,
                          fontsize=self.page.LABEL_FS)
        ax_val.tick_params(colors=T.TEXT_DIM, which="both", length=3,
                           labelsize=self.page.TICK_FS)
        ax_val.spines["right"].set_color(T.BORDER)
        ax_val.spines["top"].set_color(T.BORDER)
        ax_val.set_facecolor("none")

        handles = []
        if self._ratio_active():
            ref = self._slice_results[self._slice_ratio_ref]
            rc = np.asarray(ref["dt_centers"], dtype=float)
            rv = np.asarray(ref["vals"], dtype=float)
            re_ = np.asarray(ref["errs"], dtype=float)
            for label, r in self._slice_results.items():
                if label == self._slice_ratio_ref:
                    continue
                c = np.asarray(r["dt_centers"], dtype=float)
                v = np.asarray(r["vals"], dtype=float)
                e = np.asarray(r["errs"], dtype=float)
                # Align the reference onto this curve's dt centres (selections
                # may have been sliced with different widths).
                ref_v = np.interp(c, rc, rv)
                ref_e = np.interp(c, rc, re_)
                ratio, rerr = ratio_to_ref(v, e, ref_v, ref_e)
                lbl = f"{label}/{self._slice_ratio_ref}"
                ax_val.errorbar(c, ratio, yerr=rerr, color=r["color"], lw=1.5,
                                marker="o", ms=4, capsize=3, zorder=3)
                handles.append(Line2D([0], [0], color=r["color"], lw=1.5, label=lbl))
        else:
            for r in self._slice_results.values():
                centers = np.asarray(r["dt_centers"], dtype=float)
                vals = np.asarray(r["vals"], dtype=float)
                errs = np.asarray(r["errs"], dtype=float)
                ax_val.errorbar(centers, vals, yerr=errs, color=r["color"], lw=1.5,
                                marker="o", ms=4, capsize=3, zorder=3)
                handles.append(Line2D([0], [0], color=r["color"], lw=1.5,
                                      label=r["label"]))
        if handles:
            ax_val.legend(handles, [h.get_label() for h in handles],
                          loc="upper right", fontsize=11, facecolor=API_PLOT_BG,
                          edgecolor=T.BORDER, labelcolor=T.TEXT_PRIMARY)

        self._sig_active = True
        self._ax_sig = ax_val
        self.page.canvas.draw_idle()

    def _clear_slice_overlay(self):
        """Forget all stored vs-dt results and restore the plain dt histogram."""
        self._slice_results.clear()
        self._slice_ratio_ref = None
        if self._sel_dlg is not None:
            self._sel_dlg.refresh_ratio_combo()
        self._clear_significance()
        self._status("Cleared time-slice overlay")

    def _clear_dt_twin(self):
        """Drop the secondary (twinx) axes carrying the vs-dt overlay so a fresh
        redraw of the dt panel doesn't stack ghost axes on top of the old one."""
        ax = self._ax_sig
        if ax is not None:
            try:
                self.page.fig.delaxes(ax)
            except Exception:  # noqa: BLE001  -- axes may already be gone
                pass
            self._ax_sig = None
        self._sig_active = False

    def _clear_significance(self):
        """Remove the significance overlay and restore the normal dt histogram."""
        if not self._sig_active:
            return
        self._sig_active = False
        if self._ax_sig is not None:
            try:
                self.page.fig.delaxes(self._ax_sig)
            except Exception:  # noqa: BLE001  -- axes may already be gone
                pass
            self._ax_sig = None
        ax = self.page.ax_dt
        if ax is not None and self.df_current is not None:
            ax.clear()
            self._plot_time(self.df_current)
            self.page.canvas.draw_idle()

    def _reset_selections(self):
        """Drop all selections (e.g. when a new run is loaded) without a status
        message  -- the caller reports the higher-level action."""
        self.selections = []
        self._sel_color_idx = 0
        self._arming_selection = False
        self._set_add_armed(False)
        self._slice_results.clear()
        self._slice_ratio_ref = None
        self._selections_plotted = False
        if self._sel_dlg is not None:
            self._sel_dlg.refresh_ratio_combo()
            self._sel_dlg._spectra_args = None
            self._sel_dlg.spectra_fig.clear()
            self._sel_dlg.spectra_canvas.draw_idle()
        self._clear_significance()
        self._refresh_sel_list()

    def _plot_selections(self):
        if self.df_current is None:
            self._status("Load an API file first")
            return
        if not self.selections:
            self._status("No selections to plot  -- open Selections and drag a band")
            return
        if self._sel_dlg is not None:
            self._flash_button(self._sel_dlg.btn_plot_sel)
        self._plot_energy_overlays()
        self._plot_time_overlays()
        self._plot_xy_overlays()
        self._selections_plotted = True
        self.page.reset_nav()
        self.page.canvas.draw_idle()
        if self._sel_dlg is not None:
            self._refresh_slice_spectra()
        self._status(f"Plotted {len(self.selections)} selection(s)")

    def _refresh_slice_spectra(self):
        """Build per-slice spectra using the dialog's current settings and
        render them in the embedded canvas.  Uses the active cmb_sel band."""
        dlg = self._sel_dlg
        if dlg is None or not self.selections:
            return
        dt_slice_txt = dlg.ed_dt_slice.text().strip()
        dt_slice_w = None
        if dt_slice_txt:
            try:
                v = float(dt_slice_txt)
                if v > 0:
                    dt_slice_w = v
            except ValueError:
                pass
        try:
            min_snr = float(dlg.ed_min_snr.text().strip())
            if min_snr <= 0:
                raise ValueError
        except ValueError:
            min_snr = 3.0
        idx = dlg.cmb_sel.currentIndex()
        if idx < 0 or idx >= len(self.selections):
            idx = 0
        sel = self.selections[idx]
        if dlg._spectra_args is None:
            slices = self._build_slices(dt_slice_w, min_snr)
            if slices:
                dlg.show_slice_spectra(
                    slices, (float(sel["emin"]), float(sel["emax"])),
                    self._energy_xlabel())

    def _clear_selection_plots(self):
        """Remove selection overlays from dt and X-Y panels.  The energy
        spectrum keeps its coloured selection bands."""
        if self.df_current is None:
            return
        self._selections_plotted = False
        if self.page.ax_dt is not None:
            self.page.ax_dt.clear()
            self._plot_time(self.df_current)
        self._replot_xy()
        self.page.reset_nav()
        self.page.canvas.draw_idle()
        self._status("Selection plots cleared")

    def _plot_energy_overlays(self):
        self._compute_energy_hist(self.df_current)
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
        ax.legend(loc="upper right", fontsize=11, facecolor=API_PLOT_BG,
                  edgecolor=T.BORDER, labelcolor=T.TEXT_PRIMARY)
        self.page._style(ax)

    def _plot_time_overlays(self):
        ax = self.page.ax_dt
        if ax is None:
            return
        self._clear_dt_twin()
        ax.clear()
        base = self.df_current
        if base.shape[0] == 0:
            self.page._style(ax)
            return
        low, high = np.percentile(base[self._dt_key], [0.2, 99.5])
        # Gray base histogram of the full run, then each selection's dt
        # distribution as a coloured step outline on the same counts axis.
        ax.hist(base[self._dt_key], bins=self.tbins, range=(low, high),
                color=T.TEXT_DIM, alpha=0.30, edgecolor=API_PLOT_BG, linewidth=0.3)
        ax.set_xlabel("dt (ns)")
        ax.set_ylabel("Counts")
        for sel in self.selections:
            d = sel["df"][self._dt_key]
            if len(d) == 0:
                continue
            ax.hist(d, bins=self.tbins, range=(low, high), histtype="step",
                    color=sel["color"], linewidth=1.3)
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
            ax.legend(handles=handles, loc="upper right", fontsize=11,
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
        dataframe's raw energy column  -- so it applies directly."""
        self._flash_button(self.opts.btn_retrieve_cal)
        if self.df_current is None:
            self._status("Load an API file first")
            return
        cal = getattr(self.app, "calibration", None)
        res = cal.current_calibration() if cal is not None else None
        if res is None:
            self._status("No calibration in the Calibration tab  -- build one there "
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
        # Existing selections were cut in the old (channel) axis  -- drop them.
        self._reset_selections()
        self._configure_keys()
        self._initialize_plots()
        self.opts.btn_clear_cal.setEnabled(True)
        deg = len(coeffs) - 1
        self.opts.lbl_cal.setText(f"Calibrated → energy ({units}), degree {deg}")
        self._status(f"Retrieved calibration from the Calibration tab "
                     f"(degree {deg}, {units})")

    def _clear_calibration(self):
        """Drop the calibration and revert the panels to the original raw
        channels (rebuilds the view from the master frame, like reset)."""
        self._flash_button(self.opts.btn_clear_cal)
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
        self._configure_keys()
        self._initialize_plots()
        self.opts.btn_clear_cal.setEnabled(False)
        self.opts.lbl_cal.setText("Uncalibrated (channels)")
        self._status("Calibration cleared  -- back to raw channels")

    # -- drift correction (Shifts... window) -------------------------------------
    def _rebuild_from_master(self):
        """Re-derive the working frames from df_api and redraw (reset-style)."""
        self.df_current = self.df_api.copy()
        self.df_previous = self.df_api.copy()
        self.en_flag = self.dt_flag = self.xy_flag = 0
        self.vmax = None
        self.opts.ed_vmax.clear()
        self._reset_selections()
        self._configure_keys()
        self._initialize_plots()

    def _chan_erange(self):
        """Histogram range for the raw channel axis: always [0, max]."""
        base = self._chan_base or "energy_orig"
        return (0.0, float(self.df_api[base].max()))

    def _dt_erange(self):
        """Histogram range for the time axis: the 0.2%–99.5% span of dt, so the
        dt peak fills the histogram instead of being squashed by a few stray edge
        events. Matches the API view's dt range and the multi-run Combine window
        (see ``combine_runs``)."""
        dt = self.df_api["dt"].to_numpy(dtype=float)
        lo, hi = np.percentile(dt, [0.2, 99.5])
        if hi > lo:
            return (float(lo), float(hi))
        return (float(dt.min()), float(dt.max()))

    # energy gain shift ........................................................
    def apply_energy_gainshift(self, n_segments, method="shift", xrange=None,
                               bins=None):
        """Drift-correct the energy channels: split into time segments and align
        each onto the first, writing corrected channels to ``energy_drift``.
        ``_chan_key`` then reads that column, so the histogram, send-to-spectrum
        and any polynomial calibration all layer on the corrected channels. The
        column lives on df_api, so it survives Reset. ``bins`` overrides the
        histogram bin count used for the alignment (defaults to ``self.ebins``).

        Returns ``True`` when a pre-existing *file-provided* energy calibration
        was dropped (the shift recomputed the energy axis), so the caller can
        prompt the user to re-calibrate."""
        if self.df_api is None:
            self._status("Load an API file first")
            return False
        base = self._chan_base or "energy_orig"
        gs = apicalc.GainShift(self.df_api, n_segments=int(n_segments),
                               bins=int(bins) if bins else self.ebins,
                               erange=self._chan_erange(),
                               col=base, out_col="energy_drift")
        gs.align(method=method, xrange=xrange)
        self.df_api = gs.df
        # A gain-shift recomputes the energy axis from the raw channels, so a
        # file-provided energy_cal (one we have no polynomial to reproduce) is no
        # longer valid: drop it and mark the run uncalibrated so the corrected
        # channels are what's shown/sent. A GUI calibration (self._cal_coeffs) is
        # a polynomial we *can* re-apply to the corrected channels, so it stays.
        recal = False
        if self._cal_coeffs is None and "energy_cal" in self.df_api.columns:
            self.df_api = self.df_api.drop(columns=["energy_cal"])
            self.e_units = None
            self.opts.lbl_cal.setText("Uncalibrated (channels)")
            self.opts.btn_clear_cal.setEnabled(False)
            recal = True
        self._egain_applied = True
        self._egain_label = f"Gain-shift: {int(n_segments)} seg · {method}"
        self._rebuild_from_master()
        # Keep any polynomial calibration consistent with the corrected channels.
        if self._cal_coeffs is not None:
            self.apply_calibration(self._cal_coeffs, self.e_units)
        self._refresh_shift_labels()
        msg = f"Applied energy gain-shift ({int(n_segments)} seg, {method})"
        if recal:
            msg += "  -- energy calibration removed; re-calibrate"
        self._status(msg)
        return recal

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
    def _compose_dt_label(self):
        """Human-readable label describing the active time correction(s)."""
        parts = []
        if self._dt_segments_applied:
            parts.append(f"aligned ({self._dt_seg_desc})")
        if self._dt_shift:
            parts.append(f"{self._dt_shift:+g} ns constant")
        if not parts:
            return "No time shift"
        return "Time " + " + ".join(parts)

    def _dt_const_base(self, df):
        """Column the constant shift builds on: the segment-aligned ``dt_aligned``
        when an alignment is active (so the alignment is retained), else raw
        ``dt``."""
        if self._dt_segments_applied and "dt_aligned" in df.columns:
            return df["dt_aligned"].astype(float)
        return df["dt"].astype(float)

    def apply_dt_shift(self, shift):
        """Constant time shift composed on top of the current baseline: when a
        segment alignment is applied the constant builds on the aligned
        ``dt_aligned`` (so the alignment is preserved), otherwise on the raw
        ``dt``. Re-applying replaces the previous constant rather than
        accumulating. Writes ``dt_cal``; the columns live on df_api so they
        survive Reset; independent of the energy correction."""
        if self.df_api is None:
            return
        shift = float(shift)
        df = self.df_api.copy()
        df["dt_cal"] = self._dt_const_base(df) + shift
        self.df_api = df
        self._dt_shift = shift
        self._dt_label = self._compose_dt_label()
        self._rebuild_from_master()
        self._refresh_shift_labels()
        self._status(f"Applied time shift {shift:+g} ns")

    def apply_dt_preview_shift(self, aligned, shift, seg_desc):
        """Commit a *previewed* segment alignment together with a constant Δt in
        one step: ``dt_aligned`` is set to the per-event aligned dt the user saw
        in the preview, and ``dt_cal = dt_aligned + shift``. No re-alignment is
        run -- the preview is the base. Re-applying a different constant later
        composes on this same ``dt_aligned`` (see ``apply_dt_shift``)."""
        if self.df_api is None:
            return
        shift = float(shift)
        df = self.df_api.copy()
        df["dt_aligned"] = np.asarray(aligned, dtype=float)
        df["dt_cal"] = df["dt_aligned"] + shift
        self.df_api = df
        self._dt_segments_applied = True
        self._dt_seg_desc = seg_desc
        self._dt_shift = shift
        self._dt_label = self._compose_dt_label()
        self._rebuild_from_master()
        self._refresh_shift_labels()
        self._status(f"Applied previewed alignment + {shift:+g} ns")

    def apply_dt_gainshift(self, n_segments, method="shift", xrange=None,
                           bins=None):
        """Segment-based time alignment: split dt into time segments and align
        each onto the first, writing ``dt_aligned``. Any active constant shift
        is re-layered on top so ``dt_cal = dt_aligned + constant`` -- the two
        corrections compose and are order-independent. ``bins`` overrides the
        histogram bin count (defaults to ``self.tbins``); the alignment range is
        the 1%–99% dt span (see ``_dt_erange``)."""
        if self.df_api is None:
            self._status("Load an API file first")
            return
        gs = apicalc.GainShift(self.df_api, n_segments=int(n_segments),
                               bins=int(bins) if bins else self.tbins,
                               erange=self._dt_erange(),
                               col="dt", out_col="dt_aligned")
        gs.align(method=method, xrange=xrange)
        df = gs.df
        self._dt_segments_applied = True
        self._dt_seg_desc = f"{int(n_segments)} seg · {method}"
        # Layer any existing constant on top of the freshly aligned dt.
        df["dt_cal"] = df["dt_aligned"].astype(float) + (self._dt_shift or 0.0)
        self.df_api = df
        self._dt_label = self._compose_dt_label()
        self._rebuild_from_master()
        self._refresh_shift_labels()
        self._status(f"Applied time alignment ({int(n_segments)} seg, {method})")

    def _clear_dt_shift(self):
        """Drop the time correction (constant and/or segment) and revert to raw dt."""
        if not self._dt_corrected:
            self._status("No time correction applied")
            return
        if self.df_api is not None:
            drop = [c for c in ("dt_cal", "dt_aligned")
                    if c in self.df_api.columns]
            if drop:
                self.df_api = self.df_api.drop(columns=drop)
        self._dt_shift = None
        self._dt_segments_applied = False
        self._dt_seg_desc = ""
        self._dt_label = "No time shift"
        self._rebuild_from_master()
        self._refresh_shift_labels()
        self._status("Time correction cleared  -- back to raw dt")

    def _refresh_shift_labels(self):
        """Push the current correction state into the Shifts dialog if it's open."""
        dlg = self._shifts_dlg
        if dlg is None:
            return
        dlg.set_energy_state(self._egain_applied, self._egain_label)
        dlg.set_time_state(self._dt_corrected, self._dt_label)

    def _open_shifts(self):
        self._flash_button(self.opts.btn_shifts)
        if self.df_api is None:
            self._status("Load an API file first")
            return
        if self._shifts_dlg is None:
            self._shifts_dlg = ShiftsDialog(self)
        self._refresh_shift_labels()
        self._shifts_dlg.refresh_previews()
        self._shifts_dlg.show()
        self._shifts_dlg.raise_()

    def _open_combine(self):
        self._flash_button(self.opts.btn_combine)
        if self._combine_dlg is None:
            self._combine_dlg = CombineRunsDialog(self)
        # Seed the runs list with the currently loaded run (and its data path) so
        # the common case  -- "this run plus a few more"  -- starts pre-filled.
        if self._src_date is not None:
            self._combine_dlg.seed(self._src_date, self._src_runnr,
                                   self._src_data_path)
        self._combine_dlg.show()
        self._combine_dlg.raise_()
        self._combine_dlg.activateWindow()

    # -- 3D volume -------------------------------------------------------------
    def _open_3d(self):
        self._flash_button(self.opts.btn_3d)
        if self.df_current is None:
            self._status("Load an API file first")
            return
        if self._api3d_dlg is None:
            try:
                self._api3d_dlg = Api3DDialog(self.app)
            except Exception as exc:  # noqa: BLE001  -- QtWebEngine may be missing
                traceback.print_exc()
                self._status(f"Could not open the 3D view: {exc}")
                return
            self._api3d_dlg.btn_plot.clicked.connect(self._create_plot_3d)
        self._api3d_dlg.show()
        self._api3d_dlg.raise_()
        self._api3d_dlg.activateWindow()

    def _xyz(self, df=None):
        """Reconstructed (X, Y, Z) for *df* (default: the current view). Data
        that carries a direct X/Y/Z cloud (simulated) is used as-is;
        reconstructed-position data is mapped via apicalc.api_xyz (the
        gamma-detector position is ignored, as in the legacy 3D plot)."""
        df = self.df_current if df is None else df
        if self.xkey == "X" and "Z" in df.columns:
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
            self._set_3d_status("No events to plot  -- load or relax the filters.")
            return
        dlg = self._api3d_dlg
        num_bins = dlg.value("no_bins", int)
        iso_min = dlg.value("isomin", float)
        iso_max = dlg.value("isomax", float)
        opacity = dlg.value("opacity", float)
        surfcount = dlg.value("surfcount", int)
        if None in (num_bins, iso_min, iso_max, opacity, surfcount):
            self._set_3d_status("Check the plot inputs  -- they must be numbers.")
            return
        if num_bins < 2 or surfcount < 1:
            self._set_3d_status("Need ≥2 bins and ≥1 surface.")
            return

        self._set_3d_status("Building volume...")
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
                    self._set_3d_status("Histogram is empty/flat  -- nothing to render.")
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
        except Exception as exc:  # noqa: BLE001  -- surface render errors to the dialog
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

    def _apply_bins(self):
        """Re-bin the three panels from the Display bin boxes and redraw. Each
        box must be a positive integer; an invalid one aborts without changing
        anything."""
        o = self.opts
        parsed = {}
        for attr, ed, name in [("ebins", o.ed_ebins, "Energy bins"),
                               ("tbins", o.ed_tbins, "dt bins"),
                               ("hexbins", o.ed_xybins, "X-Y bins")]:
            try:
                val = int(ed.text().strip())
            except ValueError:
                self._status(f"{name} must be a whole number")
                return
            if val < 1:
                self._status(f"{name} must be at least 1")
                return
            parsed[attr] = val
        self.ebins, self.tbins, self.hexbins = (
            parsed["ebins"], parsed["tbins"], parsed["hexbins"])
        if self.df_current is None:
            self._status("Bin counts set  -- load an API file to apply them")
            return
        # Redraw each panel with the new binning (no file re-read), mirroring the
        # clear-then-plot pattern the filters use.
        if self.page.ax_spe is not None:
            self.page.ax_spe.clear()
            self._plot_energy(self.df_current)
        if self.page.ax_dt is not None:
            self.page.ax_dt.clear()
            self._plot_time(self.df_current)
        self._replot_xy()
        self.page.reset_nav()
        self.page.canvas.draw_idle()
        self._status(
            f"Bins → energy {self.ebins:,} · dt {self.tbins:,} · X-Y {self.hexbins:,}")

    def _confirm_uncalibrated_energy(self):
        """Warn that the energy is being saved into ``energy_cal`` without a
        calibration curve (gain-shift/alignment only). Returns True to proceed."""
        resp = QMessageBox.warning(
            self.app, "Energy not calibrated",
            "The energy axis has been gain-shifted/aligned but no calibration "
            "curve is applied (none retrieved). It will be saved into the "
            "energy_cal column, but the spectrum may not be in true energy "
            "units.\n\nRetrieve/apply a calibration first for a proper energy "
            "scale.\n\nSave the shifted (uncalibrated) energy anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return resp == QMessageBox.Yes

    def _confirm_existing_run(self, new_date, new_runnr, run_dir):
        """Ask what to do when the destination run already exists. Returns
        ``"merge"`` (add this channel, keeping the others), ``"overwrite"``
        (rebuild the run from the source) or ``"cancel"``."""
        box = QMessageBox(self.app)
        box.setWindowTitle("Run already exists")
        box.setText(
            f"Run {new_date}-{new_runnr} already exists at:\n{run_dir}\n\n"
            f"Add channel {self._src_ch}'s calibration to the existing run "
            "(keeping other channels), or overwrite the whole run from the "
            "source?")
        add_btn = box.addButton("Add channel", QMessageBox.AcceptRole)
        over_btn = box.addButton("Overwrite", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(add_btn)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is add_btn:
            return "merge"
        if clicked is over_btn:
            return "overwrite"
        return "cancel"

    def _apply_to_data(self):
        """Bake the active energy/time corrections into a *new* on-disk run.

        Re-reads every channel of the source run (all parquet chunks combined),
        adds ``energy_cal`` and/or ``dt_cal`` for the loaded channel's rows
        (other channels get NaN  -- they aren't calibrated yet), and writes the
        result as a single combined parquet under a user-chosen run number
        (date defaulting to the source run). The new run loads straight back
        through the Date/Run/Channel fields, with the calibration detected
        automatically; the source run is left untouched.

        ``energy_cal`` is written when a polynomial calibration is active
        (``e_units`` set) or an energy gain-shift has been applied; in the
        gain-shift-only case ``self.ekey`` points at ``energy_drift``, so the
        drift-corrected channels are baked in. ``dt_cal`` is written when any
        time correction is active. Time units follow the on-disk convention:
        raw ``dt`` stays in seconds, ``dt_cal`` is in ns (already so on
        df_api)."""
        self._flash_button(self.opts.btn_apply_data)
        if self.df_api is None or self._src_date is None:
            self._status("Load an API file first")
            return
        if self.flood_field:
            self._status("Flood-field runs have no calibration to apply")
            return
        energy_changed = self.e_units is not None or self._egain_applied
        dt_changed = self._dt_corrected
        if not energy_changed and not dt_changed:
            self._status("No energy or time changes to apply")
            return

        # Energy is written into the canonical ``energy_cal`` column, but when no
        # calibration curve is active (e_units is None) it only carries a
        # gain-shift/alignment of the raw channels -- not a true energy
        # calibration. Warn the user before baking it in as ``energy_cal``.
        if energy_changed and self.e_units is None:
            if not self._confirm_uncalibrated_energy():
                self._status("Apply to data cancelled  -- energy not calibrated")
                return

        dlg = ApplyToDataDialog(self._src_date, self._src_runnr, self.app)
        if dlg.exec_() != QDialog.Accepted:
            return
        new_date, new_runnr = dlg.values()
        if not new_date:
            self._status("Enter a date for the new run")
            return
        if new_runnr is None:
            self._status("New run number must be an integer")
            return

        # Resolve the destination run folder up front so we can tell whether we
        # are creating a new run or adding a channel to an existing one.
        try:
            run_dir, _, _ = read_parquet_api.run_parquet_path(
                new_date, new_runnr, self._src_data_path)
        except Exception as exc:  # noqa: BLE001
            self._status(f"Bad destination date/run: {exc}")
            return

        # When the destination run already exists, default to *merging* this
        # channel's calibration into it -- so each channel can be saved one at a
        # time into the same run, accumulating, instead of overwriting the whole
        # run from the source (which would discard previously-saved channels).
        # The user can still choose to overwrite the run from scratch.
        merge = False
        if run_dir.exists():
            choice = self._confirm_existing_run(new_date, new_runnr, run_dir)
            if choice == "cancel":
                self._status("Apply to data cancelled")
                return
            merge = (choice == "merge")

        # Base table to write: when merging, the existing destination run (so the
        # other channels' previously-saved calibration is preserved); otherwise a
        # fresh re-read of every channel of the source run. dt stays in seconds
        # on disk (the loader scales it to ns at load time).
        read_date, read_runnr = ((new_date, new_runnr) if merge
                                 else (self._src_date, self._src_runnr))
        which = "destination" if merge else "source"
        try:
            full = read_parquet_api.read_parquet_file(
                date=read_date, runnr=read_runnr, ch=None,
                data_path_txt=self._src_data_path)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._status(f"Could not re-read {which} run: {exc}")
            return
        if full is None:
            self._status(f"Could not re-read the {which} run's parquet files")
            return

        # The loaded channel's rows, in load order, line up with df_api's rows.
        mask = read_parquet_api.channel_mask(full, self._src_ch)
        n = int(mask.to_numpy().sum())
        if n != len(self.df_api):
            self._status(
                f"Row mismatch for ch {self._src_ch} ({n} in the {which} run vs "
                f"{len(self.df_api)} loaded)  -- cannot align; aborting")
            return

        # Fill only the loaded channel's rows. Create the column (NaN elsewhere)
        # only when it does not already exist, so any calibration a previous
        # channel wrote into the destination run is preserved.
        applied = []
        # When calibrated, self.ekey is "energy_cal"; gain-shift only ⇒
        # "energy_drift". Either way it becomes the canonical energy_cal column.
        if energy_changed and self.ekey in self.df_api.columns:
            if "energy_cal" not in full.columns:
                full["energy_cal"] = np.nan
            full.loc[mask, "energy_cal"] = self.df_api[self.ekey].to_numpy()
            applied.append("energy_cal")
        if dt_changed and self._dt_key in self.df_api.columns:
            if "dt_cal" not in full.columns:
                full["dt_cal"] = np.nan
            full.loc[mask, "dt_cal"] = self.df_api[self._dt_key].to_numpy()
            applied.append("dt_cal")
        if not applied:
            self._status("No energy or time changes to apply")
            return

        # A dt_cal column means "the time to use" for every event. Channels (or
        # rows) without their own time correction keep their raw dt -- convert it
        # to ns and backfill, so they are never left NaN. Without this, merging a
        # shifted channel into a run leaves the un-shifted channels with no time
        # at all (the column exists run-wide but their rows are NaN).
        if "dt_cal" in full.columns:
            missing = full["dt_cal"].isna()
            if missing.any():
                full.loc[missing, "dt_cal"] = (
                    full.loc[missing, "dt"].astype(float) * 1e9)

        try:
            out = read_parquet_api.save_combined_run(
                full, new_date, new_runnr, self._src_data_path, overwrite=True)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._status(f"Could not save calibrated run: {exc}")
            return

        # Copy the source run's settings folder (live_time / count-rate stats the
        # API calculations read) and drop/append a README recording provenance,
        # so the re-processed run carries the same metadata as its source.
        extra = self._copy_run_metadata(run_dir, new_date, new_runnr, applied,
                                        merge=merge)

        action = "Updated" if merge else "Saved"
        msg = (f"{action} calibrated run {new_date}-{new_runnr} "
               f"(ch {self._src_ch}: {', '.join(applied)}) → {out}")
        if extra:
            msg += f"  ·  {extra}"
        self._status(msg)

    def _copy_run_metadata(self, dst_run_dir, new_date, new_runnr, applied,
                           merge=False):
        """Copy the source run's ``settings/`` folder into the destination run and
        write a ``README.txt`` documenting where the data was re-processed from.
        When ``merge`` is set (a channel is being added to an existing run), the
        provenance entry is *appended* so each channel's origin is recorded.

        Best-effort: returns a short status note describing what was written, or
        an empty string if neither could be produced (the parquet save still
        stands on its own)."""
        notes = []
        try:
            src_run_dir, _, _ = read_parquet_api.run_parquet_path(
                self._src_date, self._src_runnr, self._src_data_path)
        except Exception:  # noqa: BLE001
            src_run_dir = None

        # 1) Copy the settings folder verbatim, if the source has one.
        if src_run_dir is not None:
            src_settings = src_run_dir / "settings"
            if src_settings.is_dir():
                dst_settings = dst_run_dir / "settings"
                try:
                    shutil.copytree(src_settings, dst_settings, dirs_exist_ok=True)
                    notes.append("settings copied")
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
                    notes.append("settings copy failed")

        # 2) Provenance README. Append a per-channel entry when adding a channel
        #    to an existing run; otherwise write a fresh README for the new run.
        try:
            readme = dst_run_dir / "README.txt"
            src_loc = str(src_run_dir) if src_run_dir is not None else "(unknown)"
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = (
                f"\n--- channel {self._src_ch} added on {stamp} ---\n"
                f"Source run      : {self._src_date}-{self._src_runnr} "
                f"(channel {self._src_ch})\n"
                f"Source location : {src_loc}\n"
                f"Applied columns : {', '.join(applied)}\n")
            if merge and readme.exists():
                with readme.open("a", encoding="utf-8") as fh:
                    fh.write(entry)
                notes.append("README updated")
            else:
                readme.write_text(
                    "This run was re-processed by the wara API tab.\n\n"
                    f"New run         : {new_date}-{new_runnr}\n"
                    + entry, encoding="utf-8")
                notes.append("README written")
        except Exception:  # noqa: BLE001
            traceback.print_exc()

        return "; ".join(notes)

    # -- reset / send ----------------------------------------------------------
    def _reset(self):
        self._flash_button(self.opts.btn_reset)
        if self.df_api is None:
            self.page.show_empty()
            return
        # Retrieve the original (loaded) dataframe  -- no re-read of the file, same
        # as the legacy GUI's reset_button_api. df_api is the untouched load; the
        # position columns are re-derived from it below.
        self.df_current = self.df_api.copy()
        self.df_previous = self.df_api.copy()
        self.en_flag = self.dt_flag = self.xy_flag = 0
        self._cut_energy = self._cut_time = self._cut_xy = None
        self._undo_state = None
        self.opts.btn_undo.setEnabled(False)
        self.vmax = None
        self.opts.ed_vmax.clear()
        self._reset_selections()
        self._configure_keys()
        self._initialize_plots()
        self._status("API view reset")

    # Bright fill for the click-blink, keyed by button style (objectName): each is
    # a brightened version of that button's *own* accent hue, so the blink keeps
    # the button's colour instead of switching to a different one.
    _FLASH_BG = {
        "open_btn":    "#c4a8ff",      # purple
        "primary_btn": T.ACCENT_CYAN,  # cyan
        "danger_btn":  T.ACCENT_RED,   # red
        "yellow_btn":  T.ACCENT_AMBER, # amber
        "find_btn":    "#9fb0ff",      # blue
        "mini_btn":    T.ACCENT_CYAN,
        "action_btn":  T.ACCENT_CYAN,
    }

    def _flash_button(self, button):
        """Briefly brighten a button -- a solid, bright version of its own colour
        -- so the user sees the click registered, then revert to its themed style.
        Only the colours are overridden (padding/radius come from the theme, so
        the button keeps its size), and ``repaint()`` paints the bright state
        *now*, before any blocking work freezes the event loop, so the flash
        reaches the screen."""
        bg = self._FLASH_BG.get(button.objectName(), T.ACCENT_CYAN)
        button.setStyleSheet(
            f"background-color:{bg}; border-color:{bg}; "
            f"color:{T.BG_DARK}; font-weight:800;")
        button.repaint()
        QTimer.singleShot(220, lambda: button.setStyleSheet(""))

    def _send_to_spectrum(self):
        if self.gam is None:
            # Normally populated when the energy panel draws; if no draw has
            # happened yet, bin it on demand so the first click sends instead of
            # silently doing nothing (and needing a second press). Flood-field
            # runs have no energy spectrum, so there is genuinely nothing to send.
            if self.df_current is None or self.ekey is None or self.flood_field:
                self._status("Nothing to send  -- load an API file first")
                return
            self._compute_energy_hist(self.df_current)
        # Send exactly what the panel shows (see _axis_units): a calibration's
        # units, MeV for a native physical-energy axis, else raw channels. This
        # keeps the Spectrum-tab axis label consistent with the API panel.
        units = self._axis_units()
        try:
            if units is not None:
                spect = sp.Spectrum(counts=self.gam, energies=self.gam_x, e_units=units)
            else:
                # Uncalibrated: carry the *real* channel values (gam_x bin centres)
                # on adc_channels (a pure coordinate), leaving `channels` as 0..N
                # indices for the peak-finding/fitting machinery. A calibration
                # built here fits against spect.cal_channels  -- i.e. the real
                # channel space of the API dataframe's raw energy column  -- so its
                # coefficients apply to that column directly and "Retrieve
                # calibration" round-trips correctly.
                spect = sp.Spectrum(counts=self.gam, adc_channels=self.gam_x)
        except Exception as exc:  # noqa: BLE001
            self._status(f"Could not build spectrum: {exc}")
            return
        # Load it as the active spectrum but stay on the API tab; a brief green
        # blink confirms the send instead of yanking the user away.
        self.app.load_external_spectrum(spect, "API spectrum", switch_tab=False)
        self._flash_button(self.opts.btn_send)
        self._status("Sent energy spectrum to the Spectrum tab")
