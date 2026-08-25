"""Planetary tab: NASA planetary gamma-ray / neutron spectroscopy.

First mission: **Lunar Prospector GRS** (see :mod:`wara.planetary.lp`).

Layout (like the other tabs, Options / Page / Controller trio):

* OPTIONS — mission picker, PDS search (date range + orbit phase), download,
  load-into-memory, region selection size, globe detail.
* PAGE — vertical split: a fully interactive high-resolution 3D Moon (Plotly
  in a ``QWebEngineView``, ~3/4 of the height) above a matplotlib spectrum
  canvas (~1/4) showing the summed LP-GRS gamma spectrum for the selected
  region.

The globe is the correctly-georeferenced sphere from
:mod:`wara.planetary.moon` rendered at high mesh resolution with the bundled
2048x1024 NASA/USGS albedo map. Hover shows lon/lat via a JavaScript overlay
(cheaper than per-vertex hover strings at this mesh density); clicking the
surface reports the coordinate back to Python through the page-title bridge
(``document.title = "lonlat:..."`` → ``QWebEngineView.titleChanged``), which
selects a lat/lon box, sums every loaded record inside it, and plots the
spectrum below. The selection box itself is drawn in-page with
``Plotly.addTraces`` so the (large) globe HTML never has to reload.

QtWebEngine is imported lazily and only when the tab is first activated, so
the offscreen test suite never loads it.
"""
import tempfile
from pathlib import Path

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSizePolicy,
    QScrollArea, QCheckBox, QLineEdit, QFileDialog, QSplitter, QDialog,
    QListWidget, QTextBrowser,
)
from PyQt5.QtCore import Qt, QSize, QUrl, QThread, QObject, QTimer, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

from . import theme as T
from .widgets import header, hsep, labeled_row, ComboBox, DoubleSpinBox

PLANET_PLOT_BG = T.BG_PLOT

# Globe mesh resolution per detail setting. The visual sharpness of a Plotly
# surface equals its *mesh* density (the texture is sampled at the vertices),
# so "high resolution" means a dense mesh + the 2048-px bundled texture.
GLOBE_DETAIL = {
    "Standard (1.0°)": (360, 180),
    "High (0.5°)": (720, 360),
    "Ultra (0.35°)": (1024, 512),
}
GLOBE_DETAIL_DEFAULT = "Ultra (0.35°)"

# Downloads live inside the package's planetary data folder (gitignored), so
# datasets a user decides to keep survive; the bundled orbit-metadata CSV
# (tracked in git) sits alongside them.
from wara.planetary.lp import LP_DATA_DIR as DEFAULT_DATA_DIR  # noqa: E402

# Colors for "kept" region spectra; the box on the Moon and the spectrum line
# share the color. Cyan is reserved for the active selection, grey for the
# all-data comparison curve.
KEEP_COLORS = [T.ACCENT_AMBER, T.ACCENT_GREEN, T.ACCENT_RED, "#b388ff",
               "#ff8a3d", "#ffd166", "#00bfa5", "#f06292"]

# Floor on the orbit-path overlay: subsampling never thins a selection below
# this many points (or all of them, if there are fewer), so even multi-month
# selections stay legible. Selections up to ~18 days (~2700 records/day) are
# drawn uncut; larger ones are thinned only down toward this floor.
MIN_TRACK_POINTS = 50000

# Colormaps offered for the abundance maps. Restricted to plotly.js built-in
# colorscale names (the drape is applied client-side via Plotly.restyle):
# perceptually-uniform first, then the classics.
ABUNDANCE_COLORSCALES = ["Viridis", "Cividis", "Jet", "Hot", "Portland",
                         "Electric", "Blackbody"]


# ── Background worker ─────────────────────────────────────────────────────────
class _Worker(QThread):
    """Run ``fn(progress_callback)`` off the GUI thread.

    ``fn`` receives a ``progress(str)`` callable and returns a result object,
    delivered via :attr:`done`. Exceptions are surfaced through :attr:`failed`.
    """

    progress = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):  # noqa: D102
        try:
            self.done.emit(self._fn(self.progress.emit))
        except Exception as exc:  # noqa: BLE001 — report any failure to the GUI
            self.failed.emit(str(exc))


# ── Page (globe above spectrum) ───────────────────────────────────────────────
class PlanetaryPage(QWidget):
    """Plot area for the Planetary tab: 3D globe (top ~3/4) over the regional
    spectrum canvas (bottom ~1/4)."""

    def __init__(self):
        super().__init__()
        self.setObjectName("content")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Vertical)

        # Top: globe. A dark placeholder until the tab is first activated —
        # QtWebEngine is only imported/created then (ensure_web_view).
        self.web_view = None
        self.globe_holder = QWidget()
        self.globe_holder.setStyleSheet(f"background:{PLANET_PLOT_BG};")
        holder_lay = QVBoxLayout(self.globe_holder)
        holder_lay.setContentsMargins(0, 0, 0, 0)
        self.globe_placeholder = QLabel("Loading the Moon…")
        self.globe_placeholder.setAlignment(Qt.AlignCenter)
        self.globe_placeholder.setStyleSheet(
            f"color:{T.TEXT_DIM}; font-size:15px; font-weight:600;")
        holder_lay.addWidget(self.globe_placeholder)
        self.splitter.addWidget(self.globe_holder)

        # Bottom: spectrum canvas + toolbar + readout.
        bottom = QWidget()
        blay = QVBoxLayout(bottom)
        blay.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(constrained_layout=True, facecolor=PLANET_PLOT_BG)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        row = QHBoxLayout()
        self.toolbar = NavToolbar(self.canvas, self)
        self.toolbar.setObjectName("plot_toolbar")
        self.toolbar.setIconSize(QSize(18, 18))
        T.recolor_toolbar_icons(self.toolbar, T.TEXT_PRIMARY)
        self.readout = QLabel("")
        self.readout.setStyleSheet(
            f"background:{PLANET_PLOT_BG}; color:{T.ACCENT_CYAN};"
            "font-size:13px; font-weight:600; padding:2px 8px;")
        row.addWidget(self.toolbar, 0)
        row.addWidget(self.readout, 1)
        blay.addLayout(row, 0)
        blay.addWidget(self.canvas, 1)
        self.splitter.addWidget(bottom)

        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([700, 240])
        lay.addWidget(self.splitter)

        self.show_spectrum_empty()

    # -- globe -------------------------------------------------------------
    def ensure_web_view(self):
        """Create the ``QWebEngineView`` on first use; return it (or ``None``
        if QtWebEngine is unavailable, e.g. in the offscreen test suite)."""
        if self.web_view is not None:
            return self.web_view
        try:
            from PyQt5.QtWebEngineWidgets import QWebEngineView
            from PyQt5.QtGui import QColor
        except ImportError:
            self.globe_placeholder.setText(
                "QtWebEngine is not installed — the 3D globe is unavailable.")
            return None
        self.web_view = QWebEngineView()
        self.web_view.page().setBackgroundColor(QColor(PLANET_PLOT_BG))
        self.globe_holder.layout().addWidget(self.web_view)
        self.globe_placeholder.hide()
        return self.web_view

    # -- spectrum ----------------------------------------------------------
    def show_spectrum_empty(self, msg="Load LP-GRS data, then click the Moon "
                                      "to see the regional gamma spectrum"):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=T.BORDER, fontweight="bold", wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        self._restyle()
        self.canvas.draw_idle()

    def show_spectrum(self, spectrum=None, title="", compare=None, kept=None):
        """Plot summed spectra (log-y). ``spectrum``/``title`` is the active
        selection (cyan; may be ``None`` when only kept spectra remain),
        ``compare`` an optional (label, spectrum) pair (e.g. the all-data
        average, grey), and ``kept`` a list of (label, spectrum, color) pinned
        region spectra whose colors match their boxes on the Moon."""
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        if compare is not None:
            label, comp = compare
            ax.plot(comp, drawstyle="steps-mid", color=T.TEXT_DIM, lw=1.2,
                    alpha=0.8, label=label)
        for label, spec, color in (kept or []):
            ax.plot(spec, drawstyle="steps-mid", color=color, lw=1.3,
                    label=label)
        if spectrum is not None:
            ax.plot(spectrum, drawstyle="steps-mid", color=T.ACCENT_CYAN,
                    lw=1.4, label=title)
        ax.set_yscale("log")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Counts")
        ax.legend(loc="upper right", fontsize=12)
        self._restyle()
        self.canvas.draw_idle()
        self.toolbar.update()
        self.toolbar.push_current()

    def _restyle(self):
        self.fig.set_facecolor(PLANET_PLOT_BG)
        for ax in self.fig.axes:
            ax.set_facecolor(PLANET_PLOT_BG)
            ax.tick_params(colors=T.TEXT_DIM, which="both", length=3)
            ax.xaxis.label.set_color(T.TEXT_DIM)
            ax.yaxis.label.set_color(T.TEXT_DIM)
            for sp_ in ax.spines.values():
                sp_.set_color(T.BORDER)
            ax.grid(True, color="#3c3c66", linewidth=0.7, alpha=0.8)
            leg = ax.get_legend()
            if leg is not None:
                leg.get_frame().set_facecolor(T.BG_PANEL)
                leg.get_frame().set_edgecolor(T.BORDER)
                for txt in leg.get_texts():
                    txt.set_color(T.TEXT_PRIMARY)


# ── Options column ────────────────────────────────────────────────────────────
class PlanetaryOptions(QScrollArea):
    """Scrollable options for the Planetary tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(9, 12, 9, 12); lay.setSpacing(6)

        title = QLabel("PLANETARY"); title.setObjectName("opt_title")
        lay.addWidget(title); lay.addWidget(hsep())

        def _compact(combo, chars=12):
            """Stop a combo's longest item from dictating the panel's minimum
            width: it sizes to ~``chars`` and elides the current text instead
            (the fixed-width options column clips anything wider)."""
            from PyQt5.QtWidgets import QComboBox
            combo.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(chars)
            return combo

        lay.addWidget(header("MISSION"))
        self.mission = _compact(ComboBox(), 18)
        self.mission.addItems(["Lunar Prospector GRS"])
        self.mission.setToolTip("NASA mission / instrument (more coming)")
        lay.addWidget(self.mission)

        self.dataset = _compact(ComboBox(), 16)
        self.dataset.addItems(["Raw", "Calibrated", "Elevation (LOLA)"])
        self.dataset.setToolTip(
            "Raw: the Level-3 daily spectra (this tab's spectrum workflow).\n"
            "Calibrated: the derived Level-5 elemental-abundance maps "
            "(lp-l-grs-5-elem-abundance-v1), draped over the Moon.\n"
            "Elevation: LRO LOLA global topography draped as a color map")
        row, _ = labeled_row("Data", self.dataset); lay.addWidget(row)

        from wara.planetary.abundance import ABUNDANCE_ELEMENTS
        self.element = _compact(ComboBox(), 8)
        self.element.addItems(list(ABUNDANCE_ELEMENTS))
        self.element.setToolTip(
            "Element/oxide of the abundance map (Th/K/U in ppm, oxides in "
            "weight fraction)")
        self.element.setEnabled(False)
        row, _ = labeled_row("Element", self.element); lay.addWidget(row)

        self.resolution = _compact(ComboBox(), 5)
        self.resolution.addItems(["2°", "5°", "20°"])
        self.resolution.setToolTip("Map pixel size (equal-area binning)")
        self.resolution.setEnabled(False)
        row, _ = labeled_row("Map pixels", self.resolution); lay.addWidget(row)

        self.cmap = _compact(ComboBox(), 8)
        self.cmap.addItems(list(ABUNDANCE_COLORSCALES))
        self.cmap.setToolTip("Colormap of the abundance map draped on the Moon")
        self.cmap.setEnabled(False)
        row, _ = labeled_row("Colormap", self.cmap); lay.addWidget(row)

        self.opacity = DoubleSpinBox()
        self.opacity.setRange(10.0, 100.0); self.opacity.setValue(100.0)
        self.opacity.setDecimals(0); self.opacity.setSingleStep(10.0)
        self.opacity.setSuffix(" %")
        self.opacity.setToolTip(
            "Opacity of the composition map. Below 100 % it floats as a "
            "semi-transparent layer over the grey albedo Moon, so the base "
            "shows through (including the 3D topography relief if enabled)")
        self.opacity.setEnabled(False)
        row, _ = labeled_row("Opacity", self.opacity); lay.addWidget(row)

        self.btn_info = QPushButton("Mission info…")
        self.btn_info.setObjectName("action_btn")
        self.btn_info.setToolTip(
            "The mission's PDS documentation: mission/spacecraft/instrument "
            "descriptions, data-set documents, and references")
        self.btn_info.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_info)

        self.cb_topo = QCheckBox("3D topography (LOLA)")
        self.cb_topo.setToolTip(
            "Displace the globe surface by the LRO LOLA elevation model — "
            "real 3D relief. Works with every dataset drape, so an abundance "
            "map over the relief correlates composition with topography")
        lay.addWidget(self.cb_topo)
        self.exag = DoubleSpinBox()
        self.exag.setRange(1.0, 50.0); self.exag.setValue(10.0)
        self.exag.setDecimals(0); self.exag.setSuffix(" ×")
        self.exag.setToolTip(
            "Vertical exaggeration of the relief (real lunar topography "
            "spans about ±9 km on a 1737 km sphere — invisible at 1×)")
        self.exag.setEnabled(False)
        row, _ = labeled_row("Exaggeration", self.exag); lay.addWidget(row)

        lay.addWidget(hsep()); lay.addWidget(header("PDS DATA"))
        # Filled from the bundled orbit metadata when the tab is activated.
        self.lbl_avail = QLabel("")
        self.lbl_avail.setObjectName("stat_key"); self.lbl_avail.setWordWrap(True)
        self.lbl_avail.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(self.lbl_avail)
        self.ed_start = QLineEdit("1998-01-16")
        self.ed_start.setToolTip("First measurement date (YYYY-MM-DD)")
        row, _ = labeled_row("Start", self.ed_start); lay.addWidget(row)
        self.ed_end = QLineEdit("1998-01-31")
        self.ed_end.setToolTip("Last measurement date (YYYY-MM-DD, inclusive)")
        row, _ = labeled_row("End", self.ed_end); lay.addWidget(row)

        self.phase = _compact(ComboBox())
        self.phase.addItems(["All altitudes", "High (~100 km)", "Low (~30-40 km)"])
        self.phase.setToolTip(
            "Orbit phase: the ~100 km mapping orbit (Jan-Dec 1998) or the "
            "low-altitude extended mission (Dec 1998 - Jul 1999)")
        row, _ = labeled_row("Orbit", self.phase); lay.addWidget(row)

        self.btn_search = QPushButton("Search PDS")
        self.btn_search.setObjectName("open_btn")
        self.btn_search.setToolTip(
            "List the matching daily products in the NASA PDS archive")
        self.btn_search.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_search)

        self.btn_download = QPushButton("Download")
        self.btn_download.setObjectName("action_btn")
        self.btn_download.setToolTip(
            "Download the matching products (~1 MB per day; cached, already-"
            "downloaded files are skipped)")
        self.btn_download.setCursor(Qt.PointingHandCursor)
        self.btn_download.setEnabled(False)
        lay.addWidget(self.btn_download)

        self.btn_load = QPushButton("Load into memory")
        self.btn_load.setObjectName("action_btn")
        self.btn_load.setToolTip(
            "Read every downloaded product in the date range; their records "
            "become selectable on the globe")
        self.btn_load.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_load)

        self.lbl_dir = QLabel()
        self.lbl_dir.setObjectName("stat_key"); self.lbl_dir.setWordWrap(True)
        # A path is one long unbreakable token — never let it dictate the
        # fixed-width panel's minimum width (it would clip every other row).
        self.lbl_dir.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.set_data_dir_label(DEFAULT_DATA_DIR)
        lay.addWidget(self.lbl_dir)
        self.btn_dir = QPushButton("Data folder…")
        self.btn_dir.setObjectName("action_btn")
        self.btn_dir.setToolTip("Where downloaded LP-GRS products are stored")
        self.btn_dir.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_dir)

        lay.addWidget(hsep()); lay.addWidget(header("REGION"))
        self.btn_select = QPushButton("Select region")
        self.btn_select.setObjectName("open_btn")
        self.btn_select.setCheckable(True)
        self.btn_select.setToolTip(
            "Arm region selection: the next click on the Moon selects a "
            "lat/lon box (and disarms this button again). While disarmed, "
            "clicking the globe only rotates it.")
        self.btn_select.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_select)
        self.box_size = DoubleSpinBox()
        self.box_size.setRange(1.0, 90.0); self.box_size.setValue(10.0)
        self.box_size.setSuffix(" °"); self.box_size.setDecimals(1)
        self.box_size.setToolTip(
            "Half-width of the selection box around the clicked point "
            "(degrees of latitude/longitude)")
        row, _ = labeled_row("Box half-width", self.box_size); lay.addWidget(row)

        self.cb_compare = QCheckBox("Compare to all data")
        self.cb_compare.setChecked(True)
        self.cb_compare.setToolTip(
            "Overlay the per-record average of every loaded record, scaled to "
            "the region's record count")
        lay.addWidget(self.cb_compare)

        self.cb_keep = QCheckBox("Keep spectra")
        self.cb_keep.setToolTip(
            "While checked, selecting a new region keeps the previous "
            "spectrum visible for comparison; each kept spectrum and its box "
            "on the Moon share a color. Untick to drop the kept spectra.")
        lay.addWidget(self.cb_keep)

        self.btn_send = QPushButton("Send to spectrum")
        self.btn_send.setObjectName("action_btn")
        self.btn_send.setToolTip(
            "Hand the selected region's summed spectrum to the Spectrum tab "
            "for peak finding / fitting")
        self.btn_send.setCursor(Qt.PointingHandCursor)
        self.btn_send.setEnabled(False)
        lay.addWidget(self.btn_send)

        self.btn_clear_sel = QPushButton("Clear selection")
        self.btn_clear_sel.setObjectName("action_btn")
        self.btn_clear_sel.setToolTip(
            "Remove the active selection and every kept region/spectrum")
        self.btn_clear_sel.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_clear_sel)

        lay.addWidget(hsep()); lay.addWidget(header("GLOBE"))
        self.detail = _compact(ComboBox())
        self.detail.addItems(list(GLOBE_DETAIL))
        self.detail.setCurrentText(GLOBE_DETAIL_DEFAULT)
        self.detail.setToolTip(
            "Globe mesh density. Finer = sharper Moon, larger render. "
            "Changing it rebuilds the globe (overlays are redrawn)")
        row, _ = labeled_row("Detail", self.detail); lay.addWidget(row)
        self.cb_flat = QCheckBox("2D map (equirectangular)")
        self.cb_flat.setToolTip(
            "Project the Moon onto a flat longitude/latitude map "
            "(orthographic top-down view). All overlays and the region "
            "selection keep working; 3D relief is ignored while flat")
        lay.addWidget(self.cb_flat)
        self.cb_grid = QCheckBox("Lat/lon grid")
        lay.addWidget(self.cb_grid)
        self.cb_marks = QCheckBox("Landmarks")
        self.cb_marks.setToolTip(
            "Label important reference points on the Moon (major maria, "
            "landmark craters, far-side basins, the poles, Apollo 11)")
        lay.addWidget(self.cb_marks)
        self.cb_track = QCheckBox("Show orbit path")
        self.cb_track.setToolTip(
            "Overlay the spacecraft ground track on the Moon, colored by "
            "measurement time (hover a point for the exact date, position, "
            "and altitude). Uses the loaded records, or the bundled "
            "whole-mission metadata when nothing is loaded")
        lay.addWidget(self.cb_track)

        self.status = QLabel("")
        self.status.setObjectName("stat_key"); self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(self.status)

        lay.addStretch(1)
        self.setWidget(inner)

    def set_data_dir_label(self, path):
        """Show the data folder compactly (middle-elided); the tooltip carries
        the full path."""
        from PyQt5.QtGui import QFontMetrics

        text = str(path)
        fm = QFontMetrics(self.lbl_dir.font())
        self.lbl_dir.setText(fm.elidedText(text, Qt.ElideMiddle, 210))
        self.lbl_dir.setToolTip(text)


# ── Mission info dialog ───────────────────────────────────────────────────────
class MissionInfoDialog(QDialog):
    """Browser for the mission's PDS documentation: document list on the left,
    the (80-column plain-text) document on the right, and a link to the PDS
    mission page. The controller fills :attr:`docs` once fetched."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from wara.planetary import LP_DOCUMENTS, LP_MISSION_PAGE_URL

        self.setWindowTitle("Lunar Prospector — Mission info")
        self.setStyleSheet(T.STYLESHEET)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.resize(980, 640)
        self.docs = {}                       # label -> text

        outer = QVBoxLayout(self)
        link = QLabel(
            f'Documentation from the NASA PDS Geosciences node — '
            f'<a href="{LP_MISSION_PAGE_URL}" style="color:{T.ACCENT_CYAN}">'
            f'open the LP reduced GRS/NS mission page</a>')
        link.setOpenExternalLinks(True)
        link.setObjectName("stat_key")
        outer.addWidget(link)

        row = QHBoxLayout(); row.setSpacing(8)
        self.doc_list = QListWidget()
        self.doc_list.addItems(list(LP_DOCUMENTS))
        self.doc_list.setFixedWidth(230)
        self.doc_list.currentTextChanged.connect(self._show)
        row.addWidget(self.doc_list, 0)

        self.viewer = QTextBrowser()
        self.viewer.setOpenExternalLinks(True)
        self.viewer.setStyleSheet(
            f"background:{T.BG_PLOT}; color:{T.TEXT_PRIMARY};"
            f"border:1px solid {T.BORDER};"
            "font-family:Consolas,'Cascadia Mono',monospace; font-size:12px;")
        self.viewer.setPlainText("Fetching the mission documentation…")
        row.addWidget(self.viewer, 1)
        outer.addLayout(row, 1)

    def set_docs(self, docs):
        """Receive the fetched {label: text} dict and show the selection."""
        self.docs = docs
        current = self.doc_list.currentItem()
        self._show(current.text() if current else self.doc_list.item(0).text())
        if current is None:
            self.doc_list.setCurrentRow(0)

    def _show(self, label):
        if not label:
            return
        text = self.docs.get(label)
        if text is None:
            self.viewer.setPlainText("Fetching the mission documentation…")
        else:
            self.viewer.setPlainText(text)


# ── Globe HTML (Plotly + JS hover/click bridge) ──────────────────────────────
_GLOBE_JS = """
<div id="wara-readout" style="position:fixed; top:10px; left:12px; z-index:10;
     color:#f4f6ff; background:rgba(10,10,15,0.75); border:1px solid #2a2a45;
     border-radius:4px; padding:4px 10px;
     font:600 13px 'Segoe UI',sans-serif;">lon —  lat —</div>
<script>
(function () {
  var gd = document.querySelector('.plotly-graph-div');
  var readout = document.getElementById('wara-readout');
  var nclick = 0;
  /* plotly.py >= 5.24 serialises numpy arrays as base64 "typed array specs"
     ({dtype, bdata, shape}) instead of nested JSON arrays, so gd.data[0].x
     is an *object* with no .length. Every helper below that re-derives
     geometry from the base surface must decode first, or it silently
     iterates zero times and builds an empty (and therefore invisible)
     trace. grid2d() accepts either form and always returns rows of
     numbers; it is idempotent, so it is safe on already-decoded data. */
  var DTYPES = {f8: Float64Array, f4: Float32Array,
                i1: Int8Array, u1: Uint8Array,
                i2: Int16Array, u2: Uint16Array,
                i4: Int32Array, u4: Uint32Array};
  function grid2d(v) {
    if (v === null || v === undefined) { return v; }
    if (Array.isArray(v) && Array.isArray(v[0])) { return v; }
    if (v.bdata === undefined) { return v; }
    var T = DTYPES[v.dtype];
    if (T === undefined) { return v; }
    var bin = atob(v.bdata), buf = new Uint8Array(bin.length);
    for (var k = 0; k < bin.length; k++) { buf[k] = bin.charCodeAt(k); }
    var flat = new T(buf.buffer);
    var shape = ('' + v.shape).split(',');
    if (shape.length < 2) { return Array.prototype.slice.call(flat); }
    var nrow = +shape[0], ncol = +shape[1], out = [];
    for (var i = 0; i < nrow; i++) {
      var row = [];
      for (var j = 0; j < ncol; j++) { row.push(flat[i * ncol + j]); }
      out.push(row);
    }
    return out;
  }
  function lonlat(p) {
    var r = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    var lat = Math.asin(p.z / r) * 180 / Math.PI;
    var lon = Math.atan2(p.y, p.x) * 180 / Math.PI;
    return [lon, lat];
  }
  gd.on('plotly_hover', function (d) {
    var p = d.points[0];
    if (p === undefined || p.x === undefined) return;
    var ll = lonlat(p);
    readout.textContent =
      'lon ' + ll[0].toFixed(1) + '\\u00b0  lat ' + ll[1].toFixed(1) + '\\u00b0';
  });
  gd.on('plotly_unhover', function () {
    readout.textContent = 'lon \\u2014  lat \\u2014';
  });
  /* Click -> Python bridge: QWebEngineView watches titleChanged. The counter
     makes repeated clicks on the same spot produce distinct titles. */
  gd.on('plotly_click', function (d) {
    var p = d.points[0];
    if (p === undefined || p.x === undefined) return;
    var ll = lonlat(p);
    nclick += 1;
    document.title =
      'lonlat:' + ll[0].toFixed(3) + ',' + ll[1].toFixed(3) + ':' + nclick;
  });
  /* Box helpers, called from Python via runJavaScript. Traces are tagged by
     name ('wara-sel' = the active cyan box, 'wara-kept' = pinned colored
     boxes) so no index bookkeeping is needed across adds/deletes. */
  function boxIdx(names) {
    var out = [];
    gd.data.forEach(function (t, i) {
      if (names.indexOf(t.name) !== -1) out.push(i);
    });
    return out;
  }
  function addBox(x, y, z, color, name) {
    Plotly.addTraces(gd, {type: 'scatter3d', mode: 'lines',
      x: x, y: y, z: z, line: {color: color, width: 5},
      name: name, hoverinfo: 'none', showlegend: false});
  }
  window.waraShowBox = function (x, y, z) {
    var idx = boxIdx(['wara-sel']);
    var doAdd = function () { addBox(x, y, z, '#00e5ff', 'wara-sel'); };
    if (idx.length) { Plotly.deleteTraces(gd, idx).then(doAdd); }
    else { doAdd(); }
  };
  window.waraKeepBox = function (x, y, z, color) {
    addBox(x, y, z, color, 'wara-kept');
  };
  window.waraClearSel = function () {
    var idx = boxIdx(['wara-sel']);
    if (idx.length) Plotly.deleteTraces(gd, idx);
  };
  window.waraClearKept = function () {
    var idx = boxIdx(['wara-kept']);
    if (idx.length) Plotly.deleteTraces(gd, idx);
  };
  window.waraClearBoxes = function () {
    var idx = boxIdx(['wara-sel', 'wara-kept']);
    if (idx.length) Plotly.deleteTraces(gd, idx);
  };
  /* Orbit ground track: one point per record, colored by measurement time.
     The colorbar title runs along the bar (side:'right') so it cannot be
     clipped by the plot's right edge. */
  window.waraShowTrack = function (x, y, z, c, text, title) {
    window.waraClearTrack();
    Plotly.addTraces(gd, {type: 'scatter3d', mode: 'markers',
      x: x, y: y, z: z,
      marker: {size: 2.5, color: c, colorscale: 'Viridis',
               colorbar: {title: {text: title, side: 'right',
                                  font: {color: '#aab2e0', size: 13}},
                          tickfont: {color: '#aab2e0', size: 12},
                          thickness: 12, len: 0.55,
                          x: 0.94, xanchor: 'left',
                          outlinecolor: '#2a2a45'}},
      name: 'wara-track', text: text, hoverinfo: 'text', showlegend: false});
  };
  window.waraClearTrack = function () {
    var idx = boxIdx(['wara-track']);
    if (idx.length) Plotly.deleteTraces(gd, idx);
  };
  /* Named lunar reference points (maria, craters, poles, ...). Two traces:
     dots sit on the surface, the text floats on a higher shell so labels
     are never occluded by the Moon itself. */
  window.waraShowMarks = function (xd, yd, zd, xt, yt, zt, names, hover) {
    window.waraClearMarks();
    Plotly.addTraces(gd, [
      {type: 'scatter3d', mode: 'markers', x: xd, y: yd, z: zd,
       marker: {size: 3.5, color: '#39ff14'},
       name: 'wara-marks', hovertext: hover, hoverinfo: 'text',
       showlegend: false},
      {type: 'scatter3d', mode: 'text', x: xt, y: yt, z: zt,
       text: names.map(function (n) { return '<b>' + n + '</b>'; }),
       textposition: 'middle center',
       textfont: {color: '#39ff14', size: 12},
       name: 'wara-marks', hoverinfo: 'skip', showlegend: false}
    ]);
  };
  /* Show/hide the lat-lon graticule (traces named at build time). */
  window.waraSetGrid = function (v) {
    var idx = boxIdx(['wara-grat']);
    if (idx.length) Plotly.restyle(gd, {visible: v}, idx);
  };
  /* Recolor the Moon surface with a data map (elemental abundance); the
     original albedo AND its resolved colorscale are stashed on first use and
     restored by reset (the Python-side name 'gray' is not a plotly.js
     colorscale, so the resolved [t, color] array must be kept). */
  window.waraSetSurface = function (values, cmin, cmax, title, colorscale) {
    if (window._waraAlbedo === undefined) {
      window._waraAlbedo = grid2d(gd.data[0].surfacecolor);
      window._waraAlbedoScale = gd._fullData[0].colorscale;
    }
    Plotly.restyle(gd, {surfacecolor: [values],
      colorscale: colorscale || 'Viridis',
      cmin: cmin, cmax: cmax, showscale: true,
      'colorbar.title.text': title, 'colorbar.title.side': 'right',
      'colorbar.title.font': {color: '#aab2e0', size: 13},
      'colorbar.tickfont': {color: '#aab2e0', size: 12},
      'colorbar.thickness': 12, 'colorbar.len': 0.55,
      'colorbar.x': 0.88, 'colorbar.xanchor': 'left',
      'colorbar.outlinecolor': '#2a2a45'}, [0]);
  };
  window.waraResetSurface = function () {
    if (window._waraAlbedo === undefined) return;
    Plotly.restyle(gd, {surfacecolor: [window._waraAlbedo],
      colorscale: [window._waraAlbedoScale],
      cmin: 0, cmax: 1, showscale: false}, [0]);
  };
  /* 3D topography: displace every surface vertex radially by the LOLA
     elevation (km; same grid shape as the mesh) times an exaggeration
     factor. The original vertex positions are stashed on first use and the
     elevation grid is cached, so changing the exaggeration re-uses it
     (pass null for elev). A semi-transparent composition overlay, if any,
     is rebuilt afterwards so it follows the relief. */
  window.waraSetTopo = function (elev, exag) {
    var t = gd.data[0];
    if (window._waraXYZ === undefined) {
      window._waraXYZ = {x: grid2d(t.x), y: grid2d(t.y), z: grid2d(t.z)};
    }
    if (elev !== null) { window._waraElev = elev; }
    var e = window._waraElev;
    if (e === undefined) { return; }
    var R = 1737.4, o = window._waraXYZ;
    var nx = [], ny = [], nz = [];
    for (var i = 0; i < e.length; i++) {
      var rx = [], ry = [], rz = [];
      for (var j = 0; j < e[i].length; j++) {
        var s = 1 + exag * e[i][j] / R;
        rx.push(o.x[i][j] * s);
        ry.push(o.y[i][j] * s);
        rz.push(o.z[i][j] * s);
      }
      nx.push(rx); ny.push(ry); nz.push(rz);
    }
    Plotly.restyle(gd, {x: [nx], y: [ny], z: [nz]}, [0]).then(function () {
      if (window._waraOverlay !== undefined) { window.waraSetOverlay(null); }
    });
  };
  window.waraResetTopo = function () {
    if (window._waraXYZ === undefined) { return; }
    Plotly.restyle(gd, {x: [window._waraXYZ.x], y: [window._waraXYZ.y],
      z: [window._waraXYZ.z]}, [0]).then(function () {
      if (window._waraOverlay !== undefined) { window.waraSetOverlay(null); }
    });
  };
  /* Semi-transparent composition overlay: a second surface floating just
     above the base surface (which keeps showing the albedo Moon or the
     LOLA elevation colors underneath). Its geometry is derived from the
     base's *current* vertices, so it follows any topography displacement.
     Parameters are cached; waraSetOverlay(null) rebuilds from the cache. */
  window.waraSetOverlay = function (values, cmin, cmax, title, cs, opacity) {
    if (values !== null && values !== undefined) {
      window._waraOverlay = {v: values, cmin: cmin, cmax: cmax,
                             title: title, cs: cs, op: opacity};
    }
    var ov = window._waraOverlay;
    if (ov === undefined) { return; }
    window.waraClearOverlay(true);
    var f = 1.002;
    var bx = grid2d(gd.data[0].x), by = grid2d(gd.data[0].y),
        bz = grid2d(gd.data[0].z);
    var sx = [], sy = [], sz = [];
    for (var i = 0; i < bx.length; i++) {
      var rx = [], ry = [], rz = [];
      for (var j = 0; j < bx[i].length; j++) {
        rx.push(bx[i][j] * f);
        ry.push(by[i][j] * f);
        rz.push(bz[i][j] * f);
      }
      sx.push(rx); sy.push(ry); sz.push(rz);
    }
    Plotly.addTraces(gd, {type: 'surface', x: sx, y: sy, z: sz,
      surfacecolor: ov.v, colorscale: ov.cs, cmin: ov.cmin, cmax: ov.cmax,
      opacity: ov.op, showscale: true,
      colorbar: {title: {text: ov.title, side: 'right',
                         font: {color: '#aab2e0', size: 13}},
                 tickfont: {color: '#aab2e0', size: 12},
                 thickness: 12, len: 0.55, x: 0.94, xanchor: 'left',
                 outlinecolor: '#2a2a45'},
      lighting: {ambient: 0.85, diffuse: 0.25, specular: 0.05},
      contours: {x: {highlight: false}, y: {highlight: false},
                 z: {highlight: false}},
      name: 'wara-abund', hoverinfo: 'none', showlegend: false});
  };
  window.waraClearOverlay = function (keepCache) {
    var idx = boxIdx(['wara-abund']);
    if (idx.length) { Plotly.deleteTraces(gd, idx); }
    if (!keepCache) { window._waraOverlay = undefined; }
  };
  window.waraClearMarks = function () {
    var idx = boxIdx(['wara-marks']);
    if (idx.length) Plotly.deleteTraces(gd, idx);
  };
})();
</script>
"""


def build_globe_html(n_lon, n_lat, graticule=True):
    """Full HTML for the interactive globe: the high-res Moon figure plus the
    JS hover readout / click bridge / selection-box helpers.

    The graticule is always *built into* the page (named ``wara-grat``) so the
    GUI checkbox can show/hide it instantly via ``waraSetGrid`` without a
    rebuild; ``graticule`` only sets its initial visibility."""
    from wara.planetary.moon import moon_figure, default_texture_path

    fig = moon_figure(texture=default_texture_path(2048),
                      n_lon=n_lon, n_lat=n_lat, graticule=True,
                      title="", hover=False, compact=True)
    # Everything after the surface is a graticule line trace.
    for trace in fig.data[1:]:
        trace.name = "wara-grat"
        trace.visible = bool(graticule)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    html = fig.to_html(include_plotlyjs="inline", full_html=True)
    html = html.replace(
        "<body>", f"<body style='margin:0;background:{PLANET_PLOT_BG}'>", 1)
    return html.replace("</body>", _GLOBE_JS + "</body>", 1)


# ── 2D map page (equirectangular, native-resolution texture) ─────────────────
# Unlike the globe (where visual sharpness is mesh-bound), the 2D page shows
# the albedo as a plotly layout *image* at the texture's native resolution —
# no mesh at all. It implements the exact same wara* JS API as the globe page
# (2D semantics: xyz inputs are converted to lon/lat), so the controller code
# is identical in both modes; only waraSetTopo/waraResetTopo are no-ops.
_FLAT_JS = """
<div id="wara-readout" style="position:fixed; top:10px; left:12px; z-index:10;
     color:#f4f6ff; background:rgba(10,10,15,0.75); border:1px solid #2a2a45;
     border-radius:4px; padding:4px 10px;
     font:600 13px 'Segoe UI',sans-serif;">lon —  lat —</div>
<script>
(function () {
  var gd = document.querySelector('.plotly-graph-div');
  var readout = document.getElementById('wara-readout');
  var nclick = 0;
  function idxOf(names) {
    var out = [];
    gd.data.forEach(function (t, i) {
      if (names.indexOf(t.name) !== -1) out.push(i);
    });
    return out;
  }
  function del(names) {
    var idx = idxOf(names);
    if (idx.length) Plotly.deleteTraces(gd, idx);
  }
  /* Overlay inputs arrive as spherical xyz (shared with the globe page). */
  function conv(x, y, z) {
    var lx = [], ly = [];
    for (var i = 0; i < x.length; i++) {
      var r = Math.sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
      lx.push(Math.atan2(y[i], x[i]) * 180 / Math.PI);
      ly.push(Math.asin(z[i] / r) * 180 / Math.PI);
    }
    return [lx, ly];
  }
  function axisVec(n, lo, hi) {
    var v = [];
    for (var i = 0; i < n; i++) { v.push(lo + (hi - lo) * i / (n - 1)); }
    return v;
  }
  gd.on('plotly_hover', function (d) {
    var p = d.points[0];
    if (p === undefined || p.x === undefined) return;
    readout.textContent =
      'lon ' + p.x.toFixed(1) + '\\u00b0  lat ' + p.y.toFixed(1) + '\\u00b0';
  });
  gd.on('plotly_unhover', function () {
    readout.textContent = 'lon \\u2014  lat \\u2014';
  });
  gd.on('plotly_click', function (d) {
    var p = d.points[0];
    if (p === undefined || p.x === undefined) return;
    nclick += 1;
    document.title =
      'lonlat:' + p.x.toFixed(3) + ',' + p.y.toFixed(3) + ':' + nclick;
  });
  window.waraShowBox = function (x, y, z) {
    window.waraClearSel();
    var m = conv(x, y, z);
    Plotly.addTraces(gd, {type: 'scatter', mode: 'lines',
      x: m[0], y: m[1], line: {color: '#00e5ff', width: 3},
      name: 'wara-sel', hoverinfo: 'none', showlegend: false});
  };
  window.waraKeepBox = function (x, y, z, color) {
    var m = conv(x, y, z);
    Plotly.addTraces(gd, {type: 'scatter', mode: 'lines',
      x: m[0], y: m[1], line: {color: color, width: 3},
      name: 'wara-kept', hoverinfo: 'none', showlegend: false});
  };
  window.waraClearSel = function () { del(['wara-sel']); };
  window.waraClearKept = function () { del(['wara-kept']); };
  window.waraClearBoxes = function () { del(['wara-sel', 'wara-kept']); };
  window.waraShowTrack = function (x, y, z, c, text, title) {
    window.waraClearTrack();
    var m = conv(x, y, z);
    Plotly.addTraces(gd, {type: 'scattergl', mode: 'markers',
      x: m[0], y: m[1],
      marker: {size: 4, color: c, colorscale: 'Viridis',
               colorbar: {title: {text: title, side: 'right',
                                  font: {color: '#aab2e0', size: 13}},
                          tickfont: {color: '#aab2e0', size: 12},
                          thickness: 12, len: 0.55,
                          x: 0.94, xanchor: 'left',
                          outlinecolor: '#2a2a45'}},
      name: 'wara-track', text: text, hoverinfo: 'text', showlegend: false});
  };
  window.waraClearTrack = function () { del(['wara-track']); };
  window.waraShowMarks = function (xd, yd, zd, xt, yt, zt, names, hover) {
    window.waraClearMarks();
    var m = conv(xd, yd, zd);
    Plotly.addTraces(gd, [
      {type: 'scatter', mode: 'markers', x: m[0], y: m[1],
       marker: {size: 6, color: '#39ff14'},
       name: 'wara-marks', hovertext: hover, hoverinfo: 'text',
       showlegend: false},
      {type: 'scatter', mode: 'text', x: m[0], y: m[1],
       text: names.map(function (n) { return '<b>' + n + '</b>'; }),
       textposition: 'top center',
       textfont: {color: '#39ff14', size: 12},
       name: 'wara-marks', hoverinfo: 'skip', showlegend: false}
    ]);
  };
  window.waraClearMarks = function () { del(['wara-marks']); };
  window.waraSetGrid = function (v) {
    Plotly.relayout(gd, {'xaxis.showgrid': !!v, 'yaxis.showgrid': !!v});
  };
  window.waraSetSurface = function (values, cmin, cmax, title, cs) {
    window.waraResetSurface();
    var xv = axisVec(values[0].length, -180, 180);
    var yv = axisVec(values.length, -90, 90);
    Plotly.addTraces(gd, {type: 'heatmap', x: xv, y: yv, z: values,
      colorscale: cs || 'Viridis', zmin: cmin, zmax: cmax, showscale: true,
      colorbar: {title: {text: title, side: 'right',
                         font: {color: '#aab2e0', size: 13}},
                 tickfont: {color: '#aab2e0', size: 12},
                 thickness: 12, len: 0.55, x: 0.88, xanchor: 'left',
                 outlinecolor: '#2a2a45'},
      name: 'wara-drape', hoverinfo: 'none', showlegend: false});
  };
  window.waraResetSurface = function () { del(['wara-drape']); };
  window.waraSetOverlay = function (values, cmin, cmax, title, cs, opacity) {
    window.waraClearOverlay();
    if (values === null || values === undefined) { return; }
    var xv = axisVec(values[0].length, -180, 180);
    var yv = axisVec(values.length, -90, 90);
    Plotly.addTraces(gd, {type: 'heatmap', x: xv, y: yv, z: values,
      colorscale: cs || 'Viridis', zmin: cmin, zmax: cmax,
      opacity: opacity, showscale: true,
      colorbar: {title: {text: title, side: 'right',
                         font: {color: '#aab2e0', size: 13}},
                 tickfont: {color: '#aab2e0', size: 12},
                 thickness: 12, len: 0.55, x: 0.94, xanchor: 'left',
                 outlinecolor: '#2a2a45'},
      name: 'wara-abund', hoverinfo: 'none', showlegend: false});
  };
  window.waraClearOverlay = function () { del(['wara-abund']); };
  /* No relief and no projection switching on the 2D page. */
  window.waraSetTopo = function () {};
  window.waraResetTopo = function () {};
})();
</script>
"""


def build_flat_html(texture=None, carrier=(720, 360)):
    """The 2D equirectangular map page.

    The Moon is a plotly layout *image* at the texture's native resolution
    (visual sharpness is not mesh-bound here). A fully transparent "carrier"
    heatmap at ``carrier`` resolution spans the map so hover/click events
    fire everywhere; drapes and the composition overlay are heatmaps added by
    the shared wara* JS API.
    """
    import base64
    import plotly.graph_objects as go

    from wara.planetary.moon import default_texture_path, load_texture_intensity

    tex = texture or default_texture_path(2048)
    uri = ("data:image/png;base64,"
           + base64.b64encode(Path(tex).read_bytes()).decode())

    n_lon, n_lat = carrier
    lon = np.linspace(-180.0, 180.0, n_lon)
    lat = np.linspace(-90.0, 90.0, n_lat)
    lon_g, lat_g = np.meshgrid(lon, lat)
    inten = load_texture_intensity(lon_g, lat_g, tex)
    if inten is None:
        inten = np.zeros_like(lon_g)
    carrier_trace = go.Heatmap(
        x=lon, y=lat, z=np.round(inten, 3), opacity=0.0, showscale=False,
        hoverinfo="none", name="wara-carrier", colorscale="Greys")

    axis_common = dict(
        dtick=30, gridcolor="rgba(255,255,255,0.18)", showgrid=False,
        zeroline=False, color=T.TEXT_DIM, constrain="domain",
        tickfont=dict(size=12, color=T.TEXT_DIM), linecolor=T.BORDER)
    fig = go.Figure(data=[carrier_trace])
    fig.update_layout(
        images=[dict(source=uri, xref="x", yref="y", x=-180, y=90,
                     sizex=360, sizey=180, xanchor="left", yanchor="top",
                     sizing="stretch", layer="below")],
        xaxis=dict(range=[-180, 180],
                   title=dict(text="Longitude (°E)",
                              font=dict(size=13, color=T.TEXT_DIM)),
                   **axis_common),
        yaxis=dict(range=[-90, 90], scaleanchor="x", scaleratio=1,
                   title=dict(text="Latitude (°)",
                              font=dict(size=13, color=T.TEXT_DIM)),
                   **axis_common),
        paper_bgcolor=PLANET_PLOT_BG, plot_bgcolor=PLANET_PLOT_BG,
        margin=dict(l=60, r=10, t=10, b=45), showlegend=False)
    html = fig.to_html(include_plotlyjs="inline", full_html=True)
    html = html.replace(
        "<body>", f"<body style='margin:0;background:{PLANET_PLOT_BG}'>", 1)
    return html.replace("</body>", _FLAT_JS + "</body>", 1)


def region_box_xyz(lon, lat, half_width, n=25):
    """Closed lat/lon-box outline around (``lon``, ``lat``), lifted just above
    the lunar surface. Returns ``(x, y, z)`` lists for the JS box trace. The
    box is clamped at the poles and follows parallels/meridians (not great
    circles), matching how records are selected."""
    from wara.planetary.moon import lonlat_to_xyz, R_MOON_KM

    lat_lo = max(lat - half_width, -90.0)
    lat_hi = min(lat + half_width, 90.0)
    lon_lo, lon_hi = lon - half_width, lon + half_width
    lons = np.linspace(lon_lo, lon_hi, n)
    lats = np.linspace(lat_lo, lat_hi, n)
    path_lon = np.concatenate([lons, np.full(n, lon_hi),
                               lons[::-1], np.full(n, lon_lo)])
    path_lat = np.concatenate([np.full(n, lat_lo), lats,
                               np.full(n, lat_hi), lats[::-1]])
    x, y, z = lonlat_to_xyz(path_lon, path_lat, R_MOON_KM * 1.006)
    return np.round(x, 1).tolist(), np.round(y, 1).tolist(), np.round(z, 1).tolist()


# ── Controller ────────────────────────────────────────────────────────────────
class PlanetaryController(QObject):
    """Wires PlanetaryOptions + PlanetaryPage to the PDS backend
    (:mod:`wara.planetary.lp`) and the interactive globe.

    Unlike the other tab controllers this one is a ``QObject``: its slots
    receive signals emitted from worker threads, and PyQt only delivers those
    safely (queued into the GUI thread) when the receiver is a ``QObject``.
    With a plain-Python receiver the internal slot proxy is created in the
    emitting thread and the process dies with no traceback."""

    def __init__(self, app, options: PlanetaryOptions, page: PlanetaryPage):
        super().__init__()
        self.app = app
        self.opts = options
        self.page = page
        self.data_dir = DEFAULT_DATA_DIR
        self._products = []        # last PDS search result
        self._days = []            # loaded LPGrsDay objects
        self._lat = self._lon = self._alt = self._spectra = None
        self._time64 = None        # per-record UTC timestamps (datetime64[s])
        self._meta = None          # cached bundled orbit metadata (lazy)
        self._abundance = {}       # resolution deg -> parsed abundance table
        self._lola = None          # cached LOLA DEM dict
        self._surface_is_abundance = False  # trace 0 recolored by abundance?
        self._globe_mesh = None    # (n_lon, n_lat) of the *built* globe
        self._info_dlg = None      # MissionInfoDialog (created on demand)
        self._active = None        # current selection: {lon, lat, half, spectrum, n_sel, label}
        self._kept = []            # pinned selections, each + "color"
        self._globe_ready = False
        self._globe_tmp = None
        self._flat_tmp = None      # cached 2D-map page (built on first use)
        self._worker = None
        self._on_done_cb = None
        self._wire()

    def _wire(self):
        o = self.opts
        o.btn_search.clicked.connect(self._search)
        o.btn_download.clicked.connect(self._download)
        o.btn_load.clicked.connect(self._load)
        o.btn_dir.clicked.connect(self._choose_dir)
        o.btn_select.toggled.connect(
            lambda armed: o.btn_select.setText(
                "Select region — armed" if armed else "Select region"))
        o.cb_keep.toggled.connect(self._on_keep_toggled)
        o.btn_send.clicked.connect(self._send_to_spectrum)
        o.btn_clear_sel.clicked.connect(self._clear_selection)
        # Detail is the one setting baked into the globe HTML — rebuild on
        # change (the overlays are all redrawn by _on_globe_loaded).
        o.detail.currentTextChanged.connect(
            lambda *_: self._build_globe(force=True))
        o.cb_track.toggled.connect(self._toggle_track)
        o.cb_marks.toggled.connect(self._toggle_landmarks)
        o.btn_info.clicked.connect(self._show_mission_info)
        o.dataset.currentIndexChanged.connect(self._on_dataset_changed)
        o.element.currentTextChanged.connect(lambda *_: self._refresh_abundance())
        o.resolution.currentTextChanged.connect(lambda *_: self._refresh_abundance())
        o.cmap.currentTextChanged.connect(self._refresh_drape)
        o.opacity.valueChanged.connect(lambda *_: self._refresh_abundance())
        o.cb_topo.toggled.connect(self._toggle_topo)
        o.exag.valueChanged.connect(self._on_exag_changed)
        o.cb_flat.toggled.connect(self._toggle_flat)
        # Grid shows/hides in place (the graticule is built into the page).
        o.cb_grid.toggled.connect(
            lambda v: self._js(f"waraSetGrid({str(bool(v)).lower()});"))
        # The metadata-driven orbit path follows the date/orbit controls.
        o.ed_start.editingFinished.connect(self._on_track_filter_changed)
        o.ed_end.editingFinished.connect(self._on_track_filter_changed)
        o.phase.currentIndexChanged.connect(self._on_track_filter_changed)

    # -- lifecycle -----------------------------------------------------------
    def on_activated(self):
        """Called by the app when the Planetary tab is opened."""
        if not self._globe_ready:
            self._build_globe()
        self._refresh_availability()

    def _refresh_availability(self):
        """Summarize, from the bundled metadata, what LP-GRS data exists and
        how much of it is already downloaded locally."""
        md = self.metadata()
        if md is None:
            self.opts.lbl_avail.setText("")
            return
        products = set(md["product"])
        have = sum((self.data_dir / f"{s}.xml").exists()
                   and (self.data_dir / f"{s}.dat").exists() for s in products)
        d0 = np.datetime_as_string(md["utc"].min(), unit="D")
        d1 = np.datetime_as_string(md["utc"].max(), unit="D")
        self.opts.lbl_avail.setText(
            f"{len(products)} daily products on the PDS, {d0} → {d1} "
            f"({have} downloaded locally).")

    # -- globe ---------------------------------------------------------------
    def _build_globe(self, force=False):
        view = self.page.ensure_web_view()
        if view is None or (self._globe_ready and not force):
            return
        n_lon, n_lat = GLOBE_DETAIL[self.opts.detail.currentText()]
        self._globe_mesh = (n_lon, n_lat)
        grid = self.opts.cb_grid.isChecked()
        self._status("Building the Moon…")
        self._run(lambda progress: build_globe_html(n_lon, n_lat, grid),
                  self._globe_done)

    def _globe_done(self, html):
        view = self.page.ensure_web_view()
        if view is None:
            return
        # Disconnect first so a globe rebuild doesn't stack duplicate slots.
        try:
            view.titleChanged.disconnect(self._on_title)
        except TypeError:
            pass
        view.titleChanged.connect(self._on_title)
        # Redraw the selection boxes once the (re)built page has loaded —
        # runJavaScript is a no-op while the page is still loading.
        try:
            view.loadFinished.disconnect(self._on_globe_loaded)
        except TypeError:
            pass
        view.loadFinished.connect(self._on_globe_loaded)
        with tempfile.NamedTemporaryFile(
                "w", delete=False, suffix=".html", encoding="utf-8") as f:
            f.write(html)
            path = f.name
        self._cleanup_globe_tmp()
        self._globe_tmp = path
        self._globe_ready = True
        if self.opts.cb_flat.isChecked():
            # Stay on the 2D map; the rebuilt globe shows on untick.
            self._status("Globe rebuilt — it will show when the 2D map is "
                         "turned off.")
        else:
            view.setUrl(QUrl.fromLocalFile(path))
            self._status("Globe ready. Hover for lon/lat; arm 'Select region' "
                         "then click the Moon to select.")

    def _on_globe_loaded(self, ok):
        """Re-apply the full display state after either page (3D globe or 2D
        map) finishes loading — both implement the same wara* JS API."""
        if not ok:
            return
        flat = self.opts.cb_flat.isChecked()
        self._js(f"waraSetGrid({str(self.opts.cb_grid.isChecked()).lower()});")
        for k in self._kept:
            self._js_box("waraKeepBox", k["lon"], k["lat"], k["half"],
                         color=k["color"])
        if self._active is not None:
            a = self._active
            self._js_box("waraShowBox", a["lon"], a["lat"], a["half"])
        if self.opts.cb_track.isChecked():
            self._draw_track()
        if self.opts.cb_marks.isChecked():
            self._draw_landmarks()
        self._refresh_drape()
        if self.opts.cb_topo.isChecked() and not flat:
            # New page: the JS elevation cache is gone — resend the grid.
            if self._lola is not None:
                self._apply_topo()
            else:
                self._ensure_lola(self._apply_topo)

    def _cleanup_globe_tmp(self):
        if self._globe_tmp:
            import os
            try:
                os.unlink(self._globe_tmp)
            except OSError:
                pass
            self._globe_tmp = None

    def _on_title(self, title):
        if not title.startswith("lonlat:"):
            return
        try:
            lon, lat = map(float, title.split(":")[1].split(","))
        except (IndexError, ValueError):
            return
        # Clicks only select while "Select region" is armed; each selection
        # disarms it again so orbiting the globe can't select by accident.
        if not self.opts.btn_select.isChecked():
            self._status("Arm 'Select region' to select a region "
                         f"(clicked lon {lon:.1f}°, lat {lat:.1f}°).")
            return
        self.opts.btn_select.setChecked(False)
        self.select_region(lon, lat)

    # -- PDS search / download / load -----------------------------------------
    def _dates_phase(self):
        """Read and validate the date-range and phase controls. Empty date
        fields mean "no bound"."""
        from wara.planetary import lp

        txt0 = self.opts.ed_start.text().strip()
        txt1 = self.opts.ed_end.text().strip()
        start = lp._as_date(txt0) if txt0 else None
        end = lp._as_date(txt1) if txt1 else None
        phase = {0: None, 1: "high", 2: "low"}[self.opts.phase.currentIndex()]
        return start, end, phase

    def _search(self):
        try:
            start, end, phase = self._dates_phase()
        except ValueError:
            self._status("Dates must be YYYY-MM-DD.")
            return

        def job(progress):
            from wara.planetary import list_grs_products, filter_products
            progress("Querying the PDS archive…")
            products = filter_products(list_grs_products(),
                                       start=start, end=end, phase=phase)
            return products

        self._run(job, self._search_done)

    def _search_done(self, products):
        self._products = products
        have = sum((self.data_dir / f"{p.stem}.xml").exists()
                   and (self.data_dir / f"{p.stem}.dat").exists()
                   for p in products)
        self.opts.btn_download.setEnabled(bool(products))
        if not products:
            self._status("No LP-GRS products in that range "
                         "(mission: 1998-01-16 to 1999-07-28).")
        else:
            d0, d1 = products[0].day, products[-1].day
            self._status(f"{len(products)} daily product(s), {d0} → {d1} "
                         f"({have} already downloaded).")

    def _download(self):
        products, dest = self._products, self.data_dir

        def job(progress):
            from wara.planetary import download_products
            download_products(
                products, dest,
                progress=lambda name, i, n: progress(f"Downloading {name} ({i}/{n})…"))
            return len(products)

        self._run(job, lambda n: self._status(
            f"Download complete — {n} product(s) in {dest}."))

    def _load(self):
        try:
            start, end, phase = self._dates_phase()
        except ValueError:
            self._status("Dates must be YYYY-MM-DD.")
            return
        from wara.planetary import lp
        paths = []
        for p in sorted(self.data_dir.glob("*_grs.xml")):
            m = lp._STEM_RE.match(p.stem)
            if not m:
                continue
            day = lp.doy_to_date(*map(int, m.groups()))
            if (start is None or day >= start) and (end is None or day <= end):
                if phase is None or \
                        (day >= lp.LOW_ALTITUDE_START) == (phase == "low"):
                    paths.append(p)
        if not paths:
            self._status("No downloaded products in that range — "
                         "Search PDS, then Download first.")
            return

        def job(progress):
            from wara.planetary import read_grs_day
            days = []
            for i, path in enumerate(paths):
                progress(f"Reading {path.name} ({i + 1}/{len(paths)})…")
                days.append(read_grs_day(path))
            return days

        self._run(job, self._load_done)

    def _load_done(self, days):
        self._days = days
        self._lat = np.concatenate([d.latitude for d in days])
        self._lon = np.concatenate([d.longitude for d in days])
        self._alt = np.concatenate([d.altitude_km for d in days])
        self._spectra = np.concatenate([d.spectra for d in days])
        # Absolute UTC timestamps (records_utc64 handles the extended
        # mission's continuous-DOY convention).
        from wara.planetary.lp import records_utc64
        self._time64 = np.concatenate([records_utc64(d) for d in days])
        n = len(self._lat)
        d0, d1 = days[0].day, days[-1].day
        self._status(f"Loaded {len(days)} day(s), {d0} → {d1}: {n} records. "
                     "Click the Moon to select a region.")
        # Kept spectra are snapshots of whatever was loaded when they were
        # pinned, so they survive a reload; the active selection is dropped
        # (its mask belonged to the previous dataset).
        self._active = None
        self.opts.btn_send.setEnabled(False)
        self._js("waraClearSel();")
        if self.opts.cb_track.isChecked():
            self._draw_track()
        self._replot()
        self.page.readout.setText(f"{len(days)} day(s) loaded — {n} records")

    # -- region selection ------------------------------------------------------
    def region_mask(self, lon, lat, half_width):
        """Records within the lat/lon box (longitude wraps the ±180 seam)."""
        from wara.planetary.moon import wrap_lon

        lat_lo, lat_hi = lat - half_width, lat + half_width
        mask = (self._lat >= lat_lo) & (self._lat <= lat_hi)
        # Once the box covers a pole, every longitude is inside it — only
        # constrain longitude when both box edges stay off the poles.
        if lat_hi < 90.0 and lat_lo > -90.0:
            lo = float(wrap_lon(lon - half_width))
            hi = float(wrap_lon(lon + half_width))
            if lo <= hi:
                mask &= (self._lon >= lo) & (self._lon <= hi)
            else:  # box crosses the +/-180 seam
                mask &= (self._lon >= lo) | (self._lon <= hi)
        return mask

    def select_region(self, lon, lat):
        half = float(self.opts.box_size.value())
        # While "Keep spectra" is checked, the outgoing selection is pinned
        # (spectrum stays plotted, its Moon box recolored to match) instead of
        # being replaced by the new click.
        if self.opts.cb_keep.isChecked() and self._active is not None:
            self._pin_active()
        # Draw the box on the globe even before data is loaded — the globe is
        # useful as a coordinate picker on its own.
        self._js_box("waraShowBox", lon, lat, half)
        if self._spectra is None:
            self._status(f"Selected lon {lon:.1f}°, lat {lat:.1f}° — "
                         "no data loaded yet.")
            return
        mask = self.region_mask(lon, lat, half)
        n_sel = int(np.count_nonzero(mask))
        label = (f"lon {lon:.1f}°, lat {lat:.1f}° ± {half:g}° "
                 f"({n_sel} records)")
        if n_sel == 0:
            self._active = None
            self.opts.btn_send.setEnabled(False)
            self._replot()
            self.page.readout.setText(label)
            self._status(f"0 records in {label} — LP covers ~13° of longitude "
                         "per day; load more days or enlarge the box.")
            return
        self._active = {"lon": lon, "lat": lat, "half": half, "n_sel": n_sel,
                        "spectrum": self._spectra[mask].sum(axis=0),
                        "label": label}
        self.opts.btn_send.setEnabled(True)
        self._replot()
        self.page.readout.setText(label)
        self._status(f"Summed {n_sel} records in {label}.")

    def _replot(self):
        """Redraw the spectrum panel from the current state: kept spectra in
        their box colors, the active selection in cyan on top, and (optionally)
        the scaled all-data comparison."""
        kept = [(k["label"], k["spectrum"], k["color"]) for k in self._kept]
        if self._active is None and not kept:
            if self._spectra is not None:
                n = len(self._spectra)
                self.page.show_spectrum(self._spectra.sum(axis=0),
                                        f"All loaded data ({n} records)")
            else:
                self.page.show_spectrum_empty()
            return
        spectrum = title = None
        compare = None
        if self._active is not None:
            spectrum, title = self._active["spectrum"], self._active["label"]
            if self.opts.cb_compare.isChecked() and self._spectra is not None:
                n_all = len(self._spectra)
                scale = self._active["n_sel"] / n_all
                compare = (f"all data, scaled ({n_all} records)",
                           self._spectra.sum(axis=0) * scale)
        self.page.show_spectrum(spectrum, title, compare=compare, kept=kept)

    def _pin_active(self):
        """Pin the active selection: its spectrum stays plotted and its box on
        the Moon is recolored to match the spectrum line."""
        kept = dict(self._active)
        kept["color"] = KEEP_COLORS[len(self._kept) % len(KEEP_COLORS)]
        self._kept.append(kept)
        self._active = None
        # Swap the cyan active box for a persistent one in the kept color.
        self._js("waraClearSel();")
        self._js_box("waraKeepBox", kept["lon"], kept["lat"], kept["half"],
                     color=kept["color"])

    def _on_keep_toggled(self, checked):
        """Unticking "Keep spectra" drops the pinned spectra and their boxes;
        the active selection is untouched."""
        if checked:
            return
        if self._kept:
            self._kept = []
            self._js("waraClearKept();")
            self._replot()
            self._status("Dropped the kept spectra.")

    def _send_to_spectrum(self):
        """Hand the active region's summed spectrum to the Spectrum tab (raw
        channels), mirroring the API tab's send."""
        if self._active is None:
            self._status("Click a region on the Moon first.")
            return
        if self.app is None:
            self._status("No main window to send to.")
            return
        from wara import spectrum as sp

        a = self._active
        name = (f"LP-GRS lon {a['lon']:.1f}, lat {a['lat']:.1f} "
                f"± {a['half']:g}")
        try:
            spect = sp.Spectrum(counts=a["spectrum"])
        except Exception as exc:  # noqa: BLE001 — surface build errors
            self._status(f"Could not build spectrum: {exc}")
            return
        # Load as the active spectrum but stay on this tab; a brief button
        # flash confirms the send (same pattern as the API tab).
        self.app.load_external_spectrum(spect, name, switch_tab=False)
        self._flash_button(self.opts.btn_send)
        self._status(f"Sent {a['label']} to the Spectrum tab.")

    def _flash_button(self, button):
        """Briefly brighten a button so the click visibly registered."""
        button.setStyleSheet(
            f"background-color:{T.ACCENT_CYAN}; border-color:{T.ACCENT_CYAN}; "
            f"color:{T.BG_DARK}; font-weight:800;")
        button.repaint()
        QTimer.singleShot(220, lambda: button.setStyleSheet(""))

    def _js(self, script):
        view = self.page.web_view
        if view is not None and self._globe_ready:
            view.page().runJavaScript(script)

    def _js_box(self, fn, lon, lat, half_width, color=None):
        x, y, z = region_box_xyz(lon, lat, half_width)
        args = f"{x}, {y}, {z}" + (f", '{color}'" if color else "")
        self._js(f"{fn}({args});")

    # -- orbit ground track ------------------------------------------------------
    @staticmethod
    def _track_payload(lon, lat, alt, t64):
        """Orbit-path payload from ephemeris arrays.

        Returns ``(x, y, z, color_days, hover_texts, title, n_total)``:
        sub-spacecraft positions lifted just above the surface, colored by days
        elapsed since the first point, subsampled to a floor of
        :data:`MIN_TRACK_POINTS`.
        """
        from wara.planetary.moon import lonlat_to_xyz, R_MOON_KM

        n = len(lon)
        stride = max(1, n // MIN_TRACK_POINTS)  # keep >= MIN_TRACK_POINTS points
        idx = np.arange(0, n, stride)
        x, y, z = lonlat_to_xyz(lon[idx], lat[idx], R_MOON_KM * 1.004)
        t = t64[idx]
        t0 = t64.min()
        color = (t - t0) / np.timedelta64(1, "D")
        stamps = np.datetime_as_string(t, unit="m")
        texts = [f"{ts.replace('T', ' ')} UTC · lon {lo:.1f}° lat {la:.1f}° · "
                 f"{al:.0f} km"
                 for ts, lo, la, al in zip(stamps, lon[idx], lat[idx], alt[idx])]
        title = f"days since {np.datetime_as_string(t0, unit='D')}"
        return (np.round(x, 1).tolist(), np.round(y, 1).tolist(),
                np.round(z, 1).tolist(), np.round(color, 4).tolist(), texts, title, n)

    def track_arrays(self):
        """Orbit-path payload from the loaded records (without the trailing
        total count — kept for compatibility)."""
        return self._track_payload(self._lon, self._lat, self._alt,
                                   self._time64)[:6]

    def metadata(self):
        """The bundled whole-mission orbit metadata (lazy-loaded, cached), or
        ``None`` if the CSV isn't available."""
        if self._meta is None:
            try:
                from wara.planetary import load_orbit_metadata
                self._meta = load_orbit_metadata()
            except (FileNotFoundError, OSError):
                return None
        return self._meta

    def _toggle_track(self, checked):
        if not checked:
            self._js("waraClearTrack();")
            return
        self._draw_track()

    def _on_track_filter_changed(self, *_):
        """Date/orbit controls changed: redraw the metadata-driven track."""
        if self.opts.cb_track.isChecked() and self._time64 is None:
            self._draw_track()

    def metadata_selection(self):
        """Boolean mask over the bundled metadata for the current date-range
        and orbit-phase controls (``None`` if no metadata)."""
        from wara.planetary.lp import LOW_ALTITUDE_START

        md = self.metadata()
        if md is None:
            return None
        try:
            start, end, phase = self._dates_phase()
        except ValueError:
            start = end = phase = None
        mask = np.ones(len(md["utc"]), dtype=bool)
        if start is not None:
            mask &= md["utc"] >= np.datetime64(start)
        if end is not None:
            mask &= md["utc"] < np.datetime64(end) + np.timedelta64(1, "D")
        split = np.datetime64(LOW_ALTITUDE_START)
        if phase == "high":
            mask &= md["utc"] < split
        elif phase == "low":
            mask &= md["utc"] >= split
        return mask

    def _draw_track(self):
        import json

        if self._time64 is not None:
            x, y, z, color, texts, title, n = self._track_payload(
                self._lon, self._lat, self._alt, self._time64)
            source = f"{len(x)} of {n} loaded records"
        else:
            md = self.metadata()
            if md is None:
                self._status("No data loaded and no bundled LP metadata found "
                             "— load LP-GRS data first.")
                return
            mask = self.metadata_selection()
            if not mask.any():
                self._js("waraClearTrack();")
                self._status("No LP products in that date/orbit range — "
                             "the orbit path is empty.")
                return
            x, y, z, color, texts, title, n = self._track_payload(
                md["lon"][mask], md["lat"][mask], md["alt_km"][mask],
                md["utc"][mask])
            d0 = np.datetime_as_string(md["utc"][mask].min(), unit="D")
            d1 = np.datetime_as_string(md["utc"][mask].max(), unit="D")
            source = (f"{len(x)} of {n} bundled-metadata points, "
                      f"{d0} → {d1}")
        self._js(f"waraShowTrack({x}, {y}, {z}, {color}, "
                 f"{json.dumps(texts)}, {json.dumps(title)});")
        self._status(f"Orbit path: {source}, colored by time.")

    # -- mission info --------------------------------------------------------------
    def _show_mission_info(self):
        """Open the PDS-documentation browser; documents are fetched once (in
        a worker) and cached on disk, so later opens work offline."""
        if self._info_dlg is None:
            self._info_dlg = MissionInfoDialog(self.app)
        self._info_dlg.show()
        self._info_dlg.raise_()
        docs = self._info_dlg.docs
        if docs and not any(t.startswith("(Could not fetch")
                            for t in docs.values()):
            return  # everything already loaded; failed ones retry below

        data_dir = self.data_dir

        def job(progress):
            from wara.planetary import LP_DOCUMENTS, fetch_document
            docs = {}
            for i, label in enumerate(LP_DOCUMENTS):
                progress(f"Fetching documentation ({i + 1}/{len(LP_DOCUMENTS)}): "
                         f"{label}…")
                try:
                    docs[label] = fetch_document(label, data_dir=data_dir)
                except Exception as exc:  # noqa: BLE001 — keep the other docs
                    docs[label] = (f"(Could not fetch this document: {exc})\n\n"
                                   "It will be retried the next time the "
                                   "dialog is opened.")
            return docs

        self._run(job, self._info_docs_loaded)

    def _info_docs_loaded(self, docs):
        # Failed fetches are shown but not cached in the dialog, so reopening
        # retries them.
        if self._info_dlg is not None:
            self._info_dlg.set_docs(docs)
        if any(t.startswith("(Could not fetch") for t in docs.values()):
            self._status("Mission documentation loaded (some documents "
                         "could not be fetched).")
        else:
            self._status("Mission documentation loaded.")

    # -- dataset drapes: abundance maps and LOLA elevation -------------------------
    def _on_dataset_changed(self, idx):
        abundance, elevation = idx == 1, idx == 2
        self.opts.element.setEnabled(abundance)
        self.opts.resolution.setEnabled(abundance)
        self.opts.cmap.setEnabled(abundance or elevation)
        self.opts.opacity.setEnabled(abundance)
        if abundance:
            self._refresh_abundance()
            return
        # Leaving Calibrated: drop the transparent overlay if any, and put
        # the chosen dataset on the base surface.
        self._js("waraClearOverlay();")
        self._surface_is_abundance = False
        if elevation:
            self._refresh_elevation()
        else:
            self._js("waraResetSurface();")
            self._status("Raw spectra dataset — albedo Moon restored.")

    def _refresh_drape(self, *_):
        """Re-apply whichever color drape the dataset combo selects."""
        idx = self.opts.dataset.currentIndex()
        if idx == 1:
            self._refresh_abundance()
        elif idx == 2:
            self._refresh_elevation()

    def _abundance_deg(self):
        return int(self.opts.resolution.currentText().rstrip("°"))

    def _refresh_abundance(self):
        """Drape the selected element's abundance map over the Moon (only in
        Calibrated-abundance mode; downloads the table on first use)."""
        if self.opts.dataset.currentIndex() != 1 or not self._globe_ready:
            return
        deg = self._abundance_deg()
        element = self.opts.element.currentText()
        if deg in self._abundance:
            self._apply_abundance(self._abundance[deg], element)
            return

        def job(progress):
            from wara.planetary import download_abundance, read_abundance
            progress(f"Downloading the {deg}° abundance map…")
            download_abundance(deg, data_dir=self.data_dir)
            return deg, read_abundance(deg, data_dir=self.data_dir)

        self._run(job, self._abundance_loaded)

    def _abundance_loaded(self, result):
        deg, table = result
        self._abundance[deg] = table
        # Element/resolution may have changed while downloading — re-read.
        if self.opts.dataset.currentIndex() == 1 \
                and self._abundance_deg() == deg:
            self._apply_abundance(table, self.opts.element.currentText())

    def _apply_abundance(self, table, element):
        import json

        from wara.planetary.abundance import ABUNDANCE_ELEMENTS, abundance_grid

        lon_axis, lat_axis = self._mesh_axes()
        grid = abundance_grid(table, element, lon_axis, lat_axis)
        label, unit = ABUNDANCE_ELEMENTS[element]
        # Robust display range (the Th distribution has a long KREEP tail).
        cmin = float(np.nanpercentile(grid, 1))
        cmax = float(np.nanpercentile(grid, 99))
        values = json.dumps(np.round(grid, 4).tolist())
        cmap = self.opts.cmap.currentText() or "Viridis"
        title = json.dumps(f"{label} ({unit})")
        opacity = float(self.opts.opacity.value()) / 100.0
        if opacity >= 0.995:
            # Fully opaque: recolor the base surface directly (cheapest).
            self._js("waraClearOverlay();")
            self._js(f"waraSetSurface({values}, {cmin:.6g}, {cmax:.6g}, "
                     f"{title}, {json.dumps(cmap)});")
            self._surface_is_abundance = True
            extra = ""
        else:
            # Semi-transparent overlay: float it over the grey albedo Moon.
            # Always reset the base first — after a globe rebuild the fresh
            # page shows albedo regardless of what trace 0 held before. The
            # 3D topography relief (if enabled) still shows through; only the
            # elevation *color* drape is excluded, so there is one colorbar.
            self._js("waraResetSurface();")
            self._surface_is_abundance = False
            self._js(f"waraSetOverlay({values}, {cmin:.6g}, {cmax:.6g}, "
                     f"{title}, {json.dumps(cmap)}, {opacity:.2f});")
            extra = f" at {opacity * 100:.0f} % opacity over the albedo Moon"
        self._status(f"{label} abundance map ({self._abundance_deg()}° pixels)"
                     f"{extra} — {cmin:.3g} to {cmax:.3g} {unit} "
                     "(1st-99th percentile scale).")

    # -- LOLA topography -----------------------------------------------------------
    def _ensure_lola(self, then):
        """Run ``then()`` once the LOLA DEM is available (downloading ~2 MB
        into the data folder on first use)."""
        if self._lola is not None:
            then()
            return
        data_dir = self.data_dir

        def job(progress):
            from wara.planetary import download_lola_dem, read_lola_dem
            progress("Downloading the LOLA topography model…")
            download_lola_dem(4, data_dir=data_dir)
            return read_lola_dem(4, data_dir=data_dir)

        def done(dem):
            self._lola = dem
            then()

        self._run(job, done)

    def _mesh_axes(self):
        n_lon, n_lat = self._globe_mesh or GLOBE_DETAIL[
            self.opts.detail.currentText()]
        return (np.linspace(-180.0, 180.0, n_lon),
                np.linspace(-90.0, 90.0, n_lat))

    def _refresh_elevation(self):
        if self.opts.dataset.currentIndex() != 2 or not self._globe_ready:
            return
        self._ensure_lola(self._apply_elevation)

    def _apply_elevation(self, quiet=False):
        import json

        from wara.planetary import elevation_grid

        lon_axis, lat_axis = self._mesh_axes()
        grid = elevation_grid(self._lola, lon_axis, lat_axis)
        cmin, cmax = float(grid.min()), float(grid.max())
        cmap = self.opts.cmap.currentText() or "Viridis"
        values = json.dumps(np.round(grid, 3).tolist())
        self._js(f"waraSetSurface({values}, {cmin:.6g}, {cmax:.6g}, "
                 f"{json.dumps('Elevation (km)')}, {json.dumps(cmap)});")
        if not quiet:
            self._status(f"LOLA elevation draped on the Moon — {cmin:.1f} to "
                         f"{cmax:.1f} km relative to the 1737.4 km sphere.")

    def _toggle_topo(self, checked):
        self.opts.exag.setEnabled(checked)
        if not checked:
            self._js("waraResetTopo();")
            self._status("Topography off — spherical Moon restored.")
            return
        self._ensure_lola(self._apply_topo)

    def _on_exag_changed(self, *_):
        # The elevation grid is cached in the page: re-scale in place.
        if self.opts.cb_topo.isChecked() and self._globe_ready:
            self._js(f"waraSetTopo(null, {float(self.opts.exag.value()):g});")

    def _apply_topo(self):
        import json

        if not self._globe_ready:
            return
        from wara.planetary import elevation_grid

        lon_axis, lat_axis = self._mesh_axes()
        grid = elevation_grid(self._lola, lon_axis, lat_axis)
        exag = float(self.opts.exag.value())
        values = json.dumps(np.round(grid, 3).tolist())
        self._js(f"waraSetTopo({values}, {exag:g});")
        self._status(f"LOLA 3D relief on ({exag:g}× exaggeration) — combine "
                     "with the Calibrated dataset to correlate composition "
                     "with topography.")

    # -- 2D equirectangular projection ---------------------------------------------
    def _toggle_flat(self, checked):
        if not self._globe_ready:
            return
        if checked:
            self._show_flat_page()
        else:
            view = self.page.web_view
            if view is not None and self._globe_tmp:
                view.setUrl(QUrl.fromLocalFile(self._globe_tmp))
            self._status("3D globe restored.")
        # The full display state (overlays, drapes, grid, relief) is
        # re-applied by _on_globe_loaded once the swapped page has loaded.

    def _show_flat_page(self):
        """Swap the view to the 2D map page (built once per session — it does
        not depend on the globe's mesh settings)."""
        if self._flat_tmp:
            view = self.page.web_view
            if view is not None:
                view.setUrl(QUrl.fromLocalFile(self._flat_tmp))
            self._status("2D equirectangular map — native-resolution texture; "
                         "clicks, overlays, and drapes work as on the globe.")
            return

        def job(progress):
            progress("Building the 2D map…")
            return build_flat_html()

        self._run(job, self._flat_done)

    def _flat_done(self, html):
        view = self.page.web_view
        if view is None:
            return
        with tempfile.NamedTemporaryFile(
                "w", delete=False, suffix=".html", encoding="utf-8") as f:
            f.write(html)
            self._flat_tmp = f.name
        # The user may have unticked the box while the page was building.
        if self.opts.cb_flat.isChecked():
            view.setUrl(QUrl.fromLocalFile(self._flat_tmp))
            self._status("2D equirectangular map — native-resolution texture; "
                         "clicks, overlays, and drapes work as on the globe.")

    # -- landmarks ---------------------------------------------------------------
    def _toggle_landmarks(self, checked):
        if checked:
            self._draw_landmarks()
        else:
            self._js("waraClearMarks();")

    def _draw_landmarks(self):
        import json

        from wara.planetary.moon import LUNAR_LANDMARKS, lonlat_to_xyz, R_MOON_KM

        lons, lats, names = zip(*[(lo, la, nm) for lo, la, nm in LUNAR_LANDMARKS])
        lons, lats = np.asarray(lons), np.asarray(lats)
        # Dots hug the surface; the labels sit on a higher shell so the Moon
        # can never occlude parts of the text.
        xd, yd, zd = lonlat_to_xyz(lons, lats, R_MOON_KM * 1.008)
        xt, yt, zt = lonlat_to_xyz(lons, lats, R_MOON_KM * 1.06)
        hover = [f"{nm} · lon {lo:.1f}° lat {la:.1f}°"
                 for lo, la, nm in LUNAR_LANDMARKS]
        self._js(f"waraShowMarks("
                 f"{np.round(xd, 1).tolist()}, {np.round(yd, 1).tolist()}, "
                 f"{np.round(zd, 1).tolist()}, "
                 f"{np.round(xt, 1).tolist()}, {np.round(yt, 1).tolist()}, "
                 f"{np.round(zt, 1).tolist()}, "
                 f"{json.dumps(list(names))}, {json.dumps(hover)});")

    def _clear_selection(self):
        self._js("waraClearBoxes();")
        self._active = None
        self._kept = []
        self.opts.btn_send.setEnabled(False)
        self.page.readout.setText("")
        self._replot()

    # -- misc ------------------------------------------------------------------
    def _choose_dir(self):
        path = QFileDialog.getExistingDirectory(
            self.app, "LP-GRS data folder", str(self.data_dir))
        if path:
            self.data_dir = Path(path)
            self.opts.set_data_dir_label(path)

    def _run(self, fn, on_done):
        """Run ``fn`` in a worker thread; only one job at a time.

        Worker signals connect exclusively to bound methods of this QObject
        (see the class docstring); ``on_done`` is dispatched from one."""
        if self._worker is not None and self._worker.isRunning():
            self._status("Busy — wait for the current task to finish.")
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._on_done_cb = on_done
        self._worker = _Worker(fn)
        self._worker.progress.connect(self._status)
        self._worker.done.connect(self._dispatch_done)
        self._worker.failed.connect(self._dispatch_failed)
        self._worker.start()

    def _dispatch_done(self, result):
        cb, self._on_done_cb = self._on_done_cb, None
        if cb is not None:
            cb(result)

    def _dispatch_failed(self, msg):
        self._on_done_cb = None
        self._status(f"Failed: {msg}")

    def _status(self, msg):
        self.opts.status.setText(msg)
        if self.app is not None:
            self.app.statusBar().showMessage("  " + msg)
