"""Diagnostics dialog for the API tab (``DiagnosticsDialog``): MCA spectra,
trace/binary waveforms and per-run statistics for a loaded run."""
import traceback

import numpy as np

from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.colors import to_hex
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSizePolicy, QTabWidget,
    QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt, QSize

from wara import read_parquet_api, apicalc, helper_api
from wara import spectrum as sp

from . import theme as T
from .widgets import hsep, header, labeled_row
from .api_common import API_PLOT_BG, _combo_row, _draw_axes_placeholder
from .api_dialogs import StatsInfoDialog


class DiagnosticsDialog(QDialog):
    """Pop-out diagnostics for an API run, ported from the legacy MCA / Binary
    windows. Each capability lives on its own tab:

    * **MCA** -- load the run's ``.npy`` MCA spectra and overlay them on one
      canvas; each spectrum has a colour swatch and a visibility checkbox (like
      the Spectrum tab), and "Send visible to spectrum" hands every visible
      spectrum to the Spectrum tab (first active, the rest as overlays). "Run
      stats..." opens the per-channel statistics from the latest ``-stats-``
      file (:class:`StatsInfoDialog`).
    * **Binary** -- read the run's list-mode data (trace / binary / parquet) and
      explore it across four linked views: the per-channel energy histogram
      (drag a span to pick events), the traces of the selected events, a random
      trace sampler (with pileup / CFD-error / flagged filters and an FFT
      toggle), and an energy spectrum reconstructed by integrating the traces.
      Both the energy histogram and the trace-integral spectrum can be sent to
      the Spectrum tab.

    Date and run default to the API tab's current selection (see ``seed``) but
    can be overridden here so the diagnostics can target a different run without
    disturbing the main view.
    """

    def __init__(self, controller):
        super().__init__(controller.app)
        self.c = controller
        self.setWindowTitle("API diagnostics")
        self.setStyleSheet(T.STYLESHEET)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.resize(1100, 720)

        # ── MCA state ────────────────────────────────────────────────────────
        self.mca_data = None       # 2D array, one spectrum per row
        self._mca_visible = []     # per-spectrum visibility flags
        self._mca_colors = []      # per-spectrum plot color (hex)
        self._mca_checks = []      # the visibility QCheckBoxes
        self._mca_ax = {}
        self._mca_fig = {}
        self._mca_canvas = {}
        self._mca_toolbar = {}
        self._stats_dlg = None     # StatsInfoDialog (created lazily)

        # ── Binary state ─────────────────────────────────────────────────────
        self.df_bin = None         # full loaded list-mode dataframe
        self.df_bin_ch = None      # visible-channels view (drives all downstream)
        self.df_bin_tr = None      # events inside the energy span (traces shown)
        self.df_bin_rand = None    # random-trace sample
        self._bin_chans = []       # sorted unique channel numbers
        self._bin_ch_visible = {}  # channel -> bool
        self._bin_ch_colors = {}   # channel -> hex
        self._bin_ch_checks = {}   # channel -> QCheckBox
        self._bin_ax = {}
        self._bin_fig = {}
        self._bin_canvas = {}
        self._bin_toolbar = {}
        self._bin_span = None      # SpanSelector on the energy plot
        self._bin_yscale = "log"
        self._bin_gam = self._bin_gam_x = None    # energy hist (send to spectrum)
        self._bin_gam2 = self._bin_gam_x2 = None  # trace-integral hist
        self._bin_fft = None       # (freqs, list-of-magnitudes) when FFT is on

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_mca_tab(), "MCA")
        self.tabs.addTab(self._build_binary_tab(), "Binary")
        self._fit_tabbar(self.tabs)
        root.addWidget(self.tabs, 1)

    # ── shared helpers ──────────────────────────────────────────────────────
    def _make_canvas(self, store_key, ax_store, fig_store, canvas_store,
                     toolbar_store, xlabel, yscale="log", min_h=220,
                     placeholder="Load an MCA file to begin"):
        """Build a (toolbar + canvas) column wired into the given dicts and
        return the wrapping QWidget."""
        col = QWidget()
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        fig = Figure(figsize=(6, 3), facecolor=API_PLOT_BG)
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.setMinimumHeight(min_h)
        toolbar = NavToolbar(canvas, self)
        toolbar.setObjectName("plot_toolbar")
        toolbar.setIconSize(QSize(22, 22))
        T.recolor_toolbar_icons(toolbar, T.TEXT_PRIMARY)
        ax_store[store_key] = ax
        fig_store[store_key] = fig
        canvas_store[store_key] = canvas
        toolbar_store[store_key] = toolbar
        lay.addWidget(toolbar)
        lay.addWidget(canvas, 1)
        self._draw_placeholder(ax, placeholder, xlabel, yscale)
        return col

    @staticmethod
    def _style_axis(ax, xlabel, yscale):
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, color=T.TEXT_DIM, fontsize=12)
        ax.set_facecolor(API_PLOT_BG)
        ax.tick_params(colors=T.TEXT_DIM, which="both", length=3, labelsize=11)
        for sp_ in ax.spines.values():
            sp_.set_color(T.BORDER)

    def _draw_placeholder(self, ax, message, xlabel, yscale="log"):
        _draw_axes_placeholder(ax, message, xlabel, self._style_axis, yscale)

    def _state(self, msg):
        self.lbl_mca_state.setText(msg)

    # ── seeding ─────────────────────────────────────────────────────────────
    def seed(self, date, runnr):
        """Pre-fill the MCA and Binary date/run from the API tab, leaving any
        value the user has already typed untouched."""
        for ed_date, ed_run in ((self.ed_mca_date, self.ed_mca_run),
                                (self.ed_bin_date, self.ed_bin_run)):
            if date and not ed_date.text().strip():
                ed_date.setText(str(date))
            if runnr is not None and runnr != "" and not ed_run.text().strip():
                ed_run.setText(str(runnr))

    @staticmethod
    def _palette(n):
        """*n* high-contrast colours (hex). Uses matplotlib's qualitative
        tab10/tab20 palettes (distinct neighbours, unlike a gradient) and cycles
        them for larger counts."""
        base = cm.tab10.colors if n <= 10 else cm.tab20.colors
        return [to_hex(base[i % len(base)]) for i in range(max(n, 1))]

    @staticmethod
    def _fit_tabbar(tabw):
        """Stop a QTabWidget clipping its labels.

        The global stylesheet paints the tabs bold 14px, but the tab bar sizes
        itself with the *default* font, so it ends up too narrow and the text is
        chopped on both sides. Give the bar the matching bold font (so the size
        hint is right), disable eliding, and add a little width so the labels sit
        comfortably. (Mirrors SelectionsDialog._fit_tabbar.)"""
        bar = tabw.tabBar()
        f = bar.font()
        f.setPixelSize(14)
        f.setBold(True)
        bar.setFont(f)
        bar.setElideMode(Qt.ElideNone)
        bar.setExpanding(False)
        bar.setUsesScrollButtons(False)
        tabw.setStyleSheet("QTabBar::tab { padding: 7px 24px; min-width: 96px; }")

    # ════════════════════════════════════════════════════════════════════════
    # MCA tab
    # ════════════════════════════════════════════════════════════════════════
    def _build_mca_tab(self):
        tab = QWidget()
        row = QHBoxLayout(tab)
        row.setContentsMargins(6, 6, 6, 6)
        row.setSpacing(8)

        # Plot side: one canvas overlaying every visible spectrum.
        row.addWidget(self._make_canvas(
            "main", self._mca_ax, self._mca_fig, self._mca_canvas,
            self._mca_toolbar, "Channels", min_h=460), 1)

        # Control side.
        row.addWidget(self._build_mca_controls(), 0)
        return tab

    def _build_mca_controls(self):
        side = QVBoxLayout()
        side.setSpacing(6)

        side.addWidget(header("RUN"))
        note = QLabel("Loads the run's MCA .npy spectra. Defaults to the API "
                      "tab's run; override here to inspect another.")
        note.setObjectName("stat_key"); note.setWordWrap(True)
        side.addWidget(note)
        self.ed_mca_date = QLineEdit(); self.ed_mca_date.setPlaceholderText("YYYY-MM-DD")
        r, _ = labeled_row("Date", self.ed_mca_date); side.addWidget(r)
        self.ed_mca_run = QLineEdit(); self.ed_mca_run.setPlaceholderText("e.g. 91")
        r, _ = labeled_row("Run", self.ed_mca_run); side.addWidget(r)

        self.btn_mca_load = QPushButton("Load MCA")
        self.btn_mca_load.setObjectName("open_btn")
        self.btn_mca_load.setCursor(Qt.PointingHandCursor)
        self.btn_mca_load.setToolTip("Read the run's MCA-data/*.npy spectra and "
                                     "overlay them all")
        self.btn_mca_load.clicked.connect(self._load_mca)
        side.addWidget(self.btn_mca_load)

        self.btn_mca_info = QPushButton("Run stats...")
        self.btn_mca_info.setObjectName("action_btn")
        self.btn_mca_info.setCursor(Qt.PointingHandCursor)
        self.btn_mca_info.setToolTip("Show the run's per-channel MCA statistics "
                                     "(real/live time, counts, count rates) from "
                                     "the latest -stats- file")
        self.btn_mca_info.clicked.connect(self._open_stats)
        side.addWidget(self.btn_mca_info)

        # ── Spectrum visibility list ─────────────────────────────────────────
        side.addWidget(hsep()); side.addWidget(header("SPECTRA"))
        self.cb_mca_log = QCheckBox("Log Y"); self.cb_mca_log.setChecked(True)
        self.cb_mca_log.setToolTip("Logarithmic y-axis")
        self.cb_mca_log.toggled.connect(lambda *_: self._plot_mca())
        side.addWidget(self.cb_mca_log)

        vis_btns = QHBoxLayout(); vis_btns.setContentsMargins(0, 0, 0, 0); vis_btns.setSpacing(6)
        self.btn_mca_all = QPushButton("Show all"); self.btn_mca_all.setObjectName("mini_btn")
        self.btn_mca_all.setCursor(Qt.PointingHandCursor)
        self.btn_mca_all.clicked.connect(lambda: self._set_all_mca_visible(True))
        self.btn_mca_none = QPushButton("Hide all"); self.btn_mca_none.setObjectName("mini_btn")
        self.btn_mca_none.setCursor(Qt.PointingHandCursor)
        self.btn_mca_none.clicked.connect(lambda: self._set_all_mca_visible(False))
        vis_btns.addWidget(self.btn_mca_all); vis_btns.addWidget(self.btn_mca_none)
        vbw = QWidget(); vbw.setLayout(vis_btns); side.addWidget(vbw)

        self.mca_list_area = QScrollArea()
        self.mca_list_area.setWidgetResizable(True)
        self.mca_list_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.mca_list_area.setMinimumHeight(150)
        self.mca_list_inner = QWidget()
        self.mca_list_lay = QVBoxLayout(self.mca_list_inner)
        self.mca_list_lay.setContentsMargins(0, 0, 0, 0); self.mca_list_lay.setSpacing(3)
        self.mca_list_lay.addStretch(1)
        self.mca_list_area.setWidget(self.mca_list_inner)
        side.addWidget(self.mca_list_area, 1)

        side.addWidget(hsep()); side.addWidget(header("ACTIONS"))
        self.btn_mca_send = QPushButton("Send visible to spectrum")
        self.btn_mca_send.setObjectName("open_btn")
        self.btn_mca_send.setCursor(Qt.PointingHandCursor)
        self.btn_mca_send.setToolTip("Hand every currently-visible MCA spectrum "
                                     "to the Spectrum tab (first active, rest as "
                                     "overlays; raw channels)")
        self.btn_mca_send.clicked.connect(self._send_mca)
        side.addWidget(self.btn_mca_send)
        self.btn_mca_reset = QPushButton("Reset")
        self.btn_mca_reset.setObjectName("mini_btn")
        self.btn_mca_reset.setCursor(Qt.PointingHandCursor)
        self.btn_mca_reset.clicked.connect(self._reset_mca)
        side.addWidget(self.btn_mca_reset)

        self.lbl_mca_state = QLabel(""); self.lbl_mca_state.setObjectName("stat_key")
        self.lbl_mca_state.setWordWrap(True); side.addWidget(self.lbl_mca_state)

        holder = QWidget(); holder.setFixedWidth(300); holder.setLayout(side)
        return holder

    def _run_inputs(self):
        """Validate and return (date, runnr), or None with a status message."""
        date = self.ed_mca_date.text().strip()
        run_txt = self.ed_mca_run.text().strip()
        if not date or not run_txt:
            self._state("Enter a date and run number first")
            return None
        try:
            runnr = int(run_txt)
        except ValueError:
            self._state("Run number must be an integer")
            return None
        return date, runnr

    def _load_mca(self):
        run = self._run_inputs()
        if run is None:
            return
        date, runnr = run
        try:
            self.mca_data = helper_api.read_mca(date=date, runnr=runnr)
        except Exception as exc:  # noqa: BLE001  -- surface read errors to the user
            traceback.print_exc()
            self._state(f"Could not load MCA file: {exc}")
            return
        n = len(self.mca_data)
        self._mca_visible = [True] * n
        self._mca_colors = self._palette(n)
        self._populate_mca_list()
        self._plot_mca()
        self._state(f"Loaded {n} spectra ({self.mca_data.shape[1]} channels each)")

    def _populate_mca_list(self):
        """Rebuild the per-spectrum visibility rows (swatch + label + checkbox)."""
        # Drop the old rows (everything except the trailing stretch).
        while self.mca_list_lay.count() > 1:
            item = self.mca_list_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._mca_checks = []
        for i in range(len(self.mca_data)):
            row = QWidget()
            h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
            swatch = QLabel(); swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background:{self._mca_colors[i]}; border-radius:3px;")
            h.addWidget(swatch)
            lbl = QLabel(f"Spectrum #{i}"); lbl.setObjectName("stat_key")
            h.addWidget(lbl); h.addStretch(1)
            cb = QCheckBox(); cb.setObjectName("vis_check")
            cb.setChecked(self._mca_visible[i]); cb.setToolTip("Toggle visibility")
            cb.toggled.connect(lambda vis, idx=i: self._toggle_mca(idx, vis))
            h.addWidget(cb)
            self._mca_checks.append(cb)
            self.mca_list_lay.insertWidget(self.mca_list_lay.count() - 1, row)

    def _toggle_mca(self, idx, visible):
        self._mca_visible[idx] = visible
        self._plot_mca()

    def _set_all_mca_visible(self, visible):
        if self.mca_data is None:
            return
        self._mca_visible = [visible] * len(self.mca_data)
        for cb in self._mca_checks:
            cb.blockSignals(True); cb.setChecked(visible); cb.blockSignals(False)
        self._plot_mca()

    def _plot_mca(self):
        ax = self._mca_ax["main"]
        yscale = "log" if self.cb_mca_log.isChecked() else "linear"
        if self.mca_data is None:
            self._draw_placeholder(ax, "Load an MCA file to begin", "Channels", yscale)
            self._mca_canvas["main"].draw_idle()
            return
        ax.clear()
        shown = 0
        for i, row in enumerate(self.mca_data):
            if not self._mca_visible[i]:
                continue
            ax.plot(row, color=self._mca_colors[i], lw=0.9, alpha=0.9,
                    label=f"#{i}")
            shown += 1
        ax.set_title("MCA spectra", color=T.TEXT_PRIMARY, fontsize=12)
        self._style_axis(ax, "Channels", yscale)
        if 0 < shown <= 16:
            ax.legend(loc="upper right", fontsize=9, facecolor=API_PLOT_BG,
                      edgecolor=T.BORDER, labelcolor=T.TEXT_DIM, framealpha=0.7,
                      ncol=2 if shown > 6 else 1)
        elif shown == 0:
            ax.text(0.5, 0.5, "All spectra hidden", transform=ax.transAxes,
                    ha="center", va="center", color=T.TEXT_DIM, fontsize=12)
        self._mca_fig["main"].tight_layout()
        self._mca_canvas["main"].draw_idle()
        self._mca_toolbar["main"].update()

    def _send_mca(self):
        if self.mca_data is None:
            self._state("Load an MCA file first")
            return
        specs = []
        for i, row in enumerate(self.mca_data):
            if not self._mca_visible[i]:
                continue
            try:
                specs.append((sp.Spectrum(counts=row, e_units="channels"),
                              f"MCA #{i}"))
            except Exception as exc:  # noqa: BLE001
                self._state(f"Could not build spectrum #{i}: {exc}")
                return
        if not specs:
            self._state("No visible spectra to send")
            return
        self.c.app.load_external_spectra(specs, switch_tab=False)
        self._state(f"Sent {len(specs)} spectrum(s) to the Spectrum tab")

    def _reset_mca(self):
        self.mca_data = None
        self._mca_visible = []
        self._mca_colors = []
        self._populate_mca_list_empty()
        self._plot_mca()
        self._state("")

    def _populate_mca_list_empty(self):
        while self.mca_list_lay.count() > 1:
            item = self.mca_list_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._mca_checks = []

    # ── Run stats ────────────────────────────────────────────────────────────
    def _open_stats(self):
        run = self._run_inputs()
        if run is None:
            return
        date, runnr = run
        try:
            stats, fname = helper_api.read_mca_stats(date=date, runnr=runnr)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._state(f"Could not read run stats: {exc}")
            return
        if self._stats_dlg is None:
            self._stats_dlg = StatsInfoDialog(self)
        self._stats_dlg.populate(stats, f"{date}  ·  run {runnr}", fname)
        self._stats_dlg.show()
        self._stats_dlg.raise_()
        self._stats_dlg.activateWindow()
        self._state(f"Run stats from {fname}")

    # ════════════════════════════════════════════════════════════════════════
    # Binary tab
    # ════════════════════════════════════════════════════════════════════════
    SAMPLE_RATE = 500e6     # 500 MHz digitizer -> 2 ns per trace sample
    NS_PER_PT = 2

    def _bin_state(self, msg):
        self.lbl_bin_state.setText(msg)

    def _build_binary_tab(self):
        tab = QWidget()
        row = QHBoxLayout(tab)
        row.setContentsMargins(6, 6, 6, 6)
        row.setSpacing(8)

        # Plot side: four linked views as inner tabs.
        self.bin_plot_tabs = QTabWidget()
        specs = [
            ("erg", "Energy", "Channels", "Load a run to plot the energy histogram"),
            ("ergtr", "Energy traces", "Time (ns)",
             "Drag a span on the Energy tab to show traces"),
            ("rand", "Random traces", "Time (ns)", "Sample random traces"),
            ("own", "Trace energy", "Channels", "Calc energy from trace integration"),
        ]
        for key, title, xlabel, ph in specs:
            yscale = "log" if key in ("erg", "own") else "linear"
            self.bin_plot_tabs.addTab(
                self._make_canvas(key, self._bin_ax, self._bin_fig,
                                  self._bin_canvas, self._bin_toolbar, xlabel,
                                  yscale=yscale, min_h=440, placeholder=ph),
                title)
        self._fit_tabbar(self.bin_plot_tabs)
        row.addWidget(self.bin_plot_tabs, 1)

        # Control side (scrollable -- there are several groups).
        row.addWidget(self._build_binary_controls(), 0)
        return tab

    def _build_binary_controls(self):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setFixedWidth(320)
        inner = QWidget()
        side = QVBoxLayout(inner); side.setSpacing(6)
        side.setContentsMargins(6, 6, 6, 6)

        # ── Run / load ───────────────────────────────────────────────────────
        side.addWidget(header("RUN"))
        self.ed_bin_date = QLineEdit(); self.ed_bin_date.setPlaceholderText("YYYY-MM-DD")
        r, _ = labeled_row("Date", self.ed_bin_date); side.addWidget(r)
        self.ed_bin_run = QLineEdit(); self.ed_bin_run.setPlaceholderText("e.g. 8")
        r, _ = labeled_row("Run", self.ed_bin_run); side.addWidget(r)
        self.cmb_bin_type = QComboBox()
        self.cmb_bin_type.addItems(["Trace data", "Binary data", "Parquet"])
        self.cmb_bin_type.setToolTip("Which list-mode source to read for this run")
        side.addWidget(_combo_row("Source", self.cmb_bin_type))
        self.ed_bin_bins = QLineEdit("4098"); self.ed_bin_bins.setFixedWidth(80)
        self.ed_bin_bins.setToolTip("Number of bins for the energy histograms")
        r, _ = labeled_row("Bins", self.ed_bin_bins); side.addWidget(r)
        self.btn_bin_load = QPushButton("Load")
        self.btn_bin_load.setObjectName("open_btn")
        self.btn_bin_load.setCursor(Qt.PointingHandCursor)
        self.btn_bin_load.setToolTip("Read the run's list-mode data and plot the "
                                     "per-channel energy histogram")
        self.btn_bin_load.clicked.connect(self._load_bin)
        side.addWidget(self.btn_bin_load)

        # ── Energy plot ──────────────────────────────────────────────────────
        side.addWidget(hsep()); side.addWidget(header("ENERGY"))
        en_note = QLabel("Drag a span on the Energy plot to pick events for the "
                         "Energy-traces tab. Toggle channels below; the visible "
                         "ones drive every view.")
        en_note.setObjectName("stat_key"); en_note.setWordWrap(True)
        side.addWidget(en_note)
        self.cb_bin_log = QCheckBox("Log Y"); self.cb_bin_log.setChecked(True)
        self.cb_bin_log.setToolTip("Logarithmic y-axis on the energy histogram")
        self.cb_bin_log.toggled.connect(self._on_bin_log_toggled)
        side.addWidget(self.cb_bin_log)

        chan_btns = QHBoxLayout(); chan_btns.setContentsMargins(0, 0, 0, 0); chan_btns.setSpacing(6)
        self.btn_bin_ch_all = QPushButton("Show all"); self.btn_bin_ch_all.setObjectName("mini_btn")
        self.btn_bin_ch_all.setCursor(Qt.PointingHandCursor)
        self.btn_bin_ch_all.clicked.connect(lambda: self._set_all_bin_channels(True))
        self.btn_bin_ch_none = QPushButton("Hide all"); self.btn_bin_ch_none.setObjectName("mini_btn")
        self.btn_bin_ch_none.setCursor(Qt.PointingHandCursor)
        self.btn_bin_ch_none.clicked.connect(lambda: self._set_all_bin_channels(False))
        chan_btns.addWidget(self.btn_bin_ch_all); chan_btns.addWidget(self.btn_bin_ch_none)
        cbw = QWidget(); cbw.setLayout(chan_btns); side.addWidget(cbw)

        self.bin_ch_area = QScrollArea()
        self.bin_ch_area.setWidgetResizable(True)
        self.bin_ch_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.bin_ch_area.setMinimumHeight(110)
        self.bin_ch_inner = QWidget()
        self.bin_ch_lay = QVBoxLayout(self.bin_ch_inner)
        self.bin_ch_lay.setContentsMargins(0, 0, 0, 0); self.bin_ch_lay.setSpacing(3)
        self.bin_ch_lay.addStretch(1)
        self.bin_ch_area.setWidget(self.bin_ch_inner)
        side.addWidget(self.bin_ch_area)

        self.btn_bin_send1 = QPushButton("Send energy to spectrum")
        self.btn_bin_send1.setObjectName("open_btn")
        self.btn_bin_send1.setCursor(Qt.PointingHandCursor)
        self.btn_bin_send1.setToolTip("Hand the energy histogram to the Spectrum tab")
        self.btn_bin_send1.clicked.connect(self._send_bin_energy)
        side.addWidget(self.btn_bin_send1)

        # ── Energy traces ────────────────────────────────────────────────────
        side.addWidget(hsep()); side.addWidget(header("ENERGY TRACES"))
        self.lbl_bin_ntr = QLabel("Total traces: --"); self.lbl_bin_ntr.setObjectName("stat_key")
        side.addWidget(self.lbl_bin_ntr)
        self.ed_bin_base_tr = QLineEdit("100"); self.ed_bin_base_tr.setFixedWidth(80)
        self.ed_bin_base_tr.setToolTip("Baseline window (ns) averaged and subtracted")
        r, _ = labeled_row("Baseline (ns)", self.ed_bin_base_tr); side.addWidget(r)
        self.btn_bin_norm = QPushButton("Normalize baseline")
        self.btn_bin_norm.setObjectName("mini_btn")
        self.btn_bin_norm.setCursor(Qt.PointingHandCursor)
        self.btn_bin_norm.clicked.connect(self._normalize_bin_baseline)
        side.addWidget(self.btn_bin_norm)
        self.cb_bin_leg_tr = QCheckBox("Legend")
        self.cb_bin_leg_tr.toggled.connect(self._toggle_bin_legend_tr)
        side.addWidget(self.cb_bin_leg_tr)

        # ── Random traces ────────────────────────────────────────────────────
        side.addWidget(hsep()); side.addWidget(header("RANDOM TRACES"))
        self.ed_bin_ntraces = QLineEdit("10"); self.ed_bin_ntraces.setFixedWidth(80)
        self.ed_bin_ntraces.setToolTip("How many traces to sample at random")
        r, _ = labeled_row("No. traces", self.ed_bin_ntraces); side.addWidget(r)
        self.cb_bin_pileup = QCheckBox("Pileup only")
        self.cb_bin_cfderr = QCheckBox("CFD error only")
        self.cb_bin_flag = QCheckBox("Flagged only")
        for cb in (self.cb_bin_pileup, self.cb_bin_cfderr, self.cb_bin_flag):
            side.addWidget(cb)
        rbtns = QHBoxLayout(); rbtns.setContentsMargins(0, 0, 0, 0); rbtns.setSpacing(6)
        self.btn_bin_sample = QPushButton("Sample")
        self.btn_bin_sample.setObjectName("primary_btn")
        self.btn_bin_sample.setCursor(Qt.PointingHandCursor)
        self.btn_bin_sample.clicked.connect(self._sample_bin_traces)
        self.btn_bin_fft = QPushButton("FFT"); self.btn_bin_fft.setObjectName("action_btn")
        self.btn_bin_fft.setCheckable(True); self.btn_bin_fft.setCursor(Qt.PointingHandCursor)
        self.btn_bin_fft.setToolTip("Show the FFT magnitude of the sampled traces")
        self.btn_bin_fft.clicked.connect(self._toggle_bin_fft)
        rbtns.addWidget(self.btn_bin_sample, 1); rbtns.addWidget(self.btn_bin_fft, 0)
        rbw = QWidget(); rbw.setLayout(rbtns); side.addWidget(rbw)
        self.cb_bin_leg_rand = QCheckBox("Legend")
        self.cb_bin_leg_rand.toggled.connect(self._toggle_bin_legend_rand)
        side.addWidget(self.cb_bin_leg_rand)

        # ── Trace energy ─────────────────────────────────────────────────────
        side.addWidget(hsep()); side.addWidget(header("TRACE ENERGY"))
        tr_note = QLabel("Energy from integrating each trace (baseline-subtracted) "
                         "over the bounds below. Blank bounds = whole trace.")
        tr_note.setObjectName("stat_key"); tr_note.setWordWrap(True)
        side.addWidget(tr_note)
        self.ed_bin_a = QLineEdit(); self.ed_bin_a.setFixedWidth(80)
        self.ed_bin_a.setPlaceholderText("start"); self.ed_bin_a.setToolTip("Integration start (ns)")
        r, _ = labeled_row("Bound a (ns)", self.ed_bin_a); side.addWidget(r)
        self.ed_bin_b = QLineEdit(); self.ed_bin_b.setFixedWidth(80)
        self.ed_bin_b.setPlaceholderText("end"); self.ed_bin_b.setToolTip("Integration end (ns)")
        r, _ = labeled_row("Bound b (ns)", self.ed_bin_b); side.addWidget(r)
        self.ed_bin_base = QLineEdit("100"); self.ed_bin_base.setFixedWidth(80)
        self.ed_bin_base.setToolTip("Baseline window (ns) averaged and subtracted")
        r, _ = labeled_row("Baseline (ns)", self.ed_bin_base); side.addWidget(r)
        self.btn_bin_calc = QPushButton("Calc energy")
        self.btn_bin_calc.setObjectName("primary_btn")
        self.btn_bin_calc.setCursor(Qt.PointingHandCursor)
        self.btn_bin_calc.clicked.connect(self._calc_bin_trace_energy)
        side.addWidget(self.btn_bin_calc)
        self.btn_bin_send2 = QPushButton("Send trace energy to spectrum")
        self.btn_bin_send2.setObjectName("open_btn")
        self.btn_bin_send2.setCursor(Qt.PointingHandCursor)
        self.btn_bin_send2.clicked.connect(self._send_bin_trace_energy)
        side.addWidget(self.btn_bin_send2)

        self.lbl_bin_state = QLabel(""); self.lbl_bin_state.setObjectName("stat_key")
        self.lbl_bin_state.setWordWrap(True); side.addWidget(self.lbl_bin_state)
        side.addStretch(1)

        area.setWidget(inner)
        return area

    # ── loading / channel select ─────────────────────────────────────────────
    def _bin_inputs(self):
        date = self.ed_bin_date.text().strip()
        run_txt = self.ed_bin_run.text().strip()
        if not date or not run_txt:
            self._bin_state("Enter a date and run number first")
            return None
        try:
            runnr = int(run_txt)
        except ValueError:
            self._bin_state("Run number must be an integer")
            return None
        return date, runnr

    def _bin_bins(self):
        txt = self.ed_bin_bins.text().strip()
        if not txt:
            self.ed_bin_bins.setText("4098")
            return 4098
        try:
            return max(int(txt), 1)
        except ValueError:
            self.ed_bin_bins.setText("4098")
            return 4098

    def _load_bin(self):
        # Brighten the Load button so the click registers before the (blocking)
        # list-mode read freezes the UI -- same purple flash as "Load API file".
        self.c._flash_button(self.btn_bin_load)
        run = self._bin_inputs()
        if run is None:
            return
        date, runnr = run
        kind = self.cmb_bin_type.currentText()
        self._bin_state(f"Loading {kind.lower()} ...")
        QApplication.processEvents()
        try:
            if kind == "Trace data":
                df = helper_api.read_trace_data(date=date, runnr=runnr)
            elif kind == "Binary data":
                df = helper_api.read_binary_data(date=date, runnr=runnr)
            else:
                df = read_parquet_api.read_parquet_file(date=date, runnr=runnr)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._bin_state(f"Could not load data: {exc}")
            return
        if df is None or df.shape[0] == 0:
            self._bin_state("No events found in the loaded data")
            return
        self.df_bin = df
        self.df_bin_tr = None
        self.df_bin_rand = None
        # Build the per-channel visibility list (all visible by default).
        if "channel" in df.columns:
            self._bin_chans = sorted(int(c) for c in df.channel.unique())
        else:
            self._bin_chans = []
        palette = self._palette(len(self._bin_chans))
        self._bin_ch_visible = {c: True for c in self._bin_chans}
        self._bin_ch_colors = {c: palette[i] for i, c in enumerate(self._bin_chans)}
        self._populate_bin_channels()
        self._update_bin_ch_df()
        self._plot_bin_energy()
        self._bin_state(f"Loaded {df.shape[0]:,} events "
                        f"(channels: {', '.join(map(str, self._bin_chans)) or '--'})")

    def _populate_bin_channels(self):
        """Rebuild the per-channel visibility rows (swatch + label + checkbox)."""
        while self.bin_ch_lay.count() > 1:
            item = self.bin_ch_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._bin_ch_checks = {}
        for c in self._bin_chans:
            row = QWidget()
            h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
            swatch = QLabel(); swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background:{self._bin_ch_colors[c]}; border-radius:3px;")
            h.addWidget(swatch)
            lbl = QLabel(f"Channel {c}"); lbl.setObjectName("stat_key")
            h.addWidget(lbl); h.addStretch(1)
            cb = QCheckBox(); cb.setObjectName("vis_check")
            cb.setChecked(self._bin_ch_visible[c]); cb.setToolTip("Toggle visibility")
            cb.toggled.connect(lambda vis, ch=c: self._toggle_bin_channel(ch, vis))
            h.addWidget(cb)
            self._bin_ch_checks[c] = cb
            self.bin_ch_lay.insertWidget(self.bin_ch_lay.count() - 1, row)

    def _visible_channels(self):
        return [c for c in self._bin_chans if self._bin_ch_visible.get(c, True)]

    def _update_bin_ch_df(self):
        """Recompute the working view from the currently-visible channels."""
        if self.df_bin is None:
            self.df_bin_ch = None
            return
        if "channel" not in self.df_bin.columns:
            self.df_bin_ch = self.df_bin
            return
        vis = self._visible_channels()
        self.df_bin_ch = self.df_bin[self.df_bin.channel.isin(vis)]

    def _toggle_bin_channel(self, ch, visible):
        self._bin_ch_visible[ch] = visible
        self._update_bin_ch_df()
        self._plot_bin_energy()

    def _set_all_bin_channels(self, visible):
        if not self._bin_chans:
            return
        for c in self._bin_chans:
            self._bin_ch_visible[c] = visible
            cb = self._bin_ch_checks.get(c)
            if cb is not None:
                cb.blockSignals(True); cb.setChecked(visible); cb.blockSignals(False)
        self._update_bin_ch_df()
        self._plot_bin_energy()

    # ── energy histogram ─────────────────────────────────────────────────────
    def _plot_bin_energy(self):
        ax = self._bin_ax["erg"]
        ax.clear()
        if self.df_bin is None or "energy" not in self.df_bin.columns:
            self._draw_placeholder(ax, "No energy column in this data", "Channels",
                                   self._bin_yscale)
            self._bin_canvas["erg"].draw_idle()
            return
        bins = self._bin_bins()
        vis = self._visible_channels() if self._bin_chans else [None]
        for c in vis:
            sub = self.df_bin if c is None else self.df_bin[self.df_bin.channel == c]
            cts, ed = np.histogram(sub.energy, bins=bins)
            gam_x = (ed[1:] + ed[:-1]) / 2
            color = None if c is None else self._bin_ch_colors.get(c)
            label = "all" if c is None else f"Ch {c}: {cts.sum():,} cts"
            ax.plot(gam_x, cts, lw=0.9, color=color, label=label)
        # Combined histogram of the visible channels for "send to spectrum".
        if self.df_bin_ch is not None and self.df_bin_ch.shape[0] > 0:
            cts, ed = np.histogram(self.df_bin_ch.energy, bins=bins)
            self._bin_gam = cts
            self._bin_gam_x = (ed[1:] + ed[:-1]) / 2
        else:
            self._bin_gam = self._bin_gam_x = None
        ax.set_title("Energy histogram", color=T.TEXT_PRIMARY, fontsize=12)
        self._style_axis(ax, "Channels", self._bin_yscale)
        ax.set_ylabel("Counts", color=T.TEXT_DIM, fontsize=12)
        if 0 < len(vis) <= 16:
            ax.legend(loc="upper right", fontsize=9, facecolor=API_PLOT_BG,
                      edgecolor=T.BORDER, labelcolor=T.TEXT_DIM, framealpha=0.7)
        self._bin_fig["erg"].tight_layout()
        self._bin_canvas["erg"].draw_idle()
        self._bin_toolbar["erg"].update()
        self._arm_bin_span()

    def _arm_bin_span(self):
        # Interactive energy-span selection: drag on the Energy plot to filter
        # the events whose traces appear on the Energy-traces tab.
        self._bin_span = SpanSelector(
            self._bin_ax["erg"], self._on_bin_span, "horizontal",
            useblit=True, interactive=True,
            props=dict(alpha=0.3, facecolor=T.ACCENT_CYAN))

    def _on_bin_span(self, xmin, xmax):
        if self.df_bin_ch is None or "energy" not in self.df_bin_ch.columns:
            return
        if "trace" not in self.df_bin_ch.columns:
            self._bin_state("This data has no traces to show")
            return
        mask = (self.df_bin_ch["energy"] > xmin) & (self.df_bin_ch["energy"] < xmax)
        self.df_bin_tr = self.df_bin_ch[mask].reset_index(drop=True)
        n = self.df_bin_tr.shape[0]
        self.lbl_bin_ntr.setText(f"Total traces: {n}")
        self._plot_bin_traces(self.df_bin_tr, self._bin_ax["ergtr"],
                              self._bin_canvas["ergtr"], self._bin_fig["ergtr"],
                              self.cb_bin_leg_tr.isChecked(),
                              toolbar=self._bin_toolbar["ergtr"])
        self.bin_plot_tabs.setCurrentIndex(1)
        self._bin_state(f"Selected {n} traces in [{xmin:.1f}, {xmax:.1f}]")

    # ── trace plotting helpers ───────────────────────────────────────────────
    def _trace_time_axis(self, trace):
        return np.arange(0, len(trace), 1) * self.NS_PER_PT

    def _trace_label(self, rowobj, i):
        return (f"i:{i}; E:{rowobj.energy}; pu:{rowobj.pileup}, "
                f"flag:{rowobj.trace_flag}, CFDerr:{rowobj.CFD_error}")

    # A per-event legend is only built up to this many traces. Beyond it the
    # matplotlib "best" legend solver runs over every line on each redraw (e.g.
    # when zooming), which freezes/blanks the canvas -- so the legend is skipped.
    MAX_TRACE_LEGEND = 25

    def _plot_bin_traces(self, df, ax, canvas, fig, legend_on, toolbar=None):
        ax.clear()
        if df is None or df.shape[0] == 0:
            self._draw_placeholder(ax, "No traces to show", "Time (ns)", "linear")
            canvas.draw_idle()
            return
        # Only label traces when there are few enough for a usable legend.
        label = df.shape[0] <= self.MAX_TRACE_LEGEND
        n = 0
        for i in df.index:
            trace = df.loc[i].trace
            # Traces can vary in length (and some events have none) -- give each
            # its own time axis and skip empty ones.
            if trace is None or len(trace) == 0:
                continue
            ax.plot(self._trace_time_axis(trace), trace, lw=0.8,
                    label=self._trace_label(df.loc[i], i) if label else None)
            n += 1
        if n == 0:
            self._draw_placeholder(ax, "Selected events carry no traces",
                                   "Time (ns)", "linear")
            canvas.draw_idle()
            return
        self._style_axis(ax, "Time (ns)", "linear")
        ax.set_ylabel("Amplitude (a.u.)", color=T.TEXT_DIM, fontsize=12)
        if label:
            # Explicit location: the default "best" solver is O(lines) per draw.
            leg = ax.legend(loc="upper right", fontsize=8, facecolor=API_PLOT_BG,
                            edgecolor=T.BORDER, labelcolor=T.TEXT_DIM, framealpha=0.7)
            leg.set_visible(legend_on)
        fig.tight_layout()
        canvas.draw_idle()
        # Reset the toolbar's view stack so Home/zoom act on the new data extent.
        if toolbar is not None:
            toolbar.update()

    def _normalize_bin_baseline(self):
        if self.df_bin_tr is None or self.df_bin_tr.shape[0] == 0:
            self._bin_state("Select an energy span first")
            return
        ax = self._bin_ax["ergtr"]
        ax.clear()
        baseline = self._int_or(self.ed_bin_base_tr.text(), 100)
        self.ed_bin_base_tr.setText(str(baseline))
        base_pts = max(int(baseline / self.NS_PER_PT), 1)
        n = 0
        for i in self.df_bin_tr.index:
            trace = self.df_bin_tr.loc[i].trace
            # Per-trace baseline subtraction so variable-length / empty traces
            # don't force a ragged vstack.
            if trace is None or len(trace) == 0:
                continue
            trace = np.asarray(trace, dtype=float)
            norm = trace - trace[:base_pts].mean()
            ax.plot(self._trace_time_axis(trace), norm, lw=0.8)
            n += 1
        if n == 0:
            self._draw_placeholder(ax, "Selected events carry no traces",
                                   "Time (ns)", "linear")
            self._bin_canvas["ergtr"].draw_idle()
            self._bin_state("No traces to normalize")
            return
        self._style_axis(ax, "Time (ns)", "linear")
        ax.set_ylabel("Amplitude (a.u.)", color=T.TEXT_DIM, fontsize=12)
        self._bin_fig["ergtr"].tight_layout()
        self._bin_canvas["ergtr"].draw_idle()
        self._bin_toolbar["ergtr"].update()
        self._bin_state("Baseline-normalized energy traces")

    def _toggle_bin_legend_tr(self, checked):
        ax = self._bin_ax["ergtr"]
        if ax.get_legend() is not None:
            ax.get_legend().set_visible(checked)
            self._bin_canvas["ergtr"].draw_idle()

    # ── random traces / FFT ──────────────────────────────────────────────────
    def _sample_bin_traces(self):
        self.btn_bin_fft.setChecked(False)
        self._bin_fft = None
        if self.df_bin_ch is None:
            self._bin_state("Load data first")
            return
        if "trace" not in self.df_bin_ch.columns:
            self._bin_state("This data has no traces to sample")
            return
        n = self._int_or(self.ed_bin_ntraces.text(), 10)
        self.ed_bin_ntraces.setText(str(n))
        df = self.df_bin_ch
        if self.cb_bin_pileup.isChecked():
            df = df[df.pileup == True]  # noqa: E712
        elif self.cb_bin_cfderr.isChecked():
            df = df[df.CFD_error == True]  # noqa: E712
        elif self.cb_bin_flag.isChecked():
            df = df[df.trace_flag == True]  # noqa: E712
        if df.shape[0] == 0:
            self._bin_state("No events match that filter")
            return
        n = min(n, df.shape[0])
        self.df_bin_rand = df.sample(n).sort_index()
        self._plot_bin_traces(self.df_bin_rand, self._bin_ax["rand"],
                              self._bin_canvas["rand"], self._bin_fig["rand"],
                              self.cb_bin_leg_rand.isChecked(),
                              toolbar=self._bin_toolbar["rand"])
        self.bin_plot_tabs.setCurrentIndex(2)
        self._bin_state(f"Sampled {n} random traces")

    def _toggle_bin_fft(self):
        if not self.btn_bin_fft.isChecked():
            # Back to traces.
            if self.df_bin_rand is not None:
                self._plot_bin_traces(self.df_bin_rand, self._bin_ax["rand"],
                                      self._bin_canvas["rand"], self._bin_fig["rand"],
                                      self.cb_bin_leg_rand.isChecked(),
                                      toolbar=self._bin_toolbar["rand"])
            return
        if self.df_bin_rand is None or self.df_bin_rand.shape[0] == 0:
            self._bin_state("Sample some traces first")
            self.btn_bin_fft.setChecked(False)
            return
        ffts = []
        for tr in self.df_bin_rand.trace:
            if tr is None or len(tr) == 0:
                continue
            freqs, mag = apicalc.compute_fft(signal=tr, sampling_rate=self.SAMPLE_RATE)
            ffts.append((freqs, mag))
        if not ffts:
            self._bin_state("Sampled events carry no traces")
            self.btn_bin_fft.setChecked(False)
            return
        self._bin_fft = ffts
        ax = self._bin_ax["rand"]
        ax.clear()
        # Each trace gets its own frequency axis (lengths can differ).
        for freqs, mag in ffts:
            ax.plot(freqs / 1e6, mag, lw=0.8)
        self._style_axis(ax, "Frequency (MHz)", "log")
        ax.set_ylabel("Amplitude (a.u.)", color=T.TEXT_DIM, fontsize=12)
        self._bin_fig["rand"].tight_layout()
        self._bin_canvas["rand"].draw_idle()
        self._bin_toolbar["rand"].update()
        self._bin_state("FFT of the sampled traces")

    def _toggle_bin_legend_rand(self, checked):
        ax = self._bin_ax["rand"]
        if ax.get_legend() is not None:
            ax.get_legend().set_visible(checked)
            self._bin_canvas["rand"].draw_idle()

    # ── energy from trace integration ────────────────────────────────────────
    def _calc_bin_trace_energy(self):
        if self.df_bin_ch is None:
            self._bin_state("Load data first")
            return
        if "trace" not in self.df_bin_ch.columns:
            self._bin_state("This data has no traces to integrate")
            return
        bins = self._bin_bins()
        baseline = self._int_or(self.ed_bin_base.text(), 100)
        self.ed_bin_base.setText(str(baseline))
        base_pts = max(int(baseline / self.NS_PER_PT), 1)
        a_txt, b_txt = self.ed_bin_a.text().strip(), self.ed_bin_b.text().strip()
        a = int(int(a_txt) / self.NS_PER_PT) if a_txt else None
        b = int(int(b_txt) / self.NS_PER_PT) if b_txt else None
        # Integrate each trace individually so variable-length / empty traces
        # don't force a ragged vstack.
        erg = []
        for trace in self.df_bin_ch.trace:
            if trace is None or len(trace) == 0:
                continue
            tr = np.asarray(trace, dtype=float)
            tr = tr - tr[:base_pts].mean()
            lo = 0 if a is None else a
            hi = len(tr) if b is None else b
            erg.append(tr[lo:hi].sum())
        if not erg:
            self._bin_state("Selected events carry no traces to integrate")
            return
        cts, edg = np.histogram(np.asarray(erg), bins=bins)
        self._bin_gam2 = cts
        self._bin_gam_x2 = (edg[1:] + edg[:-1]) / 2
        ax = self._bin_ax["own"]
        ax.clear()
        ax.plot(self._bin_gam_x2, cts, color=T.ACCENT_CYAN, lw=0.9)
        ax.set_title("Energy from trace integration", color=T.TEXT_PRIMARY, fontsize=12)
        self._style_axis(ax, "Integrated amplitude", "log")
        ax.set_ylabel("Counts", color=T.TEXT_DIM, fontsize=12)
        self._bin_fig["own"].tight_layout()
        self._bin_canvas["own"].draw_idle()
        self._bin_toolbar["own"].update()
        self.bin_plot_tabs.setCurrentIndex(3)
        self._bin_state("Computed energy from trace integration")

    # ── send to spectrum ─────────────────────────────────────────────────────
    def _send_bin_energy(self):
        if self._bin_gam is None:
            self._bin_state("Load data (energy histogram) first")
            return
        self._send_spectrum(self._bin_gam, self._bin_gam_x, "API binary energy")

    def _send_bin_trace_energy(self):
        if self._bin_gam2 is None:
            self._bin_state("Calc energy first")
            return
        self._send_spectrum(self._bin_gam2, self._bin_gam_x2,
                            "API binary trace energy")

    def _send_spectrum(self, counts, energies, name):
        try:
            spect = sp.Spectrum(counts=counts, energies=energies, e_units="channels")
        except Exception as exc:  # noqa: BLE001
            self._bin_state(f"Could not build spectrum: {exc}")
            return
        self.c.app.load_external_spectrum(spect, name, switch_tab=False)
        self._bin_state(f"Sent '{name}' to the Spectrum tab")

    # ── misc ──────────────────────────────────────────────────────────────────
    def _on_bin_log_toggled(self, checked):
        self._bin_yscale = "log" if checked else "linear"
        ax = self._bin_ax["erg"]
        ax.set_yscale(self._bin_yscale)
        self._bin_canvas["erg"].draw_idle()

    @staticmethod
    def _int_or(text, default):
        try:
            return int(text)
        except (TypeError, ValueError):
            return default
