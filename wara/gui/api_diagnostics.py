"""Diagnostics dialog for the API tab (``DiagnosticsDialog``): MCA spectra,
trace/binary waveforms and per-run statistics for a loaded run."""
import traceback

import numpy as np

from matplotlib.figure import Figure
from matplotlib import cm, colormaps
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QScrollArea,
    QSizePolicy, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtCore import Qt, QSize

from wara import read_parquet_api, apicalc, helper_api
from wara import pixie_trace_analysis as pta
from wara import spectrum as sp

from . import theme as T
from .widgets import hsep, header, labeled_row
from .api_common import API_PLOT_BG, _combo_row, _draw_axes_placeholder
from .api_dialogs import StatsInfoDialog, BinaryStatsDialog


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
      trace sampler (with pileup / CFD-error / flagged filters, an FFT toggle
      and "Fast Filter" / "CFD" overlays that reconstruct the Pixie-16's
      internal trigger and timing filters offline, see
      :mod:`wara.pixie_trace_analysis`), and an energy spectrum reconstructed
      by integrating the traces.
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
        self._mca_date = self._mca_run = None  # run that produced mca_data
        self._mca_visible = []     # per-spectrum visibility flags
        self._mca_colors = []      # per-spectrum plot color (hex)
        self._mca_checks = []      # the visibility QCheckBoxes
        self._mca_ax = {}
        self._mca_fig = {}
        self._mca_canvas = {}
        self._mca_toolbar = {}
        self._stats_dlg = None     # StatsInfoDialog (created lazily)
        self._bin_stats_dlg = None  # BinaryStatsDialog (created lazily)
        self._bin_kind = None      # source of the loaded df ("Trace data", ...)

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
        self._bin_date = self._bin_run = None  # run that produced df_bin
        self._bin_align = False     # LGL-sum time alignment on/off
        self._bin_lgl = {}          # {channel: (L, G)} slow-filter windows
        # {channel: (FL, FG, FastThresh, CFDThresh)} fast-trigger / CFD geometry
        # for the offline reconstruction overlaid on the Random-traces tab.
        self._bin_fast = {}
        self._bin_rand_twins = []  # twin y-axes created for those overlays
        self._bin_span = None      # SpanSelector on the energy plot
        self._bin_span_range = None  # (lo, hi) energy of the active highlight
        self._bin_yscale = "log"
        self._bin_gam = self._bin_gam_x = None    # energy hist (send to spectrum)
        self._bin_gam2 = self._bin_gam_x2 = None  # trace-integral hist
        self._bin_fft = None       # (freqs, list-of-magnitudes) when FFT is on

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_binary_tab(), "Binary")
        self.tabs.addTab(self._build_mca_tab(), "MCA")
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
        self._mca_date, self._mca_run = date, runnr
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
                              f"{self._mca_date}-{self._mca_run}-CH{i}"))
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

    def _open_bin_stats(self):
        """Per-channel statistics of the loaded list-mode data (Binary tab)."""
        if self.df_bin is None or self.df_bin.shape[0] == 0:
            self._bin_state("Load data first")
            return
        cfd = (self.cmb_bin_cfd.currentText()
               if self._bin_kind == "Trace data" else None)
        if self._bin_stats_dlg is None:
            self._bin_stats_dlg = BinaryStatsDialog(self)
        self._bin_stats_dlg.populate(
            self.df_bin, f"{self._bin_date}  ·  run {self._bin_run}",
            self._bin_kind, cfd=cfd)
        self._bin_stats_dlg.show()
        self._bin_stats_dlg.raise_()
        self._bin_stats_dlg.activateWindow()
        self._bin_state(f"Statistics for {self.df_bin.shape[0]:,} events")

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
        # A colour-coded view of the raw list-mode dataframe (no canvas).
        self.bin_plot_tabs.addTab(self._build_bin_table_tab(), "Data table")
        self._fit_tabbar(self.bin_plot_tabs)
        row.addWidget(self.bin_plot_tabs, 1)

        # Control side (scrollable -- only the group for the active plot tab shows).
        row.addWidget(self._build_binary_controls(), 0)
        self.bin_plot_tabs.currentChanged.connect(self._on_bin_tab_changed)
        self._on_bin_tab_changed(self.bin_plot_tabs.currentIndex())
        return tab

    def _on_bin_tab_changed(self, index):
        for i, grp in enumerate(self._bin_ctrl_groups):
            grp.setVisible(i == index)
        # The RUN group (date / run / source / load) lives on the Energy tab only.
        self._bin_run_grp.setVisible(self.bin_plot_tabs.tabText(index) == "Energy")
        # Fill the data table lazily the first time its tab is shown.
        if (self.bin_plot_tabs.tabText(index) == "Data table"
                and self.bin_tbl.rowCount() == 0 and self.df_bin_ch is not None):
            self._refresh_bin_table()

    # ── data-table view ──────────────────────────────────────────────────────
    def _build_bin_table_tab(self):
        """A colour-coded table of the loaded list-mode dataframe.

        Each numeric column is min-max normalized and mapped through a
        colormap, so magnitudes and per-column patterns (e.g. one channel
        dominating, or a block of piled-up events) stand out at a glance
        instead of reading as undifferentiated numbers.
        """
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        self.bin_tbl = QTableWidget()
        self.bin_tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.bin_tbl.setSelectionMode(QAbstractItemView.NoSelection)
        self.bin_tbl.setAlternatingRowColors(False)
        self.bin_tbl.setSortingEnabled(False)
        self.bin_tbl.verticalHeader().setDefaultSectionSize(22)
        self.bin_tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        lay.addWidget(self.bin_tbl)
        return wrap

    def _sync_bin_table(self):
        """Keep the data table in step with the visible-channel view.

        Rebuilds it immediately when its tab is showing; otherwise blanks it so
        the next visit to the tab repopulates lazily (channel toggles shouldn't
        pay for a table redraw the user isn't looking at).
        """
        tbl = getattr(self, "bin_tbl", None)
        if tbl is None:
            return
        idx = self.bin_plot_tabs.currentIndex()
        if self.bin_plot_tabs.tabText(idx) == "Data table":
            self._refresh_bin_table()
        else:
            tbl.setRowCount(0)

    @staticmethod
    def _tbl_numeric(series):
        """Column values as a float array if numeric/boolean, else ``None``.

        Object columns that hold plain bools/numbers (e.g. the ``pileup`` /
        ``CFD_error`` flags) are converted element-wise; array columns (traces)
        and strings stay ``None`` so they are shown but not colour-shaded.
        """
        dt = series.dtype
        if dt == bool or np.issubdtype(dt, np.number):
            try:
                return series.to_numpy(dtype=float)
            except (TypeError, ValueError):
                return None
        if dt == object:
            vals = series.to_numpy()
            out = np.full(len(vals), np.nan, dtype=float)
            ok = False
            for i, v in enumerate(vals):
                if isinstance(v, (bool, np.bool_)):
                    out[i] = 1.0 if v else 0.0; ok = True
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    out[i] = float(v); ok = True
            return out if ok else None
        return None

    @staticmethod
    def _tbl_fmt(val):
        """Compact cell text for any dataframe value."""
        if isinstance(val, (np.ndarray, list, tuple)):
            return f"[{len(val)}]"
        if isinstance(val, (bool, np.bool_)):
            return "True" if val else "False"
        if isinstance(val, (float, np.floating)):
            return f"{val:.4g}"
        return str(val)

    def _refresh_bin_table(self):
        """Fill the data table from the visible-channel view, colour-coding
        each numeric column low-to-high with the selected colormap."""
        tbl = self.bin_tbl
        df = self.df_bin_ch
        tbl.setUpdatesEnabled(False)
        try:
            tbl.clear()
            if df is None or df.shape[0] == 0:
                tbl.setRowCount(0); tbl.setColumnCount(0)
                self._bin_state("Load a run to fill the data table")
                return
            n = self._int_or(self.ed_bin_tbl_rows.text(), 500)
            n = max(min(n, df.shape[0]), 1)
            self.ed_bin_tbl_rows.setText(str(n))
            view = df.head(n)
            cols = list(view.columns)
            tbl.setColumnCount(len(cols))
            tbl.setRowCount(n)
            tbl.setHorizontalHeaderLabels([str(c) for c in cols])
            color = self.cb_bin_tbl_color.isChecked()
            cmap = colormaps[self.cmb_bin_tbl_cmap.currentText()]
            for cix, col in enumerate(cols):
                s = view[col]
                num = self._tbl_numeric(s) if color else None
                lo = hi = None
                if num is not None:
                    finite = num[np.isfinite(num)]
                    if finite.size:
                        lo, hi = float(finite.min()), float(finite.max())
                for rix in range(n):
                    item = QTableWidgetItem(self._tbl_fmt(s.iat[rix]))
                    item.setTextAlignment(Qt.AlignCenter)
                    if lo is not None and np.isfinite(num[rix]):
                        t = 0.5 if hi == lo else (num[rix] - lo) / (hi - lo)
                        r, g, b, _ = cmap(float(t))
                        item.setBackground(QBrush(
                            QColor(int(r * 255), int(g * 255), int(b * 255))))
                        lum = 0.299 * r + 0.587 * g + 0.114 * b
                        item.setForeground(QBrush(QColor(
                            "#000000" if lum > 0.55 else "#ffffff")))
                    tbl.setItem(rix, cix, item)
        finally:
            tbl.setUpdatesEnabled(True)
        self._bin_state(f"Data table: {n} of {df.shape[0]:,} rows, "
                        f"{len(cols)} columns")

    def _build_binary_controls(self):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setFixedWidth(320)
        inner = QWidget()
        side = QVBoxLayout(inner); side.setSpacing(6)
        side.setContentsMargins(6, 6, 6, 6)

        # ── Run / load ─────────────────────────────────────────────────────────
        # The run properties (date / run / source / load) belong to the Energy
        # sub-tab only -- you pick and load a run there, then explore it on the
        # other sub-tabs. Wrapped in its own group so it hides off the Energy
        # tab like the per-tab option groups below.
        self._bin_run_grp = QWidget()
        run_lay = QVBoxLayout(self._bin_run_grp)
        run_lay.setContentsMargins(0, 0, 0, 0); run_lay.setSpacing(6)
        run_lay.addWidget(header("RUN"))
        self.ed_bin_date = QLineEdit(); self.ed_bin_date.setPlaceholderText("YYYY-MM-DD")
        r, _ = labeled_row("Date", self.ed_bin_date); run_lay.addWidget(r)
        self.ed_bin_run = QLineEdit(); self.ed_bin_run.setPlaceholderText("e.g. 8")
        r, _ = labeled_row("Run", self.ed_bin_run); run_lay.addWidget(r)
        self.cmb_bin_type = QComboBox()
        self.cmb_bin_type.addItems(["Trace data", "Binary data", "Parquet"])
        self.cmb_bin_type.setToolTip("Which list-mode source to read for this run")
        run_lay.addWidget(_combo_row("Source", self.cmb_bin_type))
        self.cmb_bin_cfd = QComboBox()
        self.cmb_bin_cfd.addItems(["All", "CFD on", "CFD off"])
        self.cmb_bin_cfd.setToolTip(
            "Filter trace acquisitions by CFD state (Trace data only). "
            "'CFD on' = CFD unchanged/enabled, 'CFD off' = CFD disabled.")
        self._cmb_bin_cfd_row = _combo_row("CFD", self.cmb_bin_cfd)
        run_lay.addWidget(self._cmb_bin_cfd_row)
        self.cb_bin_align = QCheckBox("Align on LGL sums")
        self.cb_bin_align.setToolTip(
            "Locate each trace's energy-filter (leading/gap/leading) sum "
            "windows and overlay them on the Energy-traces and Random-traces "
            "plots. Traces are left untouched -- the sum windows are moved to "
            "match each trace. Applied on the fly -- no reload needed.")
        self.cb_bin_align.toggled.connect(self._toggle_bin_align)
        self._cb_bin_align_row = self.cb_bin_align
        run_lay.addWidget(self._cb_bin_align_row)
        # CFD filtering and trace alignment only apply to raw trace data
        self.cmb_bin_type.currentTextChanged.connect(
            lambda k: self._cmb_bin_cfd_row.setVisible(k == "Trace data"))
        self.cmb_bin_type.currentTextChanged.connect(
            lambda k: self._cb_bin_align_row.setVisible(k == "Trace data"))
        _is_trace = self.cmb_bin_type.currentText() == "Trace data"
        self._cmb_bin_cfd_row.setVisible(_is_trace)
        self._cb_bin_align_row.setVisible(_is_trace)
        self.ed_bin_bins = QLineEdit("4098"); self.ed_bin_bins.setFixedWidth(80)
        self.ed_bin_bins.setToolTip("Number of bins for the energy histograms")
        r, _ = labeled_row("Bins", self.ed_bin_bins); run_lay.addWidget(r)
        self.btn_bin_load = QPushButton("Load")
        self.btn_bin_load.setObjectName("open_btn")
        self.btn_bin_load.setCursor(Qt.PointingHandCursor)
        self.btn_bin_load.setToolTip("Read the run's list-mode data and plot the "
                                     "per-channel energy histogram")
        self.btn_bin_load.clicked.connect(self._load_bin)
        run_lay.addWidget(self.btn_bin_load)

        self.btn_bin_info = QPushButton("Run stats...")
        self.btn_bin_info.setObjectName("action_btn")
        self.btn_bin_info.setCursor(Qt.PointingHandCursor)
        self.btn_bin_info.setEnabled(False)
        self.btn_bin_info.setToolTip("Per-channel statistics of the loaded data: "
                                     "counts, rates, pile-up / CFD-error / trace-"
                                     "flag fractions and energy summaries")
        self.btn_bin_info.clicked.connect(self._open_bin_stats)
        run_lay.addWidget(self.btn_bin_info)
        run_lay.addWidget(hsep())
        side.addWidget(self._bin_run_grp)
        self._bin_ctrl_groups = []

        # ── Energy plot ──────────────────────────────────────────────────────
        erg_grp = QWidget(); erg_lay = QVBoxLayout(erg_grp)
        erg_lay.setContentsMargins(0, 0, 0, 0); erg_lay.setSpacing(6)
        erg_lay.addWidget(header("ENERGY"))
        en_note = QLabel("Drag a span on the Energy plot to pick events for the "
                         "Energy-traces tab. Toggle channels below; the visible "
                         "ones drive every view.")
        en_note.setObjectName("stat_key"); en_note.setWordWrap(True)
        erg_lay.addWidget(en_note)
        self.cb_bin_log = QCheckBox("Log Y"); self.cb_bin_log.setChecked(True)
        self.cb_bin_log.setToolTip("Logarithmic y-axis on the energy histogram")
        self.cb_bin_log.toggled.connect(self._on_bin_log_toggled)
        erg_lay.addWidget(self.cb_bin_log)

        chan_btns = QHBoxLayout(); chan_btns.setContentsMargins(0, 0, 0, 0); chan_btns.setSpacing(6)
        self.btn_bin_ch_all = QPushButton("Show all"); self.btn_bin_ch_all.setObjectName("mini_btn")
        self.btn_bin_ch_all.setCursor(Qt.PointingHandCursor)
        self.btn_bin_ch_all.clicked.connect(lambda: self._set_all_bin_channels(True))
        self.btn_bin_ch_none = QPushButton("Hide all"); self.btn_bin_ch_none.setObjectName("mini_btn")
        self.btn_bin_ch_none.setCursor(Qt.PointingHandCursor)
        self.btn_bin_ch_none.clicked.connect(lambda: self._set_all_bin_channels(False))
        chan_btns.addWidget(self.btn_bin_ch_all); chan_btns.addWidget(self.btn_bin_ch_none)
        cbw = QWidget(); cbw.setLayout(chan_btns); erg_lay.addWidget(cbw)

        self.bin_ch_area = QScrollArea()
        self.bin_ch_area.setWidgetResizable(True)
        self.bin_ch_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.bin_ch_area.setMinimumHeight(110)
        self.bin_ch_inner = QWidget()
        self.bin_ch_lay = QVBoxLayout(self.bin_ch_inner)
        self.bin_ch_lay.setContentsMargins(0, 0, 0, 0); self.bin_ch_lay.setSpacing(3)
        self.bin_ch_lay.addStretch(1)
        self.bin_ch_area.setWidget(self.bin_ch_inner)
        erg_lay.addWidget(self.bin_ch_area)

        self.btn_bin_send1 = QPushButton("Send energy to spectrum")
        self.btn_bin_send1.setObjectName("open_btn")
        self.btn_bin_send1.setCursor(Qt.PointingHandCursor)
        self.btn_bin_send1.setToolTip("Hand the energy histogram to the Spectrum tab")
        self.btn_bin_send1.clicked.connect(self._send_bin_energy)
        erg_lay.addWidget(self.btn_bin_send1)
        side.addWidget(erg_grp); self._bin_ctrl_groups.append(erg_grp)

        # ── Energy traces ────────────────────────────────────────────────────
        ergtr_grp = QWidget(); ergtr_lay = QVBoxLayout(ergtr_grp)
        ergtr_lay.setContentsMargins(0, 0, 0, 0); ergtr_lay.setSpacing(6)
        ergtr_lay.addWidget(header("ENERGY TRACES"))
        self.lbl_bin_ntr = QLabel("Total traces: --"); self.lbl_bin_ntr.setObjectName("stat_key")
        ergtr_lay.addWidget(self.lbl_bin_ntr)
        self.ed_bin_base_tr = QLineEdit("100"); self.ed_bin_base_tr.setFixedWidth(80)
        self.ed_bin_base_tr.setToolTip("Baseline window (ns) averaged and subtracted")
        r, _ = labeled_row("Baseline (ns)", self.ed_bin_base_tr); ergtr_lay.addWidget(r)
        self.btn_bin_norm = QPushButton("Normalize baseline")
        self.btn_bin_norm.setObjectName("mini_btn")
        self.btn_bin_norm.setCursor(Qt.PointingHandCursor)
        self.btn_bin_norm.clicked.connect(self._normalize_bin_baseline)
        ergtr_lay.addWidget(self.btn_bin_norm)
        self.cb_bin_leg_tr = QCheckBox("Legend")
        self.cb_bin_leg_tr.toggled.connect(self._toggle_bin_legend_tr)
        ergtr_lay.addWidget(self.cb_bin_leg_tr)
        side.addWidget(ergtr_grp); self._bin_ctrl_groups.append(ergtr_grp)

        # ── Random traces ────────────────────────────────────────────────────
        rand_grp = QWidget(); rand_lay = QVBoxLayout(rand_grp)
        rand_lay.setContentsMargins(0, 0, 0, 0); rand_lay.setSpacing(6)
        rand_lay.addWidget(header("RANDOM TRACES"))
        self.ed_bin_ntraces = QLineEdit("10"); self.ed_bin_ntraces.setFixedWidth(80)
        self.ed_bin_ntraces.setToolTip("How many traces to sample at random")
        r, _ = labeled_row("No. traces", self.ed_bin_ntraces); rand_lay.addWidget(r)
        self.cb_bin_pileup = QCheckBox("Pileup only")
        self.cb_bin_cfderr = QCheckBox("CFD error only")
        self.cb_bin_flag = QCheckBox("Flagged only")
        for cb in (self.cb_bin_pileup, self.cb_bin_cfderr, self.cb_bin_flag):
            rand_lay.addWidget(cb)
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
        rbw = QWidget(); rbw.setLayout(rbtns); rand_lay.addWidget(rbw)

        # ── offline fast-trigger / CFD reconstruction overlays ───────────────
        ovl_note = QLabel("Overlay the fast filter / CFD the Pixie-16 computes "
                          "internally, reconstructed offline from each trace "
                          "and the run's DSP settings.")
        ovl_note.setObjectName("stat_key"); ovl_note.setWordWrap(True)
        rand_lay.addWidget(ovl_note)
        self.cb_bin_ff = QCheckBox("Fast Filter")
        self.cb_bin_ff.setToolTip(
            "Plot the trapezoidal fast filter (manual Eq 3-1) of every sampled "
            "trace on a second y-axis, together with the channel's FastThresh "
            "register -- the fast trigger fires where the filter crosses it. "
            "Needs the run's settings file.")
        self.cb_bin_cfd = QCheckBox("CFD")
        self.cb_bin_cfd.setToolTip(
            "Plot the 500 MHz CFD trace (manual Eq 3-5) of every sampled trace "
            "on a second y-axis, together with the channel's CFDThresh "
            "register -- the timestamp comes from the zero crossing after the "
            "trace rises above it. Needs the run's settings file.")
        for cb in (self.cb_bin_ff, self.cb_bin_cfd):
            cb.setEnabled(False)
            cb.toggled.connect(self._toggle_bin_filters)
            rand_lay.addWidget(cb)

        # Custom-firmware CFD weight -- a sub-option of the CFD box above. Stock
        # 500 MHz firmware fixes the CFD weight at w=1 (correct for our standard
        # Pixie-16); our custom firmware build runs it at w=0.3125. Checked by
        # default because that is the firmware in use here. Indented under CFD
        # and only active while CFD is on, to show it modifies that overlay.
        self.cb_bin_cfd_custom_w = QCheckBox(
            f"Custom firmware (w={pta.CFD_W_CUSTOM:g})")
        self.cb_bin_cfd_custom_w.setChecked(True)
        self.cb_bin_cfd_custom_w.setToolTip(
            f"Reconstruct the CFD with the custom-firmware weight "
            f"w={pta.CFD_W_CUSTOM:g} instead of the stock w={pta.CFD_W:g} "
            "(manual Eq 3-5). Leave this on for our custom firmware; uncheck "
            "it for a standard Pixie-16, where w=1 is correct.")
        self.cb_bin_cfd_custom_w.toggled.connect(self._toggle_bin_filters)
        # Indent it under the CFD checkbox so it reads as a child option.
        _cfd_w_row = QWidget(); _cfd_w_lay = QHBoxLayout(_cfd_w_row)
        _cfd_w_lay.setContentsMargins(20, 0, 0, 0); _cfd_w_lay.setSpacing(0)
        _cfd_w_lay.addWidget(self.cb_bin_cfd_custom_w)
        rand_lay.addWidget(_cfd_w_row)
        # The sub-option is only meaningful while the CFD overlay is drawn.
        self.cb_bin_cfd_custom_w.setEnabled(False)
        self.cb_bin_cfd.toggled.connect(self.cb_bin_cfd_custom_w.setEnabled)

        self.cb_bin_leg_rand = QCheckBox("Legend")
        self.cb_bin_leg_rand.toggled.connect(self._toggle_bin_legend_rand)
        rand_lay.addWidget(self.cb_bin_leg_rand)
        side.addWidget(rand_grp); self._bin_ctrl_groups.append(rand_grp)

        # ── Trace energy ─────────────────────────────────────────────────────
        own_grp = QWidget(); own_lay = QVBoxLayout(own_grp)
        own_lay.setContentsMargins(0, 0, 0, 0); own_lay.setSpacing(6)
        own_lay.addWidget(header("TRACE ENERGY"))
        tr_note = QLabel("Energy from integrating each trace (baseline-subtracted) "
                         "over the bounds below. Blank bounds = whole trace.")
        tr_note.setObjectName("stat_key"); tr_note.setWordWrap(True)
        own_lay.addWidget(tr_note)
        self.ed_bin_a = QLineEdit(); self.ed_bin_a.setFixedWidth(80)
        self.ed_bin_a.setPlaceholderText("start"); self.ed_bin_a.setToolTip("Integration start (ns)")
        r, _ = labeled_row("Bound a (ns)", self.ed_bin_a); own_lay.addWidget(r)
        self.ed_bin_b = QLineEdit(); self.ed_bin_b.setFixedWidth(80)
        self.ed_bin_b.setPlaceholderText("end"); self.ed_bin_b.setToolTip("Integration end (ns)")
        r, _ = labeled_row("Bound b (ns)", self.ed_bin_b); own_lay.addWidget(r)
        self.ed_bin_base = QLineEdit("100"); self.ed_bin_base.setFixedWidth(80)
        self.ed_bin_base.setToolTip("Baseline window (ns) averaged and subtracted")
        r, _ = labeled_row("Baseline (ns)", self.ed_bin_base); own_lay.addWidget(r)
        self.btn_bin_calc = QPushButton("Calc energy")
        self.btn_bin_calc.setObjectName("primary_btn")
        self.btn_bin_calc.setCursor(Qt.PointingHandCursor)
        self.btn_bin_calc.clicked.connect(self._calc_bin_trace_energy)
        own_lay.addWidget(self.btn_bin_calc)
        self.btn_bin_send2 = QPushButton("Send trace energy to spectrum")
        self.btn_bin_send2.setObjectName("open_btn")
        self.btn_bin_send2.setCursor(Qt.PointingHandCursor)
        self.btn_bin_send2.clicked.connect(self._send_bin_trace_energy)
        own_lay.addWidget(self.btn_bin_send2)
        side.addWidget(own_grp); self._bin_ctrl_groups.append(own_grp)

        # ── Data table ───────────────────────────────────────────────────────
        tbl_grp = QWidget(); tbl_lay = QVBoxLayout(tbl_grp)
        tbl_lay.setContentsMargins(0, 0, 0, 0); tbl_lay.setSpacing(6)
        tbl_lay.addWidget(header("DATA TABLE"))
        tbl_note = QLabel("A colour-coded view of the loaded list-mode data. Each "
                          "numeric column is shaded low-to-high so patterns pop "
                          "out. Only the first rows are shown, for speed.")
        tbl_note.setObjectName("stat_key"); tbl_note.setWordWrap(True)
        tbl_lay.addWidget(tbl_note)
        self.ed_bin_tbl_rows = QLineEdit("500"); self.ed_bin_tbl_rows.setFixedWidth(80)
        self.ed_bin_tbl_rows.setToolTip("How many rows (from the top) to display; "
                                        "raise it to see more of the dataframe")
        r, _ = labeled_row("Rows", self.ed_bin_tbl_rows); tbl_lay.addWidget(r)
        self.cmb_bin_tbl_cmap = QComboBox()
        self.cmb_bin_tbl_cmap.addItems(["viridis", "cividis", "magma", "coolwarm"])
        self.cmb_bin_tbl_cmap.setToolTip("Colormap used to shade the numeric cells")
        self.cmb_bin_tbl_cmap.currentTextChanged.connect(
            lambda _t: self._refresh_bin_table())
        tbl_lay.addWidget(_combo_row("Colormap", self.cmb_bin_tbl_cmap))
        self.cb_bin_tbl_color = QCheckBox("Colour cells"); self.cb_bin_tbl_color.setChecked(True)
        self.cb_bin_tbl_color.setToolTip("Shade each numeric cell by its value; "
                                         "uncheck for a plain table")
        self.cb_bin_tbl_color.toggled.connect(lambda _c: self._refresh_bin_table())
        tbl_lay.addWidget(self.cb_bin_tbl_color)
        self.btn_bin_tbl_refresh = QPushButton("Refresh table")
        self.btn_bin_tbl_refresh.setObjectName("primary_btn")
        self.btn_bin_tbl_refresh.setCursor(Qt.PointingHandCursor)
        self.btn_bin_tbl_refresh.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.btn_bin_tbl_refresh.setToolTip("Rebuild the table from the currently "
                                            "visible channels")
        self.btn_bin_tbl_refresh.clicked.connect(self._refresh_bin_table)
        tbl_lay.addWidget(self.btn_bin_tbl_refresh)
        side.addWidget(tbl_grp); self._bin_ctrl_groups.append(tbl_grp)

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
                cfd = {"CFD on": "on", "CFD off": "off"}.get(
                    self.cmb_bin_cfd.currentText())
                # Raw traces are kept in memory; alignment is applied at plot
                # time so the "Align on LGL sums" checkbox needs no reload.
                df = helper_api.read_trace_data(date=date, runnr=runnr, cfd=cfd)
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
        self._bin_date, self._bin_run = date, runnr
        self._bin_kind = kind
        self.btn_bin_info.setEnabled(True)
        self.df_bin = df
        self.df_bin_tr = None
        self.df_bin_rand = None
        self._bin_span_range = None   # drop any highlight from the previous run
        # Build the per-channel visibility list (all visible by default).
        if "channel" in df.columns:
            self._bin_chans = sorted(int(c) for c in df.channel.unique())
        else:
            self._bin_chans = []
        # Per-channel energy-filter window geometry for LGL-sum alignment. Only
        # available for trace data that carries the Esum columns and a settings
        # file; if unavailable the align checkbox is disabled.
        self._bin_lgl = {}
        has_esums = {"Esum_trailing", "Esum_gap", "Esum_leading"} <= set(df.columns)
        if kind == "Trace data" and has_esums:
            for c in self._bin_chans:
                try:
                    self._bin_lgl[c] = helper_api.read_slow_filter_geometry(
                        date, runnr, c)
                except Exception:  # noqa: BLE001
                    pass
        can_align = bool(self._bin_lgl)
        self.cb_bin_align.setEnabled(can_align)
        if not can_align and self.cb_bin_align.isChecked():
            self.cb_bin_align.blockSignals(True)
            self.cb_bin_align.setChecked(False)
            self.cb_bin_align.blockSignals(False)
            self._bin_align = False
        # Per-channel fast-filter geometry / thresholds for the offline fast
        # trigger + CFD reconstruction overlaid on the Random-traces tab. Comes
        # from the run's settings JSON, so it works for trace *and* binary data
        # -- but only matters where traces were actually stored.
        self._bin_fast = {}
        if "trace" in df.columns:
            for c in self._bin_chans:
                try:
                    FL, FG, fast_thresh = pta.read_fast_trigger_geometry(
                        date, runnr, c)
                    cfd_thresh = pta.read_cfd_threshold(date, runnr, c)
                except Exception:  # noqa: BLE001
                    continue
                self._bin_fast[c] = (FL, FG, fast_thresh, cfd_thresh)
        self._set_bin_filter_boxes_enabled(bool(self._bin_fast))
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
        elif "channel" not in self.df_bin.columns:
            self.df_bin_ch = self.df_bin
        else:
            vis = self._visible_channels()
            self.df_bin_ch = self.df_bin[self.df_bin.channel.isin(vis)]
        self._sync_bin_table()

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
        # Remember the highlight so the random-trace sampler can draw only from
        # this energy window (an empty drag clears it -> sample from everything).
        self._bin_span_range = (xmin, xmax) if xmax > xmin else None
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

    def _aligned_trace(self, rowobj):
        """Return ``(trace, bounds)`` for one event.

        When "Align on LGL sums" is on, the trace is left untouched and
        ``bounds`` are the four energy-filter (trailing/gap/leading) window
        edges (in samples) located on *that trace* by :func:`find_LGL_sums` --
        i.e. the sum windows are moved to match the trace, not the other way
        around. When off or unavailable, the raw trace is returned with
        ``bounds=None``.
        """
        trace = rowobj.trace
        if trace is None or len(trace) == 0 or not self._bin_align:
            return trace, None
        geo = self._bin_lgl.get(int(getattr(rowobj, "channel", -1)))
        if geo is None:
            return trace, None
        L, G = geo
        try:
            res = helper_api.find_LGL_sums(
                trace, rowobj.Esum_trailing, rowobj.Esum_gap,
                rowobj.Esum_leading, L, G, relative=True)
        except Exception:  # noqa: BLE001
            return trace, None
        return trace, res["bounds"]

    def _draw_lgl_windows(self, ax, bounds_set):
        """Overlay the trailing | gap | leading (L | G | L) window regions."""
        for b in bounds_set:
            xs = [x * self.NS_PER_PT for x in b]
            for x in xs:
                ax.axvline(x, color=T.ACCENT_CYAN, ls="--", lw=1.0, alpha=0.7)
            ax.axvspan(xs[0], xs[1], color=T.ACCENT_CYAN, alpha=0.06)   # trailing
            ax.axvspan(xs[1], xs[2], color=T.ACCENT_AMBER, alpha=0.08)  # gap
            ax.axvspan(xs[2], xs[3], color=T.ACCENT_CYAN, alpha=0.06)   # leading

    # A per-event legend is only built up to this many traces. Beyond it the
    # matplotlib "best" legend solver runs over every line on each redraw (e.g.
    # when zooming), which freezes/blanks the canvas -- so the legend is skipped.
    MAX_TRACE_LEGEND = 25

    # ── offline fast filter / CFD overlays (Random traces) ───────────────────
    def _set_bin_filter_boxes_enabled(self, enabled):
        """Enable the Fast Filter / CFD boxes, unchecking them when they go away."""
        for cb in (self.cb_bin_ff, self.cb_bin_cfd):
            cb.setEnabled(enabled)
            if not enabled and cb.isChecked():
                cb.blockSignals(True); cb.setChecked(False); cb.blockSignals(False)
        # The custom-w sub-option follows the CFD box; blockSignals above skips
        # the cascade, so mirror CFD's checked state onto it explicitly.
        self.cb_bin_cfd_custom_w.setEnabled(enabled and self.cb_bin_cfd.isChecked())

    def _clear_bin_rand_twins(self):
        """Drop the twin y-axes from the previous Random-traces draw.

        ``ax.clear()`` empties an axes but leaves any ``twinx()`` siblings on
        the figure, so they have to be removed explicitly or every redraw
        stacks another pair of right-hand spines.
        """
        for tw in self._bin_rand_twins:
            try:
                tw.remove()
            except Exception:  # noqa: BLE001
                pass
        self._bin_rand_twins = []

    def _make_bin_twin(self, ax, label, color, offset=0):
        """A styled twin y-axis on *ax*, offset outward by *offset* points."""
        tw = ax.twinx()
        if offset:
            tw.spines["right"].set_position(("outward", offset))
        tw.set_ylabel(label, color=color, fontsize=11)
        # Sit the label right on the spine: the default padding pushes it into
        # the tick labels of whichever twin is offset further out.
        tw.yaxis.labelpad = 2
        tw.tick_params(axis="y", colors=color, labelsize=10, length=3)
        tw.spines["right"].set_color(color)
        tw.set_facecolor("none")
        self._bin_rand_twins.append(tw)
        return tw

    FF_COLOR = T.ACCENT_AMBER
    CFD_COLOR = T.ACCENT_RED

    def _draw_bin_filters(self, ax, df):
        """Overlay the offline fast filter / CFD of every trace in *df*.

        Reconstructs, per trace, the two quantities the Pixie-16 firmware
        computes in hardware but never stores (:mod:`wara.pixie_trace_analysis`)
        and draws each on its own twin y-axis together with the channel's
        ``FastThresh`` / ``CFDThresh`` register value.

        Returns ``(handles, labels)`` for the caller's legend.
        """
        want_ff = self.cb_bin_ff.isChecked()
        want_cfd = self.cb_bin_cfd.isChecked()
        if not (want_ff or want_cfd) or not self._bin_fast:
            return [], []

        # Custom firmware runs the CFD at a fractional weight (w=0.3125);
        # stock firmware fixes it at w=1.
        cfd_w = (pta.CFD_W_CUSTOM if self.cb_bin_cfd_custom_w.isChecked()
                 else pta.CFD_W)

        # With both on, the CFD axis moves outward so the spines don't overlap,
        # and the (now inner) fast-filter axis drops its label -- it would land
        # on top of the CFD tick labels. Its amber ticks and the legend below
        # still identify it.
        both = want_ff and want_cfd
        ax_ff = (self._make_bin_twin(ax, "" if both else "Fast filter",
                                     self.FF_COLOR)
                 if want_ff else None)
        ax_cfd = (self._make_bin_twin(ax, "CFD", self.CFD_COLOR,
                                      offset=52 if want_ff else 0)
                  if want_cfd else None)

        ff_thresh, cfd_thresh = set(), set()
        n = 0
        for i in df.index:
            rowobj = df.loc[i]
            trace = rowobj.trace
            if trace is None or len(trace) == 0:
                continue
            geo = self._bin_fast.get(int(getattr(rowobj, "channel", -1)))
            if geo is None:
                continue
            FL, FG, fth, cth = geo
            t = self._trace_time_axis(trace)
            if ax_ff is not None:
                ff = pta.fast_filter(trace, FL, FG)[0]
                ax_ff.plot(t, ff, lw=0.7, color=self.FF_COLOR, alpha=0.55)
                ff_thresh.add(fth)
            if ax_cfd is not None:
                cfd = pta.cfd_trace(trace, w=cfd_w)[0]
                ax_cfd.plot(t, cfd, lw=0.7, color=self.CFD_COLOR, alpha=0.55)
                cfd_thresh.add(cth)
            n += 1
        if n == 0:
            # No sampled channel had settings -- drop the empty twin axes again.
            self._clear_bin_rand_twins()
            return [], []

        handles, labels = [], []
        if ax_ff is not None:
            handles.append(Line2D([], [], color=self.FF_COLOR, lw=1.2))
            labels.append("fast filter")
            for v in sorted(ff_thresh):
                ax_ff.axhline(v, color=self.FF_COLOR, ls="--", lw=0.9, alpha=0.9)
            if ff_thresh:
                handles.append(Line2D([], [], color=self.FF_COLOR, ls="--", lw=0.9))
                labels.append("FastThresh "
                              + ", ".join(str(v) for v in sorted(ff_thresh)))
        if ax_cfd is not None:
            handles.append(Line2D([], [], color=self.CFD_COLOR, lw=1.2))
            labels.append(f"CFD (w={cfd_w:g})")
            for v in sorted(cfd_thresh):
                ax_cfd.axhline(v, color=self.CFD_COLOR, ls="--", lw=0.9, alpha=0.9)
            if cfd_thresh:
                handles.append(Line2D([], [], color=self.CFD_COLOR, ls="--", lw=0.9))
                labels.append("CFDThresh "
                              + ", ".join(str(v) for v in sorted(cfd_thresh)))
            ax_cfd.axhline(0, color=T.TEXT_DIM, lw=0.6, alpha=0.5)
        return handles, labels

    def _toggle_bin_filters(self, _checked=False):
        """Redraw the Random-traces view when a filter overlay is toggled."""
        if self.df_bin_rand is None or self.df_bin_rand.shape[0] == 0:
            self._bin_state("Sample some traces first")
            return
        if self.btn_bin_fft.isChecked():
            self._bin_state("Turn the FFT off to see the filter overlays")
            return
        self._plot_bin_rand()
        on = [n for n, cb in (("fast filter", self.cb_bin_ff), ("CFD", self.cb_bin_cfd))
              if cb.isChecked()]
        self._bin_state(f"Overlaying {' + '.join(on)}" if on
                        else "Filter overlays off")

    def _plot_bin_rand(self):
        """Draw the sampled random traces (with any filter overlays)."""
        self._plot_bin_traces(self.df_bin_rand, self._bin_ax["rand"],
                              self._bin_canvas["rand"], self._bin_fig["rand"],
                              self.cb_bin_leg_rand.isChecked(),
                              toolbar=self._bin_toolbar["rand"], overlays=True)

    def _plot_bin_traces(self, df, ax, canvas, fig, legend_on, toolbar=None,
                         overlays=False):
        if overlays:
            self._clear_bin_rand_twins()
        ax.clear()
        if df is None or df.shape[0] == 0:
            self._draw_placeholder(ax, "No traces to show", "Time (ns)", "linear")
            canvas.draw_idle()
            return
        # Only label traces when there are few enough for a usable legend.
        label = df.shape[0] <= self.MAX_TRACE_LEGEND
        n = 0
        bounds_set = []
        for i in df.index:
            rowobj = df.loc[i]
            # Traces can vary in length (and some events have none) -- give each
            # its own time axis and skip empty ones. When alignment is on, the
            # LGL-sum windows are located per-trace instead of shifting traces.
            trace, bounds = self._aligned_trace(rowobj)
            if trace is None or len(trace) == 0:
                continue
            ax.plot(self._trace_time_axis(trace), trace, lw=0.8,
                    label=self._trace_label(rowobj, i) if label else None)
            if bounds is not None and bounds not in bounds_set:
                bounds_set.append(bounds)
            n += 1
        if n == 0:
            self._draw_placeholder(ax, "Selected events carry no traces",
                                   "Time (ns)", "linear")
            canvas.draw_idle()
            return
        if bounds_set:
            self._draw_lgl_windows(ax, bounds_set)
        self._style_axis(ax, "Time (ns)", "linear")
        ax.set_ylabel("Amplitude (a.u.)", color=T.TEXT_DIM, fontsize=12)
        if label:
            # Explicit location: the default "best" solver is O(lines) per draw.
            leg = ax.legend(loc="upper right", fontsize=8, facecolor=API_PLOT_BG,
                            edgecolor=T.BORDER, labelcolor=T.TEXT_DIM, framealpha=0.7)
            leg.set_visible(legend_on)
        if overlays:
            # Drawn after _style_axis so the twin axes sit on top of the traces,
            # and legended separately -- these lines are always worth naming,
            # unlike the (optional, per-event) trace legend.
            handles, labels = self._draw_bin_filters(ax, df)
            if handles:
                # On the twin, not on `ax` -- an axes holds only one legend and
                # `ax`'s may already be the per-event trace legend above.
                self._bin_rand_twins[0].legend(
                    handles, labels, loc="upper left", fontsize=8,
                    facecolor=API_PLOT_BG, edgecolor=T.BORDER,
                    labelcolor=T.TEXT_DIM, framealpha=0.7)
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
        bounds_set = []
        for i in self.df_bin_tr.index:
            # Per-trace baseline subtraction so variable-length / empty traces
            # don't force a ragged vstack; LGL windows located when enabled.
            trace, bounds = self._aligned_trace(self.df_bin_tr.loc[i])
            if trace is None or len(trace) == 0:
                continue
            trace = np.asarray(trace, dtype=float)
            norm = trace - trace[:base_pts].mean()
            ax.plot(self._trace_time_axis(trace), norm, lw=0.8)
            if bounds is not None and bounds not in bounds_set:
                bounds_set.append(bounds)
            n += 1
        if n == 0:
            self._draw_placeholder(ax, "Selected events carry no traces",
                                   "Time (ns)", "linear")
            self._bin_canvas["ergtr"].draw_idle()
            self._bin_state("No traces to normalize")
            return
        if bounds_set:
            self._draw_lgl_windows(ax, bounds_set)
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

    def _toggle_bin_align(self, checked):
        # Alignment is applied at plot time, so just redraw the two trace views
        # from the data already in memory -- no reload.
        self._bin_align = checked
        if self.df_bin_tr is not None and self.df_bin_tr.shape[0] > 0:
            self._plot_bin_traces(self.df_bin_tr, self._bin_ax["ergtr"],
                                  self._bin_canvas["ergtr"], self._bin_fig["ergtr"],
                                  self.cb_bin_leg_tr.isChecked(),
                                  toolbar=self._bin_toolbar["ergtr"])
        if (self.df_bin_rand is not None and self.df_bin_rand.shape[0] > 0
                and not self.btn_bin_fft.isChecked()):
            self._plot_bin_rand()
        self._bin_state(f"LGL-sum alignment {'on' if checked else 'off'}")

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
        # If the user highlighted an energy region on the Energy tab, draw the
        # random sample only from that window; otherwise use all events.
        span = self._bin_span_range
        if span is not None and "energy" in df.columns:
            lo, hi = span
            df = df[(df["energy"] > lo) & (df["energy"] < hi)]
        if self.cb_bin_pileup.isChecked():
            df = df[df.pileup == True]  # noqa: E712
        elif self.cb_bin_cfderr.isChecked():
            df = df[df.CFD_error == True]  # noqa: E712
        elif self.cb_bin_flag.isChecked():
            df = df[df.trace_flag == True]  # noqa: E712
        if df.shape[0] == 0:
            self._bin_state("No events match that filter"
                            + (" in the highlighted region" if span else ""))
            return
        n = min(n, df.shape[0])
        self.df_bin_rand = df.sample(n).sort_index()
        self._plot_bin_rand()
        self.bin_plot_tabs.setCurrentIndex(2)
        region = (f" from [{span[0]:.1f}, {span[1]:.1f}]" if span else "")
        self._bin_state(f"Sampled {n} random traces{region}")

    def _toggle_bin_fft(self):
        if not self.btn_bin_fft.isChecked():
            # Back to traces (and their filter overlays).
            if self.df_bin_rand is not None:
                self._plot_bin_rand()
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
        # The filter overlays' twin axes are meaningless in the frequency domain.
        self._clear_bin_rand_twins()
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
    def _bin_ch_suffix(self):
        vis = self._visible_channels()
        if not vis:
            return "CH-all"
        return "CH" + "+".join(str(c) for c in vis)

    def _send_bin_energy(self):
        if self._bin_gam is None:
            self._bin_state("Load data (energy histogram) first")
            return
        name = f"{self._bin_date}-{self._bin_run}-{self._bin_ch_suffix()}"
        self._send_spectrum(self._bin_gam, self._bin_gam_x, name)

    def _send_bin_trace_energy(self):
        if self._bin_gam2 is None:
            self._bin_state("Calc energy first")
            return
        name = f"{self._bin_date}-{self._bin_run}-{self._bin_ch_suffix()}-trace"
        self._send_spectrum(self._bin_gam2, self._bin_gam_x2, name)

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
