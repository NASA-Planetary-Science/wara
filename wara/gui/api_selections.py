"""Plot-selections dialog for the API tab (``SelectionsDialog``): overlays
the stored energy selections and drives the time-slice fit machinery."""
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QTabWidget, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt, QSize

from . import theme as T
from .widgets import hsep, header, labeled_row
from .slicefit import (
    plot_slice_spectra, plot_slice_offset, plot_slice_waterfall,
    TECHNIQUE_LABELS, TECHNIQUE_FROM_LABEL, TECH_FIT, TECH_SNR,
)
from .api_common import API_PLOT_BG, _combo_row



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
