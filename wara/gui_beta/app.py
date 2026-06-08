"""
wara --beta : modernized WARA GUI (work in progress).

Three-column layout:
  1. NAV     — logo + WARA wordmark, the analysis sections as color-dot tabs,
               global File actions.
  2. OPTIONS — tab-specific controls (collapsible; auto-hides when a tab has none).
  3. PLOT    — the active tab's figure.

Only the Spectrum tab is wired so far (open file, plot, stats, cursor, log-Y,
reset). Remaining controls are present but inert and will be wired one by one.
"""
import os
import sys
from importlib.resources import files

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QStackedWidget, QButtonGroup, QSizePolicy, QSpacerItem,
    QFileDialog, QMessageBox, QCheckBox, QDialog, QFormLayout, QLineEdit,
    QDialogButtonBox, QFrame,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QPalette, QPixmap, QIcon
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

from wara import peaksearch as ps
from . import theme as T
from .theme import NAV_SECTIONS, TABS_WITH_OPTIONS, STYLESHEET, dot_icon, recolor_toolbar_icons
from .widgets import (
    SpectrumCanvas, SpectrumOptions, PlaceholderOptions, PlaceholderPage, header, hsep,
)
from .io import load_spectrum_file, OPEN_FILTER
from .nuclear import NuclearDatabaseDialog

LOGO_PATH = str(files("wara").joinpath("ui/wara-logo.png"))


class AxisLegendDialog(QDialog):
    """Edit the active spectrum's legend label, axis labels and description,
    and show the rest of its metadata read-only."""

    def __init__(self, parent, spect, xlabel, ylabel):
        super().__init__(parent)
        self.setWindowTitle("Axis, Legend & Spectrum Info")
        self.setStyleSheet(STYLESHEET)
        self.setMinimumWidth(420)
        outer = QVBoxLayout(self)

        # ── Editable ─────────────────────────────────────────────
        form = QFormLayout()
        self.ed_legend = QLineEdit(spect.label or "")
        self.ed_xlabel = QLineEdit(xlabel or "")
        self.ed_ylabel = QLineEdit(ylabel or "")
        self.ed_desc = QLineEdit(spect.description or "")
        form.addRow("Legend label:", self.ed_legend)
        form.addRow("X-axis label:", self.ed_xlabel)
        form.addRow("Y-axis label:", self.ed_ylabel)
        form.addRow("Description:", self.ed_desc)
        outer.addLayout(form)

        sep = QFrame(); sep.setObjectName("separator"); sep.setFrameShape(QFrame.NoFrame)
        outer.addWidget(sep)
        title = QLabel("SPECTRUM DETAILS"); title.setObjectName("section_header")
        outer.addWidget(title)

        # ── Read-only metadata ───────────────────────────────────
        info = QFormLayout()

        def num(v, fmt="{:,.0f}"):
            return fmt.format(v) if v is not None else "—"

        def row(key, value):
            val = QLabel(str(value))
            val.setStyleSheet(f"color: {T.TEXT_PRIMARY}; font-family: {T.MONO_FAMILY};")
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            info.addRow(key, val)

        ecal = spect.energy_cal
        row("Channels:", f"{len(spect.counts):,}")
        row("Total counts:", num(spect.counts.sum()))
        row("Live time (s):", num(spect.livetime, "{:,.2f}"))
        row("Real time (s):", num(spect.realtime, "{:,.2f}"))
        row("Acquisition date:", spect.acq_date if spect.acq_date is not None else "—")
        row("Count rate (cps):", "Yes" if spect.cps else "No")
        row("Energy units:", spect.e_units if spect.e_units else "—")
        row("Energy calibration:", ecal if ecal is not None else "None (uncalibrated)")
        outer.addLayout(info)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)

    def values(self):
        return (self.ed_legend.text().strip(), self.ed_xlabel.text().strip(),
                self.ed_ylabel.text().strip(), self.ed_desc.text().strip())


class SpectrumPage(QWidget):
    """Plot column for the Spectrum tab: toolbar on top + canvas."""

    def __init__(self):
        super().__init__()
        self.setObjectName("content")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(8)
        self.canvas = SpectrumCanvas()
        nav = NavToolbar(self.canvas, self)
        nav.setObjectName("plot_toolbar")
        nav.setIconSize(QSize(22, 22))
        recolor_toolbar_icons(nav, T.TEXT_PRIMARY)
        lay.addWidget(nav)
        lay.addWidget(self.canvas, stretch=1)


class WaraBetaApp(QMainWindow):
    OPT_W = 390

    def __init__(self, file_name=None):
        super().__init__()
        self.setWindowTitle("WARA  ·  Spectrum Analysis  (beta)")
        self.setMinimumSize(1480, 760)
        # Open large: 86% of the available screen (falls back to a fixed size).
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            self.resize(int(avail.width() * 0.86), int(avail.height() * 0.86))
        else:
            self.resize(2000, 1100)
        self.setStyleSheet(STYLESHEET)
        self.setWindowIcon(QIcon(LOGO_PATH))

        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(T.BG_DARK))
        self.setPalette(pal)

        self.spect = None
        self._spect_orig = None
        self.search = None
        self._remove_cal = False     # strip energy calibration in the recompute
        self._xlabel = None          # axis-label overrides (None = auto)
        self._ylabel = None
        self._active_name = None
        self._active_visible = True
        self._overlays = []          # kept spectra: list of dicts (spect/name/visible)
        self._opt_collapsed = False

        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10); root.setSpacing(10)
        root.addWidget(self._build_nav())
        root.addWidget(self._build_options())
        root.addWidget(self._build_stack(), stretch=1)

        # Permanent file-name readout on the right of the status bar.
        self._file_label = QLabel("")
        self._file_label.setStyleSheet(f"color: {T.TEXT_DIM}; padding-right: 12px;")
        self.statusBar().addPermanentWidget(self._file_label)

        self._wire_spectrum_tab()
        self._rebuild_spectra_list()
        self.statusBar().showMessage("  Ready  ·  open a spectrum file to begin")

        # Select the first tab
        self.nav_group.button(0).setChecked(True)
        self._on_nav(0)

        if file_name:
            self._load_path(file_name)

    # ── Column 1: navigation ─────────────────────────────────────────────────
    def _build_nav(self):
        panel = QWidget(); panel.setObjectName("nav_panel")
        panel.setAttribute(Qt.WA_StyledBackground, True); panel.setFixedWidth(230)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 16, 14, 16); lay.setSpacing(8)

        brand = QHBoxLayout(); brand.setSpacing(10)
        logo = QLabel(); pm = QPixmap(LOGO_PATH)
        if not pm.isNull():
            logo.setPixmap(pm.scaledToHeight(44, Qt.SmoothTransformation))
        brand.addWidget(logo)
        t = QLabel("WARA"); t.setObjectName("title")
        brand.addWidget(t); brand.addStretch(1)
        lay.addLayout(brand)
        lay.addSpacing(6); lay.addWidget(hsep())

        lay.addWidget(header("ANALYSIS"))
        self.nav_group = QButtonGroup(self); self.nav_group.setExclusive(True)
        for idx, (name, color) in enumerate(NAV_SECTIONS):
            btn = QPushButton("   " + name); btn.setObjectName("nav_btn")
            btn.setCheckable(True); btn.setCursor(Qt.PointingHandCursor)
            btn.setIcon(dot_icon(color)); btn.setIconSize(QSize(14, 14))
            self.nav_group.addButton(btn, idx); lay.addWidget(btn)
        self.nav_group.idClicked.connect(self._on_nav)

        lay.addSpacing(4); lay.addWidget(hsep())
        lay.addWidget(header("FILE"))
        self.btn_open = QPushButton("Open Spectrum"); self.btn_open.setObjectName("action_btn")
        self.btn_save = QPushButton("Save"); self.btn_save.setObjectName("action_btn")
        self.btn_clear = QPushButton("Clear"); self.btn_clear.setObjectName("action_btn")
        for b in (self.btn_open, self.btn_save, self.btn_clear):
            b.setCursor(Qt.PointingHandCursor); lay.addWidget(b)

        lay.addItem(QSpacerItem(20, 12, QSizePolicy.Minimum, QSizePolicy.Expanding))
        ver = QLabel("v1.0"); ver.setObjectName("version")
        ver.setAlignment(Qt.AlignHCenter); lay.addWidget(ver)
        return panel

    # ── Column 2: options (collapsible) ──────────────────────────────────────
    def _build_options(self):
        panel = QWidget(); panel.setObjectName("opt_panel")
        panel.setAttribute(Qt.WA_StyledBackground, True); panel.setFixedWidth(self.OPT_W)
        outer = QHBoxLayout(panel); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        self.opt_stack = QStackedWidget()
        self.spectrum_opts = SpectrumOptions()
        for name, _c in NAV_SECTIONS:
            if name == "Spectrum":
                self.opt_stack.addWidget(self.spectrum_opts)
            else:
                self.opt_stack.addWidget(PlaceholderOptions(name))
        outer.addWidget(self.opt_stack, stretch=1)

        # Vertically-centered collapse handle (thin tall strip) on the right edge.
        self.collapse_btn = QPushButton("◀"); self.collapse_btn.setObjectName("collapse_btn")
        self.collapse_btn.setCursor(Qt.PointingHandCursor); self.collapse_btn.setFixedSize(16, 48)
        self.collapse_btn.setToolTip("Collapse options panel")
        self.collapse_btn.clicked.connect(self._toggle_options)
        rail = QWidget(); rail.setFixedWidth(18)
        rl = QVBoxLayout(rail); rl.setContentsMargins(1, 4, 1, 4)
        rl.addStretch(1); rl.addWidget(self.collapse_btn, 0, Qt.AlignHCenter); rl.addStretch(1)
        outer.addWidget(rail)

        self.opt_panel = panel
        return panel

    def _toggle_options(self):
        self._opt_collapsed = not self._opt_collapsed
        self._apply_opt_state()

    def _apply_opt_state(self):
        if self._opt_collapsed:
            self.opt_stack.hide(); self.opt_panel.setFixedWidth(22)
            self.collapse_btn.setText("▶"); self.collapse_btn.setToolTip("Expand options panel")
        else:
            self.opt_stack.show(); self.opt_panel.setFixedWidth(self.OPT_W)
            self.collapse_btn.setText("◀"); self.collapse_btn.setToolTip("Collapse options panel")

    # ── Column 3: plots ──────────────────────────────────────────────────────
    def _build_stack(self):
        self.stack = QStackedWidget()
        self.spectrum_page = SpectrumPage()
        for name, _c in NAV_SECTIONS:
            if name == "Spectrum":
                self.stack.addWidget(self.spectrum_page)
            else:
                self.stack.addWidget(PlaceholderPage(name))
        return self.stack

    def _on_nav(self, idx):
        name = NAV_SECTIONS[idx][0]
        self.stack.setCurrentIndex(idx)
        self.opt_stack.setCurrentIndex(idx)
        has_opts = name in TABS_WITH_OPTIONS
        self.opt_panel.setVisible(has_opts)
        if has_opts:
            self._apply_opt_state()
        self.statusBar().showMessage(f"  {name}")

    # ── Spectrum tab wiring ──────────────────────────────────────────────────
    def _wire_spectrum_tab(self):
        self.btn_open.clicked.connect(self._open_file)
        self.btn_save.clicked.connect(self._save_spectrum)
        self.btn_clear.clicked.connect(self._clear)
        opts = self.spectrum_opts
        opts.cb_log.toggled.connect(self._replot)
        opts.cb_keep.toggled.connect(self._on_keep_toggled)
        opts.btn_reset.clicked.connect(self._reset_spectrum)
        opts.btn_remcal.clicked.connect(self._remove_calibration)
        opts.addsub_panel.btn_add.clicked.connect(self._add_all)
        opts.addsub_panel.btn_sub.clicked.connect(self._subtract_overlays)
        opts.btn_labels.clicked.connect(self._open_axis_legend)
        # Customize: each checkbox applies its option live; changing a value
        # re-applies (recomputed from the original) only when its box is ticked.
        self._cust_checks = [opts.cb_smooth, opts.cb_rebin, opts.cb_shift,
                             opts.cb_yconst, opts.cb_xconst, opts.cb_cr]
        for cb in self._cust_checks:
            cb.toggled.connect(self._recompute)
        for cb, spin in ((opts.cb_smooth, opts.smooth_spin), (opts.cb_rebin, opts.rebin),
                         (opts.cb_shift, opts.shift_box), (opts.cb_yconst, opts.yconst),
                         (opts.cb_xconst, opts.xconst)):
            spin.valueChanged.connect(lambda *_, c=cb: c.isChecked() and self._recompute())
        self.spectrum_page.canvas.cursor_moved.connect(self._on_cursor)
        # Peak finding (inline Auto-Find Peaks panel)
        pf = opts.pf_panel
        pf.btn_find.clicked.connect(self._find_peaks)
        pf.btn_clear.clicked.connect(self._clear_peaks)
        pf.detector.currentTextChanged.connect(self._apply_detector_preset)
        pf.cb_peaks.toggled.connect(self.spectrum_page.canvas.set_show_peaks)
        pf.cb_snr.toggled.connect(self._toggle_snr)
        pf.cb_manual.toggled.connect(self.spectrum_page.canvas.set_manual)
        self.spectrum_page.canvas.point_clicked.connect(self._add_manual_peak)
        opts.btn_iso.clicked.connect(self._open_nuclear_db)
        self._apply_detector_preset(pf.detector.currentText())

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Spectrum", filter=OPEN_FILTER)
        if path:
            self._load_path(path)

    def _open_axis_legend(self):
        if not self._guard():
            return
        cur_x = self._xlabel or self.spect.x_units
        cur_y = self._ylabel or {"Cts": "Counts", "CPS": "Counts/second"}.get(
            self.spect.y_label, self.spect.y_label)
        dlg = AxisLegendDialog(self, self.spect, cur_x, cur_y)
        if dlg.exec_() != QDialog.Accepted:
            return
        legend, xlabel, ylabel, desc = dlg.values()
        # Legend label + description belong to the spectrum (persist via baseline).
        self.spect.label = legend or None
        self.spect.description = desc or None
        if self._spect_orig is not None:
            self._spect_orig.label = legend or None
            self._spect_orig.description = desc or None
        # Axis labels are display overrides on the plot (None = auto).
        self._xlabel = xlabel or None
        self._ylabel = ylabel or None
        self._replot()
        self._rebuild_spectra_list()   # legend label may have changed the name
        self.statusBar().showMessage("  Axis & legend updated")

    def _save_spectrum(self):
        if not self._guard():
            return
        base = os.path.splitext(self._active_name)[0] if self._active_name else "spectrum"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Active Spectrum", base + ".txt",
            filter="Text with metadata (*.txt);;CSV (*.csv)")
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                self.spect.to_csv(path)
            else:
                self.spect.to_txt(path)
        except Exception as exc:  # noqa: BLE001 — surface write errors
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        self.statusBar().showMessage(f"  Saved → {os.path.basename(path)}")

    def _load_path(self, path):
        try:
            new_spect = load_spectrum_file(path)
        except Exception as exc:  # noqa: BLE001 — surface any read error to the user
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        # Overlay handling: if "Keep spectrum visible" is on, freeze the current
        # spectrum as an overlay before the new one becomes active.
        if self.spectrum_opts.cb_keep.isChecked():
            if self.spect is not None:
                self._overlays.append({
                    "spect": self.spect.copy(),
                    "name": self._active_name or "spectrum",
                    "visible": self._active_visible,
                })
        else:
            self._overlays = []

        self.spect = new_spect
        self._active_name = os.path.basename(path)
        self._active_visible = True
        self._remove_cal = False
        self._xlabel = self._ylabel = None
        self.spectrum_page.canvas.set_active_visible(True)
        self._spect_orig = self.spect.copy()  # baseline; customize derives from this
        self._reset_customize_checks()
        self._sync_cps_checkbox()             # reflect the file's native cps state
        self._update_overlays()
        self._rebuild_spectra_list()
        self._refresh()
        self._file_label.setText(f"📄 {self._active_name}")
        c = self.spect.counts
        n_over = len(self._overlays)
        extra = f"  ·  {n_over} overlaid" if n_over else ""
        self.statusBar().showMessage(
            f"  Loaded  ·  {len(c):,} channels  ·  {int(c.sum()):,} total counts{extra}")

    @staticmethod
    def _display_name(spect, filename):
        """Prefer the spectrum's own label; fall back to the file name."""
        label = getattr(spect, "label", None)
        return label if label else filename

    def _overlay_color(self, index):
        return T.OVERLAY_COLORS[index % len(T.OVERLAY_COLORS)]

    def _update_overlays(self):
        # Color overlays by position so a given slot keeps its color regardless
        # of which spectrum is active.
        items = [(r["spect"].x, r["spect"].counts,
                  self._display_name(r["spect"], r["name"]), self._overlay_color(i))
                 for i, r in enumerate(self._overlays) if r["visible"]]
        self.spectrum_page.canvas.set_overlays(items)

    def _on_keep_toggled(self, checked):
        # Unchecking stops overlaying and drops the spectra already kept.
        if not checked and self._overlays:
            self._overlays = []
            self._update_overlays()
            self._rebuild_spectra_list()
            self._replot()

    # ── Loaded-spectra list (active + overlays) ──────────────────────────────
    def _rebuild_spectra_list(self):
        lay = self.spectrum_opts.spectra_list
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if self.spect is None:
            ph = QLabel("No spectrum loaded"); ph.setObjectName("stat_key")
            lay.addWidget(ph)
            return
        # Active spectrum (cyan) — toggle only, no remove, already active.
        lay.addWidget(self._make_spectrum_row(
            self._display_name(self.spect, self._active_name or "spectrum"),
            T.ACCENT_CYAN, self._active_visible,
            on_toggle=self._toggle_active_visible, active=True))
        # Overlays — click name to activate, toggle visibility, or remove.
        for i, rec in enumerate(self._overlays):
            lay.addWidget(self._make_spectrum_row(
                self._display_name(rec["spect"], rec["name"]),
                self._overlay_color(i), rec["visible"],
                on_toggle=lambda v, r=rec: self._toggle_overlay(r, v),
                on_remove=lambda _=False, r=rec: self._remove_overlay(r),
                on_activate=lambda _=False, r=rec: self._activate_overlay(r)))

    def _make_spectrum_row(self, name, color, visible, on_toggle,
                           on_remove=None, on_activate=None, active=False):
        row = QWidget()
        h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
        swatch = QLabel(); swatch.setObjectName("swatch"); swatch.setFixedSize(12, 12)
        swatch.setStyleSheet(f"background:{color}; border-radius:3px;")
        h.addWidget(swatch)
        name_btn = QPushButton()
        name_btn.setObjectName("spectrum_name_active" if active else "spectrum_name")
        name_btn.setMaximumWidth(200)
        name_btn.setText(name_btn.fontMetrics().elidedText(name, Qt.ElideMiddle, 190))
        name_btn.setToolTip(name + ("  (active)" if active else "  ·  click to make active"))
        if on_activate is not None:
            name_btn.setCursor(Qt.PointingHandCursor)
            name_btn.clicked.connect(on_activate)
        h.addWidget(name_btn); h.addStretch(1)
        cb = QCheckBox(); cb.setObjectName("vis_check")
        cb.setChecked(visible); cb.setToolTip("Toggle visibility")
        cb.toggled.connect(on_toggle)
        h.addWidget(cb)
        if on_remove is not None:
            rm = QPushButton("×"); rm.setObjectName("remove_btn"); rm.setFixedSize(20, 20)
            rm.setToolTip("Remove overlay"); rm.setCursor(Qt.PointingHandCursor)
            rm.clicked.connect(on_remove)
            h.addWidget(rm)
        return row

    def _activate_overlay(self, rec):
        """Make a kept overlay the active spectrum; demote the current active
        into the overlay list. The promoted spectrum's frozen state becomes the
        new customize baseline."""
        if rec not in self._overlays:
            return
        # Freeze the current active spectrum as an overlay (takes the slot the
        # promoted one is leaving, so overlay colors stay put).
        if self.spect is not None:
            idx = self._overlays.index(rec)
            self._overlays.insert(idx, {
                "spect": self.spect.copy(),
                "name": self._active_name or "spectrum",
                "visible": self._active_visible,
            })
            self._overlays.remove(rec)
        else:
            self._overlays.remove(rec)
        self.spect = rec["spect"]
        self._spect_orig = rec["spect"].copy()
        self._active_name = rec["name"]
        self._active_visible = rec["visible"]
        self._remove_cal = False
        self.spectrum_page.canvas.set_active_visible(self._active_visible)
        self._reset_customize_checks()
        self._sync_cps_checkbox()
        self._file_label.setText(f"📄 {self._active_name}")
        self._update_overlays()
        self._rebuild_spectra_list()
        self._refresh()
        self.statusBar().showMessage(f"  Active spectrum: {self._active_name}")

    def _toggle_active_visible(self, visible):
        self._active_visible = visible
        self.spectrum_page.canvas.set_active_visible(visible)

    def _toggle_overlay(self, rec, visible):
        rec["visible"] = visible
        self._update_overlays()

    def _remove_overlay(self, rec):
        if rec in self._overlays:
            self._overlays.remove(rec)
        self._update_overlays()
        self._rebuild_spectra_list()

    # ── Add / Subtract ───────────────────────────────────────────────────────
    def _add_all(self):
        if not self._guard():
            return
        if not self._overlays:
            self.statusBar().showMessage("  Keep more than one spectrum to add")
            return
        names = [self._display_name(self.spect, self._active_name)]
        try:
            result = self.spect.copy()
            for rec in self._overlays:
                result = result + rec["spect"]
                names.append(self._display_name(rec["spect"], rec["name"]))
        except Exception as exc:  # noqa: BLE001 — mismatched bins/x-axis, etc.
            self.statusBar().showMessage(f"  {exc}")
            return
        label = " + ".join(names) if len(names) <= 3 else f"Sum of {len(names)} spectra"
        self._set_combined_result(result, label)
        self.statusBar().showMessage(f"  Added {len(names)} spectra")

    def _subtract_overlays(self):
        if not self._guard():
            return
        if not self._overlays:
            self.statusBar().showMessage("  No overlays to subtract")
            return
        scale_lt = self.spectrum_opts.addsub_panel.cb_scale_lt.isChecked()
        active_lt = self.spect.livetime
        names = []
        try:
            result = self.spect.copy()
            for rec in self._overlays:
                bg = rec["spect"]
                if scale_lt and active_lt and bg.livetime:
                    bg = bg * (active_lt / bg.livetime)
                result = result - bg
                names.append(self._display_name(rec["spect"], rec["name"]))
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"  {exc}")
            return
        active = self._display_name(self.spect, self._active_name)
        label = f"{active} − " + " − ".join(names)
        if len(label) > 40:
            label = f"{active} − {len(names)} overlay(s)"
        self._set_combined_result(result, label)
        scaled = "  (live-time scaled)" if scale_lt else ""
        self.statusBar().showMessage(f"  Subtracted {len(names)} overlay(s) from active{scaled}")

    def _set_combined_result(self, result, label):
        """Make the add/subtract result the new active spectrum, keeping the
        source spectra as overlays."""
        # Freeze the current active as an overlay (sources stay for comparison).
        self._overlays.append({
            "spect": self.spect.copy(),
            "name": self._active_name or "spectrum",
            "visible": self._active_visible,
        })
        result.label = label
        self.spect = result
        self._spect_orig = result.copy()
        self._active_name = label
        self._active_visible = True
        self._remove_cal = False
        self.spectrum_page.canvas.set_active_visible(True)
        self._reset_customize_checks()
        self._sync_cps_checkbox()
        self._file_label.setText(f"∑ {label}")
        self._update_overlays()
        self._rebuild_spectra_list()
        self._refresh()

    def _replot(self, *args):
        if self.spect is None:
            return
        self.spectrum_page.canvas.plot_spectrum(
            self.spect, log_y=self.spectrum_opts.cb_log.isChecked(),
            label=self._display_name(self.spect, self._active_name),
            xlabel=self._xlabel, ylabel=self._ylabel)

    def _refresh(self):
        """Replot and refresh the stat readouts. Any existing peak markers
        refer to the previous data, so they are cleared."""
        if self.spect is None:
            return
        self.search = None
        self.spectrum_page.canvas.clear_snr()
        self.spectrum_page.canvas.set_peaks([])
        self._replot()
        self._update_cursor_units()
        c = self.spect.counts
        self.spectrum_opts.set_stats(len(c), c.max(), c.sum())

    def _update_cursor_units(self):
        """Set the CURSOR 'X' label to the active spectrum's actual x-units."""
        xu = self.spect.x_units if self.spect is not None else None
        if not xu:
            unit = "ch/keV"
        elif "(" in xu and ")" in xu:
            unit = xu[xu.index("(") + 1:xu.rindex(")")]   # e.g. Energy (keV) -> keV
        elif xu.lower().startswith("chan"):
            unit = "ch"
        else:
            unit = xu
        self.spectrum_opts.key_cur_x.setText(f"X ({unit})")

    def _clear(self):
        self.spect = None
        self._spect_orig = None
        self._active_name = None
        self._active_visible = True
        self._overlays = []
        self._xlabel = self._ylabel = None
        self.spectrum_page.canvas.set_active_visible(True)
        self.spectrum_page.canvas.set_overlays([])
        self.spectrum_page.canvas.clear_ref_lines()
        self.spectrum_page.canvas.draw_empty()
        self.spectrum_opts.clear_stats()
        self.spectrum_opts.key_cur_x.setText("X (ch/keV)")
        self._rebuild_spectra_list()
        self._file_label.setText("")
        self.statusBar().showMessage("  Cleared")

    def _on_cursor(self, x, y):
        self.spectrum_opts.val_cur_x.setText(f"{x:.2f}")
        self.spectrum_opts.val_cur_y.setText(f"{int(y):,}")

    # ── Peak finding ─────────────────────────────────────────────────────────
    # Detector presets: (SNR, ref. channel, ref. FWHM) — mirrors legacy wara.
    DETECTOR_PRESETS = {
        "HPGe":           ("5", "420", "3"),
        "LaBr/CeBr":      ("5", "420", "12"),
        "NaI":            ("5", "420", "15"),
        "Plastic Scint.": ("5", "420", "20"),
    }

    def _apply_detector_preset(self, name):
        preset = self.DETECTOR_PRESETS.get(name)
        if not preset:
            return
        snr, ref_ch, ref_fwhm = preset
        pf = self.spectrum_opts.pf_panel
        pf.snr.setText(snr)
        pf.ref_ch.setText(ref_ch)
        pf.ref_fwhm.setText(ref_fwhm)

    def _find_peaks(self):
        if not self._guard():
            return
        pf = self.spectrum_opts.pf_panel
        try:
            min_snr = float(pf.snr.text())
            ref_x = float(pf.ref_ch.text())
            ref_fwhm = float(pf.ref_fwhm.text())
        except ValueError:
            self.statusBar().showMessage("  Enter numeric SNR, Ref. channel, and Ref. FWHM")
            return

        xrange = None
        if pf.x0.text().strip() and pf.x1.text().strip():
            try:
                xrange = [float(pf.x0.text()), float(pf.x1.text())]
            except ValueError:
                self.statusBar().showMessage("  Xrange values must be numeric")
                return

        method = "km" if pf.cb_kernel.isChecked() else "fast"
        if method == "km" and xrange is None and len(self.spect.channels) >= 9000:
            self.statusBar().showMessage(
                "  Kernel Method needs < 9000 channels — set an Xrange or uncheck it")
            return

        try:
            self.search = ps.PeakSearch(
                self.spect, ref_x, ref_fwhm, fwhm_at_0=1.0,
                min_snr=min_snr, xrange=xrange, method=method)
        except Exception as exc:  # noqa: BLE001 — surface search errors
            self.statusBar().showMessage(f"  {exc}")
            return

        idx = self.search.peaks_idx
        if idx is None or len(idx) == 0:
            self.spectrum_page.canvas.set_peaks([])
            self.statusBar().showMessage("  No peaks found — try lowering SNR")
            return
        xs = self.spect.x[idx]
        ys = self.spect.counts[idx]
        peaks = [(float(px), float(py), f"{px:.1f}") for px, py in zip(xs, ys)]
        self.spectrum_opts.pf_panel.cb_peaks.setChecked(True)
        self.spectrum_page.canvas.set_peaks(peaks)
        if self.spectrum_opts.pf_panel.cb_snr.isChecked():
            self._draw_snr()
        self.statusBar().showMessage(f"  Found {len(peaks)} peaks")

    def _toggle_snr(self, checked):
        if checked:
            self._draw_snr()
        else:
            self.spectrum_page.canvas.clear_snr()

    def _draw_snr(self):
        s = self.search
        if s is None or s.snr is None:
            self.statusBar().showMessage("  Run Find Peaks first to plot SNR")
            return
        snr = np.asarray(s.snr)
        xs = self.spect.x[s.channel_idx]
        if len(snr) != len(xs):
            self.statusBar().showMessage("  SNR curve not available for this method")
            return
        self.spectrum_page.canvas.set_snr(xs, snr, threshold=s.min_snr)

    def _add_manual_peak(self, x, y):
        if self.spect is None:
            return
        idx = int(np.argmin(np.abs(self.spect.x - x)))
        px = float(self.spect.x[idx])
        py = float(self.spect.counts[idx])
        self.spectrum_opts.pf_panel.cb_peaks.setChecked(True)
        self.spectrum_page.canvas.add_peak(px, py, f"{px:.1f}")
        self.statusBar().showMessage(f"  Added peak at {px:.1f}")

    def _clear_peaks(self):
        self.search = None
        self.spectrum_page.canvas.clear_snr()
        self.spectrum_page.canvas.set_peaks([])
        self.statusBar().showMessage("  Peaks cleared")

    def _open_nuclear_db(self):
        # Reuse one dialog instance so it remembers the last search.
        if getattr(self, "_nuc_dialog", None) is None:
            self._nuc_dialog = NuclearDatabaseDialog(self)
            self._nuc_dialog.plot_requested.connect(self._plot_ref_lines)
            self._nuc_dialog.clear_requested.connect(self._clear_ref_lines)
        self._nuc_dialog.show()
        self._nuc_dialog.raise_()
        self._nuc_dialog.activateWindow()

    MAX_REF_LINES = 50

    def _plot_ref_lines(self, lines, color):
        if not lines:
            self.statusBar().showMessage("  No lines to plot")
            return
        if len(lines) > self.MAX_REF_LINES:
            shown = len(lines)
            lines = lines[:self.MAX_REF_LINES]
            self.spectrum_page.canvas.add_ref_lines(lines, color)
            self.statusBar().showMessage(
                f"  Too many lines ({shown}) — plotted first {self.MAX_REF_LINES}; "
                "select rows to narrow down")
            return
        self.spectrum_page.canvas.add_ref_lines(lines, color)
        self.statusBar().showMessage(f"  Plotted {len(lines)} reference line(s)")

    def _clear_ref_lines(self):
        self.spectrum_page.canvas.clear_ref_lines()
        self.statusBar().showMessage("  Reference lines cleared")

    # ── Customize (live, checkbox-driven) ────────────────────────────────────
    def _guard(self):
        if self.spect is None:
            self.statusBar().showMessage("  Load a spectrum first")
            return False
        return True

    def _recompute(self, *args):
        """Rebuild the working spectrum from the original, applying every ticked
        customize option in order. Called live whenever a checkbox toggles or a
        ticked option's value changes."""
        if self._spect_orig is None:
            return
        o = self.spectrum_opts

        # Count rate needs a livetime; refuse and untick if absent.
        if o.cb_cr.isChecked() and not self._spect_orig.cps and self._spect_orig.livetime is None:
            self.statusBar().showMessage("  No livetime in file — cannot set count rate")
            o.cb_cr.blockSignals(True); o.cb_cr.setChecked(False); o.cb_cr.blockSignals(False)

        s = self._spect_orig.copy()
        try:
            # Strip calibration first so the rest operate on the channel axis.
            if self._remove_cal and s.energies is not None:
                s.remove_calibration()
            if o.cb_rebin.isChecked():
                by = max(2, o.rebin.value() - o.rebin.value() % 2)
                if by != o.rebin.value():
                    o.rebin.blockSignals(True); o.rebin.setValue(by); o.rebin.blockSignals(False)
                n = len(s.counts) - (len(s.counts) % by)
                if n != len(s.counts):
                    s.counts = s.counts[:n]; s.counts_err = s.counts_err[:n]
                    s.channels = s.channels[:n]
                    if s.energies is not None:
                        s.energies = s.energies[:n]
                s.rebin(by=by)
            if o.cb_smooth.isChecked():
                s.smooth(num=o.smooth_spin.value())
            if o.cb_shift.isChecked():
                s.gain_shift(by=o.shift_box.value(), energy=s.energies is not None)
            if o.cb_yconst.isChecked():
                c = o.yconst.value()
                s.counts = s.counts * c
                s.counts_err = s.counts_err * abs(c)
            if o.cb_xconst.isChecked():
                c = o.xconst.value()
                base = s.energies if s.energies is not None else s.channels
                s.energies = base * c
                s.x = s.energies
                s.x_units = f"Channels × {c:g}" if s.e_units is None else f"Energy ({s.e_units})"
            if o.cb_cr.isChecked() and not s.cps:
                s.normalize(by="livetime"); s.cps = True; s.y_label = "CPS"
        except Exception as exc:  # noqa: BLE001 — show op failures, keep prior plot
            self.statusBar().showMessage(f"  {exc}")
            return
        self.spect = s
        self._refresh()

    def _reset_customize_checks(self):
        for cb in self._cust_checks:
            cb.blockSignals(True); cb.setChecked(False); cb.blockSignals(False)

    def _sync_cps_checkbox(self):
        """Reflect the file's native count-rate state on load."""
        cb = self.spectrum_opts.cb_cr
        cb.blockSignals(True)
        cb.setChecked(bool(self.spect.cps) if self.spect is not None else False)
        cb.blockSignals(False)

    def _remove_calibration(self):
        if not self._guard():
            return
        if self._spect_orig is None or self._spect_orig.energies is None:
            self.statusBar().showMessage("  Spectrum has no calibration to remove")
            return
        self._remove_cal = True
        self._recompute()
        self.statusBar().showMessage("  Calibration removed  ·  Reset Spectrum to restore")

    def _reset_spectrum(self):
        if self._spect_orig is None:
            self.spectrum_page.canvas.reset_view()
            return
        self._remove_cal = False
        self._reset_customize_checks()
        self._sync_cps_checkbox()
        self.spect = self._spect_orig.copy()
        self._refresh()
        self.statusBar().showMessage("  Spectrum reset to originally loaded state")


def main():
    # First positional arg (if any) is treated as a spectrum file to open.
    file_name = None
    for a in sys.argv[1:]:
        if not a.startswith("-"):
            file_name = a
            break

    T.apply_mpl_theme()
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication([])
    app.setApplicationName("WARA")
    app.setStyleSheet(STYLESHEET)
    win = WaraBetaApp(file_name=file_name)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
