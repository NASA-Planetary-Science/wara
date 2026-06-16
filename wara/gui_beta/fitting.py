"""Drag-and-Fit: an always-on-top window that fits the peaks inside an ROI
dragged on the main plot.

Peak shapes (wara.peakfit / wara.advanced_fit): Gaussian, Skewed Gaussian,
Voigt, Pseudo-Voigt, Skewed Voigt, EMG (low-energy tail), Doniach and Hypermet
(Gaussian core + low-energy tail, the HPGe model). Backgrounds: polynomial
(degree 0–7), exponential, and a step continuum (sharp/smooth) — the latter
routes Gaussian peaks through GaussStepFit and Hypermet peaks through
FullHPGePeakFit. It reuses PeakFit.plot() for the rich fit view (residual
panel, components, 3-σ band, reduced χ² title) and wara.advanced_fit.
shape_summary() for the results table, which reports the fitted centroid /
net-count area / FWHM plus the numerically-measured FWTM, the FWTM/FWHM
tailing ratio (1.823 for a pure Gaussian) and the low-energy asymmetry. A
status line below the plot shows the live x/y cursor readout. The ROI can be
adjusted here (spin boxes) or on the main plot; the two stay in sync.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QSlider, QWidget, QTextEdit, QDialogButtonBox, QToolTip,
    QApplication,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from wara.advanced_fit import (
    fit_bkg, PeakAreaLinearBkg, MultiProfilePeakFit, HypermetFit,
    GaussStepFit, FullHPGePeakFit, shape_summary, GAUSS_FWTM_FWHM,
)
from wara.matplotlib_theme import set_theme, DARK
from . import theme as T

# Peak-shape menu → how to build the fit.
#   ("profile", name) → MultiProfilePeakFit(profile=name)
#   ("hypermet", None) → HypermetFit (Gaussian core + low-energy tail)
# "gauss" is routed through MultiProfilePeakFit too; it is identical to a
# plain PeakFit but keeps a single construction path.
PEAK_SHAPES = {
    "Gaussian":         ("profile", "gauss"),
    "Skewed Gaussian":  ("profile", "skewed"),
    "Voigt":            ("profile", "voigt"),
    "Pseudo-Voigt":     ("profile", "pvoigt"),
    "Skewed Voigt":     ("profile", "skewedvoigt"),
    "EMG (low-E tail)": ("profile", "emg"),
    "Doniach":          ("profile", "doniach"),
    "Hypermet":         ("hypermet", None),
}

# Display name → bkg argument for peakfit. Polynomial uses the degree spinner
# (degree 1 == a linear background). The two Step entries select the
# erfc/Heaviside continuum (GaussStepFit / FullHPGePeakFit).
BKG_ARGS = {
    "Polynomial":    None,
    "Exponential":   "exponential",
    "Step (sharp)":  "step-sharp",
    "Step (smooth)": "step-smooth",
}

# Results-table columns for the peak-fit method. FWHM/FWTM carry the
# spectrum's x-units, filled in at display time. FWHM is the fitted
# (Gaussian-sigma) width; FWTM, the tailing ratio and the asymmetry are
# measured numerically from the fitted profile, so they are meaningful for
# every line shape (a pure Gaussian has FWTM/FWHM = 1.823 and Asym = 0).
PEAKFIT_COLS = ["Centroid", "Area", "FWHM", "FWTM", "FWTM/FWHM", "Asym"]
AREA_COLS = ["Region", "Area", ""]

# Match the main Spectrum tab's plot background (pure black).
FIT_PLOT_BG = T.BG_PLOT


def _fmt(v, e):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if e is None or (isinstance(e, float) and np.isnan(e)):
        return f"{v:.2f}"
    return f"{v:.2f} ± {e:.2f}"


class FitCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure(constrained_layout=True, facecolor=FIT_PLOT_BG)
        super().__init__(fig)
        self.setMinimumHeight(340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


class CentroidEnergyDialog(QDialog):
    """Pair each fitted centroid (channel) with a user-entered energy before
    sending the pairs to the Calibration tab.

    The fit reports centroids in the spectrum's x-units; on a calibrated
    spectrum these were mapped back to channels, so a warning is shown.  Energy
    units are picked here unless the Calibration tab has already locked them.
    """

    def __init__(self, parent, channels, calibrated=False,
                 default_units="keV", locked_units=None):
        super().__init__(parent)
        self.setWindowTitle("Send centroids to Calibration")
        self.setStyleSheet(T.STYLESHEET)
        self._channels = list(channels)

        lay = QVBoxLayout(self)
        if calibrated:
            warn = QLabel(
                "⚠  This spectrum is on a calibrated (energy) axis. The fitted "
                "centroids were converted back to channel numbers for the "
                "calibration points.")
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color: {T.ACCENT_AMBER}; font-weight: 600;")
            lay.addWidget(warn)

        # Legacy "Note: calibration has already been initialized" warning.
        if locked_units:
            note = QLabel(
                f"Note: a calibration has already been started — units are locked "
                f"to {locked_units}.")
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {T.ACCENT_RED}; font-weight: 600;")
            lay.addWidget(note)

        # Energy units: locked to the Calibration tab's choice once points exist.
        urow = QHBoxLayout()
        self.units = QComboBox(); self.units.addItems(["eV", "keV", "MeV"])
        if locked_units:
            self.units.setCurrentText(locked_units)
            self.units.setEnabled(False)
            self.units.setToolTip("Units are locked by the existing calibration points")
        else:
            self.units.setCurrentText(default_units)
        urow.addWidget(QLabel("Energy units:")); urow.addStretch(1); urow.addWidget(self.units)
        lay.addLayout(urow)

        lay.addWidget(QLabel("Enter the energy for each fitted centroid, or use "
                             "🔍 to pick it from the database (blank = skip):"))
        self.table = QTableWidget(len(self._channels), 3)
        self.table.setHorizontalHeaderLabels(["Channel", "Energy", ""])
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        for i, ch in enumerate(self._channels):
            ch_item = QTableWidgetItem(f"{ch:.3f}")
            ch_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # read-only
            ch_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 0, ch_item)
            e_item = QTableWidgetItem("")
            e_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 1, e_item)
            # Per-row database picker — fills the Energy cell but leaves it editable.
            btn = QPushButton("🔍"); btn.setObjectName("mini_btn")
            btn.setToolTip("Pick this energy from the nuclear database")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, r=i: self._pick_energy(r))
            self.table.setCellWidget(i, 2, btn)
        self.table.setMinimumWidth(360)
        lay.addWidget(self.table)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def selected_units(self):
        return self.units.currentText()

    def _pick_energy(self, row):
        """Open the database picker and drop the chosen energy into *row*'s
        Energy cell (converted to the selected units). The cell stays editable."""
        from .nuclear import NuclearLinePicker
        picker = NuclearLinePicker(self)
        if picker.exec_() != QDialog.Accepted:
            return
        energy_keV = picker.selected_energy()
        if energy_keV is None:
            return
        value = self._from_keV(energy_keV)
        self.table.item(row, 1).setText(f"{value:.6g}")

    def _from_keV(self, energy_keV):
        """Convert a keV database energy to the dialog's selected units."""
        unit = self.units.currentText()
        if unit == "eV":
            return energy_keV * 1000.0
        if unit == "MeV":
            return energy_keV / 1000.0
        return energy_keV

    def pairs(self):
        """Return [(channel, energy_str), …] for every row (energy may be '')."""
        out = []
        for i in range(self.table.rowCount()):
            e_item = self.table.item(i, 1)
            energy = e_item.text().strip() if e_item else ""
            out.append((self._channels[i], energy))
        return out


class FitWindow(QDialog):
    """Non-modal, stays-on-top fit window driven by an ROI on the main plot."""

    roi_changed = pyqtSignal(float, float)   # emitted when the ROI is edited here
    send_to_calibration = pyqtSignal(list, str)   # ([(channel, energy_str), …], units)
    send_to_efficiency = pyqtSignal(object)       # the peak-fit object
    send_to_resolution = pyqtSignal(object)       # the peak-fit object

    def __init__(self, parent, search):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Peak fitting")
        self.setStyleSheet(T.STYLESHEET)
        self.resize(960, 840)
        set_theme(DARK)               # PeakFit.plot() uses wara's active theme
        self._search = search
        self._xrange = None
        self._roi_lo, self._roi_hi = 0.0, 1.0   # initial ROI (slider bounds)
        self.N = 1000                             # slider resolution
        self.last_fit = None                      # most recent peak-fit object
        self._iso_cache = {}                      # {round(centroid, 3): ID HTML}

        outer = QVBoxLayout(self)

        # ── Method selector ──────────────────────────────────────
        row0 = QHBoxLayout()
        self.method = QComboBox()
        self.method.addItems(["Peak fit", "Net area − linear bkg"])
        self.method.setToolTip(
            "Peak fit: model every peak in the ROI with lmfit.\n"
            "Net area − linear bkg: integrate the counts above a straight "
            "background drawn under the peak region (wara.advanced_fit)."
        )
        row0.addWidget(QLabel("Method:")); row0.addWidget(self.method)
        row0.addStretch(1)
        outer.addLayout(row0)

        # ── Model controls (peak-fit only) ───────────────────────
        self.peakfit_controls = QWidget()
        row1 = QHBoxLayout(self.peakfit_controls)
        row1.setContentsMargins(0, 0, 0, 0)
        self.shape = QComboBox(); self.shape.addItems(list(PEAK_SHAPES.keys()))
        self.shape.setToolTip(
            "Peak shape used for every peak in the ROI.\n"
            "Voigt/Pseudo-Voigt add Lorentzian wings; Skewed/EMG/Hypermet "
            "model the low-energy tail of HPGe peaks (EMG and Hypermet are "
            "the principled tailing models)."
        )
        self.bkg = QComboBox(); self.bkg.addItems(list(BKG_ARGS.keys()))
        self.bkg.setToolTip("Continuum model under the peaks")
        self.degree = QSpinBox(); self.degree.setRange(0, 7); self.degree.setValue(1)
        self.degree.setToolTip("Polynomial degree (degree 1 = linear background)")
        self.cb_shared = QCheckBox("FWHM ∝ √E")
        self.cb_shared.setToolTip("Constrain peak widths to follow the resolution curve  fwhm = a + b·√E")
        row1.addWidget(QLabel("Shape:")); row1.addWidget(self.shape)
        row1.addWidget(QLabel("Background:")); row1.addWidget(self.bkg)
        row1.addWidget(QLabel("Degree:")); row1.addWidget(self.degree)
        row1.addWidget(self.cb_shared); row1.addStretch(1)
        outer.addWidget(self.peakfit_controls)

        # The user-selectable fit options (Peak fit, Hypermet, …) are shown in
        # cyan to flag them as choices the user drives.
        opt_css = f"QComboBox, QSpinBox {{ color: {T.ACCENT_CYAN}; font-weight: 600; }}"
        for w in (self.method, self.shape, self.bkg, self.degree):
            w.setStyleSheet(opt_css)

        # ── Fit plot (residual + components + 3-σ band) ──────────
        self.canvas = FitCanvas()
        outer.addWidget(self.canvas, stretch=1)

        # ── ROI controls: single row below the plot (like legacy advanced fit) ──
        self.roi_lo = QDoubleSpinBox(); self.roi_hi = QDoubleSpinBox()
        self.slider_lo = QSlider(Qt.Horizontal); self.slider_hi = QSlider(Qt.Horizontal)
        for s in (self.roi_lo, self.roi_hi):
            s.setDecimals(2); s.setRange(-1e9, 1e9); s.setSingleStep(1.0)
            s.setMinimumWidth(96)
        for sl in (self.slider_lo, self.slider_hi):
            sl.setRange(0, self.N // 2)
        self.slider_lo.setToolTip("Trim the lower bound of the fit range")
        self.slider_hi.setToolTip("Trim the upper bound of the fit range")
        self.slider_hi.setLayoutDirection(Qt.RightToLeft)
        roi_row = QHBoxLayout()
        roi_row.addWidget(self.roi_lo)
        roi_row.addWidget(self.slider_lo, stretch=1)
        roi_row.addWidget(self.slider_hi, stretch=1)
        roi_row.addWidget(self.roi_hi)
        outer.addLayout(roi_row)

        # ── Results ──────────────────────────────────────────────
        self.lbl_status = QLabel("Drag over peaks on the main plot to fit.")
        self.lbl_status.setObjectName("stat_key")
        outer.addWidget(self.lbl_status)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.table = QTableWidget(0, len(PEAKFIT_COLS))
        self.table.setHorizontalHeaderLabels(PEAKFIT_COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(170)
        self.table.setStyleSheet("QTableView { font-size: 16px; } QHeaderView::section { font-size: 15px; }")
        outer.addWidget(self.table)

        crow = QHBoxLayout()
        self.btn_details = QPushButton("Fit details")
        self.btn_details.setObjectName("action_btn")
        self.btn_details.setToolTip("Show the full lmfit report for the current fit")
        self.btn_details.setEnabled(False)
        self.btn_details.clicked.connect(self._show_fit_details)
        crow.addWidget(self.btn_details)
        self.btn_to_cal = QPushButton("📐  Send centroids to Calibration")
        self.btn_to_cal.setObjectName("yellow_btn")
        self.btn_to_cal.setToolTip(
            "Send the fitted peak centroid(s) to the Calibration tab as points "
            "(channel axis — remove calibration first).")
        self.btn_to_cal.setEnabled(False)
        self.btn_to_cal.setCursor(Qt.PointingHandCursor)
        self.btn_to_cal.clicked.connect(self._emit_centroids)
        crow.addWidget(self.btn_to_cal)
        self.btn_to_eff = QPushButton("📈  Send to Efficiency")
        self.btn_to_eff.setObjectName("fit_btn")
        self.btn_to_eff.setToolTip(
            "Send this peak fit to the Efficiency tab to build an efficiency "
            "point from one of its peaks.")
        self.btn_to_eff.setEnabled(False)
        self.btn_to_eff.setCursor(Qt.PointingHandCursor)
        self.btn_to_eff.clicked.connect(self._emit_to_efficiency)
        crow.addWidget(self.btn_to_eff)
        self.btn_to_res = QPushButton("📏  Send to Resolution")
        self.btn_to_res.setObjectName("danger_btn")   # red — matches the Resolution tab
        self.btn_to_res.setToolTip(
            "Send this peak fit to the Resolution tab to add its peak FWHM(s) to "
            "the resolution (FWHM vs energy) curve.")
        self.btn_to_res.setEnabled(False)
        self.btn_to_res.setCursor(Qt.PointingHandCursor)
        self.btn_to_res.clicked.connect(self._emit_to_resolution)
        crow.addWidget(self.btn_to_res)
        crow.addStretch(1)
        self.btn_close = QPushButton("Close"); self.btn_close.setObjectName("action_btn")
        self.btn_close.clicked.connect(self.close)
        crow.addWidget(self.btn_close)
        outer.addLayout(crow)

        # Live refit on any setting / ROI change
        self.method.currentTextChanged.connect(self._on_method_changed)
        self.shape.currentTextChanged.connect(self._refit)
        self.bkg.currentTextChanged.connect(self._on_bkg_changed)
        self.degree.valueChanged.connect(self._refit_if_poly)
        self.cb_shared.toggled.connect(self._refit)
        self.slider_lo.valueChanged.connect(self._on_lo_slider)
        self.slider_hi.valueChanged.connect(self._on_hi_slider)
        self.roi_lo.valueChanged.connect(self._on_lo_spin)
        self.roi_hi.valueChanged.connect(self._on_hi_spin)

    # ── API used by the main window ──────────────────────────────────────────
    def set_search(self, search):
        self._search = search

    def set_roi(self, xmin, xmax):
        """Set the ROI from the main-plot span (no echo back to the span)."""
        self._roi_lo, self._roi_hi = float(xmin), float(xmax)
        self._xrange = [self._roi_lo, self._roi_hi]
        for spin in (self.roi_lo, self.roi_hi):
            spin.blockSignals(True)
            spin.setRange(self._roi_lo, self._roi_hi)
            spin.blockSignals(False)
        self._set_roi_controls(self._roi_lo, self._roi_hi)
        self._refit()

    # ── ROI control plumbing ─────────────────────────────────────────────────
    def _val_to_slider(self, v):
        span = self._roi_hi - self._roi_lo
        if span <= 0:
            return 0
        frac = (v - self._roi_lo) / span
        return int(round(min(1.0, max(0.0, frac)) * self.N))

    def _slider_to_val(self, pos):
        return self._roi_lo + (pos / self.N) * (self._roi_hi - self._roi_lo)

    def _set_roi_controls(self, lo, hi):
        """Set both sliders and both spin boxes without firing handlers."""
        widgets = (self.roi_lo, self.roi_hi, self.slider_lo, self.slider_hi)
        for wdg in widgets:
            wdg.blockSignals(True)
        self.roi_lo.setValue(lo); self.roi_hi.setValue(hi)
        self.slider_lo.setValue(self._val_to_slider(lo))
        self.slider_hi.setValue(self.N - self._val_to_slider(hi))
        for wdg in widgets:
            wdg.blockSignals(False)

    def _on_lo_slider(self, pos):
        self.roi_lo.blockSignals(True); self.roi_lo.setValue(self._slider_to_val(pos)); self.roi_lo.blockSignals(False)
        self._roi_edited()

    def _on_hi_slider(self, pos):
        self.roi_hi.blockSignals(True); self.roi_hi.setValue(self._slider_to_val(self.N - pos)); self.roi_hi.blockSignals(False)
        self._roi_edited()

    def _on_lo_spin(self, val):
        self.slider_lo.blockSignals(True); self.slider_lo.setValue(self._val_to_slider(val)); self.slider_lo.blockSignals(False)
        self._roi_edited()

    def _on_hi_spin(self, val):
        self.slider_hi.blockSignals(True); self.slider_hi.setValue(self.N - self._val_to_slider(val)); self.slider_hi.blockSignals(False)
        self._roi_edited()

    def _roi_edited(self):
        lo, hi = self.roi_lo.value(), self.roi_hi.value()
        if hi <= lo:
            return
        self._xrange = [lo, hi]
        self._refit()

    def _bkg_arg(self):
        return BKG_ARGS[self.bkg.currentText()] or f"poly{self.degree.value()}"

    def _is_step_bkg(self):
        return self.bkg.currentText().startswith("Step")

    def _on_bkg_changed(self):
        # Degree only matters for a polynomial; the resolution-curve
        # constraint is not available on the step-background classes.
        self.degree.setEnabled(self.bkg.currentText() == "Polynomial")
        step = self._is_step_bkg()
        self.cb_shared.setEnabled(not step)
        if step:
            self.cb_shared.setToolTip(
                "Not available with a step background "
                "(GaussStepFit / FullHPGePeakFit)."
            )
        else:
            self.cb_shared.setToolTip(
                "Constrain peak widths to follow the resolution curve  "
                "fwhm = a + b·√E"
            )
        self._refit()

    def _refit_if_poly(self):
        if self.bkg.currentText() == "Polynomial":
            self._refit()

    def _context_data(self):
        """Return (x, y) arrays for the full initial ROI (for dimmed context)."""
        if self._search is None:
            return None
        spec = self._search.spectrum
        xs = spec.energies if spec.energies is not None else spec.channels
        ys = spec.counts
        mask = (xs >= self._roi_lo) & (xs <= self._roi_hi)
        return xs[mask], ys[mask]

    # ── Method dispatch ──────────────────────────────────────────────────────
    def _is_area_method(self):
        return self.method.currentIndex() == 1

    def _on_method_changed(self):
        area = self._is_area_method()
        self.peakfit_controls.setVisible(not area)
        if area:
            self.slider_lo.setToolTip("Left edge of the peak — counts to its left are background")
            self.slider_hi.setToolTip("Right edge of the peak — counts to its right are background")
            self._set_columns(AREA_COLS)
        else:
            self.slider_lo.setToolTip("Trim the lower bound of the fit range")
            self.slider_hi.setToolTip("Trim the upper bound of the fit range")
            self._set_columns(PEAKFIT_COLS)
        self._refit()

    def _set_columns(self, headers):
        """Resize the results table and set its header labels."""
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

    def _refit(self):
        if self._is_area_method():
            self._refit_area()
        else:
            self._refit_peakfit()
        has_report = (self.last_fit is not None
                      and getattr(self.last_fit, "fit_result", None) is not None)
        self.btn_details.setEnabled(has_report)
        # A new fit means the peaks changed — clear any "✓ Sent" confirmation so
        # the buttons reflect that this fit hasn't been sent yet.
        self._reset_send_buttons()
        self.btn_to_cal.setEnabled(self.last_fit is not None)
        # Efficiency / Resolution need per-peak info (area, centroid, FWHM),
        # which the net-area method does not produce.
        has_peaks = self.last_fit is not None and bool(getattr(self.last_fit, "peak_info", None))
        self.btn_to_eff.setEnabled(has_peaks)
        self.btn_to_res.setEnabled(has_peaks)

    # Default object name + label for each send button (its resting appearance).
    _SEND_DEFAULTS = {
        "btn_to_cal": ("yellow_btn", "📐  Send centroids to Calibration"),
        "btn_to_eff": ("fit_btn", "📈  Send to Efficiency"),
        "btn_to_res": ("danger_btn", "📏  Send to Resolution"),
    }

    def _mark_sent(self, button, sent_text):
        """Switch a send button to the bright green "✓ Sent" confirmation look."""
        button.setText(sent_text)
        button.setObjectName("sent_btn")
        button.style().unpolish(button); button.style().polish(button)

    def _reset_send_buttons(self):
        """Restore both send buttons to their resting label / style."""
        for name, (obj, text) in self._SEND_DEFAULTS.items():
            button = getattr(self, name)
            if button.objectName() != obj:
                button.setObjectName(obj); button.setText(text)
                button.style().unpolish(button); button.style().polish(button)

    def _build_peakfit(self):
        """
        Construct the fit object for the current shape + background selection.

        Returns the fit, or None when the combination is unsupported (a status
        message is set in that case). A ValueError (no peaks in range) is left
        to propagate so the caller can fall back to a background-only fit.
        """
        search = self._search
        xr = list(self._xrange)
        shared = self.cb_shared.isChecked()
        kind, profile = PEAK_SHAPES[self.shape.currentText()]
        bkg_label = self.bkg.currentText()

        if bkg_label.startswith("Step"):
            step = "sharp" if "sharp" in bkg_label else "smooth"
            # The step-background classes are peak-shape specific.
            if kind == "hypermet":
                return FullHPGePeakFit(search, xr, step=step)
            if profile == "gauss":
                return GaussStepFit(search, xr, step=step)
            self.lbl_status.setText(
                "Step background supports the Gaussian or Hypermet shape. "
                "Choose one of those, or use a Polynomial / Exponential "
                "background with this profile."
            )
            return None

        bkg = self._bkg_arg()
        if kind == "hypermet":
            return HypermetFit(search, xr, bkg=bkg, shared_sigma=shared)
        return MultiProfilePeakFit(
            search, xr, bkg=bkg, profile=profile, shared_sigma=shared
        )

    def _refit_peakfit(self):
        if self._xrange is None or self._search is None:
            return
        try:
            fit = self._build_peakfit()
        except ValueError:
            self._fit_bkg_only()
            return
        except Exception as exc:  # noqa: BLE001
            self.last_fit = None
            self.lbl_status.setText(f"Fit failed: {exc}")
            self.table.setRowCount(0)
            self.canvas.figure.clf(); self.canvas.draw_idle()
            return
        if fit is None:                       # unsupported combo (status set)
            self.last_fit = None
            self.table.setRowCount(0)
            self.canvas.figure.clf(); self.canvas.draw_idle()
            return

        self.last_fit = fit
        ctx = self._context_data()
        fig = self.canvas.figure
        fig.clf()
        try:
            with plt.rc_context():
                fit.plot(fig=fig, context_data=ctx)
            fig.patch.set_alpha(1.0); fig.set_facecolor(FIT_PLOT_BG)
            for ax in fig.axes:
                ax.set_facecolor(FIT_PLOT_BG)
                ax.grid(True, color="#3c3c66", linewidth=0.7, alpha=0.8)
        except Exception as exc:  # noqa: BLE001
            self.lbl_status.setText(f"Plot error: {exc}")
        self.canvas.draw_idle()
        self._fill_table(fit)

    def _fit_bkg_only(self):
        """Fit a pure background (no peaks) and display the area."""
        self.last_fit = None
        spec = self._search.spectrum
        bkg_name = self._bkg_arg()
        deg = int(bkg_name.replace("poly", "")) if bkg_name.startswith("poly") else 1
        try:
            result = fit_bkg(spec, list(self._xrange), degree=deg)
        except ValueError as exc:
            self.lbl_status.setText(str(exc))
            self.table.setRowCount(0)
            self.canvas.figure.clf(); self.canvas.draw_idle()
            return

        x_fine = np.linspace(result.x[0], result.x[-1], 500)
        y_fine = result.poly(x_fine)

        ctx = self._context_data()
        fig = self.canvas.figure
        fig.clf()
        with plt.rc_context({"font.size": 12}):
            ax = fig.add_subplot(111)
            if ctx is not None:
                ax.plot(ctx[0], ctx[1], "o", color="#00e5ff", alpha=0.15, ms=5)
            ax.plot(result.x, result.y, "o", color="#00e5ff", alpha=0.5)
            ax.plot(x_fine, y_fine, color="#39ff14", lw=3,
                    label=f"Bkg: {bkg_name}")
            ax.fill_between(x_fine, 0, y_fine, color="#39ff14", alpha=0.08)
            ax.legend(loc="upper right", ncol=1, frameon=False, fontsize=10)
            ax.set_title("Background only (no peaks in range)")
            ax.set_xlabel(spec.x_units)
            fig.patch.set_alpha(1.0); fig.set_facecolor(FIT_PLOT_BG)
            ax.set_facecolor(FIT_PLOT_BG)
            ax.grid(True, color="#3c3c66", linewidth=0.7, alpha=0.8)
        self.canvas.draw_idle()

        self.lbl_status.setText("No peaks in range — background fit only")
        self._set_columns(["", "Area", ""])
        self.table.setRowCount(2)
        rows = [
            ("Raw sum", f"{result.area_raw:.2f} ± {result.area_raw_err:.2f}"),
            ("Fit",     f"{result.area_fit:.2f} ± {result.area_fit_err:.2f}"),
        ]
        for i, (lbl, val) in enumerate(rows):
            for c, text in enumerate([lbl, val, ""]):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, c, item)

    # ── Net area − linear background (wara.advanced_fit.PeakAreaLinearBkg) ─────
    def _refit_area(self):
        self.last_fit = None
        if self._search is None or self._xrange is None:
            return
        spect = self._search.spectrum
        outer_l, outer_r = self._roi_lo, self._roi_hi
        inner_l, inner_r = self.roi_lo.value(), self.roi_hi.value()
        # The two sliders carve the peak region out of the dragged ROI; the
        # flanks (outer→inner on each side) become the background ranges.
        # The spin boxes round to 2 decimals, so an inner edge resting on an ROI
        # boundary can land a hair outside the full-precision outer bound. Clamp
        # the peak edges back into the ROI before validating.
        inner_l = min(max(inner_l, outer_l), outer_r)
        inner_r = min(max(inner_r, outer_l), outer_r)
        if inner_l >= inner_r:
            self.lbl_status.setText("Set the peak start below the peak end, inside the ROI.")
            return
        # Wrap the whole path (calc + draw + table): any failure must surface in
        # the status bar rather than die silently to the terminal and leave the
        # previous (peak-fit) plot on screen.
        try:
            alb = PeakAreaLinearBkg(spect, x1=outer_l, x2=outer_r)
            alb.calculate_peak_area(x1=[outer_l, inner_l], x2=[inner_r, outer_r])
            self._draw_area(alb, spect)
            self._fill_area_table(alb)
        except Exception as exc:  # noqa: BLE001
            self.lbl_status.setText(f"Net-area fit failed: {exc}")
            self.table.setRowCount(0)
            self.canvas.figure.clf(); self.canvas.draw_idle()
            return
        self.lbl_status.setText(
            f"Net area A = {alb.A:.1f} ± {alb.sigA:.1f}    ·    "
            f"Bkg B = {alb.B:.1f} ± {alb.sigB:.1f}"
        )

    def _draw_area(self, alb, spect):
        ch_l, ch_r = alb._ch_outer_l, alb._ch_outer_r
        if spect.energies is None:
            x_full = spect.channels[ch_l:ch_r + 1].astype(float)
        else:
            x_full = spect.energies[ch_l:ch_r + 1]
        y_full = spect.counts[ch_l:ch_r + 1]

        fig = self.canvas.figure
        fig.clf()
        with plt.rc_context({"font.size": 12}):
            ax = fig.add_subplot(111)
            ax.plot(x_full, y_full, drawstyle="steps-mid", color="#00e5ff", lw=1.2,
                    label="data")
            ax.plot(alb._x_full_range, alb.y_eqn, color="#ff5d5d", lw=2,
                    label="linear background")
            # Flanks outside the dashed lines are only the background-sampling
            # regions — they are NOT part of B, so colour them neutrally.
            xf = alb._x_full_range
            flank = (xf <= alb._x_inner_l) | (xf >= alb._x_inner_r)
            # Fills use step="mid" to line up with the steps-mid data line above;
            # step="pre" would shift them half a bin left (visible on narrow peaks).
            ax.fill_between(xf, 0, alb.y_eqn, where=flank, step="mid",
                            color="#8a8fa3", alpha=0.18, label="background regions")
            # B is the background area under the peak region (between the lines).
            ax.fill_between(alb.xr, 0, alb.y_eqn_peak, step="mid",
                            color="#ff5d5d", alpha=0.15, label=f"B = {alb.B:.1f}")
            ax.fill_between(alb.xr, alb.y_eqn_peak, alb.yr, step="mid",
                            color="#39ff14", alpha=0.18, label=f"A = {alb.A:.1f}")
            # inner-edge guides, drawn up to the background line height
            y_il = alb._slope * alb._x_inner_l + alb._intercept
            y_ir = alb._slope * alb._x_inner_r + alb._intercept
            ax.vlines([alb._x_inner_l, alb._x_inner_r], 0, [y_il, y_ir],
                      linestyle=(0, (2, 1.5)), color="#ffd24a", lw=2.4, alpha=0.95)
            ax.legend(loc="upper right", frameon=False, fontsize=10)
            ax.set_xlabel(spect.x_units)
            ax.set_ylabel("Counts")
            ax.set_ylim(bottom=0)
            fig.patch.set_alpha(1.0); fig.set_facecolor(FIT_PLOT_BG)
            ax.set_facecolor(FIT_PLOT_BG)
            ax.grid(True, color="#3c3c66", linewidth=0.7, alpha=0.8)
            ax.tick_params(colors="#c8cbe0", which="both", length=3)
            for sp in ax.spines.values():
                sp.set_color("#3c3c66")
            ax.xaxis.label.set_color("#c8cbe0")
            ax.yaxis.label.set_color("#c8cbe0")
        self.canvas.draw_idle()

    def _fill_area_table(self, alb):
        self._set_columns(AREA_COLS)
        rows = [
            ("Net (A)", f"{alb.A:.2f} ± {alb.sigA:.2f}"),
            ("Bkg (B)", f"{alb.B:.2f} ± {alb.sigB:.2f}"),
        ]
        self.table.setRowCount(len(rows))
        for i, (lbl, val) in enumerate(rows):
            for c, text in enumerate([lbl, val, ""]):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, c, item)

    def _x_unit_label(self):
        """Short x-axis unit for table headers: 'keV'/'MeV'/… or 'ch'."""
        xu = self._search.spectrum.x_units    # "Channels" or "Energy (keV)"
        if "(" in xu and ")" in xu:
            return xu[xu.index("(") + 1:xu.index(")")]
        return "ch"

    def _fill_table(self, fit):
        # shape_summary = PeakFit.summary() (net-count area) plus numerically
        # measured FWTM, the FWTM/FWHM tailing ratio (1.823 for a Gaussian)
        # and the low-energy asymmetry — valid for every line shape.
        df = shape_summary(fit)
        self._iso_cache.clear()                   # centroids changed
        unit = self._x_unit_label()
        # The fitted centroid is the precise energy, so offer per-line isotope ID
        # — but only on a calibrated spectrum (otherwise the centroid is channels).
        calibrated = getattr(self._search.spectrum, "energies", None) is not None
        headers = [f"Centroid ({unit})", "Area", f"FWHM ({unit})",
                   f"FWTM ({unit})", "FWTM/FWHM", "Asym"]
        if calibrated:
            headers.append("ID")
        self._set_columns(headers)
        if calibrated:
            # Keep the ID column narrow so the data columns get the space.
            hh = self.table.horizontalHeader()
            hh.setSectionResizeMode(QHeaderView.Stretch)
            hh.setSectionResizeMode(len(headers) - 1, QHeaderView.Fixed)
            self.table.setColumnWidth(len(headers) - 1, 30)
        self._set_header_tooltips({
            2: "Fitted FWHM (from the Gaussian sigma; exact for a Gaussian, "
               "approximate for Voigt/EMG/Hypermet — see FWTM/FWHM).",
            4: f"FWTM/FWHM, measured from the profile. {GAUSS_FWTM_FWHM:.3f} "
               "for a pure Gaussian; larger means more tailing.",
            5: "Low-energy tail asymmetry (measured at tenth maximum). "
               "0 is symmetric; positive = low-energy tail.",
        })
        self.table.setRowCount(len(df))
        livetime = getattr(self._search.spectrum, "livetime", None)
        for i, row in df.iterrows():
            cells = [_fmt(row["mean"], row["mean_err"]),
                     _fmt(row["area"], row["area_err"]),
                     _fmt(row["fwhm"], row["fwhm_err"]),
                     _fmt(row["fwtm"], None),
                     _fmt(row["fwtm_fwhm"], None),
                     _fmt(row["asymmetry"], None)]
            # Hover tooltips: a count rate on the area (only when a livetime is
            # known) and a relative %FWHM (FWHM / centroid) on the FWHM.
            tips = {}
            if livetime:
                rate = row["area"] / livetime
                rate_err = (row["area_err"] / livetime
                            if row["area_err"] is not None
                            and np.isfinite(row["area_err"]) else None)
                tips[1] = (f"Rate = {_fmt(rate, rate_err)} counts/s"
                           f"  (livetime {livetime:.6g} s)")
            if row["mean"]:
                tips[2] = f"%FWHM = {100.0 * row['fwhm'] / row['mean']:.2f}%"
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                item.setTextAlignment(Qt.AlignCenter)
                if c in tips:
                    item.setToolTip(tips[c])
                self.table.setItem(i, c, item)
            if calibrated:
                self._add_id_button(i, len(cells), float(row["mean"]))

    def _add_id_button(self, table_row, col, centroid):
        """A small per-line button that identifies the fitted centroid energy."""
        btn = QPushButton("⚛"); btn.setObjectName("mini_btn")
        btn.setToolTip("Identify the likely isotope(s) for this fitted centroid")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda _=False, e=centroid, b=btn: self._identify_centroid(e, b))
        self.table.setCellWidget(table_row, col, btn)

    def _identify_centroid(self, energy, button):
        """Identify *energy* (cached per centroid), show it as a popup at the
        button, and keep it as the button's tooltip for later hovers."""
        from . import isotope_id   # lazy: pulls in the heavy identifier
        key = round(energy, 3)
        html = self._iso_cache.get(key)
        if html is None:
            # The other found peaks give context so this centroid can be flagged
            # as a possible escape peak of another line.
            spect = self._search.spectrum
            idx = getattr(self._search, "peaks_idx", None)
            context = [float(spect.x[i]) for i in idx] if idx is not None else []
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                html = isotope_id.lookup_html(energy, context=context)
            except Exception as exc:  # noqa: BLE001 — never crash the fit window
                html = f"Isotope ID failed: {exc}"
            finally:
                QApplication.restoreOverrideCursor()
            self._iso_cache[key] = html
        button.setToolTip(html)
        QToolTip.showText(button.mapToGlobal(button.rect().center()), html, button)

    def _set_header_tooltips(self, tips):
        """Attach tooltips to header sections by column index."""
        for col, tip in tips.items():
            item = self.table.horizontalHeaderItem(col)
            if item is not None:
                item.setToolTip(tip)

    @staticmethod
    def _report_to_html(report, net_areas=None):
        import html as _html
        import re
        net_areas = net_areas or {}
        amp_re = re.compile(r"^(g\d+)_amplitude\s*:")
        out = []
        section = None
        for line in report.splitlines():
            stripped = line.strip()
            # Section headers look like "[[Name]]", optionally followed by a
            # note, e.g. "[[Correlations]] (unreported correlations are < 0.1)".
            # Match on the closing "]]" (not end-of-line) so the noted headers
            # still render as titles instead of leaking into the body.
            if stripped.startswith("[[") and "]]" in stripped:
                close = stripped.index("]]")
                title = stripped[2:close]
                note = stripped[close + 2:].strip()
                section = title.lower()
                head = (
                    f'<h3 style="color:#7b8cff; font-size:16px; '
                    f'margin:16px 0 6px 0; border-bottom:1px solid #2a2a45; '
                    f'padding-bottom:4px;">{_html.escape(title)}'
                )
                if note:
                    head += (f' <span style="color:#8888c0; font-size:12px; '
                             f'font-weight:normal;">{_html.escape(note)}</span>')
                out.append(head + '</h3>')
                continue

            if not stripped:
                out.append('<div style="white-space:pre">&nbsp;</div>')
                continue

            escaped = _html.escape(line)
            if section == "model":
                # The model expression contains '=' (e.g. prefix='g1_'); colour
                # it as one block so the colour can't flip mid-line.
                escaped = f'<span style="color:#c4b5ff">{escaped}</span>'
            elif stripped.startswith("#"):
                escaped = f'<span style="color:#8888c0">{escaped}</span>'
            elif stripped.startswith("C("):
                escaped = f'<span style="color:#ffb300">{escaped}</span>'
            elif ":" in stripped:
                name, rest = escaped.split(":", 1)
                rest = rest.replace(
                    "+/-", '<span style="color:#aab2e0">&#177;</span>')
                escaped = f'<span style="color:#00e5ff">{name}</span>:{rest}'
                # Annotate gN_amplitude (lmfit's integral in counts·x-units)
                # with the net-count area shown in the results table.
                m = amp_re.match(stripped)
                if m and m.group(1) in net_areas:
                    area, area_err = net_areas[m.group(1)]
                    ann = f"net area = {area:,.1f}"
                    if area_err is not None and np.isfinite(area_err):
                        ann += f" ± {area_err:,.1f}"
                    ann += " counts"
                    escaped += (f'  <span style="color:#7cffb2">&#8594; '
                                f'{_html.escape(ann)}</span>')
            elif "=" in stripped:
                parts = escaped.split("=", 1)
                escaped = (f'<span style="color:#00e5ff">{parts[0]}</span>'
                           f'<span style="color:#aab2e0">=</span>{parts[1]}')
            out.append(f'<div style="white-space:pre">{escaped}</div>')
        return (
            '<div style="font-family: Consolas, Cascadia Mono, monospace; '
            'font-size: 14px; line-height: 1.55; color: #f4f6ff;">'
            + "\n".join(out) + '</div>'
        )

    @staticmethod
    def _net_area_map(fit):
        """Map ``g{i}`` -> (net area, net-area error) from the fit summary, so
        the report's ``amplitude`` parameters can be annotated with the same
        net-count area shown in the results table. Peaks are appended g1..gN in
        build order, matching the summary row order."""
        try:
            df = fit.summary()
        except Exception:
            return {}
        areas = {}
        for i, row in df.iterrows():
            areas[f"g{i + 1}"] = (row.get("area"), row.get("area_err"))
        return areas

    def _emit_centroids(self):
        """Convert the fitted centroids to channel positions, let the user enter
        the matching energies, and send the (channel, energy) pairs to the
        Calibration tab."""
        if self.last_fit is None:
            return
        try:
            df = shape_summary(self.last_fit)
            means = [float(m) for m in df["mean"] if np.isfinite(m)]
        except Exception:  # noqa: BLE001
            means = []
        if not means:
            return
        # Calibration is built against channels, so map the fitted centroids
        # (in the spectrum's x-units) back to channel positions. Use cal_channels
        # so API spectra resolve to their real channel space; for an ordinary
        # uncalibrated spectrum x == channels == cal_channels and this is a no-op.
        spect = self._search.spectrum
        calibrated = spect.energies is not None
        channels = [float(np.interp(m, spect.x, spect.cal_channels)) for m in means]
        default_units = spect.e_units if (calibrated and spect.e_units) else "keV"
        dlg = CentroidEnergyDialog(
            self, channels, calibrated=calibrated, default_units=default_units,
            locked_units=getattr(self, "_cal_locked_units", None))
        if dlg.exec_() != QDialog.Accepted:
            return
        pairs = dlg.pairs()
        if pairs:
            self.send_to_calibration.emit(pairs, dlg.selected_units())
            self.lbl_status.setText(
                f"Sent {len(pairs)} centroid(s) to the Calibration tab.")
            self._mark_sent(self.btn_to_cal, "✓  Sent to Calibration")

    def _emit_to_efficiency(self):
        """Hand the current peak fit to the Efficiency tab."""
        if self.last_fit is None or not getattr(self.last_fit, "peak_info", None):
            return
        n = len(self.last_fit.peak_info)
        self.send_to_efficiency.emit(self.last_fit)
        self.lbl_status.setText(f"Sent fit ({n} peak(s)) to the Efficiency tab.")
        self._mark_sent(self.btn_to_eff, "✓  Sent to Efficiency")

    def _emit_to_resolution(self):
        """Hand the current peak fit to the Resolution tab."""
        if self.last_fit is None or not getattr(self.last_fit, "peak_info", None):
            return
        n = len(self.last_fit.peak_info)
        self.send_to_resolution.emit(self.last_fit)
        self.lbl_status.setText(f"Sent fit ({n} peak(s)) to the Resolution tab.")
        self._mark_sent(self.btn_to_res, "✓  Sent to Resolution")

    def _show_fit_details(self):
        fit = self.last_fit
        if fit is None or getattr(fit, "fit_result", None) is None:
            return
        report = fit.fit_result.fit_report()
        dlg = QDialog(self)
        dlg.setWindowTitle("Fit Details")
        dlg.setStyleSheet(self.styleSheet())
        dlg.resize(720, 580)
        lay = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setStyleSheet(
            f"QTextEdit {{ background: {FIT_PLOT_BG}; border: 1px solid #2a2a45; "
            f"border-radius: 6px; padding: 10px; }}"
        )
        text.setHtml(self._report_to_html(report, self._net_area_map(fit)))
        lay.addWidget(text)
        dlg.show()

    def _on_motion(self, event):
        if event.inaxes is not None and event.xdata is not None:
            self.lbl_status.setText(f"x = {event.xdata:.2f}    y = {event.ydata:.1f}")
