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
    QSizePolicy, QSlider, QWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal

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

# Slightly lighter than the main plot (#07070f) for better contrast in the popup.
FIT_PLOT_BG = "#161622"


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


class FitWindow(QDialog):
    """Non-modal, stays-on-top fit window driven by an ROI on the main plot."""

    roi_changed = pyqtSignal(float, float)   # emitted when the ROI is edited here

    def __init__(self, parent, search):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Drag and Fit")
        self.setStyleSheet(T.STYLESHEET)
        self.resize(960, 840)
        set_theme(DARK)               # PeakFit.plot() uses wara's active theme
        self._search = search
        self._xrange = None
        self._roi_lo, self._roi_hi = 0.0, 1.0   # initial ROI (slider bounds)
        self.N = 1000                             # slider resolution
        self.last_fit = None                      # most recent peak-fit object

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

        crow = QHBoxLayout(); crow.addStretch(1)
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
        unit = self._x_unit_label()
        headers = [f"Centroid ({unit})", "Area", f"FWHM ({unit})",
                   f"FWTM ({unit})", "FWTM/FWHM", "Asym"]
        self._set_columns(headers)
        self._set_header_tooltips({
            2: "Fitted FWHM (from the Gaussian sigma; exact for a Gaussian, "
               "approximate for Voigt/EMG/Hypermet — see FWTM/FWHM).",
            4: f"FWTM/FWHM, measured from the profile. {GAUSS_FWTM_FWHM:.3f} "
               "for a pure Gaussian; larger means more tailing.",
            5: "Low-energy tail asymmetry (measured at tenth maximum). "
               "0 is symmetric; positive = low-energy tail.",
        })
        self.table.setRowCount(len(df))
        for i, row in df.iterrows():
            cells = [_fmt(row["mean"], row["mean_err"]),
                     _fmt(row["area"], row["area_err"]),
                     _fmt(row["fwhm"], row["fwhm_err"]),
                     _fmt(row["fwtm"], None),
                     _fmt(row["fwtm_fwhm"], None),
                     _fmt(row["asymmetry"], None)]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, c, item)

    def _set_header_tooltips(self, tips):
        """Attach tooltips to header sections by column index."""
        for col, tip in tips.items():
            item = self.table.horizontalHeaderItem(col)
            if item is not None:
                item.setToolTip(tip)

    def _on_motion(self, event):
        if event.inaxes is not None and event.xdata is not None:
            self.lbl_status.setText(f"x = {event.xdata:.2f}    y = {event.ydata:.1f}")
