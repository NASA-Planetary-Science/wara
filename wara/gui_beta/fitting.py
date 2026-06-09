"""Drag-and-Fit: an always-on-top window that fits the peaks inside an ROI
dragged on the main plot, using wara.peakfit (Gaussian / Skewed Gaussian +
linear / quadratic / exponential / polynomial backgrounds). It reuses
PeakFit.plot() for the rich fit view (residual panel, components, 3-σ band,
reduced χ² title) and PeakFit.summary() for the results table. The ROI can be
adjusted here (spin boxes) or on the main plot; the two stay in sync.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QSlider,
)
from PyQt5.QtCore import Qt, pyqtSignal

from wara import peakfit
from wara.advanced_fit import fit_bkg
from wara.matplotlib_theme import set_theme, DARK
from . import theme as T

# Display name → bkg argument for peakfit. Polynomial uses the degree spinner
# (degree 1 == a linear background).
BKG_ARGS = {"Polynomial": None, "Exponential": "exponential"}

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
        self.resize(960, 780)
        set_theme(DARK)               # PeakFit.plot() uses wara's active theme
        self._search = search
        self._xrange = None
        self._roi_lo, self._roi_hi = 0.0, 1.0   # initial ROI (slider bounds)
        self.N = 1000                             # slider resolution

        outer = QVBoxLayout(self)

        # ── Model controls ───────────────────────────────────────
        row1 = QHBoxLayout()
        self.shape = QComboBox(); self.shape.addItems(["Gaussian", "Skewed Gaussian"])
        self.shape.setToolTip("Peak shape used for every peak in the ROI")
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
        outer.addLayout(row1)

        # ── ROI controls: slider + spin box per bound (synced w/ main span) ──
        self.roi_lo = QDoubleSpinBox(); self.roi_hi = QDoubleSpinBox()
        self.slider_lo = QSlider(Qt.Horizontal); self.slider_hi = QSlider(Qt.Horizontal)
        for s in (self.roi_lo, self.roi_hi):
            s.setDecimals(2); s.setRange(-1e9, 1e9); s.setSingleStep(1.0)
            s.setMinimumWidth(96)
        for sl in (self.slider_lo, self.slider_hi):
            sl.setRange(0, self.N)
        self.slider_lo.setToolTip("Trim the lower bound of the fit range within the selected ROI")
        self.slider_hi.setToolTip("Trim the upper bound of the fit range within the selected ROI")
        for label, slider, spin in (("Fit low ", self.slider_lo, self.roi_lo),
                                    ("Fit high", self.slider_hi, self.roi_hi)):
            r = QHBoxLayout()
            r.addWidget(QLabel(label)); r.addWidget(slider, stretch=1); r.addWidget(spin)
            outer.addLayout(r)

        # ── Fit plot (residual + components + 3-σ band) ──────────
        self.canvas = FitCanvas()
        outer.addWidget(self.canvas, stretch=1)

        # ── Results ──────────────────────────────────────────────
        self.lbl_status = QLabel("Drag over peaks on the main plot to fit.")
        self.lbl_status.setObjectName("stat_key")
        outer.addWidget(self.lbl_status)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Centroid", "Area", "FWHM"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(170)
        self.table.setStyleSheet("QTableView { font-size: 17px; } QHeaderView::section { font-size: 17px; }")
        outer.addWidget(self.table)

        crow = QHBoxLayout(); crow.addStretch(1)
        self.btn_close = QPushButton("Close"); self.btn_close.setObjectName("action_btn")
        self.btn_close.clicked.connect(self.close)
        crow.addWidget(self.btn_close)
        outer.addLayout(crow)

        # Live refit on any setting / ROI change
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
        self.slider_hi.setValue(self._val_to_slider(hi))
        for wdg in widgets:
            wdg.blockSignals(False)

    def _on_lo_slider(self, pos):
        self.roi_lo.blockSignals(True); self.roi_lo.setValue(self._slider_to_val(pos)); self.roi_lo.blockSignals(False)
        self._roi_edited()

    def _on_hi_slider(self, pos):
        self.roi_hi.blockSignals(True); self.roi_hi.setValue(self._slider_to_val(pos)); self.roi_hi.blockSignals(False)
        self._roi_edited()

    def _on_lo_spin(self, val):
        self.slider_lo.blockSignals(True); self.slider_lo.setValue(self._val_to_slider(val)); self.slider_lo.blockSignals(False)
        self._roi_edited()

    def _on_hi_spin(self, val):
        self.slider_hi.blockSignals(True); self.slider_hi.setValue(self._val_to_slider(val)); self.slider_hi.blockSignals(False)
        self._roi_edited()

    def _roi_edited(self):
        lo, hi = self.roi_lo.value(), self.roi_hi.value()
        if hi <= lo:
            return
        self._xrange = [lo, hi]
        self._refit()

    def _bkg_arg(self):
        return BKG_ARGS[self.bkg.currentText()] or f"poly{self.degree.value()}"

    def _on_bkg_changed(self):
        self.degree.setEnabled(self.bkg.currentText() == "Polynomial")
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

    def _refit(self):
        if self._xrange is None or self._search is None:
            return
        try:
            fit = peakfit.PeakFit(
                self._search, list(self._xrange), bkg=self._bkg_arg(),
                skew=(self.shape.currentText() == "Skewed Gaussian"),
                shared_sigma=self.cb_shared.isChecked())
        except ValueError:
            self._fit_bkg_only()
            return
        except Exception as exc:  # noqa: BLE001
            self.lbl_status.setText(f"Fit failed: {exc}")
            self.table.setRowCount(0)
            self.canvas.figure.clf(); self.canvas.draw_idle()
            return

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
        self.table.setHorizontalHeaderLabels(["", "Area", ""])
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

    def _fill_table(self, fit):
        df = fit.summary()
        self.table.setRowCount(len(df))
        for i, row in df.iterrows():
            cells = [_fmt(row["mean"], row["mean_err"]),
                     _fmt(row["area"], row["area_err"]),
                     _fmt(row["fwhm"], row["fwhm_err"])]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, c, item)

    def _on_motion(self, event):
        if event.inaxes is not None and event.xdata is not None:
            self.lbl_status.setText(f"x = {event.xdata:.2f}    y = {event.ydata:.1f}")
