"""Resolution tab: detector energy resolution (FWHM vs energy).

A port of the legacy ``gui/_mixins/efficiency.py`` FWHM workflow into the wara GUI
GUI. Peak fits are pushed in from the Drag-and-Fit window; each fitted peak
contributes a (centroid, FWHM) point. The points are fitted with a resolution
model — order 1 ``FWHM = a + b·√E`` or order 2 ``FWHM = a + b·√(E + c·E²)``
(wara.resolution) — optionally forcing the origin (0, 0) and extrapolating the
curve across the whole spectrum. A points table (energy / FWHM / %FWHM) and an
optional Gaussian-components panel are shown.

Bug fixes vs. the legacy mixin (flagged inline):
* energies and FWHMs are kept paired (the legacy code sorted the two lists
  independently, which could mis-pair them);
* de-duplication is by energy only (legacy required *both* the energy and the
  FWHM to be new, dropping a real point that happened to share a FWHM value);
* the black data markers are recoloured for the dark theme.
"""
import lmfit
import numpy as np

from matplotlib.figure import Figure
from matplotlib.colors import to_rgb
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView, QCheckBox,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QSize

from wara import resolution as res
from wara.peakfit import GaussianComponents
from wara.matplotlib_theme import set_theme, DARK

from . import theme as T
from .widgets import hsep, header, ComboBox

RES_PLOT_BG = T.BG_PLOT

# Fit-model combo entries (kept short so the options panel stays narrow; the
# full formulas are in the combo tooltip and on the plot title).
FIT_ORDER1 = "Order 1"
FIT_ORDER2 = "Order 2"
_ORDER = {FIT_ORDER1: 1, FIT_ORDER2: 2}


def _is_dark(color):
    try:
        r, g, b = to_rgb(color)
    except (ValueError, TypeError):
        return False
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.25


def _fit_model(x, y, order):
    """Fit the resolution model (wara.resolution.fwhm1 / fwhm2) to the points.
    The fit is unweighted — a forced (0, 0) origin would have zero uncertainty,
    so the per-point FWHM errors are used for the error bars, not the fit."""
    if order == 1:
        return lmfit.Model(res.fwhm1).fit(y, E=x, a=0, b=0)
    return lmfit.Model(res.fwhm2).fit(y, E=x, a=0, b=0, c=0)


# ── Plot + points table ───────────────────────────────────────────────────────
class ResolutionPage(QWidget):
    """Plot area for the Resolution tab: FWHM-vs-energy fit (optionally with a
    Gaussian-components panel) above the points table."""

    # Column 0 holds a per-row 🗑 delete button.
    TABLE_COLS = ["", "N", "Energy", "FWHM", "%FWHM"]

    def __init__(self):
        super().__init__()
        self.setObjectName("content")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(8, 6), constrained_layout=True, facecolor=RES_PLOT_BG)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.toolbar = NavToolbar(self.canvas, self)
        self.toolbar.setObjectName("plot_toolbar")
        self.toolbar.setIconSize(QSize(22, 22))
        T.recolor_toolbar_icons(self.toolbar, T.TEXT_PRIMARY)
        lay.addWidget(self.toolbar, 0)
        lay.addWidget(self.canvas, 3)

        self.readout = QLabel("")
        self.readout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.readout.setStyleSheet(
            f"background:{RES_PLOT_BG}; font-size:15px; font-weight:700; padding:3px 8px;")
        lay.addWidget(self.readout, 0)

        self.table = QTableWidget(0, len(self.TABLE_COLS))
        self.table.setHorizontalHeaderLabels(self.TABLE_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMaximumHeight(190)
        T.make_headers_readable(self.table)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 32)
        lay.addWidget(self.table, 1)

        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("axes_leave_event", lambda *_: self.readout.setText(""))

        self.show_empty()

    # -- drawing ---------------------------------------------------------------
    def show_empty(self, msg="Send a peak fit to add FWHM points"):
        self.fig.clf()
        self.readout.setText("")
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color=T.BORDER, fontweight="bold", wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        self._restyle()
        self.canvas.draw_idle()
        self._reset_nav()

    def show_resolution(self, x, y, yerr, x_units, e_units, order, can_fit,
                        extrapolate=False, full_x=None, fit_lst=None, show_gauss=False):
        """Plot the FWHM points and (when *can_fit*) the resolution-model fit.

        *x*, *y*, *yerr* are paired arrays (already including the forced origin if
        requested). *fit_lst* enables the Gaussian-components panel."""
        self.fig.clf()
        self.readout.setText("")
        set_theme(DARK)

        gauss = show_gauss and bool(fit_lst)
        if gauss:
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 2])
            ax = self.fig.add_subplot(gs[0, 0])
            ax_gauss = self.fig.add_subplot(gs[1, 0])
        else:
            ax = self.fig.add_subplot(111)
            ax_gauss = None

        if can_fit:
            fit = _fit_model(x, y, order)
            bv = fit.best_values
            erg = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            if order == 1:
                yfit = res.fwhm1(erg, bv["a"], bv["b"])
                label = f"a = {bv['a']:.4g}\nb = {bv['b']:.4g}"
                title = r"$\mathrm{FWHM} = a + b\sqrt{E}$"
            else:
                yfit = res.fwhm2(erg, bv["a"], bv["b"], bv["c"])
                label = f"a = {bv['a']:.4g}\nb = {bv['b']:.4g}\nc = {bv['c']:.4g}"
                title = r"$\mathrm{FWHM} = a + b\sqrt{E + cE^2}$"
            ax.plot(erg, yfit, color=T.ACCENT_GREEN, lw=2.5, label=label, zorder=3)
            ax.set_title(title, color=T.TEXT_PRIMARY, fontsize=13)
            if extrapolate and full_x is not None and len(full_x):
                fx = np.asarray(full_x, float)
                fy = (res.fwhm1(fx, bv["a"], bv["b"]) if order == 1
                      else res.fwhm2(fx, bv["a"], bv["b"], bv["c"]))
                ax.plot(fx, fy, ls="--", lw=2, color=T.ACCENT_AMBER,
                        label="extrapolated", zorder=2)
        else:
            need = order + 1
            ax.set_title(f"Add at least {need} points for an order-{order} fit",
                         color=T.TEXT_DIM, fontsize=12)

        # Data points carry their MEASURED FWHM uncertainties (per-peak fwhm_err),
        # not the model's prediction band as the legacy backend used.
        ax.errorbar(x, y, yerr=yerr, marker="s", ms=7, ls=" ", color=T.ACCENT_CYAN,
                    mfc=T.ACCENT_CYAN, mec=T.ACCENT_CYAN, ecolor=T.ACCENT_RED,
                    elinewidth=2, capsize=5, capthick=2, label="Data", zorder=4)
        ax.set_xlabel(x_units)
        ax.set_ylabel(f"FWHM ({e_units})")
        ax.legend(loc="best", fontsize=10)

        if gauss:
            try:
                GaussianComponents(fit_obj_lst=fit_lst).plot_gauss(
                    plot_type="fwhm", fig=self.fig, ax=ax_gauss)
            except Exception:  # noqa: BLE001 — components are a best-effort extra
                ax_gauss.text(0.5, 0.5, "Gaussian components unavailable",
                              transform=ax_gauss.transAxes, ha="center", va="center",
                              color=T.TEXT_DIM, fontsize=11)
                ax_gauss.set_xticks([]); ax_gauss.set_yticks([])
            self._fix_dark_artists(ax_gauss)   # the components plot may use dark colours
        self._restyle()
        self.canvas.draw_idle()
        self._reset_nav()

    @staticmethod
    def _fix_dark_artists(ax):
        """Recolour the backend's black data markers / lines for the dark theme."""
        for line in ax.lines:
            if line.get_marker() not in ("", " ", "None", None):
                if _is_dark(line.get_markerfacecolor()):
                    line.set_markerfacecolor(T.ACCENT_CYAN)
                if _is_dark(line.get_markeredgecolor()):
                    line.set_markeredgecolor(T.ACCENT_CYAN)
            elif _is_dark(line.get_color()):
                line.set_color(T.TEXT_DIM)
        if ax.get_legend() is not None:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=10)

    def _reset_nav(self):
        self.toolbar.update()
        self.toolbar.push_current()

    def _restyle(self):
        self.fig.patch.set_alpha(1.0)
        self.fig.set_facecolor(RES_PLOT_BG)
        for ax in self.fig.axes:
            ax.set_facecolor(RES_PLOT_BG)
            ax.tick_params(colors=T.TEXT_DIM, which="both", length=3)
            ax.xaxis.label.set_color(T.TEXT_DIM)
            ax.yaxis.label.set_color(T.TEXT_DIM)
            if ax.get_title():
                ax.title.set_color(T.TEXT_PRIMARY)
            for sp_ in ax.spines.values():
                sp_.set_color(T.BORDER)
            ax.grid(True, color="#3c3c66", linewidth=0.7, alpha=0.8)
            leg = ax.get_legend()
            if leg is not None:
                leg.get_frame().set_facecolor(T.BG_PANEL)
                leg.get_frame().set_edgecolor(T.BORDER)
                for txt in leg.get_texts():
                    txt.set_color(T.TEXT_PRIMARY)

    # -- table -----------------------------------------------------------------
    def fill_table(self, records, e_units, on_delete=None):
        """Populate the points table. Each row gets a 🗑 button (column 0) that
        calls ``on_delete(record)`` when clicked."""
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for n, rec in enumerate(records, start=1):
            r = self.table.rowCount()
            self.table.insertRow(r)
            e, f = rec["energy"], rec["fwhm"]
            pct = (f / e * 100.0) if e else float("nan")
            values = [str(n), f"{e:.3f}", f"{f:.3f}",
                      "—" if not np.isfinite(pct) else f"{pct:.2f}"]
            for c, text in enumerate(values, start=1):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)
            if on_delete is not None:
                btn = QPushButton("🗑"); btn.setObjectName("remove_btn")
                btn.setToolTip("Remove this point"); btn.setCursor(Qt.PointingHandCursor)
                btn.setFixedSize(26, 22)
                btn.clicked.connect(lambda _=False, rc=rec: on_delete(rc))
                self.table.setCellWidget(r, 0, btn)
        self.table.blockSignals(False)

    # -- cursor ----------------------------------------------------------------
    def _on_motion(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            self.readout.setText("")
            return
        xlab = ax.get_xlabel() or "x"
        ylab = ax.get_ylabel() or "y"
        self.readout.setText(
            f"<span style='color:{T.ACCENT_CYAN}'>{xlab}: {event.xdata:.3f}</span>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;"
            f"<span style='color:{T.ACCENT_GREEN}'>{ylab}: {event.ydata:.4g}</span>")


# ── Options column ────────────────────────────────────────────────────────────
class ResolutionOptions(QScrollArea):
    """Scrollable options for the Resolution tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(9, 12, 9, 12); lay.setSpacing(6)

        title = QLabel("RESOLUTION"); title.setObjectName("opt_title")
        lay.addWidget(title); lay.addWidget(hsep())

        lay.addWidget(header("PEAK FIT"))
        self.lbl_fit = QLabel(
            "Send a peak fit from the Drag-and-Fit window. Each fitted peak adds "
            "an FWHM point.")
        self.lbl_fit.setObjectName("stat_key"); self.lbl_fit.setWordWrap(True)
        lay.addWidget(self.lbl_fit)

        lay.addWidget(hsep()); lay.addWidget(header("POINTS"))
        hint = QLabel("Click 🗑 next to a point to remove it.")
        hint.setObjectName("stat_key"); hint.setWordWrap(True)
        lay.addWidget(hint)
        self.btn_reset = QPushButton("Reset"); self.btn_reset.setObjectName("danger_btn")
        self.btn_reset.setToolTip("Clear every FWHM point")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_reset)

        lay.addWidget(hsep()); lay.addWidget(header("FIT"))
        frow = QHBoxLayout()
        self.fit_model = ComboBox(); self.fit_model.addItems([FIT_ORDER1, FIT_ORDER2])
        self.fit_model.setToolTip(
            "Resolution model:\n"
            "Order 1: FWHM = a + b·√E (needs ≥ 2 points).\n"
            "Order 2: FWHM = a + b·√(E + c·E²) (needs ≥ 3 points).")
        frow.addWidget(QLabel("Model:")); frow.addStretch(1); frow.addWidget(self.fit_model)
        lay.addLayout(frow)

        self.cb_origin = QCheckBox("Force origin (0, 0)")
        self.cb_origin.setChecked(True)
        self.cb_origin.setToolTip("Include the point (0, 0) in the fit (FWHM → 0 at E → 0)")
        lay.addWidget(self.cb_origin)

        self.cb_extrap = QCheckBox("Extrapolate curve")
        self.cb_extrap.setToolTip("Extend the fitted resolution curve over the full spectrum energy range")
        lay.addWidget(self.cb_extrap)

        self.cb_gauss = QCheckBox("Show fitted peaks")
        self.cb_gauss.setToolTip("Show the Gaussian components of the fitted peaks below the curve")
        lay.addWidget(self.cb_gauss)

        lay.addStretch(1)
        self.setWidget(inner)


# ── Controller ────────────────────────────────────────────────────────────────
class ResolutionController:
    """Wires a ResolutionOptions panel and ResolutionPage to the main app."""

    # Two FWHM points closer than this (in energy units) are the same peak.
    DUP_TOL = 0.5

    def __init__(self, app, options: ResolutionOptions, page: ResolutionPage):
        self.app = app
        self.opts = options
        self.page = page
        self._records = []      # [{energy, fwhm, fwhm_err}], sorted by energy
        self._fit_lst = []      # fit objects (for the Gaussian-components panel)
        self._e_units = "keV"
        self._wire()

    def _wire(self):
        o = self.opts
        o.btn_reset.clicked.connect(self._reset)
        o.fit_model.currentTextChanged.connect(lambda *_: self._replot())
        o.cb_origin.toggled.connect(lambda *_: self._replot())
        o.cb_extrap.toggled.connect(lambda *_: self._replot())
        o.cb_gauss.toggled.connect(lambda *_: self._replot())

    def _status(self, msg):
        self.app.statusBar().showMessage(f"  {msg}")

    def _units(self):
        spect = self.app.spect
        x_units = getattr(spect, "x_units", None) or "x"
        e_units = getattr(spect, "e_units", None) or "ch"
        return x_units, e_units

    # -- receiving a fit -------------------------------------------------------
    def receive_fit(self, fit):
        """Add one FWHM point per peak in the pushed fit."""
        if fit is None or not getattr(fit, "peak_info", None):
            self._status("That fit has no peaks to use for resolution")
            return
        _, self._e_units = self._units()
        added = 0
        for i, info in enumerate(fit.peak_info):
            energy = float(info["mean"])
            fwhm = float(info["fwhm"])
            try:
                fwhm_err = float(fit.peak_err[i]["fwhm_err"])
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                fwhm_err = 0.0
            if self._is_duplicate(energy):
                continue
            self._records.append({"energy": energy, "fwhm": fwhm, "fwhm_err": fwhm_err})
            added += 1
        if added:
            self._fit_lst.append(fit)
        # Keep energy/FWHM paired and ordered by energy (legacy sorted the two
        # lists independently, which could mis-pair them).
        self._records.sort(key=lambda r: r["energy"])
        self._refresh()
        n = len(fit.peak_info)
        self._status(f"Added {added} FWHM point(s) to the Resolution tab "
                     f"({n - added} duplicate(s) skipped)" if added < n
                     else f"Added {added} FWHM point(s) to the Resolution tab")

    def _is_duplicate(self, energy):
        return any(abs(energy - r["energy"]) < self.DUP_TOL for r in self._records)

    # -- points management -----------------------------------------------------
    def _delete_record(self, rec):
        """Remove a single FWHM point (its 🗑 button was clicked)."""
        if rec in self._records:
            self._records.remove(rec)
            self._refresh()
            self._status("Removed FWHM point")

    def _reset(self):
        self._records = []
        self._fit_lst = []
        self.page.fill_table([], self._e_units)
        self.page.show_empty()
        self._status("Resolution points reset")

    # -- plotting --------------------------------------------------------------
    def _refresh(self):
        self.page.fill_table(self._records, self._e_units, on_delete=self._delete_record)
        self._replot()

    def _replot(self):
        if not self._records:
            self.page.show_empty()
            return
        x_units, e_units = self._units()
        xs = [r["energy"] for r in self._records]
        ys = [r["fwhm"] for r in self._records]
        es = [r["fwhm_err"] for r in self._records]
        # The forced origin participates in the fit but is not shown in the table.
        if self.opts.cb_origin.isChecked():
            xs = [0.0] + xs
            ys = [0.0] + ys
            es = [0.0] + es
        order = _ORDER[self.opts.fit_model.currentText()]
        # Order 1 needs ≥ 2 points, order 2 ≥ 3.
        can_fit = len(xs) >= order + 1
        full_x = self.app.spect.x if self.app.spect is not None else None
        try:
            self.page.show_resolution(
                np.asarray(xs, float), np.asarray(ys, float), np.asarray(es, float),
                x_units, e_units, order, can_fit,
                extrapolate=self.opts.cb_extrap.isChecked(), full_x=full_x,
                fit_lst=self._fit_lst, show_gauss=self.opts.cb_gauss.isChecked())
        except Exception as exc:  # noqa: BLE001 — a failed fit shouldn't kill the GUI
            self._status(f"Resolution fit failed: {exc}")
            self.page.show_resolution(
                np.asarray(xs, float), np.asarray(ys, float), np.asarray(es, float),
                x_units, e_units, order, can_fit=False, fit_lst=None)
