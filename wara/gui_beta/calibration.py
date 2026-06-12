"""Calibration tab: build an energy calibration E = f(channel) and apply it.

A faithful port of the legacy ``gui/_mixins/calibration.py`` workflow into the
beta GUI:

* Calibration points live in an editable table (Use · Channel · Energy).
* Channel centroids are pushed in from the Drag-and-Fit window
  (``add_centroids``) or typed by hand into the table.
* Energies are typed into the table; units are eV / keV / MeV and can be
  changed at any time.
* Fit a polynomial of degree 1–3, optionally forcing the origin (0, 0).
* Or set the calibration directly from polynomial coefficients a + b·ch + c·ch²
  + d·ch³ (legacy "set equations").
* Apply the calibration to the active spectrum, or remove an existing one.

Anything beyond the legacy feature set (smart auto-matching, piecewise models,
Nuclear-Database energy sourcing) is intentionally left out — it will be built
on top of this baseline.
"""
import io

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QLineEdit, QCheckBox, QMessageBox, QApplication,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from wara import spectrum as sp
from wara import energy_calibration as ec
from wara.matplotlib_theme import set_theme, DARK

from . import theme as T
from .widgets import hsep, header, stat_row, ComboBox, SpinBox

CAL_PLOT_BG = "#161622"

# Points-table columns.
USE_COL, CH_COL, E_COL = range(3)


def _mathtext_pixmap(mathtext, color=T.TEXT_PRIMARY, fontsize=11):
    """Render a matplotlib *mathtext* string (``$...$``) to a QPixmap.

    Qt's QLabel can't typeset math, so we let matplotlib's built-in mathtext
    engine draw the equation onto a transparent canvas and hand back the
    rasterised result. Rendered at 2x and tagged with a device-pixel-ratio so
    it stays crisp on HiDPI screens without looking oversized.
    """
    fig = Figure(figsize=(0.1, 0.1))
    fig.patch.set_alpha(0.0)
    fig.text(0.0, 0.0, mathtext, color=color, fontsize=fontsize)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True,
                bbox_inches="tight", pad_inches=0.02, dpi=200)
    buf.seek(0)
    pix = QPixmap()
    pix.loadFromData(buf.getvalue(), "PNG")
    pix.setDevicePixelRatio(2.0)   # 200 dpi ≈ 2x a 100-dpi screen
    return pix


# ── Plot column ──────────────────────────────────────────────────────────────
class CalibrationPage(QWidget):
    """Plot area for the Calibration tab: calibration curve + residuals."""

    def __init__(self):
        super().__init__()
        self.setObjectName("content")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(8, 6), constrained_layout=True, facecolor=CAL_PLOT_BG)
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas)
        self.show_empty()

    def show_empty(self, msg="Add at least two calibration points, then Calibrate"):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color=T.BORDER, fontweight="bold", wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        self._restyle()
        self.canvas.draw_idle()

    def show_polynomial(self, cal):
        """Draw an EnergyCalibration fit with its residual panel."""
        self.fig.clf()
        set_theme(DARK)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 4])
        ax_res = self.fig.add_subplot(gs[0, 0])
        ax_fit = self.fig.add_subplot(gs[1, 0])
        cal.plot(residual=True, ax_fit=ax_fit, ax_res=ax_res)
        # EnergyCalibration.plot labels the axis "Energy [units]"; the beta GUI
        # uses parentheses for units, so relabel.
        ax_fit.set_ylabel(f"Energy ({cal.e_units})")
        self._recolor_points(ax_fit)
        self._restyle()
        self.canvas.draw_idle()

    @staticmethod
    def _recolor_points(ax):
        """EnergyCalibration.plot draws the data as black squares, which vanish
        on the dark background — repaint them in a light, high-contrast colour."""
        for line in ax.lines:
            if line.get_marker() in ("s", "o", "."):
                line.set_markerfacecolor(T.ACCENT_CYAN)
                line.set_markeredgecolor(T.ACCENT_CYAN)
        # The legend captured the data marker while it was still black; rebuild
        # it from the now-cyan artists so its swatch keeps the errorbar look
        # (cyan square + red bars), not just a bare marker.
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=10)

    def show_equation(self, channels, predicted, label, units):
        """Draw a calibration curve set directly from coefficients."""
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.plot(channels, predicted, lw=2.5, color=T.ACCENT_GREEN, label=label)
        ax.set_xlabel("Channels")
        ax.set_ylabel(f"Energy ({units})")
        ax.legend(fontsize=11)
        self._restyle()
        self.canvas.draw_idle()

    def _restyle(self):
        """Force the calibration backend's own theme onto the dark GUI palette."""
        self.fig.patch.set_alpha(1.0)
        self.fig.set_facecolor(CAL_PLOT_BG)
        for ax in self.fig.axes:
            ax.set_facecolor(CAL_PLOT_BG)
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


# ── Options column ───────────────────────────────────────────────────────────
class CalibrationOptions(QScrollArea):
    """Scrollable options for the Calibration tab. Widgets are exposed as
    attributes; the CalibrationController wires the behaviour."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(9, 12, 9, 12); lay.setSpacing(6)

        title = QLabel("CALIBRATION"); title.setObjectName("opt_title")
        lay.addWidget(title); lay.addWidget(hsep())

        # ── Calibration points table ─────────────────────────────────
        lay.addWidget(header("POINTS"))
        self.tbl_points = QTableWidget(0, 3)
        self.tbl_points.setHorizontalHeaderLabels(["Use", "Channel", "Energy"])
        self.tbl_points.verticalHeader().setVisible(False)
        self.tbl_points.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_points.setMinimumHeight(170)
        hh = self.tbl_points.horizontalHeader()
        hh.setSectionResizeMode(USE_COL, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(CH_COL, QHeaderView.Stretch)
        hh.setSectionResizeMode(E_COL, QHeaderView.Stretch)
        self.tbl_points.setToolTip(
            "Channel ↔ energy pairs for the fit. Untick a row to keep it on the "
            "plot but exclude it from the fit. Both columns are editable.")
        lay.addWidget(self.tbl_points)

        prow = QHBoxLayout()
        self.btn_add_row = QPushButton("Add row"); self.btn_add_row.setObjectName("mini_btn")
        self.btn_add_row.setToolTip("Add an empty row for manual channel/energy entry")
        self.btn_remove = QPushButton("Remove"); self.btn_remove.setObjectName("mini_btn")
        self.btn_remove.setToolTip("Remove the selected point row(s)")
        self.btn_reset = QPushButton("Reset"); self.btn_reset.setObjectName("mini_btn")
        self.btn_reset.setToolTip("Clear every calibration point and the plot")
        for b in (self.btn_add_row, self.btn_remove, self.btn_reset):
            b.setCursor(Qt.PointingHandCursor); prow.addWidget(b)
        prow.addStretch(1)
        lay.addLayout(prow)
        self.cb_origin = QCheckBox("Force origin (0, 0)")
        self.cb_origin.setChecked(True)
        self.cb_origin.setToolTip("Include the point (channel 0, energy 0) in the fit")
        lay.addWidget(self.cb_origin)

        # ── Polynomial fit (refits automatically on every change) ────
        lay.addWidget(hsep()); lay.addWidget(header("FIT"))
        urow = QHBoxLayout()
        self.units = ComboBox(); self.units.addItems(["eV", "keV", "MeV"])
        self.units.setCurrentText("keV")
        self.units.setToolTip(
            "Energy units of the values entered above. Change it any time — the "
            "entered energies are converted to the new units (e.g. keV → MeV "
            "divides by 1000) and the fit re-runs.")
        urow.addWidget(QLabel("Units:")); urow.addStretch(1); urow.addWidget(self.units)
        lay.addLayout(urow)

        drow = QHBoxLayout()
        self.degree = SpinBox(); self.degree.setRange(1, 3); self.degree.setValue(1)
        self.degree.setToolTip("Polynomial degree (1 = linear, 2 = quadratic, 3 = cubic)")
        drow.addWidget(QLabel("Degree:")); drow.addStretch(1); drow.addWidget(self.degree)
        lay.addLayout(drow)

        # ── Set from equation ────────────────────────────────────────
        lay.addWidget(hsep()); lay.addWidget(header("SET FROM EQUATION"))
        eqn_help = QLabel()
        eqn_help.setPixmap(_mathtext_pixmap(
            r"$E = a + b\cdot ch + c\cdot ch^{2} + d\cdot ch^{3}$"))
        eqn_help.setToolTip("E = a + b·ch + c·ch² + d·ch³")
        lay.addWidget(eqn_help)

        # One labelled row per coefficient ("a =", "b =", ...) so each box is
        # tied to its constant in the equation above.
        self.coef_a = QLineEdit(); self.coef_a.setPlaceholderText("constant")
        self.coef_b = QLineEdit(); self.coef_b.setPlaceholderText("· ch")
        self.coef_c = QLineEdit(); self.coef_c.setPlaceholderText("· ch²  (optional)")
        self.coef_d = QLineEdit(); self.coef_d.setPlaceholderText("· ch³  (optional)")
        for name, ed in (("a", self.coef_a), ("b", self.coef_b),
                         ("c", self.coef_c), ("d", self.coef_d)):
            crow = QHBoxLayout(); crow.setSpacing(5)
            tag = QLabel(f"{name} ="); tag.setObjectName("stat_key"); tag.setMinimumWidth(24)
            crow.addWidget(tag); crow.addWidget(ed)
            lay.addLayout(crow)
        self.btn_set_eqn = QPushButton("Set from equation"); self.btn_set_eqn.setObjectName("action_btn")
        self.btn_set_eqn.setToolTip("Build the calibration directly from the coefficients above")
        self.btn_set_eqn.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_set_eqn)

        # ── Result + apply ───────────────────────────────────────────
        lay.addWidget(hsep()); lay.addWidget(header("RESULT"))
        r, self.val_r2 = stat_row("R²", T.ACCENT_GREEN); lay.addWidget(r)

        self.btn_apply = QPushButton("Apply to spectrum"); self.btn_apply.setObjectName("primary_btn")
        self.btn_apply.setToolTip("Replace the spectrum's x-axis with the calibrated energy")
        self.btn_apply.setEnabled(False)
        self.btn_remove_cal = QPushButton("Remove Calibration"); self.btn_remove_cal.setObjectName("danger_btn")
        self.btn_remove_cal.setToolTip("Strip the energy axis and revert to channel numbers")
        for b in (self.btn_apply, self.btn_remove_cal):
            b.setCursor(Qt.PointingHandCursor); lay.addWidget(b)

        lay.addStretch(1)
        self.setWidget(inner)


# ── Controller ───────────────────────────────────────────────────────────────
class CalibrationController:
    """Wires a CalibrationOptions panel and CalibrationPage to the main app."""

    def __init__(self, app, options: CalibrationOptions, page: CalibrationPage):
        self.app = app
        self.opts = options
        self.page = page
        self.predicted = None      # energy predicted for every channel
        self.ecal_eqn = ""         # textual equation, stored on the spectrum
        self._loading = False      # suppress auto-refit during bulk table edits
        self._units = options.units.currentText()   # units the table values are in
        self._wire()

    # -- helpers ---------------------------------------------------------------
    def _status(self, msg):
        self.app.statusBar().showMessage(f"  {msg}")

    def _wire(self):
        o = self.opts
        o.btn_add_row.clicked.connect(lambda: (self._add_row(), self._auto_calibrate()))
        o.btn_remove.clicked.connect(self._remove_rows)
        o.btn_reset.clicked.connect(self._reset)
        o.btn_set_eqn.clicked.connect(self._calibrate_equation)
        o.btn_apply.clicked.connect(self._apply)
        o.btn_remove_cal.clicked.connect(self.app._remove_calibration)
        # Re-fit automatically whenever the points or fit settings change, so a
        # calibration line is always drawn from the current pairs (legacy behaviour).
        o.tbl_points.cellChanged.connect(self._on_cell_changed)
        o.degree.valueChanged.connect(lambda *_: self._auto_calibrate())
        o.cb_origin.toggled.connect(lambda *_: self._auto_calibrate())
        # Units are freely changeable; converting the entered values to the new
        # units (then re-fitting) is handled in _on_units_changed.
        o.units.currentTextChanged.connect(self._on_units_changed)

    def _on_cell_changed(self, *_):
        if not self._loading:
            self._auto_calibrate()

    def _set_result(self, eqn, r2=None):
        self.ecal_eqn = eqn
        self.opts.val_r2.setText(f"{r2:.6f}" if r2 is not None else "—")
        self.opts.btn_apply.setEnabled(self.predicted is not None)

    def _fill_coeffs(self, coeffs):
        """Populate the 'set from equation' boxes with the fitted coefficients
        (a, b, c, d) so the two paths always agree. Unused terms are cleared."""
        boxes = (self.opts.coef_a, self.opts.coef_b, self.opts.coef_c, self.opts.coef_d)
        for i, ed in enumerate(boxes):
            ed.blockSignals(True)
            ed.setText(f"{coeffs[i]:.6E}" if i < len(coeffs) else "")
            ed.blockSignals(False)

    # -- units ------------------------------------------------------------------
    # Energy-unit scale factors relative to eV.
    _UNIT_SCALE = {"eV": 1.0, "keV": 1.0e3, "MeV": 1.0e6}

    def _has_energy(self):
        """True once at least one row carries a usable channel+energy pair."""
        return bool(self._read_points(used_only=False)[1])

    def locked_units(self):
        """Energy-units string currently in effect, reported to the Drag-and-Fit
        window so pushed centroids share the calibration's units. Returns None
        until an energy has been entered (no calibration started yet). The combo
        itself stays editable in the Calibration tab; the name is kept for the
        Drag-and-Fit handshake in app.py/fitting.py.
        """
        return self.opts.units.currentText() if self._has_energy() else None

    def _on_units_changed(self, new_units):
        """Convert every entered value to the newly selected units, then re-fit.

        Changing the combo is a unit *conversion*, not just a relabel: the
        numbers were typed in the previously selected units, so e.g. switching
        keV → MeV must divide them by 1000. We rescale both the table's energy
        cells and the 'set from equation' coefficients (every polynomial
        coefficient scales by the same factor), then refit so the curve, R²,
        and calibrated axis all follow.
        """
        old_units = self._units
        if new_units == old_units or new_units not in self._UNIT_SCALE:
            return
        # value_new = value_old * scale_old / scale_new
        factor = self._UNIT_SCALE[old_units] / self._UNIT_SCALE[new_units]
        self._units = new_units
        self._convert_values(factor)
        self._auto_calibrate()

    def _convert_values(self, factor):
        """Multiply all numeric energy cells and equation coefficients by
        *factor* (a unit-conversion scale). Blank/non-numeric entries are left
        untouched."""
        o = self.opts
        was_loading = self._loading
        self._loading = True               # batch the edits; caller refits once
        for r in range(o.tbl_points.rowCount()):
            e_item = o.tbl_points.item(r, E_COL)
            try:
                val = float((e_item.text() if e_item else "").strip())
            except (AttributeError, ValueError):
                continue
            o.tbl_points.setItem(r, E_COL, QTableWidgetItem(f"{val * factor:.10g}"))
        self._loading = was_loading
        for ed in (o.coef_a, o.coef_b, o.coef_c, o.coef_d):
            txt = ed.text().strip()
            if not txt:
                continue
            try:
                c = float(txt)
            except ValueError:
                continue
            ed.blockSignals(True)
            ed.setText(f"{c * factor:.6E}")
            ed.blockSignals(False)

    # -- points table ----------------------------------------------------------
    def _add_row(self, channel="", energy="", use=True):
        o = self.opts
        was_loading = self._loading
        self._loading = True            # batch the item writes; caller refits once
        row = o.tbl_points.rowCount()
        o.tbl_points.insertRow(row)
        use_item = QTableWidgetItem()
        use_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        use_item.setCheckState(Qt.Checked if use else Qt.Unchecked)
        use_item.setTextAlignment(Qt.AlignCenter)
        o.tbl_points.setItem(row, USE_COL, use_item)
        ch_txt = f"{channel:.3f}" if isinstance(channel, (int, float)) else str(channel)
        e_txt = f"{energy:.4g}" if isinstance(energy, (int, float)) else str(energy)
        o.tbl_points.setItem(row, CH_COL, QTableWidgetItem(ch_txt))
        o.tbl_points.setItem(row, E_COL, QTableWidgetItem(e_txt))
        self._loading = was_loading
        return row

    def _remove_rows(self):
        rows = sorted({i.row() for i in self.opts.tbl_points.selectedIndexes()}, reverse=True)
        self._loading = True
        for r in rows:
            self.opts.tbl_points.removeRow(r)
        self._loading = False
        self._auto_calibrate()

    def _reset(self):
        self._loading = True
        self.opts.tbl_points.setRowCount(0)
        self._loading = False
        self.predicted = None
        self._fill_coeffs([])              # clear the equation boxes too
        self._set_result("—")
        self.page.show_empty()
        self._status("Calibration reset")

    def _read_points(self, used_only=True):
        """Return (channels, energies) for rows with valid numbers."""
        o = self.opts
        chans, ergs = [], []
        for r in range(o.tbl_points.rowCount()):
            use = o.tbl_points.item(r, USE_COL).checkState() == Qt.Checked
            if used_only and not use:
                continue
            ch_item, e_item = o.tbl_points.item(r, CH_COL), o.tbl_points.item(r, E_COL)
            try:
                ch = float((ch_item.text() if ch_item else "").strip())
                e = float((e_item.text() if e_item else "").strip())
            except (AttributeError, ValueError):
                continue
            chans.append(ch); ergs.append(e)
        return chans, ergs

    def _existing_channels(self):
        """Channel values already in the table (energy column may be empty)."""
        out = []
        for r in range(self.opts.tbl_points.rowCount()):
            ch_item = self.opts.tbl_points.item(r, CH_COL)
            try:
                out.append(round(float(ch_item.text().strip()), 3))
            except (AttributeError, ValueError):
                continue
        return out

    def _dup_tolerance(self, ch):
        """Half a peak FWHM at *ch* — two centroids closer than this are the
        same (unresolvable) peak. Falls back to 1 channel without a search."""
        s = self.app.search
        if s is not None and hasattr(s, "fwhm"):
            try:
                return max(0.5 * float(s.fwhm(ch)), 0.5)
            except Exception:  # noqa: BLE001 — fall back to a fixed tolerance
                pass
        return 1.0

    def _duplicate_channel(self, ch):
        """Return an existing channel within tolerance of *ch*, else None."""
        tol = self._dup_tolerance(ch)
        for existing in self._existing_channels():
            if abs(existing - ch) <= tol:
                return existing
        return None

    # -- centroids -------------------------------------------------------------
    def _add_unique(self, pairs):
        """Add (channel, energy) rows, skipping any channel that duplicates a
        peak already in the table (same peak within half a FWHM). Returns
        (n_added, [duplicate_channels])."""
        added, dupes = 0, []
        for ch, energy in pairs:
            ch = float(ch)
            if self._duplicate_channel(ch) is not None:
                dupes.append(ch)
                continue
            self._add_row(channel=ch, energy=str(energy))
            added += 1
        return added, dupes

    def _warn_duplicates(self, dupes):
        self._show_warning(
            "Duplicate peak",
            "Each peak can only be used once in a calibration. These were "
            "already present and were skipped:\n  "
            + ", ".join(f"channel {d:.2f}" for d in dupes))

    def _show_warning(self, title, text):
        """Modal warning that stays above the (always-on-top) Drag-and-Fit
        window — parenting to the main window alone leaves it hidden behind that
        window, which blocks the event loop and freezes the GUI."""
        parent = QApplication.activeWindow() or self.app
        box = QMessageBox(QMessageBox.Warning, title, text, QMessageBox.Ok, parent)
        box.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        box.exec_()

    def add_centroids(self, pairs, units=None):
        """Add (channel, energy) pairs pushed from the Drag-and-Fit window.

        ``pairs`` is a list of ``(channel, energy_str)``; the energy may be an
        empty string, in which case only the channel is filled in. ``units`` (if
        given) sets the energy units. A peak that is already in the calibration
        (e.g. the same peak fitted twice) is rejected with a warning instead of
        being added again."""
        # Adopt the pushed units only before a calibration is under way (no
        # energy yet) — here it declares the units of the incoming energies. Once
        # points exist the Drag-and-Fit dialog locks to the tab's units, so this
        # never silently reinterprets values already in the table.
        if units and not self._has_energy():
            self.opts.units.setCurrentText(units)
        added, dupes = self._add_unique(pairs)
        self._auto_calibrate()
        if dupes:
            self._warn_duplicates(dupes)
        if added:
            self._status(f"Added {added} centroid(s) from the fit")

    # -- validation ------------------------------------------------------------
    def _validation_error(self, chans, ergs):
        """Return a message if the points are inconsistent (a duplicate peak or
        a repeated energy), else None. Mirrors the legacy 'not repeated' guard
        for points typed directly into the table."""
        for i in range(len(chans)):
            for j in range(i + 1, len(chans)):
                if abs(chans[i] - chans[j]) <= self._dup_tolerance(chans[i]):
                    return (f"Two points share channel ~{chans[i]:.2f} — "
                            "each peak can only be used once.")
        seen = set()
        for e in ergs:
            key = round(e, 3)
            if key in seen:
                return (f"Two points share energy {e:g} — "
                        "each energy can only be used once.")
            seen.add(key)
        return None

    def _fit_failed(self, msg):
        """Clear the result and show *msg* on the plot + status bar."""
        self.predicted = None
        self._set_result("—")
        self.page.show_empty(msg)
        self._status(msg)

    # -- polynomial calibration (automatic) ------------------------------------
    def _auto_calibrate(self):
        """Re-fit and replot from the current points. Called on every change, so
        a calibration line is always shown once enough valid points exist."""
        if self.app.spect is None:
            return
        chans, ergs = self._read_points()
        if len(chans) == 0:                    # nothing to fit yet
            self.predicted = None
            self._set_result("—")
            self.page.show_empty()
            return
        # Reject inconsistent points (duplicate peak / repeated energy) typed
        # directly into the table, before they reach the fit.
        err = self._validation_error(chans, ergs)
        if err is not None:
            self._fit_failed(err)
            return
        if self.opts.cb_origin.isChecked() and 0.0 not in chans:
            chans = [0.0] + chans
            ergs = [0.0] + ergs
        # A degree-n polynomial needs at least n+1 points; say so plainly rather
        # than silently dropping the degree or drawing nothing.
        n = self.opts.degree.value()
        if len(chans) < n + 1:
            self._fit_failed(
                f"Need at least {n + 1} points for a degree-{n} fit "
                f"(have {len(chans)}{' incl. origin' if self.opts.cb_origin.isChecked() else ''}).")
            return
        units = self.opts.units.currentText()
        try:
            cal = ec.EnergyCalibration(mean_vals=chans, erg=ergs,
                                       channels=self.app.spect.channels, n=n, e_units=units)
        except Exception as exc:  # noqa: BLE001 — surface fit errors to the user
            self.predicted = None
            self._set_result("—")
            self.page.show_empty(str(exc))
            self._status(f"Calibration failed: {exc}")
            return
        self.predicted = cal.predicted
        self.page.show_polynomial(cal)
        coeffs = list(cal.fit.best_values.values())
        eqn = " + ".join(f"{c:.6E}*ch^{i}" for i, c in enumerate(coeffs))
        self._fill_coeffs(coeffs)              # mirror the fit into the equation boxes
        self._set_result(eqn, cal.rsquared)

    # -- set from equation -----------------------------------------------------
    def _calibrate_equation(self):
        if not self.app._guard():
            return
        ch = self.app.spect.channels.astype(float)
        units = self.opts.units.currentText()
        coeffs, terms = [], []
        for i, ed in enumerate((self.opts.coef_a, self.opts.coef_b,
                                self.opts.coef_c, self.opts.coef_d)):
            txt = ed.text().strip()
            if not txt:
                coeffs.append(0.0)
                continue
            try:
                c = float(txt)
            except ValueError:
                self._status(f"Coefficient '{txt}' is not a number")
                return
            coeffs.append(c)
            terms.append(f"{c:.6E}*ch^{i}")
        if all(c == 0.0 for c in coeffs):
            self._status("Enter at least one non-zero coefficient")
            return
        predicted = sum(c * ch ** i for i, c in enumerate(coeffs))
        self.predicted = predicted
        eqn = " + ".join(terms)
        self.page.show_equation(ch, predicted, eqn, units)
        self._set_result(eqn)
        self._status("Calibration set from equation — review the plot, then Apply")

    # -- apply -----------------------------------------------------------------
    def _apply(self):
        if not self.app._guard():
            return
        if self.predicted is None:
            # The Apply button is disabled in this state, but guard anyway.
            self._status("No valid calibration yet — add enough points or set an equation")
            return
        spect = self.app.spect
        units = self.opts.units.currentText()
        new = sp.Spectrum(
            counts=spect.counts,
            energies=self.predicted,
            e_units=units,
            realtime=spect.realtime,
            livetime=spect.livetime,
            cps=spect.cps,
            acq_date=spect.acq_date,
            energy_cal=self.ecal_eqn,
            description=spect.description,
            label=spect.label,
        )
        app = self.app
        app.spect = new
        app._spect_orig = new.copy()
        app._remove_cal = False
        app._reset_customize_checks()
        app._refresh()
        app._rebuild_spectra_list()
        self._status("Calibration applied to the active spectrum")
