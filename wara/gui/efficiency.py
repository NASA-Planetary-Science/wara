"""Efficiency tab: build a detector photopeak-efficiency curve.

Workflow (wara GUI):

* A peak fit is pushed in from the Drag-and-Fit window
  (``EfficiencyController.receive_fit``). Every peak in the fit becomes one
  editable row in the points table; the energy and net area come from the fit.
* The user fills in the source parameters per row — half-life, initial activity,
  branching ratio, counting time, time elapsed. The input cells are
  **unit-aware** and accept an optional uncertainty in parentheses: values like
  ``5 µCi (0.1)``, ``30 y (2)`` or ``600 s`` are parsed and converted to standard
  units (Bq, s); the uncertainty in ``(…)`` inherits the value's unit unless it
  states its own. Hovering a cell shows the original text that was typed.
* A 🔍 button on each row opens the Nuclear Database picker; choosing a line
  auto-fills the isotope, its half-life and the branching ratio (line intensity).
* As soon as a row holds every required value its efficiency is computed and the
  point is plotted (efficiency in percent vs energy). The accumulated points can
  be fitted with a linear polynomial (order 1) or the log-polynomial efficiency
  curve (order 2), and the y-axis toggled log/linear.

Bug fixes carried over from the original port: ``calculate_error`` is called with
``livetime_sig`` / ``t_elapsed_sig`` in the right order, and the plotted error
bars are scaled to percent to match the points.
"""
import re

import numpy as np

from matplotlib.figure import Figure
from matplotlib.colors import to_rgb
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView, QCheckBox,
    QDialog, QSizePolicy, QApplication, QStyledItemDelegate,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor

from wara import efficiency as eff
from wara.matplotlib_theme import set_theme, DARK

from . import theme as T
from .widgets import hsep, header, ComboBox

# Match the Spectrum/Calibration tabs' plot background (pure black).
EFF_PLOT_BG = T.BG_PLOT

# Fit-model combo entries.
FIT_NONE, FIT_LINEAR, FIT_CURVE = "None", "Order 1 (linear)", "Order 2 (log-poly)"

# Per-cell stored data: the value and uncertainty in standard units.
VALUE_ROLE = Qt.UserRole
UNC_ROLE = Qt.UserRole + 1

# ── Unit recognition ──────────────────────────────────────────────────────────
# Time units → seconds. Years use 365 d to match wara.efficiency's convention.
_TIME_UNITS = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0, "m": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "d": 86400.0, "day": 86400.0, "days": 86400.0,
    "w": 604800.0, "wk": 604800.0, "week": 604800.0, "weeks": 604800.0,
    "y": 365 * 86400.0, "yr": 365 * 86400.0, "yrs": 365 * 86400.0,
    "year": 365 * 86400.0, "years": 365 * 86400.0,
}
# Activity units → becquerel.
_ACTIVITY_UNITS = {
    "bq": 1.0, "kbq": 1e3, "mbq": 1e6, "gbq": 1e9,
    "ci": 3.7e10, "mci": 3.7e7, "uci": 3.7e4, "µci": 3.7e4, "nci": 3.7e1,
    "pci": 3.7e-2,
}

_NUM_RE = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Zµ%]*)\s*$")
# A trailing "(…)" uncertainty, e.g. "30 y (2)" or "5 µCi (0.1 µCi)".
_PAREN_RE = re.compile(r"^(.*?)\(([^)]*)\)\s*$")

# Column categories used for parsing / standard-unit labels.
CAT_TIME, CAT_ACTIVITY, CAT_RATIO, CAT_NUMBER = "time", "activity", "ratio", "number"


def _to_standard(num, unit, category):
    """Convert a (number, unit-token) pair to standard units for *category*.
    Returns ``(value, std_label)``; raises ``ValueError`` on an unknown unit."""
    unit = unit.lower()
    if category == CAT_NUMBER:
        if unit:
            raise ValueError(f"'{unit}' — enter a plain number here")
        return num, ""
    if category == CAT_TIME:
        if not unit:
            return num, "s"
        if unit in _TIME_UNITS:
            return num * _TIME_UNITS[unit], "s"
        raise ValueError(f"Unknown time unit '{unit}'")
    if category == CAT_ACTIVITY:
        if not unit:
            return num, "Bq"
        if unit in _ACTIVITY_UNITS:
            return num * _ACTIVITY_UNITS[unit], "Bq"
        raise ValueError(f"Unknown activity unit '{unit}'")
    # ratio
    if not unit:
        return num, ""
    if unit == "%":
        return num / 100.0, ""
    raise ValueError(f"'{unit}' is not valid for a branching ratio")


def parse_quantity(text, category):
    """Parse a user-entered quantity (with an optional ``(uncertainty)``) into
    standard units.

    Returns ``(value, uncertainty, std_label)`` (e.g. ``(1800.0, 60.0, "s")``),
    or ``None`` for a blank entry. The uncertainty defaults to 0 and, when given
    as a bare number in ``(…)``, inherits the value's unit. Raises ``ValueError``
    on an unparseable value or unknown unit. A bare number is assumed to already
    be in standard units; ``%`` is accepted for branching ratios."""
    if text is None:
        return None
    t = text.strip()
    if not t:
        return None
    m = _PAREN_RE.match(t)
    if m:
        main, unc_text = m.group(1).strip(), m.group(2).strip()
    else:
        main, unc_text = t, ""

    mv = _NUM_RE.match(main)
    if not mv:
        raise ValueError(f"Cannot read a number from '{text}'")
    value, label = _to_standard(float(mv.group(1)), mv.group(2), category)

    unc = 0.0
    if unc_text:
        mu = _NUM_RE.match(unc_text)
        if not mu:
            raise ValueError(f"Cannot read the uncertainty in '{text}'")
        u_unit = mu.group(2) or mv.group(2)        # bare ± inherits the value's unit
        unc, _ = _to_standard(float(mu.group(1)), u_unit, category)
    return value, unc, label


# A "YYYY-MM-DD to YYYY-MM-DD" elapsed-time range (separator: "to", "→" or "->").
_DATE_RANGE_RE = re.compile(
    r"^\s*(\d{4}-\d{1,2}-\d{1,2})\s*(?:to|→|->)\s*(\d{4}-\d{1,2}-\d{1,2})\s*$", re.I)


def parse_elapsed(text):
    """Parse the Elapsed-time cell. Accepts a ``"YYYY-MM-DD to YYYY-MM-DD"`` date
    range (converted to seconds via ``efficiency.calculate_t_elapsed``) as well
    as a unit-bearing duration like ``"16 y"`` (with an optional ``(±)``).
    Returns ``(value, uncertainty, "s")`` or ``None`` for a blank cell."""
    if text is None:
        return None
    t = text.strip()
    if not t:
        return None
    m = _DATE_RANGE_RE.match(t)
    if m:
        try:
            secs = float(eff.calculate_t_elapsed(m.group(1), m.group(2)))
        except ValueError:
            raise ValueError("Dates must be valid calendar dates (YYYY-MM-DD)")
        if secs < 0:
            raise ValueError("The end date is before the initial date")
        return secs, 0.0, "s"
    return parse_quantity(t, CAT_TIME)


def _format_cell(value, unc, label):
    """Render a normalised cell: ``"<value> <label> (<unc>)"`` (the ``(…)`` part
    is omitted when the uncertainty is zero)."""
    base = f"{value:.6g} {label}".strip()
    return f"{base} ({unc:.6g})" if unc else base


def _is_dark(color):
    """True for near-black colours (luminance < 0.25) — the backend plot helpers
    draw black markers/lines that vanish on the dark GUI background."""
    try:
        r, g, b = to_rgb(color)
    except (ValueError, TypeError):
        return False
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.25


class _PeakData:
    """Minimal stand-in for a peak-fit object exposing just the centroid, net
    area and area error the efficiency backend reads. Lets the (now editable)
    Energy / Net-counts cells drive the calculation instead of the fit object,
    so user overrides take effect."""

    def __init__(self, mean, area, area_err):
        self.peak_info = [{"mean": mean, "area": area, "fwhm": 0.0}]
        self.peak_err = [{"area_err": area_err, "fwhm_err": 0.0}]


# ── Column layout ─────────────────────────────────────────────────────────────
# Column 0 is a per-row 🗑 delete button, column 1 the 🔍 nuclide lookup.
(COL_TRASH, COL_LOOKUP, COL_ISOTOPE, COL_ENERGY, COL_COUNTS, COL_THALF, COL_A0,
 COL_BR, COL_LT, COL_TE, COL_ANOW, COL_EFF) = range(12)

HEADERS = ["", "", "Isotope", "Energy", "Net counts", "Half-life", "Activity A₀",
           "Branching", "Count time", "Elapsed", "Activity Aₜ", "Efficiency (%)"]

# Unit category for each parseable column. Energy and Net counts are plain
# numbers (seeded from the fit, but editable); the rest carry units.
COL_CATEGORY = {
    COL_ENERGY: CAT_NUMBER, COL_COUNTS: CAT_NUMBER,
    COL_THALF: CAT_TIME, COL_A0: CAT_ACTIVITY, COL_BR: CAT_RATIO,
    COL_LT: CAT_TIME, COL_TE: CAT_TIME,
}

# Placeholder hints shown (dim) in empty editable cells. The "(±)" reminds the
# user that an uncertainty can be appended in parentheses.
COL_PLACEHOLDER = {
    COL_ISOTOPE: "🔍 or type",
    COL_ENERGY: "keV",
    COL_COUNTS: "counts (±)",
    COL_THALF: "e.g. 30 y (2)",
    COL_A0: "e.g. 5 µCi (0.1)",
    COL_BR: "e.g. 0.85 (0.01)",
    COL_LT: "e.g. 600 s (5)",
    COL_TE: "16 y  or  2006-06-01 to 2022-08-19",
}

# Columns the user types into (Energy / Net counts are editable too).
EDITABLE_COLS = {COL_ISOTOPE} | set(COL_CATEGORY)
# Required values a row needs before it can be computed.
REQUIRED_COLS = (COL_ENERGY, COL_COUNTS, COL_THALF, COL_A0, COL_BR, COL_LT, COL_TE)


class _PlaceholderDelegate(QStyledItemDelegate):
    """Paints a dim placeholder hint in empty cells and sets the same hint on the
    cell editor, so the points table reads like a form to fill in."""

    def __init__(self, placeholders, parent=None):
        super().__init__(parent)
        self._ph = placeholders

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        ph = self._ph.get(index.column())
        if ph and hasattr(editor, "setPlaceholderText"):
            editor.setPlaceholderText(ph)
        return editor

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if index.data(Qt.DisplayRole):
            return
        ph = self._ph.get(index.column())
        if not ph:
            return
        painter.save()
        painter.setPen(QColor(T.TEXT_DIM))
        painter.drawText(option.rect.adjusted(6, 0, -6, 0),
                         Qt.AlignVCenter | Qt.AlignLeft, ph)
        painter.restore()


# ── Plot + points-input table column ──────────────────────────────────────────
class EfficiencyPage(QWidget):
    """Plot area for the Efficiency tab: efficiency-vs-energy curve (with an
    optional residual panel) above the editable points table."""

    def __init__(self):
        super().__init__()
        self.setObjectName("content")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(8, 6), constrained_layout=True, facecolor=EFF_PLOT_BG)
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
            f"background:{EFF_PLOT_BG}; font-size:15px; font-weight:700; padding:3px 8px;")
        lay.addWidget(self.readout, 0)

        # Editable efficiency points (one row per pushed peak). Columns stretch to
        # fill the full plot width (no dead space to the right of the table).
        self.table = QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMaximumHeight(220)
        self.table.setItemDelegate(_PlaceholderDelegate(COL_PLACEHOLDER, self.table))
        # Size columns to fit their titles (e.g. "Activity A₀", "Net counts");
        # the last column stretches to fill the rest, the lookup stays narrow.
        T.make_headers_readable(self.table)   # stretches the last (Efficiency) col
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_TRASH, QHeaderView.Fixed)
        hh.setSectionResizeMode(COL_LOOKUP, QHeaderView.Fixed)
        self.table.setColumnWidth(COL_TRASH, 32)
        self.table.setColumnWidth(COL_LOOKUP, 34)
        lay.addWidget(self.table, 1)

        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("axes_leave_event", lambda *_: self.readout.setText(""))

        self.show_empty()

    # -- drawing ---------------------------------------------------------------
    def show_empty(self, msg="Send a peak fit, then fill in the source parameters"):
        self.fig.clf()
        self.readout.setText("")
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color=T.BORDER, fontweight="bold", wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        self._restyle()
        self.canvas.draw_idle()
        self._reset_nav()

    def show_points(self, x, y, yerr, e_units, log_y, log_x=False, order=None):
        """Plot the efficiency points; when *order* is 1 or 2 overlay the fit and
        its residual panel."""
        self.fig.clf()
        self.readout.setText("")
        set_theme(DARK)
        if order in (1, 2):
            gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 4])
            ax_res = self.fig.add_subplot(gs[0, 0])
            ax_fit = self.fig.add_subplot(gs[1, 0])
        else:
            ax_res = None
            ax_fit = self.fig.add_subplot(111)

        eff.plot_points(e_vals=x, eff_vals=y, err_vals=yerr, e_units=e_units, ax=ax_fit)
        if order in (1, 2):
            eff.eff_fit(x, y, yerr, order=order, ax_fit=ax_fit, ax_res=ax_res)
            ax_fit.set_xlabel(f"Energy ({e_units})")
            ax_fit.set_ylabel("Efficiency (%)")
        if log_y:
            ax_fit.set_yscale("log")
        if log_x:
            ax_fit.set_xscale("log")
            if ax_res is not None:
                ax_res.set_xscale("log")
        # The backend helpers draw black markers / zero-lines; lift them off the
        # dark background.
        for ax in self.fig.axes:
            self._fix_dark_artists(ax)
        self._restyle()
        self.canvas.draw_idle()
        self._reset_nav()

    @staticmethod
    def _fix_dark_artists(ax):
        """Recolour the backend's black data markers (→ cyan) and any black
        lines / zero-rules (→ dim) so they are visible on the dark theme."""
        for line in ax.lines:
            if line.get_marker() not in ("", " ", "None", None):
                if _is_dark(line.get_markerfacecolor()):
                    line.set_markerfacecolor(T.ACCENT_CYAN)
                if _is_dark(line.get_markeredgecolor()):
                    line.set_markeredgecolor(T.ACCENT_CYAN)
            elif _is_dark(line.get_color()):
                line.set_color(T.TEXT_DIM)
        for coll in ax.collections:        # hlines (residual zero-rule) etc.
            try:
                colors = coll.get_colors()
            except AttributeError:
                continue
            if len(colors) and all(_is_dark(c) for c in colors):
                coll.set_color(T.TEXT_DIM)
        # Rebuild the legend so its swatch shows the recoloured marker.
        if ax.get_legend() is not None:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=10)

    def _reset_nav(self):
        self.toolbar.update()
        self.toolbar.push_current()

    def _restyle(self):
        """Force the backend's own theme onto the dark GUI palette."""
        self.fig.patch.set_alpha(1.0)
        self.fig.set_facecolor(EFF_PLOT_BG)
        for ax in self.fig.axes:
            ax.set_facecolor(EFF_PLOT_BG)
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
class EfficiencyOptions(QScrollArea):
    """Scrollable options for the Efficiency tab. Widgets are exposed as
    attributes; the EfficiencyController wires the behaviour."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(9, 12, 9, 12); lay.setSpacing(6)

        title = QLabel("EFFICIENCY"); title.setObjectName("opt_title")
        lay.addWidget(title); lay.addWidget(hsep())

        lay.addWidget(header("PEAK FIT"))
        self.lbl_fit = QLabel(
            "Send a peak fit from the Drag-and-Fit window. Each peak becomes a "
            "row to fill in.")
        self.lbl_fit.setObjectName("stat_key"); self.lbl_fit.setWordWrap(True)
        lay.addWidget(self.lbl_fit)
        hint = QLabel(
            "Tip: cells accept units and an uncertainty in parentheses — "
            "“5 µCi (0.1)”, “30 y (2)”, “600 s”. Elapsed time also takes a date "
            "range, e.g. “2006-06-01 to 2022-08-19”. Use 🔍 to pull a nuclide's "
            "half-life and branching ratio from the database.")
        hint.setObjectName("stat_key"); hint.setWordWrap(True)
        lay.addWidget(hint)

        # ── Points management ────────────────────────────────────────
        lay.addWidget(hsep()); lay.addWidget(header("POINTS"))
        phint = QLabel("Click 🗑 next to a row to remove it.")
        phint.setObjectName("stat_key"); phint.setWordWrap(True)
        lay.addWidget(phint)
        self.btn_reset = QPushButton("Reset"); self.btn_reset.setObjectName("danger_btn")
        self.btn_reset.setToolTip("Clear every efficiency point")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_reset)

        # ── Fit model + y-scale ──────────────────────────────────────
        lay.addWidget(hsep()); lay.addWidget(header("FIT"))
        frow = QHBoxLayout()
        self.fit_model = ComboBox()
        self.fit_model.addItems([FIT_NONE, FIT_LINEAR, FIT_CURVE])
        self.fit_model.setToolTip(
            "Order 1: weighted linear polynomial (needs ≥ 2 points).\n"
            "Order 2: log-polynomial efficiency curve "
            "eff = (a0 + a1·lnE + a2·lnE² + a3·lnE³)/E (needs ≥ 4 points).")
        frow.addWidget(QLabel("Model:")); frow.addStretch(1); frow.addWidget(self.fit_model)
        lay.addLayout(frow)

        self.cb_logy = QCheckBox("Log Y")
        self.cb_logy.setChecked(True)
        self.cb_logy.setToolTip("Plot efficiency on a logarithmic y-axis")
        lay.addWidget(self.cb_logy)

        self.cb_logx = QCheckBox("Log X")
        self.cb_logx.setToolTip("Plot energy on a logarithmic x-axis (log-log with Log Y)")
        lay.addWidget(self.cb_logx)

        lay.addStretch(1)
        self.setWidget(inner)


# ── Controller ────────────────────────────────────────────────────────────────
class EfficiencyController:
    """Wires an EfficiencyOptions panel and EfficiencyPage to the main app."""

    def __init__(self, app, options: EfficiencyOptions, page: EfficiencyPage):
        self.app = app
        self.opts = options
        self.page = page
        self.tbl = page.table
        self._records = []          # one dict per table row: fit/which_peak/energy/point
        self._e_units = "keV"       # energy units of the points (from the spectrum)
        self._loading = False       # suppress cell-changed handling during edits
        self._wire()

    def _wire(self):
        o = self.opts
        o.btn_reset.clicked.connect(self._reset)
        o.fit_model.currentTextChanged.connect(lambda *_: self._replot())
        o.cb_logy.toggled.connect(lambda *_: self._replot())
        o.cb_logx.toggled.connect(lambda *_: self._replot())
        self.tbl.cellChanged.connect(self._on_cell_changed)

    def _status(self, msg):
        self.app.statusBar().showMessage(f"  {msg}")

    def _current_units(self):
        spect = self.app.spect
        if spect is not None and getattr(spect, "e_units", None):
            return spect.e_units
        return "keV"

    # -- receiving a fit -------------------------------------------------------
    def receive_fit(self, fit):
        """Accept a peak fit pushed from the Drag-and-Fit window: add one
        editable row per peak and switch to the Efficiency tab."""
        if fit is None or not getattr(fit, "peak_info", None):
            self._status("That fit has no peaks to use for efficiency")
            return
        self._e_units = self._current_units()
        n = len(fit.peak_info)
        for wp in range(n):
            self._append_row(fit, wp)
        self.opts.lbl_fit.setText(
            f"Added {n} peak{'s' if n != 1 else ''}. Fill in each row's source "
            "parameters (cells accept units and a (±) uncertainty; 🔍 pulls from "
            "the database).")
        # Stay on the current tab; the user opens Efficiency when ready.
        self._replot()
        self._status(f"Added {n} peak(s) to the Efficiency tab — open it to enter the source parameters")

    # -- table construction ----------------------------------------------------
    @staticmethod
    def _peak_value(fit, attr, which_peak, key, default=0.0):
        try:
            return float(getattr(fit, attr)[which_peak][key])
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return default

    def _append_row(self, fit, which_peak):
        energy = self._peak_value(fit, "peak_info", which_peak, "mean")
        area = self._peak_value(fit, "peak_info", which_peak, "area")
        area_err = self._peak_value(fit, "peak_err", which_peak, "area_err")
        rec = {"fit": fit, "which_peak": which_peak, "energy": energy,
               "a_now": None, "point": None}
        self._records.append(rec)

        self._loading = True
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        for col in range(len(HEADERS)):
            item = QTableWidgetItem("")
            if col in EDITABLE_COLS:
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            else:
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                item.setTextAlignment(Qt.AlignCenter)
            self.tbl.setItem(r, col, item)
        # Seed Energy and Net counts from the fit (both stay editable; the user
        # may override either). Store the standard value/uncertainty directly so
        # the seed matches what the parser would produce on a manual edit.
        self._seed_cell(r, COL_ENERGY, energy, 0.0)
        self._seed_cell(r, COL_COUNTS, area, area_err)
        # 🗑 delete + 🔍 lookup buttons (capture the record so they survive row
        # removal/reordering).
        trash = QPushButton("🗑"); trash.setObjectName("remove_btn")
        trash.setToolTip("Remove this efficiency point")
        trash.setCursor(Qt.PointingHandCursor); trash.setFixedSize(26, 22)
        trash.clicked.connect(lambda _=False, rc=rec: self._delete_record(rc))
        self.tbl.setCellWidget(r, COL_TRASH, trash)
        btn = QPushButton("🔍"); btn.setObjectName("mini_btn")
        btn.setToolTip("Pick a nuclide from the database to fill half-life & branching ratio")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda _=False, rc=rec: self._open_lookup(rc))
        self.tbl.setCellWidget(r, COL_LOOKUP, btn)
        self._loading = False

    def _seed_cell(self, row, col, value, unc):
        """Write a value+uncertainty into a cell as if the parser produced it."""
        it = self.tbl.item(row, col)
        it.setData(VALUE_ROLE, value)
        it.setData(UNC_ROLE, unc)
        it.setText(_format_cell(value, unc, ""))

    # -- cell editing / unit parsing -------------------------------------------
    def _on_cell_changed(self, row, col):
        if self._loading:
            return
        if col in COL_CATEGORY:
            self._normalize_cell(row, col)
        self._recompute_row(row)
        self._replot()

    def _normalize_cell(self, row, col):
        """Parse the cell, convert value + uncertainty to standard units, store
        them on the item, and show the converted text with the original kept as a
        tooltip. An unparseable value is flagged but left as typed."""
        item = self.tbl.item(row, col)
        if item is None:
            return
        raw = item.text()
        try:
            # The Elapsed column also accepts a "date to date" range.
            parsed = (parse_elapsed(raw) if col == COL_TE
                      else parse_quantity(raw, COL_CATEGORY[col]))
        except ValueError as exc:
            item.setData(VALUE_ROLE, None)
            item.setData(UNC_ROLE, None)
            item.setToolTip(f"⚠ {exc}")
            return
        self._loading = True
        if parsed is None:
            item.setData(VALUE_ROLE, None)
            item.setData(UNC_ROLE, None)
            item.setText("")
            item.setToolTip("")
        else:
            value, unc, label = parsed
            item.setData(VALUE_ROLE, value)
            item.setData(UNC_ROLE, unc)
            item.setText(_format_cell(value, unc, label))
            item.setToolTip(f"entered: {raw.strip()}")
        self._loading = False

    def _cell_value(self, row, col):
        item = self.tbl.item(row, col)
        return None if item is None else item.data(VALUE_ROLE)

    def _cell_unc(self, row, col):
        item = self.tbl.item(row, col)
        u = None if item is None else item.data(UNC_ROLE)
        return 0.0 if u is None else u

    def _recompute_row(self, row):
        """Compute one row's efficiency when every required value is present;
        otherwise clear its result cell. The centroid and net area come from the
        (editable) Energy / Net-counts cells."""
        if row < 0 or row >= len(self._records):
            return
        rec = self._records[row]
        vals = {c: self._cell_value(row, c) for c in REQUIRED_COLS}
        if any(vals[c] is None for c in REQUIRED_COLS) or not vals[COL_THALF] > 0:
            self._clear_computed(row)
            rec["a_now"] = rec["point"] = None
            return
        energy = vals[COL_ENERGY]
        peak = _PeakData(energy, vals[COL_COUNTS], self._cell_unc(row, COL_COUNTS))
        try:
            obj = eff.Efficiency(
                t_half=vals[COL_THALF], A0=vals[COL_A0], Br=vals[COL_BR],
                livetime=vals[COL_LT], t_elapsed=vals[COL_TE], which_peak=0)
            obj.calculate_efficiency(peak)
            # Activity at the time of counting (with the line's branching ratio):
            # A_now = N_emitted / livetime  (see efficiency.calculate_N_emitted).
            a_now = float(obj.calculate_N_emitted()) / vals[COL_LT]
            # Keyword args keep livetime_sig / t_elapsed_sig in the right slots.
            obj.calculate_error(
                peak,
                t_half_sig=self._cell_unc(row, COL_THALF),
                A0_sig=self._cell_unc(row, COL_A0),
                Br_sig=self._cell_unc(row, COL_BR),
                livetime_sig=self._cell_unc(row, COL_LT),
                t_elapsed_sig=self._cell_unc(row, COL_TE))
        except Exception as exc:  # noqa: BLE001 — surface compute errors per row
            self._clear_computed(row)
            rec["a_now"] = rec["point"] = None
            self._status(f"Row {row + 1}: {exc}")
            return
        eff_pct, err_pct = obj.eff * 100.0, obj.error * 100.0
        self._set_computed(row, a_now, eff_pct, err_pct)
        rec["a_now"] = a_now
        rec["point"] = (energy, eff_pct, err_pct)

    def _set_computed(self, row, a_now, eff_pct, err_pct):
        """Fill the locked Activity Aₜ and Efficiency result cells."""
        self._loading = True
        self.tbl.item(row, COL_ANOW).setText(f"{a_now:.4g} Bq")
        self.tbl.item(row, COL_EFF).setText(f"{eff_pct:.4g} ({err_pct:.2g})")
        self._loading = False

    def _clear_computed(self, row):
        self._loading = True
        self.tbl.item(row, COL_ANOW).setText("")
        self.tbl.item(row, COL_EFF).setText("")
        self._loading = False

    # -- nuclide lookup --------------------------------------------------------
    def _open_lookup(self, rec):
        from .nuclear import NuclearLinePicker   # lazy: avoids import cycle
        row = self._records.index(rec)
        dlg = NuclearLinePicker(QApplication.activeWindow() or self.app)
        # Aim the search at the row's energy in a database that carries
        # half-life and intensity.
        energy = self._cell_value(row, COL_ENERGY)
        if energy is None:
            energy = rec["energy"]
        dlg.db_combo.setCurrentText("Common lab sources")
        dlg.ed_element.setText("")
        dlg.ed_energy.setText(f"{energy:.2f}")
        dlg.ed_range.setText("3")
        dlg._search()
        if dlg.exec_() != QDialog.Accepted:
            return
        record = dlg.selected_record()
        if record is None:
            self._status("No database line selected")
            return
        self._apply_lookup(row, record)

    def _apply_lookup(self, row, record):
        self._loading = True
        iso = record.get("isotope", "")
        if iso:
            self.tbl.item(row, COL_ISOTOPE).setText(iso)
        hl = record.get("half_life")
        if hl is not None and hl > 0:
            it = self.tbl.item(row, COL_THALF)
            it.setData(VALUE_ROLE, hl); it.setData(UNC_ROLE, 0.0)
            it.setText(_format_cell(hl, 0.0, "s"))
            it.setToolTip(f"from database: {iso} half-life")
        inten = record.get("intensity")
        if inten is not None:
            br = inten / 100.0
            it = self.tbl.item(row, COL_BR)
            it.setData(VALUE_ROLE, br); it.setData(UNC_ROLE, 0.0)
            it.setText(_format_cell(br, 0.0, ""))
            it.setToolTip(f"from database: {iso} intensity {inten:g}%")
        self._loading = False
        self._recompute_row(row)
        self._replot()
        missing = [c for c in (COL_A0, COL_LT, COL_TE) if self._cell_value(row, c) is None]
        if missing:
            self._status(f"Filled {iso or 'nuclide'} — still need activity / counting / elapsed time")
        else:
            self._status(f"Filled {iso} from the database")

    # -- points management -----------------------------------------------------
    def _delete_record(self, rec):
        """Remove a single efficiency point (its 🗑 button was clicked). Records
        and table rows stay aligned, so the remaining 🗑/🔍 buttons still resolve
        to the right row via ``_records.index``."""
        if rec not in self._records:
            return
        row = self._records.index(rec)
        self._loading = True
        self.tbl.removeRow(row)
        self._loading = False
        del self._records[row]
        self._replot()
        self._status("Removed efficiency point")

    def _reset(self):
        self._loading = True
        self.tbl.setRowCount(0)
        self._loading = False
        self._records = []
        self.opts.fit_model.blockSignals(True)
        self.opts.fit_model.setCurrentText(FIT_NONE)
        self.opts.fit_model.blockSignals(False)
        self.page.show_empty()
        self._status("Efficiency points reset")

    # -- plotting --------------------------------------------------------------
    def _selected_order(self):
        choice = self.opts.fit_model.currentText()
        if choice == FIT_LINEAR:
            return 1
        if choice == FIT_CURVE:
            return 2
        return None

    def _points(self):
        """(energy, eff%, ±eff%) arrays for every completed row, sorted by
        energy so the fit/line is drawn left-to-right."""
        pts = sorted((rec["point"] for rec in self._records if rec["point"] is not None),
                     key=lambda p: p[0])
        if not pts:
            return None
        arr = np.array(pts, dtype=float)
        return arr[:, 0], arr[:, 1], arr[:, 2]

    def _replot(self):
        pts = self._points()
        if pts is None:
            self.page.show_empty()
            return
        x, y, yerr = pts
        order = self._selected_order()
        if order == 1 and len(x) < 2:
            self._status("Need at least 2 points for an order-1 fit")
            order = None
        elif order == 2 and len(x) < 4:
            self._status("Need at least 4 points for an order-2 fit")
            order = None
        log_y = self.opts.cb_logy.isChecked()
        log_x = self.opts.cb_logx.isChecked()
        try:
            self.page.show_points(x, y, yerr, self._e_units,
                                  log_y=log_y, log_x=log_x, order=order)
        except Exception as exc:  # noqa: BLE001 — a failed fit shouldn't kill the GUI
            self._status(f"Efficiency fit failed: {exc}")
            self.page.show_points(x, y, yerr, self._e_units,
                                  log_y=log_y, log_x=log_x, order=None)
