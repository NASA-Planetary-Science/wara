"""Small pop-up dialogs for the API tab: manual filters, energy-selection
labeling, the 3D-view parameters, apply-shifts-to-data confirmation, and the
per-run statistics table. The heavyweight dialogs live in their own modules
(``api_shifts``, ``api_combine``, ``api_diagnostics``, ``api_selections``).
"""
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QColorDialog, QAbstractItemView,
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

from . import theme as T
from .widgets import header, labeled_row
from .api_common import API_PLOT_BG


class ApiFilterDialog(QDialog):
    """Manual min/max entry for the X-Y, time and energy filters  -- the typed
    equivalent of dragging a selector on the plot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API filters")
        self.setStyleSheet(T.STYLESHEET)
        self.setMinimumWidth(320)
        lay = QVBoxLayout(self)
        lay.addWidget(header("MANUAL FILTERS"))
        lay.addWidget(QLabel("Leave a pair blank to skip that filter."))

        grid = QGridLayout()
        grid.addWidget(QLabel("min"), 0, 1)
        grid.addWidget(QLabel("max"), 0, 2)
        self.fields = {}
        for r, (key, label) in enumerate(
                [("x", "X"), ("y", "Y"), ("t", "dt"), ("e", "Energy")], start=1):
            grid.addWidget(QLabel(label), r, 0)
            lo, hi = QLineEdit(), QLineEdit()
            for e in (lo, hi):
                e.setFixedWidth(90)
            grid.addWidget(lo, r, 1)
            grid.addWidget(hi, r, 2)
            self.fields[key] = (lo, hi)
        lay.addLayout(grid)

        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_apply = QPushButton("Apply"); self.btn_apply.setObjectName("open_btn")
        self.btn_apply.setCursor(Qt.PointingHandCursor)
        row.addWidget(self.btn_apply)
        lay.addLayout(row)

    def pair(self, key):
        """Return (min, max) floats for *key*, or None if either box is empty."""
        lo, hi = self.fields[key]
        lo_t, hi_t = lo.text().strip(), hi.text().strip()
        if lo_t == "" or hi_t == "":
            return None
        try:
            return float(lo_t), float(hi_t)
        except ValueError:
            return None


class EnergySelectionDialog(QDialog):
    """Label + colour entry for a new energy selection, shown after the user
    drags a band on the energy spectrum with 'Add selection' armed."""

    def __init__(self, emin, emax, color, n_events, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label energy selection")
        self.setStyleSheet(T.STYLESHEET)
        self.setMinimumWidth(300)
        self._color = color

        lay = QVBoxLayout(self)
        lay.addWidget(header("ENERGY SELECTION"))
        info = QLabel(f"Range  {emin:g} – {emax:g}\n{n_events:,} events")
        info.setObjectName("stat_key")
        lay.addWidget(info)

        self.ed_label = QLineEdit()
        self.ed_label.setPlaceholderText("e.g. Fe, Si, 511 keV")
        self.ed_label.setFixedWidth(150)
        row, _ = labeled_row("Label", self.ed_label)
        lay.addWidget(row)

        self.btn_color = QPushButton("Pick colour")
        self.btn_color.setCursor(Qt.PointingHandCursor)
        self.btn_color.clicked.connect(self._pick_color)
        crow, _ = labeled_row("Colour", self.btn_color)
        lay.addWidget(crow)
        self._apply_swatch()

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _pick_color(self):
        col = QColorDialog.getColor(QColor(self._color), self, "Selection colour")
        if col.isValid():
            self._color = col.name()
            self._apply_swatch()

    def _apply_swatch(self):
        # Paint the button in the chosen colour so it doubles as a live swatch.
        self.btn_color.setStyleSheet(
            f"background-color:{self._color}; color:{T.BG_DARK}; "
            f"border:2px solid {self._color}; border-radius:5px; "
            f"padding:6px 12px; font-weight:800;")

    def label(self):
        return self.ed_label.text().strip()

    def color(self):
        return self._color


class Api3DDialog(QDialog):
    """Pop-out Plotly volume render of the reconstructed X-Y-Z hit cloud.

    Port of the legacy ``WindowAPI3D`` / ``create_plot_api3D``. The controller
    fills :attr:`browser` (a ``QWebEngineView``) with a Plotly ``Volume`` figure
    built from the *current* (filtered) event dataframe; the controls here only
    set the histogram resolution and isosurface look. Like the legacy window the
    gamma-detector position is ignored (``api_xyz(use_det=False)``), so no
    detector-position fields are exposed."""

    # (label, attribute, default) for the numeric controls.
    FIELDS = [
        ("No. of bins", "no_bins", "50"),
        ("Iso min", "isomin", "0.1"),
        ("Iso max", "isomax", "0.8"),
        ("Opacity", "opacity", "0.1"),
        ("Surface count", "surfcount", "20"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API 3D volume")
        self.setStyleSheet(T.STYLESHEET)
        # Maximise/minimise help when orbiting the volume.
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.resize(1040, 700)

        # QtWebEngine is imported lazily (as the legacy GUI does via its .ui) so
        # the offscreen test suite never has to load it. AA_ShareOpenGLContexts
        # is already set in app.main() before the QApplication is created.
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        from PyQt5.QtGui import QColor

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(8)

        self.browser = QWebEngineView()
        self.browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.browser.setMinimumWidth(560)
        # The QWebEngineView renders white by default  -- paint the page background
        # dark so the blank initial page, the load flash, and any margin around
        # the Plotly figure all match the dark theme instead of flashing white.
        self.browser.page().setBackgroundColor(QColor(API_PLOT_BG))
        self.browser.setHtml(
            f"<html><body style='margin:0;background:{API_PLOT_BG}'></body></html>")
        lay.addWidget(self.browser, 1)

        side = QVBoxLayout(); side.setSpacing(6)
        side.addWidget(header("3D VOLUME"))
        note = QLabel("Volume render of the X-Y-Z hit cloud reconstructed from "
                      "the current (filtered) events.")
        note.setObjectName("stat_key"); note.setWordWrap(True)
        side.addWidget(note)

        self.fields = {}
        for label, attr, default in self.FIELDS:
            ed = QLineEdit(default); ed.setFixedWidth(70)
            row, _ = labeled_row(label, ed)
            side.addWidget(row)
            self.fields[attr] = ed

        self.btn_plot = QPushButton("Plot"); self.btn_plot.setObjectName("open_btn")
        self.btn_plot.setCursor(Qt.PointingHandCursor)
        side.addWidget(self.btn_plot)

        self.status = QLabel(""); self.status.setObjectName("stat_key")
        self.status.setWordWrap(True)
        side.addWidget(self.status)
        side.addStretch(1)

        holder = QWidget(); holder.setFixedWidth(230); holder.setLayout(side)
        lay.addWidget(holder, 0)

    def value(self, attr, cast):
        """Read a control as *cast* (int/float), or None if it isn't valid."""
        try:
            return cast(self.fields[attr].text().strip())
        except (ValueError, KeyError):
            return None


class ApplyToDataDialog(QDialog):
    """Ask for the destination run when baking the active calibration/time-shift
    into a new on-disk run. The date defaults to the loaded run's date (editable)
    and the run number must be entered by the user  -- writing to a *new* run keeps
    the source run untouched and avoids the loader concatenating duplicates."""

    def __init__(self, date, runnr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Apply to data  -- save calibrated run")
        self.setStyleSheet(T.STYLESHEET)
        self.setMinimumWidth(340)

        lay = QVBoxLayout(self)
        lay.addWidget(header("SAVE CALIBRATED RUN"))
        note = QLabel(
            "Combine all of the source run's parquet files and bake the active "
            "energy calibration and/or time shift into energy_cal / dt_cal, then "
            "save as a new run. Pick a new run number so the original run is left "
            "untouched.")
        note.setObjectName("stat_key"); note.setWordWrap(True)
        lay.addWidget(note)

        self.ed_date = QLineEdit(str(date or "")); self.ed_date.setFixedWidth(150)
        drow, _ = labeled_row("Date", self.ed_date)
        lay.addWidget(drow)
        self.ed_run = QLineEdit(); self.ed_run.setFixedWidth(150)
        self.ed_run.setPlaceholderText("new run number")
        rrow, _ = labeled_row("Run", self.ed_run)
        lay.addWidget(rrow)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def values(self):
        """Return (date_str, runnr_int). runnr is None if it isn't an integer."""
        date = self.ed_date.text().strip()
        try:
            runnr = int(self.ed_run.text().strip())
        except ValueError:
            runnr = None
        return date, runnr


class StatsInfoDialog(QDialog):
    """Per-channel MCA run statistics from the run's latest ``-stats-`` file,
    shown as a read-only table (one row per channel)."""

    # (stats dict key, column header, value formatter).
    COLUMNS = [
        ("real_time", "Real time (s)", lambda v: f"{v:,.3f}"),
        ("live_time", "Live time (s)", lambda v: f"{v:,.3f}"),
        ("input_counts", "Input counts", lambda v: f"{v:,.0f}"),
        ("input_count_rate", "Input CR (cps)", lambda v: f"{v:,.1f}"),
        ("output_counts", "Output counts", lambda v: f"{v:,.0f}"),
        ("output_count_rate", "Output CR (cps)", lambda v: f"{v:,.1f}"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MCA run statistics")
        self.setStyleSheet(T.STYLESHEET)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.resize(720, 480)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(6)
        lay.addWidget(header("MCA RUN STATISTICS"))
        self.lbl_meta = QLabel(""); self.lbl_meta.setObjectName("stat_key")
        self.lbl_meta.setWordWrap(True)
        lay.addWidget(self.lbl_meta)

        self.table = QTableWidget(0, 1 + len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(
            ["Channel"] + [h for _, h, _ in self.COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self.table, 1)

    def populate(self, stats, run_label, fname):
        module = stats.get("module", "?")
        self.lbl_meta.setText(
            f"{run_label}  ·  module {module}  ·  source: {fname}")
        # Channel count = length of the first per-channel list present.
        n = 0
        for key, _, _ in self.COLUMNS:
            val = stats.get(key)
            if isinstance(val, list):
                n = max(n, len(val))
        self.table.setRowCount(n)
        for r in range(n):
            ch_item = QTableWidgetItem(str(r))
            ch_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 0, ch_item)
            for c, (key, _, fmt) in enumerate(self.COLUMNS, start=1):
                val = stats.get(key)
                if isinstance(val, list) and r < len(val):
                    try:
                        text = fmt(val[r])
                    except Exception:  # noqa: BLE001
                        text = str(val[r])
                else:
                    text = "--"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, item)
