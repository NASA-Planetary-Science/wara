"""Small pop-up dialogs for the API tab: manual filters, energy-selection
labeling, the 3D-view parameters, apply-shifts-to-data confirmation, and the
per-run statistics table. The heavyweight dialogs live in their own modules
(``api_shifts``, ``api_combine``, ``api_diagnostics``, ``api_selections``).
"""
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QColorDialog, QAbstractItemView, QComboBox,
    QCheckBox,
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

    Port of the legacy ``WindowAPI3D`` / ``create_plot_api3D``, upgraded to the
    full ``apicalc.api_xyz`` reconstruction. The controller fills
    :attr:`browser` (a ``QWebEngineView``) with a Plotly ``Volume`` figure built
    from the *current* (filtered) event dataframe. The right-hand controls set
    the histogram resolution / isosurface look plus the reconstruction
    geometry: the gamma-detector position relative to the neutron source (used
    by the default "Full (detector)" model), the source-YAP distance, the beam
    energy, and the depth calibration. "Simple (no detector)" is the previous
    behaviour (straight-ray, detector ignored), kept as the backup model."""

    # (label, attribute, default) for the numeric render controls.
    FIELDS = [
        ("No. of bins", "no_bins", "50"),
        ("Iso min", "isomin", "0.1"),
        ("Iso max", "isomax", "0.8"),
        ("Opacity", "opacity", "0.1"),
        ("Surface count", "surfcount", "20"),
    ]

    # (label, attribute, default) for the reconstruction-geometry controls.
    # Frame: origin at the neutron source, +z up toward the YAP, sample at -z.
    RECON_FIELDS = [
        ("Det X [cm]", "det_x", "-18"),
        ("Det Y [cm]", "det_y", "0"),
        ("Det Z [cm]", "det_z", "-25"),
        ("Src-YAP [cm]", "z_t", "6.7"),
        ("Beam [keV]", "beam_kev", "50"),
        ("Sample Z [cm]", "sample_z", ""),
        ("t offset [ns]", "toffset", ""),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API 3D volume")
        self.setStyleSheet(T.STYLESHEET)
        # Maximise/minimise help when orbiting the volume.
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        # Tall enough for the render + reconstruction controls and the status
        # messages under them.
        self.resize(1040, 800)

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

        side.addWidget(header("RECONSTRUCTION"))
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["Full (detector)", "Simple (no detector)"])
        self.cmb_model.setToolTip(
            "Full: apicalc.api_xyz with the gamma-detector position (the "
            "gamma flight time from the interaction point to the detector is "
            "solved for). Simple: the previous straight-ray reconstruction "
            "that ignores the detector (backup).")
        mrow, _ = labeled_row("Model", self.cmb_model)
        side.addWidget(mrow)
        for label, attr, default in self.RECON_FIELDS:
            ed = QLineEdit(default); ed.setFixedWidth(70)
            row, _ = labeled_row(label, ed)
            side.addWidget(row)
            self.fields[attr] = ed
        self.fields["det_x"].setToolTip(
            "Gamma-detector centre relative to the neutron source [cm]; "
            "+z points up toward the YAP, so the sample side is -z.")
        self.fields["det_y"].setToolTip(self.fields["det_x"].toolTip())
        self.fields["det_z"].setToolTip(self.fields["det_x"].toolTip())
        self.fields["z_t"].setToolTip("Neutron source to YAP-face distance [cm]")
        self.fields["beam_kev"].setToolTip(
            "Deuteron beam energy [keV] for the center-of-mass correction")
        self.fields["sample_z"].setToolTip(
            "Optional: depth [cm] of the sample FRONT FACE. When set (and "
            "t offset is blank), the timing offset is auto-fitted so the "
            "rising edge of the dominant dt peak lands at this depth. Leave "
            "blank if the sample position is unknown.")
        self.fields["toffset"].setToolTip(
            "Timing offset [ns] subtracted from dt. A value overrides "
            "everything; blank with a Sample Z auto-fits the offset; both "
            "blank uses dt as-is (assumes the dt spectrum is already "
            "calibrated so 0 ns = production time).")

        self.cb_r2 = QCheckBox("1/r² correction")
        self.cb_r2.setToolTip(
            "Weight each event by (r/r_mean)^p, where r is the distance from "
            "the reconstructed point to the gamma detector -- undoes the "
            "higher count density on the detector side of the target. p = 2 "
            "is a point-like 1/r² response; a nearby/large detector behaves "
            "softer (smaller p). Full (detector) model only.")
        side.addWidget(self.cb_r2)
        ed = QLineEdit("2"); ed.setFixedWidth(70)
        prow, _ = labeled_row("p (2 = 1/r²)", ed)
        side.addWidget(prow)
        self.fields["r2_power"] = ed
        ed.setToolTip(self.cb_r2.toolTip())

        self.btn_plot = QPushButton("Plot"); self.btn_plot.setObjectName("open_btn")
        self.btn_plot.setCursor(Qt.PointingHandCursor)
        # Pressing Enter in any of the option fields re-plots.
        self.btn_plot.setAutoDefault(True)
        self.btn_plot.setDefault(True)
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

    def recon_params(self):
        """The reconstruction inputs as a dict, or None if a field is invalid.

        Keys: ``det_pos`` (x, y, z) [cm], ``z_t`` [cm], ``beam_kev``,
        ``sample_z`` [cm] (float, or None = unknown), ``toffset_ns`` (float,
        or None = not given), ``use_det`` (bool -- False selects the backup
        straight-ray model). The timing calibration follows from the two
        optional fields: a ``toffset_ns`` value wins; else a ``sample_z``
        auto-fits the offset; both None means dt is used as-is (already
        calibrated). ``r2_correction`` (bool) and ``r2_power`` (float) drive
        the optional detector solid-angle weighting."""
        vals = {}
        for attr in ("det_x", "det_y", "det_z", "z_t", "beam_kev", "r2_power"):
            v = self.value(attr, float)
            if v is None:
                return None
            vals[attr] = v
        # optional fields: blank -> None, anything else must be a number
        for attr in ("sample_z", "toffset"):
            txt = self.fields[attr].text().strip()
            if not txt:
                vals[attr] = None
            else:
                try:
                    vals[attr] = float(txt)
                except ValueError:
                    return None
        return {
            "det_pos": (vals["det_x"], vals["det_y"], vals["det_z"]),
            "z_t": vals["z_t"],
            "beam_kev": vals["beam_kev"],
            "sample_z": vals["sample_z"],
            "toffset_ns": vals["toffset"],
            "use_det": self.cmb_model.currentIndex() == 0,
            "r2_correction": self.cb_r2.isChecked(),
            "r2_power": vals["r2_power"],
        }


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


def _count_pct(g, col, total):
    """'n (p%)' for the number of True rows of *g[col]* (percent of len(g))."""
    if col not in g.columns or len(g) == 0:
        return "--"
    n = int(g[col].astype(bool).sum())
    return f"{n:,} ({100.0 * n / len(g):.2f}%)"


def _span_seconds(g):
    """Acquisition span (s) from the timestamp column (ns), or None."""
    if "timestamp" not in g.columns or len(g) == 0:
        return None
    span = float(g["timestamp"].max() - g["timestamp"].min())
    return span / 1e9 if span > 0 else None


class BinaryStatsDialog(QDialog):
    """Rich per-channel statistics for the loaded list-mode data on the Binary
    tab (trace / binary / parquet), computed directly from the dataframe.

    Unlike :class:`StatsInfoDialog` (which reads the run's ``-stats-`` JSON), this
    summarises the events actually loaded: counts and rates, pile-up / CFD-error
    / trace-flag fractions, and energy / baseline / timing summaries per channel,
    with a totals row.
    """

    # (header, tooltip, per-channel formatter fn(group, total) -> str,
    #  required-column or None). Columns whose required column is absent from the
    #  dataframe are dropped, so the table adapts to trace/binary/parquet data.
    COLUMNS = [
        ("Events", "Number of events",
         lambda g, tot: f"{len(g):,}", None),
        ("% total", "Share of all loaded events",
         lambda g, tot: f"{100.0 * len(g) / tot:.2f}%" if tot else "--", None),
        ("Duration (s)", "Timestamp span of these events",
         lambda g, tot: (f"{_span_seconds(g):,.3f}"
                         if _span_seconds(g) is not None else "--"), "timestamp"),
        ("Rate (cps)", "Events / timestamp span",
         lambda g, tot: (f"{len(g) / _span_seconds(g):,.1f}"
                         if _span_seconds(g) else "--"), "timestamp"),
        ("Pile-up", "Events flagged as piled-up",
         lambda g, tot: _count_pct(g, "pileup", tot), "pileup"),
        ("CFD errors", "Events with a CFD error (forced trigger)",
         lambda g, tot: _count_pct(g, "CFD_error", tot), "CFD_error"),
        ("Trace flags", "Events with the trace out-of-range flag set",
         lambda g, tot: _count_pct(g, "trace_flag", tot), "trace_flag"),
        ("Traces", "Events carrying a non-empty trace",
         lambda g, tot: (f"{int(g['trace'].map(lambda t: t is not None and len(t) > 0).sum()):,}"
                         if "trace" in g.columns else "--"), "trace"),
        ("Aligned", "Events successfully time-aligned",
         lambda g, tot: (f"{int(g['align_shift'].notna().sum()):,}"
                         if "align_shift" in g.columns else "--"), "align_shift"),
        ("E mean", "Mean energy",
         lambda g, tot: f"{g['energy'].mean():,.1f}" if len(g) else "--", "energy"),
        ("E median", "Median energy",
         lambda g, tot: f"{g['energy'].median():,.0f}" if len(g) else "--", "energy"),
        ("E min", "Minimum energy",
         lambda g, tot: f"{g['energy'].min():,.0f}" if len(g) else "--", "energy"),
        ("E max", "Maximum energy",
         lambda g, tot: f"{g['energy'].max():,.0f}" if len(g) else "--", "energy"),
        ("Baseline", "Mean baseline",
         lambda g, tot: f"{g['baseline'].mean():,.1f}" if len(g) else "--", "baseline"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("List-mode data statistics")
        self.setStyleSheet(T.STYLESHEET)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.resize(960, 460)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(6)
        lay.addWidget(header("LIST-MODE DATA STATISTICS"))
        self.lbl_meta = QLabel(""); self.lbl_meta.setObjectName("stat_key")
        self.lbl_meta.setWordWrap(True)
        lay.addWidget(self.lbl_meta)

        self.table = QTableWidget(0, 0)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        lay.addWidget(self.table, 1)

    def populate(self, df, run_label, kind, cfd=None, align=None):
        total = int(len(df))
        # Only keep columns whose backing dataframe column exists.
        cols = [c for c in self.COLUMNS if c[3] is None or c[3] in df.columns]

        # ── metadata summary ──────────────────────────────────────────────────
        parts = [run_label, f"source: {kind}"]
        if cfd:
            parts.append(f"CFD: {cfd}")
        if align:
            parts.append(f"align: {align}")
        parts.append(f"{total:,} events")
        chans = (sorted(int(c) for c in df["channel"].unique())
                 if "channel" in df.columns else [])
        if chans:
            parts.append(f"channels: {', '.join(map(str, chans))}")
        span = _span_seconds(df)
        if span is not None:
            parts.append(f"duration: {span:,.2f} s")
            parts.append(f"rate: {total / span:,.1f} cps")
        meta = "  ·  ".join(parts)
        meta += "\ncolumns: " + ", ".join(df.columns)
        self.lbl_meta.setText(meta)

        # ── per-channel table (+ totals row) ──────────────────────────────────
        self.table.setColumnCount(1 + len(cols))
        self.table.setHorizontalHeaderLabels(["Channel"] + [h for h, *_ in cols])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c, (_h, tip, _fn, _need) in enumerate(cols, start=1):
            self.table.horizontalHeaderItem(c).setToolTip(tip)

        rows = [(str(ch), df[df["channel"] == ch]) for ch in chans]
        if len(rows) != 1:               # skip a redundant Total when single-channel
            rows.append(("Total", df))
        self.table.setRowCount(len(rows))
        for r, (label, g) in enumerate(rows):
            ch_item = QTableWidgetItem(label)
            ch_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 0, ch_item)
            for c, (_h, _tip, fn, _need) in enumerate(cols, start=1):
                try:
                    text = fn(g, total)
                except Exception:  # noqa: BLE001
                    text = "--"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, item)
