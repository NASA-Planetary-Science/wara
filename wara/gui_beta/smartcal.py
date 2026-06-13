"""Smart Calibration dialog for the beta GUI Calibration tab.

Gathers peak channels (imported from Find Peaks / manual selection, or typed in)
and reference energies (picked from the Nuclear Database, or typed in), then runs
the outlier-tolerant matcher :func:`wara.energy_calibration.smart_calibration_auto`
to pair them automatically. The matched channel↔energy pairs are pushed into the
Calibration tab's points table, where the existing auto-fit / plot / Apply
pipeline takes over.

Energies are handled in keV (the Nuclear Database unit); the points table's units
are set to keV when results are applied.
"""
from PyQt5.QtWidgets import (
    QDialog, QWidget, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QVBoxLayout, QHBoxLayout, QLineEdit, QSpinBox,
)
from PyQt5.QtCore import Qt

from wara import energy_calibration as ec
from . import theme as T


def _editable_item(text):
    it = QListWidgetItem(text)
    it.setFlags(it.flags() | Qt.ItemIsEditable)
    return it


def _panel_btn_style(accent):
    """Mini-button stylesheet tinted with *accent*, so each panel's buttons read
    as belonging together (channels vs energies)."""
    return (
        f"QPushButton {{ background-color:#10101a; color:{accent}; "
        f"border:1px solid {accent}; border-radius:4px; padding:3px 6px; "
        f"font-size:13px; font-weight:600; }}"
        f"QPushButton:hover {{ background-color:{accent}; color:#0a0a0f; }}")


def _panel_header(text, accent):
    h = QLabel(text)
    h.setStyleSheet(f"color:{accent}; font-size:13px; font-weight:800; letter-spacing:1px;")
    return h


class SmartCalibrationDialog(QDialog):
    """Two-column picker (peak channels + reference energies) that auto-matches
    them into a calibration."""

    def __init__(self, app, controller, parent=None):
        super().__init__(parent)
        self.app = app
        self.controller = controller
        self.setWindowTitle("Smart Calibration")
        self.setStyleSheet(T.STYLESHEET)
        self.setMinimumSize(560, 520)

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Collect peak channels and reference energies, then Match & "
            "Calibrate. Spurious peaks and absent lines are left unmatched.")
        intro.setObjectName("stat_key"); intro.setWordWrap(True)
        lay.addWidget(intro)

        cols = QHBoxLayout()
        cols.addWidget(self._build_channels_col(), 1)
        cols.addWidget(self._build_energies_col(), 1)
        lay.addLayout(cols)

        # Controls: degree + tolerance + match.
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Degree:"))
        self.degree = QSpinBox(); self.degree.setRange(1, 3)
        self.degree.setValue(int(controller.opts.degree.value()))
        self.degree.setToolTip("Polynomial degree of the calibration")
        ctl.addWidget(self.degree)
        ctl.addSpacing(12)
        ctl.addWidget(QLabel("Tolerance (keV):"))
        self.tol = QLineEdit(); self.tol.setPlaceholderText("auto")
        self.tol.setMaximumWidth(80)
        self.tol.setToolTip("Energy match tolerance; blank = auto (0.3 × smallest line gap)")
        ctl.addWidget(self.tol)
        ctl.addStretch(1)
        self.btn_match = QPushButton("Match && Calibrate")
        self.btn_match.setObjectName("primary_btn"); self.btn_match.setCursor(Qt.PointingHandCursor)
        self.btn_match.setToolTip("Pair the channels to energies and fill the points table")
        self.btn_match.clicked.connect(self._match)
        ctl.addWidget(self.btn_match)
        lay.addLayout(ctl)

        self.result = QLabel(""); self.result.setObjectName("stat_key")
        self.result.setWordWrap(True); self.result.setTextFormat(Qt.RichText)
        lay.addWidget(self.result)

        crow = QHBoxLayout(); crow.addStretch(1)
        self.btn_close = QPushButton("Close"); self.btn_close.setObjectName("action_btn")
        self.btn_close.setCursor(Qt.PointingHandCursor); self.btn_close.clicked.connect(self.close)
        crow.addWidget(self.btn_close)
        lay.addLayout(crow)

    # ── column builders ───────────────────────────────────────────────────────
    def _build_channels_col(self):
        box = QWidget(); v = QVBoxLayout(box); v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(_panel_header("PEAK CHANNELS", T.ACCENT_CYAN))
        self.lst_ch = QListWidget()
        self.lst_ch.setToolTip("Peak positions in channel number (double-click to edit)")
        v.addWidget(self.lst_ch)
        row = QHBoxLayout()
        self.btn_import = QPushButton("Import peaks")
        self.btn_import.setToolTip("Pull peaks already found / selected on the spectrum")
        self.btn_import.clicked.connect(self._import_peaks)
        b_add = QPushButton("Add")
        b_add.clicked.connect(lambda: self._add_blank(self.lst_ch))
        b_rm = QPushButton("Remove")
        b_rm.clicked.connect(lambda: self._remove_sel(self.lst_ch))
        b_clr = QPushButton("Clear")
        b_clr.clicked.connect(self.lst_ch.clear)
        for b in (self.btn_import, b_add, b_rm, b_clr):
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(_panel_btn_style(T.ACCENT_CYAN))   # channels → cyan
            row.addWidget(b)
        v.addLayout(row)
        return box

    def _build_energies_col(self):
        box = QWidget(); v = QVBoxLayout(box); v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(_panel_header("REFERENCE ENERGIES (keV)", T.ACCENT_GREEN))
        self.lst_en = QListWidget()
        self.lst_en.setToolTip("Known line energies in keV (double-click to edit)")
        v.addWidget(self.lst_en)
        row = QHBoxLayout()
        self.btn_db = QPushButton("From database…")
        self.btn_db.setToolTip("Pick one or more lines from the Nuclear Database")
        self.btn_db.clicked.connect(self._from_database)
        b_add = QPushButton("Add")
        b_add.clicked.connect(lambda: self._add_blank(self.lst_en))
        b_rm = QPushButton("Remove")
        b_rm.clicked.connect(lambda: self._remove_sel(self.lst_en))
        b_clr = QPushButton("Clear")
        b_clr.clicked.connect(self.lst_en.clear)
        for b in (self.btn_db, b_add, b_rm, b_clr):
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(_panel_btn_style(T.ACCENT_GREEN))   # energies → green
            row.addWidget(b)
        v.addLayout(row)
        return box

    # ── list helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _add_blank(lst):
        it = _editable_item("")
        lst.addItem(it); lst.setCurrentItem(it); lst.editItem(it)

    @staticmethod
    def _remove_sel(lst):
        for it in lst.selectedItems():
            lst.takeItem(lst.row(it))

    @staticmethod
    def _values(lst):
        """Numeric values currently in a list, skipping blanks/non-numbers."""
        out = []
        for i in range(lst.count()):
            txt = lst.item(i).text().strip()
            if not txt:
                continue
            try:
                out.append(float(txt))
            except ValueError:
                continue
        return out

    def _add_values(self, lst, values):
        for v in values:
            lst.addItem(_editable_item(f"{v:.10g}"))

    # ── actions ───────────────────────────────────────────────────────────────
    def _import_peaks(self):
        s = self.app.search
        spect = self.app.spect
        if s is None or getattr(s, "peaks_idx", None) is None or len(s.peaks_idx) == 0:
            self._info("No peaks on the spectrum yet — run Find Peaks (or add channels manually).")
            return
        chans = [float(spect.channels[i]) for i in s.peaks_idx]
        existing = set(round(c, 3) for c in self._values(self.lst_ch))
        added = [c for c in chans if round(c, 3) not in existing]
        self._add_values(self.lst_ch, added)
        self._info(f"Imported {len(added)} peak channel(s).")

    def _from_database(self):
        from .nuclear import NuclearLinePicker
        picker = NuclearLinePicker(self, multi=True)
        if picker.exec_() != QDialog.Accepted:
            return
        energies = picker.selected_energies()
        if not energies:
            self._info("No lines selected.")
            return
        self._add_values(self.lst_en, energies)
        self._info(f"Added {len(energies)} energy line(s).")

    def _match(self):
        if self.app.spect is None:
            self._error("Load a spectrum first.")
            return
        channels = self._values(self.lst_ch)
        energies = self._values(self.lst_en)
        n = self.degree.value()
        if len(channels) < n + 1 or len(energies) < n + 1:
            self._error(f"Need at least {n + 1} channels and {n + 1} energies for a degree-{n} fit.")
            return
        tol = None
        if self.tol.text().strip():
            try:
                tol = float(self.tol.text())
            except ValueError:
                self._error("Tolerance must be a number (or blank for auto).")
                return
        nch = len(self.app.spect.channels)
        try:
            res = ec.smart_calibration_auto(
                channels, energies, n=n, tol=tol, channel_range=(0, nch))
        except ValueError as exc:
            self._error(str(exc))
            return
        self.controller.set_smart_results(res["pairs"], units="keV")
        self._show_result(res)

    # ── result / status display ───────────────────────────────────────────────
    def _show_result(self, res):
        um_ch = ", ".join(f"{c:g}" for c in res["unmatched_channels"]) or "none"
        um_en = ", ".join(f"{e:g}" for e in res["unmatched_energies"]) or "none"
        self.result.setText(
            f"<span style='color:{T.ACCENT_GREEN}'>Matched "
            f"{res['n_matched']} pairs &nbsp;·&nbsp; RMSE {res['rmse']:.3g} keV "
            f"&nbsp;·&nbsp; R² {res['r2']:.6f}</span><br>"
            f"<span style='color:{T.TEXT_DIM}'>Unmatched peaks: {um_ch}<br>"
            f"Unmatched energies: {um_en}</span>")

    def _info(self, msg):
        self.result.setText(f"<span style='color:{T.TEXT_DIM}'>{msg}</span>")

    def _error(self, msg):
        self.result.setText(f"<span style='color:{T.ACCENT_RED}'>{msg}</span>")
