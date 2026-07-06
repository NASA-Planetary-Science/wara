"""Nuclear Database: searchable gamma-ray line libraries + overlay dialog.

Loads the line databases shipped in wara/nuclear-data and lets the user filter
by element/energy, browse the lines, and emit them for overlay on the spectrum.
"""
import re
from importlib.resources import files

import pandas as pd

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QCheckBox, QTableView, QAbstractItemView,
    QColorDialog, QHeaderView, QFrame, QDialogButtonBox,
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, QAbstractTableModel, pyqtSignal

from wara import parse_NIST
from . import theme as T

# Display name → data file. Order matters (combo order).
DATABASES = {
    "Common lab sources":        "Common_lab_sources.csv",
    "Natural radiation":         "Natural_radiation.csv",
    "Delayed activation (IAEA)": "Delayed_activation_IAEA.csv",
    "Neutron capture (CapGam)":  "Capture_CapGam.csv",
    "Neutron capture (IAEA)":    "Capture_IAEA.csv",
    "Inelastic (Baghdad)":       "Inelastic_Baghdad.csv",
    "TALYS 14 MeV":              "Talys-14MeV.csv",
}


def load_database(name):
    fname = DATABASES[name]
    path = str(files("wara").joinpath("nuclear-data", fname))
    df = pd.read_csv(path)
    if "Info" not in df.columns:
        df["Info"] = ""
    return df


def _evaluable(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def filter_lines(df, element_text, energy_text, range_text):
    """Filter a database by element token(s) and an optional energy window."""
    out = df
    if element_text.strip():
        tokens = [t.strip().lower() for t in element_text.split(",") if t.strip()]
        # Isotope column is mass-first (e.g. "137Cs"); normalize to lowercase alnum.
        iso_norm = out["Isotope"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        mask = pd.Series(False, index=out.index)
        for tok in tokens:
            letters = re.sub(r"[^a-z]", "", tok)
            digits = re.sub(r"[^0-9]", "", tok)
            if digits and letters:
                # accept either order: "137cs" (mass-first) or "cs137" (element-first)
                mask |= iso_norm.isin({digits + letters, letters + digits})
            elif letters:
                # element only → all isotopes of that element (symbol at the end)
                mask |= iso_norm.str.contains(rf"{letters}$", regex=True, na=False)
            elif digits:
                mask |= iso_norm.str.contains(digits, na=False)
        out = out[mask]
    if _evaluable(energy_text):
        energy = float(energy_text)
        rng = float(range_text) if _evaluable(range_text) else 1.0
        out = out[(out["Energy (keV)"] > energy - rng) & (out["Energy (keV)"] < energy + rng)]
    return out.reset_index(drop=True)


class _PandasModel(QAbstractTableModel):
    def __init__(self, df):
        super().__init__()
        self._df = df

    def rowCount(self, _parent=None):
        return len(self._df)

    def columnCount(self, _parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and index.isValid():
            val = self._df.iat[index.row(), index.column()]
            if pd.isna(val):
                return ""
            # Format floats with significant figures so floating-point noise
            # (e.g. 6129.88999999999) collapses to a clean value (6129.89),
            # while real decimals and tiny intensities are preserved.
            if isinstance(val, float):   # numpy.float64 subclasses float
                return format(val, ".10g")
            return str(val)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)

    def sort(self, column, order=Qt.AscendingOrder):
        """Sort by a column on header click (pandas handles numeric vs text)."""
        if self._df.empty:
            return
        col = self._df.columns[column]
        self.beginResetModel()
        self._df = self._df.sort_values(
            by=col, ascending=(order == Qt.AscendingOrder),
            kind="mergesort", ignore_index=True)
        self.endResetModel()


def abundance_html(element):
    """Rich-text natural isotopic composition: colored 'isotope : value' pairs."""
    res = parse_NIST.isotopic_abundance(element)
    if isinstance(res, str):           # warning message (plain)
        return res
    parts = []
    for k, v in res.items():
        m = re.match(r"(\d+)([A-Za-z]+)-(\d+)", k)   # '26Fe-54' -> '54Fe'
        name = f"{m.group(3)}{m.group(2)}" if m else k
        parts.append(
            f'<span style="color:{T.ACCENT_CYAN}; font-weight:700">{name}</span>'
            f'<span style="color:{T.TEXT_DIM}"> : </span>'
            f'<span style="color:{T.TEXT_PRIMARY}">{round(v * 100, 3)}%</span>'
        )
    if not parts:
        return "No stable isotopes found"
    return "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join(parts)


class NuclearDatabaseDialog(QDialog):
    """Non-modal dialog: gamma-line database overlay + isotopic abundances."""

    plot_requested = pyqtSignal(list, str)   # (list of (energy, label), color)
    clear_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nuclear Database")
        self.setStyleSheet(T.STYLESHEET)
        self.setMinimumSize(640, 600)
        self.resize(780, 760)
        self._df = pd.DataFrame()
        self._color = T.REF_LINE

        lay = QVBoxLayout(self)

        form = QFormLayout()
        self.db_combo = QComboBox(); self.db_combo.addItems(list(DATABASES.keys()))
        self.db_combo.setCurrentText("TALYS 14 MeV")
        self.db_combo.setToolTip("Choose a gamma-ray line database")
        self.ed_element = QLineEdit("56Fe")
        self.ed_element.setPlaceholderText("e.g. Co60, Cs137  (blank = all)")
        self.ed_element.setToolTip("Filter by element or isotope (Co60, 60Co, Co-60, or Co); blank = all")
        self.ed_energy = QLineEdit(); self.ed_energy.setPlaceholderText("keV (optional)")
        self.ed_energy.setToolTip("Keep only lines near this energy (keV)")
        self.ed_range = QLineEdit(); self.ed_range.setPlaceholderText("± keV (default 1)")
        self.ed_range.setToolTip("Energy window ± keV around the energy above (default 1)")
        form.addRow("Database:", self.db_combo)
        form.addRow("Element / isotope:", self.ed_element)
        form.addRow("Energy:", self.ed_energy)
        form.addRow("Range:", self.ed_range)
        lay.addLayout(form)

        self.btn_search = QPushButton("Search"); self.btn_search.setObjectName("primary_btn")
        self.btn_search.setToolTip("Search the selected database with the filters above")
        self.btn_search.setCursor(Qt.PointingHandCursor)
        self.btn_search.clicked.connect(self._search)
        self.lbl_count = QLabel(""); self.lbl_count.setObjectName("stat_key")
        srow = QHBoxLayout(); srow.addWidget(self.btn_search); srow.addWidget(self.lbl_count); srow.addStretch(1)
        lay.addLayout(srow)

        self.table = QTableView()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setToolTip("Click a column header to sort; select rows to plot only those")
        lay.addWidget(self.table, stretch=1)

        # Satellite-line + label options
        opts = QHBoxLayout()
        self.cb_sep = QCheckBox("Single escape")
        self.cb_sep.setToolTip("Also mark the single-escape peak (E − 511 keV)")
        self.cb_dep = QCheckBox("Double escape")
        self.cb_dep.setToolTip("Also mark the double-escape peak (E − 1022 keV)")
        self.cb_ce = QCheckBox("Compton edge")
        self.cb_ce.setToolTip("Also mark the Compton edge for each line")
        self.cb_labels = QCheckBox("Include labels"); self.cb_labels.setChecked(True)
        self.cb_labels.setToolTip("Draw the isotope and energy text next to each line")
        for cb in (self.cb_sep, self.cb_dep, self.cb_ce, self.cb_labels):
            opts.addWidget(cb)
        opts.addStretch(1)
        lay.addLayout(opts)

        # Color picker + plot/clear
        btns = QHBoxLayout()
        btns.addWidget(QLabel("Line color:"))
        self.btn_color = QPushButton(); self.btn_color.setFixedSize(28, 22)
        self.btn_color.setCursor(Qt.PointingHandCursor)
        self.btn_color.setToolTip("Pick the color for plotted lines")
        self.btn_color.clicked.connect(self._pick_color)
        self._update_swatch()
        btns.addWidget(self.btn_color)
        btns.addSpacing(12)
        self.btn_plot = QPushButton("Plot Lines"); self.btn_plot.setObjectName("primary_btn")
        self.btn_plot.setToolTip("Overlay the selected lines (or up to 50 if none are selected)")
        self.btn_clear = QPushButton("Clear Lines"); self.btn_clear.setObjectName("danger_btn")
        self.btn_clear.setToolTip("Remove all overlaid reference lines from the plot")
        for b in (self.btn_plot, self.btn_clear):
            b.setCursor(Qt.PointingHandCursor)
        self.btn_plot.clicked.connect(lambda: self.plot_requested.emit(self._lines(), self._color))
        self.btn_clear.clicked.connect(self.clear_requested.emit)
        btns.addWidget(self.btn_plot); btns.addWidget(self.btn_clear); btns.addStretch(1)
        lay.addLayout(btns)

        # Compact isotopic-abundance lookup
        sep = QFrame(); sep.setObjectName("separator"); sep.setFrameShape(QFrame.NoFrame)
        lay.addWidget(sep)
        ab = QHBoxLayout()
        self.ed_abund = QLineEdit(); self.ed_abund.setMaximumWidth(90)
        self.ed_abund.setPlaceholderText("Fe")
        self.ed_abund.setToolTip("Element symbol to look up natural isotopic abundances")
        self.btn_abund = QPushButton("Look up"); self.btn_abund.setObjectName("action_btn")
        self.btn_abund.setToolTip("Show natural isotopic abundances (NIST) for the element")
        self.btn_abund.setCursor(Qt.PointingHandCursor)
        self.btn_abund.clicked.connect(self._lookup_abundance)
        self.ed_abund.returnPressed.connect(self._lookup_abundance)
        ab.addWidget(QLabel("Isotopic abundance:"))
        ab.addWidget(self.ed_abund); ab.addWidget(self.btn_abund); ab.addStretch(1)
        lay.addLayout(ab)
        self.lbl_abund = QLabel(""); self.lbl_abund.setObjectName("stat_key")
        self.lbl_abund.setWordWrap(True)
        lay.addWidget(self.lbl_abund)

        self.btn_close = QPushButton("Close"); self.btn_close.setObjectName("action_btn")
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.clicked.connect(self.close)
        crow = QHBoxLayout(); crow.addStretch(1); crow.addWidget(self.btn_close)
        lay.addLayout(crow)

        self._search()

    def _update_swatch(self):
        self.btn_color.setStyleSheet(
            f"background:{self._color}; border:1px solid {T.BORDER_BTN}; border-radius:4px;")

    def _pick_color(self):
        col = QColorDialog.getColor(QColor(self._color), self, "Line color")
        if col.isValid():
            self._color = col.name()
            self._update_swatch()

    def _search(self):
        try:
            df = load_database(self.db_combo.currentText())
            df = filter_lines(df, self.ed_element.text(), self.ed_energy.text(), self.ed_range.text())
        except Exception as exc:  # noqa: BLE001
            self.lbl_count.setText(str(exc))
            return
        self._df = df
        self.table.setModel(_PandasModel(df))
        T.make_headers_readable(self.table)
        self.lbl_count.setText(f"{len(df)} lines  ·  select rows to plot a subset")

    def _lines(self):
        model = self.table.model()
        if model is None or model.rowCount() == 0:
            return []
        df = model._df   # the (possibly sorted) data currently shown
        sel = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        rows = df.iloc[sel] if sel else df
        sep, dep, ce = self.cb_sep.isChecked(), self.cb_dep.isChecked(), self.cb_ce.isChecked()
        labels = self.cb_labels.isChecked()
        lines = []

        def add(energy, text):
            lines.append((energy, text if labels else ""))

        for _, r in rows.iterrows():
            try:
                e = float(r["Energy (keV)"])
            except (TypeError, ValueError):
                continue
            iso = str(r["Isotope"])
            add(e, f"{iso}  {e:.1f}")
            if sep and e > 1022:
                add(e - 511, f"{iso}  {e - 511:.1f} SEP")
            if dep and e > 1022:
                add(e - 1022, f"{iso}  {e - 1022:.1f} DEP")
            if ce:
                ce_e = e - e / (1 + 2 * e / 511)
                add(ce_e, f"{iso}  {ce_e:.1f} CE")
        return lines

    def _lookup_abundance(self):
        element = self.ed_abund.text().strip()
        self.lbl_abund.setTextFormat(Qt.RichText)
        self.lbl_abund.setText(abundance_html(element) if element else "")


# Last line-picker search, remembered across opens so reusing it is faster.
_LAST_LINE_SEARCH = {"database": "Common lab sources", "element": "", "energy": "", "range": ""}


class NuclearLinePicker(QDialog):
    """Compact, modal database picker: search the gamma-line libraries and pick
    one line's energy. Reuses the same load/filter helpers as the full Nuclear
    Database dialog. ``selected_energy()`` returns the chosen energy in keV.

    The last search (database + filters) is remembered across opens so the
    table comes back pre-populated."""

    def __init__(self, parent=None, element=None, multi=False):
        super().__init__(parent)
        self._multi = multi
        self.setWindowTitle("Select energies from database" if multi
                            else "Select energy from database")
        self.setStyleSheet(T.STYLESHEET)
        # Stay above the (always-on-top) Drag-and-Fit window and its child dialog.
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setMinimumSize(680, 640)
        self.resize(740, 720)
        self._df = pd.DataFrame()
        last = _LAST_LINE_SEARCH

        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.db_combo = QComboBox(); self.db_combo.addItems(list(DATABASES.keys()))
        self.db_combo.setCurrentText(last["database"])
        self.db_combo.setToolTip("Gamma-ray line database to search")
        # An explicit element wins; otherwise restore the last-searched isotope.
        self.ed_element = QLineEdit(element if element is not None else last["element"])
        self.ed_element.setPlaceholderText("e.g. Co60, Cs137  (blank = all)")
        self.ed_energy = QLineEdit(last["energy"]); self.ed_energy.setPlaceholderText("keV (optional)")
        self.ed_range = QLineEdit(last["range"]); self.ed_range.setPlaceholderText("± keV (default 1)")
        form.addRow("Database:", self.db_combo)
        form.addRow("Element / isotope:", self.ed_element)
        form.addRow("Energy:", self.ed_energy)
        form.addRow("Range:", self.ed_range)
        lay.addLayout(form)

        self.btn_search = QPushButton("Search"); self.btn_search.setObjectName("primary_btn")
        self.btn_search.setCursor(Qt.PointingHandCursor)
        self.btn_search.clicked.connect(self._search)
        lay.addWidget(self.btn_search)

        self.table = QTableView()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection if multi
                                    else QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setToolTip("Pick a line, then OK (or double-click a row)")
        self.table.doubleClicked.connect(lambda *_: self.accept())
        lay.addWidget(self.table, stretch=1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        # Make Enter run a search (not accept the dialog), so typing an isotope
        # and pressing Enter filters the list rather than closing with nothing.
        for b in (bb.button(QDialogButtonBox.Ok), bb.button(QDialogButtonBox.Cancel)):
            b.setAutoDefault(False)
            b.setDefault(False)
        self.btn_search.setAutoDefault(True)
        self.btn_search.setDefault(True)

        self._search()

    def _search(self):
        try:
            df = load_database(self.db_combo.currentText())
            df = filter_lines(df, self.ed_element.text(),
                              self.ed_energy.text(), self.ed_range.text())
        except Exception:  # noqa: BLE001 — leave the previous results on error
            return
        self._df = df
        self.table.setModel(_PandasModel(df))
        T.make_headers_readable(self.table)
        # Remember this search so the next picker opens pre-populated.
        _LAST_LINE_SEARCH.update(
            database=self.db_combo.currentText(),
            element=self.ed_element.text(),
            energy=self.ed_energy.text(),
            range=self.ed_range.text())

    def _selected_row(self):
        model = self.table.model()
        if model is None or model.rowCount() == 0:
            return None
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return None
        return model._df.iloc[sel[0].row()]

    def selected_energy(self):
        """Chosen line energy in keV, or None if nothing is selected."""
        row = self._selected_row()
        if row is None:
            return None
        try:
            return float(row["Energy (keV)"])
        except (TypeError, ValueError, KeyError):
            return None

    def selected_isotope(self):
        row = self._selected_row()
        return "" if row is None else str(row.get("Isotope", ""))

    def selected_record(self):
        """Full record for the chosen line: isotope plus any of energy /
        intensity / half-life present in the database (missing fields → None).
        Used by the Efficiency tab to auto-fill half-life and branching ratio."""
        row = self._selected_row()
        if row is None:
            return None

        def num(col):
            try:
                val = row[col]
            except (KeyError, TypeError):
                return None
            return float(val) if not pd.isna(val) else None

        return {
            "isotope": str(row.get("Isotope", "") or ""),
            "energy": num("Energy (keV)"),
            "intensity": num("Intensity (%)"),
            "half_life": num("Half life (s)"),
        }

    def selected_energies(self):
        """All chosen line energies in keV (for ``multi=True``). Empty if none."""
        model = self.table.model()
        if model is None or model.rowCount() == 0:
            return []
        out = []
        for i in sorted({idx.row() for idx in self.table.selectionModel().selectedRows()}):
            try:
                out.append(float(model._df.iloc[i]["Energy (keV)"]))
            except (TypeError, ValueError, KeyError):
                continue
        return out

    def selected_lines(self):
        """``(energy_keV, isotope)`` for each chosen row (for ``multi=True``).

        Rows whose energy can't be parsed are skipped; the isotope falls back to
        an empty string when the column is missing."""
        model = self.table.model()
        if model is None or model.rowCount() == 0:
            return []
        out = []
        for i in sorted({idx.row() for idx in self.table.selectionModel().selectedRows()}):
            row = model._df.iloc[i]
            try:
                energy = float(row["Energy (keV)"])
            except (TypeError, ValueError, KeyError):
                continue
            out.append((energy, str(row.get("Isotope", "") or "")))
        return out
