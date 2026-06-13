"""Smart Calibration dialog for the beta GUI Calibration tab.

Gathers peak channels (imported from Find Peaks / manual selection, or typed in)
and reference energies (picked from the Nuclear Database, or typed in), then runs
the outlier-tolerant matcher :func:`wara.energy_calibration.smart_calibration_auto`
to pair them automatically. The matched channel↔energy pairs are pushed into the
Calibration tab's points table, where the existing auto-fit / plot / Apply
pipeline takes over.

Energies are handled in keV (the Nuclear Database unit); the points table's units
are set to keV when results are applied.

Persistent **Favorite energies** are stored in ``~/.wara/favorite_energies.json``
as one or more *named lists* (e.g. "API"), each a collection of reference
energies with an isotope attached to every energy (sourced from the Nuclear
Database or typed in). A list is chosen from a drop-down and its energies pushed
into the energies column with a couple of clicks. See
:class:`FavoriteEnergiesDialog`.
"""
import json
import os

from PyQt5.QtWidgets import (
    QDialog, QWidget, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QVBoxLayout, QHBoxLayout, QLineEdit, QSpinBox,
    QComboBox, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QHeaderView, QInputDialog, QMessageBox,
)
from PyQt5.QtCore import Qt

from wara import energy_calibration as ec
from . import theme as T


# Where favorite energies persist between sessions. Auto-created on first save;
# tests monkeypatch this module-level value to a temp file. The on-disk schema
# (version 2) is {"version": 2, "lists": {name: [{"energy": float,
# "isotope": str}, ...]}}; the old flat version-1 file is migrated on read.
FAV_PATH = os.path.join(os.path.expanduser("~"), ".wara", "favorite_energies.json")


def _clean_entries(entries, tol=1e-6):
    """Return entries sorted by energy with near-duplicate energies dropped.

    Each output entry is a ``{"energy": float, "isotope": str}`` dict."""
    cleaned, seen = [], []
    for ent in sorted(entries, key=lambda e: e["energy"]):
        e = ent["energy"]
        if seen and abs(e - seen[-1]) <= tol:
            continue
        seen.append(e)
        cleaned.append({"energy": float(e), "isotope": str(ent.get("isotope", ""))})
    return cleaned


def _coerce_entries(raw):
    """Normalise raw JSON entries (dicts or bare numbers) into clean entries."""
    out = []
    for item in raw if isinstance(raw, list) else []:
        try:
            if isinstance(item, dict):
                e = float(item["energy"])
                iso = str(item.get("isotope", "") or "")
            else:                       # bare number (v1 / hand-edited file)
                e = float(item)
                iso = ""
        except (TypeError, ValueError, KeyError):
            continue
        out.append({"energy": e, "isotope": iso})
    return _clean_entries(out)


def load_favorite_lists():
    """Saved favorite lists as ``{name: [entry, ...]}``; ``{}`` if none/unreadable.

    Transparently migrates the old flat version-1 file (a single ``energies``
    array) into a list named "Favorites"."""
    try:
        with open(FAV_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if isinstance(data, dict) and isinstance(data.get("lists"), dict):
        return {str(name): _coerce_entries(entries)
                for name, entries in data["lists"].items()}
    # version-1 fallback: {"energies": [...]} or a bare list.
    raw = data.get("energies", []) if isinstance(data, dict) else data
    entries = _coerce_entries(raw)
    return {"Favorites": entries} if entries else {}


def save_favorite_lists(lists):
    """Persist ``{name: entries}`` to :data:`FAV_PATH` (entries cleaned per list).

    Returns the cleaned dict. Raises ``OSError`` if the file can't be written."""
    clean = {str(name): _coerce_entries(entries) for name, entries in lists.items()}
    os.makedirs(os.path.dirname(FAV_PATH), exist_ok=True)
    with open(FAV_PATH, "w", encoding="utf-8") as fh:
        json.dump({"version": 2, "lists": clean}, fh, indent=2)
    return clean


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
        # Sourcing row: pull energies from the Nuclear Database or your favorites.
        srow = QHBoxLayout()
        self.btn_db = QPushButton("From database…")
        self.btn_db.setToolTip("Pick one or more lines from the Nuclear Database")
        self.btn_db.clicked.connect(self._from_database)
        self.btn_fav = QPushButton("Favorites…")
        self.btn_fav.setToolTip("Add from — and manage — your saved favorite energies")
        self.btn_fav.clicked.connect(self._from_favorites)
        for b in (self.btn_db, self.btn_fav):
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(_panel_btn_style(T.ACCENT_GREEN))   # energies → green
            srow.addWidget(b)
        v.addLayout(srow)
        # Edit row: manual add / remove / clear.
        row = QHBoxLayout()
        b_add = QPushButton("Add")
        b_add.clicked.connect(lambda: self._add_blank(self.lst_en))
        b_rm = QPushButton("Remove")
        b_rm.clicked.connect(lambda: self._remove_sel(self.lst_en))
        b_clr = QPushButton("Clear")
        b_clr.clicked.connect(self.lst_en.clear)
        for b in (b_add, b_rm, b_clr):
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

    def _from_favorites(self):
        dlg = FavoriteEnergiesDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        energies = dlg.selected_energies()
        if not energies:
            self._info("No favorite energies to add.")
            return
        existing = set(round(e, 6) for e in self._values(self.lst_en))
        added = [e for e in energies if round(e, 6) not in existing]
        self._add_values(self.lst_en, added)
        self._info(f"Added {len(added)} favorite energy(ies).")

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


EN_COL, ISO_COL = 0, 1


class FavoriteEnergiesDialog(QDialog):
    """Manage named lists of favorite reference energies and pick from them.

    Each named list (e.g. "API") holds energies with an isotope attached to
    every one — sourced from the Nuclear Database or typed in. Lists are chosen
    from a drop-down; the entries table is editable in place and every add /
    edit / remove is saved immediately to :data:`FAV_PATH` (so they survive
    across sessions). :meth:`selected_energies` returns the highlighted energies
    — or *all* of the current list when nothing is selected.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading = False
        self._lists = load_favorite_lists()      # {name: [{"energy","isotope"}]}
        self.setWindowTitle("Favorite energies")
        self.setStyleSheet(T.STYLESHEET)
        # Stay above the always-on-top Drag-and-Fit window and its child dialogs.
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setMinimumSize(460, 540)

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Reusable named lists of reference energies, each tagged with an "
            "isotope. Pick a list, then Add to drop its energies into the column.")
        intro.setObjectName("stat_key"); intro.setWordWrap(True)
        lay.addWidget(intro)

        # List selector + management.
        lrow = QHBoxLayout()
        lrow.addWidget(QLabel("List:"))
        self.cmb = QComboBox()
        self.cmb.setToolTip("Choose a saved list of favorite energies")
        self.cmb.currentTextChanged.connect(self._on_list_changed)
        lrow.addWidget(self.cmb, 1)
        b_new = QPushButton("New")
        b_new.setToolTip("Create a new, empty list")
        b_new.clicked.connect(self._new_list)
        b_ren = QPushButton("Rename")
        b_ren.setToolTip("Rename the current list")
        b_ren.clicked.connect(self._rename_list)
        b_del = QPushButton("Delete")
        b_del.setToolTip("Delete the current list")
        b_del.clicked.connect(self._delete_list)
        for b in (b_new, b_ren, b_del):
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(_panel_btn_style(T.ACCENT_GREEN))
            lrow.addWidget(b)
        lay.addLayout(lrow)

        # Entries table: Energy (keV) | Isotope.
        self.tbl = QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(["Energy (keV)", "Isotope"])
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.setToolTip("Double-click a cell to edit. Energies are in keV.")
        self.tbl.cellChanged.connect(self._on_cell_changed)
        lay.addWidget(self.tbl, 1)

        # Entry management.
        erow = QHBoxLayout()
        b_add = QPushButton("Add row")
        b_add.setToolTip("Add a blank row to type an energy + isotope")
        b_add.clicked.connect(self._add_row_blank)
        b_db = QPushButton("From database…")
        b_db.setToolTip("Add energies + isotopes picked from the Nuclear Database")
        b_db.clicked.connect(self._add_from_database)
        b_rm = QPushButton("Remove")
        b_rm.setToolTip("Delete the selected rows")
        b_rm.clicked.connect(self._remove_rows)
        for b in (b_add, b_db, b_rm):
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(_panel_btn_style(T.ACCENT_GREEN))
            erow.addWidget(b)
        lay.addLayout(erow)

        self.status = QLabel(""); self.status.setObjectName("stat_key")
        self.status.setWordWrap(True); self.status.setTextFormat(Qt.RichText)
        lay.addWidget(self.status)

        # Accept (add to the column) / close.
        brow = QHBoxLayout(); brow.addStretch(1)
        btn_close = QPushButton("Close"); btn_close.setObjectName("action_btn")
        btn_close.setCursor(Qt.PointingHandCursor); btn_close.clicked.connect(self.reject)
        self.btn_add = QPushButton("Add to energies"); self.btn_add.setObjectName("primary_btn")
        self.btn_add.setCursor(Qt.PointingHandCursor)
        self.btn_add.setToolTip("Add the selected energies (or the whole list) to the energies column")
        self.btn_add.clicked.connect(self.accept)
        brow.addWidget(btn_close); brow.addWidget(self.btn_add)
        lay.addLayout(brow)

        self._refresh_combo()

    # ── list (combo) management ─────────────────────────────────────────────────
    def _refresh_combo(self, select=None):
        """Rebuild the list drop-down and show the selected (or first) list."""
        self._loading = True
        self.cmb.clear()
        names = sorted(self._lists, key=str.lower)
        self.cmb.addItems(names)
        if select and select in self._lists:
            self.cmb.setCurrentText(select)
        elif names:
            self.cmb.setCurrentIndex(0)
        self._loading = False
        self._load_table(self._current_list_name())

    def _current_list_name(self):
        return self.cmb.currentText().strip()

    def _on_list_changed(self, name):
        if not self._loading:
            self._load_table(name.strip())

    def _new_list(self):
        name, ok = QInputDialog.getText(self, "New list", "Name for the new list:")
        if not ok:
            return
        name = name.strip()
        if not name:
            self._warn("Enter a name for the list.")
            return
        if name in self._lists:
            self._warn(f"A list named '{name}' already exists.")
            return
        self._lists[name] = []
        self._save()
        self._refresh_combo(select=name)
        self._note(f"Created list '{name}'")

    def _rename_list(self):
        old = self._current_list_name()
        if not old:
            return
        name, ok = QInputDialog.getText(self, "Rename list", "New name:", text=old)
        if not ok:
            return
        name = name.strip()
        if not name or name == old:
            return
        if name in self._lists:
            self._warn(f"A list named '{name}' already exists.")
            return
        self._lists[name] = self._lists.pop(old)
        self._save()
        self._refresh_combo(select=name)

    def _delete_list(self):
        name = self._current_list_name()
        if not name:
            return
        if not self._confirm(f"Delete the list '{name}'? This cannot be undone."):
            return
        self._lists.pop(name, None)
        self._save()
        self._refresh_combo()
        self._note(f"Deleted list '{name}'")

    def _ensure_list(self):
        """Guarantee a current list exists to add rows into; create one if not."""
        if self._current_list_name():
            return True
        name, i = "Favorites", 2
        while name in self._lists:
            name = f"Favorites {i}"; i += 1
        self._lists[name] = []
        self._save()
        self._refresh_combo(select=name)
        return True

    # ── entries table ───────────────────────────────────────────────────────────
    def _load_table(self, name):
        self._loading = True
        self.tbl.setRowCount(0)
        for ent in self._lists.get(name, []):
            self._append_row(ent["energy"], ent["isotope"])
        self._loading = False

    def _append_row(self, energy="", isotope=""):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        e_txt = f"{energy:.10g}" if isinstance(energy, (int, float)) else str(energy)
        self.tbl.setItem(r, EN_COL, QTableWidgetItem(e_txt))
        self.tbl.setItem(r, ISO_COL, QTableWidgetItem(str(isotope)))
        return r

    def _read_table(self):
        """Entries currently in the table, skipping rows without a numeric energy."""
        out = []
        for r in range(self.tbl.rowCount()):
            e_item = self.tbl.item(r, EN_COL)
            if e_item is None or not e_item.text().strip():
                continue
            try:
                e = float(e_item.text().strip())
            except ValueError:
                continue
            iso_item = self.tbl.item(r, ISO_COL)
            out.append({"energy": e, "isotope": iso_item.text().strip() if iso_item else ""})
        return out

    def _on_cell_changed(self, _r, _c):
        if not self._loading:
            self._persist_current()

    def _add_row_blank(self):
        self._ensure_list()
        self._loading = True          # blank rows aren't persisted until typed in
        r = self._append_row("", "")
        self._loading = False
        self.tbl.setCurrentCell(r, EN_COL)
        self.tbl.editItem(self.tbl.item(r, EN_COL))

    def _add_from_database(self):
        self._ensure_list()
        from .nuclear import NuclearLinePicker
        picker = NuclearLinePicker(self, multi=True)
        if picker.exec_() != QDialog.Accepted:
            return
        lines = picker.selected_lines()
        if not lines:
            self._note("No lines selected.")
            return
        self._loading = True
        for energy, isotope in lines:
            self._append_row(energy, isotope)
        self._loading = False
        self._persist_current()
        self._note(f"Added {len(lines)} line(s) from the database.")

    def _remove_rows(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True)
        if not rows:
            return
        self._loading = True
        for r in rows:
            self.tbl.removeRow(r)
        self._loading = False
        self._persist_current()

    # ── persistence ─────────────────────────────────────────────────────────────
    def _persist_current(self):
        name = self._current_list_name()
        if not name:
            return
        self._lists[name] = self._read_table()
        self._save()

    def _save(self):
        try:
            self._lists = save_favorite_lists(self._lists)
        except OSError as exc:
            self._warn(f"Could not save favorites: {exc}")

    # ── selection / status ──────────────────────────────────────────────────────
    def selected_energies(self):
        """Highlighted energies in keV, or all of the current list when none are."""
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()})
        if not rows:
            rows = range(self.tbl.rowCount())
        out = []
        for r in rows:
            item = self.tbl.item(r, EN_COL)
            if item is None:
                continue
            try:
                out.append(float(item.text().strip()))
            except ValueError:
                continue
        return out

    def _confirm(self, text):
        box = QMessageBox(QMessageBox.Question, "Favorite energies", text,
                          QMessageBox.Yes | QMessageBox.No, self)
        box.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        return box.exec_() == QMessageBox.Yes

    def _note(self, msg):
        self.status.setText(f"<span style='color:{T.ACCENT_GREEN}'>{msg}</span>")

    def _warn(self, msg):
        self.status.setText(f"<span style='color:{T.ACCENT_RED}'>{msg}</span>")
