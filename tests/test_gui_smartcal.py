"""
Offscreen GUI tests for the Smart Calibration dialog (wara.gui.smartcal)
and its integration with the Calibration tab controller.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QDialog, QListWidgetItem

from wara import spectrum as sp
from wara.peaksearch import PeakSearch
from wara.gui.app import WaraApp
from wara.gui import smartcal
from wara.gui.smartcal import (
    SmartCalibrationDialog, FavoriteEnergiesDialog,
    load_favorite_lists, save_favorite_lists,
)
from wara.gui.calibration import CH_COL, E_COL


# True linear calibration of the synthetic spectrum: E = 0.5*ch + 3.
SLOPE, INTERCEPT = 0.5, 3.0
PEAK_CHANNELS = (500, 1200, 2500, 3300)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _synthetic_spectrum():
    ch = np.arange(4096)
    counts = np.full_like(ch, 5.0, dtype=float)
    for c in PEAK_CHANNELS:
        counts += 1200.0 * np.exp(-0.5 * ((ch - c) / 4.0) ** 2)
    return sp.Spectrum(counts=counts)


@pytest.fixture
def app_with_peaks(qapp):
    w = WaraApp()
    spect = _synthetic_spectrum()
    w.spect = spect
    w._spect_orig = spect.copy()
    w._active_name = "synthetic"
    w._refresh()
    w.search = PeakSearch(spect, ref_x=420, ref_fwhm=3, fwhm_at_0=1.0, min_snr=20)
    yield w
    w.close()


def _energies(channels):
    return [INTERCEPT + SLOPE * c for c in channels]


def _fill(lst, values):
    for v in values:
        lst.addItem(QListWidgetItem(f"{v:g}"))


def _table(w):
    return w.calibration.opts.tbl_points


# ---------------------------------------------------------------------------

class TestSmartCalDialog:
    def test_match_populates_table(self, app_with_peaks):
        w = app_with_peaks
        dlg = SmartCalibrationDialog(w, w.calibration, w)
        _fill(dlg.lst_ch, PEAK_CHANNELS)
        _fill(dlg.lst_en, _energies(PEAK_CHANNELS))
        dlg.degree.setValue(1)
        dlg._match()
        cal = w.calibration
        assert cal.predicted is not None
        # Table holds exactly the 4 matched pairs — no origin row.
        tbl = _table(w)
        assert tbl.rowCount() == len(PEAK_CHANNELS)
        chans = sorted(float(tbl.item(r, CH_COL).text()) for r in range(tbl.rowCount()))
        assert chans == [float(c) for c in PEAK_CHANNELS]
        assert all(float(tbl.item(r, CH_COL).text()) != 0.0 for r in range(tbl.rowCount()))
        assert cal.opts.units.currentText() == "keV"

    def test_match_drops_spurious_channel(self, app_with_peaks):
        w = app_with_peaks
        dlg = SmartCalibrationDialog(w, w.calibration, w)
        _fill(dlg.lst_ch, list(PEAK_CHANNELS) + [1900])   # 1900 is spurious
        _fill(dlg.lst_en, _energies(PEAK_CHANNELS))
        dlg._match()
        assert "Matched 4" in dlg.result.text()
        assert _table(w).rowCount() == 4                  # spurious one excluded

    def test_too_few_points_errors(self, app_with_peaks):
        w = app_with_peaks
        dlg = SmartCalibrationDialog(w, w.calibration, w)
        _fill(dlg.lst_ch, [500])
        _fill(dlg.lst_en, [253.0])
        before = _table(w).rowCount()
        dlg._match()
        assert "at least" in dlg.result.text().lower()
        assert _table(w).rowCount() == before            # table untouched

    def test_no_match_shows_error(self, app_with_peaks):
        w = app_with_peaks
        dlg = SmartCalibrationDialog(w, w.calibration, w)
        _fill(dlg.lst_ch, [10, 400, 905])
        _fill(dlg.lst_en, [5.0, 933.0, 17.0])            # no monotonic consistent set
        dlg.tol.setText("0.01")
        dlg._match()
        assert "No calibration matched" in dlg.result.text()

    def test_import_peaks_from_search(self, app_with_peaks):
        w = app_with_peaks
        dlg = SmartCalibrationDialog(w, w.calibration, w)
        dlg._import_peaks()
        vals = dlg._values(dlg.lst_ch)
        assert len(vals) == len(w.search.peaks_idx)
        # Importing again does not duplicate channels.
        dlg._import_peaks()
        assert len(dlg._values(dlg.lst_ch)) == len(vals)

    def test_import_without_search_warns(self, qapp):
        w = WaraApp()
        w.spect = _synthetic_spectrum()
        w._spect_orig = w.spect.copy()
        w._refresh()
        w.search = None
        try:
            dlg = SmartCalibrationDialog(w, w.calibration, w)
            dlg._import_peaks()
            assert "Find Peaks" in dlg.result.text()
            assert dlg._values(dlg.lst_ch) == []
        finally:
            w.close()

    def test_energy_precision_preserved(self, app_with_peaks):
        """7-significant-figure reference energies must not be rounded away."""
        w = app_with_peaks
        dlg = SmartCalibrationDialog(w, w.calibration, w)
        dlg._add_values(dlg.lst_ch, [float(c) for c in PEAK_CHANNELS])
        precise = [253.2481, 603.5119, 1253.913, 1653.444]
        dlg._add_values(dlg.lst_en, precise)
        dlg.tol.setText("50")            # loose: keep all 4 matched
        dlg._match()
        tbl = _table(w)
        ergs = sorted(float(tbl.item(r, E_COL).text()) for r in range(tbl.rowCount()))
        assert ergs == pytest.approx(sorted(precise), abs=1e-3)

    def test_values_skips_blank_and_nonnumeric(self, app_with_peaks):
        w = app_with_peaks
        dlg = SmartCalibrationDialog(w, w.calibration, w)
        for txt in ("100", "", "abc", "250.5"):
            dlg.lst_ch.addItem(QListWidgetItem(txt))
        assert dlg._values(dlg.lst_ch) == [100.0, 250.5]


class TestSetSmartResults:
    def test_replaces_table_and_unforces_origin(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        # Pre-seed a stray point and leave Force-origin ticked.
        cal._add_row(channel=10.0, energy=5.0)
        assert cal.opts.cb_origin.isChecked()
        pairs = [(500.0, 253.0), (1200.0, 603.0), (2500.0, 1253.0)]
        cal.set_smart_results(pairs, units="keV")
        tbl = _table(w)
        assert tbl.rowCount() == 3                         # replaced, not appended
        chans = sorted(float(tbl.item(r, CH_COL).text()) for r in range(3))
        assert chans == [500.0, 1200.0, 2500.0]
        # Force-origin is switched off so the smart fit uses only matched points.
        assert not cal.opts.cb_origin.isChecked()
        assert cal.predicted is not None


class TestMultiLinePicker:
    def test_selected_energies_returns_list(self, qapp):
        from wara.gui.nuclear import NuclearLinePicker
        picker = NuclearLinePicker(multi=True, element="Co60")
        model = picker.table.model()
        if model is not None and model.rowCount() >= 2:
            picker.table.selectRow(0)
            picker.table.selectRow(1)   # ExtendedSelection keeps both
        out = picker.selected_energies()
        assert isinstance(out, list)
        picker.close()


@pytest.fixture
def fav_file(tmp_path, monkeypatch):
    """Point favorite-energy persistence at a throwaway file."""
    path = tmp_path / "favorite_energies.json"
    monkeypatch.setattr(smartcal, "FAV_PATH", str(path))
    return path


def _energies_of(lst):
    """Energies of a saved list (a list of {'energy','isotope'} dicts)."""
    return [e["energy"] for e in lst]


class TestFavoritePersistence:
    def test_roundtrip_named_lists_with_isotopes(self, fav_file):
        lists = {"API": [
            {"energy": 1332.5, "isotope": "Co-60"},
            {"energy": 1173.2, "isotope": "Co-60"},
        ]}
        save_favorite_lists(lists)
        loaded = load_favorite_lists()
        assert list(loaded) == ["API"]
        # Sorted by energy, isotopes preserved.
        assert loaded["API"] == [
            {"energy": 1173.2, "isotope": "Co-60"},
            {"energy": 1332.5, "isotope": "Co-60"},
        ]

    def test_dedup_within_list(self, fav_file):
        cleaned = save_favorite_lists({"L": [
            {"energy": 511.0, "isotope": "annih"},
            {"energy": 511.0, "isotope": "dup"},
        ]})
        assert cleaned["L"] == [{"energy": 511.0, "isotope": "annih"}]

    def test_load_missing_file_returns_empty(self, fav_file):
        assert not fav_file.exists()
        assert load_favorite_lists() == {}

    def test_load_corrupt_file_returns_empty(self, fav_file):
        fav_file.write_text("{ not json", encoding="utf-8")
        assert load_favorite_lists() == {}

    def test_v1_file_migrates_to_favorites_list(self, fav_file):
        fav_file.write_text('{"version": 1, "energies": [20.0, 10.0]}', encoding="utf-8")
        loaded = load_favorite_lists()
        assert list(loaded) == ["Favorites"]
        assert loaded["Favorites"] == [
            {"energy": 10.0, "isotope": ""},
            {"energy": 20.0, "isotope": ""},
        ]

    def test_skips_nonnumeric_energy_entries(self, fav_file):
        cleaned = save_favorite_lists({"L": [
            {"energy": 10.0, "isotope": "a"},
            {"energy": "oops", "isotope": "b"},
            {"energy": 20.0, "isotope": "c"},
        ]})
        assert _energies_of(cleaned["L"]) == [10.0, 20.0]


class TestFavoriteEnergiesDialog:
    def test_combo_lists_and_loads_table(self, qapp, fav_file):
        save_favorite_lists({
            "API": [{"energy": 100.0, "isotope": "X"}],
            "BG":  [{"energy": 200.0, "isotope": "Y"}, {"energy": 300.0, "isotope": "Z"}],
        })
        dlg = FavoriteEnergiesDialog()
        assert sorted(dlg.cmb.itemText(i) for i in range(dlg.cmb.count())) == ["API", "BG"]
        dlg.cmb.setCurrentText("BG")
        assert dlg.tbl.rowCount() == 2
        assert dlg.selected_energies() == [200.0, 300.0]
        dlg.close()

    def test_new_list_creates_and_selects(self, qapp, fav_file, monkeypatch):
        monkeypatch.setattr(smartcal.QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("API", True)))
        dlg = FavoriteEnergiesDialog()
        dlg._new_list()
        assert dlg._current_list_name() == "API"
        assert "API" in load_favorite_lists()
        dlg.close()

    def test_add_row_and_edit_persists(self, qapp, fav_file, monkeypatch):
        monkeypatch.setattr(smartcal.QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("API", True)))
        dlg = FavoriteEnergiesDialog()
        dlg._new_list()
        dlg._add_row_blank()
        r = dlg.tbl.rowCount() - 1
        dlg.tbl.item(r, smartcal.EN_COL).setText("661.7")    # commits → save
        dlg.tbl.item(r, smartcal.ISO_COL).setText("Cs-137")
        loaded = load_favorite_lists()["API"]
        assert loaded == [{"energy": 661.7, "isotope": "Cs-137"}]
        dlg.close()

    def test_add_from_database_uses_energy_and_isotope(self, qapp, fav_file, monkeypatch):
        monkeypatch.setattr(smartcal.QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("API", True)))
        # Stub the database picker: accept with two (energy, isotope) lines.
        from wara.gui import nuclear
        monkeypatch.setattr(nuclear.NuclearLinePicker, "exec_", lambda self: QDialog.Accepted)
        monkeypatch.setattr(nuclear.NuclearLinePicker, "selected_lines",
                            lambda self: [(1173.2, "Co-60"), (1332.5, "Co-60")])
        dlg = FavoriteEnergiesDialog()
        dlg._new_list()
        dlg._add_from_database()
        assert load_favorite_lists()["API"] == [
            {"energy": 1173.2, "isotope": "Co-60"},
            {"energy": 1332.5, "isotope": "Co-60"},
        ]
        dlg.close()

    def test_remove_rows_persists(self, qapp, fav_file):
        save_favorite_lists({"API": [
            {"energy": 100.0, "isotope": "a"},
            {"energy": 200.0, "isotope": "b"},
            {"energy": 300.0, "isotope": "c"},
        ]})
        dlg = FavoriteEnergiesDialog()
        dlg.tbl.selectRow(1)                      # the 200.0 row
        dlg._remove_rows()
        assert _energies_of(load_favorite_lists()["API"]) == [100.0, 300.0]
        dlg.close()

    def test_rename_list(self, qapp, fav_file, monkeypatch):
        save_favorite_lists({"API": [{"energy": 100.0, "isotope": "a"}]})
        monkeypatch.setattr(smartcal.QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("PGAA", True)))
        dlg = FavoriteEnergiesDialog()
        dlg._rename_list()
        lists = load_favorite_lists()
        assert "PGAA" in lists and "API" not in lists
        assert dlg._current_list_name() == "PGAA"
        dlg.close()

    def test_delete_list(self, qapp, fav_file, monkeypatch):
        save_favorite_lists({"API": [{"energy": 100.0, "isotope": "a"}],
                             "BG": [{"energy": 200.0, "isotope": "b"}]})
        monkeypatch.setattr(FavoriteEnergiesDialog, "_confirm", lambda self, text: True)
        dlg = FavoriteEnergiesDialog()
        dlg.cmb.setCurrentText("API")
        dlg._delete_list()
        assert list(load_favorite_lists()) == ["BG"]
        dlg.close()

    def test_selected_returns_only_highlighted(self, qapp, fav_file):
        save_favorite_lists({"API": [
            {"energy": 10.0, "isotope": "a"},
            {"energy": 20.0, "isotope": "b"},
            {"energy": 30.0, "isotope": "c"},
        ]})
        dlg = FavoriteEnergiesDialog()
        dlg.tbl.selectRow(0)
        dlg.tbl.item(2, smartcal.EN_COL).setSelected(True)
        assert dlg.selected_energies() == [10.0, 30.0]
        dlg.close()


class TestFromFavoritesIntegration:
    def test_from_favorites_adds_to_energy_column(self, app_with_peaks, fav_file, monkeypatch):
        save_favorite_lists({"API": [
            {"energy": 253.0, "isotope": "a"},
            {"energy": 603.0, "isotope": "b"},
        ]})
        w = app_with_peaks
        dlg = SmartCalibrationDialog(w, w.calibration, w)
        # Auto-accept the favorites dialog; with nothing selected it returns all.
        monkeypatch.setattr(FavoriteEnergiesDialog, "exec_", lambda self: QDialog.Accepted)

        dlg._from_favorites()
        assert sorted(dlg._values(dlg.lst_en)) == [253.0, 603.0]
        # Adding again does not duplicate the already-present energies.
        dlg._from_favorites()
        assert sorted(dlg._values(dlg.lst_en)) == [253.0, 603.0]
