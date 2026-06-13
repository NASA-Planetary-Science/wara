"""
Offscreen GUI tests for the Smart Calibration dialog (wara.gui_beta.smartcal)
and its integration with the Calibration tab controller.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QListWidgetItem

from wara import spectrum as sp
from wara.peaksearch import PeakSearch
from wara.gui_beta.app import WaraBetaApp
from wara.gui_beta.smartcal import SmartCalibrationDialog
from wara.gui_beta.calibration import CH_COL, E_COL


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
    w = WaraBetaApp()
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
        w = WaraBetaApp()
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
        from wara.gui_beta.nuclear import NuclearLinePicker
        picker = NuclearLinePicker(multi=True, element="Co60")
        model = picker.table.model()
        if model is not None and model.rowCount() >= 2:
            picker.table.selectRow(0)
            picker.table.selectRow(1)   # ExtendedSelection keeps both
        out = picker.selected_energies()
        assert isinstance(out, list)
        picker.close()
