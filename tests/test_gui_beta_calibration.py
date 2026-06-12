"""
Offscreen GUI regression tests for the wara --beta Calibration tab
(wara.gui_beta.calibration: CalibrationOptions / CalibrationPage /
CalibrationController).

Covers the workflow: adding centroids into the points table (as the
Drag-and-Fit window does), fitting a polynomial, setting the calibration
directly from coefficients, changing units, and applying the calibration to
the active spectrum.
"""

import os

# Must be set before the first QApplication is created (during collection).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QTableWidgetItem

from wara import spectrum as sp
from wara.peaksearch import PeakSearch
from wara.gui_beta.app import WaraBetaApp
from wara.gui_beta.calibration import CH_COL, E_COL, USE_COL


PEAK_CHANNELS = (500, 1200, 2500, 3300)
# Linear truth used to label the synthetic peaks: E = 0.5*ch + 10.
SLOPE, INTERCEPT = 0.5, 10.0


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def captured_warnings(monkeypatch):
    """Capture (not display) warning dialogs — a real modal would block/crash
    the offscreen tests."""
    calls = []
    monkeypatch.setattr(
        "wara.gui_beta.calibration.CalibrationController._show_warning",
        lambda self, title, text: calls.append((title, text)))
    return calls


def _synthetic_spectrum():
    ch = np.arange(4096)
    counts = np.full_like(ch, 5.0, dtype=float)
    for c in PEAK_CHANNELS:
        counts += 800.0 * np.exp(-0.5 * ((ch - c) / 4.0) ** 2)
    return sp.Spectrum(counts=counts)


@pytest.fixture
def app_with_peaks(qapp):
    w = WaraBetaApp()
    spect = _synthetic_spectrum()
    w.spect = spect
    w._spect_orig = spect.copy()
    w._active_name = "synthetic"
    w._refresh()
    w.search = PeakSearch(spect, ref_x=420, ref_fwhm=3, fwhm_at_0=1.0, min_snr=5)
    yield w
    w.close()


def _table(w):
    return w.calibration.opts.tbl_points


def _label_energies(w):
    """Fill the Energy column from the linear truth for every row."""
    tbl = _table(w)
    for r in range(tbl.rowCount()):
        chv = float(tbl.item(r, CH_COL).text())
        tbl.setItem(r, E_COL, QTableWidgetItem(f"{SLOPE * chv + INTERCEPT:.4f}"))


def _add_peak_channels(w):
    """Populate the points table with PEAK_CHANNELS (energies blank), mirroring
    how centroids arrive from the Drag-and-Fit window."""
    w.calibration.add_centroids([(float(c), "") for c in PEAK_CHANNELS])


# ---------------------------------------------------------------------------
# Adding centroids + polynomial calibration
# ---------------------------------------------------------------------------

class TestAddAndCalibrate:
    def test_add_centroids_fills_table(self, app_with_peaks):
        w = app_with_peaks
        _add_peak_channels(w)
        assert _table(w).rowCount() == len(PEAK_CHANNELS)

    def test_adding_same_channels_is_idempotent(self, app_with_peaks):
        w = app_with_peaks
        _add_peak_channels(w)
        _add_peak_channels(w)                # no duplicate channels
        assert _table(w).rowCount() == len(PEAK_CHANNELS)

    def test_linear_calibration_recovers_truth(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        cal.opts.cb_origin.setChecked(False)     # truth is E = 0.5ch + 10 (not through 0)
        _add_peak_channels(w)
        _label_energies(w)                       # fitting is automatic on edit
        assert cal.predicted is not None
        assert float(cal.opts.val_r2.text()) > 0.999999
        # E at channel 0 and the slope come straight from the fit.
        assert cal.predicted[0] == pytest.approx(INTERCEPT, abs=1e-2)
        assert cal.predicted[100] - cal.predicted[0] == pytest.approx(100 * SLOPE, abs=1e-3)
        assert cal.opts.btn_apply.isEnabled()

    def test_unticked_rows_excluded_from_fit(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _add_peak_channels(w)
        _label_energies(w)
        _table(w).item(0, USE_COL).setCheckState(Qt.Unchecked)
        chans, ergs = cal._read_points()
        assert len(chans) == len(PEAK_CHANNELS) - 1

    def test_reset_clears_points(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _add_peak_channels(w)
        cal._reset()
        assert _table(w).rowCount() == 0
        assert cal.predicted is None
        assert not cal.opts.btn_apply.isEnabled()


# ---------------------------------------------------------------------------
# Apply / remove
# ---------------------------------------------------------------------------

class TestApply:
    def test_apply_sets_energy_axis(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        cal.opts.cb_origin.setChecked(False)
        _add_peak_channels(w)
        _label_energies(w)
        cal._apply()
        assert w.spect.energies is not None
        assert w.spect.x_units == "Energy (keV)"
        assert w.spect.energies[0] == pytest.approx(INTERCEPT, abs=1.0)

    def test_apply_disabled_before_fit(self, app_with_peaks):
        w = app_with_peaks
        assert not w.calibration.opts.btn_apply.isEnabled()


# ---------------------------------------------------------------------------
# Set from equation
# ---------------------------------------------------------------------------

class TestEquation:
    def test_set_from_equation(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        cal.opts.coef_a.setText(str(INTERCEPT))
        cal.opts.coef_b.setText(str(SLOPE))
        cal._calibrate_equation()
        assert cal.predicted is not None
        assert cal.predicted[0] == pytest.approx(INTERCEPT)
        assert cal.predicted[200] == pytest.approx(SLOPE * 200 + INTERCEPT)
        assert cal.opts.btn_apply.isEnabled()
        cal._apply()
        assert w.spect.energies is not None

    def test_fit_populates_equation_boxes(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        cal.opts.cb_origin.setChecked(False)
        _add_peak_channels(w)
        _label_energies(w)                       # E = 0.5ch + 10, degree 1
        assert float(cal.opts.coef_a.text()) == pytest.approx(INTERCEPT, abs=1e-2)
        assert float(cal.opts.coef_b.text()) == pytest.approx(SLOPE, abs=1e-6)
        assert cal.opts.coef_c.text() == ""      # unused terms cleared
        assert cal.opts.coef_d.text() == ""

    def test_equation_rejects_non_numeric(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        cal.opts.coef_a.setText("oops")
        cal._calibrate_equation()
        assert cal.predicted is None
        assert not cal.opts.btn_apply.isEnabled()


# ---------------------------------------------------------------------------
# Centroids pushed from the Drag-and-Fit window
# ---------------------------------------------------------------------------

class TestCentroidPush:
    def test_add_centroid_energy_pairs(self, app_with_peaks):
        w = app_with_peaks
        w.calibration.add_centroids([(512.3, "260.5"), (1190.7, "")])
        chans = [float(_table(w).item(r, CH_COL).text())
                 for r in range(_table(w).rowCount())]
        assert chans == [pytest.approx(512.3), pytest.approx(1190.7)]
        # First row carries the entered energy; second was left blank.
        assert _table(w).item(0, E_COL).text() == "260.5"
        assert _table(w).item(1, E_COL).text() == ""

    def test_add_centroids_dedupes(self, app_with_peaks):
        w = app_with_peaks
        w.calibration.add_centroids([(512.3, "")])
        w.calibration.add_centroids([(512.3, ""), (800.0, "")])   # 512.3 already present
        assert _table(w).rowCount() == 2

    def test_same_peak_twice_rejected_with_warning(self, app_with_peaks, captured_warnings):
        w = app_with_peaks
        cal = w.calibration
        cal.add_centroids([(500.0, "260.0")])
        # Re-fitting the same peak gives a slightly different centroid; it is
        # within a FWHM of the existing point, so it must be rejected.
        cal.add_centroids([(500.3, "260.0")])
        assert _table(w).rowCount() == 1                # the duplicate was skipped
        assert captured_warnings, "expected a duplicate-peak warning"
        title, text = captured_warnings[-1]
        assert title == "Duplicate peak"
        assert "channel 500.30" in text

    def test_distinct_nearby_peaks_allowed(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        # Far enough apart (many FWHM) to be genuinely different peaks.
        cal.add_centroids([(500.0, "260.0")])
        cal.add_centroids([(560.0, "290.0")])
        assert _table(w).rowCount() == 2

    def test_pushed_pair_is_usable_in_fit(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        cal.opts.cb_origin.setChecked(False)
        # Channel/energy pairs along E = 0.5*ch + 10 — fits automatically.
        cal.add_centroids([(500.0, "260.0"), (1200.0, "610.0"), (2500.0, "1260.0")])
        assert cal.predicted is not None
        assert cal.predicted[0] == pytest.approx(INTERCEPT, abs=1e-2)

    def test_add_centroids_sets_units(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        cal.add_centroids([(500.0, "0.26"), (1200.0, "0.61")], units="MeV")
        assert cal.opts.units.currentText() == "MeV"
        assert cal.locked_units() == "MeV"      # reported once a point has energy


# ---------------------------------------------------------------------------
# Validation safety: repeated energy, duplicate channel typed in, degree count
# ---------------------------------------------------------------------------

class TestValidation:
    def test_repeated_energy_blocks_fit(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        cal.opts.cb_origin.setChecked(False)
        cal._add_row(channel=500.0, energy=260.0)
        cal._add_row(channel=1200.0, energy=260.0)   # same energy on two peaks
        cal._auto_calibrate()
        assert cal.predicted is None
        assert not cal.opts.btn_apply.isEnabled()

    def test_typed_duplicate_channel_blocks_fit(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        cal.opts.cb_origin.setChecked(False)
        cal._add_row(channel=500.0, energy=260.0)
        cal._add_row(channel=500.2, energy=610.0)    # same peak (within a FWHM)
        cal._auto_calibrate()
        assert cal.predicted is None
        assert not cal.opts.btn_apply.isEnabled()

    def test_degree_requires_enough_points(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        cal.opts.cb_origin.setChecked(False)
        cal.opts.degree.setValue(2)                  # quadratic needs 3 points
        cal._add_row(channel=500.0, energy=260.0)
        cal._add_row(channel=1200.0, energy=610.0)
        cal._auto_calibrate()
        assert cal.predicted is None                 # only 2 points → blocked
        assert not cal.opts.btn_apply.isEnabled()
        cal._add_row(channel=2500.0, energy=1260.0)  # third point unblocks it
        cal._auto_calibrate()
        assert cal.predicted is not None
        assert cal.opts.btn_apply.isEnabled()

    def test_apply_blocked_without_calibration(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        cal._apply()                                 # nothing calibrated yet
        assert w.spect.energies is None              # spectrum untouched


# ---------------------------------------------------------------------------
# Auto-fit, force origin, and units locking
# ---------------------------------------------------------------------------

class TestAutoFitAndUnits:
    def test_force_origin_checked_by_default(self, app_with_peaks):
        assert app_with_peaks.calibration.opts.cb_origin.isChecked()

    def test_single_point_with_force_origin_draws_line(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        cal.opts.cb_origin.setChecked(True)
        cal._add_row(channel=1000.0, energy=510.0)
        cal._auto_calibrate()
        # One point + the origin is enough for a line.
        assert cal.predicted is not None
        assert cal.predicted[0] == pytest.approx(0.0, abs=1e-6)
        assert cal.predicted[1000] == pytest.approx(510.0, rel=1e-3)

    def test_force_origin_toggle_refits_live(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        cal.opts.cb_origin.setChecked(False)
        # Two points whose free fit has a non-zero intercept.
        cal.add_centroids([(1000.0, "510.0"), (2000.0, "1000.0")])
        free_intercept = cal.predicted[0]
        assert free_intercept == pytest.approx(20.0, abs=1.0)
        # Toggling origin adds (0,0) as a point and refits automatically,
        # pulling the intercept toward zero (legacy set_origin behaviour).
        cal.opts.cb_origin.setChecked(True)
        assert abs(cal.predicted[0]) < abs(free_intercept)
        assert cal.predicted[0] != pytest.approx(free_intercept, abs=1.0)

    def test_units_stay_changeable_and_reset_clears(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        assert cal.opts.units.isEnabled()
        cal.add_centroids([(500.0, "260.0"), (1200.0, "610.0")])
        assert cal.opts.units.isEnabled()        # units remain changeable
        assert cal.locked_units() == "keV"       # reported to the Drag-and-Fit window
        cal._reset()
        assert cal.opts.units.isEnabled()
        assert cal.locked_units() is None        # no points → nothing to report

    def test_changing_units_converts_values_and_refits(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        # cb_origin defaults to True, so two points + the origin give a line.
        cal.add_centroids([(500.0, "260.0"), (1200.0, "610.0")])
        assert cal.predicted is not None
        kev_pred0 = cal.predicted[0]
        cal.opts.units.setCurrentText("MeV")     # keV → MeV: values ÷ 1000
        # The entered energies are converted, not just relabelled.
        tbl = _table(w)
        ergs = sorted(float(tbl.item(r, E_COL).text())
                      for r in range(tbl.rowCount())
                      if tbl.item(r, E_COL) and tbl.item(r, E_COL).text().strip())
        assert ergs == [pytest.approx(0.26), pytest.approx(0.61)]
        # The fit follows: the predicted curve is now 1000x smaller.
        assert cal.predicted[0] == pytest.approx(kev_pred0 / 1000.0)
        cal._apply()
        assert w.spect.x_units == "Energy (MeV)"

    def test_units_round_trip_preserves_values(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        cal.opts.cb_origin.setChecked(False)
        # A 7-significant-figure energy that would lose precision if the
        # conversion were formatted too coarsely.
        cal._add_row(channel=500.0, energy="1332.501")
        cal._add_row(channel=1200.0, energy="610.0")
        cal._auto_calibrate()
        cal.opts.units.setCurrentText("eV")      # keV → eV: ×1000
        cal.opts.units.setCurrentText("keV")     # eV → keV: ÷1000 (back to start)
        tbl = _table(w)
        ergs = sorted(float(tbl.item(r, E_COL).text()) for r in range(tbl.rowCount()))
        assert ergs == [pytest.approx(610.0), pytest.approx(1332.501)]
