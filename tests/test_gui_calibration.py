"""
Offscreen GUI regression tests for the wara Calibration tab
(wara.gui.calibration: CalibrationOptions / CalibrationPage /
CalibrationController).

Covers the workflow: adding centroids into the points table (as the
Drag-and-Fit window does), fitting a polynomial, setting the calibration
directly from coefficients, changing units, and applying the calibration to
the active spectrum.

A "Force origin (0, 0)" checkbox (ticked by default) adds the origin to the fit.
Tests that need a fit which does *not* pass through the origin clear it via
``_set_origin(w, False)``.
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
from wara.gui.app import WaraApp
from wara.gui import calibration as calmod
from wara.gui.calibration import CH_COL, E_COL, USE_COL


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
        "wara.gui.calibration.CalibrationController._show_warning",
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
    w = WaraApp()
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


def _set_origin(w, checked):
    """Tick/untick the 'Force origin (0, 0)' checkbox."""
    w.calibration.opts.cb_origin.setChecked(checked)


def _peak_rows(w):
    """(channel, energy_text) for every non-origin row."""
    tbl = _table(w)
    rows = []
    for r in range(tbl.rowCount()):
        ch_it = tbl.item(r, CH_COL)
        try:
            chv = float(ch_it.text())
        except (AttributeError, ValueError):
            continue
        if chv == 0.0:
            continue
        e_it = tbl.item(r, E_COL)
        rows.append((chv, e_it.text() if e_it else ""))
    return rows


def _peak_energies(w):
    """Sorted numeric energies of the non-origin rows."""
    return sorted(float(e) for _, e in _peak_rows(w) if e.strip())


def _label_energies(w):
    """Fill the Energy column of the non-origin rows from the linear truth."""
    tbl = _table(w)
    for r in range(tbl.rowCount()):
        chv = float(tbl.item(r, CH_COL).text())
        if chv == 0.0:                       # leave the origin row at (0, 0)
            continue
        tbl.setItem(r, E_COL, QTableWidgetItem(f"{SLOPE * chv + INTERCEPT:.4f}"))


def _add_peak_channels(w):
    """Populate the points table with PEAK_CHANNELS (energies blank), mirroring
    how centroids arrive from the Drag-and-Fit window."""
    w.calibration.add_centroids([(float(c), "") for c in PEAK_CHANNELS])


# ---------------------------------------------------------------------------
# Force-origin checkbox
# ---------------------------------------------------------------------------

class TestForceOrigin:
    def test_origin_row_shown_when_checked_by_default(self, app_with_peaks):
        w = app_with_peaks
        assert w.calibration.opts.cb_origin.isChecked()  # ticked by default
        assert _table(w).rowCount() == 1                  # and shown as a (0,0) row
        assert float(_table(w).item(0, CH_COL).text()) == 0.0
        assert float(_table(w).item(0, E_COL).text()) == 0.0

    def test_unchecking_removes_origin_row(self, app_with_peaks):
        w = app_with_peaks
        _set_origin(w, False)
        assert _table(w).rowCount() == 0
        _set_origin(w, True)
        assert _table(w).rowCount() == 1                  # comes back when re-checked


# ---------------------------------------------------------------------------
# Adding centroids + polynomial calibration
# ---------------------------------------------------------------------------

class TestAddAndCalibrate:
    def test_add_centroids_fills_table(self, app_with_peaks):
        w = app_with_peaks
        _add_peak_channels(w)
        assert _table(w).rowCount() == len(PEAK_CHANNELS) + 1   # + origin row

    def test_adding_same_channels_is_idempotent(self, app_with_peaks):
        w = app_with_peaks
        _add_peak_channels(w)
        _add_peak_channels(w)                # no duplicate channels
        assert _table(w).rowCount() == len(PEAK_CHANNELS) + 1

    def test_linear_calibration_recovers_truth(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)                    # truth is E = 0.5ch + 10 (not through 0)
        _add_peak_channels(w)
        _label_energies(w)                       # fitting is automatic on edit
        assert cal.predicted is not None
        # E at channel 0 and the slope come straight from the fit.
        assert cal.predicted[0] == pytest.approx(INTERCEPT, abs=1e-2)
        assert cal.predicted[100] - cal.predicted[0] == pytest.approx(100 * SLOPE, abs=1e-3)
        assert cal.opts.btn_apply.isEnabled()

    def test_unticked_rows_excluded_from_fit(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)
        _add_peak_channels(w)
        _label_energies(w)
        before = len(cal._read_points()[0])
        _table(w).item(0, USE_COL).setCheckState(Qt.Unchecked)
        assert len(cal._read_points()[0]) == before - 1

    def test_reset_restores_origin_row(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _add_peak_channels(w)
        cal._reset()
        # Force-origin is still on, so reset leaves just the (0,0) row.
        assert _table(w).rowCount() == 1
        assert float(_table(w).item(0, CH_COL).text()) == 0.0
        assert cal.predicted is None
        assert not cal.opts.btn_apply.isEnabled()

    def test_reset_button_is_the_only_reset(self, app_with_peaks):
        """The Calibration tab exposes a single Reset button (the old in-tab
        Remove-Calibration button was retired in favour of the Spectrum tab's)."""
        cal = app_with_peaks.calibration
        assert cal.opts.btn_reset.text() == "Reset"
        assert not hasattr(cal.opts, "btn_remove_cal")

    def test_plot_has_navigation_toolbar(self, app_with_peaks):
        """The calibration plot offers zoom/pan/home via a nav toolbar, and a
        replot resets its history without error."""
        page = app_with_peaks.calibration_page
        assert hasattr(page, "toolbar")
        labels = {a.text() for a in page.toolbar.actions() if a.text()}
        assert {"Home", "Pan", "Zoom"} <= labels
        page.show_empty()                 # exercises _reset_nav after a replot

    def test_selecting_row_highlights_point(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)
        _add_peak_channels(w)
        _label_energies(w)
        assert cal.predicted is not None
        assert cal.page._hl is None                       # nothing selected yet
        # Select the first non-origin row → a highlight ring is drawn.
        peak_row = next(r for r in range(_table(w).rowCount())
                        if float(_table(w).item(r, CH_COL).text()) != 0.0)
        _table(w).selectRow(peak_row)
        assert cal.page._hl is not None
        hx, hy = cal.page._hl.get_xdata()[0], cal.page._hl.get_ydata()[0]
        assert hx == pytest.approx(float(_table(w).item(peak_row, CH_COL).text()))
        assert hy == pytest.approx(float(_table(w).item(peak_row, E_COL).text()))

    def test_highlight_survives_refit(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)
        _add_peak_channels(w)
        _label_energies(w)
        _table(w).selectRow(1)
        assert cal.page._hl is not None
        cal._auto_calibrate()                              # replot
        assert cal.page._hl is not None                   # re-applied

    def test_clearing_selection_clears_highlight(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)
        _add_peak_channels(w)
        _label_energies(w)
        _table(w).selectRow(1)
        assert cal.page._hl is not None
        _table(w).clearSelection()
        assert cal.page._hl is None

    def test_points_sort_ascending_on_add(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)                # drop the (0,0) row for a clean check
        # A completed row sorts when its last cell commits (here the energy cell).
        for ch, e in ((1500.0, 760.0), (300.0, 160.0), (900.0, 460.0)):
            r = cal._add_row(channel=ch, energy=e)
            cal._on_cell_changed(r, E_COL)
        tbl = _table(w)
        chans = [float(tbl.item(r, CH_COL).text()) for r in range(tbl.rowCount())]
        assert chans == [300.0, 900.0, 1500.0]

    def test_partial_entry_does_not_sort(self, app_with_peaks):
        """Typing only the channel of a new pair must not reshuffle the table —
        the row waits in place until the energy is also entered."""
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)
        cal._add_row(channel=1000.0, energy=510.0)   # an existing complete row
        cal._on_cell_changed(0, E_COL)
        r = cal._add_row(channel=50.0, energy="")     # new row, channel only
        cal._on_cell_changed(r, CH_COL)               # channel committed, energy blank
        tbl = _table(w)
        # The incomplete row stays last (not sorted above the 1000 row yet).
        assert float(tbl.item(tbl.rowCount() - 1, CH_COL).text()) == 50.0
        # Once the energy is filled in, the row sorts into place.
        tbl.item(r, E_COL).setText("25.0")            # fires cellChanged → complete
        chans = [float(tbl.item(i, CH_COL).text()) for i in range(tbl.rowCount())]
        assert chans == [50.0, 1000.0]                # the 50 row sorted in front

    def test_sort_keeps_use_flag_with_row(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)
        cal._add_row(channel=1500.0, energy=760.0, use=False)   # unticked
        cal._add_row(channel=300.0, energy=160.0, use=True)
        cal._sort_points()
        tbl = _table(w)
        # After sorting: 300 (ticked) first, 1500 (unticked) last.
        order = [(float(tbl.item(r, CH_COL).text()),
                  tbl.item(r, USE_COL).checkState() == Qt.Checked)
                 for r in range(tbl.rowCount())]
        assert order == [(300.0, True), (1500.0, False)]


# ---------------------------------------------------------------------------
# Apply / remove
# ---------------------------------------------------------------------------

class TestApply:
    def test_apply_sets_energy_axis(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)
        _add_peak_channels(w)
        _label_energies(w)
        cal._apply()
        assert w.spect.energies is not None
        assert w.spect.x_units == "Energy (keV)"
        assert w.spect.energies[0] == pytest.approx(INTERCEPT, abs=1.0)

    def test_apply_disabled_before_fit(self, app_with_peaks):
        w = app_with_peaks
        assert not w.calibration.opts.btn_apply.isEnabled()

    def test_apply_marks_button_applied(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)
        _add_peak_channels(w)
        _label_energies(w)
        assert cal.opts.btn_apply.objectName() == "primary_btn"
        cal._apply()
        assert cal.opts.btn_apply.objectName() == "applied_btn"
        assert "Applied" in cal.opts.btn_apply.text()
        # Editing the calibration afterwards reverts the button to "pending".
        cal._add_row(channel=900.0, energy=460.0)
        cal._auto_calibrate()
        assert cal.opts.btn_apply.objectName() == "primary_btn"

    def test_remove_calibration_reverts_apply_button(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)
        _add_peak_channels(w)
        _label_energies(w)
        cal._apply()
        assert cal.opts.btn_apply.objectName() == "applied_btn"
        # Removing the calibration from the Spectrum tab reverts the button.
        w._remove_calibration()
        assert cal.opts.btn_apply.objectName() == "primary_btn"
        assert cal.opts.btn_apply.text() == "Apply to spectrum"


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
        _set_origin(w, False)
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
        rows = _peak_rows(w)
        chans = [c for c, _ in rows]
        assert chans == [pytest.approx(512.3), pytest.approx(1190.7)]
        # First pushed row carries the entered energy; second was left blank.
        assert rows[0][1] == "260.5"
        assert rows[1][1] == ""

    def test_add_centroids_dedupes(self, app_with_peaks):
        w = app_with_peaks
        w.calibration.add_centroids([(512.3, "")])
        w.calibration.add_centroids([(512.3, ""), (800.0, "")])   # 512.3 already present
        assert len(_peak_rows(w)) == 2

    def test_same_peak_twice_rejected_with_warning(self, app_with_peaks, captured_warnings):
        w = app_with_peaks
        cal = w.calibration
        cal.add_centroids([(500.0, "260.0")])
        # Re-fitting the same peak gives a slightly different centroid; it is
        # within a FWHM of the existing point, so it must be rejected.
        cal.add_centroids([(500.3, "260.0")])
        assert len(_peak_rows(w)) == 1                  # the duplicate was skipped
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
        assert len(_peak_rows(w)) == 2

    def test_pushed_pair_is_usable_in_fit(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)
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
        _set_origin(w, False)
        cal._add_row(channel=500.0, energy=260.0)
        cal._add_row(channel=1200.0, energy=260.0)   # same energy on two peaks
        cal._auto_calibrate()
        assert cal.predicted is None
        assert not cal.opts.btn_apply.isEnabled()

    def test_typed_duplicate_channel_blocks_fit(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)
        cal._add_row(channel=500.0, energy=260.0)
        cal._add_row(channel=500.2, energy=610.0)    # same peak (within a FWHM)
        cal._auto_calibrate()
        assert cal.predicted is None
        assert not cal.opts.btn_apply.isEnabled()

    def test_degree_requires_enough_points(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)
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
# Auto-fit, origin point, and units locking
# ---------------------------------------------------------------------------

class TestAutoFitAndUnits:
    def test_single_point_with_origin_draws_line(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, True)                         # ticked by default, but be explicit
        cal._add_row(channel=1000.0, energy=510.0)
        cal._auto_calibrate()
        # One point + the origin is enough for a line.
        assert cal.predicted is not None
        assert cal.predicted[0] == pytest.approx(0.0, abs=1e-6)
        assert cal.predicted[1000] == pytest.approx(510.0, rel=1e-3)

    def test_origin_toggle_refits_live(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)
        # Two points whose free fit has a non-zero intercept.
        cal.add_centroids([(1000.0, "510.0"), (2000.0, "1000.0")])
        free_intercept = cal.predicted[0]
        assert free_intercept == pytest.approx(20.0, abs=1.0)
        # Re-ticking the origin adds (0,0) to the fit and refits automatically,
        # pulling the intercept toward zero.
        _set_origin(w, True)
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
        assert cal.locked_units() is None        # empty table → nothing to report

    def test_changing_units_converts_values_and_refits(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        # Origin ticked by default, so two points + the origin give a line.
        cal.add_centroids([(500.0, "260.0"), (1200.0, "610.0")])
        assert cal.predicted is not None
        kev_pred0 = cal.predicted[0]
        cal.opts.units.setCurrentText("MeV")     # keV → MeV: values ÷ 1000
        # The entered energies are converted, not just relabelled.
        assert _peak_energies(w) == [pytest.approx(0.26), pytest.approx(0.61)]
        # The fit follows: the predicted curve is now 1000x smaller.
        assert cal.predicted[0] == pytest.approx(kev_pred0 / 1000.0)
        cal._apply()
        assert w.spect.x_units == "Energy (MeV)"

    def test_units_round_trip_preserves_values(self, app_with_peaks):
        w = app_with_peaks
        cal = w.calibration
        _set_origin(w, False)
        # A 7-significant-figure energy that would lose precision if the
        # conversion were formatted too coarsely.
        cal._add_row(channel=500.0, energy="1332.501")
        cal._add_row(channel=1200.0, energy="610.0")
        cal._auto_calibrate()
        cal.opts.units.setCurrentText("eV")      # keV → eV: ×1000
        cal.opts.units.setCurrentText("keV")     # eV → keV: ÷1000 (back to start)
        assert _peak_energies(w) == [pytest.approx(610.0), pytest.approx(1332.501)]


# ---------------------------------------------------------------------------
# Saving / loading / deleting calibration files
# ---------------------------------------------------------------------------

@pytest.fixture
def cal_dir(tmp_path, monkeypatch):
    """Redirect saved-calibration storage to a temp directory."""
    d = tmp_path / "calibrations"
    monkeypatch.setattr(calmod, "CAL_DIR", str(d))
    return d


@pytest.fixture
def confirm_yes(monkeypatch):
    """Auto-accept the modal confirm (overwrite / delete) dialogs."""
    monkeypatch.setattr(
        "wara.gui.calibration.CalibrationController._confirm",
        lambda self, title, text: True)


def _make_calibration(w):
    """Build a valid (non-origin) linear calibration in the table."""
    cal = w.calibration
    _set_origin(w, False)
    _add_peak_channels(w)
    _label_energies(w)
    assert cal.predicted is not None
    return cal


class TestSaveLoadDelete:
    def test_save_creates_file_and_lists_it(self, app_with_peaks, cal_dir):
        w = app_with_peaks
        cal = _make_calibration(w)
        cal.opts.cal_name.setText("hpge_co60")
        cal._save_calibration()
        assert (cal_dir / "hpge_co60.json").exists()
        assert "hpge_co60" in [cal.opts.cal_files.itemText(i)
                               for i in range(cal.opts.cal_files.count())]

    def test_save_requires_a_calibration(self, app_with_peaks, cal_dir):
        w = app_with_peaks
        cal = w.calibration                      # empty table → no fit
        cal.opts.cal_name.setText("empty")
        cal._save_calibration()
        assert not (cal_dir / "empty.json").exists()

    def test_save_sanitizes_name(self, app_with_peaks, cal_dir):
        w = app_with_peaks
        cal = _make_calibration(w)
        cal.opts.cal_name.setText("bad/na me*?")
        cal._save_calibration()
        files = list(cal_dir.glob("*.json"))
        assert len(files) == 1
        assert "/" not in files[0].stem and "*" not in files[0].stem

    def test_round_trip_restores_points_units_degree(self, app_with_peaks, cal_dir):
        w = app_with_peaks
        cal = _make_calibration(w)
        cal.opts.degree.setValue(1)
        cal.opts.cal_name.setText("roundtrip")
        cal._save_calibration()
        saved_pred0 = cal.predicted[0]
        saved_rows = _peak_rows(w)
        # Wipe the table, then load the file back.
        cal._reset()
        assert cal.predicted is None
        cal.opts.cal_files.setCurrentText("roundtrip")
        cal._load_calibration()
        assert cal.predicted is not None
        assert cal.predicted[0] == pytest.approx(saved_pred0, abs=1e-6)
        assert _peak_rows(w) == saved_rows
        assert cal.opts.units.currentText() == "keV"

    def test_load_equation_only_calibration(self, app_with_peaks, cal_dir):
        w = app_with_peaks
        cal = w.calibration
        cal.opts.coef_a.setText(str(INTERCEPT))
        cal.opts.coef_b.setText(str(SLOPE))
        cal._calibrate_equation()
        cal.opts.cal_name.setText("eqn_only")
        cal._save_calibration()
        cal._reset()
        cal.opts.cal_files.setCurrentText("eqn_only")
        cal._load_calibration()
        assert cal.predicted is not None
        assert cal.predicted[200] == pytest.approx(SLOPE * 200 + INTERCEPT)

    def test_overwrite_when_confirmed(self, app_with_peaks, cal_dir, confirm_yes):
        w = app_with_peaks
        cal = _make_calibration(w)
        cal.opts.cal_name.setText("dup")
        cal._save_calibration()
        cal.opts.cal_name.setText("dup")
        cal._save_calibration()                  # confirm_yes → overwrites
        assert len(list(cal_dir.glob("*.json"))) == 1

    def test_delete_removes_file_and_entry(self, app_with_peaks, cal_dir, confirm_yes):
        w = app_with_peaks
        cal = _make_calibration(w)
        cal.opts.cal_name.setText("to_delete")
        cal._save_calibration()
        assert (cal_dir / "to_delete.json").exists()
        cal.opts.cal_files.setCurrentText("to_delete")
        cal._delete_calibration()
        assert not (cal_dir / "to_delete.json").exists()
        assert cal.opts.cal_files.count() == 0
        assert not cal.opts.btn_cal_load.isEnabled()

    def test_refresh_discovers_external_files(self, app_with_peaks, cal_dir):
        w = app_with_peaks
        cal = w.calibration
        cal_dir.mkdir(parents=True, exist_ok=True)
        (cal_dir / "external.json").write_text(
            '{"units": "keV", "degree": 1, "points": [], "coeffs": ["", "", "", ""]}')
        cal.refresh_saved()
        assert "external" in [cal.opts.cal_files.itemText(i)
                              for i in range(cal.opts.cal_files.count())]
