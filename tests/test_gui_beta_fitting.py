"""
Offscreen GUI regression tests for the wara --beta Drag-and-Fit window
(wara.gui_beta.fitting.FitWindow), focused on the "Net area − linear bkg"
method (wara.advanced_fit.PeakAreaLinearBkg).

Regression target: the area path mapped the dragged ROI to full-precision outer
bounds but read the inner peak edges from spin boxes rounded to 2 decimals. An
inner edge resting on an ROI boundary could round a hair outside the outer
bound, so the old guard ``outer_l <= inner_l < inner_r <= outer_r`` rejected
valid input with "Set the peak start below the peak end" — intermittently,
depending on how the ROI's fractional value rounded.
"""

import os

# Must be set before the first QApplication is created (during collection).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication

from wara import file_reader
from wara.peaksearch import PeakSearch
from wara.gui_beta.fitting import FitWindow


DATA_DIR = Path(__file__).parent.parent / "examples" / "data"
CSV_WITH_CAL = str(DATA_DIR / "test_data_cebr_cal.csv")
TXT_HPGE = str(DATA_DIR / "test_data_hpge_NH3.txt")   # calibrated, has a livetime


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def search():
    spect = file_reader.read_csv(CSV_WITH_CAL)
    return PeakSearch(spectrum=spect, ref_x=1150, ref_fwhm=30.0,
                      fwhm_at_0=1.0, min_snr=3, method="km")


@pytest.fixture
def window(qapp, search):
    w = FitWindow(None, search)
    yield w
    w.close()


def _area_row(w):
    """Return the (label, value) text of the 'Net (A)' table row, or None."""
    for i in range(w.table.rowCount()):
        item = w.table.item(i, 0)
        if item is not None and item.text() == "Net (A)":
            return item.text(), w.table.item(i, 1).text()
    return None


# ---------------------------------------------------------------------------
# Method switching wiring
# ---------------------------------------------------------------------------

class TestMethodSwitch:
    def test_switch_hides_peakfit_controls_and_relabels(self, window):
        window.show()
        window.set_roi(1106, 1214)
        window.method.setCurrentIndex(1)   # Net area − linear bkg
        assert not window.peakfit_controls.isVisible()
        # The ROI sliders are relabeled for the net-area workflow. The
        # descriptive text now lives in the slider tooltips (and the table
        # header), not separate QLabels.
        assert "Left edge of the peak" in window.slider_lo.toolTip()
        assert "Right edge of the peak" in window.slider_hi.toolTip()

    def test_switch_back_restores_peakfit_controls(self, window):
        window.show()
        window.set_roi(1106, 1214)
        window.method.setCurrentIndex(1)
        window.method.setCurrentIndex(0)   # back to Peak fit
        assert window.peakfit_controls.isVisible()
        assert "Trim the lower bound" in window.slider_lo.toolTip()


# ---------------------------------------------------------------------------
# Net-area path — the rounding regression
# ---------------------------------------------------------------------------

class TestNetAreaPath:
    def test_fractional_roi_boundary_does_not_error(self, window):
        """ROI bounds that round OUTSIDE the spin-box 2-decimal grid must still
        compute (the original intermittent bug)."""
        window.method.setCurrentIndex(1)
        # 1106.0104 -> spin 1106.01 (below true min);
        # 1214.2480 -> spin 1214.25 (above true max).
        window.set_roi(1106.0104579, 1214.2480461)
        status = window.lbl_status.text()
        assert status.startswith("Net area A ="), status
        assert "peak start below the peak end" not in status

    def test_default_endpoints_populate_table(self, window):
        window.method.setCurrentIndex(1)
        window.set_roi(1106.0104579, 1214.2480461)
        row = _area_row(window)
        assert row is not None, "Net (A) row missing from results table"
        assert "±" in row[1]

    def test_example_ranges_reproduce_known_area(self, window):
        """Outer ROI [1060, 1260] with inner edges [1105, 1214] mirrors
        example_advfit_area_bkg.py -> net area A ≈ 9626.36."""
        window.method.setCurrentIndex(1)
        window.set_roi(1060, 1260)
        # Move the peak-edge spin boxes to the example's inner bounds.
        window.roi_lo.setValue(1105)
        window.roi_hi.setValue(1214)
        label, value = _area_row(window)
        assert value.startswith("9626.36"), value

    def test_degenerate_roi_is_rejected_cleanly(self, window):
        """A collapsed ROI (outer_l == outer_r) hits the inner_l >= inner_r
        guard and reports it instead of raising."""
        window.method.setCurrentIndex(1)
        window.set_roi(1106, 1214)
        # Force a degenerate ROI and re-run the area path directly.
        window._roi_hi = window._roi_lo
        window._refit_area()
        assert "peak start below the peak end" in window.lbl_status.text()


# ---------------------------------------------------------------------------
# Peak-fit table: area units (livetime) and header units.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hpge_search():
    spect = file_reader.read_txt(TXT_HPGE)
    return PeakSearch(spect, ref_x=420, ref_fwhm=3, min_snr=15, method="fast")


@pytest.fixture
def hpge_window(qapp, hpge_search):
    w = FitWindow(None, hpge_search)
    yield w
    w.close()


class TestPeakFitTable:
    def test_headers_carry_x_units(self, window):
        window.show()
        window.set_roi(1106, 1214)   # cebr is calibrated in keV
        n = window.table.columnCount()
        headers = [window.table.horizontalHeaderItem(c).text() for c in range(n)]
        assert headers == ["Centroid (keV)", "Area", "FWHM (keV)",
                           "FWTM (keV)", "FWTM/FWHM", "Asym"]

    def test_area_is_raw_counts_despite_livetime(self, hpge_search, hpge_window):
        """Regression: peak area is net counts (not counts/sec) even when the
        spectrum has a livetime, so the GUI shows the backend area directly and
        it matches the plot and the net-area method."""
        lt = hpge_search.spectrum.livetime
        assert lt and not hpge_search.spectrum.cps   # this file does have one
        pk = float(hpge_search.spectrum.energies[hpge_search.peaks_idx[5]])
        hpge_window.show()
        hpge_window.set_roi(pk - 8, pk + 8)
        gui_area = float(hpge_window.table.item(0, 1).text().split("±")[0])
        from wara.peakfit import PeakFit
        backend_area = PeakFit(hpge_search, [pk - 8, pk + 8],
                               bkg="poly1").peak_info[0]["area"]
        # peak_info["area"] is already net counts; the GUI displays it as-is
        # (no livetime scaling), so the two match exactly.
        assert gui_area == pytest.approx(backend_area, rel=1e-6)
        assert gui_area > 100   # plainly counts, not a ~1-count/sec rate


# ---------------------------------------------------------------------------
# New peak-fitting features surfaced in the beta GUI: all profiles, the step
# background, and the enriched results table (FWTM / tailing / asymmetry).
# ---------------------------------------------------------------------------

from wara.gui_beta.fitting import PEAK_SHAPES, BKG_ARGS


class TestProfilesAndModels:
    def test_all_shapes_offered(self, window):
        items = [window.shape.itemText(i) for i in range(window.shape.count())]
        for name in ("Gaussian", "Voigt", "Pseudo-Voigt", "Skewed Voigt",
                     "EMG (low-E tail)", "Doniach", "Hypermet"):
            assert name in items

    @pytest.mark.parametrize("shape", list(PEAK_SHAPES))
    def test_every_profile_fits(self, hpge_window, shape):
        hpge_window.show()
        hpge_window.shape.setCurrentText(shape)
        hpge_window.bkg.setCurrentText("Polynomial")
        hpge_window.set_roi(6100, 6150)   # tailed 6129 keV peak
        assert hpge_window.table.rowCount() == 1
        area = float(hpge_window.table.item(0, 1).text().split("±")[0])
        assert area > 0

    def test_hypermet_beats_gaussian_chi2(self, hpge_window):
        """Hypermet should beat Gaussian on the tailed peak (reduced χ²)."""

        def redchi(shape):
            hpge_window.shape.setCurrentText(shape)
            hpge_window.bkg.setCurrentText("Polynomial")
            hpge_window.set_roi(6100, 6150)
            return hpge_window.last_fit.fit_result.redchi

        hpge_window.show()
        assert redchi("Hypermet") < redchi("Gaussian")


class TestStepBackgroundInGui:
    def test_gaussian_step_dispatches(self, hpge_window):
        hpge_window.show()
        hpge_window.shape.setCurrentText("Gaussian")
        hpge_window.bkg.setCurrentText("Step (sharp)")
        hpge_window.set_roi(6100, 6150)
        assert hpge_window.table.rowCount() == 1
        assert "bkg_step_amplitude" in hpge_window.last_fit.fit_result.best_values

    def test_hypermet_step_dispatches(self, hpge_window):
        hpge_window.show()
        hpge_window.shape.setCurrentText("Hypermet")
        hpge_window.bkg.setCurrentText("Step (smooth)")
        hpge_window.set_roi(6100, 6150)
        assert hpge_window.table.rowCount() == 1

    def test_unsupported_combo_messages(self, hpge_window):
        hpge_window.show()
        hpge_window.shape.setCurrentText("Voigt")
        hpge_window.bkg.setCurrentText("Step (sharp)")
        hpge_window.set_roi(6100, 6150)
        assert hpge_window.table.rowCount() == 0
        assert "Step background supports" in hpge_window.lbl_status.text()

    def test_shared_sigma_disabled_for_step(self, hpge_window):
        hpge_window.show()
        hpge_window.bkg.setCurrentText("Step (sharp)")
        assert not hpge_window.cb_shared.isEnabled()
        hpge_window.bkg.setCurrentText("Polynomial")
        assert hpge_window.cb_shared.isEnabled()


class TestEnrichedTable:
    def test_gaussian_ratio_and_asymmetry(self, window):
        """A Gaussian fit reads ~1.823 for the tailing ratio and ~0 asymmetry."""
        window.show()
        window.shape.setCurrentText("Gaussian")
        window.bkg.setCurrentText("Polynomial")
        window.set_roi(1106, 1214)
        ratio = float(window.table.item(0, 4).text())
        asym = float(window.table.item(0, 5).text())
        assert ratio == pytest.approx(1.82, abs=0.03)
        assert asym == pytest.approx(0.0, abs=0.03)

    def test_fit_stored_with_chi2(self, window):
        window.show()
        window.shape.setCurrentText("Voigt")
        window.set_roi(1106, 1214)
        assert window.last_fit is not None
        assert window.last_fit.fit_result.redchi > 0

    def test_area_method_resets_columns(self, window):
        window.show()
        window.set_roi(1106, 1214)            # peak fit -> 6 columns
        assert window.table.columnCount() == 6
        window.method.setCurrentIndex(1)       # net area -> 3 columns
        assert window.table.columnCount() == 3
        window.method.setCurrentIndex(0)       # back -> 6 columns
        assert window.table.columnCount() == 6
