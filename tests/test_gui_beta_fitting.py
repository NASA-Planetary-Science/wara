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
        assert window._slider_labels["lo"].text().strip() == "Peak start"
        assert window._slider_labels["hi"].text().strip() == "Peak end"

    def test_switch_back_restores_peakfit_controls(self, window):
        window.show()
        window.set_roi(1106, 1214)
        window.method.setCurrentIndex(1)
        window.method.setCurrentIndex(0)   # back to Peak fit
        assert window.peakfit_controls.isVisible()
        assert window._slider_labels["lo"].text().strip() == "Fit low"


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
        headers = [window.table.horizontalHeaderItem(c).text() for c in range(3)]
        assert headers == ["Centroid (keV)", "Area", "FWHM (keV)"]

    def test_area_is_raw_counts_despite_livetime(self, hpge_search, hpge_window):
        """Regression: PeakFit divides area by livetime (counts/sec); the GUI
        must display raw counts so it matches the plot and the net-area method."""
        lt = hpge_search.spectrum.livetime
        assert lt and not hpge_search.spectrum.cps   # this file does have one
        pk = float(hpge_search.spectrum.energies[hpge_search.peaks_idx[5]])
        hpge_window.show()
        hpge_window.set_roi(pk - 8, pk + 8)
        # The GUI area = backend area (a rate) * livetime = raw counts.
        gui_area = float(hpge_window.table.item(0, 1).text().split("±")[0])
        from wara.peakfit import PeakFit
        backend_area = PeakFit(hpge_search, [pk - 8, pk + 8],
                               bkg="poly1").peak_info[0]["area"]
        assert gui_area == pytest.approx(backend_area * lt, rel=1e-6)
        assert gui_area > 100   # plainly counts, not a ~1-count/sec rate
