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
        # A calibrated spectrum also gets a trailing "ID" (isotope-ID) column.
        assert headers == ["Centroid (keV)", "Area", "FWHM (keV)",
                           "FWTM (keV)", "FWTM/FWHM", "Asym", "ID"]

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
        window.set_roi(1106, 1214)            # peak fit -> 6 cols + ID (calibrated)
        assert window.table.columnCount() == 7
        window.method.setCurrentIndex(1)       # net area -> 3 columns
        assert window.table.columnCount() == 3
        window.method.setCurrentIndex(0)       # back -> 6 cols + ID
        assert window.table.columnCount() == 7


# ---------------------------------------------------------------------------
# Fit-details report rendering (_report_to_html / _net_area_map)
# ---------------------------------------------------------------------------

class TestFitDetailsReport:
    # A noted header ("[[Correlations]] (unreported ...)") must still become a
    # title, and the model expression must not flip colour at its inner '='.
    SAMPLE = (
        "[[Model]]\n"
        "    (Model(gaussian, prefix='g1_') + Model(polynomial, prefix='p_'))\n"
        "[[Variables]]\n"
        "    g1_amplitude:  142870.937 +/- 302.4 (0.21%) (init = 21083.7)\n"
        "    g1_center:     1330.0 +/- 0.01 (init = 1330)\n"
        "[[Correlations]] (unreported correlations are < 0.100)\n"
        "    C(g1_amplitude, g1_sigma) = +0.3220\n"
    )

    def test_noted_header_becomes_title(self):
        html = FitWindow._report_to_html(self.SAMPLE)
        # The Correlations line, despite its trailing note, is an <h3> title and
        # the note rides along — it does not leak into the body as a data row.
        assert "<h3" in html and ">Correlations" in html
        assert "unreported correlations" in html
        assert "[[Correlations]]" not in html

    def test_model_line_single_colour(self):
        html = FitWindow._report_to_html(self.SAMPLE)
        # The model expression (which contains prefix='g1_') is one colour span,
        # so the colour cannot change midway through the line.
        body = html.split(">Variables")[0]
        assert body.count("color:#c4b5ff") == 1

    def test_amplitude_annotated_with_net_area(self, window):
        window.show()
        window.shape.setCurrentText("Gaussian")
        window.set_roi(1106, 1214)
        fit = window.last_fit
        html = FitWindow._report_to_html(
            fit.fit_result.fit_report(), FitWindow._net_area_map(fit))
        assert "net area" in html and "counts" in html
        # The annotated value matches the net-count area in the results table.
        area = fit.summary().iloc[0]["area"]
        assert f"net area = {area:,.1f}" in html

    def test_net_area_map_matches_summary(self, window):
        window.show()
        window.set_roi(1106, 1214)
        fit = window.last_fit
        amap = FitWindow._net_area_map(fit)
        for i, row in fit.summary().iterrows():
            assert amap[f"g{i + 1}"][0] == pytest.approx(row["area"])


# ---------------------------------------------------------------------------
# Send-centroids dialog: per-row database energy picker
# ---------------------------------------------------------------------------

class TestCentroidEnergyDialog:
    def _dialog(self, qapp, **kw):
        from wara.gui_beta.fitting import CentroidEnergyDialog
        return CentroidEnergyDialog(None, [500.0, 1200.0], **kw)

    def test_has_per_row_database_buttons(self, qapp):
        dlg = self._dialog(qapp)
        assert dlg.table.columnCount() == 3
        assert dlg.table.cellWidget(0, 2) is not None
        assert dlg.table.cellWidget(1, 2) is not None

    def test_pick_energy_fills_cell_in_selected_units(self, qapp, monkeypatch):
        import wara.gui_beta.nuclear as nuc
        from PyQt5.QtWidgets import QDialog

        class FakePicker:
            def __init__(self, parent=None, element=""):
                pass

            def exec_(self):
                return QDialog.Accepted

            def selected_energy(self):
                return 1332.5            # keV

        monkeypatch.setattr(nuc, "NuclearLinePicker", FakePicker)

        dlg = self._dialog(qapp, default_units="MeV")
        dlg._pick_energy(0)
        assert dlg.table.item(0, 1).text() == "1.3325"     # converted keV -> MeV
        # The pair is then reported with that energy; the cell stays editable.
        assert dlg.pairs()[0] == (500.0, "1.3325")

    def test_pick_energy_cancel_leaves_cell_blank(self, qapp, monkeypatch):
        import wara.gui_beta.nuclear as nuc
        from PyQt5.QtWidgets import QDialog

        class FakePicker:
            def __init__(self, parent=None, element=""):
                pass

            def exec_(self):
                return QDialog.Rejected

            def selected_energy(self):
                return 1332.5

        monkeypatch.setattr(nuc, "NuclearLinePicker", FakePicker)

        dlg = self._dialog(qapp)
        dlg._pick_energy(0)
        assert dlg.table.item(0, 1).text() == ""


class TestNuclearLinePickerMemory:
    def test_remembers_last_search(self, qapp):
        import wara.gui_beta.nuclear as nuc
        # Reset to a known baseline so the test is order-independent.
        nuc._LAST_LINE_SEARCH.update(
            database="Common lab sources", element="", energy="", range="")
        p1 = nuc.NuclearLinePicker(None)
        p1.db_combo.setCurrentText("TALYS 14 MeV")
        p1.ed_element.setText("56Fe")
        p1._search()
        p2 = nuc.NuclearLinePicker(None)
        assert p2.db_combo.currentText() == "TALYS 14 MeV"
        assert p2.ed_element.text() == "56Fe"
        assert p2.table.model().rowCount() > 0      # opens pre-populated


class TestSendButtonStates:
    """The 'Send to Calibration / Efficiency' buttons flip to a bright '✓ Sent'
    look on a successful send, and reset when a new fit is computed."""

    def test_send_to_efficiency_marks_sent(self, window):
        window.show()
        window.set_roi(1106, 1214)
        assert window.btn_to_eff.objectName() == "fit_btn"
        window._emit_to_efficiency()
        assert window.btn_to_eff.objectName() == "sent_btn"
        assert "Sent" in window.btn_to_eff.text()

    def test_new_fit_resets_sent_state(self, window):
        window.show()
        window.set_roi(1106, 1214)
        window._emit_to_efficiency()
        assert window.btn_to_eff.objectName() == "sent_btn"
        window.set_roi(1100, 1220)                       # a new fit
        assert window.btn_to_eff.objectName() == "fit_btn"
        assert window.btn_to_eff.text() == "📈  Send to Efficiency"

    def test_mark_and_reset_calibration_button(self, window):
        window.show()
        window.set_roi(1106, 1214)
        window._mark_sent(window.btn_to_cal, "✓  Sent to Calibration")
        assert window.btn_to_cal.objectName() == "sent_btn"
        window._reset_send_buttons()
        assert window.btn_to_cal.objectName() == "yellow_btn"
        assert window.btn_to_cal.text() == "📐  Send centroids to Calibration"


class TestIsotopeIdColumn:
    """The Drag-and-Fit results table offers per-line isotope ID (a 🧬 button)
    when the spectrum is calibrated, identifying the precise fitted centroid."""

    def test_id_column_present_when_calibrated(self, window):
        window.show()
        window.set_roi(1106, 1214)
        headers = [window.table.horizontalHeaderItem(i).text()
                   for i in range(window.table.columnCount())]
        assert headers[-1] == "ID"
        btn = window.table.cellWidget(0, window.table.columnCount() - 1)
        assert btn is not None and btn.text() == "⚛"

    def test_clicking_id_button_identifies_centroid(self, window):
        window.show()
        window.set_roi(1106, 1214)
        col = window.table.columnCount() - 1
        btn = window.table.cellWidget(0, col)
        centroid = float(window.last_fit.peak_info[0]["mean"])
        window._identify_centroid(centroid, btn)
        assert "keV" in btn.toolTip()                      # colored HTML popup text
        assert round(centroid, 3) in window._iso_cache     # cached for re-hover

    def test_no_id_column_when_uncalibrated(self, qapp):
        import numpy as np
        from wara import spectrum as sp
        from wara.peaksearch import PeakSearch
        ch = np.arange(4096)
        counts = np.full_like(ch, 5.0, dtype=float)
        counts += 6000.0 * np.exp(-0.5 * ((ch - 2000) / 3.0) ** 2)
        spect = sp.Spectrum(counts=counts)                 # channel axis, no energy
        w = FitWindow(None, PeakSearch(spect, ref_x=2000, ref_fwhm=3,
                                       fwhm_at_0=1.0, min_snr=4))
        w.set_roi(1960, 2040)
        headers = [w.table.horizontalHeaderItem(i).text()
                   for i in range(w.table.columnCount())]
        assert "ID" not in headers
        w.close()
