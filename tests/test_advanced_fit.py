"""
Pytest tests for wara.advanced_fit.

Covers:
  * PeakAreaLinearBkg — net peak area above a linear background, using both the
    point/range linear fit (``calculate_peak_area``) and the per-side average
    method (``calculate_peak_area_avg``).
  * ContinuumFit — polynomial baseline fit that masks out detected peaks.

These tests mirror the example scripts and run on the real example spectra
(test_data_cebr_cal.csv / test_data_cebr.csv for the area methods, and
test_data_hpge_NH3.txt for the continuum fit) so the fitting paths are
exercised on actual detector data.
"""

from pathlib import Path

import pytest

from wara import file_reader
from wara import peaksearch as ps
from wara import advanced_fit as adv


# ---------------------------------------------------------------------------
# Example data (same files used by the example scripts)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "examples" / "data"
CSV_WITH_CAL = str(DATA_DIR / "test_data_cebr_cal.csv")   # energy-calibrated
CSV_NO_CAL = str(DATA_DIR / "test_data_cebr.csv")         # channels only
TXT_HPGE = str(DATA_DIR / "test_data_hpge_NH3.txt")       # peak-rich HPGe

# Background regions used by example_advfit_area_bkg.py (energy keV / channel no.)
X1_RANGE = [1060, 1105]
X2_RANGE = [1214, 1260]


@pytest.fixture
def spec_cal():
    return file_reader.read_csv(CSV_WITH_CAL)


@pytest.fixture
def spec_chan():
    return file_reader.read_csv(CSV_NO_CAL)


# ---------------------------------------------------------------------------
# Construction / input validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_non_spectrum_raises(self):
        with pytest.raises(Exception, match="must be a Spectrum"):
            adv.PeakAreaLinearBkg([1, 2, 3], x1=1060, x2=1260)

    def test_results_unset_before_calculation(self, spec_cal):
        alb = adv.PeakAreaLinearBkg(spec_cal, x1=1060, x2=1260)
        assert alb.prange is None
        assert alb.A == 0 and alb.B == 0

    def test_roi_bounds_stored(self, spec_cal):
        # x1/x2 are energies for a calibrated spectrum; stored as channel idx.
        alb = adv.PeakAreaLinearBkg(spec_cal, x1=1060, x2=1260)
        assert alb._ch_roi_l < alb._ch_roi_r


# ---------------------------------------------------------------------------
# _x_to_ch helper
# ---------------------------------------------------------------------------

class TestXToCh:
    def test_channel_identity(self, spec_chan):
        assert adv.PeakAreaLinearBkg._x_to_ch(1267, spec_chan) == 1267

    def test_energy_maps_to_first_channel_at_or_above(self, spec_cal):
        ch = adv.PeakAreaLinearBkg._x_to_ch(1060, spec_cal)
        # energy at that channel must be >= requested, and one channel lower < it
        assert spec_cal.energies[ch] >= 1060
        assert spec_cal.energies[ch - 1] < 1060


# ---------------------------------------------------------------------------
# calculate_peak_area — linear fit through background points
# Reference values produced from the example spectra (see example script).
# ---------------------------------------------------------------------------

class TestCalculatePeakAreaCalibrated:
    @pytest.fixture
    def fit(self, spec_cal):
        alb = adv.PeakAreaLinearBkg(spec_cal, x1=1060, x2=1260)
        alb.calculate_peak_area(x1=X1_RANGE, x2=X2_RANGE)
        return alb

    def test_net_area(self, fit):
        assert fit.A == pytest.approx(9626.36, rel=1e-3)

    def test_background_area(self, fit):
        assert fit.B == pytest.approx(31309.64, rel=1e-3)

    def test_total_equals_net_plus_bkg(self, fit):
        assert fit.A + fit.B == pytest.approx(fit.yr.sum())

    def test_inner_edges_in_channels(self, fit):
        assert fit.pchrange == [1269, 1372]

    def test_peak_region_energy_bounds(self, fit):
        assert fit.prange[0] == pytest.approx(1106.01, rel=1e-3)
        assert fit.prange[1] == pytest.approx(1214.24, rel=1e-3)

    def test_errors_positive(self, fit):
        assert fit.sigA > 0
        assert fit.sigB > 0

    def test_net_area_is_significant(self, fit):
        # Net peak area should be many sigma above zero on this real peak.
        assert fit.A / fit.sigA > 10


class TestCalculatePeakAreaChannels:
    @pytest.fixture
    def fit(self, spec_chan):
        alb = adv.PeakAreaLinearBkg(spec_chan, x1=1060, x2=1260)
        alb.calculate_peak_area(x1=X1_RANGE, x2=X2_RANGE)
        return alb

    def test_net_area(self, fit):
        assert fit.A == pytest.approx(1827.43, rel=1e-3)

    def test_inner_edges_are_input_channels(self, fit):
        # No calibration -> inner edges are just the input channel numbers.
        assert fit.pchrange == [1105, 1214]

    def test_total_equals_net_plus_bkg(self, fit):
        assert fit.A + fit.B == pytest.approx(fit.yr.sum())


# ---------------------------------------------------------------------------
# calculate_peak_area_avg — per-side average background
# ---------------------------------------------------------------------------

class TestCalculatePeakAreaAvg:
    def test_net_area_matches_linfit_closely(self, spec_cal):
        alb = adv.PeakAreaLinearBkg(spec_cal, x1=1060, x2=1260)
        alb.calculate_peak_area_avg(x1=X1_RANGE, x2=X2_RANGE)
        # Average-background method agrees with the linear fit to ~0.1%.
        assert alb.A == pytest.approx(9632.0, rel=2e-3)

    def test_scalar_input_raises(self, spec_cal):
        alb = adv.PeakAreaLinearBkg(spec_cal, x1=1060, x2=1260)
        with pytest.raises(ValueError, match="2-element lists"):
            alb.calculate_peak_area_avg(x1=1080, x2=1240)

    def test_large_gap_collapses_region(self, spec_cal):
        alb = adv.PeakAreaLinearBkg(spec_cal, x1=1060, x2=1260)
        with pytest.raises(ValueError, match="Gap is too large"):
            alb.calculate_peak_area_avg(x1=X1_RANGE, x2=X2_RANGE, gap=1000)

    def test_gap_shrinks_peak_region(self, spec_cal):
        alb_nogap = adv.PeakAreaLinearBkg(spec_cal, x1=1060, x2=1260)
        alb_nogap.calculate_peak_area_avg(x1=X1_RANGE, x2=X2_RANGE)
        alb_gap = adv.PeakAreaLinearBkg(spec_cal, x1=1060, x2=1260)
        alb_gap.calculate_peak_area_avg(x1=X1_RANGE, x2=X2_RANGE, gap=10)
        # A positive gap pulls both inner edges inward -> narrower peak region.
        assert alb_gap.pchrange[0] > alb_nogap.pchrange[0]
        assert alb_gap.pchrange[1] < alb_nogap.pchrange[1]


# ---------------------------------------------------------------------------
# ContinuumFit — polynomial baseline that masks out detected peaks.
# Mirrors examples/peakfit/example_continuum_fit.py (real HPGe spectrum).
# ---------------------------------------------------------------------------

CONT_XRANGE = [180, 540]


@pytest.fixture(scope="module")
def hpge_search():
    spect = file_reader.read_txt(TXT_HPGE)
    return ps.PeakSearch(spect, ref_x=420, ref_fwhm=3, min_snr=15, method="fast")


class TestContinuumFitValidation:
    def test_non_peaksearch_raises(self, spec_cal):
        with pytest.raises(TypeError, match="must be a PeakSearch"):
            adv.ContinuumFit(spec_cal, xrange=CONT_XRANGE)

    def test_degree_too_high_raises(self, hpge_search):
        with pytest.raises(ValueError, match=r"degree must be in \[0, 7\]"):
            adv.ContinuumFit(hpge_search, xrange=CONT_XRANGE, degree=8)

    def test_degree_negative_raises(self, hpge_search):
        with pytest.raises(ValueError, match=r"degree must be in \[0, 7\]"):
            adv.ContinuumFit(hpge_search, xrange=CONT_XRANGE, degree=-1)

    def test_too_few_points_after_masking_raises(self, hpge_search):
        # A 2-keV window can't supply the 8 points a degree-7 polynomial needs.
        with pytest.raises(ValueError, match="continuum points available"):
            adv.ContinuumFit(hpge_search, xrange=[200, 202], degree=7)


class TestContinuumFit:
    @pytest.fixture
    def cont(self, hpge_search):
        return adv.ContinuumFit(hpge_search, xrange=CONT_XRANGE,
                                degree=3, mask_fwhm=3.0)

    def test_search_found_peaks(self, hpge_search):
        assert len(hpge_search.peaks_idx) > 0

    def test_degree_stored(self, cont):
        assert cont.degree == 3

    def test_masking_removes_peak_channels(self, cont):
        in_range = (cont.x >= CONT_XRANGE[0]) & (cont.x <= CONT_XRANGE[1])
        kept = int(cont.continuum_mask.sum())
        # Some channels are kept, some masked out under the peaks.
        assert 0 < kept < int(in_range.sum())

    def test_fit_result_finite(self, cont):
        assert cont.fit_result is not None
        assert cont.fit_result.redchi > 0

    def test_evaluate_spans_range(self, cont):
        in_range = (cont.x >= CONT_XRANGE[0]) & (cont.x <= CONT_XRANGE[1])
        y = cont.evaluate()
        assert len(y) == int(in_range.sum())

    def test_evaluate_at_explicit_x(self, cont):
        val = cont.evaluate([CONT_XRANGE[0]])
        assert val.shape == (1,)

    def test_subtract_returns_paired_arrays(self, cont):
        x_sub, y_sub = cont.subtract()
        assert len(x_sub) == len(y_sub) > 0

    def test_subtract_is_counts_minus_continuum(self, cont):
        import numpy as np
        x_sub, y_sub = cont.subtract()
        resid = y_sub + cont.evaluate(x_sub)
        in_range = (cont.x >= CONT_XRANGE[0]) & (cont.x <= CONT_XRANGE[1])
        counts = cont.search.spectrum.counts[in_range]
        assert np.allclose(resid, counts)

    def test_default_xrange_spans_full_spectrum(self, hpge_search):
        cont = adv.ContinuumFit(hpge_search, degree=3)
        assert cont.xrange[0] == pytest.approx(float(hpge_search.spectrum.energies[0]))
        assert cont.xrange[1] == pytest.approx(float(hpge_search.spectrum.energies[-1]))

    def test_higher_degree_lowers_or_equals_redchi(self, hpge_search):
        # More flexible polynomial should not fit worse on the kept points.
        c1 = adv.ContinuumFit(hpge_search, xrange=CONT_XRANGE, degree=1)
        c5 = adv.ContinuumFit(hpge_search, xrange=CONT_XRANGE, degree=5)
        assert c5.fit_result.redchi <= c1.fit_result.redchi
