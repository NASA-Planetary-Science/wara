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


# ---------------------------------------------------------------------------
# GaussStepFit — Gaussian peak on a step background.
# Mirrors examples/peakfit/example_gauss_step_fit.py. The isolated ~1158 keV
# peak in the calibrated CeBr spectrum sits on a clear Compton step, so the
# step background gives an excellent fit (redchi < 1). The default step is a
# sharp Heaviside discontinuity; step="smooth" uses an erfc transition.
# ---------------------------------------------------------------------------

STEP_XRANGE = [1070, 1260]


@pytest.fixture
def cebr_search(spec_cal):
    return ps.PeakSearch(spec_cal, ref_x=1220, ref_fwhm=31,
                         fwhm_at_0=1.0, min_snr=5)


class TestSharpStepBackgroundFunction:
    def test_steps_down_across_center(self):
        # Below the centroid -> offset + amplitude; above -> offset.
        left = adv.sharp_step_background(-5.0, step_amplitude=100.0,
                                         step_center=0.0, offset=20.0)
        right = adv.sharp_step_background(5.0, step_amplitude=100.0,
                                          step_center=0.0, offset=20.0)
        assert left == pytest.approx(120.0)
        assert right == pytest.approx(20.0)

    def test_value_at_center_is_halfway(self):
        mid = adv.sharp_step_background(0.0, step_amplitude=100.0,
                                        step_center=0.0, offset=20.0)
        assert mid == pytest.approx(70.0)

    def test_transition_is_discontinuous(self):
        import numpy as np
        # No width parameter: just below vs just above the centroid jumps
        # the full step height (a hard discontinuity).
        eps = 1e-9
        below = adv.sharp_step_background(-eps, step_amplitude=100.0,
                                          step_center=0.0, offset=20.0)
        above = adv.sharp_step_background(eps, step_amplitude=100.0,
                                          step_center=0.0, offset=20.0)
        assert np.isclose(below - above, 100.0)


class TestSmoothStepBackgroundFunction:
    def test_steps_down_across_center(self):
        # Far left -> offset + amplitude; far right -> offset.
        left = adv.step_background(-1e6, step_amplitude=100.0,
                                   step_center=0.0, step_sigma=1.0, offset=20.0)
        right = adv.step_background(1e6, step_amplitude=100.0,
                                    step_center=0.0, step_sigma=1.0, offset=20.0)
        assert left == pytest.approx(120.0)
        assert right == pytest.approx(20.0)

    def test_value_at_center_is_halfway(self):
        mid = adv.step_background(0.0, step_amplitude=100.0,
                                  step_center=0.0, step_sigma=1.0, offset=20.0)
        assert mid == pytest.approx(70.0)


class TestGaussStepFit:
    """Default (sharp) step background."""

    @pytest.fixture
    def fit(self, cebr_search):
        return adv.GaussStepFit(cebr_search, STEP_XRANGE)

    def test_sharp_is_default(self, fit):
        assert fit.step == "sharp"
        assert fit.bkg == "step"
        assert fit.bkg_key == "bkg_"

    def test_no_step_sigma_parameter(self, fit):
        # The sharp step has no width — only height, location, offset.
        assert "bkg_step_sigma" not in fit.fit_result.params

    def test_good_fit_quality(self, fit):
        # A linear background can't track the Compton step; the step model
        # fits this peak essentially at the noise level.
        assert fit.fit_result.redchi == pytest.approx(0.8886, rel=1e-3)

    def test_single_peak_found(self, fit):
        assert len(fit.peak_info) == 1

    def test_peak_centroid(self, fit):
        assert fit.peak_info[0]["mean"] == pytest.approx(1158.85, rel=1e-4)

    def test_peak_area_and_fwhm(self, fit):
        assert fit.peak_info[0]["area"] == pytest.approx(9951.29, rel=1e-3)
        assert fit.peak_info[0]["fwhm"] == pytest.approx(44.89, rel=1e-3)

    def test_step_parameters(self, fit):
        bv = fit.fit_result.best_values
        assert bv["bkg_step_amplitude"] == pytest.approx(27.44, rel=1e-2)
        assert bv["bkg_offset"] == pytest.approx(284.20, rel=1e-3)
        # Positive step height -> higher on the low-energy side, as expected.
        assert bv["bkg_step_amplitude"] > 0

    def test_step_tied_to_peak(self, fit):
        bv = fit.fit_result.best_values
        assert bv["bkg_step_center"] == bv["g1_center"]

    def test_summary_one_row(self, fit):
        df = fit.summary()
        assert len(df) == 1
        assert {"mean", "area", "fwhm"} <= set(df.columns)

    def test_errors_finite_and_positive(self, fit):
        import numpy as np
        err = fit.peak_err[0]
        for key in ("mean_err", "area_err", "fwhm_err"):
            assert np.isfinite(err[key]) and err[key] > 0

    def test_shared_sigma_not_supported(self, cebr_search):
        with pytest.raises(NotImplementedError, match="shared_sigma"):
            adv.GaussStepFit(cebr_search, STEP_XRANGE, shared_sigma=True)

    def test_unknown_step_raises(self, cebr_search):
        with pytest.raises(ValueError, match="Unknown step"):
            adv.GaussStepFit(cebr_search, STEP_XRANGE, step="wobble")

    def test_hints_override_applied(self, cebr_search):
        # Pin the centroid and confirm the fit honours it.
        fit = adv.GaussStepFit(
            cebr_search, STEP_XRANGE,
            hints={"g1_center": {"value": 1160.0, "vary": False}},
        )
        assert fit.fit_result.best_values["g1_center"] == pytest.approx(1160.0)

    def test_plot_runs_headless(self, fit):
        import matplotlib
        matplotlib.use("Agg")
        fit.plot()

    def test_plotted_peak_component_is_smooth(self, fit):
        # Regression: the drawn peak must not inherit the step's
        # discontinuity. The component curve is a smooth Gaussian lifted
        # onto the background, so its largest jump between adjacent channels
        # is far below the step height.
        import numpy as np
        comps = fit.fit_result.eval_components()
        curve = fit._component_curve(comps, 0)
        max_jump = np.abs(np.diff(curve)).max()
        step_height = fit.fit_result.best_values["bkg_step_amplitude"]
        assert max_jump < step_height

    def test_json_roundtrip(self, fit, cebr_search, tmp_path):
        path = tmp_path / "gauss_step.json"
        fit.save_json(path)
        restored = adv.GaussStepFit.load_json(path, cebr_search)
        assert restored.step == "sharp"
        assert restored.bkg == "step"
        assert restored.fit_result.best_values["g1_center"] == pytest.approx(
            fit.fit_result.best_values["g1_center"]
        )


class TestGaussStepFitSmooth:
    """Optional smoothed (erfc) step background."""

    @pytest.fixture
    def fit(self, cebr_search):
        return adv.GaussStepFit(cebr_search, STEP_XRANGE, step="smooth")

    def test_smooth_selected(self, fit):
        assert fit.step == "smooth"
        assert fit.bkg == "step-smooth"

    def test_has_step_sigma_tied_to_peak(self, fit):
        bv = fit.fit_result.best_values
        assert "bkg_step_sigma" in fit.fit_result.params
        assert bv["bkg_step_center"] == bv["g1_center"]
        assert bv["bkg_step_sigma"] == bv["g1_sigma"]

    def test_good_fit_quality(self, fit):
        assert fit.fit_result.redchi == pytest.approx(0.9256, rel=1e-3)

    def test_peak_centroid_and_area(self, fit):
        assert fit.peak_info[0]["mean"] == pytest.approx(1157.99, rel=1e-4)
        assert fit.peak_info[0]["area"] == pytest.approx(9952.06, rel=1e-3)

    def test_json_roundtrip_preserves_smooth(self, fit, cebr_search, tmp_path):
        path = tmp_path / "gauss_step_smooth.json"
        fit.save_json(path)
        restored = adv.GaussStepFit.load_json(path, cebr_search)
        assert restored.step == "smooth"
        assert "bkg_step_sigma" in restored.fit_result.params


# ---------------------------------------------------------------------------
# Peak-shape diagnostics — peak_shape_metrics / shape_metrics / shape_summary.
# Widths and tailing measured numerically from the fitted profile, valid for
# any line shape (fixes the sigma*2.3548 FWHM that only holds for a Gaussian).
# ---------------------------------------------------------------------------

import numpy as np


class TestPeakShapeMetricsFunction:
    def test_gaussian_constant(self):
        # FWTM/FWHM of a pure Gaussian is sqrt(ln10/ln2).
        assert adv.GAUSS_FWTM_FWHM == pytest.approx(1.822615, rel=1e-5)

    def test_pure_gaussian_widths(self):
        sigma, mu = 5.0, 100.0
        x = np.linspace(40, 160, 6000)
        y = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        m = adv.peak_shape_metrics(x, y)
        assert m["fwhm"] == pytest.approx(2.354820 * sigma, rel=1e-3)
        assert m["fwtm"] == pytest.approx(4.291932 * sigma, rel=1e-3)
        assert m["fwtm_fwhm"] == pytest.approx(adv.GAUSS_FWTM_FWHM, rel=1e-3)
        assert m["asymmetry"] == pytest.approx(0.0, abs=1e-3)

    def test_low_energy_tail_positive_asymmetry(self):
        # Build a peak that is wider on the low-energy (left) side.
        mu, sig = 100.0, 5.0
        x = np.linspace(40, 160, 6000)
        left = np.exp(-((x - mu) ** 2) / (2 * (2.5 * sig) ** 2))  # broad left
        right = np.exp(-((x - mu) ** 2) / (2 * sig**2))           # narrow right
        y = np.where(x < mu, left, right)
        m = adv.peak_shape_metrics(x, y)
        assert m["hwhm_left"] > m["hwhm_right"]
        assert m["asymmetry"] > 0          # low-energy tail -> positive
        assert m["fwtm_fwhm"] > adv.GAUSS_FWTM_FWHM

    def test_nan_when_apex_on_edge(self):
        x = np.linspace(0, 10, 100)
        y = np.linspace(1, 5, 100)  # monotonic, max at the right edge
        m = adv.peak_shape_metrics(x, y)
        assert np.isnan(m["fwhm"])

    def test_nan_when_all_zero(self):
        x = np.linspace(0, 10, 100)
        m = adv.peak_shape_metrics(x, np.zeros_like(x))
        assert all(np.isnan(v) for v in m.values())


class TestShapeMetricsOnFits:
    @pytest.fixture
    def gauss_fit(self, cebr_search):
        from wara import peakfit as pf
        return pf.PeakFit(cebr_search, STEP_XRANGE, bkg="linear")

    def test_one_dict_per_peak(self, gauss_fit):
        mets = adv.shape_metrics(gauss_fit)
        assert len(mets) == len(gauss_fit.peak_info) == 1

    def test_gaussian_fit_ratio_is_reference(self, gauss_fit):
        m = adv.shape_metrics(gauss_fit)[0]
        assert m["fwtm_fwhm"] == pytest.approx(adv.GAUSS_FWTM_FWHM, rel=1e-3)

    def test_measured_fwhm_matches_sigma_fwhm_for_gaussian(self, gauss_fit):
        # For a true Gaussian the numeric FWHM should match sigma*2.3548.
        m = adv.shape_metrics(gauss_fit)[0]
        assert m["fwhm"] == pytest.approx(gauss_fit.peak_info[0]["fwhm"], rel=1e-3)

    def test_shape_summary_adds_columns(self, gauss_fit):
        df = adv.shape_summary(gauss_fit)
        for col in ("fwhm_meas", "fwtm", "fwtm_fwhm", "asymmetry"):
            assert col in df.columns
        # base summary columns are preserved
        assert {"mean", "area", "fwhm"} <= set(df.columns)
        assert len(df) == 1

    def test_shape_summary_works_for_step_fit(self, cebr_search):
        # Subclass (step background) peaks flow through unchanged.
        fit = adv.GaussStepFit(cebr_search, STEP_XRANGE)
        df = adv.shape_summary(fit)
        assert df["fwtm_fwhm"].iloc[0] == pytest.approx(
            adv.GAUSS_FWTM_FWHM, rel=5e-3
        )


# ---------------------------------------------------------------------------
# MultiProfilePeakFit tail profiles — EMG / SkewedVoigt (Tier A).
# Exercised on the HPGe 399.5 keV peak.
# ---------------------------------------------------------------------------

HPGE_PEAK_XRANGE = [390, 410]


class TestTailProfiles:
    def test_emg_and_skewedvoigt_registered(self):
        assert "emg" in adv._PROFILE_MODELS
        assert "skewedvoigt" in adv._PROFILE_MODELS

    def test_unknown_profile_raises(self, hpge_search):
        with pytest.raises(ValueError, match="Unknown profile"):
            adv.MultiProfilePeakFit(hpge_search, HPGE_PEAK_XRANGE, profile="nope")

    def test_emg_fit_converges(self, hpge_search):
        fit = adv.MultiProfilePeakFit(
            hpge_search, HPGE_PEAK_XRANGE, bkg="linear", profile="emg"
        )
        assert fit.fit_result.success
        assert fit.peak_info[0]["area"] > 0
        # centroid lands on the ~399.5 keV line
        assert fit.peak_info[0]["mean"] == pytest.approx(399.5, abs=1.0)

    def test_emg_tail_is_low_energy(self, hpge_search):
        # Our EMG is the *left*-tailed variant (lmfit's tails high), so a
        # genuinely tailed peak shows a positive measured asymmetry.
        fit = adv.MultiProfilePeakFit(
            hpge_search, HYPERMET_XRANGE, bkg="linear", profile="emg"
        )
        assert "g1_tau" in fit.fit_result.params
        assert fit.fit_result.best_values["g1_tau"] > 0
        assert adv.shape_metrics(fit)[0]["asymmetry"] > 0

    def test_emg_not_worse_than_gaussian(self, hpge_search):
        # On a tailed HPGe peak the EMG should fit at least as well as a
        # plain Gaussian (it adds a tail degree of freedom).
        g = adv.MultiProfilePeakFit(
            hpge_search, HPGE_PEAK_XRANGE, bkg="linear", profile="gauss"
        )
        emg = adv.MultiProfilePeakFit(
            hpge_search, HPGE_PEAK_XRANGE, bkg="linear", profile="emg"
        )
        assert emg.fit_result.redchi <= g.fit_result.redchi * 1.05

    def test_skewedvoigt_fit_converges(self, hpge_search):
        fit = adv.MultiProfilePeakFit(
            hpge_search, HPGE_PEAK_XRANGE, bkg="linear", profile="skewedvoigt"
        )
        assert fit.fit_result.success
        assert fit.peak_info[0]["area"] > 0

    def test_shape_metrics_on_emg(self, hpge_search):
        fit = adv.MultiProfilePeakFit(
            hpge_search, HPGE_PEAK_XRANGE, bkg="linear", profile="emg"
        )
        m = adv.shape_metrics(fit)[0]
        assert np.isfinite(m["fwhm"]) and m["fwhm"] > 0
        assert np.isfinite(m["fwtm_fwhm"])


# ---------------------------------------------------------------------------
# Hypermet peak model (Tier B) — Gaussian core + low-energy exponential tail.
# The strong, visibly tailed 200.1 keV HPGe peak is the canonical case:
# a plain Gaussian gives redchi ~71, the Hypermet ~2.6.
# ---------------------------------------------------------------------------

HYPERMET_XRANGE = [196, 204]


class TestHypermetFunction:
    def test_unit_area_amplitude(self):
        x = np.linspace(50, 150, 200000)
        amp = 1000.0
        y = adv.hypermet(x, amplitude=amp, center=100.0, sigma=5.0,
                         tail_fraction=0.3, tail_tau=8.0)
        trapezoid = getattr(np, "trapezoid", np.trapz)
        assert trapezoid(y, x) == pytest.approx(amp, rel=1e-3)

    def test_zero_tail_reduces_to_gaussian(self):
        x = np.linspace(50, 150, 4000)
        g = adv.hypermet(x, 1000.0, 100.0, 5.0, tail_fraction=0.0, tail_tau=8.0)
        ref = 1000.0 * np.exp(-0.5 * ((x - 100.0) / 5.0) ** 2) / (
            5.0 * np.sqrt(2 * np.pi)
        )
        assert np.allclose(g, ref, rtol=1e-6)

    def test_tail_on_low_energy_side(self):
        x = np.linspace(50, 150, 4000)
        y = adv.hypermet(x, 1000.0, 100.0, 5.0, tail_fraction=0.3, tail_tau=8.0)
        dx = x[1] - x[0]
        mass_left = y[x < 100.0].sum() * dx
        mass_right = y[x >= 100.0].sum() * dx
        assert mass_left > mass_right  # low-energy tail carries more area

    def test_emg_left_unit_stable_small_tau(self):
        # erfcx formulation must stay finite where exp*erfc would overflow.
        x = np.linspace(-100, 100, 50000)
        y = adv._emg_left_unit(x, 0.0, 5.0, 0.3)
        assert np.all(np.isfinite(y))
        assert np.all(y >= 0)


class TestHypermetFit:
    @pytest.fixture
    def fit(self, hpge_search):
        return adv.HypermetFit(hpge_search, HYPERMET_XRANGE, bkg="linear")

    @pytest.fixture
    def gauss(self, hpge_search):
        return adv.MultiProfilePeakFit(
            hpge_search, HYPERMET_XRANGE, bkg="linear", profile="gauss"
        )

    def test_has_tail_parameters(self, fit):
        params = fit.fit_result.params
        assert "g1_tail_fraction" in params
        assert "g1_tail_tau" in params

    def test_beats_gaussian_on_tailed_peak(self, fit, gauss):
        # The whole point: on a tailed HPGe peak the Hypermet fits far better.
        assert fit.fit_result.redchi < 0.2 * gauss.fit_result.redchi

    def test_tail_fraction_significant(self, fit):
        tf = fit.fit_result.best_values["g1_tail_fraction"]
        assert 0.05 < tf < 0.95   # a real, bounded tail

    def test_area_is_total_and_matches_gaussian(self, fit, gauss):
        # Unit-area amplitude => total area; close to (slightly above) the
        # Gaussian area since it now also counts the tail.
        assert fit.peak_info[0]["area"] == pytest.approx(
            gauss.peak_info[0]["area"], rel=0.05
        )

    def test_measured_shape_shows_low_energy_tail(self, fit):
        m = adv.shape_metrics(fit)[0]
        assert m["fwtm_fwhm"] > adv.GAUSS_FWTM_FWHM   # tailed, not Gaussian
        assert m["asymmetry"] > 0                     # low-energy tail

    def test_centroid_on_line(self, fit):
        assert fit.peak_info[0]["mean"] == pytest.approx(200.05, abs=0.2)

    def test_plot_runs_headless(self, fit):
        import matplotlib
        matplotlib.use("Agg")
        fit.plot()

    def test_json_roundtrip(self, fit, hpge_search, tmp_path):
        path = tmp_path / "hypermet.json"
        fit.save_json(path)
        restored = adv.HypermetFit.load_json(path, hpge_search)
        assert restored.fit_result.best_values["g1_tail_fraction"] == pytest.approx(
            fit.fit_result.best_values["g1_tail_fraction"], rel=1e-6
        )


# ---------------------------------------------------------------------------
# FullHPGePeakFit — Hypermet tail on a smoothed step (the full HPGe shape).
# On the tight 200.1 keV window it is the best of all models (gauss ~71,
# hypermet ~2.6, full ~2.2).
# ---------------------------------------------------------------------------

FULL_XRANGE = [196, 204]


class TestFullHPGePeakFit:
    @pytest.fixture
    def fit(self, hpge_search):
        return adv.FullHPGePeakFit(hpge_search, FULL_XRANGE)

    @pytest.fixture
    def gauss(self, hpge_search):
        return adv.MultiProfilePeakFit(
            hpge_search, FULL_XRANGE, bkg="linear", profile="gauss"
        )

    def test_mro_composes_tail_and_step(self):
        names = [c.__name__ for c in adv.FullHPGePeakFit.__mro__]
        # Hypermet peak component takes priority; step background follows.
        assert names.index("_HypermetPeakMixin") < names.index("GaussStepFit")

    def test_sharp_step_is_default(self, fit):
        # The sharp step is the robust default: it shares only the centroid
        # with the tail (not sigma), avoiding the smooth-step degeneracy.
        assert fit.step == "sharp"
        assert fit.bkg == "step"
        assert fit.bkg_key == "bkg_"

    def test_has_tail_and_step_parameters(self, fit):
        params = fit.fit_result.params
        assert "g1_tail_fraction" in params
        assert "g1_tail_tau" in params
        assert "bkg_step_amplitude" in params
        assert "bkg_step_sigma" not in params  # sharp step has no width

    def test_step_tied_to_peak(self, fit):
        bv = fit.fit_result.best_values
        assert bv["bkg_step_center"] == bv["g1_center"]

    def test_converges(self, fit):
        assert fit.fit_result.success

    def test_converges_on_tailed_peak_without_step(self, hpge_search):
        # Regression: the 6129 keV peak is strongly tailed but sits on a
        # flat continuum (no Compton step). With the smooth step this fit
        # diverged (step shares sigma with the tail -> singular); the sharp
        # step default reduces the step to ~0 and still captures the tail.
        fit = adv.FullHPGePeakFit(hpge_search, [6100, 6150])
        gauss = adv.MultiProfilePeakFit(
            hpge_search, [6100, 6150], bkg="linear", profile="gauss"
        )
        assert fit.fit_result.success
        assert fit.fit_result.redchi < 0.2 * gauss.fit_result.redchi
        assert fit.fit_result.best_values["g1_tail_fraction"] > 0.2

    def test_beats_gaussian(self, fit, gauss):
        assert fit.fit_result.redchi < 0.1 * gauss.fit_result.redchi

    def test_tail_fraction_significant(self, fit):
        tf = fit.fit_result.best_values["g1_tail_fraction"]
        assert 0.05 < tf < 0.95

    def test_area_total_matches_gaussian(self, fit, gauss):
        assert fit.peak_info[0]["area"] == pytest.approx(
            gauss.peak_info[0]["area"], rel=0.05
        )

    def test_measured_shape_shows_low_energy_tail(self, fit):
        m = adv.shape_metrics(fit)[0]
        assert m["fwtm_fwhm"] > adv.GAUSS_FWTM_FWHM
        assert m["asymmetry"] > 0

    def test_plot_runs_headless(self, fit):
        import matplotlib
        matplotlib.use("Agg")
        fit.plot()

    def test_shared_sigma_not_supported(self, hpge_search):
        with pytest.raises(NotImplementedError, match="shared_sigma"):
            adv.FullHPGePeakFit(hpge_search, FULL_XRANGE, shared_sigma=True)

    def test_json_roundtrip(self, fit, hpge_search, tmp_path):
        path = tmp_path / "full_hpge.json"
        fit.save_json(path)
        restored = adv.FullHPGePeakFit.load_json(path, hpge_search)
        assert restored.step == "sharp"
        assert restored.fit_result.best_values["g1_tail_fraction"] == pytest.approx(
            fit.fit_result.best_values["g1_tail_fraction"], rel=1e-6
        )
        assert restored.fit_result.best_values["bkg_step_amplitude"] == pytest.approx(
            fit.fit_result.best_values["bkg_step_amplitude"], rel=1e-6
        )


# ---------------------------------------------------------------------------
# Peak area units — net counts, NOT counts/sec.
# Regression: the area was being divided by livetime, reporting a rate that
# was orders of magnitude too small (and silently breaking efficiency.py,
# which compares it against an emitted-counts number).
# ---------------------------------------------------------------------------

class TestAreaIsNetCounts:
    def test_area_not_divided_by_livetime(self, hpge_search):
        from wara.peakfit import PeakFit

        spect = hpge_search.spectrum
        lt = spect.livetime
        assert lt and lt > 1 and not spect.cps   # HPGe file has a 28800 s livetime

        fit = PeakFit(hpge_search, [6100, 6150], bkg="linear")
        area = fit.peak_info[0]["area"]

        # area == amplitude / mean-bin-width, with NO livetime division.
        amp = fit.fit_result.best_values["g1_amplitude"]
        x = spect.energies
        che = spect.channels[(x > 6100) & (x < 6150)]
        correct_f = float(np.diff(spect.energies[che]).mean())
        assert area == pytest.approx(amp / correct_f, rel=1e-6)

        # Net counts here are ~10^5; the old (wrong) cps value was ~5.
        assert area > 1e4
        assert area / lt < 100   # the rate it must NOT be reporting

    def test_area_matches_across_methods(self, hpge_search):
        # All fitters share the same area convention (net counts).
        from wara.peakfit import PeakFit
        g = PeakFit(hpge_search, [6100, 6150], bkg="linear")
        full = adv.FullHPGePeakFit(hpge_search, [6100, 6150])
        # Both > 10^4 counts; the tailed fit recovers a bit more (the tail).
        assert g.peak_info[0]["area"] > 1e4
        assert full.peak_info[0]["area"] > g.peak_info[0]["area"]


# ---------------------------------------------------------------------------
# Doppler-broadened companion peaks (MultiProfilePeakFit doppler=).
# A broad, recoil-broadened line is many times the detector resolution and is
# often too wide/shallow for the peak finder to flag (cf. Evans et al. 2006,
# the broad atmospheric C/O lines). `doppler=` declares such a line as a broad
# Gaussian. The broad-line tests mirror examples/peakfit/example_doppler_peak.py
# (the ~2495 keV line of the HPGe NH3 spectrum, which the search skips).
# ---------------------------------------------------------------------------


class TestNormalizeDoppler:
    def test_none_gives_empty(self):
        assert adv._normalize_doppler(None) == []

    def test_scalar_energy(self):
        assert adv._normalize_doppler(2211) == [{"energy": 2211.0}]

    def test_list_of_energies(self):
        out = adv._normalize_doppler([2211, 2230])
        assert [d["energy"] for d in out] == [2211.0, 2230.0]

    def test_dict_passthrough(self):
        out = adv._normalize_doppler({"energy": 2211, "fwhm": 8})
        assert out == [{"energy": 2211, "fwhm": 8}]

    def test_mixed_list(self):
        out = adv._normalize_doppler([2211, {"energy": 2230, "max_fwhm": 18}])
        assert len(out) == 2 and out[1]["max_fwhm"] == 18

    def test_missing_energy_raises(self):
        with pytest.raises(ValueError, match="needs an 'energy'"):
            adv._normalize_doppler({"fwhm": 8})

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown doppler key"):
            adv._normalize_doppler({"energy": 2211, "widht": 8})


# --- A lone broad line (no narrow peaks) — mirrors the example --------------
BROAD_XRANGE = [2462, 2530]
BROAD_SPEC = [{"energy": 2498, "fwhm": 25, "min_fwhm": 10, "max_fwhm": 50}]


@pytest.fixture(scope="module")
def broad_search():
    """Search over the broad ~2495 keV NH3 line the peak finder skips."""
    spect = file_reader.read_txt(TXT_HPGE)
    return ps.PeakSearch(spect, ref_x=2495, ref_fwhm=3.5, fwhm_at_0=1.0,
                         min_snr=8, method="fast")


class TestDopplerBroadLine:
    @pytest.fixture(scope="module")
    def fit(self, broad_search):
        return adv.MultiProfilePeakFit(broad_search, BROAD_XRANGE, bkg="poly1",
                                       profile="gauss", doppler=BROAD_SPEC)

    def test_base_fit_raises_without_doppler(self, broad_search):
        # No narrow peak is detected, so an ordinary fit cannot be set up.
        with pytest.raises(ValueError, match="No peaks found"):
            adv.MultiProfilePeakFit(broad_search, BROAD_XRANGE, bkg="poly1",
                                    profile="gauss")

    def test_fits_lone_broad_line(self, fit):
        # The declared companion is the only component, and it is fitted well.
        assert len(fit.peak_info) == 1
        assert fit._free_sigma_idx == {0}
        assert fit.fit_result.redchi < 4.0

    def test_recovers_broad_symmetric_line(self, fit):
        broad = fit.summary().iloc[0]
        assert 20.0 < broad["fwhm"] < 50.0          # genuinely broad
        assert 2490.0 < broad["mean"] < 2510.0      # near the declared energy
        assert broad["area"] > 0
        assert np.isfinite(broad["area_err"])

    def test_width_bounds_respected(self, fit):
        # The fitted FWHM stays within the min_fwhm / max_fwhm bounds.
        assert 10.0 <= fit.summary().iloc[0]["fwhm"] <= 50.0

    def test_seed_independent(self, broad_search):
        # The result should not depend on the width seed for this clean line.
        widths = []
        for seed in (15, 25, 40):
            spec = [dict(BROAD_SPEC[0], fwhm=seed)]
            f = adv.MultiProfilePeakFit(broad_search, BROAD_XRANGE, bkg="poly1",
                                        profile="gauss", doppler=spec)
            widths.append(f.summary().iloc[0]["fwhm"])
        assert max(widths) - min(widths) < 1.0


# --- A companion together with narrow peaks (API mechanics) -----------------
MECH_XRANGE = [2202, 2240]
MECH_KW = dict(profile="skewed", bkg="poly1")
# Broad companion near the 2211 keV bump; replace_narrow drops the auto peak
# the finder places on the broad apex (~2213 keV).
MECH_SPEC = [{"energy": 2211, "fwhm": 10, "max_fwhm": 22,
              "replace_narrow": True, "suppress_fwhm": 4}]


@pytest.fixture(scope="module")
def h_search():
    """Search over the 2223 keV hydrogen region (has narrow peaks)."""
    spect = file_reader.read_txt(TXT_HPGE)
    return ps.PeakSearch(spect, ref_x=2224, ref_fwhm=3.5, fwhm_at_0=1.0,
                         min_snr=50, method="fast")


class TestDopplerMechanics:
    def test_none_matches_no_doppler(self, h_search):
        base = adv.MultiProfilePeakFit(h_search, MECH_XRANGE, **MECH_KW)
        explicit = adv.MultiProfilePeakFit(h_search, MECH_XRANGE, doppler=None,
                                           **MECH_KW)
        assert len(explicit.peak_info) == len(base.peak_info)
        assert explicit._free_sigma_idx == set()

    def test_companion_appended_and_free(self, h_search):
        fit = adv.MultiProfilePeakFit(h_search, MECH_XRANGE, doppler=MECH_SPEC,
                                      **MECH_KW)
        # The companion is the last peak and is exempt from shared_sigma.
        assert fit._free_sigma_idx == {len(fit.peak_info) - 1}
        broad = fit.summary().iloc[-1]
        assert broad["fwhm"] > 1.5 * fit.summary().iloc[:-1]["fwhm"].median()

    def test_replace_narrow_culls_detection(self, h_search):
        with_replace = adv.MultiProfilePeakFit(
            h_search, MECH_XRANGE, doppler=MECH_SPEC, **MECH_KW)
        keep = [dict(MECH_SPEC[0], replace_narrow=False)]
        without = adv.MultiProfilePeakFit(
            h_search, MECH_XRANGE, doppler=keep, **MECH_KW)
        # replace_narrow drops the auto peak on the broad apex, so one fewer.
        assert len(with_replace.peak_info) == len(without.peak_info) - 1

    def test_multiple_companions(self, h_search):
        fit = adv.MultiProfilePeakFit(h_search, MECH_XRANGE,
                                      doppler=[2211, 2230], **MECH_KW)
        assert len(fit._free_sigma_idx) == 2

    def test_companion_is_plain_gaussian_under_emg(self, h_search):
        # Doppler broadening is symmetric: the companion must be a plain
        # Gaussian (no tail parameter) even when the narrow peaks are EMG.
        kw = dict(MECH_KW, profile="emg")
        fit = adv.MultiProfilePeakFit(h_search, MECH_XRANGE, doppler=MECH_SPEC,
                                      **kw)
        prefix = sorted(fit._doppler_bounds)[0]
        names = set(fit.fit_result.params.keys())
        assert f"{prefix}sigma" in names
        assert f"{prefix}tau" not in names      # gaussian companion, no EMG tail
        assert any(k.endswith("tau") for k in names)   # narrow peaks are EMG


# ---------------------------------------------------------------------------
# Peak constraints (peaks= / fix_center / fix_fwhm / ratios) on a synthetic
# doublet with a known answer. Mirrors examples/peakfit/example_peak_constraints.
# A strong line at 1330 keV with a weak (~12%), overlapping companion at 1339
# keV on a flat background — the peak finder merges them, so the constraints
# carry the physical knowledge needed to recover the truth.
# ---------------------------------------------------------------------------

DBL_BIN = 0.5
DBL_SIGMA = 2.6
DBL_FWHM = 2.355 * DBL_SIGMA
DBL_RATIO = 2600.0 / 22000.0
DBL_XRANGE = [1314, 1358]
DBL_PEAKS = [1330, 1339]


@pytest.fixture(scope="module")
def doublet_search():
    from wara.spectrum import Spectrum

    e = np.arange(1300.0, 1380.0, DBL_BIN)
    g = lambda h, c: h * np.exp(-0.5 * ((e - c) / DBL_SIGMA) ** 2)
    truth = 350.0 + g(22000.0, 1330.0) + g(2600.0, 1339.0)
    counts = np.random.default_rng(12).poisson(truth).astype(float)
    spect = Spectrum(counts=counts, energies=e, e_units="keV", livetime=100)
    return ps.PeakSearch(spect, ref_x=1330, ref_fwhm=DBL_FWHM, fwhm_at_0=0.5,
                         min_snr=8, method="km")


def _weak(fit):
    """The weak (~1339 keV) peak's summary row."""
    s = fit.summary()
    return s.iloc[(s["mean"] - 1339).abs().idxmin()]


class TestPeakConstraints:
    def test_auto_merges_the_doublet(self, doublet_search):
        auto = adv.MultiProfilePeakFit(doublet_search, DBL_XRANGE, bkg="poly0",
                                       profile="gauss")
        # The finder misses the weak line: one peak and a poor fit.
        assert len(auto.peak_info) == 1
        assert auto.fit_result.redchi > 50

    def test_peaks_resolves_doublet(self, doublet_search):
        fit = adv.MultiProfilePeakFit(doublet_search, DBL_XRANGE, bkg="poly0",
                                      profile="gauss", peaks=DBL_PEAKS)
        assert len(fit.peak_info) == 2
        assert fit.fit_result.redchi < 2.0
        # Recovers the true weak-line net area (~33890 counts) within errors.
        w = _weak(fit)
        assert w["area"] == pytest.approx(33890, rel=0.05)

    def test_peaks_without_autodetection(self, doublet_search):
        # A flat sub-window where the finder sees nothing still fits via peaks=.
        fit = adv.MultiProfilePeakFit(doublet_search, [1300, 1312], bkg="poly0",
                                      profile="gauss", peaks=[1306])
        assert len(fit.peak_info) == 1

    def test_fix_center_pins_centroid(self, doublet_search):
        fit = adv.MultiProfilePeakFit(doublet_search, DBL_XRANGE, bkg="poly0",
                                      profile="gauss", peaks=DBL_PEAKS,
                                      fix_center={1339: 1339.0})
        w = _weak(fit)
        assert w["mean"] == pytest.approx(1339.0, abs=1e-6)
        assert np.isnan(w["mean_err"]) or w["mean_err"] < 0.02

    def test_fix_fwhm_pins_width(self, doublet_search):
        fit = adv.MultiProfilePeakFit(doublet_search, DBL_XRANGE, bkg="poly0",
                                      profile="gauss", peaks=DBL_PEAKS,
                                      fix_fwhm={1339: DBL_FWHM})
        w = _weak(fit)
        assert w["fwhm"] == pytest.approx(DBL_FWHM, abs=1e-6)
        assert np.isnan(w["fwhm_err"]) or w["fwhm_err"] < 0.02

    def test_fix_fwhm_bounds(self, doublet_search):
        fit = adv.MultiProfilePeakFit(doublet_search, DBL_XRANGE, bkg="poly0",
                                      profile="gauss", peaks=DBL_PEAKS,
                                      fix_fwhm={1339: (5.0, 7.0)})
        assert 5.0 <= _weak(fit)["fwhm"] <= 7.0

    def test_ratios_tie_areas(self, doublet_search):
        fit = adv.MultiProfilePeakFit(doublet_search, DBL_XRANGE, bkg="poly0",
                                      profile="gauss", peaks=DBL_PEAKS,
                                      ratios=[(1339, 1330, DBL_RATIO)])
        s = fit.summary()
        strong = s.iloc[(s["mean"] - 1330).abs().idxmin()]
        weak = s.iloc[(s["mean"] - 1339).abs().idxmin()]
        assert weak["area"] / strong["area"] == pytest.approx(DBL_RATIO, rel=1e-4)

    def test_ratios_shrink_weak_area_error(self, doublet_search):
        free = adv.MultiProfilePeakFit(doublet_search, DBL_XRANGE, bkg="poly0",
                                       profile="gauss", peaks=DBL_PEAKS)
        tied = adv.MultiProfilePeakFit(doublet_search, DBL_XRANGE, bkg="poly0",
                                       profile="gauss", peaks=DBL_PEAKS,
                                       ratios=[(1339, 1330, DBL_RATIO)])
        assert _weak(tied)["area_err"] < _weak(free)["area_err"]

    def test_ratios_self_reference_raises(self, doublet_search):
        with pytest.raises(ValueError, match="same peak"):
            adv.MultiProfilePeakFit(doublet_search, DBL_XRANGE, bkg="poly0",
                                    profile="gauss", peaks=DBL_PEAKS,
                                    ratios=[(1330, 1330, 1.0)])

    def test_constraint_far_from_any_peak_raises(self, doublet_search):
        with pytest.raises(ValueError, match="No fitted peak within"):
            adv.MultiProfilePeakFit(doublet_search, DBL_XRANGE, bkg="poly0",
                                    profile="gauss", peaks=DBL_PEAKS,
                                    fix_center={1500: 1500.0})

    def test_user_hints_win_over_constraint(self, doublet_search):
        # An explicit hint on the same parameter takes precedence over fix_center.
        fit = adv.MultiProfilePeakFit(
            doublet_search, DBL_XRANGE, bkg="poly0", profile="gauss",
            peaks=DBL_PEAKS, fix_center={1339: 1339.0},
            hints={"g2_center": {"value": 1340.0, "vary": False}})
        assert _weak(fit)["mean"] == pytest.approx(1340.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Peak constraints on REAL data — the overlapping 932.6/937.0 keV doublet in
# the HPGe NH3 spectrum. Mirrors examples/peakfit/example_peak_constraints_real.
# ---------------------------------------------------------------------------

REAL_XRANGE = [927, 943]
REAL_PEAKS = [932.5, 937.0]
REAL_RES = 2.49        # detector FWHM at ~935 keV (from clean lines)


@pytest.fixture(scope="module")
def real_doublet_search():
    spect = file_reader.read_txt(TXT_HPGE)
    return ps.PeakSearch(spect, ref_x=935, ref_fwhm=2.5, fwhm_at_0=0.5,
                         min_snr=6, method="km")


class TestPeakConstraintsRealData:
    def test_peaks_gives_two_resolved_lines(self, real_doublet_search):
        fit = adv.MultiProfilePeakFit(real_doublet_search, REAL_XRANGE,
                                      bkg="poly1", profile="gauss",
                                      peaks=REAL_PEAKS)
        assert len(fit.peak_info) == 2
        means = sorted(p["mean"] for p in fit.peak_info)
        assert means[0] == pytest.approx(932.3, abs=0.5)
        assert means[1] == pytest.approx(936.8, abs=0.5)

    def test_fix_fwhm_pins_both_widths(self, real_doublet_search):
        fit = adv.MultiProfilePeakFit(
            real_doublet_search, REAL_XRANGE, bkg="poly1", profile="gauss",
            peaks=REAL_PEAKS, fix_fwhm={932.5: REAL_RES, 937.0: REAL_RES})
        assert all(p["fwhm"] == pytest.approx(REAL_RES, abs=1e-6)
                   for p in fit.peak_info)

    def test_fix_fwhm_shrinks_area_errors(self, real_doublet_search):
        free = adv.MultiProfilePeakFit(real_doublet_search, REAL_XRANGE,
                                       bkg="poly1", profile="gauss",
                                       peaks=REAL_PEAKS)
        fixed = adv.MultiProfilePeakFit(
            real_doublet_search, REAL_XRANGE, bkg="poly1", profile="gauss",
            peaks=REAL_PEAKS, fix_fwhm={932.5: REAL_RES, 937.0: REAL_RES})
        free_err = max(e["area_err"] for e in free.peak_err)
        fixed_err = max(e["area_err"] for e in fixed.peak_err)
        assert fixed_err < free_err

    def test_ratios_tie_real_doublet(self, real_doublet_search):
        fit = adv.MultiProfilePeakFit(real_doublet_search, REAL_XRANGE,
                                      bkg="poly1", profile="gauss",
                                      peaks=REAL_PEAKS,
                                      ratios=[(937.0, 932.5, 0.70)])
        s = fit.summary().sort_values("mean")
        lo, hi = s.iloc[0], s.iloc[1]
        assert hi["area"] / lo["area"] == pytest.approx(0.70, rel=1e-4)
