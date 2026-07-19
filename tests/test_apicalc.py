"""
Pytest tests for apicalc module.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wara.apicalc import (
    GainShift,
    calc_own_pos,
    compute_fft,
    dfe,
    dfet,
    dft,
    dftxy,
    dftxye,
    dfxy,
    dfxy_exclude,
    dfxye,
    approximate_fa,
    find_data_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_df():
    """Minimal DataFrame with energy, dt, X, Y columns."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "energy": rng.uniform(0, 2000, n),
        "dt":     rng.uniform(-50, 150, n),
        "X":      rng.uniform(-1, 1, n),
        "Y":      rng.uniform(-1, 1, n),
    })


@pytest.fixture
def abcd_df():
    """DataFrame with A, B, C, D columns for calc_own_pos."""
    rng = np.random.default_rng(1)
    n = 100
    return pd.DataFrame({
        "A": rng.uniform(0.1, 1.0, n),
        "B": rng.uniform(0.1, 1.0, n),
        "C": rng.uniform(0.1, 1.0, n),
        "D": rng.uniform(0.1, 1.0, n),
    })


# ---------------------------------------------------------------------------
# calc_own_pos
# ---------------------------------------------------------------------------

class TestCalcOwnPos:
    def test_adds_x2_y2_columns(self, abcd_df):
        result = calc_own_pos(abcd_df)
        assert "X2" in result.columns
        assert "Y2" in result.columns

    def test_values_in_range(self, abcd_df):
        result = calc_own_pos(abcd_df)
        assert result["X2"].between(-1, 1).all()
        assert result["Y2"].between(-1, 1).all()

    def test_returns_copy(self, abcd_df):
        result = calc_own_pos(abcd_df)
        assert result is not abcd_df

    def test_idempotent_when_columns_exist(self, abcd_df):
        first = calc_own_pos(abcd_df)
        second = calc_own_pos(first)
        assert second is first  # returns early without recalculating

    def test_type_error_on_non_dataframe(self):
        with pytest.raises(TypeError, match="pandas dataframe"):
            calc_own_pos({"A": 1, "B": 2, "C": 3, "D": 4})


# ---------------------------------------------------------------------------
# DataFrame filter functions
# ---------------------------------------------------------------------------

class TestDfe:
    def test_filters_within_range(self, base_df):
        result = dfe(base_df, erange=[500, 1000])
        assert (result["energy"] > 500).all()
        assert (result["energy"] < 1000).all()

    def test_empty_result_for_impossible_range(self, base_df):
        result = dfe(base_df, erange=[9000, 9001])
        assert len(result) == 0

    def test_type_error_on_non_dataframe(self):
        with pytest.raises(TypeError):
            dfe([1, 2, 3], erange=[0, 100])


class TestDfxy:
    def test_filters_within_xy(self, base_df):
        result = dfxy(base_df, xrange=[-0.5, 0.5], yrange=[-0.5, 0.5])
        assert (result["X"] > -0.5).all()
        assert (result["X"] < 0.5).all()
        assert (result["Y"] > -0.5).all()
        assert (result["Y"] < 0.5).all()

    def test_type_error_on_non_dataframe(self):
        with pytest.raises(TypeError):
            dfxy("not a df", xrange=[0, 1], yrange=[0, 1])


class TestDfxyExclude:
    def test_excludes_interior(self, base_df):
        xrange = [-0.3, 0.3]
        yrange = [-0.3, 0.3]
        included = dfxy(base_df, xrange=xrange, yrange=yrange)
        excluded = dfxy_exclude(base_df, xrange=xrange, yrange=yrange)
        assert len(included) + len(excluded) == len(base_df)

    def test_type_error_on_non_dataframe(self):
        with pytest.raises(TypeError):
            dfxy_exclude("bad", xrange=[0, 1], yrange=[0, 1])


class TestDfxye:
    def test_filters_all_three(self, base_df):
        result = dfxye(base_df, xrange=[-0.5, 0.5], yrange=[-0.5, 0.5], erange=[500, 1500])
        assert (result["energy"] > 500).all()
        assert (result["energy"] < 1500).all()
        assert (result["X"] > -0.5).all()
        assert (result["X"] < 0.5).all()

    def test_type_error(self):
        with pytest.raises(TypeError):
            dfxye(None, xrange=[0, 1], yrange=[0, 1], erange=[0, 100])


class TestDft:
    def test_filters_time(self, base_df):
        result = dft(base_df, trange=[0, 100])
        assert (result["dt"] > 0).all()
        assert (result["dt"] < 100).all()

    def test_type_error(self):
        with pytest.raises(TypeError, match="pandas dataframe"):
            dft("not a df", trange=[0, 100])


class TestDfet:
    def test_filters_energy_and_time(self, base_df):
        result = dfet(base_df, erange=[500, 1500], trange=[0, 100])
        assert (result["energy"] > 500).all()
        assert (result["energy"] < 1500).all()
        assert (result["dt"] > 0).all()
        assert (result["dt"] < 100).all()

    def test_type_error(self):
        with pytest.raises(TypeError, match="pandas dataframe"):
            dfet(42, erange=[0, 100], trange=[0, 50])


class TestDftxy:
    def test_filters_xy_and_time(self, base_df):
        result = dftxy(base_df, xrange=[-0.5, 0.5], yrange=[-0.5, 0.5], trange=[0, 100])
        assert (result["X"] > -0.5).all()
        assert (result["dt"] > 0).all()
        assert (result["dt"] < 100).all()

    def test_type_error(self):
        with pytest.raises(TypeError):
            dftxy([], xrange=[0, 1], yrange=[0, 1], trange=[0, 100])


class TestDftxye:
    def test_filters_all_four(self, base_df):
        result = dftxye(
            base_df,
            xrange=[-0.5, 0.5], yrange=[-0.5, 0.5],
            erange=[500, 1500], trange=[0, 100],
        )
        assert (result["energy"] > 500).all()
        assert (result["dt"] > 0).all()
        assert (result["X"] > -0.5).all()

    def test_type_error(self):
        with pytest.raises(TypeError):
            dftxye(None, xrange=[0, 1], yrange=[0, 1], erange=[0, 100], trange=[0, 50])


# ---------------------------------------------------------------------------
# compute_fft
# ---------------------------------------------------------------------------

class TestComputeFFT:
    def test_dominant_frequency_detected(self):
        sr = 1000.0
        t = np.linspace(0, 1, int(sr), endpoint=False)
        freq_true = 50.0
        signal = np.sin(2 * np.pi * freq_true * t)
        freqs, mag = compute_fft(signal, sr)
        peak_freq = freqs[np.argmax(mag)]
        assert peak_freq == pytest.approx(freq_true, abs=1.0)

    def test_only_positive_frequencies(self):
        sr = 500.0
        signal = np.random.default_rng(7).normal(size=256)
        freqs, mag = compute_fft(signal, sr)
        assert (freqs >= 0).all()

    def test_output_lengths_match(self):
        sr = 100.0
        signal = np.ones(64)
        freqs, mag = compute_fft(signal, sr)
        assert len(freqs) == len(mag)

    def test_dc_component_for_constant_signal(self):
        sr = 100.0
        signal = np.ones(128) * 5.0
        freqs, mag = compute_fft(signal, sr)
        assert freqs[0] == pytest.approx(0.0)
        assert mag[0] == pytest.approx(5.0, rel=1e-6)


# ---------------------------------------------------------------------------
# approximate_fa
# ---------------------------------------------------------------------------

class TestApproximateFa:
    def test_small_sample_small_fraction(self):
        fa = approximate_fa(L=100, S=1)
        assert 0 < fa < 1

    def test_large_sample_clips_to_one(self):
        fa = approximate_fa(L=5, S=100)
        assert fa == pytest.approx(1.0)

    def test_fraction_never_exceeds_one(self):
        for S in [1, 5, 10, 50, 200]:
            fa = approximate_fa(L=30, S=S)
            assert fa <= 1.0


# ---------------------------------------------------------------------------
# find_data_path — date parsing validation
# ---------------------------------------------------------------------------

class TestFindDataPath:
    def test_unparseable_date_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not parse date string"):
            find_data_path(date="not-a-date", runnr=1)


# ---------------------------------------------------------------------------
# api_xyz — does not mutate caller's DataFrame
# ---------------------------------------------------------------------------

class TestApiXyzNoMutation:
    def test_toffset_does_not_mutate_input(self):
        from wara.apicalc import api_xyz
        rng = np.random.default_rng(5)
        n = 500
        df = pd.DataFrame({
            "X2": rng.uniform(-0.5, 0.5, n),
            "Y2": rng.uniform(-0.5, 0.5, n),
            "dt": rng.uniform(0, 100, n),
        })
        dt_original = df["dt"].copy()
        api_xyz(df, toffset=10.0)
        pd.testing.assert_series_equal(df["dt"], dt_original)


# ---------------------------------------------------------------------------
# api_xyz — full center-of-mass-corrected, relativistic model
# ---------------------------------------------------------------------------

def _xy_df(n=600, dt_ns=20.0, seed=3, with_corners=False):
    """Synthetic alpha image filling the YAP square, dt in seconds."""
    rng = np.random.default_rng(seed)
    x2 = rng.uniform(-0.6, 0.6, n)
    y2 = rng.uniform(-0.6, 0.6, n)
    dt = rng.normal(dt_ns * 1e-9, 1e-9, n)
    if with_corners:
        # A,B,C,D such that calc_own_pos reproduces (x2, y2)
        tot = 1.0
        B = (1 + x2 + y2) / 4 * tot
        A = (1 - x2 + y2) / 4 * tot
        C = (1 + x2 - y2) / 4 * tot
        D = (1 - x2 - y2) / 4 * tot
        return pd.DataFrame({"A": A, "B": B, "C": C, "D": D, "dt": dt})
    return pd.DataFrame({"X2": x2, "Y2": y2, "dt": dt})


class TestApiXyzFull:
    def test_returns_three_finite_arrays(self):
        from wara.apicalc import api_xyz
        df = _xy_df()
        X, Y, Z = api_xyz(df, dt_unit="s", use_det=False)
        assert len(X) == len(Y) == len(Z) == len(df)
        for arr in (X, Y, Z):
            assert np.isfinite(arr).all()

    def test_central_alpha_points_down_and_on_axis(self):
        from wara.apicalc import api_xyz
        # uniform fill (defines the edges) plus central events appended
        df = _xy_df(n=600)
        df = pd.concat([df, pd.DataFrame({"X2": [0.0, 0.0, 0.0],
                                          "Y2": [0.0, 0.0, 0.0],
                                          "dt": [20e-9] * 3})],
                       ignore_index=True)
        X, Y, Z = api_xyz(df, dt_unit="s", use_det=False)
        # the appended central rows are the last three
        assert np.allclose(X[-3:], 0.0, atol=1.0)   # within 1 cm of axis
        assert np.allclose(Y[-3:], 0.0, atol=1.0)
        assert (Z[-3:] < 0).all()                   # downward (-z)

    def test_builds_x2y2_from_corners(self):
        from wara.apicalc import api_xyz
        df = _xy_df(with_corners=True)
        X, Y, Z = api_xyz(df, dt_unit="s", use_det=False)
        assert np.isfinite(Z).all()
        assert (np.nanmedian(Z) < 0)

    def test_detector_term_changes_depth(self):
        from wara.apicalc import api_xyz
        df = _xy_df()
        _, _, Z_det = api_xyz(df, dt_unit="s", use_det=True)
        _, _, Z_ray = api_xyz(df, dt_unit="s", use_det=False)
        assert not np.allclose(np.nanmedian(Z_det), np.nanmedian(Z_ray))

    def test_relativistic_differs_from_nonrelativistic(self):
        from wara.apicalc import api_xyz
        df = _xy_df()
        _, _, Z_rel = api_xyz(df, dt_unit="s", use_det=False, relativistic=True)
        _, _, Z_nr = api_xyz(df, dt_unit="s", use_det=False, relativistic=False)
        # ~1% neutron-speed difference -> measurable depth shift
        assert abs(np.nanmedian(Z_rel) - np.nanmedian(Z_nr)) > 0.1

    def test_com_correction_tilts_direction(self):
        from wara.apicalc import api_xyz
        df = _xy_df()
        Xc, Yc, _ = api_xyz(df, dt_unit="s", use_det=False, use_com=True)
        X0, Y0, _ = api_xyz(df, dt_unit="s", use_det=False, use_com=False)
        assert not np.allclose(Xc, X0) or not np.allclose(Yc, Y0)

    def test_dt_unit_ns_matches_seconds(self):
        from wara.apicalc import api_xyz
        df_s = _xy_df()
        df_ns = df_s.copy()
        df_ns["dt"] = df_s["dt"] * 1e9
        Zs = api_xyz(df_s, dt_unit="s", use_det=False)[2]
        Zns = api_xyz(df_ns, dt_unit="ns", use_det=False)[2]
        np.testing.assert_allclose(Zs, Zns, rtol=1e-9)

    def test_unphysical_events_masked_only_with_detector(self):
        from wara.apicalc import api_xyz
        # prompt-gamma-like events: dt ~ 0, so c*dta < |D| and no positive
        # neutron flight time exists -> NaN with the detector term; the
        # straight-ray model must keep returning finite values (backup path).
        df = _xy_df(n=600)
        df = pd.concat([df, pd.DataFrame({"X2": [0.1, -0.1, 0.0],
                                          "Y2": [0.0, 0.1, -0.1],
                                          "dt": [-6e-9] * 3})],
                       ignore_index=True)
        _, _, Z_det = api_xyz(df, dt_unit="s", use_det=True)
        _, _, Z_ray = api_xyz(df, dt_unit="s", use_det=False)
        assert np.isnan(Z_det[-3:]).all()
        assert np.isfinite(Z_ray[-3:]).all()
        # the normal (physical) events stay finite in both models
        assert np.isfinite(Z_det[:-3]).all()

    def test_invalid_dt_unit_raises(self):
        from wara.apicalc import api_xyz
        with pytest.raises(ValueError, match="dt_unit"):
            api_xyz(_xy_df(), dt_unit="minutes")


class TestFitToffset:
    @staticmethod
    def _front_edge(Z, zrange=(-80, 0), bins=160):
        """Shallow (closest-to-source) half-max edge of the depth histogram."""
        Z = Z[np.isfinite(Z)]
        h, ed = np.histogram(Z, bins=bins, range=zrange)
        ctr = 0.5 * (ed[:-1] + ed[1:])
        above = np.flatnonzero(h >= h.max() / 2)
        return float(ctr[above[-1]])   # largest Z = shallowest

    def test_places_front_face_at_z0(self):
        from wara.apicalc import api_xyz, fit_toffset
        df = _xy_df(n=6000, dt_ns=35.0)     # 1 ns sigma -> thin "sample"
        dt_ns = df["dt"].to_numpy() * 1e9
        toff = fit_toffset(dt_ns, z0=-25.0, z_t=6.7,
                           det_pos=(-18.0, 0.0, -25.0), dt_unit="ns")
        _, _, Z = api_xyz(df, det_pos=(-18.0, 0.0, -25.0), toffset=toff * 1e-9,
                          dt_unit="s", use_det=True)
        # the front (shallow) edge of the depth distribution is the promise
        assert abs(self._front_edge(Z) - (-25.0)) < 8.0

    def test_peak_anchor_places_mode_at_z0(self):
        from wara.apicalc import api_xyz, fit_toffset
        df = _xy_df(n=6000, dt_ns=35.0)
        dt_ns = df["dt"].to_numpy() * 1e9
        toff = fit_toffset(dt_ns, z0=-25.0, z_t=6.7,
                           det_pos=(-18.0, 0.0, -25.0), dt_unit="ns",
                           anchor="peak")
        _, _, Z = api_xyz(df, det_pos=(-18.0, 0.0, -25.0), toffset=toff * 1e-9,
                          dt_unit="s", use_det=True)
        Zf = Z[np.isfinite(Z)]
        h, ed = np.histogram(Zf, bins=160, range=(-80, 0))
        zmode = float(0.5 * (ed[:-1] + ed[1:])[np.argmax(h)])
        assert abs(zmode - (-25.0)) < 8.0

    def test_invalid_anchor_raises(self):
        from wara.apicalc import fit_toffset
        with pytest.raises(ValueError, match="anchor"):
            fit_toffset([1.0], anchor="median")

    def test_units_consistent(self):
        from wara.apicalc import fit_toffset
        rng = np.random.default_rng(1)
        dt_s = rng.normal(30e-9, 2e-9, 2000)
        t_s = fit_toffset(dt_s, dt_unit="s")
        t_ns = fit_toffset(dt_s * 1e9, dt_unit="ns")
        assert t_ns == pytest.approx(t_s * 1e9, rel=1e-6)

    def test_empty_and_bad_unit(self):
        from wara.apicalc import fit_toffset
        assert fit_toffset([], dt_unit="ns") == 0.0
        with pytest.raises(ValueError, match="dt_unit"):
            fit_toffset([1.0], dt_unit="minutes")


class TestApiXyzSimpleStillAvailable:
    def test_simple_returns_arrays(self):
        from wara.apicalc import api_xyz_simple
        rng = np.random.default_rng(8)
        n = 400
        df = pd.DataFrame({
            "X2": rng.uniform(-0.6, 0.6, n),
            "Y2": rng.uniform(-0.6, 0.6, n),
            "dt": rng.uniform(10, 30, n),   # ns (legacy convention)
        })
        X, Y, Z = api_xyz_simple(df, use_det=False)
        assert len(X) == len(Y) == len(Z) == n


# ---------------------------------------------------------------------------
# GainShift.align — gain-drift correction across time segments
# ---------------------------------------------------------------------------

class TestGainShiftAlign:
    @staticmethod
    def _drifting_df(n_seg=8, per=5000, peak0=30000.0, total_drift=700.0, seed=0):
        """List-mode frame whose peak drifts linearly up by total_drift over the
        run (segment 0 at peak0, the last at peak0 + total_drift)."""
        rng = np.random.default_rng(seed)
        drifts = np.linspace(0, total_drift, n_seg)
        parts = [pd.DataFrame({"energy_orig": rng.normal(peak0 + d, 300, per)})
                 for d in drifts]
        return pd.concat(parts, ignore_index=True), drifts

    def _seg_means(self, cal):
        ecal = cal.df["energy_cal"].to_numpy()
        return np.array([ecal[pos].mean() for pos in cal._seg_idx])

    def test_align_writes_energy_cal_and_removes_drift(self):
        df, _ = self._drifting_df()
        cal = GainShift(df, n_segments=8, bins=2**12, erange=(0, 2**16))
        cal.align()
        assert "energy_cal" in cal.df.columns
        means = self._seg_means(cal)
        # Every segment's corrected peak collapses onto the reference: the spread
        # of per-segment means shrinks from ~700 ch (raw drift) to a few bins.
        assert means.max() - means.min() < 60        # raw spread was ~700
        assert abs(means.mean() - 30000.0) < 60

    def test_reference_segment_has_zero_shift(self):
        df, _ = self._drifting_df()
        cal = GainShift(df, n_segments=8, bins=2**12, erange=(0, 2**16))
        params = cal.align(reference=0)
        assert params[0]["intercept"] == pytest.approx(0.0, abs=1e-6)
        # Shifts grow more negative for later (higher-drift) segments.
        intercepts = [p["intercept"] for p in params]
        assert intercepts[-1] < intercepts[1] < 0

    def test_align_is_not_cumulative(self):
        df, _ = self._drifting_df()
        cal = GainShift(df, n_segments=8, bins=2**12, erange=(0, 2**16))
        first = [p["intercept"] for p in cal.align()]
        second = [p["intercept"] for p in cal.align()]   # re-run from energy_orig
        np.testing.assert_allclose(first, second)

    def test_align_with_xrange_window(self):
        df, _ = self._drifting_df()
        cal = GainShift(df, n_segments=8, bins=2**12, erange=(0, 2**16))
        cal.align(xrange=(28000, 32000))                 # focus on the peak band
        means = self._seg_means(cal)
        assert means.max() - means.min() < 60

    def test_plot_aligned_requires_align_first(self):
        df, _ = self._drifting_df()
        cal = GainShift(df, n_segments=8, bins=2**12, erange=(0, 2**16))
        with pytest.raises(RuntimeError, match="align"):
            cal.plot_aligned()
        cal.align()
        fig = cal.plot_aligned()
        assert fig is not None
        plt.close(fig)

    def test_shift_method_keeps_slope_one(self):
        df, _ = self._drifting_df()
        cal = GainShift(df, n_segments=8, bins=2**12, erange=(0, 2**16))
        params = cal.align(method="shift")
        assert all(p["slope"] == 1.0 for p in params)

    def test_invalid_method_raises(self):
        df, _ = self._drifting_df()
        cal = GainShift(df, n_segments=8, bins=2**12, erange=(0, 2**16))
        with pytest.raises(ValueError, match="shift.*linear"):
            cal.align(method="quadratic")

    # -- linear (slope + offset) ------------------------------------------------

    @staticmethod
    def _affine_two_peak_df(n_seg=8, per_peak=4000, p1=15000.0, p2=45000.0,
                            gain_to=1.08, off_to=-300.0, seed=1):
        """Two peaks whose channels drift affinely over the run:
        ch_obs = a_i + b_i * ch_true, with b_i ramping 1.0 -> gain_to."""
        rng = np.random.default_rng(seed)
        b = np.linspace(1.0, gain_to, n_seg)
        a = np.linspace(0.0, off_to, n_seg)
        parts = []
        for i in range(n_seg):
            e1 = a[i] + b[i] * rng.normal(p1, 200, per_peak)
            e2 = a[i] + b[i] * rng.normal(p2, 200, per_peak)
            parts.append(pd.DataFrame({"energy_orig": np.concatenate([e1, e2])}))
        return pd.concat(parts, ignore_index=True), b

    @staticmethod
    def _band_centroids(cal, center, half=6000):
        ec = cal.df["energy_cal"].to_numpy()
        out = []
        for pos in cal._seg_idx:
            v = ec[pos]; m = (v >= center - half) & (v <= center + half)
            out.append(v[m].mean() if m.any() else np.nan)
        return np.array(out)

    def test_linear_recovers_gain_slope(self):
        df, b = self._affine_two_peak_df()
        cal = GainShift(df, n_segments=8, bins=2**12, erange=(0, 2**16))
        params = cal.align(method="linear")
        # The correction inverts the per-segment gain: slope ≈ 1 / b_i.
        assert params[0]["slope"] == pytest.approx(1.0, abs=0.02)
        assert params[-1]["slope"] == pytest.approx(1.0 / b[-1], abs=0.03)

    def test_linear_aligns_both_peaks_better_than_shift(self):
        df, _ = self._affine_two_peak_df()
        lin = GainShift(df, n_segments=8, bins=2**12, erange=(0, 2**16))
        lin.align(method="linear")
        sh = GainShift(df, n_segments=8, bins=2**12, erange=(0, 2**16))
        sh.align(method="shift")
        # The low peak is what a single shift cannot also fix once the gain has
        # stretched the axis; the linear model should align it markedly better.
        lo_lin = np.nanmax(self._band_centroids(lin, 15000)) - \
            np.nanmin(self._band_centroids(lin, 15000))
        lo_sh = np.nanmax(self._band_centroids(sh, 15000)) - \
            np.nanmin(self._band_centroids(sh, 15000))
        assert lo_lin < lo_sh

    def test_linear_falls_back_to_shift_with_single_peak(self):
        # One peak gives no leverage on the slope, so linear degrades to a pure
        # shift (slope ≈ 1) rather than fitting noise.
        df, _ = self._drifting_df()
        cal = GainShift(df, n_segments=8, bins=2**12, erange=(0, 2**16))
        params = cal.align(method="linear")
        assert all(p["slope"] == pytest.approx(1.0, abs=1e-9) for p in params)
        means = self._seg_means(cal)
        assert means.max() - means.min() < 60        # still removes the drift
