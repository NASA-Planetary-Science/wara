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
