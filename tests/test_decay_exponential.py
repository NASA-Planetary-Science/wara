"""
Pytest tests for decay_exponential module.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wara.decay_exponential import (
    single_decay_plus_constant,
    double_decay_plus_constant,
    guess_halflife,
    Decay_exp,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def make_single_decay(t, A=1000.0, k=0.05, C=50.0, noise=10.0):
    y = single_decay_plus_constant(t, A, k, C)
    y += RNG.normal(0, noise, size=len(t))
    yerr = np.full_like(t, noise)
    return y, yerr


def make_double_decay(t, A1=700.0, A2=300.0, k1=0.05, k2=0.25, C=30.0, noise=10.0):
    y = double_decay_plus_constant(t, A1, A2, k1, k2, C)
    y += RNG.normal(0, noise, size=len(t))
    yerr = np.full_like(t, noise)
    return y, yerr


@pytest.fixture
def t():
    return np.linspace(1, 100, 80)


@pytest.fixture
def single_data(t):
    y, yerr = make_single_decay(t)
    return y, yerr


@pytest.fixture
def double_data(t):
    y, yerr = make_double_decay(t)
    return y, yerr


@pytest.fixture
def fitter_single(t, single_data):
    y, yerr = single_data
    return Decay_exp(x=t, y=y, yerr=yerr)


@pytest.fixture
def fitter_double(t, double_data):
    y, yerr = double_data
    return Decay_exp(x=t, y=y, yerr=yerr)


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

class TestModelFunctions:
    def test_single_decay_at_zero(self):
        assert single_decay_plus_constant(0, A=100, k=1, C=10) == pytest.approx(110.0)

    def test_single_decay_large_t(self):
        val = single_decay_plus_constant(1000, A=100, k=1, C=10)
        assert val == pytest.approx(10.0, abs=1e-6)

    def test_double_decay_at_zero(self):
        val = double_decay_plus_constant(0, A1=100, A2=50, k1=1, k2=2, C=5)
        assert val == pytest.approx(155.0)

    def test_double_decay_large_t(self):
        val = double_decay_plus_constant(1000, A1=100, A2=50, k1=1, k2=2, C=5)
        assert val == pytest.approx(5.0, abs=1e-6)


# ---------------------------------------------------------------------------
# guess_halflife
# ---------------------------------------------------------------------------

class TestGuessHalflife:
    def test_typical_decay(self):
        t = np.linspace(0.1, 100, 200)
        y = 1000 * np.exp(-np.log(2) / 20 * t) + 10
        thalf = guess_halflife(t, y)
        assert thalf == pytest.approx(20.0, rel=0.2)

    def test_empty_mask_no_crash(self):
        # All y values are below ymid — mask is always False
        t = np.linspace(1, 10, 50)
        y = np.ones(50) * 5.0  # flat; ymid == y everywhere, y > ymid is empty
        result = guess_halflife(t, y)
        assert result > 0

    def test_zero_first_x_no_crash(self):
        # Data starts at x=0 — should not produce division by zero in callers
        t = np.linspace(0, 50, 100)
        y = 500 * np.exp(-0.1 * t) + 20
        result = guess_halflife(t, y)
        assert result >= 0


# ---------------------------------------------------------------------------
# Decay_exp — construction
# ---------------------------------------------------------------------------

class TestDecayExpInit:
    def test_stores_attributes(self, t, single_data):
        y, yerr = single_data
        d = Decay_exp(x=t, y=y, yerr=yerr)
        assert np.array_equal(d.x_data, t)
        assert np.array_equal(d.y_data, y)
        assert np.array_equal(d.yerr, yerr)
        assert d.degree is None

    def test_none_yerr_accepted(self, t, single_data):
        y, _ = single_data
        d = Decay_exp(x=t, y=y, yerr=None)
        assert d.yerr is None


# ---------------------------------------------------------------------------
# fit_single_decay
# ---------------------------------------------------------------------------

class TestFitSingleDecay:
    def test_sets_degree(self, fitter_single):
        fitter_single.fit_single_decay()
        assert fitter_single.degree == 1

    def test_fit_result_exists(self, fitter_single):
        fitter_single.fit_single_decay()
        assert hasattr(fitter_single, "fit_result")

    def test_k_positive(self, fitter_single):
        fitter_single.fit_single_decay()
        assert fitter_single.fit_result.best_values["k"] > 0

    def test_k_close_to_truth(self, fitter_single):
        fitter_single.fit_single_decay()
        k_fit = fitter_single.fit_result.best_values["k"]
        assert k_fit == pytest.approx(0.05, rel=0.15)

    def test_no_yerr(self, t, single_data):
        y, _ = single_data
        d = Decay_exp(x=t, y=y, yerr=None)
        d.fit_single_decay()
        assert hasattr(d, "fit_result")


# ---------------------------------------------------------------------------
# fit_double_decay
# ---------------------------------------------------------------------------

class TestFitDoubleDecay:
    def test_sets_degree(self, fitter_double):
        fitter_double.fit_double_decay()
        assert fitter_double.degree == 2

    def test_fit_result_exists(self, fitter_double):
        fitter_double.fit_double_decay()
        assert hasattr(fitter_double, "fit_result")

    def test_k1_k2_both_positive(self, fitter_double):
        fitter_double.fit_double_decay()
        bv = fitter_double.fit_result.best_values
        assert bv["k1"] > 0
        assert bv["k2"] > 0

    def test_no_yerr(self, t, double_data):
        y, _ = double_data
        d = Decay_exp(x=t, y=y, yerr=None)
        d.fit_double_decay()
        assert hasattr(d, "fit_result")


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::UserWarning")
class TestPlot:
    def test_plot_raises_before_fit(self, fitter_single):
        with pytest.raises(RuntimeError, match="fit_single_decay"):
            fitter_single.plot()

    def test_single_plot_runs(self, fitter_single):
        fitter_single.fit_single_decay()
        fitter_single.plot()
        plt.close("all")

    def test_double_plot_runs(self, fitter_double):
        fitter_double.fit_double_decay()
        fitter_double.plot()
        plt.close("all")

    def test_plot_returns_axes(self, fitter_single):
        fitter_single.fit_single_decay()
        result = fitter_single.plot()
        ax_fit, ax_res = result
        assert isinstance(ax_fit, plt.Axes)
        plt.close("all")

    def test_plot_show_components_single(self, fitter_single):
        fitter_single.fit_single_decay()
        fitter_single.plot(show_components=True)
        plt.close("all")

    def test_plot_show_components_double(self, fitter_double):
        fitter_double.fit_double_decay()
        fitter_double.plot(show_components=True)
        plt.close("all")

    def test_plot_with_provided_ax(self, fitter_single):
        fitter_single.fit_single_decay()
        fig, ax = plt.subplots()
        fitter_single.plot(ax_fit=ax)
        plt.close("all")
