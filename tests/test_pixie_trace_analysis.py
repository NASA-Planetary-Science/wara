"""Tests for wara.pixie_trace_analysis (offline fast filter / CFD reconstruction)."""
import numpy as np
import pytest

from wara.pixie_trace_analysis import (
    cfd_trace,
    compute_cfd_timing,
    fast_filter,
    find_cfd_crossing,
    find_fast_trigger,
)


def _step_trace(n=60, step_at=30, amplitude=1000.0, baseline=100.0):
    t = np.full(n, baseline, dtype=float)
    t[step_at:] = amplitude
    return t


def _pulse_trace(n=60, rise_at=25, amplitude=1000.0, baseline=100.0, decay=15.0):
    """Step rise followed by an exponential decay, like a real detector pulse."""
    t = np.full(n, baseline, dtype=float)
    x = np.arange(n - rise_at, dtype=float)
    t[rise_at:] = baseline + amplitude * np.exp(-x / decay)
    return t


def test_fast_filter_flat_trace_is_zero():
    # A perfectly flat trace has equal leading/trailing sums everywhere.
    t = np.full(50, 100.0)
    ff = fast_filter(t, FL=5, FG=2)
    valid = ff[2 * 5 + 2 - 1:]
    assert np.allclose(valid, 0.0)


def test_fast_filter_step_response():
    # A step of height A after FL+FG samples: once the leading window is
    # fully past the step and the trailing one fully before it, FF = FL*A.
    FL, FG = 5, 2
    baseline, amp = 0.0, 10.0
    step_at = 20
    t = _step_trace(n=60, step_at=step_at, amplitude=amp, baseline=baseline)
    ff = fast_filter(t, FL, FG)[0]
    # once fully saturated (leading window entirely past the step, trailing
    # window entirely before it)
    i = step_at + FL - 1
    assert ff[i] == pytest.approx(FL * (amp - baseline))


def test_find_fast_trigger_crosses_at_expected_sample():
    FL, FG = 5, 2
    amp = 20.0
    step_at = 25
    t = _step_trace(n=60, step_at=step_at, amplitude=amp, baseline=0.0)
    ff = fast_filter(t, FL, FG)
    threshold = FL * amp / 2  # crosses partway up the rising trapezoid
    idx = find_fast_trigger(ff, threshold)
    assert idx[0] > 0
    assert ff[0, idx[0]] >= threshold
    assert ff[0, idx[0] - 1] < threshold


def test_find_fast_trigger_no_crossing_returns_minus_one():
    t = np.full(40, 5.0)
    ff = fast_filter(t, FL=3, FG=2)
    idx = find_fast_trigger(ff, threshold=1e6)
    assert idx[0] == -1


def test_cfd_trace_symmetric_pulse_has_zero_crossing():
    # A rising step produces a positive-then-negative bipolar CFD response
    # (Eq 3-5) with a zero crossing near the step.
    n = 60
    step_at = 30
    t = _step_trace(n=n, step_at=step_at, amplitude=100.0, baseline=0.0)
    cfd = cfd_trace(t)
    finite = np.isfinite(cfd)
    assert finite.any()
    signs = np.sign(cfd[finite])
    # response must change sign (bipolar) around the step
    assert (signs > 0).any() and (signs < 0).any()


def test_find_cfd_crossing_matches_manual_sign_convention():
    # Build a CFD-like trace by hand: positive then crossing to negative at
    # a known fractional position between samples 9 and 10.
    cfd = np.zeros(20)
    cfd[9] = 2.0
    cfd[10] = -6.0  # f = 2 / (2 - (-6)) = 0.25 -> crossing at 9.25
    start_index = np.array([0])
    pos, valid = find_cfd_crossing(cfd[None, :], cfd_threshold=1.0, start_index=start_index)
    assert valid[0]
    assert pos[0] == pytest.approx(9.25)


def test_find_cfd_crossing_gated_by_threshold_skips_noise():
    # A small noise-level zero crossing before the real pulse must be
    # ignored because it never rises above cfd_threshold.
    cfd = np.zeros(30)
    cfd[5] = 0.5   # noise-level crossing, below threshold
    cfd[6] = -0.5
    cfd[20] = 5.0  # real pulse, above threshold
    cfd[21] = -15.0
    pos, valid = find_cfd_crossing(cfd[None, :], cfd_threshold=2.0, start_index=np.array([0]))
    assert valid[0]
    assert pos[0] == pytest.approx(20 + 5.0 / (5.0 - (-15.0)))


def test_find_cfd_crossing_invalid_start_index():
    cfd = np.zeros((1, 10))
    pos, valid = find_cfd_crossing(cfd, cfd_threshold=1.0, start_index=np.array([-1]))
    assert not valid[0]
    assert np.isnan(pos[0])


def test_compute_cfd_timing_batch_shapes():
    n_traces, n_samples = 4, 60
    rng = np.random.default_rng(0)
    T = np.stack([_pulse_trace(n=n_samples, rise_at=20 + i, amplitude=50.0)
                  for i in range(n_traces)])
    T = T + rng.normal(0, 0.01, T.shape)  # tiny noise, avoids exact ties
    res = compute_cfd_timing(T, FL=5, FG=2, fast_threshold=20.0, cfd_threshold=1.0)
    assert res["fast_filter"].shape == T.shape
    assert res["cfd_trace"].shape == T.shape
    assert res["fast_trigger_index"].shape == (n_traces,)
    assert res["cfd_position"].shape == (n_traces,)
    assert res["cfd_valid"].shape == (n_traces,)
    assert res["cfd_valid"].all()


def test_fast_filter_and_cfd_accept_1d_input():
    t = _step_trace(n=40, step_at=15, amplitude=30.0)
    ff = fast_filter(t, FL=4, FG=2)
    cfd = cfd_trace(t)
    assert ff.ndim == 2 and ff.shape[0] == 1
    assert cfd.ndim == 2 and cfd.shape[0] == 1
