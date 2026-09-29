"""Tests for the fast-filter-triggered PSD on raw trace matrices
(wara.neutron_psd: fast_filter, fast_filter_trigger, triggered_psd,
trace_matrix, TracePSD) -- the backend of PSD on API list-mode runs."""
import numpy as np
import pytest

from wara import neutron_psd as npsd

PEDESTAL = 8000
DT_NS = 2.0


def pulse(n=500, start=200.0, amp=3000.0, tau=5.0, rise=2.0):
    """Positive PIXIE-like pulse (ADC) on a pedestal, starting at *start*
    (samples, may be fractional): linear rise then exponential decay."""
    x = np.arange(n) - start
    body = np.where(x < 0, 0.0,
                    np.where(x < rise, amp * x / rise,
                             amp * np.exp(-(x - rise) / tau)))
    return PEDESTAL + body


def make_matrix(n_each=60, seed=0):
    """Short-tail ("gamma") and long-tail ("neutron") pulses at jittered start
    times and amplitudes, as a uint16 matrix like the parquet Trace column."""
    rng = np.random.default_rng(seed)
    rows, is_long = [], []
    for i in range(2 * n_each):
        long_tail = i % 2 == 0
        tr = pulse(start=rng.uniform(195, 215), amp=rng.uniform(1500, 6000),
                   tau=25.0 if long_tail else 6.0)
        rows.append(np.round(tr + rng.normal(0, 2, tr.size)).astype(np.uint16))
        is_long.append(long_tail)
    return np.stack(rows), np.array(is_long)


def test_fast_filter_settles_at_step_height():
    step = np.full((1, 100), 100.0); step[0, :50] = 0.0
    ff, first = npsd.fast_filter(step, rise=5, gap=2)
    assert first == 11
    assert ff.max() == pytest.approx(100.0)
    assert ff[0, 0] == pytest.approx(0.0)       # flat baseline reads zero


def test_trigger_tracks_pulse_start():
    a = pulse(start=200.0)[None]
    b = pulse(start=207.0)[None]
    ta = npsd.fast_filter_trigger(a, threshold=30)[0]
    tb = npsd.fast_filter_trigger(b, threshold=30)[0]
    assert tb - ta == pytest.approx(7.0, abs=1e-9)
    assert 195 < ta < 205                       # at the rising edge


def test_small_or_missing_pulse_has_no_trigger():
    small = pulse(amp=20.0)[None]               # never reaches 30 ADC
    flat = np.full((1, 500), float(PEDESTAL))
    trig = npsd.fast_filter_trigger(np.vstack([small, flat]), threshold=30)
    assert np.isnan(trig).all()
    psd, q, ok = npsd.triggered_psd(np.vstack([small, flat]), DT_NS, trig)
    assert not ok.any() and np.isnan(psd).all()


def test_psd_is_shift_invariant_and_separates_tails():
    a = pulse(start=200.0, tau=25.0)
    b = pulse(start=211.0, tau=25.0)            # same pulse, whole-sample shift
    c = pulse(start=205.4, tau=25.0)            # ... and a sub-sample one
    g = pulse(start=204.0, tau=6.0)
    tr = np.vstack([a, b, c, g])
    trig = npsd.fast_filter_trigger(tr, threshold=30)
    psd, q, ok = npsd.triggered_psd(tr, DT_NS, trig)
    assert ok.all()
    assert psd[0] == pytest.approx(psd[1], abs=1e-5)   # tail truncates at the record end
    assert q[0] == pytest.approx(q[1], rel=1e-5)
    # A 2-sample rise times to within ~0.4 sample (linear interpolation of the
    # threshold crossing), a small PSD jitter against the tail separation.
    assert psd[0] == pytest.approx(psd[2], abs=0.02)
    assert psd[0] > psd[3] + 0.2                # long tail -> larger PSD
    assert (0 < psd).all() and (psd < 1).all()


def test_default_gates_are_relative_to_trigger():
    p = npsd.TracePSD(pulse()[None], DT_NS).compute()
    assert (p.pre_ns, p.prompt_ns, p.tail_ns) == (30.0, 20.0, None)
    start, prompt, tail = p.gate_times()
    assert prompt - start == pytest.approx(50.0)
    assert tail == pytest.approx(p.time_ns[-1])  # tail runs to the trace end
    assert p.threshold == 30.0


def test_tracepsd_matrix_populations_and_chunking():
    mat, is_long = make_matrix()
    p = npsd.TracePSD(mat, DT_NS).compute()
    assert p.valid.all()
    assert np.nanmin(p.psd[is_long]) > np.nanmax(p.psd[~is_long])
    small = npsd.TracePSD(mat, DT_NS, chunk=7).compute()
    np.testing.assert_allclose(small.psd, p.psd)
    np.testing.assert_allclose(small.trigger, p.trigger)


def test_regate_reuses_triggers_and_retrigger_resets():
    mat, _ = make_matrix(n_each=10)
    p = npsd.TracePSD(mat, DT_NS).compute()
    trig, psd0 = p.trigger, p.psd.copy()
    p.set_gates(prompt_ns=30.0, tail_ns=200.0)
    p.compute()
    assert p.trigger is trig                    # not re-triggered
    assert not np.allclose(p.psd, psd0)
    p.set_gates(tail_to_end=True)
    assert p.tail_ns is None
    p.set_trigger(threshold=100.0)
    assert p.trigger is None
    p.compute()
    assert np.nanmean(p.trigger) > np.nanmean(trig)   # higher threshold -> later


def test_trace_matrix_flags_missing_and_odd_length_rows():
    good = np.arange(10, dtype=np.uint16)
    col = [good, None, good + 1, np.array([], np.uint16),
           np.arange(7, dtype=np.uint16)]
    mat, has = npsd.trace_matrix(col)
    assert mat.dtype == np.uint16 and mat.shape == (5, 10)
    assert has.tolist() == [True, False, True, False, False]
    assert (mat[1] == 0).all()
    np.testing.assert_array_equal(mat[2], good + 1)
    with pytest.raises(ValueError):
        npsd.trace_matrix([None, []])


def test_from_column_masks_rows_without_trace():
    mat, _ = make_matrix(n_each=3)
    col = list(mat) + [None]
    p = npsd.TracePSD.from_column(col, DT_NS).compute()
    assert p.n_events == 7
    assert p.valid[:6].all() and not p.valid[6]
    assert np.isnan(p.trigger[6])


def test_aligned_traces_share_the_trigger_and_baseline():
    mat, _ = make_matrix(n_each=5)
    p = npsd.TracePSD(mat, DT_NS).compute()
    al = p.aligned(np.arange(len(mat)))
    k = int(round(p.trigger_ns / DT_NS))
    # Aligned on the fast-filter trigger, every pulse is rising right after it.
    assert (al[:, k + 3] > al[:, k - 3]).all()
    assert np.abs(al[:, :50].mean(axis=1)).max() < 1.0   # baseline removed



def test_far_off_trigger_is_rejected():
    """A pulse near the end of its record has its tail cut off, so its Q_total
    and PSD are wrong: events whose trigger is >10 % of the trace length from
    the run's median trigger are marked invalid."""
    mat, _ = make_matrix()
    late = np.round(pulse(start=480.0, tau=25.0)).astype(np.uint16)
    mat = np.vstack([mat, late])
    p = npsd.TracePSD(mat, DT_NS).compute()
    assert p.n_misaligned == 1
    assert not p.valid[-1] and np.isnan(p.q_total[-1]) and np.isnan(p.psd[-1])
    assert p.valid[:-1].all()
    keep = npsd.TracePSD(mat, DT_NS, max_trigger_offset_frac=None).compute()
    assert keep.valid[-1] and keep.n_misaligned == 0


def test_trace_matrix_arrow_column_matches_object_column():
    """An Arrow-backed list column (as the API tab loads ``Trace``) stacks to
    the same matrix and flags as the per-event object path."""
    pa = pytest.importorskip("pyarrow")
    import pandas as pd
    good = np.arange(10, dtype=np.uint16)
    col = [good, None, good + 1, np.array([], np.uint16),
           np.arange(7, dtype=np.uint16)]
    arr = pa.array([None if t is None else t.tolist() for t in col],
                   type=pa.list_(pa.uint16()))
    ser = pd.Series(pd.arrays.ArrowExtensionArray(arr))
    for column in (ser, arr):
        mat, has = npsd.trace_matrix(column)
        ref_mat, ref_has = npsd.trace_matrix(col)
        assert mat.dtype == np.uint16
        np.testing.assert_array_equal(mat, ref_mat)
        np.testing.assert_array_equal(has, ref_has)
    with pytest.raises(ValueError):
        npsd.trace_matrix(pa.array([None, []], type=pa.list_(pa.uint16())))
