"""Tests for wara.helper_api trace alignment."""
import numpy as np
import pandas as pd
import pytest

from wara.helper_api import align_traces


def _pulse(n, rise_at, amplitude=1000.0, baseline=100.0, decay=15.0):
    t = np.full(n, baseline, dtype=float)
    x = np.arange(n - rise_at, dtype=float)
    t[rise_at:] = baseline + amplitude * np.exp(-x / decay)
    return t


@pytest.mark.parametrize("method", ["edge", "fast", "peak"])
def test_align_traces_handles_mixed_trace_lengths(method):
    # A run where channels record different trace lengths: the minority-length
    # channel must still be aligned instead of being left with NaN shifts.
    rows = ([{"channel": 1, "trace": _pulse(80, 30 + k % 3)} for k in range(4)]
            + [{"channel": 12, "trace": _pulse(200, 60 + k % 3)} for k in range(8)])
    out = align_traces(pd.DataFrame(rows), method=method)

    assert out["align_shift"].notna().all()
    for ch, n in ((1, 80), (12, 200)):
        g = out[out.channel == ch]
        assert all(len(t) == n for t in g["trace"])
        peaks = {int(np.argmax(t)) for t in g["trace"]}
        assert len(peaks) == 1          # every pulse lands on the same sample
