"""Tests for wara.combine_runs."""
import numpy as np
import pandas as pd
import pytest

from wara import combine_runs as cr


def _run(energies, dts):
    return pd.DataFrame({"channel": 4, "energy": np.asarray(energies, float),
                         "dt": np.asarray(dts, float)})


def test_run_spectra_energy_spans_zero_to_pooled_max():
    # A sparse high tail (e.g. ADC saturation at 65535) must stay on the
    # overlay, matching the main API view's [0, max] energy range.
    rng = np.random.default_rng(0)
    a = np.r_[rng.uniform(0, 5000, 1000), 65535.0]
    b = rng.uniform(0, 5000, 1000)
    runs = [("d1", 1, _run(a, np.zeros_like(a))),
            ("d2", 1, _run(b, np.zeros_like(b)))]
    spectra = cr.run_spectra(runs, 4, "energy", bins=64)
    width = 65535.0 / 64
    for s in spectra:
        assert s.energies[0] == pytest.approx(width / 2)
        assert s.energies[-1] == pytest.approx(65535.0 - width / 2)
    assert spectra[0].counts[-1] == 1
    assert spectra[0].counts.sum() == len(a)


def test_run_spectra_time_keeps_percentile_trim():
    dts = np.r_[np.linspace(0, 1e-7, 2000), 1e-3]  # one far outlier
    runs = [("d1", 1, _run(np.ones_like(dts), dts)),
            ("d2", 1, _run(np.ones_like(dts), dts))]
    spectra = cr.run_spectra(runs, 4, "time", bins=32)
    assert spectra[0].energies[-1] < 1e3  # ns; the 1e6 ns outlier is trimmed
