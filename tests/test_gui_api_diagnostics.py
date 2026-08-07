"""Offscreen GUI tests for the API diagnostics dialog's Binary tab
(``wara.gui.api_diagnostics.DiagnosticsDialog``).

Focus: the "Fast Filter" / "CFD" overlays on the Random-traces view, which
reconstruct the Pixie-16's internal trigger and timing filters offline with
:mod:`wara.pixie_trace_analysis`. Real runs live on a local data path, so the
list-mode reader and the two DSP-settings helpers are monkeypatched with a
small synthetic event table.
"""

import os

# Must be set before the first QApplication is created (during collection).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QWidget

from wara import helper_api, pixie_trace_analysis as pta
from wara.gui import api_diagnostics as diag_mod
from wara.gui.api_diagnostics import DiagnosticsDialog

# Per-channel DSP settings the stubbed readers hand back: (FL, FG, FastThresh)
# and CFDThresh, deliberately different per channel so the threshold lines and
# their legend labels have to cope with more than one value.
GEOM = {4: (10, 10, 500), 7: (10, 10, 900)}
CFD_THRESH = {4: 20, 7: 5}
N_EVENTS = 24


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _trace(rise_at, amplitude, n=180, baseline=1600.0, decay=60.0, seed=0):
    """A step rise with an exponential tail, like a real detector pulse."""
    rng = np.random.default_rng(seed)
    t = np.full(n, baseline, dtype=float)
    x = np.arange(n - rise_at, dtype=float)
    t[rise_at:] = baseline + amplitude * np.exp(-x / decay)
    return t + rng.normal(0, 2.0, n)


def _synthetic_traces():
    rows = []
    for i in range(N_EVENTS):
        ch = 4 if i % 2 == 0 else 7
        rows.append({
            "channel": ch,
            "energy": 500.0 + 40 * i,
            "trace": _trace(rise_at=50 + (i % 5), amplitude=900.0 + 20 * i, seed=i),
            "pileup": False,
            "CFD_error": False,
            "trace_flag": False,
            "CFD_fraction": 0.5,
        })
    return pd.DataFrame(rows)


class _StubController:
    """The two members DiagnosticsDialog touches on its controller."""

    def __init__(self, app):
        self.app = app

    def _flash_button(self, btn):
        pass


@pytest.fixture
def dlg(qapp, monkeypatch):
    df = _synthetic_traces()
    monkeypatch.setattr(helper_api, "read_trace_data", lambda *a, **k: df.copy())
    monkeypatch.setattr(diag_mod.helper_api, "read_trace_data",
                        lambda *a, **k: df.copy())
    # No settings file on disk -- serve the DSP registers from GEOM/CFD_THRESH.
    monkeypatch.setattr(diag_mod.pta, "read_fast_trigger_geometry",
                        lambda date, runnr, ch: GEOM[ch])
    monkeypatch.setattr(diag_mod.pta, "read_cfd_threshold",
                        lambda date, runnr, ch: CFD_THRESH[ch])
    # LGL alignment is a separate feature; keep it unavailable and quiet.
    monkeypatch.setattr(diag_mod.helper_api, "read_slow_filter_geometry",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError))
    # `_load_bin` pumps the event loop so the UI repaints before the (blocking)
    # read. Nothing here is under test, and running it offscreen tears down the
    # widgets other GUI test modules leave alive -- which aborts the process.
    monkeypatch.setattr(diag_mod.QApplication, "processEvents",
                        staticmethod(lambda *a, **k: None))

    parent = QWidget()
    d = DiagnosticsDialog(_StubController(parent))
    d.ed_bin_date.setText("2025-04-18")
    d.ed_bin_run.setText("8")
    d.cmb_bin_type.setCurrentText("Trace data")
    d.ed_bin_ntraces.setText("4")
    yield d
    d.close()
    parent.close()


def _rand_ax(d):
    return d._bin_ax["rand"]


def test_filter_boxes_start_disabled_and_enable_after_load(dlg):
    assert not dlg.cb_bin_ff.isEnabled() and not dlg.cb_bin_cfd.isEnabled()
    dlg._load_bin()
    assert dlg._bin_fast == {4: (10, 10, 500, 20), 7: (10, 10, 900, 5)}
    assert dlg.cb_bin_ff.isEnabled() and dlg.cb_bin_cfd.isEnabled()


def test_filter_boxes_disabled_when_settings_unavailable(dlg, monkeypatch):
    monkeypatch.setattr(diag_mod.pta, "read_fast_trigger_geometry",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError))
    dlg._load_bin()
    assert dlg._bin_fast == {}
    assert not dlg.cb_bin_ff.isEnabled() and not dlg.cb_bin_cfd.isEnabled()


def test_no_overlay_axes_until_a_box_is_checked(dlg):
    dlg._load_bin()
    dlg._sample_bin_traces()
    assert len(_rand_ax(dlg).lines) == 4     # one line per sampled trace
    assert dlg._bin_rand_twins == []
    assert len(dlg._bin_fig["rand"].axes) == 1


def test_fast_filter_overlay_plots_one_curve_per_trace(dlg):
    dlg._load_bin()
    dlg._sample_bin_traces()
    dlg.cb_bin_ff.setChecked(True)
    assert len(dlg._bin_rand_twins) == 1
    tw = dlg._bin_rand_twins[0]
    # 4 fast-filter curves + one dashed FastThresh line per distinct threshold
    n_thresh = len({GEOM[c][2] for c in dlg.df_bin_rand.channel})
    assert len(tw.lines) == 4 + n_thresh
    assert "fast filter" in dlg.lbl_bin_state.text()


def test_fast_filter_overlay_matches_pixie_trace_analysis(dlg):
    """The plotted curve is exactly what pixie_trace_analysis computes."""
    dlg._load_bin()
    dlg._sample_bin_traces()
    dlg.cb_bin_ff.setChecked(True)
    row = dlg.df_bin_rand.iloc[0]
    FL, FG, _ = GEOM[int(row.channel)]
    expected = pta.fast_filter(row.trace, FL, FG)[0]
    plotted = dlg._bin_rand_twins[0].lines[0].get_ydata()
    assert np.allclose(plotted, expected, equal_nan=True)
    # x axis is time in ns, not samples
    xs = dlg._bin_rand_twins[0].lines[0].get_xdata()
    assert xs[1] - xs[0] == dlg.NS_PER_PT


def test_threshold_lines_use_the_dsp_register_values(dlg):
    dlg._load_bin()
    dlg._sample_bin_traces()
    dlg.cb_bin_ff.setChecked(True)
    dlg.cb_bin_cfd.setChecked(True)
    ax_ff, ax_cfd = dlg._bin_rand_twins
    chans = {int(c) for c in dlg.df_bin_rand.channel}
    # Dashed horizontals sit at FastThresh / CFDThresh of the sampled channels.
    def _dashed(ax):
        return {ln.get_ydata()[0] for ln in ax.lines if ln.get_linestyle() == "--"}

    ff_levels, cfd_levels = _dashed(ax_ff), _dashed(ax_cfd)
    assert ff_levels == {GEOM[c][2] for c in chans}
    assert cfd_levels == {CFD_THRESH[c] for c in chans}
    labels = [t.get_text() for t in ax_ff.get_legend().get_texts()]
    assert any(lb.startswith("FastThresh") for lb in labels)
    assert any(lb.startswith("CFDThresh") for lb in labels)


def test_cfd_custom_firmware_weight_on_by_default(dlg):
    """The custom-firmware (w=0.3125) box defaults to checked, and the plotted
    CFD is reconstructed with that weight, not the stock w=1."""
    assert dlg.cb_bin_cfd_custom_w.isChecked()
    dlg._load_bin()
    dlg._sample_bin_traces()
    dlg.cb_bin_cfd.setChecked(True)
    ax_cfd = dlg._bin_rand_twins[0]
    row = dlg.df_bin_rand.iloc[0]
    expected = pta.cfd_trace(row.trace, w=pta.CFD_W_CUSTOM)[0]
    plotted = ax_cfd.lines[0].get_ydata()
    assert np.allclose(plotted, expected, equal_nan=True)
    # and NOT the stock w=1 reconstruction (the two differ where the CFD rises)
    stock = pta.cfd_trace(row.trace, w=pta.CFD_W)[0]
    assert not np.allclose(plotted, stock, equal_nan=True)


def test_cfd_stock_weight_when_custom_unchecked(dlg):
    """Unchecking the custom-firmware box reconstructs the CFD at stock w=1."""
    dlg._load_bin()
    dlg._sample_bin_traces()
    dlg.cb_bin_cfd_custom_w.setChecked(False)
    dlg.cb_bin_cfd.setChecked(True)
    ax_cfd = dlg._bin_rand_twins[0]
    row = dlg.df_bin_rand.iloc[0]
    expected = pta.cfd_trace(row.trace, w=pta.CFD_W)[0]
    plotted = ax_cfd.lines[0].get_ydata()
    assert np.allclose(plotted, expected, equal_nan=True)


def test_both_overlays_use_separate_twin_axes(dlg):
    dlg._load_bin()
    dlg._sample_bin_traces()
    dlg.cb_bin_ff.setChecked(True)
    dlg.cb_bin_cfd.setChecked(True)
    assert len(dlg._bin_rand_twins) == 2
    ax_ff, ax_cfd = dlg._bin_rand_twins
    # The CFD spine is pushed outward so the two right-hand axes don't overlap.
    assert ax_cfd.spines["right"].get_position()[0] == "outward"
    assert ax_ff.spines["right"].get_position() != ax_cfd.spines["right"].get_position()


def test_twin_axes_do_not_accumulate_across_redraws(dlg):
    dlg._load_bin()
    dlg._sample_bin_traces()
    dlg.cb_bin_ff.setChecked(True)
    dlg.cb_bin_cfd.setChecked(True)
    for _ in range(3):
        dlg._sample_bin_traces()
    assert len(dlg._bin_rand_twins) == 2
    assert len(dlg._bin_fig["rand"].axes) == 3   # base + two twins


def test_unchecking_removes_the_overlay_axes(dlg):
    dlg._load_bin()
    dlg._sample_bin_traces()
    dlg.cb_bin_ff.setChecked(True)
    dlg.cb_bin_cfd.setChecked(True)
    dlg.cb_bin_ff.setChecked(False)
    assert len(dlg._bin_rand_twins) == 1
    dlg.cb_bin_cfd.setChecked(False)
    assert dlg._bin_rand_twins == []
    assert len(dlg._bin_fig["rand"].axes) == 1
    assert len(_rand_ax(dlg).lines) == 4         # the raw traces survive


def test_fft_drops_the_overlays_and_restores_them(dlg):
    dlg._load_bin()
    dlg._sample_bin_traces()
    dlg.cb_bin_ff.setChecked(True)
    dlg.btn_bin_fft.setChecked(True)
    dlg._toggle_bin_fft()
    assert dlg._bin_rand_twins == []
    assert len(dlg._bin_fig["rand"].axes) == 1
    dlg.btn_bin_fft.setChecked(False)
    dlg._toggle_bin_fft()
    assert len(dlg._bin_rand_twins) == 1


def test_toggling_before_sampling_is_a_no_op(dlg):
    dlg._load_bin()
    dlg.cb_bin_ff.setChecked(True)
    assert dlg._bin_rand_twins == []
    assert "Sample some traces first" in dlg.lbl_bin_state.text()


def test_reload_without_settings_unchecks_the_boxes(dlg, monkeypatch):
    dlg._load_bin()
    dlg._sample_bin_traces()
    dlg.cb_bin_ff.setChecked(True)
    monkeypatch.setattr(diag_mod.pta, "read_fast_trigger_geometry",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError))
    dlg._load_bin()
    assert not dlg.cb_bin_ff.isChecked() and not dlg.cb_bin_cfd.isChecked()
    assert not dlg.cb_bin_ff.isEnabled()
