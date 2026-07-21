"""Tests for the Neutrons PSD tab: the wara.neutron_psd numerics and the
offscreen wara.gui.neutrons (NeutronsOptions / NeutronsPage / NeutronsController)
front-end, driven with synthetic two-population (fast/slow) trace data.

Covers: polarity auto-detection, baseline correction, MCA/PSD computation, the
validity mask, dataset load + full redraw, the MCA-span and PSD-box
cross-filters (AND semantics), marker-drag recompute, and clear-selection.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from wara import neutron_psd as npsd

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QMainWindow

from wara.gui.app import WaraApp


# ── Synthetic dataset ───────────────────────────────────────────────────────────
def make_traces(n=400, n_samples=300, negative=True, seed=1):
    """Two pulse populations sharing a rise but differing in tail length
    (short-tail ≈ gamma, long-tail ≈ neutron), on a common baseline offset."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 300.0, n_samples).astype(np.float32)
    t0 = 50.0
    traces = np.zeros((n, n_samples), dtype=np.float32)
    for i in range(n):
        long_tail = i % 2 == 0
        tau = 40.0 if long_tail else 12.0
        amp = rng.uniform(0.5, 2.0)
        pulse = np.where(t >= t0, amp * np.exp(-(t - t0) / tau), 0.0)
        rise = np.where((t >= t0 - 4) & (t < t0), amp * (t - (t0 - 4)) / 4.0, 0.0)
        trace = pulse + rise + rng.normal(0, 0.01, n_samples) + 0.3  # +0.3 baseline
        traces[i] = -trace if negative else trace
    mean = traces.mean(axis=0)
    return t, traces, mean


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def synth_npz(tmp_path):
    t, traces, mean = make_traces()
    p = tmp_path / "synthetic_traces.npz"
    np.savez(p, time_ns=t, A_traces=traces, A_mean=mean)
    return str(p)


# ── Compute module ──────────────────────────────────────────────────────────────
def test_orient_positive_flips_negative_pulses():
    t, traces, mean = make_traces(negative=True)
    oriented, sign = npsd.orient_positive(traces, reference=mean)
    assert sign == -1.0
    # After orientation the dominant excursion is positive.
    assert oriented.max() > abs(oriented.min())


def test_pipeline_separates_two_populations():
    t, traces, mean = make_traces()
    nt = npsd.NeutronTraces(t, traces, mean=mean).compute()
    assert nt.sign == -1.0
    assert nt.valid.sum() > 0
    # Even/odd traces were built with long/short tails → distinct PSD medians.
    long_tail = np.zeros(nt.n_traces, bool); long_tail[::2] = True
    psd_long = np.nanmedian(nt.psd[nt.valid & long_tail])
    psd_short = np.nanmedian(nt.psd[nt.valid & ~long_tail])
    assert psd_long != pytest.approx(psd_short, abs=0.05)


def test_baseline_correction_preserves_dtype_and_zeroes_pretrigger():
    t, traces, mean = make_traces()
    corr = npsd.baseline_correct(-traces, t, pre_trigger_ns=40.0)
    assert corr.dtype == traces.dtype  # stays float32 (memory bound)
    pre = corr[:, t < 40.0]
    assert abs(pre.mean()) < 1e-3


def test_threshold_filters_low_pulses():
    t, traces, mean = make_traces()
    nt = npsd.NeutronTraces(t, traces, mean=mean)
    nt.set_params(threshold_v=0.0); nt.compute()
    n_low = int(nt.valid.sum())
    nt.set_params(threshold_v=5.0); nt.compute()  # above every pulse
    assert nt.valid.sum() < n_low


def test_from_npz_roundtrip(synth_npz):
    nt = npsd.NeutronTraces.from_npz(synth_npz).compute()
    assert nt.n_traces == 400
    assert nt.energy.shape == (400,)


def test_from_npz_rejects_bad_archive(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, foo=np.arange(3))
    with pytest.raises(ValueError):
        npsd.NeutronTraces.from_npz(str(p))


@pytest.mark.parametrize("ext", ["npy", "txt", "csv"])
def test_from_file_loads_plain_formats(tmp_path, ext):
    """.npy/.txt/.csv hold one trace per row; the time axis is synthesised."""
    _, traces, _ = make_traces(n=20, n_samples=64)
    p = tmp_path / f"traces.{ext}"
    if ext == "npy":
        np.save(p, traces)
    else:
        np.savetxt(p, traces, delimiter="," if ext == "csv" else " ")
    nt = npsd.NeutronTraces.from_file(str(p)).compute()
    assert nt.n_traces == 20
    assert nt.time_ns.shape == (64,)
    assert np.allclose(nt.time_ns, np.arange(64))  # sample-index time axis
    assert nt.energy.shape == (20,)


def test_from_file_reads_nonstandard_npz(tmp_path):
    """A .npz without the PicoScope keys falls back to its first 2-D array."""
    _, traces, _ = make_traces(n=15, n_samples=48)
    p = tmp_path / "plain.npz"
    np.savez(p, whatever=traces)
    nt = npsd.NeutronTraces.from_file(str(p)).compute()
    assert nt.n_traces == 15 and nt.time_ns.shape == (48,)


def test_from_file_rejects_unknown_extension(tmp_path):
    p = tmp_path / "traces.xyz"
    p.write_text("nope")
    with pytest.raises(ValueError):
        npsd.load_trace_file(str(p))


# ── GUI controller ──────────────────────────────────────────────────────────────
@pytest.fixture
def neutrons(qapp):
    """The Neutrons controller from a real WaraApp, with a synthetic dataset."""
    w = WaraApp()
    t, traces, mean = make_traces()
    nt = npsd.NeutronTraces(t, traces, mean=mean).compute()
    w.neutrons.nt = nt
    # Simulate the user having committed the gates so the MCA/PSD panels are
    # populated (the default post-load state draws traces only).
    w.neutrons._analysis_ready = True
    w.neutrons._redraw()
    yield w.neutrons
    w.close()


def test_tab_registered():
    from wara.gui.theme import NAV_SECTIONS, TABS_WITH_OPTIONS
    names = [n for n, _ in NAV_SECTIONS]
    assert "Neutrons" in names
    # Sits directly above Planetary.
    assert names.index("Neutrons") == names.index("Planetary") - 1
    assert "Neutrons" in TABS_WITH_OPTIONS


def test_load_defers_mca_psd_until_first_selection(qapp):
    """On load only the Traces panel is drawn; the MCA/PSD selectors attach
    (panels populate) only after the user commits the gate markers."""
    w = WaraApp()
    t, traces, mean = make_traces()
    nt = npsd.NeutronTraces(t, traces, mean=mean).compute()
    c = w.neutrons
    c.nt = nt
    c._energy_range = (None, None); c._psd_region = None; c._psd_selections = []
    c._analysis_ready = False
    c._redraw()
    # Traces are shown, but MCA/PSD panels are pending: no selectors yet.
    assert c._analysis_ready is False
    assert c.page._span is None and c.page._rect is None
    assert c.opts.lbl_sel.text() == "—"
    # A Traces-panel selection (committing the gates) populates the panels.
    c._on_params(nt.threshold_v, nt.gate_start_ns, nt.prompt_end_ns, nt.tail_end_ns)
    assert c._analysis_ready is True
    assert c.page._span is not None and c.page._rect is not None
    assert "of" in c.opts.lbl_sel.text()
    w.close()


def test_redraw_reports_counts(neutrons):
    nt = neutrons.nt
    assert "of" in neutrons.opts.lbl_sel.text()
    assert neutrons.opts.lbl_gate.text().endswith("ns")


def test_energy_span_filter(neutrons):
    nt = neutrons.nt
    e = nt.energy[nt.valid]
    lo, hi = np.percentile(e, [40, 60])
    neutrons._on_energy_range(float(lo), float(hi))
    mask = neutrons._selection_mask()
    assert mask.sum() < nt.valid.sum()
    assert np.all((nt.energy[mask] >= lo) & (nt.energy[mask] <= hi))


def test_psd_box_and_span_and_together(neutrons):
    """MCA span and PSD box AND together to narrow the selection further."""
    nt = neutrons.nt
    e = nt.energy[nt.valid]; p = nt.psd[nt.valid]
    neutrons._on_energy_range(float(np.percentile(e, 20)), float(np.percentile(e, 80)))
    after_span = neutrons._selection_mask().sum()
    neutrons._on_psd_region((float(np.percentile(e, 30)), float(np.percentile(e, 70)),
                             float(np.percentile(p, 50)), float(np.percentile(p, 99))))
    after_both = neutrons._selection_mask().sum()
    assert after_both <= after_span


def test_marker_drag_recomputes(neutrons):
    nt = neutrons.nt
    before = nt.psd.copy()
    neutrons._on_params(0.0, nt.gate_start_ns, nt.prompt_end_ns + 20.0,
                        nt.tail_end_ns)
    # Moving the prompt/tail boundary changes the PSD of the pulses.
    assert not np.allclose(before, neutrons.nt.psd, equal_nan=True)


def test_clear_selection(neutrons):
    nt = neutrons.nt
    neutrons._on_energy_range(0.1, 0.2)
    neutrons._clear_selection()
    assert neutrons._energy_range == (None, None)
    assert neutrons._psd_region is None
    assert neutrons._selection_mask().sum() == nt.valid.sum()


# ── Armed "PSD selections" mode ─────────────────────────────────────────────────
def _arm(neutrons, on):
    neutrons.opts.btn_psd_select.setChecked(on)  # fires toggled -> _on_arm


def test_arming_disables_mca_span_and_clears(neutrons):
    neutrons._on_energy_range(0.1, 0.2)
    _arm(neutrons, True)
    assert neutrons._armed and neutrons.page.armed
    assert neutrons._energy_range == (None, None)
    # The MCA span is ignored while armed.
    neutrons._on_energy_range(0.1, 0.2)
    assert neutrons._energy_range == (None, None)


def test_multiple_psd_selections_get_distinct_colors(neutrons):
    from wara.gui import theme as T
    nt = neutrons.nt
    e = nt.energy[nt.valid]; p = nt.psd[nt.valid]
    _arm(neutrons, True)
    neutrons._on_psd_add((float(np.percentile(e, 10)), float(np.percentile(e, 40)),
                          float(np.percentile(p, 10)), float(np.percentile(p, 50))))
    neutrons._on_psd_add((float(np.percentile(e, 50)), float(np.percentile(e, 90)),
                          float(np.percentile(p, 50)), float(np.percentile(p, 90))))
    assert len(neutrons._psd_selections) == 2
    colors = [s["color"] for s in neutrons._psd_selections]
    assert colors[0] != colors[1]
    assert colors == [T.OVERLAY_COLORS[0], T.OVERLAY_COLORS[1]]
    # Each selection becomes its own trace group and MCA/PSD overlay.
    groups = neutrons._selection_groups()
    assert len(groups) == 2
    assert len(neutrons._psd_rects()) == 2
    # The union drives the selected count.
    assert neutrons._selection_mask().sum() > 0


def test_selection_mask_is_union_when_armed(neutrons):
    nt = neutrons.nt
    e = nt.energy[nt.valid]; p = nt.psd[nt.valid]
    _arm(neutrons, True)
    r1 = (float(np.percentile(e, 10)), float(np.percentile(e, 30)),
          float(np.percentile(p, 10)), float(np.percentile(p, 40)))
    r2 = (float(np.percentile(e, 60)), float(np.percentile(e, 90)),
          float(np.percentile(p, 60)), float(np.percentile(p, 95)))
    neutrons._on_psd_add(r1)
    only1 = neutrons._selection_mask().sum()
    neutrons._on_psd_add(r2)
    both = neutrons._selection_mask().sum()
    assert both >= only1
    assert both == ((neutrons._region_mask(r1) | neutrons._region_mask(r2))
                    & nt.valid).sum()


def test_disarming_restores_single_mode(neutrons):
    nt = neutrons.nt
    _arm(neutrons, True)
    neutrons._on_psd_add((0.0, 1.0, -1.0, 1.0))
    _arm(neutrons, False)
    assert not neutrons._armed and not neutrons.page.armed
    assert neutrons._psd_selections == []
    # Back to amplitude-coloured single group covering the cross-filter.
    groups = neutrons._selection_groups()
    assert len(groups) == 1 and groups[0][1] is None
    assert neutrons._selection_mask().sum() == nt.valid.sum()
