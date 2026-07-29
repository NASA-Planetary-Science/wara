"""Offscreen GUI regression tests for the wara API tab
(wara.gui.api: ApiOptions / ApiPage / ApiController).

The real parquet loader needs run files on a local data path, so the data
access (``read_parquet_api.read_parquet_file`` and the three apicalc settings
helpers) is monkeypatched with a small synthetic event table. This exercises the
controller logic — panel build, live energy/time/X-Y filters, vmax, reset and
send-to-Spectrum — without any data files.
"""

import os
from pathlib import Path

# Must be set before the first QApplication is created (during collection).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QDialog

from wara import read_parquet_api, apicalc
from wara.gui.app import WaraApp
from wara.gui import api as api_mod
from wara.gui.theme import NAV_SECTIONS


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _StubBrowser:
    """Stand-in for the dialog's QWebEngineView so tests never load QtWebEngine."""

    def __init__(self):
        self.urls = []

    def setUrl(self, url):
        self.urls.append(url)


class _Stub3DDialog(QWidget):
    """Drop-in for Api3DDialog with the interface the controller touches, minus
    the QtWebEngine browser (offscreen CI can't render WebEngine)."""

    DEFAULTS = {"no_bins": 12, "isomin": 0.1, "isomax": 0.8,
                "opacity": 0.1, "surfcount": 10}
    # Mirrors Api3DDialog.recon_params() with its default field values
    # (sample_z is set here, unlike the blank GUI default, so the depth
    # calibration tests exercise the auto-fit path).
    RECON = {"det_pos": (-18.0, 0.0, -25.0), "z_t": 6.7, "beam_kev": 50.0,
             "sample_z": -25.0, "toffset_ns": None, "use_det": True,
             "r2_correction": False, "r2_power": 2.0}

    def __init__(self, parent=None):
        super().__init__()
        self.btn_plot = QPushButton(self)
        self.status = QLabel(self)
        self.browser = _StubBrowser()

    def value(self, attr, cast):
        return cast(self.DEFAULTS[attr])

    def recon_params(self):
        return dict(self.RECON)


class _StubSelDialog:
    """Stand-in for EnergySelectionDialog: accepts immediately with a fixed
    label/colour (no modal exec_)."""

    LABEL = "Fe"
    COLOR = "#ff0000"

    def __init__(self, emin, emax, color, n_events, parent=None):
        self.emin, self.emax = emin, emax

    def exec_(self):
        return QDialog.Accepted

    def label(self):
        return self.LABEL

    def color(self):
        return self.COLOR


def _synthetic_events(n=4000, seed=0):
    """A minimal API event table: the columns the controller reads plus the
    A/B/C/D quadrants calc_own_pos needs to reconstruct X2/Y2."""
    rng = np.random.default_rng(seed)
    energy = rng.uniform(0, 6000, n)
    return pd.DataFrame({
        "dt": rng.normal(0.0, 15.0, n),          # already in ns post-scale
        "energy": energy,
        "energy_orig": energy,
        "A": rng.uniform(1, 100, n),
        "B": rng.uniform(1, 100, n),
        "C": rng.uniform(1, 100, n),
        "D": rng.uniform(1, 100, n),
    })


def _synthetic_flat_events(n=4000, seed=1, with_alpha=False):
    """A flat-field (ch 9) event table: only the A/B/C/D quadrants for the X-Y
    alpha map, with NO gamma energy/dt columns -- matching the on-disk flat
    run. With *with_alpha*, add the "alpha" energy column that triggers the
    split (alpha spectrum + X-Y) view."""
    rng = np.random.default_rng(seed)
    cols = {
        "A": rng.uniform(1, 100, n),
        "B": rng.uniform(1, 100, n),
        "C": rng.uniform(1, 100, n),
        "D": rng.uniform(1, 100, n),
    }
    if with_alpha:
        cols["alpha"] = rng.uniform(0, 4000, n)
    return pd.DataFrame(cols)


@pytest.fixture
def api(qapp, monkeypatch):
    df = _synthetic_events()
    # The loader multiplies dt by 1e9 (s→ns); feed seconds so the result is ns.
    monkeypatch.setattr(read_parquet_api, "read_parquet_file",
                        lambda *a, **k: df.copy().assign(dt=df["dt"] / 1e9))
    monkeypatch.setattr(apicalc, "get_total_time", lambda *a, **k: 14386.0)
    monkeypatch.setattr(apicalc, "get_total_counts", lambda *a, **k: 2.5e9)
    monkeypatch.setattr(apicalc, "calculate_neutron_yield", lambda *a, **k: 4.3e6)

    w = WaraApp()
    idx = [name for name, _ in NAV_SECTIONS].index("API")
    w.nav_group.button(idx).setChecked(True)
    w._on_nav(idx)
    c = w.api
    c.opts.ed_date.setText("2023-07-02")
    c.opts.ed_run.setText("91")
    c.opts.ed_ch.setText("5")
    yield w, c
    w.close()


def test_load_builds_panels_and_info(api):
    _w, c = api
    c._load()
    assert c.page.ax_spe is not None and c.page.ax_dt is not None and c.page.ax_xy is not None
    assert c.xkey == "X2" and c.ykey == "Y2" and c.ekey == "energy_orig"
    assert c.gam is not None and len(c.gam) == c.ebins
    assert c.df_current.shape[0] == 4000
    # Interactive cuts are off by default — selectors attach only when toggled on.
    assert len(c._selectors) == 0
    c.opts.btn_interactive.setChecked(True)
    assert len(c._selectors) == 3
    # No colorbar: the X-Y intensity is read via the hover lookup instead.
    assert c.page.xy_offsets is not None and len(c.page.xy_offsets) > 0
    assert c.page.xy_values is not None and len(c.page.xy_values) > 0
    info = c.opts.lbl_info.text()
    assert "Neutron yield" in info
    # The event count moved off the X-Y plot title into RUN INFO. The readout is
    # rich text (bold labels, coloured values), so label and value aren't adjacent.
    assert "Processed counts" in info and "4,000" in info
    # Values are wrapped in coloured spans; labels are bold.
    assert "<b>" in info and "color:" in info
    assert c.page.ax_xy.get_title() == ""


def _flat_field_ctrl(qapp, monkeypatch, df):
    """Build the app on the API tab, stub the loader to return *df*, and seed the
    ch-9 flat-field run fields. Returns (window, controller)."""
    monkeypatch.setattr(read_parquet_api, "read_parquet_file",
                        lambda *a, **k: df.copy())
    monkeypatch.setattr(apicalc, "get_total_time", lambda *a, **k: 14386.0)
    monkeypatch.setattr(apicalc, "get_total_counts", lambda *a, **k: 2.5e9)
    monkeypatch.setattr(apicalc, "calculate_neutron_yield", lambda *a, **k: 4.3e6)

    w = WaraApp()
    idx = [name for name, _ in NAV_SECTIONS].index("API")
    w.nav_group.button(idx).setChecked(True)
    w._on_nav(idx)
    c = w.api
    c.opts.ed_date.setText("2025-07-13")
    c.opts.ed_run.setText("14")
    c.opts.ed_ch.setText("9")
    return w, c


def test_flat_field_ch9_plots_only_xy(qapp, monkeypatch):
    """Loading ch 9 (flat field) with no alpha column builds only the X-Y map
    and never looks for a (missing) energy column. Regression: _configure_keys
    used to fall through to df["energy"].max() for flat runs and raise
    KeyError."""
    w, c = _flat_field_ctrl(qapp, monkeypatch, _synthetic_flat_events())
    try:
        c._load()  # must not raise KeyError on the missing energy column
        assert c.flat_field is True and c._flat_alpha is False
        # Only the X-Y map exists; the energy/dt/alpha panels are absent.
        assert c.page.ax_xy is not None
        assert c.page.ax_spe is None and c.page.ax_dt is None
        assert c.page.ax_alpha is None
        # Energy keys are cleared; the X-Y map still reconstructs X2/Y2.
        assert c.ekey is None and c._chan_base is None
        assert c.xkey == "X2" and c.ykey == "Y2"
        assert c.page.xy_offsets is not None and len(c.page.xy_offsets) > 0
    finally:
        w.close()


def test_flat_field_ch9_alpha_splits_view(qapp, monkeypatch):
    """A flat-field run carrying an "alpha" column splits the canvas: the alpha
    energy spectrum (left) alongside the X-Y map (right). The alpha spectrum is
    binned with the shared Energy-bins box; the alpha column drives the energy
    axis (ekey) for interactive filtering."""
    w, c = _flat_field_ctrl(qapp, monkeypatch,
                            _synthetic_flat_events(with_alpha=True))
    try:
        c._load()
        assert c.flat_field is True and c._flat_alpha is True
        # Split view: alpha spectrum + X-Y map, no gamma energy/dt panels.
        assert c.page.ax_alpha is not None and c.page.ax_xy is not None
        assert c.page.ax_spe is None and c.page.ax_dt is None
        # The alpha spectrum drew a curve; X-Y map still reconstructs X2/Y2.
        assert len(c.page.ax_alpha.get_lines()) == 1
        assert c.xkey == "X2" and c.ykey == "Y2"
        assert c.page.xy_offsets is not None and len(c.page.xy_offsets) > 0
        # The alpha column becomes the energy axis so span cuts filter on it.
        assert c.ekey == "alpha" and c.erange[0] == 0.0
        # The shared Energy-bins box re-bins the alpha spectrum.
        c.opts.ed_ebins.setText("256")
        c._apply_bins()
        assert c.ebins == 256
        assert len(c.page.ax_alpha.get_lines()) == 1
    finally:
        w.close()


def test_flat_field_alpha_interactive_cross_filter(qapp, monkeypatch):
    """With both alpha and X-Y present, interactive cuts cross-filter: an alpha
    energy window updates the X-Y map, and an X-Y region updates the alpha
    spectrum. Undo restores the prior state."""
    w, c = _flat_field_ctrl(qapp, monkeypatch,
                            _synthetic_flat_events(with_alpha=True))
    try:
        c._load()
        n0 = c.df_current.shape[0]
        # Interactive mode attaches an alpha span + an X-Y rectangle selector.
        c.opts.btn_interactive.setChecked(True)
        assert len(c._selectors) == 2
        # Alpha energy window: fewer events, X-Y reflects the cut, markers drawn.
        amax = c.erange[1]
        c.apply_energy_filter(0.2 * amax, 0.6 * amax)
        n_alpha = c.df_current.shape[0]
        assert 0 < n_alpha < n0
        assert c._cut_energy is not None
        # X-Y region further narrows and refreshes the alpha spectrum.
        c.apply_xy_filter(-0.3, 0.3, -0.3, 0.3)
        assert c.df_current.shape[0] <= n_alpha
        assert c._cut_xy is not None
        assert len(c.page.ax_alpha.get_lines()) >= 1
        # Undo rolls back the X-Y cut.
        c._undo()
        assert c.df_current.shape[0] == n_alpha
    finally:
        w.close()


def test_flat_field_no_alpha_not_interactive(qapp, monkeypatch):
    """A flat-field run without an alpha column has nothing to cross-filter, so
    enabling interactive cuts attaches no selectors."""
    w, c = _flat_field_ctrl(qapp, monkeypatch, _synthetic_flat_events())
    try:
        c._load()
        assert c._flat_alpha is False
        c.opts.btn_interactive.setChecked(True)
        assert len(c._selectors) == 0
    finally:
        w.close()


def test_bin_boxes_prepopulated_with_defaults(api):
    _w, c = api
    assert c.opts.ed_ebins.text() == str(api_mod.DEFAULT_EBINS)
    assert c.opts.ed_tbins.text() == str(api_mod.DEFAULT_TBINS)
    assert c.opts.ed_xybins.text() == str(api_mod.DEFAULT_HEXBINS)


def test_apply_bins_rebins_panels(api):
    _w, c = api
    c._load()
    c.opts.ed_ebins.setText("1024")
    c.opts.ed_tbins.setText("256")
    c.opts.ed_xybins.setText("40")
    c._apply_bins()
    assert (c.ebins, c.tbins, c.hexbins) == (1024, 256, 40)
    # The energy histogram is rebinned to the new count.
    assert len(c.gam) == 1024


def test_apply_bins_rejects_bad_value(api):
    _w, c = api
    c._load()
    c.opts.ed_ebins.setText("oops")
    c._apply_bins()
    # Invalid entry aborts without touching the bin counts.
    assert c.ebins == api_mod.DEFAULT_EBINS


def test_grids_off_and_dt_bin_edges(api):
    from matplotlib.colors import to_rgba
    _w, c = api
    c._load()
    # No grid lines on any of the three panels.
    for ax in (c.page.ax_spe, c.page.ax_dt, c.page.ax_xy):
        assert not any(gl.get_visible()
                       for gl in ax.get_xgridlines() + ax.get_ygridlines())
    dt = c.page.ax_dt
    assert dt.patches, "dt histogram should have bar patches"
    # Bars carry dark (plot-background) edges with a non-zero line width.
    # (Compare RGB only; the hist alpha folds into the edge's alpha channel.)
    assert to_rgba(dt.patches[0].get_edgecolor())[:3] == to_rgba(api_mod.API_PLOT_BG)[:3]
    assert dt.patches[0].get_linewidth() > 0


def test_xy_hover_reports_hexagon_count(api):
    _w, c = api
    c._load()
    page = c.page
    # Hovering a drawn hexagon's centre reports that hexagon's count.
    cx, cy = page.xy_offsets[0]
    cnt = page.xy_count_at(cx, cy)
    assert cnt == pytest.approx(page.xy_values[0])
    # Far outside the X-Y plane (beyond the pick radius) reports nothing.
    assert page.xy_count_at(1e6, 1e6) is None


def test_xy_hover_undoes_log_scale(api):
    _w, c = api
    c._load()
    c.opts.cb_xy_log.setChecked(True)        # array stores log10(count)
    page = c.page
    assert page.xy_log is True
    cx, cy = page.xy_offsets[0]
    cnt = page.xy_count_at(cx, cy)
    assert cnt == pytest.approx(10 ** page.xy_values[0])


def test_processed_count_tracks_filters(api):
    _w, c = api
    c._load()
    c.apply_energy_filter(0, 3000)
    n = c.df_current.shape[0]
    info = c.opts.lbl_info.text()
    assert "Processed counts" in info and f"{n:,}" in info
    assert c.page.ax_xy.get_title() == ""


def test_energy_filter_reduces_events(api):
    _w, c = api
    c._load()
    c.apply_energy_filter(0, 3000)
    assert c.df_current.shape[0] < 4000
    assert c.en_flag == 1


def test_time_and_xy_filters(api):
    _w, c = api
    c._load()
    before = c.df_current.shape[0]
    c.apply_t_filter(-10, 10)
    assert c.df_current.shape[0] < before
    assert c.dt_flag == 1
    n_after_t = c.df_current.shape[0]
    c.apply_xy_filter(-0.2, 0.2, -0.2, 0.2)
    assert c.df_current.shape[0] <= n_after_t
    assert c.xy_flag == 1


def test_filter_clears_toolbar_home_history(api):
    """After a filter redraws a panel, the nav toolbar's pre-filter 'home' view
    must be dropped so pressing Home can't restore the stale (wider) range and
    squash the filtered dt distribution."""
    _w, c = api
    c._load()
    tb = c.page.toolbar
    tb.push_current()                       # capture a home view, as zoom/pan would
    assert tb._nav_stack() is not None      # something is on the history stack
    c.apply_energy_filter(0, 3000)          # data changed → history must be cleared
    assert tb._nav_stack() is None          # 'home' no longer holds the stale view


def test_vmax_forces_linear_scale(api):
    _w, c = api
    c._load()
    c.opts.cb_xy_log.setChecked(True)
    c.opts.ed_vmax.setText("50")
    c._apply_vmax()
    assert c.vmax == 50.0
    assert c.opts.cb_xy_log.isChecked() is False


def test_reset_restores_full_data(api):
    _w, c = api
    c._load()
    c.apply_energy_filter(0, 1000)
    assert c.df_current.shape[0] < 4000
    c._reset()
    assert c.df_current.shape[0] == 4000
    assert c.en_flag == 0 and c.dt_flag == 0 and c.xy_flag == 0


def test_send_to_spectrum_stays_on_tab_and_blinks(api):
    w, c = api
    c._load()
    assert c.opts.btn_send.objectName() == "open_btn"
    c._send_to_spectrum()
    assert w.spect is not None
    assert len(w.spect.counts) == c.ebins
    # Stays on the API tab; the button keeps its label/objectName and only blinks
    # (a transient inline stylesheet, reverted by a QTimer) to confirm the send.
    assert type(w.stack.currentWidget()).__name__ == "ApiPage"
    assert c.opts.btn_send.objectName() == "open_btn"
    assert c.opts.btn_send.text() == api_mod.SEND_DEFAULT_TEXT
    assert "background-color" in c.opts.btn_send.styleSheet()


def test_reset_uses_cached_df_without_reloading(api, monkeypatch):
    w, c = api
    calls = {"n": 0}
    orig = read_parquet_api.read_parquet_file
    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    monkeypatch.setattr(read_parquet_api, "read_parquet_file", counting)
    c._load()
    assert calls["n"] == 1
    c.apply_energy_filter(0, 3000)
    assert c.df_current.shape[0] < 4000
    c._reset()
    assert calls["n"] == 1                      # reset did NOT re-read the file
    assert c.df_current.shape[0] == 4000        # original dataframe restored
    assert c.en_flag == 0 and c.dt_flag == 0 and c.xy_flag == 0
    # Reset flashes the button brighter (a transient inline stylesheet).
    assert c.opts.btn_reset.styleSheet() != ""


def test_send_twice_numbers_duplicate_names(api):
    w, c = api
    c._load()
    w.spectrum_opts.cb_keep.setChecked(True)    # keep previous spectra as overlays
    c._send_to_spectrum()
    assert w._active_name == "API spectrum"
    c._send_to_spectrum()
    assert w._active_name == "API spectrum 2"
    c._send_to_spectrum()
    assert w._active_name == "API spectrum 3"


def test_long_name_is_elided_with_full_tooltip(api):
    w, _c = api
    long_name = "📄 " + "WidthOverflow" * 30
    w._set_file_label(long_name)
    assert w._file_label.text() != long_name          # shortened to fit
    assert len(w._file_label.text()) < len(long_name)
    assert w._file_label.toolTip() == long_name        # full name on hover


def test_3d_view_guards_before_load(api):
    _w, c = api
    c._open_3d()                       # no data yet → no dialog, just a status nudge
    assert c._api3d_dlg is None


def test_3d_view_renders_volume(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "Api3DDialog", _Stub3DDialog)
    c._open_3d()
    assert isinstance(c._api3d_dlg, _Stub3DDialog)
    c._create_plot_3d()
    # A temp HTML file was produced and handed to the (stub) browser.
    assert c._api3d_tmp is not None and os.path.exists(c._api3d_tmp)
    assert len(c._api3d_dlg.browser.urls) == 1
    assert "Rendered" in c._api3d_dlg.status.text()
    # A second render swaps the temp file (no accumulation).
    first = c._api3d_tmp
    c._create_plot_3d()
    assert not os.path.exists(first)
    assert c._api3d_tmp != first
    c._cleanup_3d_tmp()


def test_3d_xyz_reconstruction_is_finite(api):
    _w, c = api
    c._load()
    X, Y, Z = c._xyz()
    assert len(X) == len(Y) == len(Z) == c.df_current.shape[0]
    assert np.all(np.isfinite(np.asarray(X)))
    assert np.all(np.isfinite(np.asarray(Z)))


def test_3d_full_model_calibrates_depth_to_sample_z(api, monkeypatch):
    """With the auto-fitted t offset, the front (shallow) edge of the depth
    distribution lands near the dialog's Sample Z depth -- the front-face
    promise, in cm, so the ns/seconds unit handling in the controller must be
    right."""
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "Api3DDialog", _Stub3DDialog)
    c._open_3d()
    recon = c._api3d_dlg.recon_params()
    toff = apicalc.fit_toffset(
        c.df_current[c._dt_key], z0=recon["sample_z"], z_t=recon["z_t"],
        det_pos=recon["det_pos"], dt_unit="ns")
    # the promise: an on-axis event at the rising-edge dt reconstructs to
    # Sample Z. Find the edge the same way fit_toffset does, append central
    # probe events at that dt, and check where they land (in cm, so the
    # ns/seconds unit handling in the controller must be right too).
    dt = c.df_current[c._dt_key].to_numpy(dtype=float)
    lo, hi = np.percentile(dt, [0.2, 99.8])
    h, ed = np.histogram(dt, bins=400, range=(lo, hi))
    ctr = 0.5 * (ed[:-1] + ed[1:])
    imax = int(np.argmax(h))
    half = h[imax] / 2.0
    i = int(np.flatnonzero(h[: imax + 1] < half)[-1])
    frac = (half - h[i]) / max(h[i + 1] - h[i], 1)
    dt_edge = ctr[i] + frac * (ctr[i + 1] - ctr[i])
    probe = c.df_current.iloc[:3].copy()
    probe["X2"] = 0.0
    probe["Y2"] = 0.0
    probe[c._dt_key] = dt_edge
    c.df_current = pd.concat([c.df_current, probe], ignore_index=True)
    _, _, Z = c._xyz(recon=recon, toffset_ns=toff)
    Z = np.asarray(Z, dtype=float)
    assert np.isfinite(Z[-3:]).all()
    assert np.allclose(Z[-3:], recon["sample_z"], atol=8.0)   # cm


def test_3d_toffset_modes(api, monkeypatch):
    """Sample Z and t offset are both optional: value in t offset wins; blank
    t offset with a Sample Z auto-fits; both blank trusts dt (offset 0)."""
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "Api3DDialog", _Stub3DDialog)
    c._open_3d()
    calls = []
    real_fit = apicalc.fit_toffset
    monkeypatch.setattr(apicalc, "fit_toffset",
                        lambda *a, **k: (calls.append(1), real_fit(*a, **k))[1])
    # both blank: dt trusted as calibrated, no auto fit
    c._api3d_dlg.RECON = dict(_Stub3DDialog.RECON, sample_z=None,
                              toffset_ns=None)
    c._create_plot_3d()
    assert not calls
    assert "dt used as calibrated" in c._api3d_dlg.status.text()
    # manual offset only: used directly, still no auto fit
    c._api3d_dlg.RECON = dict(_Stub3DDialog.RECON, sample_z=None,
                              toffset_ns=3.0)
    c._create_plot_3d()
    assert not calls
    assert "auto t offset" not in c._api3d_dlg.status.text()
    # Sample Z given, t offset blank: auto fit kicks in
    c._api3d_dlg.RECON = dict(_Stub3DDialog.RECON, toffset_ns=None)
    c._create_plot_3d()
    assert calls
    assert "auto t offset" in c._api3d_dlg.status.text()
    c._cleanup_3d_tmp()


def test_3d_r2_correction_weights_volume(api, monkeypatch):
    """The 1/r² correction renders (status note present) and actually changes
    the volume histogram relative to the uncorrected render."""
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "Api3DDialog", _Stub3DDialog)
    c._open_3d()
    import plotly.graph_objs as go
    dlg = c._api3d_dlg
    recon_off = dict(_Stub3DDialog.RECON)
    recon_on = dict(_Stub3DDialog.RECON, r2_correction=True)
    t_off = c._volume_trace(go, c.df_current, 12, 0.1, 0.8, 0.1, 10,
                            recon=recon_off, toffset_ns=0.0)
    t_on = c._volume_trace(go, c.df_current, 12, 0.1, 0.8, 0.1, 10,
                           recon=recon_on, toffset_ns=0.0)
    assert t_off is not None and t_on is not None
    assert not np.allclose(np.asarray(t_off.value), np.asarray(t_on.value))
    dlg.RECON = recon_on
    c._create_plot_3d()
    assert "1/r² correction" in dlg.status.text()
    c._cleanup_3d_tmp()


def test_3d_view_empty_data_sets_status(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "Api3DDialog", _Stub3DDialog)
    c._open_3d()
    c.apply_energy_filter(1e9, 1e9 + 1)      # filter everything out
    assert c.df_current.shape[0] == 0
    c._create_plot_3d()                       # must not raise
    assert "No events" in c._api3d_dlg.status.text()
    assert c._api3d_tmp is None               # nothing rendered


def test_add_energy_selection_stores_cut_without_filtering(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._arm_selection()
    assert c._arming_selection is True
    c._on_energy_span(0, 3000)              # armed → creates a selection
    assert c._arming_selection is False     # disarmed after the drag
    assert len(c.selections) == 1
    sel = c.selections[0]
    assert sel["label"] == "Fe" and sel["color"] == "#ff0000"
    assert 0 < sel["df"].shape[0] <= 4000
    # A selection is non-destructive: the underlying data is untouched.
    assert c.df_current.shape[0] == 4000
    assert c.en_flag == 0


def test_unarmed_energy_drag_still_filters(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c.opts.btn_interactive.setChecked(True)  # interactive on but not armed for a selection
    c._on_energy_span(0, 3000)              # not armed → filters as before
    assert c.df_current.shape[0] < 4000
    assert c.en_flag == 1
    assert c.selections == []


def test_plot_selections_overlays_all_panels(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._arm_selection(); c._on_energy_span(0, 1000)
    c._arm_selection(); c._on_energy_span(1000, 2000)
    c._plot_selections()
    # Energy panel gets a legend entry per selection.
    assert c.page.ax_spe.get_legend() is not None
    # X-Y panel: gray base hexbin + one coloured hexbin layer per selection.
    from matplotlib.collections import PolyCollection
    polys = [c for c in c.page.ax_xy.collections if isinstance(c, PolyCollection)]
    assert len(polys) == 3                       # base + 2 selections
    assert c.page.ax_xy.get_legend() is not None
    # dt panel has histogram patches (base + selections).
    assert len(c.page.ax_dt.patches) > 0


def test_remove_and_clear_selections(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._arm_selection(); c._on_energy_span(0, 1000)
    c._arm_selection(); c._on_energy_span(1000, 2000)
    assert len(c.selections) == 2
    c._remove_selection(c.selections[0])
    assert len(c.selections) == 1
    c._clear_selections()
    assert c.selections == []


def test_load_resets_selections(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._arm_selection(); c._on_energy_span(0, 3000)
    assert len(c.selections) == 1
    c._load()                               # a fresh run clears stale selections
    assert c.selections == []


# -- SelectionsDialog ----------------------------------------------------------

def test_selections_dialog_opens_and_holds_list(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._arm_selection(); c._on_energy_span(0, 2000)
    assert len(c.selections) == 1
    # Dialog not open yet -- count label on the options panel is updated.
    assert "1 selection" in c.opts.lbl_sel_count.text()
    # Opening the dialog populates its list.
    c._open_selections()
    assert c._sel_dlg is not None
    # Reusing: opening again just raises the same dialog.
    dlg = c._sel_dlg
    c._open_selections()
    assert c._sel_dlg is dlg


def test_selections_dialog_remove_clears_significance(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._arm_selection(); c._on_energy_span(500, 2000)
    c._open_selections()
    # Force-plant a significance overlay state (no real matplotlib twinx needed).
    c._sig_active = True
    c._remove_selection(c.selections[0])
    assert c.selections == []
    assert not c._sig_active    # cleared by _reset_selections -> _clear_significance


# -- time-slice fits vs dt -----------------------------------------------------

def test_build_slices_returns_spectra(api):
    _w, c = api
    c._load()
    slices = c._build_slices(dt_slice_w=None)
    assert len(slices) >= 1
    s = slices[0]
    assert {"idx", "t0", "t1", "tc", "spe", "search"} <= set(s)
    # Each slice carries an energy Spectrum binned to the tab's ebins, and the
    # PeakSearch has run (km method populates .snr).
    assert len(s["spe"].counts) == c.ebins
    assert s["search"].snr is not None
    # Energy units propagate from _axis_units: only an applied calibration
    # (energy_cal) is energy; everything else is channels (no units).
    expected_units = c.e_units if c.ekey == "energy_cal" else None
    assert s["spe"].e_units == expected_units


def test_build_slices_caps_count(api):
    _w, c = api
    c._load()
    from wara.gui.slicefit import MAX_SLICES
    # An absurdly small width would ask for a huge slice count; it must cap.
    slices = c._build_slices(dt_slice_w=1e-12)
    assert 1 <= len(slices) <= MAX_SLICES


def test_band_snr_reads_search_snr(api):
    _w, c = api
    c._load()
    from wara.gui.slicefit import band_snr
    slices = c._build_slices(dt_slice_w=None)
    val = band_snr(slices[0]["search"], (500.0, 2000.0))
    # A finite, non-negative SNR (snr is clipped at 0 by PeakSearch).
    assert val >= 0.0 or val != val  # >=0 or NaN if band off-axis


def test_slice_fit_window_area_mode_emits(api):
    _w, c = api
    c._load()
    from wara.gui.slicefit import SliceFitWindow, TECH_FIT
    slices = c._build_slices(dt_slice_w=None)
    win = SliceFitWindow(c.app, slices, (500.0, 2000.0), "Fe", "#ff0000")
    # Switch the per-slice Method to Net area − linear bkg.
    win.method.setCurrentIndex(1)
    assert win._is_area_method()
    captured = {}
    win.results_ready.connect(lambda p: captured.update(p))
    # Step forward and back to exercise per-slice ROI restore + refit, then emit.
    win._step(+1)
    win._step(-1)
    win._emit_results()
    win.close()
    assert captured["label"] == "Fe"
    assert captured["technique"] == TECH_FIT
    assert len(captured["vals"]) == len(slices)
    assert len(captured["dt_centers"]) == len(slices)
    # Net-area mode must produce finite values (regression: the ROI edges used
    # to collapse to a single point, making every net area NaN).
    assert any(np.isfinite(v) for v in captured["vals"])


def test_build_slices_respects_min_snr(api):
    _w, c = api
    c._load()
    s_lo = c._build_slices(dt_slice_w=None, min_snr=1.0)
    s_hi = c._build_slices(dt_slice_w=None, min_snr=42.0)
    assert s_lo[0]["search"].min_snr == 1.0
    assert s_hi[0]["search"].min_snr == 42.0


def test_slice_fit_window_peak_selection(api):
    _w, c = api
    c._load()
    from wara.gui.slicefit import SliceFitWindow
    slices = c._build_slices(dt_slice_w=None, min_snr=1.0)
    win = SliceFitWindow(c.app, slices, (500.0, 2000.0), "Fe", "#ff0000")
    assert not win._is_area_method()   # opens in Peak-fit mode
    # Selecting no peaks must yield exactly 0 for the slice (none → 0).
    win._selected_by_slice[win._cur] = set()
    v, e = win._current_metric()
    assert v == 0.0 and e == 0.0
    # If the fit found peaks, the table carries a leading "Use" checkbox column
    # and selecting all peaks gives their summed area.
    if win._peak_rows:
        assert win.table.horizontalHeaderItem(0).text() == "Use"
        assert win.table.cellWidget(0, 0) is not None
        win._selected_by_slice[win._cur] = set(range(len(win._peak_rows)))
        total = sum(r[1] for r in win._peak_rows)
        v2, _e2 = win._current_metric()
        assert abs(v2 - total) < 1e-6
    captured = {}
    win.results_ready.connect(lambda p: captured.update(p))
    win._emit_results()
    win.close()
    assert len(captured["vals"]) == len(slices)


def test_slice_fit_table_keeps_data_columns(qapp):
    """Regression: the inserted 'Use' checkbox column must not clobber the
    Centroid/Area/FWHM cells across repeated refits, and units stay keV."""
    from wara import spectrum as sp, peaksearch as ps
    from wara.gui.slicefit import SliceFitWindow
    x = np.linspace(700, 900, 2048)
    peak = 50 + 4000 * np.exp(-0.5 * ((x - 780) / 3) ** 2)
    slices = []
    for i in range(3):
        cts = (peak * (1 + 0.1 * i)).astype(int)
        spe = sp.Spectrum(counts=cts, energies=x, e_units="keV", label="t")
        se = ps.PeakSearch(spe, 420 * 2048 / 2 ** 11, 12 * 2048 / 2 ** 11,
                           fwhm_at_0=1.0, min_snr=3)
        slices.append(dict(idx=i, t0=i * 2.0, t1=(i + 1) * 2.0,
                           tc=i * 2.0 + 1.0, spe=spe, search=se))
    win = SliceFitWindow(None, slices, (760.0, 800.0), "Fe", "#f00")
    # Step around to force the multi-refit path that exposed the bug.
    win._step(+1); win._step(-1)
    t = win.table
    assert t.horizontalHeaderItem(0).text() == "Use"
    assert t.horizontalHeaderItem(1).text().startswith("Centroid")
    assert "keV" in t.horizontalHeaderItem(1).text()      # units, not "ch"
    assert win._peak_rows                                  # the peak was fit
    # Column 0 is the checkbox widget; column 1 carries the centroid TEXT.
    assert t.cellWidget(0, 0) is not None
    assert t.cellWidget(0, 1) is None
    assert t.item(0, 1) is not None and t.item(0, 1).text() not in ("", "-")
    win.close()


def test_spectra_view_toggle(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._open_selections()
    d = c._sel_dlg
    slices = c._build_slices(dt_slice_w=None)
    d.show_slice_spectra(slices, (500.0, 2000.0), c._energy_xlabel())
    # Overlay: a single log-y axis.
    assert len(d.spectra_fig.axes) == 1
    assert d.spectra_fig.axes[0].get_yscale() == "log"
    # Offset: ridgeline traces + a dt colour bar (second axes).
    d.cmb_spectra_view.setCurrentText("Offset")
    assert len(d.spectra_fig.axes) == 2
    assert "offset" in d.spectra_fig.axes[0].get_ylabel().lower()
    # Waterfall: 2-D heatmap (energy × dt) + a counts colour bar.
    d.cmb_spectra_view.setCurrentText("Waterfall")
    assert len(d.spectra_fig.axes) == 2
    assert d.spectra_fig.axes[0].get_ylabel() == "dt (ns)"
    # Back to overlay.
    d.cmb_spectra_view.setCurrentText("Overlay")
    assert len(d.spectra_fig.axes) == 1
    assert d.spectra_fig.axes[0].get_yscale() == "log"


def test_receive_slice_results_overlays(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._arm_selection(); c._on_energy_span(500, 2000)
    sel = c.selections[0]
    c._receive_slice_results(dict(
        label=sel["label"], color=sel["color"], technique="snr",
        dt_centers=[1.0, 2.0, 3.0], vals=[1.0, 2.0, 1.5],
        errs=[0.0, 0.0, 0.0], ylabel="Peak SNR"))
    assert c._sig_active
    assert c._ax_sig is not None
    # A secondary y-axis was added to the dt panel.
    assert c._ax_sig in c.page.fig.axes
    assert sel["label"] in c._slice_results


def test_clear_slice_overlay_restores_dt_panel(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._arm_selection(); c._on_energy_span(500, 2000)
    sel = c.selections[0]
    c._receive_slice_results(dict(
        label=sel["label"], color=sel["color"], technique="snr",
        dt_centers=[1.0, 2.0], vals=[1.0, 2.0], errs=[0.0, 0.0],
        ylabel="Peak SNR"))
    n_axes_with_overlay = len(c.page.fig.axes)
    c._clear_slice_overlay()
    assert not c._sig_active
    assert c._ax_sig is None
    assert c._slice_results == {}
    # The secondary axes was removed.
    assert len(c.page.fig.axes) == n_axes_with_overlay - 1


def test_open_selections_disables_interactive_cuts(api):
    _w, c = api
    c._load()
    c.opts.btn_interactive.setChecked(True)        # arm interactive cuts
    assert c.opts.btn_interactive.isChecked()
    assert len(c._selectors) == 3                  # energy / dt / X-Y selectors
    c._open_selections()
    # Opening Selections must turn cuts off and detach the span/rect selectors.
    assert not c.opts.btn_interactive.isChecked()
    assert c._selectors == []


def test_clear_selections_clears_slice_results(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._arm_selection(); c._on_energy_span(500, 2000)
    sel = c.selections[0]
    c._receive_slice_results(dict(
        label=sel["label"], color=sel["color"], technique="snr",
        dt_centers=[1.0, 2.0], vals=[1.0, 2.0], errs=[0.0, 0.0],
        ylabel="Peak SNR"))
    assert c._slice_results
    c._clear_selections()
    # The stored result must be gone so it can't reappear on the next overlay.
    assert c._slice_results == {}
    assert not c._sig_active


def test_remove_selection_drops_its_slice_result(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._arm_selection(); c._on_energy_span(500, 2000)
    sel = c.selections[0]
    c._receive_slice_results(dict(
        label=sel["label"], color=sel["color"], technique="snr",
        dt_centers=[1.0, 2.0], vals=[1.0, 2.0], errs=[0.0, 0.0],
        ylabel="Peak SNR"))
    c._remove_selection(sel)
    assert sel["label"] not in c._slice_results


def test_remove_one_selection_keeps_others_overlay(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._arm_selection(); c._on_energy_span(500, 1000)
    c._arm_selection(); c._on_energy_span(1500, 2000)
    # The stub uses a fixed label; give them distinct ones for the results map.
    c.selections[0]["label"] = "A"
    c.selections[1]["label"] = "B"
    for s in c.selections:
        c._receive_slice_results(dict(
            label=s["label"], color=s["color"], technique="snr",
            dt_centers=[1.0, 2.0], vals=[1.0, 2.0], errs=[0.0, 0.0],
            ylabel="Peak SNR"))
    assert set(c._slice_results) == {"A", "B"}
    c._remove_selection(c.selections[0])    # remove "A"
    # A's curve is gone; B's stays drawn.
    assert set(c._slice_results) == {"B"}
    assert c._sig_active
    assert c._ax_sig in c.page.fig.axes


def test_ratio_to_ref_math():
    from wara.gui.slicefit import ratio_to_ref
    r, e = ratio_to_ref([10.0, 20.0], [1.0, 2.0], [5.0, 10.0], [0.5, 1.0])
    assert np.allclose(r, [2.0, 2.0])
    # err = |r| * sqrt((num_err/num)^2 + (den_err/den)^2)
    assert np.allclose(e, [2.0 * np.sqrt(0.01 + 0.01)] * 2)
    # A non-positive denominator yields NaN (drops out of the plot).
    r0, e0 = ratio_to_ref([5.0], [1.0], [0.0], [0.0])
    assert np.isnan(r0[0]) and np.isnan(e0[0])


def _inject_two_results(c):
    for lbl, col, vals in (("Si", "#f00", [10.0, 20.0, 30.0]),
                           ("Mg", "#0f0", [5.0, 8.0, 9.0])):
        c._receive_slice_results(dict(
            label=lbl, color=col, technique="snr", dt_centers=[1.0, 2.0, 3.0],
            vals=vals, errs=[1.0, 1.0, 1.0], ylabel="Peak SNR"))


def test_slice_ratio_overlay_toggle(api):
    _w, c = api
    c._load()
    _inject_two_results(c)
    assert not c._ratio_active()                  # no reference selected yet
    c._set_slice_ratio_ref("Si")
    assert c._ratio_active()
    assert c._slice_overlay_ylabel() == "Ratio to Si"
    assert c._sig_active and c._ax_sig in c.page.fig.axes
    c._set_slice_ratio_ref(None)                  # back to absolute
    assert not c._ratio_active()
    assert c._slice_overlay_ylabel() == "Peak SNR"


def test_ratio_combo_enables_with_two_results(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._open_selections()
    d = c._sel_dlg
    assert not d.cmb_ratio_ref.isEnabled()        # no results
    c._receive_slice_results(dict(
        label="Si", color="#f00", technique="snr", dt_centers=[1.0],
        vals=[1.0], errs=[0.0], ylabel="Peak SNR"))
    assert not d.cmb_ratio_ref.isEnabled()        # only one
    c._receive_slice_results(dict(
        label="Mg", color="#0f0", technique="snr", dt_centers=[1.0],
        vals=[1.0], errs=[0.0], ylabel="Peak SNR"))
    assert d.cmb_ratio_ref.isEnabled()            # two → enabled
    items = [d.cmb_ratio_ref.itemText(i) for i in range(d.cmb_ratio_ref.count())]
    assert items[0] == "(absolute)" and "Si" in items and "Mg" in items


def test_remove_reference_resets_ratio(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._arm_selection(); c._on_energy_span(500, 1000)
    c._arm_selection(); c._on_energy_span(1500, 2000)
    c.selections[0]["label"] = "Si"
    c.selections[1]["label"] = "Mg"
    for s in c.selections:
        c._receive_slice_results(dict(
            label=s["label"], color=s["color"], technique="snr",
            dt_centers=[1.0, 2.0], vals=[10.0, 20.0], errs=[1.0, 1.0],
            ylabel="Peak SNR"))
    c._set_slice_ratio_ref("Si")
    assert c._ratio_active()
    c._remove_selection(c.selections[0])          # remove the reference (Si)
    assert c._slice_ratio_ref is None
    assert not c._ratio_active()


def test_filter_clears_slice_overlay(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._arm_selection(); c._on_energy_span(500, 2000)
    sel = c.selections[0]
    c._receive_slice_results(dict(
        label=sel["label"], color=sel["color"], technique="snr",
        dt_centers=[1.0, 2.0], vals=[1.0, 2.0], errs=[0.0, 0.0],
        ylabel="Peak SNR"))
    assert c._sig_active
    c.apply_energy_filter(0, 3000)
    assert not c._sig_active    # cleared by the filter


def test_3d_renders_one_volume_per_selection(api, monkeypatch):
    _w, c = api
    c._load()
    monkeypatch.setattr(api_mod, "EnergySelectionDialog", _StubSelDialog)
    c._arm_selection(); c._on_energy_span(0, 2000)
    c._arm_selection(); c._on_energy_span(2000, 4000)
    monkeypatch.setattr(api_mod, "Api3DDialog", _Stub3DDialog)
    c._open_3d()
    c._create_plot_3d()
    assert c._api3d_tmp is not None and os.path.exists(c._api3d_tmp)
    assert "2 selection(s)" in c._api3d_dlg.status.text()
    c._cleanup_3d_tmp()


def test_send_uncalibrated_spectrum_carries_real_channels(api):
    w, c = api
    c._load()                                # raw → uncalibrated channel axis
    assert c.e_units is None
    c._send_to_spectrum()
    spect = w.spect
    assert spect is not None
    # The real channels (gam_x) ride on adc_channels — the coordinate a
    # calibration fits against (cal_channels) so it round-trips to the dataframe.
    # Everything the user interacts with (channels, x, drag/fit) stays on the
    # 0..N index axis so peak-finding and drag-and-fit behave like any file.
    np.testing.assert_allclose(spect.adc_channels, c.gam_x)
    np.testing.assert_allclose(spect.cal_channels, c.gam_x)
    np.testing.assert_array_equal(spect.channels, np.arange(len(spect.counts)))
    np.testing.assert_array_equal(spect.x, np.arange(len(spect.counts)))
    assert spect.adc_channels.max() > len(spect.channels)   # proves real channels


def test_send_units_match_channel_panel(api):
    # Regression: a channel panel must not send an energy (MeV) spectrum.
    # _native_units could be "MeV" (a native-energy column was detected) while
    # the working axis is still channels (e.g. raw or a drift-corrected axis);
    # the send must mirror the panel label, not _native_units.
    w, c = api
    c._load()
    c._native_units = "MeV"                  # pretend native energy was detected
    assert c.ekey == "energy_orig"           # but the working axis is channels
    assert c._energy_xlabel() == "Channels"
    assert c._axis_units() is None
    c._send_to_spectrum()
    spect = w.spect
    # Channels → rides on adc_channels with no energy units, NOT MeV energies.
    assert spect.energies is None
    assert not spect.e_units
    np.testing.assert_allclose(spect.adc_channels, c.gam_x)


def test_send_works_without_a_prior_energy_draw(api):
    # Regression: the first Send used to silently do nothing when the energy
    # panel hadn't drawn yet (self.gam still None), forcing a second press.
    # Send must bin the histogram on demand from df_current.
    w, c = api
    c._load()
    c.gam = c.gam_x = None          # simulate "no draw happened yet"
    c._send_to_spectrum()
    assert w.spect is not None
    assert c.gam is not None         # Send populated it on demand


def test_apply_calibration_calibrates_dataframe(api):
    _w, c = api
    c._load()
    assert c.ekey == "energy_orig" and c.e_units is None
    c.apply_calibration([0.0, 0.5], units="keV")   # E = 0.5 * channel
    assert c.e_units == "keV"
    assert c.ekey == "energy_cal"
    assert "energy_cal" in c.df_current.columns and "energy_cal" in c.df_api.columns
    np.testing.assert_allclose(
        c.df_current["energy_cal"].to_numpy(),
        0.5 * c.df_current["energy_orig"].to_numpy())
    # erange + axis label follow the calibration.
    assert c.erange[1] == pytest.approx(0.5 * c.df_api["energy_orig"].max())
    assert c.page.ax_spe.get_xlabel() == "Energy (keV)"
    assert c.opts.lbl_cal.text().startswith("Calibrated")


def test_retrieve_calibration_from_calibration_tab(api):
    w, c = api
    c._load()
    # Stand up a calibration on the Calibration tab (E = 0.5 * channel, keV).
    cal = w.calibration
    cal.opts.coef_a.setText("0")
    cal.opts.coef_b.setText("0.5")
    cal.predicted = np.zeros(8)              # marks "a calibration exists"
    assert cal.current_calibration() == ([0.0, 0.5], "keV")
    c._retrieve_calibration()
    assert c.e_units == "keV" and c.ekey == "energy_cal"
    np.testing.assert_allclose(
        c.df_current["energy_cal"].to_numpy(),
        0.5 * c.df_current["energy_orig"].to_numpy())


def test_retrieve_calibration_guards_when_none(api):
    _w, c = api
    c._load()
    assert c.app.calibration.current_calibration() is None   # nothing built yet
    c._retrieve_calibration()                # must not raise or calibrate
    assert c.e_units is None and c.ekey == "energy_orig"
    assert "energy_cal" not in c.df_current.columns


def test_send_to_spectrum_uses_calibrated_energy(api):
    w, c = api
    c._load()
    c.apply_calibration([0.0, 0.5], units="keV")
    c._send_to_spectrum()
    assert w.spect is not None
    assert w.spect.e_units == "keV"
    assert w.spect.energies is not None


def test_load_clears_calibration(api):
    _w, c = api
    c._load()
    c.apply_calibration([0.0, 0.5], units="keV")
    assert c.e_units == "keV"
    c._load()                                    # a fresh run resets calibration
    assert c.e_units is None and c.ekey == "energy_orig"
    assert "energy_cal" not in c.df_current.columns
    assert c.opts.lbl_cal.text() == "Uncalibrated (channels)"


def test_clear_calibration_reverts_to_channels(api):
    _w, c = api
    c._load()
    c.apply_calibration([0.0, 0.5], units="keV")
    assert c.ekey == "energy_cal" and c.opts.btn_clear_cal.isEnabled()
    c._clear_calibration()
    assert c.e_units is None and c._cal_coeffs is None
    assert c.ekey == "energy_orig"
    assert "energy_cal" not in c.df_current.columns
    assert "energy_cal" not in c.df_api.columns
    assert c.df_current.shape[0] == 4000           # back to the full original data
    assert c.page.ax_spe.get_xlabel() == "Channels"
    assert not c.opts.btn_clear_cal.isEnabled()
    assert c.opts.lbl_cal.text() == "Uncalibrated (channels)"


def test_clear_button_starts_disabled_and_load_disables_it(api):
    _w, c = api
    assert not c.opts.btn_clear_cal.isEnabled()
    c._load()
    c.apply_calibration([0.0, 0.5], units="keV")
    assert c.opts.btn_clear_cal.isEnabled()
    c._load()                                       # a new run disables it again
    assert not c.opts.btn_clear_cal.isEnabled()


def test_calibration_survives_reset(api):
    _w, c = api
    c._load()
    c.apply_calibration([0.0, 0.5], units="keV")
    c.apply_energy_filter(100, 500)              # filter in calibrated energy
    assert c.df_current.shape[0] < 4000
    c._reset()                                   # reset keeps the calibration
    assert c.e_units == "keV" and c.ekey == "energy_cal"
    assert c.df_current.shape[0] == 4000


# ── Time (dt) shift ──────────────────────────────────────────────────────────

def test_apply_dt_shift_adds_dt_cal(api):
    _w, c = api
    c._load()
    assert c._dt_key == "dt"
    raw = c.df_api["dt"].to_numpy().copy()
    c.apply_dt_shift(12.5)                        # dt_cal = dt + 12.5
    assert c._dt_shift == 12.5
    assert c._dt_key == "dt_cal"
    assert "dt_cal" in c.df_current.columns and "dt_cal" in c.df_api.columns
    np.testing.assert_allclose(c.df_current["dt_cal"].to_numpy(), raw + 12.5)
    assert c._dt_corrected
    assert "12.5 ns" in c._dt_label and "constant" in c._dt_label.lower()
    assert "shifted" in c.page.ax_dt.get_xlabel().lower()


def test_apply_dt_shift_is_not_cumulative(api):
    _w, c = api
    c._load()
    raw = c.df_api["dt"].to_numpy().copy()
    c.apply_dt_shift(10.0)
    c.apply_dt_shift(-3.0)                        # recomputed from original dt
    np.testing.assert_allclose(c.df_current["dt_cal"].to_numpy(), raw - 3.0)
    assert c._dt_shift == -3.0


def test_dt_shift_survives_reset_with_energy_cal(api):
    # The explicit requirement: Reset remembers BOTH calibrations.
    _w, c = api
    c._load()
    c.apply_calibration([0.0, 0.5], units="keV")
    c.apply_dt_shift(7.0)
    raw = c.df_api["dt"].to_numpy().copy()
    c.apply_t_filter(*np.percentile(c.df_current["dt_cal"], [10, 90]))
    assert c.df_current.shape[0] < 4000
    c._reset()
    assert c.e_units == "keV" and c.ekey == "energy_cal"   # energy cal kept
    assert c._dt_shift == 7.0 and c._dt_key == "dt_cal"     # time shift kept
    assert c.df_current.shape[0] == 4000
    np.testing.assert_allclose(c.df_current["dt_cal"].to_numpy(), raw + 7.0)


def test_clear_dt_shift_is_independent_of_energy_cal(api):
    _w, c = api
    c._load()
    c.apply_calibration([0.0, 0.5], units="keV")
    c.apply_dt_shift(5.0)
    c._clear_dt_shift()                           # only the shift goes away
    assert c._dt_shift is None and c._dt_key == "dt"
    assert not c._dt_corrected
    assert "dt_cal" not in c.df_current.columns and "dt_cal" not in c.df_api.columns
    # Energy calibration is untouched.
    assert c.e_units == "keV" and c.ekey == "energy_cal"
    assert "energy_cal" in c.df_current.columns


def test_clear_calibration_is_independent_of_dt_shift(api):
    _w, c = api
    c._load()
    c.apply_calibration([0.0, 0.5], units="keV")
    c.apply_dt_shift(5.0)
    c._clear_calibration()                        # only the energy cal goes away
    assert c.e_units is None and "energy_cal" not in c.df_current.columns
    # Time shift is untouched.
    assert c._dt_shift == 5.0 and c._dt_key == "dt_cal"
    assert "dt_cal" in c.df_current.columns


# ── Apply to data ────────────────────────────────────────────────────────────
#
# "Apply to data" now re-reads every channel of the source run, bakes the active
# calibration/time-shift onto the loaded channel's rows, and writes the combined
# result to a *new* run via read_parquet_api.save_combined_run. The apply_env
# fixture supplies a two-channel (4 + 5) source run, stubs the destination
# dialog, and captures what would have been written instead of touching disk.

def _combined_events(n_per=2000, seed=1):
    """A two-channel API source run: channels 4 and 5, dt in seconds (the
    on-disk convention — the loader scales it to ns)."""
    rng = np.random.default_rng(seed)
    frames = []
    for ch in (4, 5):
        energy = rng.uniform(0, 6000, n_per)
        frames.append(pd.DataFrame({
            "channel": np.full(n_per, ch),
            "dt": rng.normal(0.0, 15.0, n_per) / 1e9,   # seconds
            "energy": energy,
            "energy_orig": energy,
            "A": rng.uniform(1, 100, n_per),
            "B": rng.uniform(1, 100, n_per),
            "C": rng.uniform(1, 100, n_per),
            "D": rng.uniform(1, 100, n_per),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def apply_env(api, monkeypatch):
    """(controller, combined_frame, saved) wired so Apply to data writes nowhere:
    save_combined_run is captured into ``saved`` and the destination dialog auto-
    accepts run number 999."""
    _w, c = api
    combined = _combined_events()

    def fake_read(date, runnr, ch=None, flat_field=False, data_path_txt=None):
        if ch is None:
            return combined.copy()
        sub = combined[read_parquet_api.channel_mask(combined, ch)]
        return sub.reset_index(drop=True).copy()
    monkeypatch.setattr(read_parquet_api, "read_parquet_file", fake_read)

    saved = {}

    def fake_save(df, date, runnr, data_path=None, overwrite=False):
        saved.update(df=df.copy(), date=date, runnr=runnr, overwrite=overwrite)
        return Path("saved.parquet")
    monkeypatch.setattr(read_parquet_api, "save_combined_run", fake_save)
    # Destination run "doesn't exist" → no overwrite prompt.
    monkeypatch.setattr(read_parquet_api, "run_parquet_path",
                        lambda *a, **k: (Path("__no_such_run__"), Path("p"), "RUN"))

    class _StubApplyDialog:
        def __init__(self, date, runnr, parent=None):
            self._date = date
        def exec_(self):
            return QDialog.Accepted
        def values(self):
            return (self._date, 999)
    monkeypatch.setattr(api_mod, "ApplyToDataDialog", _StubApplyDialog)

    return c, combined, saved


def test_apply_to_data_bakes_both_columns(apply_env):
    c, combined, saved = apply_env
    c._load()
    c.apply_calibration([0.0, 0.5], units="keV")  # energy_cal = 0.5 * energy_orig
    c.apply_dt_shift(7.0)                          # dt_cal = dt(ns) + 7
    c._apply_to_data()

    df = saved["df"]
    assert saved["runnr"] == 999
    # All channels, all rows preserved in the combined output.
    assert len(df) == len(combined)
    assert {"energy_cal", "dt_cal"}.issubset(df.columns)
    m5 = read_parquet_api.channel_mask(df, 5).to_numpy()
    m4 = read_parquet_api.channel_mask(df, 4).to_numpy()
    # Loaded channel (5) carries the calibration; the rest stays NaN.
    np.testing.assert_allclose(
        df.loc[m5, "energy_cal"].to_numpy(),
        0.5 * df.loc[m5, "energy_orig"].to_numpy())
    assert np.isnan(df.loc[m4, "energy_cal"].to_numpy()).all()
    # dt_cal = raw dt (seconds) → ns, plus the 7 ns shift; raw dt left in seconds.
    np.testing.assert_allclose(
        df.loc[m5, "dt_cal"].to_numpy(),
        df.loc[m5, "dt"].to_numpy() * 1e9 + 7.0)
    # The un-shifted channel keeps its raw dt (in ns) -- never NaN -- so a dt_cal
    # column is usable for every channel.
    np.testing.assert_allclose(
        df.loc[m4, "dt_cal"].to_numpy(),
        df.loc[m4, "dt"].to_numpy() * 1e9)


def test_apply_to_data_only_dt_when_no_energy_change(apply_env):
    c, _combined, saved = apply_env
    c._load()
    c.apply_dt_shift(3.0)
    c._apply_to_data()
    df = saved["df"]
    assert "dt_cal" in df.columns
    # No energy calibration was applied, so no energy_cal column is created.
    assert "energy_cal" not in df.columns


def test_apply_to_data_noop_without_changes(apply_env):
    c, _combined, saved = apply_env
    c._load()
    c._apply_to_data()
    assert saved == {}            # nothing written when nothing changed


def test_apply_to_data_energy_cal_from_gainshift_only(apply_env, monkeypatch):
    """Gain-shift alone (no polynomial calibration) should produce energy_cal,
    after the user confirms the "not calibrated" warning."""
    c, _combined, saved = apply_env
    c._load()
    c.apply_energy_gainshift(4, method="shift")
    assert "energy_drift" in c.df_api.columns and "energy_cal" not in c.df_api.columns
    # Gain-shift-only ⇒ no calibration curve ⇒ the warning must fire; accept it.
    warned = {"n": 0}
    def _ok():
        warned["n"] += 1
        return True
    monkeypatch.setattr(c, "_confirm_uncalibrated_energy", _ok)
    c._apply_to_data()
    assert warned["n"] == 1
    df = saved["df"]
    assert "energy_cal" in df.columns
    m5 = read_parquet_api.channel_mask(df, 5).to_numpy()
    np.testing.assert_array_equal(
        df.loc[m5, "energy_cal"].to_numpy(), c.df_api["energy_drift"].to_numpy())


def test_apply_to_data_uncalibrated_energy_can_be_cancelled(apply_env, monkeypatch):
    """Declining the "not calibrated" warning aborts the save entirely."""
    c, _combined, saved = apply_env
    c._load()
    c.apply_energy_gainshift(4, method="shift")
    monkeypatch.setattr(c, "_confirm_uncalibrated_energy", lambda: False)
    c._apply_to_data()
    assert saved == {}            # nothing written when the warning is declined


def test_apply_to_data_calibrated_energy_skips_warning(apply_env, monkeypatch):
    """A real calibration curve (e_units set) must not trigger the warning."""
    c, _combined, saved = apply_env
    c._load()
    c.apply_calibration([0.0, 0.5], units="keV")
    called = {"n": 0}
    monkeypatch.setattr(c, "_confirm_uncalibrated_energy",
                        lambda: called.__setitem__("n", called["n"] + 1) or True)
    c._apply_to_data()
    assert called["n"] == 0       # calibrated energy needs no warning
    assert "energy_cal" in saved["df"].columns


def test_apply_to_data_ignores_cuts(apply_env):
    """Apply to data writes the full combined run regardless of df_current cuts."""
    c, combined, saved = apply_env
    c._load()
    c.apply_calibration([0.0, 0.5], units="keV")
    c.apply_dt_shift(5.0)
    # A dt cut shrinks df_current but must not affect what's written.
    dt_vals = c.df_api["dt_cal"]
    c.apply_t_filter(float(dt_vals.min()), float(dt_vals.quantile(0.5)))
    assert len(c.df_current) < len(c.df_api)
    c._apply_to_data()
    df = saved["df"]
    assert len(df) == len(combined)        # full combined run written
    assert {"energy_cal", "dt_cal"}.issubset(df.columns)


def test_apply_to_data_merges_channels_into_one_run(api, monkeypatch, tmp_path):
    """Saving channel 5 then channel 4 to the SAME destination run accumulates
    both channels' calibration -- the second save must not clobber the first.
    """
    _w, c = api
    combined = _combined_events()
    disk = {}                                  # runnr -> saved combined frame

    def fake_read(date, runnr, ch=None, flat_field=False, data_path_txt=None):
        base = disk.get(runnr, combined)       # destination reads back; source = raw
        if ch is None:
            return base.copy()
        sub = base[read_parquet_api.channel_mask(base, ch)]
        return sub.reset_index(drop=True).copy()
    monkeypatch.setattr(read_parquet_api, "read_parquet_file", fake_read)

    def fake_save(df, date, runnr, data_path=None, overwrite=False):
        disk[runnr] = df.copy()
        (tmp_path / f"run-{runnr}").mkdir(parents=True, exist_ok=True)
        return Path("saved.parquet")
    monkeypatch.setattr(read_parquet_api, "save_combined_run", fake_save)

    def fake_path(date, runnr, data_path=None, make=False):
        run_dir = tmp_path / f"run-{runnr}"
        return run_dir, run_dir / "parquet-data", "RUN"
    monkeypatch.setattr(read_parquet_api, "run_parquet_path", fake_path)

    class _StubApplyDialog:
        def __init__(self, date, runnr, parent=None):
            self._date = date
        def exec_(self):
            return QDialog.Accepted
        def values(self):
            return (self._date, 999)
    monkeypatch.setattr(api_mod, "ApplyToDataDialog", _StubApplyDialog)
    # The existing-run prompt always chooses "merge" (add the channel).
    monkeypatch.setattr(c, "_confirm_existing_run", lambda *a, **k: "merge")

    # 1) Channel 5 → new run 999.
    c.opts.ed_ch.setText("5")
    c._load()
    c.apply_calibration([0.0, 0.5], units="keV")
    c._apply_to_data()
    df1 = disk[999]
    m4 = read_parquet_api.channel_mask(df1, 4).to_numpy()
    assert np.isnan(df1.loc[m4, "energy_cal"].to_numpy()).all()  # ch4 not yet done

    # 2) Channel 4 → same run 999 (merge). Channel 5 must survive.
    c.opts.ed_ch.setText("4")
    c._load()
    c.apply_calibration([0.0, 0.25], units="keV")
    c._apply_to_data()
    df2 = disk[999]
    m5 = read_parquet_api.channel_mask(df2, 5).to_numpy()
    m4 = read_parquet_api.channel_mask(df2, 4).to_numpy()
    # Both channels now calibrated, each with its own coefficient.
    np.testing.assert_allclose(df2.loc[m5, "energy_cal"].to_numpy(),
                               0.5 * df2.loc[m5, "energy_orig"].to_numpy())
    np.testing.assert_allclose(df2.loc[m4, "energy_cal"].to_numpy(),
                               0.25 * df2.loc[m4, "energy_orig"].to_numpy())


def test_apply_to_data_unshifted_channel_keeps_raw_dt(apply_env):
    """When a dt_cal column is created for the shifted channel, the other channel
    (no shift applied) must keep its raw dt in ns, not be left NaN."""
    c, combined, saved = apply_env
    c._load()                                     # loads ch 5
    c.apply_calibration([0.0, 0.5], units="keV")
    c.apply_dt_shift(8.0)
    c._apply_to_data()
    df = saved["df"]
    m4 = read_parquet_api.channel_mask(df, 4).to_numpy()
    # ch4 got no shift -> dt_cal is raw dt (s) converted to ns, never NaN.
    assert not np.isnan(df.loc[m4, "dt_cal"].to_numpy()).any()
    np.testing.assert_allclose(
        df.loc[m4, "dt_cal"].to_numpy(), df.loc[m4, "dt"].to_numpy() * 1e9)
    # ch4 has no energy calibration, so energy_cal stays NaN (no valid fallback).
    assert np.isnan(df.loc[m4, "energy_cal"].to_numpy()).all()


def test_load_clears_dt_shift(api):
    _w, c = api
    c._load()
    c.apply_dt_shift(9.0)
    c._load()                                     # a fresh run resets the shift
    assert c._dt_shift is None and c._dt_key == "dt"
    assert not c._dt_corrected
    assert "dt_cal" not in c.df_current.columns
    assert c._dt_label == "No time shift"


# ── Energy gain-shift (drift correction) + Shifts dialog ──────────────────────

def test_energy_gainshift_switches_chan_key(api):
    _w, c = api
    c._load()
    assert c._chan_key == "energy_orig" and not c._egain_applied
    c.apply_energy_gainshift(4, method="shift")
    assert c._egain_applied
    assert c._chan_key == "energy_drift"          # corrected channels are active
    assert "energy_drift" in c.df_current.columns and "energy_drift" in c.df_api.columns
    assert c.ekey == "energy_drift"               # uncalibrated → bins on corrected ch


def test_energy_gainshift_survives_reset(api):
    _w, c = api
    c._load()
    c.apply_energy_gainshift(4, method="shift")
    c.apply_energy_filter(*np.percentile(c.df_current["energy_drift"], [10, 90]))
    assert c.df_current.shape[0] < 4000
    c._reset()
    assert c._egain_applied and c._chan_key == "energy_drift"
    assert c.df_current.shape[0] == 4000


def test_clear_energy_gainshift_reverts(api):
    _w, c = api
    c._load()
    c.apply_energy_gainshift(4)
    c.clear_energy_gainshift()
    assert not c._egain_applied and c._chan_key == "energy_orig"
    assert "energy_drift" not in c.df_current.columns
    assert "energy_drift" not in c.df_api.columns


def test_energy_gainshift_layers_under_polynomial_cal(api):
    _w, c = api
    c._load()
    c.apply_energy_gainshift(4, method="shift")
    c.apply_calibration([0.0, 0.5], units="keV")
    assert c.ekey == "energy_cal" and c._chan_key == "energy_drift"
    # energy_cal must derive from the drift-corrected channels, not the raw ones.
    np.testing.assert_allclose(
        c.df_current["energy_cal"].to_numpy(),
        0.5 * c.df_current["energy_drift"].to_numpy())


def test_gainshift_after_calibration_reapplies_cal(api):
    # Applying the gain-shift while a calibration is active must re-derive
    # energy_cal from the freshly corrected channels (not leave it stale).
    _w, c = api
    c._load()
    c.apply_calibration([0.0, 0.5], units="keV")
    c.apply_energy_gainshift(4, method="shift")
    np.testing.assert_allclose(
        c.df_current["energy_cal"].to_numpy(),
        0.5 * c.df_current["energy_drift"].to_numpy())


def test_dt_gainshift_writes_dt_cal(api):
    _w, c = api
    c._load()
    c.apply_dt_gainshift(4, method="shift")
    assert c._dt_segments_applied and c._dt_shift is None
    assert c._dt_corrected and c._dt_key == "dt_cal"
    assert "dt_cal" in c.df_current.columns


def test_constant_and_segment_dt_compose(api):
    import numpy as np
    _w, c = api
    c._load()
    # Constant first, then segment alignment: the alignment must retain the
    # constant rather than recompute dt_cal from raw dt.
    c.apply_dt_shift(5.0)
    assert c._dt_shift == 5.0 and not c._dt_segments_applied
    c.apply_dt_gainshift(4)
    assert c._dt_shift == 5.0 and c._dt_segments_applied      # constant retained
    aligned = c.df_api["dt_aligned"].to_numpy()
    np.testing.assert_allclose(c.df_api["dt_cal"].to_numpy(), aligned + 5.0)
    # A further constant builds on the aligned baseline (not cumulative) and
    # keeps the alignment in place -- the original reported bug.
    c.apply_dt_shift(2.0)
    assert c._dt_shift == 2.0 and c._dt_segments_applied
    np.testing.assert_allclose(c.df_api["dt_cal"].to_numpy(), aligned + 2.0)


def test_energy_and_time_corrections_independent(api):
    _w, c = api
    c._load()
    c.apply_energy_gainshift(4)
    c.apply_dt_gainshift(4)
    assert c._egain_applied and c._dt_segments_applied
    assert c._chan_key == "energy_drift" and c._dt_key == "dt_cal"
    c.clear_energy_gainshift()                    # clearing one leaves the other
    assert not c._egain_applied and c._dt_segments_applied
    assert "dt_cal" in c.df_current.columns


def test_shifts_dialog_opens_and_drives_controller(api):
    _w, c = api
    c._load()
    c._open_shifts()
    dlg = c._shifts_dlg
    assert dlg is not None and dlg.tabs.count() == 2
    dlg._apply_segments("energy")                 # Apply on the Energy tab
    assert c._egain_applied
    dlg.f_const.setText("3.0"); dlg._apply_constant()
    assert c._dt_shift == 3.0
    dlg._clear("energy")
    assert not c._egain_applied
    dlg.close()


def test_constant_commits_previewed_time_alignment(api):
    # Reported bug: Preview the time alignment (not Apply), then apply a constant
    # Δt. The constant must commit the previewed alignment first and layer on top,
    # not discard it by shifting raw dt.
    import numpy as np
    _w, c = api
    c._load()
    c._open_shifts()
    dlg = c._shifts_dlg
    dlg.f_nseg["time"].setText("4")
    dlg._preview("time", prospective=True)            # preview only, not applied
    assert dlg._time_preview_pending and not c._dt_segments_applied
    dlg.f_const.setText("-10.0"); dlg._apply_constant()
    assert c._dt_segments_applied and c._dt_shift == -10.0
    np.testing.assert_allclose(c.df_api["dt_cal"].to_numpy(),
                               c.df_api["dt_aligned"].to_numpy() - 10.0)
    dlg.close()


def test_constant_without_preview_uses_raw_dt(api):
    # Bypass case: no alignment previewed -> a constant shifts raw dt directly.
    import numpy as np
    _w, c = api
    c._load()
    c._open_shifts()
    dlg = c._shifts_dlg
    raw = c.df_api["dt"].to_numpy().copy()
    dlg.f_const.setText("4.0"); dlg._apply_constant()
    assert not c._dt_segments_applied and c._dt_shift == 4.0
    np.testing.assert_allclose(c.df_api["dt_cal"].to_numpy(), raw + 4.0)
    dlg.close()


def test_shifts_dialog_open_does_not_compute_alignment(api, monkeypatch):
    # Opening the window must only show the raw segments; the shift alignment is
    # computed only when the user clicks Preview (or Apply), never on open.
    from wara import apicalc
    _w, c = api
    c._load()
    calls = {"align": 0}
    orig_align = apicalc.GainShift.align
    monkeypatch.setattr(apicalc.GainShift, "align",
                        lambda self, *a, **k: calls.__setitem__("align", calls["align"] + 1)
                        or orig_align(self, *a, **k))
    c._open_shifts()
    dlg = c._shifts_dlg
    assert calls["align"] == 0                     # nothing aligned on open
    # The aligned panel shows the placeholder hint, not an alignment.
    aligned_text = " ".join(t.get_text() for t in dlg.ax_aligned["energy"].texts)
    assert "Preview" in aligned_text
    dlg._preview("energy", prospective=True)        # user asks for the preview
    assert calls["align"] == 1                      # now it computes
    dlg.close()


def test_load_guards_blank_inputs(api):
    _w, c = api
    c.opts.ed_date.setText("")
    c.opts.ed_run.setText("")
    c._load()                      # must not raise
    assert c.df_current is None    # nothing loaded
