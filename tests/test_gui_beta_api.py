"""Offscreen GUI regression tests for the wara --beta API tab
(wara.gui_beta.api: ApiOptions / ApiPage / ApiController).

The real parquet loader needs run files on a local data path, so the data
access (``read_parquet_api.read_parquet_file`` and the three apicalc settings
helpers) is monkeypatched with a small synthetic event table. This exercises the
controller logic — panel build, live energy/time/X-Y filters, vmax, reset and
send-to-Spectrum — without any data files.
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
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QDialog

from wara import read_parquet_api, apicalc
from wara.gui_beta.app import WaraBetaApp
from wara.gui_beta import api as api_mod
from wara.gui_beta.theme import NAV_SECTIONS


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

    def __init__(self, parent=None):
        super().__init__()
        self.btn_plot = QPushButton(self)
        self.status = QLabel(self)
        self.browser = _StubBrowser()

    def value(self, attr, cast):
        return cast(self.DEFAULTS[attr])


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


@pytest.fixture
def api(qapp, monkeypatch):
    df = _synthetic_events()
    # The loader multiplies dt by 1e9 (s→ns); feed seconds so the result is ns.
    monkeypatch.setattr(read_parquet_api, "read_parquet_file",
                        lambda *a, **k: df.copy().assign(dt=df["dt"] / 1e9))
    monkeypatch.setattr(apicalc, "get_total_time", lambda *a, **k: 14386.0)
    monkeypatch.setattr(apicalc, "get_total_counts", lambda *a, **k: 2.5e9)
    monkeypatch.setattr(apicalc, "calculate_neutron_yield", lambda *a, **k: 4.3e6)

    w = WaraBetaApp()
    idx = [name for name, _ in NAV_SECTIONS].index("API")
    w.nav_group.button(idx).setChecked(True)
    w._on_nav(idx)
    c = w.api
    c.opts.ed_date.setText("2023-07-02")
    c.opts.ed_run.setText("91")
    c.opts.ed_ch.setText("5")
    c.opts.rb_raw.setChecked(True)
    yield w, c
    w.close()


def test_load_builds_panels_and_info(api):
    _w, c = api
    c._load()
    assert c.page.ax_spe is not None and c.page.ax_dt is not None and c.page.ax_xy is not None
    assert c.xkey == "X2" and c.ykey == "Y2" and c.ekey == "energy_orig"
    assert c.gam is not None and len(c.gam) == c.ebins
    assert c.df_current.shape[0] == 4000
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


def test_send_to_spectrum_stays_on_tab_and_marks_button(api):
    w, c = api
    c._load()
    assert c.opts.btn_send.objectName() == "open_btn"
    c._send_to_spectrum()
    assert w.spect is not None
    assert len(w.spect.counts) == c.ebins
    # Stays on the API tab; the button turns into the green "Sent" confirmation.
    assert type(w.stack.currentWidget()).__name__ == "ApiPage"
    assert c.opts.btn_send.objectName() == "sent_btn"
    assert c.opts.btn_send.text() == api_mod.SEND_SENT_TEXT
    assert "sent" in c.opts.btn_send.text().lower()


def test_send_button_resets_when_spectrum_changes(api):
    _w, c = api
    c._load()
    c._send_to_spectrum()
    assert c.opts.btn_send.objectName() == "sent_btn"
    # A filter recomputes the energy histogram → the confirmation clears.
    c.apply_t_filter(-10, 10)
    assert c.opts.btn_send.objectName() == "open_btn"
    assert c.opts.btn_send.text() == api_mod.SEND_DEFAULT_TEXT


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
    # The sent spectrum's channel axis is gam_x (real channels), not 0..nbins-1,
    # so a calibration built on it round-trips back to the dataframe.
    np.testing.assert_allclose(spect.channels, c.gam_x)
    np.testing.assert_allclose(spect.x, c.gam_x)
    assert spect.channels.max() > len(spect.channels)   # proves real channels


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


def test_load_guards_blank_inputs(api):
    _w, c = api
    c.opts.ed_date.setText("")
    c.opts.ed_run.setText("")
    c._load()                      # must not raise
    assert c.df_current is None    # nothing loaded
