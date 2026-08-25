"""Offscreen GUI regression tests for the Planetary tab
(wara.gui.planetary: PlanetaryOptions / PlanetaryPage / PlanetaryController).

QtWebEngine is never loaded here — the page defers creating the web view until
the tab is first activated in a real session, so these tests exercise the
options panel, the PDS search/load plumbing, the region selection/summing
logic, the click bridge, and the globe HTML builder, all headless.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")

import re
from datetime import date

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication

from wara.gui import planetary as P
from wara.planetary.lp import LPGrsDay, LPGrsProduct
from wara.planetary.moon import R_MOON_KM, xyz_to_lonlat


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def tab(qapp):
    opts = P.PlanetaryOptions()
    page = P.PlanetaryPage()
    ctl = P.PlanetaryController(None, opts, page)
    yield ctl
    page.deleteLater(); opts.deleteLater()


def _fake_day(stem="1998_016_grs", lat=None, lon=None, n=None):
    lat = np.asarray(lat if lat is not None else [0.0, 40.0, 80.0, -40.0])
    n = len(lat)
    lon = np.asarray(lon if lon is not None else np.zeros(n))
    spectra = np.tile(np.arange(512, dtype=float), (n, 1))
    return LPGrsDay(
        stem=stem, day=date(1998, 1, 16),
        spectra=spectra, rejected=spectra * 0.1,
        deadtime=np.full(n, 0.04), overload=np.zeros(n),
        temperature=np.full(n, -28.0), time_doy=np.full(n, 16.5),
        altitude_km=np.full(n, 100.0), latitude=lat, longitude=lon,
    )


# ── Options panel ─────────────────────────────────────────────────────────────
def test_options_defaults(tab):
    o = tab.opts
    assert o.mission.currentText() == "Lunar Prospector GRS"
    assert o.ed_start.text() == "1998-01-16"
    assert o.phase.currentIndex() == 0          # All altitudes
    assert o.box_size.value() == 10.0
    assert o.detail.currentText() == P.GLOBE_DETAIL_DEFAULT
    assert not o.btn_download.isEnabled()       # enabled only after a search


def test_dates_phase_parses_controls(tab):
    tab.opts.ed_start.setText("1998-06-01")
    tab.opts.ed_end.setText("1999-02-01")
    tab.opts.phase.setCurrentIndex(2)
    start, end, phase = tab._dates_phase()
    assert start == date(1998, 6, 1) and end == date(1999, 2, 1)
    assert phase == "low"


def test_search_done_reports_and_enables_download(tab, tmp_path):
    tab.data_dir = tmp_path
    p = LPGrsProduct(stem="1998_016_grs", day=date(1998, 1, 16),
                     url_xml="http://x/1998_016_grs.xml",
                     url_dat="http://x/1998_016_grs.dat")
    tab._search_done([p])
    assert tab.opts.btn_download.isEnabled()
    assert "1 daily product(s)" in tab.opts.status.text()
    assert "(0 already downloaded)" in tab.opts.status.text()
    # Mark it downloaded → the count updates.
    (tmp_path / "1998_016_grs.xml").write_text("x")
    (tmp_path / "1998_016_grs.dat").write_text("x")
    tab._search_done([p])
    assert "(1 already downloaded)" in tab.opts.status.text()
    tab._search_done([])
    assert not tab.opts.btn_download.isEnabled()


# ── Loading and region selection ─────────────────────────────────────────────
def test_load_done_concatenates_days(tab):
    tab._load_done([_fake_day(), _fake_day(stem="1998_017_grs")])
    assert len(tab._lat) == 8
    assert tab._spectra.shape == (8, 512)
    assert "8 records" in tab.opts.status.text()
    # The full-data spectrum is plotted below the globe.
    assert len(tab.page.fig.axes) == 1
    assert tab.page.fig.axes[0].get_yscale() == "log"


def test_region_mask_box_and_seam(tab):
    tab._load_done([_fake_day(lat=[0.0, 40.0, 80.0, 0.0],
                              lon=[0.0, 0.0, 0.0, 179.0])])
    mask = tab.region_mask(0.0, 40.0, 10.0)
    assert list(mask) == [False, True, False, False]
    # Box crossing the ±180 seam catches the lon=179 record.
    mask = tab.region_mask(-175.0, 0.0, 10.0)
    assert list(mask) == [False, False, False, True]


def test_region_mask_over_pole_ignores_longitude(tab):
    tab._load_done([_fake_day(lat=[85.0, 85.0, 85.0],
                              lon=[-170.0, 0.0, 120.0])])
    # Box centered near the pole covers it → all longitudes at lat≥75 selected.
    mask = tab.region_mask(0.0, 85.0, 10.0)
    assert list(mask) == [True, True, True]


def test_select_region_sums_and_reports(tab):
    tab._load_done([_fake_day(lat=[0.0, 40.0, 41.0, -40.0])])
    tab.opts.box_size.setValue(5.0)
    tab.select_region(0.0, 40.0)
    assert "2 records" in tab.page.readout.text()
    # Sum of two identical arange(512) spectra.
    line = [l for l in tab.page.fig.axes[0].lines
            if l.get_color() == P.T.ACCENT_CYAN][0]
    assert np.allclose(line.get_ydata(), 2 * np.arange(512))
    # Comparison overlay (scaled all-data average) is present by default.
    assert len(tab.page.fig.axes[0].lines) == 2


def test_select_region_empty_shows_message(tab):
    tab._load_done([_fake_day(lat=[0.0, 1.0, 2.0, 3.0])])
    tab.opts.box_size.setValue(2.0)
    tab.select_region(90.0, -60.0)
    assert "0 records" in tab.opts.status.text()


def test_select_region_without_data_only_reports(tab):
    tab.select_region(10.0, 20.0)
    assert "no data loaded yet" in tab.opts.status.text()


def test_clear_selection_restores_full_spectrum(tab):
    tab._load_done([_fake_day()])
    tab.select_region(0.0, 40.0)
    tab._clear_selection()
    assert tab.page.readout.text() == ""


# ── Keep spectra (checkbox: pin the outgoing selection on each new click) ────
def test_keep_checked_pins_previous_selection(tab):
    tab._load_done([_fake_day(lat=[0.0, 40.0, 41.0, -40.0])])
    tab.opts.box_size.setValue(5.0)
    tab.opts.cb_keep.setChecked(True)
    tab.select_region(0.0, 40.0)           # nothing to pin yet
    assert tab._kept == [] and tab._active is not None
    tab.select_region(0.0, -40.0)          # pins the 40° region
    assert len(tab._kept) == 1
    assert tab._kept[0]["color"] == P.KEEP_COLORS[0]
    assert "lat 40.0°" in tab._kept[0]["label"]
    assert np.allclose(tab._kept[0]["spectrum"], 2 * np.arange(512))
    tab.select_region(0.0, 0.0)            # pins the −40° region too
    assert [k["color"] for k in tab._kept] == P.KEEP_COLORS[:2]


def test_keep_unchecked_replaces_selection(tab):
    tab._load_done([_fake_day(lat=[0.0, 40.0, 41.0, -40.0])])
    tab.opts.box_size.setValue(5.0)
    assert not tab.opts.cb_keep.isChecked()
    tab.select_region(0.0, 40.0)
    tab.select_region(0.0, -40.0)
    assert tab._kept == []                 # nothing pinned while unchecked
    assert "lat -40.0°" in tab._active["label"]


def test_kept_spectra_stay_plotted_with_active(tab):
    tab._load_done([_fake_day(lat=[0.0, 40.0, 41.0, -40.0])])
    tab.opts.box_size.setValue(5.0)
    tab.opts.cb_compare.setChecked(False)
    tab.opts.cb_keep.setChecked(True)
    tab.select_region(0.0, 40.0)
    tab.select_region(0.0, -40.0)
    ax = tab.page.fig.axes[0]
    colors = [l.get_color() for l in ax.lines]
    # One kept line in its palette color + the active selection in cyan.
    assert colors == [P.KEEP_COLORS[0], P.T.ACCENT_CYAN]
    labels = [l.get_label() for l in ax.lines]
    assert "2 records" in labels[0] and "1 records" in labels[1]


def test_unchecking_keep_drops_kept_but_not_active(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab._load_done([_fake_day(lat=[0.0, 40.0, 41.0, -40.0])])
    tab.opts.cb_keep.setChecked(True)
    tab.select_region(0.0, 40.0)
    tab.select_region(0.0, -40.0)
    assert len(tab._kept) == 1
    calls.clear()
    tab.opts.cb_keep.setChecked(False)
    assert tab._kept == []
    assert "waraClearKept();" in calls     # kept boxes removed from the Moon
    assert tab._active is not None         # active selection untouched
    colors = [l.get_color() for l in tab.page.fig.axes[0].lines]
    assert P.T.ACCENT_CYAN in colors and P.KEEP_COLORS[0] not in colors


def test_clear_selection_drops_kept_and_active(tab):
    tab._load_done([_fake_day(lat=[0.0, 40.0, 41.0, -40.0])])
    tab.opts.cb_keep.setChecked(True)
    tab.select_region(0.0, 40.0)
    tab.select_region(0.0, -40.0)
    tab._clear_selection()
    assert tab._kept == [] and tab._active is None
    # Back to the all-data spectrum.
    labels = [l.get_label() for l in tab.page.fig.axes[0].lines]
    assert any("All loaded data" in s for s in labels)


def test_load_drops_active_but_keeps_pinned(tab):
    tab._load_done([_fake_day(lat=[0.0, 40.0, 41.0, -40.0])])
    tab.opts.cb_keep.setChecked(True)
    tab.select_region(0.0, 40.0)
    tab.select_region(0.0, -40.0)
    tab._load_done([_fake_day()])          # reload a new dataset
    assert tab._active is None and len(tab._kept) == 1
    colors = [l.get_color() for l in tab.page.fig.axes[0].lines]
    assert colors == [P.KEEP_COLORS[0]]    # pinned snapshot still plotted


# ── Orbit ground track ────────────────────────────────────────────────────────
def test_track_arrays_positions_time_and_hover(tab):
    tab._load_done([_fake_day(lat=[0.0, 40.0, 80.0, -40.0],
                              lon=[10.0, 20.0, 30.0, 40.0])])
    x, y, z, color, texts, title = tab.track_arrays()
    assert len(x) == len(color) == len(texts) == 4
    # Points sit just above the surface.
    r = np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2 + np.asarray(z) ** 2)
    assert np.all(r > R_MOON_KM)
    # Color = days since the first record; the fake day is a single instant.
    assert np.allclose(color, 0.0)
    assert title == "days since 1998-01-16"
    # Hover text carries UTC date, position, and altitude.
    assert "1998-01-16" in texts[0] and "UTC" in texts[0]
    assert "lon 20.0°" in texts[1] and "lat 40.0°" in texts[1]
    assert "100 km" in texts[0]


def test_track_time_spans_days(tab):
    d1 = _fake_day(stem="1998_016_grs")
    d2 = _fake_day(stem="1998_017_grs")
    d2.time_doy = np.full(d2.n_records, 17.5)   # one day later
    tab._load_done([d1, d2])
    *_, color, _, _ = tab.track_arrays()
    assert np.isclose(max(color) - min(color), 1.0)


def test_track_subsamples_but_keeps_floor(tab, monkeypatch):
    monkeypatch.setattr(P, "MIN_TRACK_POINTS", 3)
    tab._load_done([_fake_day(lat=np.linspace(-80, 80, 8))])
    x, *_ = tab.track_arrays()
    # stride = 8 // 3 = 2 → ceil(8/2) = 4 points, never below the floor of 3.
    assert len(x) == 4
    assert len(x) >= 3


def _fake_meta(tab):
    """Inject a small fake metadata table (avoids depending on the real CSV)."""
    tab._meta = {
        "utc": np.array(["1998-01-16T00:00", "1998-01-16T00:08",
                         "1999-07-28T00:00"], dtype="datetime64[s]"),
        "product": np.array(["1998_016_grs"] * 2 + ["1999_209_grs"]),
        "lon": np.array([10.0, 11.0, -120.0]),
        "lat": np.array([-80.0, 0.0, 80.0]),
        "alt_km": np.array([100.0, 100.5, 30.0]),
    }


def test_toggle_track_without_data_uses_bundled_metadata(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    _fake_meta(tab)
    tab.opts.ed_start.setText("")           # no bounds → whole mission
    tab.opts.ed_end.setText("")
    tab.opts.cb_track.setChecked(True)
    assert any(c.startswith("waraShowTrack(") for c in calls)
    assert "3 of 3 bundled-metadata points" in tab.opts.status.text()
    assert "1998-01-16 → 1999-07-28" in tab.opts.status.text()


def test_metadata_track_respects_date_and_phase(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    _fake_meta(tab)
    # Date range keeps only the 1998-01-16 rows.
    mask = tab.metadata_selection()          # defaults: 1998-01-16..1998-01-31
    assert list(mask) == [True, True, False]
    # Low orbit phase keeps only the 1999 row.
    tab.opts.ed_start.setText(""); tab.opts.ed_end.setText("")
    tab.opts.phase.setCurrentIndex(2)        # Low (~30-40 km)
    assert list(tab.metadata_selection()) == [False, False, True]
    tab.opts.phase.setCurrentIndex(1)        # High (~100 km)
    assert list(tab.metadata_selection()) == [True, True, False]


def test_metadata_track_redraws_on_filter_change(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    _fake_meta(tab)
    tab.opts.ed_start.setText("")            # whole mission
    tab.opts.ed_end.setText("")
    tab.opts.cb_track.setChecked(True)
    calls.clear()
    tab.opts.phase.setCurrentIndex(2)        # orbit change → redraw (1999 row)
    assert any(c.startswith("waraShowTrack(") for c in calls)
    calls.clear()
    tab.opts.ed_start.setText("1999-09-01")  # empty range → cleared + message
    tab.opts.ed_start.editingFinished.emit()
    assert "waraClearTrack();" in calls
    assert "No LP products in that date/orbit range" in tab.opts.status.text()


def test_toggle_track_no_data_no_metadata_reports(tab, monkeypatch):
    monkeypatch.setattr(tab, "metadata", lambda: None)
    tab.opts.cb_track.setChecked(True)
    assert "no bundled LP metadata" in tab.opts.status.text()


def test_toggle_track_draws_and_clears(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab._load_done([_fake_day()])
    tab.opts.cb_track.setChecked(True)
    assert any(c.startswith("waraShowTrack(") for c in calls)
    calls.clear()
    tab.opts.cb_track.setChecked(False)
    assert calls == ["waraClearTrack();"]


def test_load_redraws_track_when_enabled(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab.opts.cb_track.blockSignals(True)        # simulate pre-checked box
    tab.opts.cb_track.setChecked(True)
    tab.opts.cb_track.blockSignals(False)
    tab._load_done([_fake_day()])
    assert any(c.startswith("waraShowTrack(") for c in calls)


def test_availability_label_from_metadata(tab, tmp_path):
    tab.data_dir = tmp_path
    tab._meta = {
        "utc": np.array(["1998-01-16T00:00", "1999-07-28T23:52"],
                        dtype="datetime64[s]"),
        "product": np.array(["1998_016_grs", "1999_209_grs"]),
        "lon": np.zeros(2), "lat": np.zeros(2), "alt_km": np.full(2, 100.0),
    }
    tab._refresh_availability()
    text = tab.opts.lbl_avail.text()
    assert "2 daily products" in text
    assert "1998-01-16" in text and "1999-07-28" in text
    assert "(0 downloaded locally)" in text
    (tmp_path / "1998_016_grs.xml").write_text("x")
    (tmp_path / "1998_016_grs.dat").write_text("x")
    tab._refresh_availability()
    assert "(1 downloaded locally)" in tab.opts.lbl_avail.text()


# ── Calibrated dataset (abundance maps) ──────────────────────────────────────
def _fake_abundance_table():
    return {
        "lat_s": np.array([-90.0, 0.0]), "lat_n": np.array([0.0, 90.0]),
        "lon_w": np.array([-180.0, -180.0]), "lon_e": np.array([180.0, 180.0]),
        "Th": np.array([1.0, 4.0]), "K": np.array([300.0, 800.0]),
        "U": np.array([0.2, 0.5]), "FeO": np.array([0.05, 0.1]),
        "TiO2": np.array([0.001, 0.004]), "MgO": np.array([0.07, 0.08]),
        "Al2O3": np.array([0.27, 0.22]), "SiO2": np.array([0.45, 0.44]),
        "CaO": np.array([0.16, 0.15]),
    }


def test_dataset_dropdown_defaults_and_gating(tab):
    assert tab.opts.dataset.currentIndex() == 0            # raw by default
    assert not tab.opts.element.isEnabled()
    assert not tab.opts.resolution.isEnabled()
    assert not tab.opts.cmap.isEnabled()
    assert tab.opts.element.currentText() == "Th"          # the famous LP map
    assert tab.opts.cmap.currentText() == "Viridis"
    assert not hasattr(tab.opts, "btn_globe")              # rebuild is automatic


def test_detail_change_rebuilds_globe(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_build_globe", lambda force=False: calls.append(force))
    tab.opts.detail.setCurrentText("Standard (1.0°)")
    assert calls == [True]


def test_colormap_choice_applies_to_drape(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab._globe_ready = True
    tab._globe_mesh = (36, 18)
    tab._abundance[2] = _fake_abundance_table()
    tab.opts.dataset.setCurrentIndex(1)
    assert tab.opts.cmap.isEnabled()
    assert '"Viridis"' in calls[-1]
    calls.clear()
    tab.opts.cmap.setCurrentText("Jet")                    # re-drapes live
    assert any('"Jet"' in c and c.startswith("waraSetSurface(") for c in calls)
    # Only plotly.js built-in colorscale names are offered.
    items = [tab.opts.cmap.itemText(i) for i in range(tab.opts.cmap.count())]
    assert items == P.ABUNDANCE_COLORSCALES


def test_switch_to_abundance_drapes_map(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab._globe_ready = True
    tab._globe_mesh = (36, 18)
    tab._abundance[2] = _fake_abundance_table()            # pre-cached: no net
    tab.opts.dataset.setCurrentIndex(1)
    assert tab.opts.element.isEnabled() and tab.opts.resolution.isEnabled()
    assert any(c.startswith("waraSetSurface(") for c in calls)
    assert "Th (ppm)" in calls[-1]
    assert "Th abundance map" in tab.opts.status.text()
    # Element change re-drapes with the new title.
    calls.clear()
    tab.opts.element.setCurrentText("FeO")
    assert any("FeO (wt. fraction)" in c for c in calls)


def test_switch_back_to_raw_restores_albedo(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab._globe_ready = True
    tab._globe_mesh = (36, 18)
    tab._abundance[2] = _fake_abundance_table()
    tab.opts.dataset.setCurrentIndex(1)
    calls.clear()
    tab.opts.dataset.setCurrentIndex(0)
    assert calls == ["waraClearOverlay();", "waraResetSurface();"]
    assert not tab.opts.element.isEnabled()


def test_globe_reload_reapplies_abundance(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab._globe_ready = True
    tab._globe_mesh = (36, 18)
    tab._abundance[2] = _fake_abundance_table()
    tab.opts.dataset.setCurrentIndex(1)
    calls.clear()
    tab._on_globe_loaded(True)
    assert any(c.startswith("waraSetSurface(") for c in calls)


# ── Mission info dialog ───────────────────────────────────────────────────────
def test_mission_info_dialog_lists_docs_and_switches(qapp):
    from wara.planetary import LP_DOCUMENTS
    dlg = P.MissionInfoDialog()
    assert dlg.doc_list.count() == len(LP_DOCUMENTS)
    docs = {label: f"text of {label}" for label in LP_DOCUMENTS}
    dlg.set_docs(docs)
    assert dlg.viewer.toPlainText() == f"text of {list(LP_DOCUMENTS)[0]}"
    dlg.doc_list.setCurrentRow(2)
    assert dlg.viewer.toPlainText() == f"text of {list(LP_DOCUMENTS)[2]}"
    dlg.deleteLater()


def test_mission_info_fetches_once_and_retries_failures(tab, monkeypatch):
    from wara.planetary import LP_DOCUMENTS
    runs = []
    monkeypatch.setattr(tab, "_run", lambda fn, done: runs.append(fn))
    monkeypatch.setattr(P.MissionInfoDialog, "show", lambda self: None)
    monkeypatch.setattr(P.MissionInfoDialog, "raise_", lambda self: None)
    tab._show_mission_info()
    assert len(runs) == 1                       # first open fetches
    tab._info_docs_loaded({l: "ok" for l in LP_DOCUMENTS})
    tab._show_mission_info()
    assert len(runs) == 1                       # fully loaded → no refetch
    # A failed doc makes the next open retry.
    docs = {l: "ok" for l in LP_DOCUMENTS}
    docs["References"] = "(Could not fetch this document: timeout)"
    tab._info_docs_loaded(docs)
    assert "some documents" in tab.opts.status.text()
    tab._show_mission_info()
    assert len(runs) == 2


# ── Transparent composition overlay ──────────────────────────────────────────
def test_opacity_control_defaults_and_gating(tab):
    assert tab.opts.opacity.value() == 100.0
    assert not tab.opts.opacity.isEnabled()
    tab.opts.dataset.setCurrentIndex(1)
    assert tab.opts.opacity.isEnabled()


def test_opaque_abundance_recolors_base_no_overlay(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab._globe_ready = True
    tab._globe_mesh = (36, 18)
    tab._abundance[2] = _fake_abundance_table()
    tab.opts.dataset.setCurrentIndex(1)
    assert any(c.startswith("waraSetSurface(") for c in calls)
    assert not any(c.startswith("waraSetOverlay(") for c in calls)
    assert tab._surface_is_abundance


def test_transparent_overlay_over_albedo(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab._globe_ready = True
    tab._globe_mesh = (36, 18)
    tab._abundance[2] = _fake_abundance_table()
    tab.opts.dataset.setCurrentIndex(1)
    calls.clear()
    tab.opts.opacity.setValue(60.0)
    # Base restored to albedo, then the semi-transparent overlay added.
    assert "waraResetSurface();" in calls
    assert any(c.startswith("waraSetOverlay(") and "0.60" in c for c in calls)
    assert not tab._surface_is_abundance
    assert "60 % opacity over the albedo Moon" in tab.opts.status.text()


def test_transparent_overlay_from_elevation_base_uses_albedo(tab, monkeypatch):
    # Coming from the elevation drape, a semi-transparent abundance overlay
    # must float over the *albedo* Moon, never the elevation color map —
    # abundance-over-elevation drew two colorbars (dropped in 6a94cd2).
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab._globe_ready = True
    tab._globe_mesh = (36, 18)
    tab._abundance[2] = _fake_abundance_table()
    tab._lola = _fake_lola()
    tab.opts.dataset.setCurrentIndex(2)          # view elevation first...
    calls.clear()
    tab.opts.dataset.setCurrentIndex(1)          # ...then composition
    tab.opts.opacity.setValue(50.0)
    # Base reset to albedo + overlay on top; no elevation color drape left.
    assert "waraResetSurface();" in calls
    assert any(c.startswith("waraSetOverlay(") and "0.50" in c for c in calls)
    assert not any(c.startswith("waraSetSurface(") and "Elevation (km)" in c
                   for c in calls)
    assert not tab._surface_is_abundance
    assert "over the albedo Moon" in tab.opts.status.text()
    # Leaving Calibrated clears the overlay.
    calls.clear()
    tab.opts.dataset.setCurrentIndex(0)
    assert "waraClearOverlay();" in calls


def test_opacity_back_to_full_removes_overlay(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab._globe_ready = True
    tab._globe_mesh = (36, 18)
    tab._abundance[2] = _fake_abundance_table()
    tab.opts.dataset.setCurrentIndex(1)
    tab.opts.opacity.setValue(40.0)
    calls.clear()
    tab.opts.opacity.setValue(100.0)
    assert "waraClearOverlay();" in calls
    assert any(c.startswith("waraSetSurface(") for c in calls)


# ── LOLA topography ───────────────────────────────────────────────────────────
def _fake_lola():
    lat = np.linspace(90, -90, 8)
    return {"elev_km": np.tile(np.linspace(-2.0, 4.0, 16), (8, 1)), "ppd": 4}


def test_elevation_dataset_drapes_lola(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab._globe_ready = True
    tab._globe_mesh = (36, 18)
    tab._lola = _fake_lola()
    tab.opts.dataset.setCurrentIndex(2)         # Elevation (LOLA)
    assert tab.opts.cmap.isEnabled()
    assert not tab.opts.element.isEnabled()     # abundance-only controls
    assert any(c.startswith("waraSetSurface(") and "Elevation (km)" in c
               for c in calls)
    assert "LOLA elevation" in tab.opts.status.text()


def test_topo_toggle_displaces_and_restores(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab._globe_ready = True
    tab._globe_mesh = (36, 18)
    tab._lola = _fake_lola()
    assert not tab.opts.exag.isEnabled()
    tab.opts.cb_topo.setChecked(True)
    assert tab.opts.exag.isEnabled()
    assert any(c.startswith("waraSetTopo(") for c in calls)
    assert any("10" in c for c in calls if c.startswith("waraSetTopo("))
    calls.clear()
    # Exaggeration changes re-use the page-cached elevation (null payload).
    tab.opts.exag.setValue(25.0)
    assert calls == ["waraSetTopo(null, 25);"]
    calls.clear()
    tab.opts.cb_topo.setChecked(False)
    assert calls == ["waraResetTopo();"]
    assert not tab.opts.exag.isEnabled()


def test_globe_reload_reapplies_topo_and_drape(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab._globe_ready = True
    tab._globe_mesh = (36, 18)
    tab._lola = _fake_lola()
    tab.opts.dataset.setCurrentIndex(2)
    tab.opts.cb_topo.setChecked(True)
    calls.clear()
    tab._on_globe_loaded(True)
    assert any(c.startswith("waraSetSurface(") for c in calls)   # drape back
    assert any(c.startswith("waraSetTopo([") for c in calls)     # full resend


# ── 2D equirectangular projection ─────────────────────────────────────────────
def test_flat_checkbox_defaults_off(tab):
    assert not tab.opts.cb_flat.isChecked()


class _FakePage:
    def __init__(self, log):
        self._log = log

    def runJavaScript(self, script, *a):
        self._log.append(("js", script))


class _FakeSignal:
    def connect(self, *a):
        pass

    def disconnect(self, *a):
        pass


class _FakeView:
    def __init__(self, log):
        self._log = log
        self._page = _FakePage(log)
        self.titleChanged = _FakeSignal()
        self.loadFinished = _FakeSignal()

    def setUrl(self, url):
        self._log.append(("url", os.path.normpath(url.toLocalFile())))

    def page(self):
        return self._page


def _norm(p):
    return os.path.normpath(p)


def test_flat_toggle_swaps_to_native_map_page(tab, monkeypatch, tmp_path):
    log = []
    tab.page.web_view = _FakeView(log)
    tab._globe_ready = True
    tab._globe_tmp = str(tmp_path / "globe.html")
    # The page build runs in a worker; run it synchronously with a stub html.
    monkeypatch.setattr(tab, "_run",
                        lambda fn, done: done("<html>flat</html>"))
    tab.opts.cb_flat.setChecked(True)
    urls = [v for k, v in log if k == "url"]
    assert len(urls) == 1 and urls[0] == _norm(tab._flat_tmp)
    assert "native-resolution" in tab.opts.status.text()
    # The page is cached: toggling again re-uses it (no second build).
    monkeypatch.setattr(tab, "_run",
                        lambda fn, done: pytest.fail("must not rebuild"))
    tab.opts.cb_flat.setChecked(False)
    tab.opts.cb_flat.setChecked(True)
    urls = [v for k, v in log if k == "url"]
    assert urls[-2:] == [_norm(tab._globe_tmp), _norm(tab._flat_tmp)]


def test_flat_off_returns_to_globe_page(tab, tmp_path):
    log = []
    tab.page.web_view = _FakeView(log)
    tab._globe_ready = True
    tab._globe_tmp = str(tmp_path / "globe.html")
    tab._flat_tmp = str(tmp_path / "flat.html")
    tab.opts.cb_flat.setChecked(True)
    tab.opts.cb_flat.setChecked(False)
    urls = [v for k, v in log if k == "url"]
    assert urls == [_norm(tab._flat_tmp), _norm(tab._globe_tmp)]
    assert "3D globe restored" in tab.opts.status.text()


def test_page_load_reapplies_state_but_skips_topo_when_flat(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab._globe_ready = True
    tab._lola = _fake_lola()
    tab.opts.cb_topo.blockSignals(True); tab.opts.cb_topo.setChecked(True)
    tab.opts.cb_topo.blockSignals(False)
    tab.opts.cb_flat.blockSignals(True); tab.opts.cb_flat.setChecked(True)
    tab.opts.cb_flat.blockSignals(False)
    tab._on_globe_loaded(True)
    assert calls[0] == "waraSetGrid(false);"     # grid state re-applied
    assert not any(c.startswith("waraSetTopo(") for c in calls)  # flat: no relief
    calls.clear()
    tab.opts.cb_flat.blockSignals(True); tab.opts.cb_flat.setChecked(False)
    tab.opts.cb_flat.blockSignals(False)
    tab._on_globe_loaded(True)                   # back on the globe page
    assert any(c.startswith("waraSetTopo(") for c in calls)


def test_globe_rebuild_while_flat_does_not_switch_pages(tab, tmp_path):
    log = []
    tab.page.web_view = _FakeView(log)
    tab.page.ensure_web_view = lambda: tab.page.web_view
    tab._globe_ready = True
    tab._flat_tmp = str(tmp_path / "flat.html")
    tab.opts.cb_flat.blockSignals(True); tab.opts.cb_flat.setChecked(True)
    tab.opts.cb_flat.blockSignals(False)
    tab._globe_done("<html>globe</html>")
    assert [v for k, v in log if k == "url"] == []   # stayed on the 2D map
    assert "will show when the 2D map is turned off" in tab.opts.status.text()
    assert tab._globe_tmp                            # rebuilt page kept for later


def test_build_flat_html_has_native_image_and_full_api():
    pytest.importorskip("plotly")
    html = P.build_flat_html()
    assert "data:image/png;base64," in html          # native-resolution texture
    assert "wara-carrier" in html                    # hover/click carrier
    for fn in ("waraShowBox", "waraKeepBox", "waraClearBoxes", "waraShowTrack",
               "waraShowMarks", "waraSetGrid", "waraSetSurface",
               "waraResetSurface", "waraSetOverlay", "waraClearOverlay",
               "waraSetTopo"):
        assert fn in html, fn
    assert "document.title" in html                  # click bridge


# ── Landmarks ─────────────────────────────────────────────────────────────────
def test_lunar_landmarks_are_well_formed():
    from wara.planetary import LUNAR_LANDMARKS
    assert len(LUNAR_LANDMARKS) >= 15
    names = [nm for _, _, nm in LUNAR_LANDMARKS]
    assert len(names) == len(set(names))            # unique names
    for lon, lat, name in LUNAR_LANDMARKS:
        assert -180.0 <= lon < 180.0 and -90.0 <= lat <= 90.0, name
    assert "Mare Imbrium" in names and "South pole" in names


def test_grid_and_landmarks_default_off(tab):
    assert not tab.opts.cb_marks.isChecked()
    assert not tab.opts.cb_grid.isChecked()


def test_toggle_landmarks_draws_and_clears(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab.opts.cb_marks.setChecked(True)
    assert len(calls) == 1 and calls[0].startswith("waraShowMarks(")
    assert "Mare Imbrium" in calls[0]
    # Landmark dots sit just above the surface (no data needed).
    assert tab._spectra is None
    calls.clear()
    tab.opts.cb_marks.setChecked(False)
    assert calls == ["waraClearMarks();"]


def test_globe_load_redraws_landmarks_when_checked(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab.opts.cb_marks.blockSignals(True)     # pre-checked box, no toggle draw
    tab.opts.cb_marks.setChecked(True)
    tab.opts.cb_marks.blockSignals(False)
    tab._on_globe_loaded(True)
    assert any(c.startswith("waraShowMarks(") for c in calls)
    # Unchecked → not redrawn.
    tab.opts.cb_marks.blockSignals(True)
    tab.opts.cb_marks.setChecked(False)
    tab.opts.cb_marks.blockSignals(False)
    calls.clear()
    tab._on_globe_loaded(True)
    assert not any(c.startswith("waraShowMarks(") for c in calls)


# ── Send to Spectrum ──────────────────────────────────────────────────────────
class _FakeApp:
    def __init__(self):
        self.received = []

    def load_external_spectrum(self, spect, name, switch_tab=True):
        self.received.append((spect, name, switch_tab))

    def statusBar(self):
        class _SB:
            def showMessage(self, *_):
                pass
        return _SB()


def test_send_to_spectrum_hands_region_sum(qapp):
    opts = P.PlanetaryOptions(); page = P.PlanetaryPage()
    fake = _FakeApp()
    ctl = P.PlanetaryController(fake, opts, page)
    ctl._load_done([_fake_day(lat=[0.0, 40.0, 41.0, -40.0])])
    ctl.opts.box_size.setValue(5.0)
    ctl.select_region(0.0, 40.0)
    ctl._send_to_spectrum()
    assert len(fake.received) == 1
    spect, name, switch = fake.received[0]
    assert np.allclose(spect.counts, 2 * np.arange(512))
    assert "LP-GRS" in name and "lat 40.0" in name
    assert switch is False                 # stay on the Planetary tab
    page.deleteLater(); opts.deleteLater()


def test_send_without_selection_reports(tab):
    tab._send_to_spectrum()
    assert "Click a region" in tab.opts.status.text()


# ── Click bridge ──────────────────────────────────────────────────────────────
def test_title_bridge_parses_lonlat_when_armed(tab, monkeypatch):
    got = []
    monkeypatch.setattr(tab, "select_region", lambda lon, lat: got.append((lon, lat)))
    tab.opts.btn_select.setChecked(True)       # arm
    tab._on_title("lonlat:12.500,-33.250:7")
    assert got == [(12.5, -33.25)]
    # Each selection disarms the button — the next click must not select.
    assert not tab.opts.btn_select.isChecked()
    tab._on_title("lonlat:1.000,2.000:8")
    assert len(got) == 1
    assert "Arm 'Select region'" in tab.opts.status.text()
    tab.opts.btn_select.setChecked(True)
    tab._on_title("Some Page Title")           # non-bridge titles are ignored
    tab._on_title("lonlat:garbage")            # malformed payloads are ignored
    assert len(got) == 1
    assert tab.opts.btn_select.isChecked()     # bad payloads don't disarm


def test_select_button_label_reflects_armed_state(tab):
    assert tab.opts.btn_select.text() == "Select region"
    tab.opts.btn_select.setChecked(True)
    assert tab.opts.btn_select.text() == "Select region — armed"
    tab.opts.btn_select.setChecked(False)
    assert tab.opts.btn_select.text() == "Select region"


# ── Region box geometry ───────────────────────────────────────────────────────
def test_region_box_outline_sits_above_surface():
    x, y, z = P.region_box_xyz(30.0, 20.0, 10.0)
    r = np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2 + np.asarray(z) ** 2)
    assert np.all(r > R_MOON_KM)               # lifted off the surface
    lon, lat = xyz_to_lonlat(np.asarray(x), np.asarray(y), np.asarray(z))
    assert lat.min() >= 9.9 and lat.max() <= 30.1
    assert lon.min() >= 19.8 and lon.max() <= 40.2


def test_region_box_clamps_at_pole():
    _, _, z = P.region_box_xyz(0.0, 88.0, 10.0)
    lat_max = np.degrees(np.arcsin(np.max(z) / (R_MOON_KM * 1.006)))
    assert lat_max <= 90.0 + 1e-6


# ── Globe HTML ────────────────────────────────────────────────────────────────
def test_globe_html_contains_bridge_and_moon():
    pytest.importorskip("plotly")
    html = P.build_globe_html(90, 45, graticule=True)
    assert "plotly-graph-div" in html
    assert "waraShowBox" in html and "waraClearBoxes" in html
    assert "waraKeepBox" in html and "waraClearSel" in html
    assert "waraShowTrack" in html and "waraClearTrack" in html
    assert "waraShowMarks" in html and "waraClearMarks" in html
    assert "waraSetGrid" in html and "wara-grat" in html
    assert "waraSetSurface" in html and "waraResetSurface" in html
    assert "waraSetTopo" in html and "waraResetTopo" in html
    assert "waraSetOverlay" in html and "waraClearOverlay" in html

    assert "plotly_click" in html and "document.title" in html
    assert "wara-readout" in html              # JS hover readout overlay


def test_grid_checkbox_toggles_in_place(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab.opts.cb_grid.setChecked(True)
    assert calls == ["waraSetGrid(true);"]
    calls.clear()
    tab.opts.cb_grid.setChecked(False)
    assert calls == ["waraSetGrid(false);"]


def test_globe_js_decodes_base64_typed_arrays():
    """plotly.py >= 5.24 serialises numpy arrays as base64 typed-array specs
    ({dtype, bdata, shape}) rather than nested JSON lists, so ``gd.data[0].x``
    is an object with no ``.length``. Any helper that re-derives geometry from
    the base surface must go through ``grid2d()`` first, or it iterates zero
    times and silently builds an empty -- and therefore invisible -- trace
    (this is what broke the semi-transparent abundance overlay and the
    topography relief on machines with a newer plotly)."""
    js = P._GLOBE_JS
    assert "grid2d" in js and "bdata" in js and "atob" in js
    # Every read of a base-surface data array must be wrapped in grid2d(...).
    code = re.sub(r"/\*.*?\*/", "", js, flags=re.S)   # drop /* comments */
    for m in re.finditer(r"\S*gd\.data\[0\]\.(x|y|z|surfacecolor)", code):
        assert m.group(0).startswith("grid2d("), m.group(0)


def test_globe_figure_uses_typed_array_specs():
    """Guards the assumption above: confirm the installed plotly really does
    emit base64 typed-array specs for the Moon mesh. If this ever stops being
    true the decoder is harmlessly bypassed (grid2d passes nested arrays
    through), so this is documentation, not a constraint."""
    pytest.importorskip("plotly")
    html = P.build_globe_html(8, 6, graticule=False)
    i = html.rfind("Plotly.newPlot(")
    payload = html[i:i + 4000]
    assert '"bdata"' in payload or '"x":[[' in payload


def test_globe_html_graticule_initial_visibility():
    pytest.importorskip("plotly")
    html = P.build_globe_html(90, 45, graticule=False)
    # Graticule traces are present (toggleable) but start hidden.
    assert "wara-grat" in html and '"visible":false' in html


def test_empty_date_fields_mean_no_bound(tab):
    tab.opts.ed_start.setText("")
    tab.opts.ed_end.setText("")
    start, end, phase = tab._dates_phase()
    assert start is None and end is None and phase is None


def test_landmarks_js_has_dot_and_label_shells(tab, monkeypatch):
    calls = []
    monkeypatch.setattr(tab, "_js", lambda script: calls.append(script))
    tab._draw_landmarks()
    script = calls[0]
    assert script.startswith("waraShowMarks(")
    # 8 args: dots xyz, labels xyz, names, hover — labels on a higher shell.
    from wara.planetary.moon import LUNAR_LANDMARKS
    n_args = script.count("[")               # 8 top-level JSON arrays
    assert n_args >= 8
    assert "Mare Imbrium" in script


def test_globe_detail_settings_are_valid():
    assert P.GLOBE_DETAIL_DEFAULT in P.GLOBE_DETAIL
    for n_lon, n_lat in P.GLOBE_DETAIL.values():
        assert n_lon == 2 * n_lat              # equirectangular aspect
