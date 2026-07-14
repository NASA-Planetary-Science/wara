"""Offscreen GUI tests for the Spectrum-tab 'Isotope ID' hover feature
(wara.gui: SpectrumOptions.cb_isotope_id + SpectrumCanvas hover overlay +
WaraApp._refresh_isotope_id, which calls wara.nuclide_identificator)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication

from wara import spectrum as sp
from wara import peaksearch as ps
from wara.gui.app import WaraApp


# Synthetic 16O-rich spectrum, linearly calibrated (E = 1.5·ch).
PEAK_ENERGIES = (846.8, 3684.5, 6129.9, 6917.1)
SLOPE = 1.5


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _calibrated_spectrum():
    ch = np.arange(8192)
    counts = np.full_like(ch, 5.0, dtype=float)
    for e in PEAK_ENERGIES:
        counts += 4000.0 * np.exp(-0.5 * ((ch - e / SLOPE) / 3.0) ** 2)
    return sp.Spectrum(counts=counts, energies=SLOPE * ch, e_units="keV")


@pytest.fixture
def app(qapp):
    w = WaraApp()
    spect = _calibrated_spectrum()
    w.spect = spect
    w._spect_orig = spect.copy()
    w._active_name = "syn"
    w._refresh()
    yield w
    w.close()


def _find_peaks(w):
    pf = w.spectrum_opts.pf_panel
    pf.snr.setText("4"); pf.ref_ch.setText("4000"); pf.ref_fwhm.setText("3")
    w._find_peaks()


def _fake_motion(canvas, px, py):
    """A matplotlib-like motion event at data point (px, py)."""
    canvas.figure.canvas.draw()
    disp = canvas.ax.transData.transform((px, py))

    class _Ev:
        pass
    ev = _Ev()
    ev.inaxes = canvas.ax
    ev.x, ev.y = float(disp[0]), float(disp[1])
    ev.xdata, ev.ydata = px, py
    return ev


# ── checkbox + activation conditions ──────────────────────────────────────────
def test_checkbox_off_by_default(app):
    assert not app.spectrum_opts.cb_isotope_id.isChecked()
    assert not app.spectrum_page.canvas._iso_on


def test_inactive_without_calibration(qapp):
    w = WaraApp()
    ch = np.arange(4096)
    counts = np.full_like(ch, 5.0, dtype=float)
    counts += 4000.0 * np.exp(-0.5 * ((ch - 2000) / 3.0) ** 2)
    w.spect = sp.Spectrum(counts=counts)          # uncalibrated (channel axis)
    w._spect_orig = w.spect.copy(); w._active_name = "u"; w._refresh()
    w.search = ps.PeakSearch(w.spect, ref_x=2000, ref_fwhm=3, fwhm_at_0=1.0, min_snr=4)
    w.spectrum_page.canvas.set_peaks([(2000.0, 4000.0, "2000")])
    w.spectrum_opts.cb_isotope_id.setChecked(True)
    assert not w.spectrum_page.canvas._iso_on        # calibration is required
    w.close()


def test_inactive_without_peaks(app):
    app.spectrum_opts.cb_isotope_id.setChecked(True)   # calibrated but no search
    assert not app.spectrum_page.canvas._iso_on


def test_active_when_calibrated_with_peaks_and_identifies(app):
    _find_peaks(app)
    app.spectrum_opts.cb_isotope_id.setChecked(True)
    canvas = app.spectrum_page.canvas
    assert canvas._iso_on
    assert len(canvas._iso_info) == len(canvas.peak_xs())
    # The 6129 keV peak's hover text should name 16O (TALYS) and the energy.
    key = next(k for k in canvas._iso_info if abs(k - 6129.9) < app._iso_tol(6129.9))
    text = canvas._iso_info[key]
    assert "16O" in text
    assert "keV" in text


def test_clearing_peaks_disables(app):
    _find_peaks(app)
    app.spectrum_opts.cb_isotope_id.setChecked(True)
    assert app.spectrum_page.canvas._iso_on
    app._clear_peaks()
    assert not app.spectrum_page.canvas._iso_on


# ── canvas hover overlay (isolated from the identifier) ───────────────────────
def test_hover_shows_and_hides_annotation(app):
    canvas = app.spectrum_page.canvas
    canvas.set_peaks([(e, 4000.0, f"{e:.0f}") for e in PEAK_ENERGIES])
    canvas.set_isotope_id(True, {round(6129.9, 3): "6129.9 keV<br>TALYS 14 MeV:  16O 98%"})

    # Hover the 6129.9 peak → tooltip appears with that text.
    canvas._update_iso_hover(_fake_motion(canvas, 6129.9, 4000.0))
    assert canvas._iso_hover == round(6129.9, 3)
    assert "16O" in canvas._iso_label.text()

    # Hover a peak with no info (846.8) → tooltip hidden.
    canvas._update_iso_hover(_fake_motion(canvas, 846.8, 4000.0))
    assert canvas._iso_hover is None


def test_zero_percent_candidates_are_dropped(qapp):
    import pandas as pd
    from wara.gui.app import WaraApp
    results = {
        "DB1": pd.DataFrame({"Energy": [100.0, 100.0],
                             "Isotope": ["AA", "BB"],          # 98 % and ~0 %
                             "Probability": [0.98, 0.002]}),
        "DB2": pd.DataFrame({"Energy": [100.0],
                             "Isotope": ["CC"], "Probability": [0.001]}),  # ~0 %
    }
    text = WaraApp._format_iso_text(100.0, results)
    assert "AA" in text and "98%" in text          # rank-1 candidate kept
    assert "BB" not in text          # rank-2 0 % candidate dropped
    assert "DB2" not in text         # whole database dropped (only 0 % match)
    assert " 0%" not in text


def test_isotope_names_are_cyan(qapp):
    import pandas as pd
    from wara.gui import isotope_id, theme as T
    results = {"DB": pd.DataFrame({"Energy": [661.7], "Isotope": ["137Cs"],
                                   "Probability": [0.9], "Lines seen": [1]})}
    html = isotope_id.format_html(661.7, results)
    assert f"color:{T.ACCENT_CYAN}'>137Cs" in html      # isotope name in cyan


def test_escape_peaks_shown_as_possibility_not_candidate(qapp):
    import pandas as pd
    from wara.gui import isotope_id
    # 6129.9 is identified (16O); 5618.9 is shown as a SEP possibility of it.
    results = {"DB": pd.DataFrame({"Energy": [6129.9], "Isotope": ["16O"],
                                   "Probability": [0.98], "Lines seen": [4]})}
    esc_html = isotope_id.format_html(5618.9, results, escapes=[("SEP", 6129.9)])
    assert "possibly SEP of 6129.9 keV" in esc_html
    assert "16O" in esc_html                        # parent isotope named
    # The 6129.9 peak itself carries no escape note.
    assert "possibly" not in isotope_id.format_html(6129.9, results)


def test_gui_identify_excludes_escape_peaks_from_scoring(qapp):
    from wara.gui import isotope_id
    res = isotope_id.identify([6129.9, 5618.9])
    for frame in res.values():
        rows = frame[abs(frame["Energy"] - 5618.9) < 1e-6]
        rows = rows[rows["Isotope"].notna()]
        assert (rows["Match type"] == "full").all()   # no SEP/DEP candidates


def test_set_isotope_id_off_hides_annotation(app):
    canvas = app.spectrum_page.canvas
    canvas.set_peaks([(6129.9, 4000.0, "6129")])
    canvas.set_isotope_id(True, {round(6129.9, 3): "text"})
    canvas._update_iso_hover(_fake_motion(canvas, 6129.9, 4000.0))
    assert canvas._iso_hover is not None
    canvas.set_isotope_id(False)
    assert canvas._iso_hover is None
    assert not canvas._iso_on


# ── drop-down panel: backward compat + database / Z-range controls ─────────────
def test_checkbox_alias_still_toggles_activation(app):
    # The on/off checkbox now lives inside iso_panel but keeps the old alias.
    assert app.spectrum_opts.cb_isotope_id is app.spectrum_opts.iso_panel.cb_enable
    _find_peaks(app)
    app.spectrum_opts.cb_isotope_id.setChecked(True)
    assert app.spectrum_page.canvas._iso_on


def test_panel_defaults(app):
    iso = app.spectrum_opts.iso_panel
    assert len(iso.selected_databases()) == len(iso.db_checks)   # all on
    assert iso.z_range() == (1, 118)                             # full range


def test_deselecting_database_changes_hover_result(app):
    _find_peaks(app)
    iso = app.spectrum_opts.iso_panel
    app.spectrum_opts.cb_isotope_id.setChecked(True)
    canvas = app.spectrum_page.canvas
    key = next(k for k in canvas._iso_info if abs(k - 6129.9) < app._iso_tol(6129.9))
    assert "TALYS 14 MeV" in canvas._iso_info[key]
    # Drop the TALYS library → its 16O line disappears from the hover text.
    iso.db_checks["TALYS 14 MeV"].setChecked(False)
    assert "TALYS 14 MeV" not in canvas._iso_info[key]


def test_z_range_excludes_element_in_hover(app):
    _find_peaks(app)
    iso = app.spectrum_opts.iso_panel
    app.spectrum_opts.cb_isotope_id.setChecked(True)
    canvas = app.spectrum_page.canvas
    key = next(k for k in canvas._iso_info if abs(k - 6129.9) < app._iso_tol(6129.9))
    assert "16O" in canvas._iso_info[key]
    # Restrict to Z ≥ 50: oxygen (Z=8) can no longer be a candidate.
    iso.z_min.setValue(50)
    assert "16O" not in canvas._iso_info[key]


def test_export_table_builder_columns_and_rows(app):
    from wara.gui import isotope_id
    _find_peaks(app)
    energies = [float(x) for x in app.spectrum_page.canvas.peak_xs()]
    df = isotope_id.best_per_database_table(energies)
    assert list(df.columns) == [
        "Energy (keV)", "Database", "Isotope (parent)", "Element", "Daughter",
        "Decay", "Line strength", "Strength type", "Probability (%)",
        "Matched line (keV)", "Lines seen"]
    assert df["Isotope (parent)"].astype(str).str.contains("16O").any()
    assert (df["Element"] == "O").any()
    # Every match carries a line strength with its column label (mb / %).
    talys = df[df["Database"] == "TALYS 14 MeV"].iloc[0]
    assert talys["Strength type"] == "XS (mb)" and talys["Line strength"] > 0


def test_export_table_includes_decay_daughter():
    # A ¹³⁷Cs source line: the parent decays to ¹³⁷Ba, which emits the 661.7 keV
    # gamma — both the daughter and the decay mode come through.
    from wara.gui import isotope_id
    df = isotope_id.best_per_database_table([661.7], tol=3.0)
    cs = df[(df["Database"] == "Common lab sources")
            & (df["Isotope (parent)"] == "137Cs")].iloc[0]
    assert cs["Daughter"] == "137Ba"
    assert cs["Decay"] == "B-"
    assert cs["Strength type"] == "Intensity (%)"


def test_export_writes_csv(app, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog
    _find_peaks(app)
    out = tmp_path / "iso.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "CSV (*.csv)")))
    app._export_isotope_table()
    assert out.exists()
    import pandas as pd
    saved = pd.read_csv(out)
    assert "Isotope (parent)" in saved.columns and len(saved) > 0
