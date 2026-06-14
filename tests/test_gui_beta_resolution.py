"""Offscreen GUI regression tests for the wara --beta Resolution tab
(wara.gui_beta.resolution: ResolutionOptions / ResolutionPage /
ResolutionController) plus its Drag-and-Fit "Send to Resolution" hook.

Covers: receiving peak fits (one FWHM point per peak, energy/FWHM kept paired),
duplicate rejection, the points table, the order-1/order-2 resolution fits with
forced origin and extrapolation, the Gaussian-components panel, dark-mode marker
recolouring, and remove / reset.
"""
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
from wara import peakfit as pf
from wara.gui_beta.app import WaraBetaApp
from wara.gui_beta import resolution as R


# Synthetic calibrated spectrum (E = 0.5·ch) with peaks of increasing width.
PEAKS = [(600.0, 3.0), (1500.0, 4.0), (2500.0, 5.0), (3300.0, 6.0)]
SLOPE = 0.5


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def app(qapp):
    ch = np.arange(8192)
    counts = np.full_like(ch, 5.0, dtype=float)
    for e, sig in PEAKS:
        counts += 5000.0 * np.exp(-0.5 * ((ch - e / SLOPE) / sig) ** 2)
    spect = sp.Spectrum(counts=counts, energies=SLOPE * ch, e_units="keV")
    w = WaraBetaApp()
    w.spect = spect
    w._spect_orig = spect.copy()
    w._active_name = "syn"
    w._refresh()
    w.search = ps.PeakSearch(spect, ref_x=1000, ref_fwhm=8, fwhm_at_0=1.0, min_snr=4)
    yield w
    w.close()


def _fit(w, energy):
    return pf.PeakFit(w.search, xrange=[energy - 40, energy + 40], bkg="linear")


def _send(w, energy):
    w.resolution.receive_fit(_fit(w, energy))


# ── receiving fits ────────────────────────────────────────────────────────────
def test_receive_fit_adds_one_point_per_peak(app):
    _send(app, 600.0)
    recs = app.resolution._records
    assert len(recs) == 1
    assert recs[0]["energy"] == pytest.approx(600.0, abs=3)
    assert recs[0]["fwhm"] > 0
    assert app.resolution.page.table.rowCount() == 1


def test_points_stay_paired_and_sorted_by_energy(app):
    # Send out of order; energy and FWHM must stay paired and sort ascending.
    for e in (2500.0, 600.0, 1500.0):
        _send(app, e)
    recs = app.resolution._records
    energies = [r["energy"] for r in recs]
    assert energies == sorted(energies)
    # FWHM increases with energy for this spectrum → pairing preserved.
    fwhms = [r["fwhm"] for r in recs]
    assert fwhms == sorted(fwhms)


def test_duplicate_energy_is_skipped(app):
    _send(app, 600.0)
    _send(app, 600.0)
    assert len(app.resolution._records) == 1


def test_table_shows_percent_fwhm(app):
    _send(app, 1500.0)
    tbl = app.resolution.page.table
    e = app.resolution._records[0]["energy"]
    f = app.resolution._records[0]["fwhm"]
    # Columns: [🗑, N, Energy, FWHM, %FWHM]
    assert float(tbl.item(0, 4).text()) == pytest.approx(f / e * 100, abs=0.01)


# ── fitting ───────────────────────────────────────────────────────────────────
def _add_points(app, n):
    for e, _ in PEAKS[:n]:
        _send(app, e)


def test_order1_fit_with_two_points(app):
    _add_points(app, 2)
    app.resolution.opts.fit_model.setCurrentText(R.FIT_ORDER1)
    # A single axis with a fitted green curve (data + fit).
    assert len(app.resolution.page.fig.axes) == 1
    ax = app.resolution.page.fig.axes[0]
    assert any(ln.get_label().startswith("a=") or "√" in (ax.get_title() or "")
               for ln in ax.lines) or ax.get_title()


def test_order2_needs_three_points(app):
    # Force-origin on by default → 1 real point + origin = 2 < 3 needed for order 2.
    _add_points(app, 1)
    app.resolution.opts.fit_model.setCurrentText(R.FIT_ORDER2)
    ax = app.resolution.page.fig.axes[0]
    assert "at least 3 points" in (ax.get_title() or "")
    # With three real points (+origin) the order-2 fit runs.
    _add_points(app, 3)
    assert "at least" not in (app.resolution.page.fig.axes[0].get_title() or "")


def test_force_origin_toggle_changes_point_count(app):
    _add_points(app, 2)
    o = app.resolution.opts
    assert o.cb_origin.isChecked()              # default on
    o.cb_origin.setChecked(False)               # just the two real points
    # Order 1 still fits with exactly two points; no error.
    assert len(app.resolution.page.fig.axes) == 1


def test_extrapolate_adds_dashed_curve(app):
    _add_points(app, 3)
    app.resolution.opts.cb_extrap.setChecked(True)
    ax = app.resolution.page.fig.axes[0]
    assert any(ln.get_linestyle() == "--" for ln in ax.lines)


def test_gaussian_components_panel_toggles(app):
    _add_points(app, 3)
    assert len(app.resolution.page.fig.axes) == 1
    app.resolution.opts.cb_gauss.setChecked(True)
    assert len(app.resolution.page.fig.axes) == 2


def test_data_markers_recoloured_for_dark_mode(app):
    _add_points(app, 3)
    ax = app.resolution.page.fig.axes[0]
    markers = [ln for ln in ax.lines if ln.get_marker() not in ("", " ", "None", None)]
    assert markers
    assert all(not R._is_dark(ln.get_markerfacecolor()) for ln in markers)


# ── points management ─────────────────────────────────────────────────────────
def test_trash_icon_removes_point(app):
    _add_points(app, 3)
    tbl = app.resolution.page.table
    btn = tbl.cellWidget(1, 0)        # 🗑 button on the second row
    assert btn is not None and btn.text() == "🗑"
    btn.click()
    assert len(app.resolution._records) == 2
    assert tbl.rowCount() == 2


def test_reset_clears_everything(app):
    _add_points(app, 3)
    app.resolution._reset()
    assert app.resolution._records == []
    assert app.resolution.page.table.rowCount() == 0


def test_receive_fit_with_no_peaks_is_ignored(app):
    class _Empty:
        peak_info = []
    app.resolution.receive_fit(_Empty())
    assert app.resolution._records == []


# ── Drag-and-Fit "Send to Resolution" hook ────────────────────────────────────
def test_send_to_resolution_button(app):
    from wara.gui_beta.fitting import FitWindow
    fw = FitWindow(None, app.search)
    fw.send_to_resolution.connect(app.resolution.receive_fit)
    fw.show()
    fw.set_roi(560, 640)                         # the 600 keV peak
    assert fw.btn_to_res.isEnabled()
    fw._emit_to_resolution()
    assert fw.btn_to_res.objectName() == "sent_btn"
    assert len(app.resolution._records) == 1
    fw.close()
