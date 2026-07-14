"""Offscreen tests for the Spectrum tab cursor readout (moved below the plot)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication

from wara import spectrum as sp
from wara.gui.app import WaraApp


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def app(qapp):
    w = WaraApp()
    spect = sp.Spectrum(counts=np.linspace(1, 100, 256))
    w.spect = spect
    w._spect_orig = spect.copy()
    w._active_name = "synthetic"
    w._refresh()
    yield w
    w.close()


def test_cursor_menu_removed(app):
    """The old CURSOR stat rows are gone from the options panel."""
    opts = app.spectrum_opts
    assert not hasattr(opts, "val_cur_x")
    assert not hasattr(opts, "val_cur_y")
    assert not hasattr(opts, "key_cur_x")


def test_customize_preserves_zoom(app):
    """Engaging a Customize option must not reset the pan/zoom (like Show peaks)."""
    ax = app.spectrum_page.canvas.ax
    ax.set_xlim(50, 120); ax.set_ylim(0, 40)
    app.spectrum_opts.customize_panel.cb_smooth.setChecked(True)   # toggled → live _recompute
    assert tuple(round(v) for v in ax.get_xlim()) == (50, 120)
    assert tuple(round(v) for v in ax.get_ylim()) == (0, 40)


def test_plain_refresh_resets_zoom(app):
    """A normal refresh (new data) still autoscales to the full view."""
    ax = app.spectrum_page.canvas.ax
    ax.set_xlim(50, 120)
    app._refresh()
    assert ax.get_xlim()[0] < 50           # reset well beyond the zoomed window


def test_resize_keeps_active_spectrum(app):
    """Resizing the window must not blank the active spectrum.

    resizeEvent used to call draw_empty() unconditionally, which nulls
    canvas._spect; afterwards the visibility toggle (_redraw) could not restore
    it because _redraw short-circuits when _spect is None. Regression: the
    spectrum survives a resize and the axes still hold its line.
    """
    from PyQt5.QtCore import QSize
    from PyQt5.QtGui import QResizeEvent
    canvas = app.spectrum_page.canvas
    assert canvas._spect is not None
    canvas.resizeEvent(QResizeEvent(QSize(640, 480), QSize(800, 600)))
    assert canvas._spect is not None          # spectrum not destroyed
    assert canvas.ax.lines or canvas.ax.collections   # its artists are still drawn
    # And the visibility toggle still round-trips (it relies on _spect).
    canvas.set_active_visible(False)
    canvas.set_active_visible(True)
    assert canvas._spect is not None


def test_readout_label_exists_below_plot(app):
    assert hasattr(app.spectrum_page, "readout")


def test_on_cursor_updates_readout(app):
    app._on_cursor(123.45, 678.0)
    txt = app.spectrum_page.readout.text()
    assert "123.45" in txt
    assert "678" in txt
    assert "Counts" in txt


def test_fwhm_pct_at_662_helper():
    """The convenience factory reproduces the quoted resolution at 662 keV and
    scales as √E away from it."""
    fwhm = sp.Spectrum.fwhm_pct_at_662(3.0)          # 3% at 662 keV, in keV
    assert fwhm(662.0) == pytest.approx(0.03 * 662.0)
    # Scintillator scaling: FWHM(E) ∝ √E.
    assert fwhm(4 * 662.0) == pytest.approx(2 * fwhm(662.0))
    # MeV axis: pass e_ref in MeV.
    fwhm_mev = sp.Spectrum.fwhm_pct_at_662(3.0, e_ref=0.662)
    assert fwhm_mev(0.662) == pytest.approx(0.03 * 0.662)


def test_broaden_needs_calibration(app):
    """Ticking Broaden on an uncalibrated spectrum unticks itself and leaves the
    counts untouched (no energy axis to broaden along)."""
    cust = app.spectrum_opts.customize_panel
    before = app.spect.counts.copy()
    cust.cb_broaden.setChecked(True)                  # toggled → live _recompute
    assert not cust.cb_broaden.isChecked()            # guard unticked it
    np.testing.assert_array_equal(app.spect.counts, before)


def test_broaden_applies_on_calibrated_spectrum(qapp):
    """On a calibrated spectrum, Broaden redistributes counts (total conserved
    up to Poisson resampling) without dropping the energy axis."""
    w = WaraApp()
    n = 256
    counts = np.zeros(n); counts[128] = 10000.0       # a single sharp line
    spect = sp.Spectrum(counts=counts, energies=np.linspace(1, 1500, n),
                        e_units="keV")
    w.spect = spect; w._spect_orig = spect.copy()
    w._active_name = "line"; w._refresh()
    cust = w.spectrum_opts.customize_panel
    cust.cb_broaden.setChecked(True)
    assert cust.cb_broaden.isChecked()
    # The delta spread into neighbouring channels: the peak channel dropped and
    # its neighbours gained.
    assert w.spect.counts[128] < 10000.0
    assert w.spect.counts[127] > 0 or w.spect.counts[129] > 0
    assert w.spect.energies is not None               # calibration preserved
    w.close()


def test_readout_uses_x_units(app):
    app._update_cursor_units()
    app._on_cursor(50.0, 10.0)
    # Uncalibrated spectrum → channel axis label.
    assert "ch" in app.spectrum_page.readout.text().lower()


def test_clear_empties_readout(app):
    app._on_cursor(10.0, 20.0)
    app._clear()
    assert app.spectrum_page.readout.text() == ""


def test_long_spectrum_name_is_elided_not_widening_panel(app):
    """A long file name is elided to a capped width so it stays visible but does
    not force the fixed-width options panel wider (which clipped the buttons on
    the right). The full name remains in the tooltip."""
    from PyQt5.QtWidgets import QPushButton
    long_name = "a_really_quite_long_spectrum_file_name_2024_run42_cal.txt"
    row = app._make_spectrum_row(
        long_name, "#00e5ff", True, on_toggle=lambda *_: None,
        on_remove=lambda *_: None, on_activate=lambda *_: None)
    name_btn = row.findChild(QPushButton, "spectrum_name")
    assert name_btn.maximumWidth() <= 160          # width-capped to fit the panel
    assert name_btn.text()                          # still visible (not empty)
    assert name_btn.text() != long_name             # ...and shortened (elided)
    assert long_name in name_btn.toolTip()          # full name in the tooltip
