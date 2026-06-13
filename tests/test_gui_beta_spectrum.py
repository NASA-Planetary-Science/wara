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
from wara.gui_beta.app import WaraBetaApp


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def app(qapp):
    w = WaraBetaApp()
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
    app.spectrum_opts.cb_smooth.setChecked(True)   # toggled → live _recompute
    assert tuple(round(v) for v in ax.get_xlim()) == (50, 120)
    assert tuple(round(v) for v in ax.get_ylim()) == (0, 40)


def test_plain_refresh_resets_zoom(app):
    """A normal refresh (new data) still autoscales to the full view."""
    ax = app.spectrum_page.canvas.ax
    ax.set_xlim(50, 120)
    app._refresh()
    assert ax.get_xlim()[0] < 50           # reset well beyond the zoomed window


def test_readout_label_exists_below_plot(app):
    assert hasattr(app.spectrum_page, "readout")


def test_on_cursor_updates_readout(app):
    app._on_cursor(123.45, 678.0)
    txt = app.spectrum_page.readout.text()
    assert "123.45" in txt
    assert "678" in txt
    assert "Counts" in txt


def test_readout_uses_x_units(app):
    app._update_cursor_units()
    app._on_cursor(50.0, 10.0)
    # Uncalibrated spectrum → channel axis label.
    assert "ch" in app.spectrum_page.readout.text().lower()


def test_clear_empties_readout(app):
    app._on_cursor(10.0, 20.0)
    app._clear()
    assert app.spectrum_page.readout.text() == ""
