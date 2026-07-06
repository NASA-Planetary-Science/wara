"""Offscreen GUI regression tests for the wara Efficiency tab
(wara.gui.efficiency: EfficiencyOptions / EfficiencyPage /
EfficiencyController + the unit parser).

Covers the table-driven workflow: receiving a peak fit (one editable row per
peak), entering unit-aware source parameters, the per-row nuclide lookup, live
efficiency computation, fitting (order 1 / order 2), and remove / reset.
"""

import os

# Must be set before the first QApplication is created (during collection).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDialog

from wara import spectrum as sp
from wara import peaksearch as ps
from wara import peakfit as pf
from wara.gui.app import WaraApp
from wara.gui import efficiency as E


SLOPE, INTERCEPT = 0.5, 10.0
PEAK_CHANNELS = (600, 1200, 2000, 2800)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _synthetic_spectrum():
    ch = np.arange(4096)
    counts = np.full_like(ch, 5.0, dtype=float)
    for c in PEAK_CHANNELS:
        counts += 1500.0 * np.exp(-0.5 * ((ch - c) / 4.0) ** 2)
    erg = SLOPE * ch + INTERCEPT
    return sp.Spectrum(counts=counts, energies=erg, e_units="keV", livetime=600)


@pytest.fixture
def app(qapp):
    w = WaraApp()
    spect = _synthetic_spectrum()
    w.spect = spect
    w._spect_orig = spect.copy()
    w._active_name = "synthetic"
    w._refresh()
    w.search = ps.PeakSearch(spect, ref_x=420, ref_fwhm=3, fwhm_at_0=1.0, min_snr=5)
    yield w
    w.close()


def _fit(w, ch_lo, ch_hi, bkg="linear"):
    x = w.spect.x
    return pf.PeakFit(w.search, xrange=[x[ch_lo], x[ch_hi]], bkg=bkg)


def _setc(w, row, col, text):
    """Type *text* into a cell (mirrors a user edit → fires cellChanged)."""
    w.efficiency.tbl.item(row, col).setText(text)


def _fill_row(w, row, thalf="30 y", a0="5 µCi", br="0.85", lt="600 s", te="16 y"):
    _setc(w, row, E.COL_THALF, thalf)
    _setc(w, row, E.COL_A0, a0)
    _setc(w, row, E.COL_BR, br)
    _setc(w, row, E.COL_LT, lt)
    _setc(w, row, E.COL_TE, te)


def _eff_text(w, row):
    return w.efficiency.tbl.item(row, E.COL_EFF).text()


def _add_filled_points(w, n):
    """Accumulate *n* completed efficiency rows (one single-peak fit each)."""
    spans = [(550, 650), (1150, 1250), (1950, 2050), (2750, 2850)]
    for i in range(n):
        lo, hi = spans[i]
        w.efficiency.receive_fit(_fit(w, lo, hi))
        _fill_row(w, w.efficiency.tbl.rowCount() - 1)


# ── unit parser ───────────────────────────────────────────────────────────────
def test_parse_time_units():
    assert E.parse_quantity("30 y", E.CAT_TIME)[0] == pytest.approx(30 * 365 * 86400)
    assert E.parse_quantity("600 s", E.CAT_TIME)[0] == pytest.approx(600)
    assert E.parse_quantity("30 min", E.CAT_TIME)[0] == pytest.approx(1800)
    assert E.parse_quantity("2 h", E.CAT_TIME)[0] == pytest.approx(7200)
    # A bare number is already in the standard unit (seconds); zero uncertainty.
    assert E.parse_quantity("42", E.CAT_TIME) == (42.0, 0.0, "s")


def test_parse_activity_units():
    assert E.parse_quantity("1 Bq", E.CAT_ACTIVITY)[0] == pytest.approx(1)
    assert E.parse_quantity("5 µCi", E.CAT_ACTIVITY)[0] == pytest.approx(5 * 3.7e4)
    assert E.parse_quantity("5 uCi", E.CAT_ACTIVITY)[0] == pytest.approx(5 * 3.7e4)
    assert E.parse_quantity("1e-6 Ci", E.CAT_ACTIVITY)[0] == pytest.approx(3.7e4)


def test_parse_ratio_and_blank():
    assert E.parse_quantity("0.85", E.CAT_RATIO) == (0.85, 0.0, "")
    assert E.parse_quantity("85%", E.CAT_RATIO)[0] == pytest.approx(0.85)
    assert E.parse_quantity("   ", E.CAT_TIME) is None


def test_parse_inline_uncertainty():
    # Bare uncertainty inherits the value's unit.
    val, unc, label = E.parse_quantity("30 y (2)", E.CAT_TIME)
    assert val == pytest.approx(30 * 365 * 86400)
    assert unc == pytest.approx(2 * 365 * 86400)
    assert label == "s"
    # Uncertainty may carry its own unit.
    val, unc, _ = E.parse_quantity("1 h (30 min)", E.CAT_TIME)
    assert val == pytest.approx(3600) and unc == pytest.approx(1800)
    # Activity + percent ratio uncertainties.
    assert E.parse_quantity("5 µCi (0.1)", E.CAT_ACTIVITY)[1] == pytest.approx(0.1 * 3.7e4)
    assert E.parse_quantity("85% (1%)", E.CAT_RATIO)[1] == pytest.approx(0.01)


def test_parse_elapsed_date_range():
    import datetime
    secs, unc, label = E.parse_elapsed("2006-06-01 to 2022-08-19")
    expected = (datetime.datetime(2022, 8, 19)
                - datetime.datetime(2006, 6, 1)).days * 24 * 3600
    assert secs == pytest.approx(expected)
    assert unc == 0.0 and label == "s"
    # Falls back to a unit-bearing duration when it is not a date range.
    assert E.parse_elapsed("16 y")[0] == pytest.approx(16 * 365 * 86400)


def test_parse_elapsed_reversed_dates_raise():
    with pytest.raises(ValueError):
        E.parse_elapsed("2022-08-19 to 2006-06-01")


def test_parse_unknown_unit_raises():
    with pytest.raises(ValueError):
        E.parse_quantity("30 banana", E.CAT_TIME)
    with pytest.raises(ValueError):
        E.parse_quantity("5 furlongs", E.CAT_ACTIVITY)
    with pytest.raises(ValueError):
        E.parse_quantity("30 y (2 banana)", E.CAT_TIME)


# ── receive_fit ───────────────────────────────────────────────────────────────
def test_receive_fit_adds_one_row_per_peak_without_switching_tab(app):
    fit = _fit(app, 1150, 2050, bkg="poly2")
    assert len(fit.peak_info) >= 2
    # Start on the Spectrum tab; receiving a fit must NOT steal focus to it.
    app.nav_group.button(0).setChecked(True)
    app._on_nav(0)
    app.efficiency.receive_fit(fit)
    assert app.efficiency.tbl.rowCount() == len(fit.peak_info)
    assert len(app.efficiency._records) == len(fit.peak_info)
    assert app.stack.currentWidget() is app.spectrum_page
    # Energy + Net counts are seeded from the fit and BOTH editable now.
    e_item = app.efficiency.tbl.item(0, E.COL_ENERGY)
    c_item = app.efficiency.tbl.item(0, E.COL_COUNTS)
    assert e_item.flags() & Qt.ItemIsEditable
    assert c_item.flags() & Qt.ItemIsEditable
    assert e_item.data(E.VALUE_ROLE) == pytest.approx(fit.peak_info[0]["mean"], abs=2)
    assert c_item.data(E.VALUE_ROLE) == pytest.approx(fit.peak_info[0]["area"], rel=1e-6)
    assert c_item.data(E.UNC_ROLE) == pytest.approx(fit.peak_err[0]["area_err"], rel=1e-6)


def test_receive_fit_with_no_peaks_adds_nothing(app):
    class _Empty:
        peak_info = []
    app.efficiency.receive_fit(_Empty())
    assert app.efficiency.tbl.rowCount() == 0


# ── unit-aware cell entry + live compute ──────────────────────────────────────
def test_cell_converts_units_and_keeps_original_in_tooltip(app):
    app.efficiency.receive_fit(_fit(app, 1150, 1250))
    _setc(app, 0, E.COL_THALF, "30 y")
    item = app.efficiency.tbl.item(0, E.COL_THALF)
    assert item.data(E.VALUE_ROLE) == pytest.approx(30 * 365 * 86400)
    assert item.text().endswith("s")
    assert "30 y" in item.toolTip()


def test_complete_row_computes_efficiency(app):
    app.efficiency.receive_fit(_fit(app, 1150, 1250))
    _fill_row(app, 0)
    rec = app.efficiency._records[0]
    assert rec["point"] is not None
    energy, eff_pct, err_pct = rec["point"]
    assert eff_pct > 0
    # The single Efficiency cell shows "value (± unc)".
    assert "(" in _eff_text(app, 0)


def test_inline_uncertainty_stored_and_used(app):
    app.efficiency.receive_fit(_fit(app, 1150, 1250))
    _fill_row(app, 0, thalf="30 y (2)")
    item = app.efficiency.tbl.item(0, E.COL_THALF)
    assert item.data(E.VALUE_ROLE) == pytest.approx(30 * 365 * 86400)
    assert item.data(E.UNC_ROLE) == pytest.approx(2 * 365 * 86400)
    assert app.efficiency._records[0]["point"] is not None


def test_editing_net_counts_scales_efficiency(app):
    app.efficiency.receive_fit(_fit(app, 1150, 1250))
    _fill_row(app, 0)
    base = app.efficiency._records[0]["point"][1]
    # Net counts drives N_detected → doubling it doubles the efficiency.
    area = app.efficiency.tbl.item(0, E.COL_COUNTS).data(E.VALUE_ROLE)
    _setc(app, 0, E.COL_COUNTS, f"{2 * area:.12g}")
    assert app.efficiency._records[0]["point"][1] == pytest.approx(2 * base, rel=1e-5)


def test_editing_energy_moves_point(app):
    app.efficiency.receive_fit(_fit(app, 1150, 1250))
    _fill_row(app, 0)
    _setc(app, 0, E.COL_ENERGY, "999")
    assert app.efficiency._records[0]["point"][0] == pytest.approx(999.0)


def test_activity_at_count_time_column(app):
    app.efficiency.receive_fit(_fit(app, 1150, 1250))
    _fill_row(app, 0)
    rec = app.efficiency._records[0]
    # A_now = A0 · exp(-ln2 · t_elapsed / t_half) · Br
    A0 = 5 * 3.7e4
    t_half = 30 * 365 * 86400
    t_elapsed = 16 * 365 * 86400
    expected = A0 * np.exp(-np.log(2) * t_elapsed / t_half) * 0.85
    assert rec["a_now"] == pytest.approx(expected, rel=1e-6)
    anow_item = app.efficiency.tbl.item(0, E.COL_ANOW)
    assert anow_item.text() != "" and "Bq" in anow_item.text()
    assert not (anow_item.flags() & Qt.ItemIsEditable)       # locked


def test_elapsed_date_range_in_cell_computes(app):
    app.efficiency.receive_fit(_fit(app, 1150, 1250))
    _fill_row(app, 0, te="2006-06-01 to 2022-08-19")
    item = app.efficiency.tbl.item(0, E.COL_TE)
    assert item.data(E.VALUE_ROLE) == pytest.approx(
        E.parse_elapsed("2006-06-01 to 2022-08-19")[0])
    assert "2006-06-01" in item.toolTip()
    assert app.efficiency._records[0]["point"] is not None


def test_incomplete_row_has_no_point(app):
    app.efficiency.receive_fit(_fit(app, 1150, 1250))
    _setc(app, 0, E.COL_THALF, "30 y")     # only one value entered
    assert app.efficiency._records[0]["point"] is None
    assert app.efficiency.tbl.item(0, E.COL_EFF).text() == ""


def test_invalid_unit_flags_cell_and_blocks_compute(app):
    app.efficiency.receive_fit(_fit(app, 1150, 1250))
    _fill_row(app, 0)
    assert app.efficiency._records[0]["point"] is not None
    _setc(app, 0, E.COL_THALF, "30 zonks")
    item = app.efficiency.tbl.item(0, E.COL_THALF)
    assert item.data(E.VALUE_ROLE) is None
    assert item.toolTip().startswith("⚠")
    assert app.efficiency._records[0]["point"] is None


# ── nuclide lookup ────────────────────────────────────────────────────────────
class _StubField:
    def setText(self, *a):
        pass

    def setCurrentText(self, *a):
        pass


@pytest.fixture
def stub_picker(monkeypatch):
    """Replace NuclearLinePicker with a stub returning a canned record."""
    box = {"record": {"isotope": "60Co", "energy": 1332.5,
                      "intensity": 99.85, "half_life": 1.663e8},
           "result": QDialog.Accepted}

    class _Stub:
        def __init__(self, parent=None):
            self.db_combo = _StubField()
            self.ed_element = _StubField()
            self.ed_energy = _StubField()
            self.ed_range = _StubField()

        def _search(self):
            pass

        def exec_(self):
            return box["result"]

        def selected_record(self):
            return box["record"]

    monkeypatch.setattr("wara.gui.nuclear.NuclearLinePicker", _Stub)
    return box


def test_lookup_fills_half_life_and_branching(app, stub_picker):
    app.efficiency.receive_fit(_fit(app, 1150, 1250))
    rec = app.efficiency._records[0]
    app.efficiency._open_lookup(rec)
    tbl = app.efficiency.tbl
    assert tbl.item(0, E.COL_ISOTOPE).text() == "60Co"
    assert tbl.item(0, E.COL_THALF).data(E.VALUE_ROLE) == pytest.approx(1.663e8)
    assert tbl.item(0, E.COL_BR).data(E.VALUE_ROLE) == pytest.approx(0.9985)


def test_lookup_then_filling_rest_computes(app, stub_picker):
    app.efficiency.receive_fit(_fit(app, 1150, 1250))
    rec = app.efficiency._records[0]
    app.efficiency._open_lookup(rec)
    # Lookup filled t½ and Br; supply the remaining inputs.
    _setc(app, 0, E.COL_A0, "5 µCi")
    _setc(app, 0, E.COL_LT, "600 s")
    _setc(app, 0, E.COL_TE, "16 y")
    assert rec["point"] is not None


def test_lookup_cancelled_changes_nothing(app, stub_picker):
    stub_picker["result"] = QDialog.Rejected
    app.efficiency.receive_fit(_fit(app, 1150, 1250))
    rec = app.efficiency._records[0]
    app.efficiency._open_lookup(rec)
    assert app.efficiency.tbl.item(0, E.COL_ISOTOPE).text() == ""


# ── fit models ────────────────────────────────────────────────────────────────
def test_order1_fit_runs_with_two_points(app):
    _add_filled_points(app, 2)
    app.efficiency.opts.fit_model.setCurrentText(E.FIT_LINEAR)
    assert len(app.efficiency.page.fig.axes) == 2


def test_order2_fit_runs_with_four_points(app):
    _add_filled_points(app, 4)
    app.efficiency.opts.fit_model.setCurrentText(E.FIT_CURVE)
    assert len(app.efficiency.page.fig.axes) == 2


def test_order2_with_too_few_points_falls_back(app):
    _add_filled_points(app, 2)
    app.efficiency.opts.fit_model.setCurrentText(E.FIT_CURVE)
    assert len(app.efficiency.page.fig.axes) == 1


def test_logy_toggle_replots(app):
    _add_filled_points(app, 2)
    app.efficiency.opts.cb_logy.setChecked(False)
    assert app.efficiency.page.fig.axes[-1].get_yscale() == "linear"
    app.efficiency.opts.cb_logy.setChecked(True)
    assert app.efficiency.page.fig.axes[-1].get_yscale() == "log"


def test_logx_toggle_replots(app):
    _add_filled_points(app, 2)
    ax = app.efficiency.page.fig.axes[-1]
    assert ax.get_xscale() == "linear"
    app.efficiency.opts.cb_logx.setChecked(True)
    assert app.efficiency.page.fig.axes[-1].get_xscale() == "log"


def test_data_markers_recoloured_for_dark_mode(app):
    from wara.gui.efficiency import _is_dark
    _add_filled_points(app, 2)
    ax = app.efficiency.page.fig.axes[-1]
    markers = [ln for ln in ax.lines if ln.get_marker() not in ("", " ", "None", None)]
    assert markers, "expected at least the data-point markers"
    # No marker should be left black (invisible on the dark background).
    assert all(not _is_dark(ln.get_markerfacecolor()) for ln in markers)


# ── remove / reset ────────────────────────────────────────────────────────────
def test_trash_icon_removes_point_and_keeps_records_in_sync(app):
    _add_filled_points(app, 3)
    tbl = app.efficiency.tbl
    btn = tbl.cellWidget(1, E.COL_TRASH)        # 🗑 button on the second row
    assert btn is not None and btn.text() == "🗑"
    btn.click()
    assert tbl.rowCount() == 2
    assert len(app.efficiency._records) == 2


def test_reset_clears_everything(app):
    _add_filled_points(app, 2)
    app.efficiency._reset()
    assert app.efficiency.tbl.rowCount() == 0
    assert app.efficiency._records == []
    assert app.efficiency.opts.fit_model.currentText() == E.FIT_NONE
