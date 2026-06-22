"""Time-slice fitting for the API tab.

Replaces the old "significance vs dt" overlay with a depth-profiling workflow
ported from ``examples/other/anorthosite-olivine-ilmenite-cal.py``:

* The current dt range is split into slices of a user-defined width.  For each
  slice the events are histogrammed into an energy spectrum (one
  :class:`wara.spectrum.Spectrum` + :class:`wara.peaksearch.PeakSearch` per
  slice), all sharing the tab's energy binning and range.
* :func:`plot_slice_spectra` overlays every per-slice spectrum (coloured by
  time, plasma map) — the example's "Energy spectra per time slice" figure —
  with the active selection's energy band shaded.  It is drawn into the canvas
  embedded in the Energy Selections dialog (with a navigation toolbar).
* The *Fit* technique opens :class:`SliceFitWindow` (a
  :class:`~wara.gui_beta.fitting.FitWindow` subclass) where the user steps
  through the slices and adjusts the fit per slice, reusing the Spectrum tab's
  fitting machinery.  Its **Method** selector chooses, per slice, between a peak
  fit (sum the checked peaks via the in-table checkboxes) and a net area above a
  linear background — so that choice is made here rather than up front.
  ``Plot vs dt`` harvests the per-slice value and emits it back to the
  controller, which overlays the value-vs-dt curve on the dt panel.  *SNR*
  needs no fitting, so the controller computes it directly and skips this
  window.

The fit window keeps each slice's region independently (per-slice windows): the
selection band seeds the ROI, and the FitWindow inner sliders carve the peak /
background out of it exactly as on the Spectrum tab.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from PyQt5.QtWidgets import (
    QHBoxLayout, QWidget, QLabel, QPushButton, QCheckBox,
    QTableWidgetItem, QHeaderView,
)
from PyQt5.QtCore import Qt, pyqtSignal

from wara.advanced_fit import MultiProfilePeakFit, shape_summary
from .fitting import FitWindow, FIT_PLOT_BG
from . import theme as T


# Technique identifiers shared with the controller / Selections dialog.
# "Fit" opens the interactive window where the per-slice Method selector picks
# Peak fit vs Net area − linear bkg; "SNR" needs no fitting.
TECH_FIT = "fit"
TECH_SNR = "snr"

TECHNIQUE_LABELS = {
    TECH_FIT: "Fit",
    TECH_SNR: "SNR",
}
# Reverse map for reading the Selections-dialog combo back to an identifier.
TECHNIQUE_FROM_LABEL = {v: k for k, v in TECHNIQUE_LABELS.items()}

YLABELS = {
    TECH_FIT: "Area (counts)",
    TECH_SNR: "Peak SNR",
}

# Cap the number of slices so an over-fine dt width can't spawn hundreds of
# PeakSearch builds and freeze the UI.
MAX_SLICES = 40


def band_snr(search, band):
    """Peak SNR for an energy band: max of ``search.snr`` inside the band.

    ``search.snr`` is the per-energy-bin SNR array (method 'km'), aligned to the
    spectrum's energy axis.  Returns NaN if the band falls outside the axis.
    """
    snr = getattr(search, "snr", None)
    if snr is None:
        return float("nan")
    e = search.spectrum.energies
    if e is None:
        e = search.spectrum.channels
    inside = (e >= band[0]) & (e <= band[1])
    if not np.any(inside):
        return float("nan")
    return float(np.nanmax(snr[inside]))


def ratio_to_ref(num, num_err, den, den_err):
    """Per-dt ratio num/den with uncorrelated error propagation.

    Returns ``(ratio, ratio_err)`` arrays.  Bins where the reference is
    non-positive or the ratio isn't finite are NaN so they drop out of the plot.
    """
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    num_err = np.asarray(num_err, float)
    den_err = np.asarray(den_err, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
        rerr = np.abs(r) * np.sqrt((num_err / num) ** 2 + (den_err / den) ** 2)
    bad = (den <= 0) | ~np.isfinite(r)
    r = np.where(bad, np.nan, r)
    rerr = np.where(np.isfinite(rerr) & ~bad, rerr, np.nan)
    return r, rerr


def slice_metric(search, band):
    """Headless (value, err) for one slice, matching the fit window's default.

    Used to pre-populate every slice so the vs-dt overlay is available before
    the user has stepped through each one.  The window opens in Peak-fit mode,
    so the default is the fitted Gaussian peak nearest the band centre.
    """
    lo, hi = float(band[0]), float(band[1])
    if hi <= lo:
        return float("nan"), float("nan")
    try:
        fit = MultiProfilePeakFit(search, [lo, hi], bkg="poly1", profile="gauss")
        return _nearest_peak_area(fit, 0.5 * (lo + hi))
    except Exception:  # noqa: BLE001 — a bad slice just yields no point
        return float("nan"), float("nan")


def _nearest_peak_area(fit, center):
    """(area, area_err) of the fitted peak whose centroid is nearest *center*."""
    if fit is None or not getattr(fit, "peak_info", None):
        return float("nan"), float("nan")
    df = shape_summary(fit)
    if df is None or len(df) == 0:
        return float("nan"), float("nan")
    k = (df["mean"] - center).abs().idxmin()
    return float(df.loc[k, "area"]), float(df.loc[k, "area_err"])


def plot_slice_spectra(fig, slices, band, x_label):
    """Draw the per-slice energy spectra (plasma map) onto *fig*.

    Shows the full spectrum for every slice (so the user can pan/zoom with the
    embedded toolbar); the selection's energy band is shaded for reference.
    """
    fig.clear()
    fig.set_facecolor(FIT_PLOT_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(FIT_PLOT_BG)
    cmap = plt.get_cmap("plasma")
    n = len(slices)
    colors = cmap(np.linspace(0, 1, max(1, n)))
    for j, s in enumerate(slices):
        x = s["spe"].energies
        ax.plot(x, s["spe"].counts, color=colors[j], lw=1.0, alpha=0.75,
                label=f"t=[{s['t0']:.1f}, {s['t1']:.1f}] ns")
    if band is not None:
        ax.axvspan(band[0], band[1], color=T.ACCENT_AMBER, alpha=0.12)
    ax.set_yscale("log")
    ax.set_xlabel(x_label, color=T.TEXT_PRIMARY, fontsize=12)
    ax.set_ylabel("Counts", color=T.TEXT_PRIMARY, fontsize=12)
    ax.tick_params(colors=T.TEXT_DIM, which="both", length=3, labelsize=11)
    for sp in ax.spines.values():
        sp.set_color(T.BORDER)
    ax.grid(True, color="#3c3c66", linewidth=0.6, alpha=0.7)
    if n <= 16:
        ax.legend(fontsize=11, ncol=2, facecolor=FIT_PLOT_BG,
                  edgecolor=T.BORDER, labelcolor=T.TEXT_PRIMARY)
    fig.tight_layout()


def plot_slice_offset(fig, slices, band, x_label):
    """Draw the per-slice spectra as an offset (ridgeline) plot onto *fig*.

    Each slice's spectrum is offset vertically by a constant step and filled to
    its baseline, earliest dt at the bottom; a colour bar maps colour → dt.
    Linear y (offsets don't read on a log axis); the band is shaded for
    reference.
    """
    fig.clear()
    fig.set_facecolor(FIT_PLOT_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(FIT_PLOT_BG)
    cmap = plt.get_cmap("plasma")
    n = len(slices)
    colors = cmap(np.linspace(0, 1, max(1, n)))
    # Offset step: a fraction of the tallest spectrum so traces cascade without
    # fully overlapping.
    peak = max((float(s["spe"].counts.max()) for s in slices), default=1.0) or 1.0
    step = 0.35 * peak
    for j, s in enumerate(slices):
        x = s["spe"].energies
        base = j * step
        y = s["spe"].counts + base
        ax.fill_between(x, base, y, color=colors[j], alpha=0.12, zorder=j)
        ax.plot(x, y, color=colors[j], lw=1.0, alpha=0.95, zorder=j)
    if band is not None:
        ax.axvspan(band[0], band[1], color=T.ACCENT_AMBER, alpha=0.10)
    ax.set_xlabel(x_label, color=T.TEXT_PRIMARY, fontsize=12)
    ax.set_ylabel("Counts (offset per dt slice)", color=T.TEXT_PRIMARY, fontsize=12)
    ax.tick_params(colors=T.TEXT_DIM, which="both", length=3, labelsize=11)
    for sp in ax.spines.values():
        sp.set_color(T.BORDER)
    ax.grid(True, axis="x", color="#3c3c66", linewidth=0.6, alpha=0.7)
    # Colour bar mapping the plasma colour to dt.
    t_lo, t_hi = slices[0]["t0"], slices[-1]["t1"]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(t_lo, t_hi))
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("dt (ns)", color=T.TEXT_PRIMARY, fontsize=12)
    cb.ax.tick_params(colors=T.TEXT_DIM, labelsize=11)
    cb.outline.set_edgecolor(T.BORDER)
    fig.tight_layout()


def plot_slice_waterfall(fig, slices, band, x_label):
    """Draw the per-slice spectra as a 2-D waterfall heatmap onto *fig*.

    Energy on x, dt on y, colour = counts (log scale).  Each row is one dt
    slice; the band is marked with vertical guide lines.
    """
    fig.clear()
    fig.set_facecolor(FIT_PLOT_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(FIT_PLOT_BG)

    energies = slices[0]["spe"].energies
    # Cell edges: energy midpoints, and the dt slice boundaries.
    de = np.diff(energies)
    x_edges = np.concatenate((
        [energies[0] - de[0] / 2],
        energies[:-1] + de / 2,
        [energies[-1] + de[-1] / 2],
    ))
    y_edges = np.array([s["t0"] for s in slices] + [slices[-1]["t1"]], dtype=float)
    z = np.array([s["spe"].counts for s in slices], dtype=float)   # [n_dt, n_e]
    zmasked = np.ma.masked_less_equal(z, 0)                        # log-safe
    zmax = float(z.max()) if z.size and z.max() > 0 else 1.0
    norm = LogNorm(vmin=1.0, vmax=max(zmax, 2.0))

    mesh = ax.pcolormesh(x_edges, y_edges, zmasked, cmap="plasma", norm=norm,
                         shading="flat")
    if band is not None:
        ax.axvline(band[0], color=T.ACCENT_AMBER, lw=1.2, alpha=0.8)
        ax.axvline(band[1], color=T.ACCENT_AMBER, lw=1.2, alpha=0.8)
    ax.set_xlabel(x_label, color=T.TEXT_PRIMARY, fontsize=12)
    ax.set_ylabel("dt (ns)", color=T.TEXT_PRIMARY, fontsize=12)
    ax.tick_params(colors=T.TEXT_DIM, which="both", length=3, labelsize=11)
    for sp in ax.spines.values():
        sp.set_color(T.BORDER)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("Counts", color=T.TEXT_PRIMARY, fontsize=12)
    cb.ax.tick_params(colors=T.TEXT_DIM, labelsize=11)
    cb.outline.set_edgecolor(T.BORDER)
    fig.tight_layout()


class SliceFitWindow(FitWindow):
    """FitWindow that steps through dt slices, harvesting per-slice peak area.

    The selection band is the fixed outer ROI for every slice; the inner
    sliders carve the peak / background and are remembered per slice.  ``Plot
    vs dt`` emits the harvested (dt_centre, value, error) series.
    """

    results_ready = pyqtSignal(object)   # payload dict (see _emit_results)

    def __init__(self, parent, slices, band, sel_label, sel_color):
        super().__init__(parent, slices[0]["search"])
        self.setWindowTitle(f"Time-slice fits — {sel_label}")
        self.slices = slices
        self.band = (float(band[0]), float(band[1]))
        self.technique = TECH_FIT
        self.sel_label = sel_label
        self.sel_color = sel_color
        self._cur = 0
        self._inner_by_slice = {}     # idx -> (lo, hi) inner peak edges
        self._metric_by_slice = {}    # idx -> (value, err)
        # Peak selection (Peak-fit method only): which fitted peaks to sum into
        # the slice area.  Peaks are dynamic across dt, so the chosen set is kept
        # per slice, by row index.  None ⇒ not yet visited (default to the peak
        # nearest the band centre on first fit).
        self._selected_by_slice = {}  # idx -> set(peak row indices)
        self._peak_rows = []          # current slice: [(centroid, area, err), …]
        self._building = True
        self._loaded = False          # a slice has been loaded into the controls

        # The send-to-Calibration/Efficiency/Resolution buttons aren't part of
        # this depth-profiling flow — hide them to keep the bar focused.
        for b in (self.btn_to_cal, self.btn_to_eff, self.btn_to_res):
            b.hide()

        # Start in Peak-fit mode with a Gaussian; the Method selector stays
        # enabled so the user can switch to Net area − linear bkg per slice.
        self.method.setCurrentIndex(0)
        self.shape.setCurrentText("Gaussian")

        # Slice navigation row, inserted just under the method selector.
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_prev.setObjectName("action_btn")
        self.btn_prev.setCursor(Qt.PointingHandCursor)
        self.btn_next = QPushButton("Next ▶")
        self.btn_next.setObjectName("action_btn")
        self.btn_next.setCursor(Qt.PointingHandCursor)
        self.lbl_slice = QLabel("")
        self.lbl_slice.setObjectName("stat_key")
        self.btn_plot = QPushButton("Plot vs dt")
        self.btn_plot.setObjectName("primary_btn")
        self.btn_plot.setCursor(Qt.PointingHandCursor)
        self.btn_plot.setToolTip(
            "Overlay this selection's per-slice value vs dt on the dt panel")
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(+1))
        self.btn_plot.clicked.connect(self._emit_results)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        nav.addWidget(self.lbl_slice, 1)
        # Live value readout for the current slice (summed checked peaks, or the
        # net area in linear-background mode).
        self.lbl_peak_sum = QLabel(""); self.lbl_peak_sum.setObjectName("stat_key")
        nav.addWidget(self.lbl_peak_sum)
        nav.addWidget(self.btn_plot)
        self.layout().insertLayout(1, nav)

        # Pre-fit every slice headlessly so the overlay is ready immediately.
        for i, s in enumerate(self.slices):
            self._inner_by_slice[i] = self.band
            self._metric_by_slice[i] = slice_metric(s["search"], self.band)
        self._building = False
        self._goto(0)

    # ── slice navigation ─────────────────────────────────────────────────────
    def _step(self, delta):
        self._goto(self._cur + delta)

    def _goto(self, i):
        i = max(0, min(len(self.slices) - 1, i))
        self._save_current()
        self._cur = i
        s = self.slices[i]
        self.set_search(s["search"])
        # Outer ROI is always the band; set_roi resets the inner edges to full
        # and triggers a refit, then we restore this slice's saved inner edges.
        self._building = True
        self.set_roi(self.band[0], self.band[1])
        inner = self._inner_by_slice.get(i, self.band)
        self._set_roi_controls(inner[0], inner[1])
        self._building = False
        self._loaded = True
        self._refit()
        self.lbl_slice.setText(
            f"slice {i + 1}/{len(self.slices)}  ·  "
            f"t = [{s['t0']:.1f}, {s['t1']:.1f}] ns")
        self.btn_prev.setEnabled(i > 0)
        self.btn_next.setEnabled(i < len(self.slices) - 1)

    def _save_current(self):
        """Store the inner edges of the slice we're leaving (metric is captured
        on refit).  Skipped before the first slice is loaded, otherwise the
        stale default spinner values would clobber the seeded band edges."""
        if not self.slices or not self._loaded:
            return
        self._inner_by_slice[self._cur] = (self.roi_lo.value(), self.roi_hi.value())

    # ── peak selection (Peak-fit method) ──────────────────────────────────────
    def _set_columns(self, headers):
        """Drop a checkbox column added on a previous fill before the base
        re-lays the table.  ``setColumnCount`` alone leaves the column-0 cell
        widgets behind, which would otherwise shift into the data columns on
        the next ``insertColumn``.  This is the single choke point every
        table-filling path (peak fit, bkg-only, net area) routes through."""
        h0 = self.table.horizontalHeaderItem(0)
        if h0 is not None and h0.text() == "Use":
            self.table.removeColumn(0)
        super()._set_columns(headers)

    def _fill_table(self, fit):
        """Capture this slice's peaks, let the base build the standard results
        table, then prepend a checkbox column so the user can pick which peaks
        feed the slice area.  Only the Peak-fit method routes through here
        (the net-area method uses its own table), so the column is always added."""
        self._capture_peak_rows(fit)
        super()._fill_table(fit)
        self._add_select_column()

    def _capture_peak_rows(self, fit):
        """Store [(centroid, area, err), …] for the current fit and, on a
        slice's first visit, default the selection to the peak nearest the band
        centre (preserving the original single-peak behaviour)."""
        rows = []
        if fit is not None and getattr(fit, "peak_info", None):
            try:
                df = shape_summary(fit)
                for _, r in df.iterrows():
                    err = r["area_err"]
                    rows.append((float(r["mean"]), float(r["area"]),
                                 float(err) if err is not None else float("nan")))
            except Exception:  # noqa: BLE001
                rows = []
        self._peak_rows = rows
        if self._selected_by_slice.get(self._cur) is None:
            if rows:
                center = 0.5 * (self.band[0] + self.band[1])
                k = int(np.argmin([abs(c - center) for c, _, _ in rows]))
                self._selected_by_slice[self._cur] = {k}
            else:
                self._selected_by_slice[self._cur] = set()

    def _add_select_column(self):
        """Insert a leading 'Use' column of checkboxes (one per peak row)."""
        table = self.table
        table.insertColumn(0)
        table.setHorizontalHeaderItem(0, QTableWidgetItem("Use"))
        sel = self._selected_by_slice.get(self._cur, set())
        n = min(table.rowCount(), len(self._peak_rows))
        for i in range(n):
            cb = QCheckBox()
            cb.setChecked(i in sel)
            cb.setCursor(Qt.PointingHandCursor)
            cb.setToolTip("Include this peak's area in the slice value")
            cb.toggled.connect(lambda checked, idx=i: self._on_peak_toggle(idx, checked))
            # Wrap so the checkbox sits centred in the cell.
            wrap = QWidget(); wl = QHBoxLayout(wrap)
            wl.setContentsMargins(0, 0, 0, 0); wl.setAlignment(Qt.AlignCenter)
            wl.addWidget(cb)
            table.setCellWidget(i, 0, wrap)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        table.setColumnWidth(0, 44)

    def _on_peak_toggle(self, idx, checked):
        sel = self._selected_by_slice.setdefault(self._cur, set())
        if checked:
            sel.add(idx)
        else:
            sel.discard(idx)
        self._metric_by_slice[self._cur] = self._current_metric()
        self._update_peak_sum()

    def _update_peak_sum(self):
        v, e = self._current_metric()
        kind = "Net area" if self._is_area_method() else "Σ peak area"
        self.lbl_peak_sum.setText(f"{kind} = {v:,.1f} ± {e:,.1f}")

    # ── metric capture ───────────────────────────────────────────────────────
    def _refit(self):
        super()._refit()
        if self._building:
            return
        # In Peak-fit mode, when the fit found nothing the base routes to a
        # background-only view (no _fill_table), so clear the stale peak list.
        if not self._is_area_method() and not (
                self.last_fit is not None
                and getattr(self.last_fit, "peak_info", None)):
            self._peak_rows = []
            self._selected_by_slice.setdefault(self._cur, set())
        self._metric_by_slice[self._cur] = self._current_metric()
        self._update_peak_sum()

    def _current_metric(self):
        if self._is_area_method():
            alb = getattr(self, "_last_area", None)
            if alb is not None:
                return float(alb.A), float(alb.sigA)
            return float("nan"), float("nan")
        # Peak-fit: sum the areas of the checked peaks (none checked ⇒ 0).
        sel = self._selected_by_slice.get(self._cur, set())
        valid = [i for i in sel if 0 <= i < len(self._peak_rows)]
        if not valid:
            return 0.0, 0.0
        area = sum(self._peak_rows[i][1] for i in valid)
        err2 = sum(self._peak_rows[i][2] ** 2 for i in valid
                   if np.isfinite(self._peak_rows[i][2]))
        return float(area), float(np.sqrt(err2))

    # ── emit ─────────────────────────────────────────────────────────────────
    def _emit_results(self):
        self._save_current()
        self._metric_by_slice[self._cur] = self._current_metric()
        centers, vals, errs = [], [], []
        for i, s in enumerate(self.slices):
            v, e = self._metric_by_slice.get(i, (float("nan"), float("nan")))
            centers.append(s["tc"]); vals.append(v); errs.append(e)
        self.results_ready.emit(dict(
            label=self.sel_label, color=self.sel_color, technique=self.technique,
            dt_centers=centers, vals=vals, errs=errs,
            ylabel=YLABELS.get(self.technique, "Value")))
        self.lbl_status.setText(
            f"Plotted {self.sel_label} ({TECHNIQUE_LABELS[self.technique]}) vs dt.")
