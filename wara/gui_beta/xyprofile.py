"""X-Y tile selections for the API tab.

A spatial selection workflow built on the dt slice-fit machinery in
:mod:`slicefit`.  The pipeline (three inner sub-tabs of the Selections dialog's
"X-Y selections" tab):

1. **X-Y map** — the current event hexbin is shown live.  *Build tiles* lays a
   uniform grid of user-sized rectangles over it; clicking a tile toggles it
   into the selected set (highlighted in the tile's colour).
2. **Tile spectra** — one energy spectrum per selected tile, overlaid in the
   tile colours.  The user drags energy *bands* on this overlay.
3. **Area vs position** — picking a band and *Fit tiles* opens the interactive
   :class:`TileFitWindow` (the same stepping fit window the Energy tab uses),
   stepping through each selected tile's spectrum to harvest a Gaussian +
   linear-background net peak area.  *Plot vs X / Y* sends one point per tile
   (area at the tile centre) to the area-vs-X and area-vs-Y plots.

Larger tiles pool more counts per spectrum; smaller tiles resolve position.
"""
import numpy as np

from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

from wara import spectrum as sp
from wara import peaksearch as ps

from .slicefit import SliceFitWindow
from . import theme as T


# Cap the grid so an over-fine tile size can't blow up memory (the per-tile
# count table is n_tiles × energy-bins) or spawn a PeakSearch per tile on Fit.
# 50 x 50 — the spectra binning still bounds the table; tile_spectra is O(N).
MAX_TILES = 2500


class TileGrid:
    """A uniform rectangular tiling of an X-Y plane extent.

    ``x_edges`` / ``y_edges`` are the column / row boundaries (length nx+1 /
    ny+1).  Tile ``(col, row)`` spans ``x_edges[col:col+2]`` ×
    ``y_edges[row:row+2]``; row 0 is the lowest Y.
    """

    def __init__(self, x_edges, y_edges):
        self.x_edges = np.asarray(x_edges, dtype=float)
        self.y_edges = np.asarray(y_edges, dtype=float)
        self.nx = len(self.x_edges) - 1
        self.ny = len(self.y_edges) - 1

    @property
    def n_tiles(self):
        return self.nx * self.ny

    def x_centers(self):
        return 0.5 * (self.x_edges[:-1] + self.x_edges[1:])

    def y_centers(self):
        return 0.5 * (self.y_edges[:-1] + self.y_edges[1:])

    def bounds(self, col, row):
        return (float(self.x_edges[col]), float(self.x_edges[col + 1]),
                float(self.y_edges[row]), float(self.y_edges[row + 1]))

    def indices(self, x, y):
        """Per-event ``(col, row, inside)`` arrays for coordinate arrays.

        Events outside the gridded extent get col/row clamped but
        ``inside`` False so callers can drop them.
        """
        col = np.digitize(x, self.x_edges) - 1
        row = np.digitize(y, self.y_edges) - 1
        inside = (col >= 0) & (col < self.nx) & (row >= 0) & (row < self.ny)
        col = np.clip(col, 0, max(self.nx - 1, 0))
        row = np.clip(row, 0, max(self.ny - 1, 0))
        return col, row, inside


def make_tiles(plane, tile_w, tile_h):
    """Build a :class:`TileGrid` of ``tile_w`` × ``tile_h`` tiles over *plane*.

    *plane* is ``(xlo, xhi, ylo, yhi)``.  Tiling starts at the low corner and
    steps by the tile size; the final column / row is clamped to the plane edge
    (so the last tile may be narrower).  Raises ``ValueError`` for non-positive
    sizes or when the resulting grid would exceed :data:`MAX_TILES`.
    """
    xlo, xhi, ylo, yhi = (float(v) for v in plane)
    if not (tile_w > 0 and tile_h > 0):
        raise ValueError("Tile width and length must be positive")
    span_x, span_y = xhi - xlo, yhi - ylo
    if not (span_x > 0 and span_y > 0):
        raise ValueError("The X-Y plane has no extent")
    nx = max(1, int(np.ceil(span_x / tile_w)))
    ny = max(1, int(np.ceil(span_y / tile_h)))
    if nx * ny > MAX_TILES:
        raise ValueError(
            f"{nx}×{ny} = {nx * ny} tiles exceeds the {MAX_TILES}-tile cap; "
            "use a larger tile size")
    x_edges = np.minimum(xlo + tile_w * np.arange(nx + 1), xhi)
    y_edges = np.minimum(ylo + tile_h * np.arange(ny + 1), yhi)
    x_edges[0], y_edges[0] = xlo, ylo
    return TileGrid(x_edges, y_edges)


def tile_color(tiles, col, row):
    """Stable colour for a tile, cycling the theme overlay palette by linear
    index so the map highlight and the overlaid spectrum always match."""
    idx = row * tiles.nx + col
    return T.OVERLAY_COLORS[idx % len(T.OVERLAY_COLORS)]


def tile_spectra(df, xkey, ykey, ekey, tiles, ebins, erange):
    """Histogram each tile's events into an energy spectrum.

    Returns ``(centers, counts)`` where *centers* is the shared energy-bin
    centre array and *counts* is an ``(nx, ny)`` object grid of per-tile count
    arrays (None where a tile has no events).

    Vectorised: every event is binned once into a (tile × energy) table via a
    single ``bincount``, so the cost is O(N) regardless of the tile count (the
    old per-tile masking loop was O(N · n_tiles) and didn't scale to fine grids).
    """
    x = df[xkey].to_numpy(dtype=float)
    y = df[ykey].to_numpy(dtype=float)
    e = df[ekey].to_numpy(dtype=float)
    col, row, inside = tiles.indices(x, y)
    e0, e1 = float(erange[0]), float(erange[1])
    edges = np.linspace(e0, e1, ebins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    nx, ny = tiles.nx, tiles.ny
    grid = np.empty((nx, ny), dtype=object)
    grid[:] = None
    if e1 <= e0:
        return centers, grid
    inb = inside & (e >= e0) & (e < e1)
    tile_idx = (row * nx + col)[inb]
    ebin = ((e[inb] - e0) / (e1 - e0) * ebins).astype(np.intp)
    np.clip(ebin, 0, ebins - 1, out=ebin)
    flat = tile_idx * ebins + ebin
    counts2d = np.bincount(flat, minlength=nx * ny * ebins).reshape(nx * ny, ebins)
    for c in range(nx):
        for r in range(ny):
            cc = counts2d[r * nx + c]          # a view into counts2d
            if cc.any():
                grid[c, r] = cc
    return centers, grid


def _tile_search(centers, counts, ebins, e_units):
    """Build a Spectrum + PeakSearch for one tile's spectrum.

    The PeakSearch reference is scaled to the binning exactly as the dt-slice
    builder does (420 ch / 12 ch FWHM tuned at 2**11 bins).
    """
    ref_x = max(1.0, 420.0 * ebins / 2 ** 11)
    ref_fwhm = max(1.0, 12.0 * ebins / 2 ** 11)
    spe = sp.Spectrum(counts=np.asarray(counts, float), energies=centers,
                      e_units=e_units)
    search = ps.PeakSearch(spe, ref_x, ref_fwhm, fwhm_at_0=1.0, min_snr=3.0)
    return spe, search


def tile_slices(centers, counts, selected, tiles, ebins, e_units, mode="tile"):
    """Build SliceFitWindow-style slice dicts for the selected tiles.

    ``mode="tile"`` → one slice per tile, carrying both ``xc`` and ``yc`` (the
    area is then plotted vs X *and* vs Y).  ``mode="x"`` / ``"y"`` → the tiles
    sharing each column / row are **summed** into one combined spectrum, giving
    one slice per occupied column (plotted vs X) or row (vs Y) — the orthogonal
    axis is collapsed for better statistics.  Each dict carries a ``caption``
    for the fit window plus zeroed ``t0``/``t1``/``tc`` for the base window's
    bookkeeping.
    """
    xcen, ycen = tiles.x_centers(), tiles.y_centers()
    nan = float("nan")
    slices = []
    if mode == "tile":
        for k, (c, r) in enumerate(sorted(selected)):
            cc = counts[c, r]
            cc = np.zeros_like(centers) if cc is None else cc
            spe, search = _tile_search(centers, cc, ebins, e_units)
            slices.append(dict(
                idx=k, t0=0.0, t1=0.0, tc=0.0,
                xc=float(xcen[c]), yc=float(ycen[r]), col=c, row=r,
                spe=spe, search=search, label=f"({c},{r})",
                caption=f"tile (x, y) = ({xcen[c]:.2f}, {ycen[r]:.2f})"))
        return slices
    # Combined: group the selected tiles by the kept axis, sum their spectra.
    groups = {}
    for (c, r) in selected:
        groups.setdefault(c if mode == "x" else r, []).append((c, r))
    for k, key in enumerate(sorted(groups)):
        members = groups[key]
        summed = np.zeros_like(centers)
        for (c, r) in members:
            cc = counts[c, r]
            if cc is not None:
                summed = summed + cc
        spe, search = _tile_search(centers, summed, ebins, e_units)
        ntxt = f"{len(members)} tile" + ("s" if len(members) != 1 else "")
        if mode == "x":
            pos = float(xcen[key]); xc, yc, lbl = pos, nan, f"x={pos:.2f}"
            cap = f"x = {pos:.2f}  ({ntxt})"
        else:
            pos = float(ycen[key]); xc, yc, lbl = nan, pos, f"y={pos:.2f}"
            cap = f"y = {pos:.2f}  ({ntxt})"
        slices.append(dict(
            idx=k, t0=0.0, t1=0.0, tc=0.0, xc=xc, yc=yc,
            spe=spe, search=search, label=lbl, caption=cap))
    return slices


# ── drawing ────────────────────────────────────────────────────────────────
# Grid lines and selected-tile fill on the X-Y map. Bright lines so the grid
# reads over the dark plasma hexbin; selected tiles are a neutral grey (they no
# longer compete with the colourful hexbin) labelled with their coordinate,
# which keys back to the matching coloured spectrum in the overlay legend.
GRID_LINE = "#e6e8f2"
SELECTED_FILL = "#b8bccc"


def draw_grid_overlay(ax, tiles, selected):
    """Overlay the tile grid lines and shade the selected tiles on the X-Y map.

    Selected tiles are filled neutral grey and labelled with their ``(col,row)``
    coordinate, drawn in that tile's spectrum colour so it matches the overlay
    legend.  Returns the created artists so the caller can remove just the
    overlay (leaving the expensive hexbin underneath) on the next refresh.
    """
    artists = []
    for xe in tiles.x_edges:
        artists.append(ax.axvline(xe, color=GRID_LINE, lw=1.0, alpha=0.8,
                                   zorder=4))
    for ye in tiles.y_edges:
        artists.append(ax.axhline(ye, color=GRID_LINE, lw=1.0, alpha=0.8,
                                   zorder=4))
    # Scale the label to the tile size so it sits inside without overflowing;
    # hide it when tiles get too small to carry legible text.
    show_labels = len(selected) <= 120
    for (c, r) in selected:
        x0, x1, y0, y1 = tiles.bounds(c, r)
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=SELECTED_FILL,
                         edgecolor=GRID_LINE, alpha=0.55, lw=1.4, zorder=5)
        ax.add_patch(rect)
        artists.append(rect)
        if show_labels:
            txt = ax.text(
                0.5 * (x0 + x1), 0.5 * (y0 + y1), f"({c},{r})",
                ha="center", va="center", zorder=6, fontsize=9,
                fontweight="bold", color=tile_color(tiles, c, r))
            txt.set_path_effects([pe.withStroke(linewidth=2.4,
                                                foreground=T.BG_DARK)])
            artists.append(txt)
    return artists


def plot_tile_overlay(ax, centers, counts, selected, tiles, bands, x_label):
    """Draw the selected tiles' energy spectra overlaid into *ax* (cleared)."""
    ax.clear()
    ax.set_facecolor(T.BG_PLOT)
    if not selected:
        ax.text(0.5, 0.5, "Select tiles on the X-Y map to overlay their spectra",
                ha="center", va="center", transform=ax.transAxes,
                color=T.TEXT_DIM, fontsize=12)
    else:
        for (c, r) in sorted(selected):
            cc = counts[c, r]
            if cc is None:
                continue
            ax.plot(centers, np.where(cc > 0, cc, np.nan),
                    color=tile_color(tiles, c, r), lw=0.9, alpha=0.85,
                    label=f"({c},{r})")
        for b in bands:
            ax.axvspan(b["emin"], b["emax"], color=b["color"], alpha=0.18)
        ax.set_yscale("log")
        if 0 < len(selected) <= 12:
            ax.legend(fontsize=10, ncol=2, facecolor=T.BG_PLOT,
                      edgecolor=T.BORDER, labelcolor=T.TEXT_PRIMARY)
    ax.set_xlabel(x_label, color=T.TEXT_PRIMARY, fontsize=12)
    ax.set_ylabel("Counts", color=T.TEXT_PRIMARY, fontsize=12)
    ax.tick_params(colors=T.TEXT_DIM, which="both", length=3, labelsize=11)
    for sp_ in ax.spines.values():
        sp_.set_color(T.BORDER)
    ax.grid(True, color="#3c3c66", linewidth=0.6, alpha=0.7)


def _area_panel(ax, results, axis, ylabel, unit):
    """Overlay every *results* entry that has data on this axis ("X" / "Y")."""
    ax.set_facecolor(T.BG_PLOT)
    n_drawn = 0
    for res in results:
        pos = np.asarray(res["x"] if axis == "X" else res["y"], float)
        vals = np.asarray(res["vals"], float)
        errs = np.asarray(res["errs"], float)
        finite = np.isfinite(pos)
        if not finite.any():
            continue
        p, v, e = pos[finite], vals[finite], errs[finite]
        order = np.argsort(p)
        color = res.get("color") or T.ACCENT_CYAN
        ax.errorbar(p[order], v[order], yerr=e[order], fmt="o-", color=color,
                    ecolor=T.TEXT_DIM, elinewidth=1.0, capsize=3, lw=1.2, ms=6,
                    mfc=color, mec=T.BG_DARK, label=res.get("label", ""))
        n_drawn += 1
    suffix = f" ({unit})" if unit else ""
    ax.set_xlabel(f"{axis}{suffix}", color=T.TEXT_PRIMARY, fontsize=12)
    ax.set_ylabel(ylabel, color=T.TEXT_PRIMARY, fontsize=11)
    ax.tick_params(colors=T.TEXT_DIM, which="both", length=3, labelsize=10)
    for sp_ in ax.spines.values():
        sp_.set_color(T.BORDER)
    ax.grid(True, color="#3c3c66", linewidth=0.6, alpha=0.7)
    if n_drawn >= 1:
        ax.legend(fontsize=10, facecolor=T.BG_PLOT, edgecolor=T.BORDER,
                  labelcolor=T.TEXT_PRIMARY)


def plot_area_overlays(fig, results):
    """Draw the net-peak-area profiles of the *visible* fitted bands onto *fig*.

    *results* is a list of payloads emitted by :class:`TileFitWindow`
    (``dict(label, color, x, y, vals, errs, ylabel, pos_unit, mode)``).  Each
    band is drawn in its own colour and overlaid; a panel is shown for an axis
    only when some visible band has data there (so a vs-X-only set draws one
    panel, while any per-tile band brings in both X and Y).
    """
    fig.clear()
    fig.set_facecolor(T.BG_PLOT)
    if not results:
        ax = fig.add_subplot(111)
        ax.set_facecolor(T.BG_PLOT)
        ax.text(0.5, 0.5, "Fit a band to plot net area vs position",
                ha="center", va="center", transform=ax.transAxes,
                color=T.TEXT_DIM, fontsize=12)
        for sp_ in ax.spines.values():
            sp_.set_color(T.BORDER)
        ax.set_xticks([]); ax.set_yticks([])
        return
    unit = results[0].get("pos_unit") or ""
    ylabel = results[0].get("ylabel", "Net peak area")
    needs_x = any(np.isfinite(np.asarray(r["x"], float)).any() for r in results)
    needs_y = any(np.isfinite(np.asarray(r["y"], float)).any() for r in results)
    if needs_x and needs_y:
        ax_x = fig.add_subplot(211)
        ax_y = fig.add_subplot(212)
        _area_panel(ax_x, results, "X", ylabel, unit)
        _area_panel(ax_y, results, "Y", ylabel, unit)
        ax_x.set_title("Net area vs position", color=T.TEXT_PRIMARY, fontsize=12)
    else:
        ax = fig.add_subplot(111)
        axis = "X" if needs_x else "Y"
        _area_panel(ax, results, axis, ylabel, unit)
        ax.set_title("Net area vs position", color=T.TEXT_PRIMARY, fontsize=12)
    fig.tight_layout()


class TileFitWindow(SliceFitWindow):
    """Interactive per-tile fit window: steps through the selected tiles'
    spectra harvesting a net peak area, then emits one point per tile (with the
    tile centre) for the area-vs-X / area-vs-Y plots.

    Reuses the Energy tab's :class:`~wara.gui_beta.slicefit.SliceFitWindow`
    wholesale; only the captions and the emitted payload differ.
    """

    def __init__(self, parent, slices, band, band_label, color, pos_unit="",
                 mode="tile"):
        super().__init__(parent, slices, band, band_label, color)
        self.pos_unit = pos_unit
        self.mode = mode
        self.setWindowTitle(f"Tile fits — {band_label}")
        plot_txt = {"tile": "Plot vs X / Y", "x": "Plot vs X",
                    "y": "Plot vs Y"}.get(mode, "Plot")
        self.btn_plot.setText(plot_txt)
        self.btn_plot.setToolTip("Plot the net peak area vs position")

    def _slice_caption(self, i, s):
        return f"{i + 1}/{len(self.slices)}  ·  {s.get('caption', '')}"

    def _emit_results(self):
        self._save_current()
        self._metric_by_slice[self._cur] = self._current_metric()
        xs, ys, vals, errs = [], [], [], []
        for i, s in enumerate(self.slices):
            v, e = self._metric_by_slice.get(i, (float("nan"), float("nan")))
            xs.append(s["xc"]); ys.append(s["yc"])
            vals.append(v); errs.append(e)
        self.results_ready.emit(dict(
            label=self.sel_label, color=self.sel_color,
            x=xs, y=ys, vals=vals, errs=errs,
            ylabel="Net peak area", pos_unit=self.pos_unit, mode=self.mode))
        axis_txt = {"tile": "vs X / Y", "x": "vs X", "y": "vs Y"}.get(
            self.mode, "")
        self.lbl_status.setText(f"Plotted {self.sel_label} {axis_txt}.")
