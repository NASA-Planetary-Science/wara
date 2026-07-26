"""Neutrons tab: interactive pulse-shape-discrimination (PSD) explorer.

Loads a PicoScope ``.npz`` acquisition of digitized PMT traces and shows three
linked, cross-filtering panels:

* **Traces** (top-left) — a representative sample of baseline-corrected pulses
  coloured by PSD ratio, with *draggable* markers: a horizontal voltage
  threshold and three vertical lines setting the gate start, the prompt/tail
  boundary and the tail end. Dragging any marker recomputes the energy/PSD of
  every pulse and refreshes the other two panels.
* **MCA** (bottom-left) — the pulse-integral (energy) spectrum. Drag a
  horizontal span to restrict the traces and PSD panels to that energy window.
* **PSD** (right) — a 2-D energy-vs-PSD histogram. Drag a rectangle to restrict
  the traces and MCA panels to that energy×PSD region.

The MCA span and PSD rectangle are independent filters that AND together; the
Traces panel always shows a sample of the surviving pulses.

Arming **Figure of merit** turns the PSD rubber band into a slice selector: the
pulses inside the dragged box are projected onto the PSD axis and fitted with a
double Gaussian in a pop-out window (:class:`FOMDialog`), which reports
``FOM = |mu_n - mu_gamma| / (FWHM_gamma + FWHM_n)``. Disarming closes it.

The numerics live in :mod:`wara.neutron_psd`; this module is purely the Qt/
matplotlib front-end.
"""
import numpy as np

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector, RectangleSelector
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea,
    QFileDialog, QSizePolicy, QLineEdit, QDialog, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QCheckBox,
)
from PyQt5.QtCore import Qt, QSize, QTimer

from wara.neutron_psd import NeutronTraces
from wara import spectrum as sp

from . import theme as T
from .widgets import hsep, header, stat_row, SpinBox, ComboBox, labeled_row

# Pixel tolerance for grabbing a draggable marker line.
_GRAB_PX = 8


# ── Plot page ──────────────────────────────────────────────────────────────────
class NeutronsPage(QWidget):
    """Three linked PSD panels with draggable markers and drag-selection.

    The controller sets the ``on_params`` / ``on_energy_range`` /
    ``on_psd_region`` callbacks and drives drawing through :meth:`draw_traces`,
    :meth:`draw_mca` and :meth:`draw_psd`.
    """

    def __init__(self):
        super().__init__()
        self.setObjectName("content")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(12, 7), constrained_layout=True, facecolor=T.BG_PLOT)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.toolbar = NavToolbar(self.canvas, self)
        self.toolbar.setObjectName("plot_toolbar")
        self.toolbar.setIconSize(QSize(22, 22))
        T.recolor_toolbar_icons(self.toolbar, T.TEXT_PRIMARY)
        lay.addWidget(self.toolbar, 0)
        lay.addWidget(self.canvas, 1)

        self.readout = QLabel("")
        self.readout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.readout.setStyleSheet(
            f"background:{T.BG_PLOT}; font-size:14px; font-weight:700; padding:3px 8px;")
        lay.addWidget(self.readout, 0)

        self.ax_traces = self.ax_mca = self.ax_psd = None
        self._marker_lines = {}     # key -> Line2D (draggable)
        self._drag_key = None       # which marker is currently being dragged
        self._span = None           # MCA SpanSelector
        self._rect = None           # PSD RectangleSelector
        self.armed = False          # armed for multi-box "PSD selections" mode
        self.fom_armed = False      # armed for figure-of-merit slice selection

        # Axis / marker units, overridden per data source (V for PicoScope,
        # ADC for PIXIE). ``thresh_scale`` scales the stored threshold for its
        # readout (×1000 -> mV for volts, ×1 for ADC counts).
        self.amp_unit = "V"
        self.charge_unit = "V·ns"
        self.thresh_scale = 1000.0
        self.thresh_unit = "mV"

        # Controller callbacks (no-ops until wired).
        self.on_params = None        # (threshold_v, gate, prompt, tail) on release
        self.on_energy_range = None  # (lo, hi) or (None, None) to clear
        self.on_psd_region = None    # (elo, ehi, plo, phi) or None to clear
        self.on_psd_add = None       # (elo, ehi, plo, phi) — armed additive box
        self.on_fom_region = None    # (elo, ehi, plo, phi) — FOM slice box

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        self._build_axes()
        self.show_empty()

    # -- axes / styling --------------------------------------------------------
    def _build_axes(self):
        self.fig.clf()
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15])
        self.ax_traces = self.fig.add_subplot(gs[0, 0])
        self.ax_mca = self.fig.add_subplot(gs[1, 0])
        self.ax_psd = self.fig.add_subplot(gs[:, 1])

    def set_fom_mode(self, on):
        """Arm/disarm figure-of-merit mode: the PSD rubber band then selects the
        slice to project into the pop-out FOM window instead of cross-filtering.

        The canvas layout is unchanged; the caller must redraw so the PSD panel
        picks up the new selector styling and title."""
        self.fom_armed = bool(on)

    def draw_pending(self, msg):
        """Blank the MCA and PSD panels with a prompt and detach their
        selectors, until the user commits gate markers on the Traces panel."""
        self._detach_selectors()
        for ax, sub in ((self.ax_mca, "MCA spectrum"), (self.ax_psd, "PSD vs. energy")):
            ax.clear()
            ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center",
                    va="center", color=T.TEXT_DIM, fontsize=11, fontweight="bold",
                    wrap=True)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(sub)
            self._style(ax, grid=False)

    def show_empty(self, msg="Load a trace file (.npz, .npy, .txt, .csv) to begin"):
        self._detach_selectors()
        self._marker_lines = {}
        self._build_axes()
        for ax in (self.ax_mca, self.ax_psd):
            ax.set_xticks([]); ax.set_yticks([])
        self.ax_traces.text(
            0.5, 0.5, msg, transform=self.ax_traces.transAxes, ha="center",
            va="center", fontsize=14, color=T.BORDER, fontweight="bold", wrap=True)
        self.ax_traces.set_xticks([]); self.ax_traces.set_yticks([])
        self._style(self.ax_traces); self._style(self.ax_mca); self._style(self.ax_psd)
        self.readout.setText("")
        self.canvas.draw_idle()

    def _style(self, ax, grid=True):
        ax.set_facecolor(T.BG_PLOT)
        ax.tick_params(colors=T.TEXT_DIM, which="both", length=3, labelsize=9)
        ax.xaxis.label.set_color(T.TEXT_DIM)
        ax.yaxis.label.set_color(T.TEXT_DIM)
        ax.xaxis.label.set_fontsize(10); ax.yaxis.label.set_fontsize(10)
        if ax.get_title():
            ax.title.set_color(T.TEXT_PRIMARY); ax.title.set_fontsize(11)
        for sp in ax.spines.values():
            sp.set_color(T.BORDER)
        if grid:
            ax.grid(True, color=T.GRID, linewidth=0.6, alpha=0.5)
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor(T.BG_PANEL)
            leg.get_frame().set_edgecolor(T.BORDER)
            for txt in leg.get_texts():
                txt.set_color(T.TEXT_PRIMARY)

    # -- drawing: traces panel -------------------------------------------------
    def draw_traces(self, time_ns, groups, threshold_v, gate_start,
                    prompt_end, tail_end, n_shown, n_total, log_y=False,
                    average=False):
        """Draw the sampled traces plus the draggable markers.

        *groups* is a list of ``(traces, color)`` tuples, each an already-
        subsampled (k, n_samples) array. When *color* is ``None`` the pulses are
        coloured by their own peak amplitude (default single-selection mode);
        otherwise every pulse in the group is drawn in that solid colour (the
        armed "PSD selections" mode). *n_total* is how many pulses passed the
        filter across all groups.

        With *average* set each group holds a single already-averaged trace,
        drawn as one bold line (one per PSD selection when armed); *n_shown* is
        then how many pulses went into those averages.
        """
        ax = self.ax_traces
        ax.clear()
        self._marker_lines = {}
        drawn = sum(len(tr) for tr, _c in groups)
        if drawn and average:
            for traces, color in groups:
                if not len(traces):
                    continue
                ax.plot(time_ns, traces[0], lw=2.0, alpha=0.95, zorder=5,
                        color=color or T.ACCENT_CYAN)
            ax.set_title(f"Traces — average of {n_shown:,} of "
                         f"{n_total:,} selected pulses")
        elif drawn:
            for traces, color in groups:
                if not len(traces):
                    continue
                if color is None:
                    amp = traces.max(axis=1)
                    lo, hi = float(amp.min()), float(amp.max())
                    norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-6)
                    cmap = mpl.colormaps["viridis"]
                    # Dimmest pulses first so the bright ones sit on top.
                    for i in np.argsort(amp):
                        ax.plot(time_ns, traces[i], lw=0.6, alpha=0.5,
                                color=cmap(norm(amp[i])))
                else:
                    for tr in traces:
                        ax.plot(time_ns, tr, lw=0.6, alpha=0.5, color=color)
            ax.set_title(f"Traces — {n_shown} of {n_total} selected pulses")
        else:
            ax.text(0.5, 0.5, "No pulses in the current selection",
                    transform=ax.transAxes, ha="center", va="center",
                    color=T.TEXT_DIM, fontsize=12, fontweight="bold")
            ax.set_title("Traces")

        # Shaded prompt / tail gate windows.
        ax.axvspan(gate_start, prompt_end, color=T.ACCENT_RED, alpha=0.10, zorder=0)
        ax.axvspan(prompt_end, tail_end, color=T.SNR_PURPLE, alpha=0.10, zorder=0)

        # Draggable markers (grabbed by proximity in _on_press).
        self._marker_lines["threshold"] = ax.axhline(
            threshold_v, color=T.ACCENT_RED, ls="--", lw=1.4, zorder=6,
            label=f"threshold ({threshold_v * self.thresh_scale:.0f} {self.thresh_unit})")
        self._marker_lines["gate"] = ax.axvline(
            gate_start, color=T.ACCENT_CYAN, ls="--", lw=1.4, zorder=6, label="gate start")
        self._marker_lines["prompt"] = ax.axvline(
            prompt_end, color=T.ACCENT_AMBER, ls="--", lw=1.4, zorder=6, label="prompt end")
        self._marker_lines["tail"] = ax.axvline(
            tail_end, color=T.ACCENT_GREEN, ls="--", lw=1.4, zorder=6, label="tail end")

        ax.set_xlabel("Time (ns)")
        ax.set_ylabel(f"Amplitude ({self.amp_unit})")
        ax.set_xlim(float(time_ns[0]), float(time_ns[-1]))
        # Log amplitude masks the near-zero baseline (and any negative
        # undershoot), leaving the positive pulse body -- useful for the tail.
        ax.set_yscale("log" if log_y else "linear")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        self._style(ax)

    # -- drawing: MCA panel ----------------------------------------------------
    def draw_mca(self, energy_all, overlays, bins, energy_range, log_y=True):
        """Histogram all valid energies with per-selection subsets overlaid.

        *overlays* is a list of ``(energy_array, color, label)`` tuples drawn as
        filled histograms on top of the full spectrum. Marks the active MCA
        energy band and (re)attaches the horizontal span selector.
        """
        ax = self.ax_mca
        ax.clear()
        if len(energy_all):
            rng = (float(energy_all.min()), float(energy_all.max()))
            ax.hist(energy_all, bins=bins, range=rng, histtype="step",
                    color=T.ACCENT_CYAN, lw=1.2, label="all")
            for energy_sel, color, label in overlays:
                if len(energy_sel):
                    ax.hist(energy_sel, bins=bins, range=rng, histtype="stepfilled",
                            color=color, alpha=0.5, label=label)
        ax.set_yscale("log" if log_y else "linear")
        if energy_range[0] is not None:
            for x in energy_range:
                ax.axvline(x, color=T.ACCENT_RED, ls=":", lw=1.3, zorder=6)
        ax.set_xlabel(f"Pulse integral / energy ({self.charge_unit})")
        ax.set_ylabel("Counts")
        ax.set_title("MCA spectrum — drag to select an energy window")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
        self._style(ax)
        self._attach_span()

    # -- drawing: PSD panel ----------------------------------------------------
    def draw_psd(self, energy, psd, bins, energy_range, rects, log_counts=True):
        """2-D energy-vs-PSD histogram with the active MCA band and one or more
        selection rectangles overlaid; (re)attaches the rectangle selector.

        *rects* is a list of ``(elo, ehi, plo, phi, color, linestyle)`` boxes.
        The page's ``armed`` flag switches the title/selector styling into the
        multi-selection ("PSD selections") mode.
        """
        ax = self.ax_psd
        ax.clear()
        if len(energy):
            ex = (float(energy.min()), float(energy.max()))
            plo, phi = np.percentile(psd, [1.0, 99.0])
            pad = 0.05 * (phi - plo) if phi > plo else 0.1
            ax.hist2d(energy, psd, bins=[bins, bins],
                      range=[list(ex), [plo - pad, phi + pad]],
                      norm=LogNorm() if log_counts else None, cmap="jet")
        if energy_range[0] is not None:
            for x in energy_range:
                ax.axvline(x, color=T.TEXT_PRIMARY, ls=":", lw=1.2, zorder=6)
        for elo, ehi, plo_, phi_, color, ls in rects:
            ax.add_patch(Rectangle(
                (elo, plo_), ehi - elo, phi_ - plo_, fill=False,
                edgecolor=color, ls=ls, lw=1.8, zorder=7))
        ax.set_xlabel(f"Energy / pulse integral ({self.charge_unit})")
        ax.set_ylabel("PSD = 1 - Q_prompt / Q_tail")
        if self.armed:
            title = "PSD vs. energy — draw coloured selection boxes"
        elif self.fom_armed:
            title = "PSD vs. energy — drag a slice across BOTH bands"
        else:
            title = "PSD vs. energy — drag a box to select"
        ax.set_title(title)
        self._style(ax, grid=False)
        self._attach_rect()

    def finish_draw(self):
        # Reset the navigation toolbar's zoom/pan history so the freshly drawn
        # view becomes the new 'Home'. Without this, after a selection redraws a
        # panel with a new axis range, pressing Home restores the stale pre-
        # selection limits and squashes/cuts off the filtered traces.
        try:
            self.toolbar.update()
        except Exception:  # noqa: BLE001
            pass
        self.canvas.draw_idle()

    # -- selectors -------------------------------------------------------------
    def _attach_span(self):
        if self._span is not None:
            try:
                self._span.disconnect_events()
            except Exception:  # noqa: BLE001
                pass
            self._span = None
        # While armed (PSD selections or FOM) the MCA span is disabled so
        # selections are made only on the PSD panel.
        if self.armed or self.fom_armed:
            return
        self._span = SpanSelector(
            self.ax_mca, self._span_done, "horizontal", useblit=True,
            interactive=True, drag_from_anywhere=True,
            props=dict(alpha=0.25, facecolor=T.ACCENT_AMBER))

    def _attach_rect(self):
        if self._rect is not None:
            try:
                self._rect.disconnect_events()
            except Exception:  # noqa: BLE001
                pass
        # PSD selections armed: each drag commits a new box (non-interactive, no
        # lingering handles). FOM armed / normal: a single interactive,
        # re-draggable box. The armed rubber-bands need a bold, high-contrast
        # outline so they read against the bright jet histogram while dragged.
        if self.armed:
            props = dict(facecolor=T.TEXT_PRIMARY, edgecolor=T.TEXT_PRIMARY,
                         linewidth=2.5, linestyle="--", alpha=0.35, fill=True)
        elif self.fom_armed:
            props = dict(facecolor=T.ACCENT_GREEN, edgecolor=T.ACCENT_GREEN,
                         linewidth=2.5, linestyle="--", alpha=0.25, fill=True)
        else:
            props = dict(facecolor=T.TEXT_PRIMARY, edgecolor=T.ACCENT_RED,
                         alpha=0.15, fill=True)
        self._rect = RectangleSelector(
            self.ax_psd, self._rect_done, useblit=True, button=[1],
            minspanx=0, minspany=0, spancoords="pixels",
            interactive=not self.armed, props=props)

    def _detach_selectors(self):
        for s in (self._span, self._rect):
            if s is not None:
                try:
                    s.disconnect_events()
                    s.set_visible(False)
                except Exception:  # noqa: BLE001
                    pass
        self._span = self._rect = None

    def _span_done(self, lo, hi):
        if self.on_energy_range is None:
            return
        if hi - lo <= 0:
            self.on_energy_range(None, None)
        else:
            self.on_energy_range(float(lo), float(hi))

    def _rect_done(self, epress, erelease):
        if epress.xdata is None or erelease.xdata is None:
            return
        elo, ehi = sorted((epress.xdata, erelease.xdata))
        plo, phi = sorted((epress.ydata, erelease.ydata))
        degenerate = ehi - elo <= 0 or phi - plo <= 0
        region = None if degenerate else (float(elo), float(ehi), float(plo), float(phi))
        # Armed → additive coloured selection; FOM armed → the projection slice;
        # otherwise the single cross-filter box.
        if self.armed and self.on_psd_add is not None:
            if region is not None:
                self.on_psd_add(region)
        elif self.fom_armed:
            if self.on_fom_region is not None:
                self.on_fom_region(region)
        elif self.on_psd_region is not None:
            self.on_psd_region(region)

    # -- draggable markers -----------------------------------------------------
    def _on_press(self, event):
        if (event.inaxes is not self.ax_traces or event.button != 1
                or not self._marker_lines or self.toolbar.mode):
            return
        self._drag_key = self._nearest_marker(event)

    def _nearest_marker(self, event):
        """Return the marker key within grab tolerance of the cursor, else None."""
        ax = self.ax_traces
        best_key, best_px = None, _GRAB_PX
        # Vertical markers: compare cursor x to the line's x in pixels.
        for key in ("gate", "prompt", "tail"):
            line = self._marker_lines.get(key)
            if line is None:
                continue
            xline = line.get_xdata()[0]
            px = abs(ax.transData.transform((xline, 0))[0]
                     - ax.transData.transform((event.xdata, 0))[0])
            if px < best_px:
                best_key, best_px = key, px
        # Horizontal threshold: compare cursor y in pixels.
        line = self._marker_lines.get("threshold")
        if line is not None:
            yline = line.get_ydata()[0]
            px = abs(ax.transData.transform((0, yline))[1]
                     - ax.transData.transform((0, event.ydata))[1])
            if px < best_px:
                best_key, best_px = "threshold", px
        return best_key

    def _on_motion(self, event):
        if self._drag_key is None:
            self._cursor_readout(event)
            return
        if event.inaxes is not self.ax_traces or event.xdata is None:
            return
        line = self._marker_lines[self._drag_key]
        if self._drag_key == "threshold":
            line.set_ydata([event.ydata, event.ydata])
        else:
            line.set_xdata([event.xdata, event.xdata])
        self.canvas.draw_idle()

    def _on_release(self, event):
        if self._drag_key is None or self.on_params is None:
            self._drag_key = None
            return
        self._drag_key = None
        th = self._marker_lines["threshold"].get_ydata()[0]
        gate = self._marker_lines["gate"].get_xdata()[0]
        prompt = self._marker_lines["prompt"].get_xdata()[0]
        tail = self._marker_lines["tail"].get_xdata()[0]
        # Keep the gate ordering sane before handing the values back.
        gate, prompt, tail = sorted((gate, prompt, tail))
        self.on_params(float(th), float(gate), float(prompt), float(tail))

    def _cursor_readout(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            self.readout.setText("")
            return
        xlab = ax.get_xlabel() or "x"
        ylab = ax.get_ylabel() or "y"
        self.readout.setText(
            f"<span style='color:{T.ACCENT_CYAN}'>{xlab}: {event.xdata:.4g}</span>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;"
            f"<span style='color:{T.ACCENT_GREEN}'>{ylab}: {event.ydata:.4g}</span>")


# ── Options column ──────────────────────────────────────────────────────────────
class NeutronsOptions(QScrollArea):
    """Scrollable options for the Neutrons tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(9, 12, 9, 12); lay.setSpacing(6)

        title = QLabel("NEUTRONS"); title.setObjectName("opt_title")
        lay.addWidget(title); lay.addWidget(hsep())

        lay.addWidget(header("FILE"))
        self.btn_load = QPushButton("Load traces")
        self.btn_load.setObjectName("open_btn")
        self.btn_load.setCursor(Qt.PointingHandCursor)
        self.btn_load.setToolTip(
            "Load a trace file: a PicoScope .npz (time_ns + A_traces) or an "
            ".npy / .txt / .csv holding one trace per row")
        lay.addWidget(self.btn_load)
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setObjectName("stat_key"); self.lbl_file.setWordWrap(True)
        lay.addWidget(self.lbl_file)

        # ── PIXIE-16 run source ───────────────────────────────────────────────
        lay.addWidget(hsep()); lay.addWidget(header("PIXIE RUN"))
        pix_hint = QLabel(
            "Load time-aligned traces straight from a PIXIE run (same date / run "
            "/ channel as the API page). Traces are aligned per the option below "
            "before PSD.")
        pix_hint.setObjectName("stat_key"); pix_hint.setWordWrap(True)
        lay.addWidget(pix_hint)

        self.ed_date = QLineEdit(); self.ed_date.setPlaceholderText("YYYY-MM-DD")
        r, _ = labeled_row("Date", self.ed_date); lay.addWidget(r)
        self.ed_run = QLineEdit(); self.ed_run.setPlaceholderText("e.g. 19")
        r, _ = labeled_row("Run", self.ed_run); lay.addWidget(r)
        self.cmb_channel = ComboBox()
        self.cmb_channel.setToolTip("Detector channel to analyze (populated on load)")
        r, _ = labeled_row("Channel", self.cmb_channel); lay.addWidget(r)
        self.cmb_align = ComboBox()
        self.cmb_align.addItems(["Fast filter", "Edge", "Peak", "None"])
        self.cmb_align.setToolTip(
            "Time-alignment applied to every trace before PSD. 'Fast filter' "
            "(default) aligns on the reconstructed fast trigger.")
        r, _ = labeled_row("Align", self.cmb_align); lay.addWidget(r)
        self.cmb_cfd = ComboBox()
        self.cmb_cfd.addItems(["CFD on", "CFD off"])
        self.cmb_cfd.setToolTip("Which CFD acquisition to read (on = enabled).")
        r, _ = labeled_row("CFD", self.cmb_cfd); lay.addWidget(r)

        self.btn_load_pixie = QPushButton("Load PIXIE run")
        self.btn_load_pixie.setObjectName("open_btn")
        self.btn_load_pixie.setCursor(Qt.PointingHandCursor)
        self.btn_load_pixie.setToolTip(
            "Read the run with the chosen alignment / CFD and run PSD on the "
            "selected channel. Change the channel to re-analyze without re-reading.")
        lay.addWidget(self.btn_load_pixie)

        self.btn_pixie_stats = QPushButton("PIXIE stats")
        self.btn_pixie_stats.setObjectName("primary_btn")
        self.btn_pixie_stats.setCursor(Qt.PointingHandCursor)
        self.btn_pixie_stats.setEnabled(False)
        self.btn_pixie_stats.setToolTip(
            "Per-channel event counts, CFD errors, pile-up, trace flags and "
            "energy summary for the loaded PIXIE run.")
        lay.addWidget(self.btn_pixie_stats)

        lay.addWidget(hsep()); lay.addWidget(header("GATING"))
        hint = QLabel(
            "• Q_prompt = ∫ gate start → prompt end\n"
            "• Q_tail = ∫ prompt end → tail end\n"
            "• Energy = ∫ gate start → tail end\n"
            "• PSD = 1 − Q_prompt / Q_tail")
        hint.setObjectName("stat_key"); hint.setWordWrap(True)
        lay.addWidget(hint)
        self.row_thresh, self.lbl_thresh = stat_row("Threshold", T.ACCENT_RED)
        self.row_gate, self.lbl_gate = stat_row("Gate start", T.ACCENT_CYAN)
        self.row_prompt, self.lbl_prompt = stat_row("Prompt end", T.ACCENT_AMBER)
        self.row_tail, self.lbl_tail = stat_row("Tail end", T.ACCENT_GREEN)
        for r in (self.row_thresh, self.row_gate, self.row_prompt, self.row_tail):
            lay.addWidget(r)

        lay.addWidget(hsep()); lay.addWidget(header("DISPLAY"))
        drow = QHBoxLayout()
        self.spin_shown = SpinBox(); self.spin_shown.setRange(10, 2000)
        self.spin_shown.setValue(150); self.spin_shown.setSingleStep(10)
        self.spin_shown.setMaximumWidth(84)
        self.spin_shown.setToolTip("Maximum number of traces drawn (a random sample)")
        lbl_shown = QLabel("Traces:"); lbl_shown.setToolTip(
            "Maximum number of traces drawn (a random sample)")
        drow.addWidget(lbl_shown); drow.addStretch(1); drow.addWidget(self.spin_shown)
        lay.addLayout(drow)
        mrow = QHBoxLayout()
        self.spin_bins_mca = SpinBox(); self.spin_bins_mca.setRange(32, 8192)
        self.spin_bins_mca.setValue(512); self.spin_bins_mca.setSingleStep(32)
        self.spin_bins_mca.setMaximumWidth(84)
        self.spin_bins_mca.setToolTip("Histogram bins for the MCA spectrum")
        mrow.addWidget(QLabel("MCA bins:")); mrow.addStretch(1)
        mrow.addWidget(self.spin_bins_mca)
        lay.addLayout(mrow)
        prow = QHBoxLayout()
        self.spin_bins_psd = SpinBox(); self.spin_bins_psd.setRange(32, 1024)
        self.spin_bins_psd.setValue(300); self.spin_bins_psd.setSingleStep(32)
        self.spin_bins_psd.setMaximumWidth(84)
        self.spin_bins_psd.setToolTip("2-D histogram bins for the PSD-vs-energy panel")
        prow.addWidget(QLabel("PSD bins:")); prow.addStretch(1)
        prow.addWidget(self.spin_bins_psd)
        lay.addLayout(prow)

        self.cb_mca_log = QCheckBox("MCA log Y"); self.cb_mca_log.setChecked(True)
        self.cb_mca_log.setToolTip("Logarithmic y-axis on the MCA spectrum")
        lay.addWidget(self.cb_mca_log)
        self.cb_traces_log = QCheckBox("Traces log Y")
        self.cb_traces_log.setToolTip("Logarithmic y-axis on the Traces panel")
        lay.addWidget(self.cb_traces_log)
        self.cb_psd_log = QCheckBox("PSD log counts"); self.cb_psd_log.setChecked(True)
        self.cb_psd_log.setToolTip("Logarithmic colour scale for the PSD density")
        lay.addWidget(self.cb_psd_log)
        self.cb_avg_trace = QCheckBox("Average trace only")
        self.cb_avg_trace.setToolTip(
            "Replace the individual traces with a single averaged pulse: the "
            "mean of the random sample drawn above, or — when 'PSD selections' "
            "is ON — the mean of every pulse inside each coloured PSD box "
            "(one averaged trace per selection, in its own colour).")
        lay.addWidget(self.cb_avg_trace)
        avg_hint = QLabel(
            "Averages the drawn sample; with PSD selections ON it averages all "
            "selected pulses.")
        avg_hint.setObjectName("stat_key"); avg_hint.setWordWrap(True)
        lay.addWidget(avg_hint)

        lay.addWidget(hsep()); lay.addWidget(header("SELECTION"))
        selhint = QLabel(
            "Drag a horizontal span on the MCA panel or a box on the PSD panel "
            "to cross-filter the pulses.")
        selhint.setObjectName("stat_key"); selhint.setWordWrap(True)
        lay.addWidget(selhint)
        self.row_sel, self.lbl_sel = stat_row("Selected", T.ACCENT_AMBER)
        # Let the value wrap: a long armed-mode count must never force the
        # fixed-width options panel wider (which would push buttons off-screen).
        self.lbl_sel.setWordWrap(True)
        lay.addWidget(self.row_sel)

        self.btn_psd_select = QPushButton("PSD selections")
        self.btn_psd_select.setObjectName("arm_btn")
        self.btn_psd_select.setCheckable(True)
        self.btn_psd_select.setCursor(Qt.PointingHandCursor)
        self.btn_psd_select.setToolTip(
            "Arm multi-selection mode: draw several coloured boxes on the PSD "
            "panel, each shown as its own colour in the Traces and MCA panels. "
            "Disarm to return to single-selection cross-filtering.")
        lay.addWidget(self.btn_psd_select)

        self.btn_fom = QPushButton("Figure of merit")
        self.btn_fom.setObjectName("arm_btn")
        self.btn_fom.setCheckable(True)
        self.btn_fom.setCursor(Qt.PointingHandCursor)
        self.btn_fom.setToolTip(
            "Arm figure-of-merit mode: drag a box on the PSD panel spanning "
            "BOTH the gamma and the neutron band and the FOM pops up in its own "
            "window. The pulses inside are projected onto the PSD axis, fitted "
            "with a double Gaussian, and reduced to "
            "FOM = |μ_n − μ_γ| / (FWHM_γ + FWHM_n). FOM ≥ 1.27 is the usual "
            "threshold for clean separation. Disarm to close the window and "
            "return to normal cross-filtering.")
        lay.addWidget(self.btn_fom)
        fom_hint = QLabel(
            "The FOM box only sets the projected slice — it does not "
            "cross-filter the panels. Each new box refreshes the pop-out.")
        fom_hint.setObjectName("stat_key"); fom_hint.setWordWrap(True)
        lay.addWidget(fom_hint)
        frow = QHBoxLayout()
        self.spin_bins_fom = SpinBox(); self.spin_bins_fom.setRange(10, 1000)
        self.spin_bins_fom.setValue(80); self.spin_bins_fom.setSingleStep(10)
        self.spin_bins_fom.setMaximumWidth(84)
        self.spin_bins_fom.setToolTip(
            "Histogram bins of the 1-D PSD projection fitted for the FOM")
        lbl_fbins = QLabel("FOM bins:")
        lbl_fbins.setToolTip("Histogram bins of the 1-D PSD projection fitted "
                             "for the FOM")
        frow.addWidget(lbl_fbins); frow.addStretch(1); frow.addWidget(self.spin_bins_fom)
        lay.addLayout(frow)
        self.row_fom, self.lbl_fom = stat_row("FOM", T.ACCENT_GREEN)
        self.lbl_fom.setWordWrap(True)
        lay.addWidget(self.row_fom)

        self.btn_reset = QPushButton("Clear selection")
        self.btn_reset.setObjectName("danger_btn")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setToolTip("Clear the MCA span and all PSD box selections")
        lay.addWidget(self.btn_reset)

        self.btn_send_spec = QPushButton("Send to spectrum")
        self.btn_send_spec.setObjectName("primary_btn")
        self.btn_send_spec.setCursor(Qt.PointingHandCursor)
        self.btn_send_spec.setEnabled(False)
        self.btn_send_spec.setToolTip(
            "Send the current MCA spectrum (the selected pulses, or all valid "
            "pulses when nothing is selected) to the Spectrum tab")
        lay.addWidget(self.btn_send_spec)

        # Let full-width buttons shrink with the fixed-width panel instead of
        # forcing the scroll content wider than the viewport (which would clip
        # their right edge). They still stretch to fill the available width.
        for b in (self.btn_load, self.btn_load_pixie, self.btn_pixie_stats,
                  self.btn_psd_select, self.btn_fom, self.btn_reset,
                  self.btn_send_spec):
            b.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        lay.addStretch(1)
        self.setWidget(inner)


# ── PIXIE stats dialog ────────────────────────────────────────────────────────
class PixieStatsDialog(QDialog):
    """Per-channel event statistics for a loaded PIXIE run: trace counts, CFD
    errors, pile-up, trace-out-of-range flags and an energy summary."""

    # (header, per-channel-frame -> formatted string).
    COLUMNS = [
        ("Traces", lambda g: f"{len(g):,}"),
        ("CFD errors", lambda g: PixieStatsDialog._count_pct(g, "CFD_error")),
        ("Pile-up", lambda g: PixieStatsDialog._count_pct(g, "pileup")),
        ("Trace flags", lambda g: PixieStatsDialog._count_pct(g, "trace_flag")),
        ("Empty traces", lambda g: f"{int(g['trace'].map(lambda t: t is None or len(t) == 0).sum()):,}"
            if "trace" in g.columns else "--"),
        ("Median E", lambda g: f"{g['energy'].median():,.0f}"
            if "energy" in g.columns and len(g) else "--"),
        ("Max E", lambda g: f"{g['energy'].max():,.0f}"
            if "energy" in g.columns and len(g) else "--"),
    ]

    @staticmethod
    def _count_pct(g, col):
        if col not in g.columns or len(g) == 0:
            return "--"
        n = int(g[col].astype(bool).sum())
        return f"{n:,} ({100.0 * n / len(g):.2f}%)"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PIXIE trace statistics")
        self.setStyleSheet(T.STYLESHEET)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.resize(760, 380)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(6)
        lay.addWidget(header("PIXIE TRACE STATISTICS"))
        self.lbl_meta = QLabel(""); self.lbl_meta.setObjectName("stat_key")
        self.lbl_meta.setWordWrap(True)
        lay.addWidget(self.lbl_meta)

        self.table = QTableWidget(0, 1 + len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(
            ["Channel"] + [h for h, _ in self.COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self.table, 1)

    def populate(self, df, desc):
        self.lbl_meta.setText(f"{desc}  ·  {len(df):,} total events")
        if "channel" in df.columns:
            chans = sorted(int(c) for c in df.channel.unique())
            groups = [(str(c), df[df.channel == c]) for c in chans]
        else:
            groups = []
        # A "Total" row across all channels always closes the table.
        rows = groups + [("Total", df)]
        self.table.setRowCount(len(rows))
        for r, (label, g) in enumerate(rows):
            ch_item = QTableWidgetItem(label)
            ch_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 0, ch_item)
            for c, (_hdr, fmt) in enumerate(self.COLUMNS, start=1):
                try:
                    text = fmt(g)
                except Exception:  # noqa: BLE001
                    text = "--"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, item)


# ── Figure-of-merit pop-out ───────────────────────────────────────────────────
class FOMDialog(QDialog):
    """Pop-out window with the 1-D PSD projection of the selected slice, its
    double-Gaussian fit and the resulting figure of merit.

    Opened (and refreshed in place) by the controller every time a slice is
    dragged on the PSD panel while the "Figure of merit" button is armed. It is
    non-modal and stays on top so the selection can be adjusted with the window
    open; closing it does not disarm the mode.
    """

    #: FOM at or above this is the usual threshold for clean n/γ separation.
    GOOD_FOM = 1.27

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Figure of merit")
        self.setStyleSheet(T.STYLESHEET)
        self.resize(760, 560)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(6)
        lay.addWidget(header("NEUTRON / GAMMA FIGURE OF MERIT"))

        self.fig = Figure(figsize=(7.4, 4.6), constrained_layout=True,
                          facecolor=T.BG_PLOT)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavToolbar(self.canvas, self)
        self.toolbar.setObjectName("plot_toolbar")
        self.toolbar.setIconSize(QSize(22, 22))
        T.recolor_toolbar_icons(self.toolbar, T.TEXT_PRIMARY)
        lay.addWidget(self.toolbar, 0)
        lay.addWidget(self.canvas, 1)

        self.ax = self.fig.add_subplot(111)
        self.lbl = QLabel("")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet(
            f"background:{T.BG_PLOT}; font-size:13px; font-weight:700; "
            f"padding:4px 8px;")
        lay.addWidget(self.lbl, 0)

    def _style(self, grid=True):
        ax = self.ax
        ax.set_facecolor(T.BG_PLOT)
        ax.tick_params(colors=T.TEXT_DIM, which="both", length=3, labelsize=9)
        ax.xaxis.label.set_color(T.TEXT_DIM); ax.yaxis.label.set_color(T.TEXT_DIM)
        if ax.get_title():
            ax.title.set_color(T.TEXT_PRIMARY); ax.title.set_fontsize(11)
        for s in ax.spines.values():
            s.set_color(T.BORDER)
        if grid:
            ax.grid(True, color=T.GRID, linewidth=0.6, alpha=0.5)
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor(T.BG_PANEL)
            leg.get_frame().set_edgecolor(T.BORDER)
            for txt in leg.get_texts():
                txt.set_color(T.TEXT_PRIMARY)

    def show_message(self, message):
        """Blank the plot and explain why there is no fit."""
        ax = self.ax
        ax.clear()
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center",
                va="center", color=T.TEXT_DIM, fontsize=11, fontweight="bold",
                wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Figure of merit")
        self._style(grid=False)
        self.lbl.setText("")
        self.canvas.draw_idle()

    def plot(self, result, region=None, charge_unit="V·ns"):
        """Draw *result* (a :class:`wara.neutron_psd.FOMResult`) for the slice
        *region* — ``(elo, ehi, plo, phi)`` — in units *charge_unit*."""
        ax = self.ax
        ax.clear()
        centers = result.centers
        ax.bar(centers, result.counts, width=np.diff(result.edges),
               color=T.TEXT_DIM, alpha=0.35, align="center",
               label="PSD projection")
        # Oversample the fit so the Gaussians stay smooth against the bars.
        xs = np.linspace(centers[0], centers[-1], 600)
        ax.plot(xs, result.component(xs, "gamma"), color=T.ACCENT_CYAN, lw=1.5,
                ls="--", label=f"γ: μ={result.mu_gamma:.4g}, "
                               f"FWHM={result.fwhm_gamma:.3g}")
        ax.plot(xs, result.component(xs, "neutron"), color=T.ACCENT_AMBER, lw=1.5,
                ls="--", label=f"n: μ={result.mu_n:.4g}, FWHM={result.fwhm_n:.3g}")
        ax.plot(xs, result.curve(xs), color=T.ACCENT_RED, lw=2.0,
                label="double Gaussian")
        for mu in (result.mu_gamma, result.mu_n):
            ax.axvline(mu, color=T.TEXT_PRIMARY, ls=":", lw=1.0, zorder=6)

        ok = np.isfinite(result.fom) and result.fom >= self.GOOD_FOM
        color = T.ACCENT_GREEN if ok else T.ACCENT_RED
        ax.text(0.02, 0.97,
                f"FOM = {result.fom:.3f}\n"
                f"S = {result.separation:.4g}\n"
                f"ΣFWHM = {result.fwhm_gamma + result.fwhm_n:.4g}\n"
                f"{result.n_events:,} pulses · R² = {result.r_squared:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=10,
                fontweight="bold", color=color,
                bbox=dict(facecolor=T.BG_PANEL, edgecolor=T.BORDER, alpha=0.85,
                          boxstyle="round,pad=0.35"))
        sub = ""
        if region is not None:
            elo, ehi, _plo, _phi = region
            sub = f" — slice {elo:.4g}–{ehi:.4g} {charge_unit}"
        ax.set_title(f"FOM = S / (FWHM_γ + FWHM_n){sub}")
        ax.set_xlabel("PSD = 1 - Q_prompt / Q_tail")
        ax.set_ylabel("Counts")
        ax.legend(loc="upper right", fontsize=8)
        self._style()

        verdict = ("clean separation (≥ 1.27)" if ok
                   else f"below the {self.GOOD_FOM} clean-separation threshold")
        self.lbl.setText(
            f"<span style='color:{color}'>FOM = {result.fom:.3f}</span> "
            f"<span style='color:{T.TEXT_DIM}'>— {verdict} · "
            f"S = {result.separation:.4g} · "
            f"FWHM γ/n = {result.fwhm_gamma:.4g} / {result.fwhm_n:.4g}</span>")
        self.canvas.draw_idle()


# ── Controller ──────────────────────────────────────────────────────────────────
class NeutronsController:
    """Wires the NeutronsOptions panel and NeutronsPage to the loaded dataset."""

    def __init__(self, app, options: NeutronsOptions, page: NeutronsPage):
        self.app = app
        self.opts = options
        self.page = page
        self.nt = None                 # NeutronTraces or None
        self._energy_range = (None, None)   # MCA span filter
        self._psd_region = None             # PSD rectangle filter (single mode)
        self._psd_selections = []           # armed mode: list of {region, color}
        self._fom_region = None             # FOM mode: projected slice box
        self._fom_result = None             # FOMResult for that slice
        self._fom_error = None              # why the last FOM fit failed
        self._fom_dialog = None             # FOMDialog pop-out (lazily created)
        # On load only the Traces panel is drawn; the MCA and PSD panels stay
        # blank until the user makes their first Traces-panel selection (i.e.
        # commits the gate markers), which flips this True.
        self._analysis_ready = False
        self._rng = np.random.default_rng(0)
        # Cached PIXIE run so switching channel doesn't re-read from disk.
        self._pixie_df = None
        self._pixie_dt = None
        self._pixie_desc = ""
        self._wire()

    # Bright flash colours keyed by button objectName (mirrors the API tab).
    _FLASH_BG = {"open_btn": "#c4a8ff", "primary_btn": T.ACCENT_CYAN,
                 "danger_btn": T.ACCENT_RED}

    def _flash_button(self, button):
        """Briefly brighten a button so the click registers before the (blocking)
        load freezes the event loop, then revert to the themed style."""
        bg = self._FLASH_BG.get(button.objectName(), T.ACCENT_CYAN)
        button.setStyleSheet(
            f"background-color:{bg}; border-color:{bg}; "
            f"color:{T.BG_DARK}; font-weight:800;")
        button.repaint()
        QTimer.singleShot(220, lambda: button.setStyleSheet(""))

    def _set_page_units(self, amp_unit, charge_unit, thresh_scale, thresh_unit):
        """Set the axis / threshold units used by the plot page (V for PicoScope,
        ADC for PIXIE)."""
        self.page.amp_unit = amp_unit
        self.page.charge_unit = charge_unit
        self.page.thresh_scale = thresh_scale
        self.page.thresh_unit = thresh_unit

    def _wire(self):
        self.opts.btn_load.clicked.connect(self._load)
        self.opts.btn_load_pixie.clicked.connect(self._load_pixie)
        self.opts.cmb_channel.activated.connect(lambda *_: self._rebuild_pixie_channel())
        self.opts.btn_pixie_stats.clicked.connect(self._show_pixie_stats)
        self.opts.btn_send_spec.clicked.connect(self._send_to_spectrum)
        self.opts.btn_reset.clicked.connect(self._clear_selection)
        self.opts.btn_psd_select.toggled.connect(self._on_arm)
        self.opts.btn_fom.toggled.connect(self._on_fom_arm)
        self.opts.spin_bins_fom.valueChanged.connect(lambda *_: self._refit_fom())
        self.opts.spin_shown.valueChanged.connect(lambda *_: self._redraw_traces_only())
        self.opts.cb_traces_log.toggled.connect(lambda *_: self._redraw_traces_only())
        self.opts.cb_avg_trace.toggled.connect(lambda *_: self._redraw_traces_only())
        self.opts.spin_bins_mca.valueChanged.connect(lambda *_: self._redraw())
        self.opts.spin_bins_psd.valueChanged.connect(lambda *_: self._redraw())
        self.opts.cb_mca_log.toggled.connect(lambda *_: self._redraw())
        self.opts.cb_psd_log.toggled.connect(lambda *_: self._redraw())
        self.page.on_params = self._on_params
        self.page.on_energy_range = self._on_energy_range
        self.page.on_psd_region = self._on_psd_region
        self.page.on_psd_add = self._on_psd_add
        self.page.on_fom_region = self._on_fom_region

    def _status(self, msg):
        self.app.statusBar().showMessage(f"  {msg}")

    # -- loading ---------------------------------------------------------------
    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self.app, "Load neutron trace file",
            filter="Trace files (*.npz *.npy *.txt *.csv *.dat);;All files (*)")
        if not path:
            return
        self._flash_button(self.opts.btn_load)
        try:
            self._status("Loading traces…")
            self.app.repaint()
            nt = NeutronTraces.from_file(path)
            nt.compute()
        except Exception as exc:  # noqa: BLE001 — surface load/compute errors in the UI
            self.nt = None
            self.page.show_empty(f"Could not load file:\n{exc}")
            self.opts.lbl_file.setText("No file loaded")
            self._status(f"Failed to load: {exc}")
            return
        self.nt = nt
        # PicoScope traces are in volts; reset the page units in case the last
        # load was a PIXIE (ADC) run.
        self._set_page_units("V", "V·ns", 1000.0, "mV")
        self._pixie_df = None
        self.opts.btn_pixie_stats.setEnabled(False)
        self._energy_range = (None, None)
        self._psd_region = None
        self._psd_selections = []
        # Fresh load: show traces only until the user adjusts the gates.
        self._analysis_ready = False
        if self.opts.btn_psd_select.isChecked():
            self.opts.btn_psd_select.setChecked(False)  # fires _on_arm to reset
        if self.opts.btn_fom.isChecked():
            self.opts.btn_fom.setChecked(False)         # fires _on_fom_arm
        self._clear_fom()
        import os
        self.opts.lbl_file.setText(
            f"{os.path.basename(path)}\n{nt.n_traces:,} traces · "
            f"{'flipped' if nt.sign < 0 else 'positive'} polarity")
        self._redraw()
        self._status(f"Loaded {nt.n_traces:,} traces — adjust the gate markers "
                     "to compute the PSD and MCA")

    # -- loading: PIXIE run ----------------------------------------------------
    _ALIGN_MAP = {"Fast filter": "fast", "Edge": "edge", "Peak": "peak",
                  "None": None}
    _CFD_MAP = {"CFD on": "on", "CFD off": "off"}

    def _load_pixie(self):
        """Read a PIXIE run (with the chosen alignment / CFD), populate the
        channel list and analyze the selected channel."""
        from wara import helper_api

        date = self.opts.ed_date.text().strip()
        run_txt = self.opts.ed_run.text().strip()
        if not date or not run_txt:
            self._status("Enter a PIXIE date and run number first")
            return
        try:
            runnr = int(run_txt)
        except ValueError:
            self._status("Run number must be an integer")
            return
        align = self._ALIGN_MAP.get(self.opts.cmb_align.currentText(), "fast")
        cfd = self._CFD_MAP.get(self.opts.cmb_cfd.currentText(), "on")

        self._flash_button(self.opts.btn_load_pixie)
        try:
            self._status("Reading PIXIE run…")
            self.app.repaint()
            df = helper_api.read_trace_data(date=date, runnr=runnr, cfd=cfd,
                                            align=align)
            dt_ns = helper_api.read_sample_interval_ns(date, runnr)
        except Exception as exc:  # noqa: BLE001 — surface load errors in the UI
            self._pixie_df = None
            self.nt = None
            self.page.show_empty(f"Could not read PIXIE run:\n{exc}")
            self._status(f"Failed to read run: {exc}")
            return

        self._pixie_df = df
        self._pixie_dt = dt_ns
        self._pixie_desc = f"{date} run {runnr} · {align or 'no'} align · {cfd}"
        # PIXIE traces are raw ADC, not volts.
        self._set_page_units("ADC", "ADC·ns", 1.0, "ADC")
        self.opts.btn_pixie_stats.setEnabled(True)

        # Populate the channel combo (keep the current pick if still present).
        chans = (sorted(int(c) for c in df.channel.unique())
                 if "channel" in df.columns else [])
        cur = self.opts.cmb_channel.currentText()
        cb = self.opts.cmb_channel
        cb.blockSignals(True)
        cb.clear()
        cb.addItems([str(c) for c in chans])
        if cur in [str(c) for c in chans]:
            cb.setCurrentText(cur)
        cb.blockSignals(False)

        self._rebuild_pixie_channel()

    def _rebuild_pixie_channel(self):
        """(Re)build the PSD dataset for the selected channel from the cached
        PIXIE run — no disk re-read."""
        if self._pixie_df is None:
            return
        txt = self.opts.cmb_channel.currentText()
        channel = int(txt) if txt else None
        align = self._ALIGN_MAP.get(self.opts.cmb_align.currentText(), "fast")
        try:
            nt = NeutronTraces.from_pixie(
                channel=channel, align=align, df=self._pixie_df,
                dt_ns=self._pixie_dt)
            nt.compute()
        except Exception as exc:  # noqa: BLE001
            self.nt = None
            self.page.show_empty(f"Could not analyze channel:\n{exc}")
            self._status(f"Failed: {exc}")
            return
        self.nt = nt
        self._energy_range = (None, None)
        self._psd_region = None
        self._psd_selections = []
        self._analysis_ready = False
        if self.opts.btn_psd_select.isChecked():
            self.opts.btn_psd_select.setChecked(False)
        if self.opts.btn_fom.isChecked():
            self.opts.btn_fom.setChecked(False)
        self._clear_fom()
        self.opts.lbl_file.setText(
            f"PIXIE {self._pixie_desc} · ch {channel}\n{nt.n_traces:,} traces")
        self._redraw()
        self._status(f"Loaded {nt.n_traces:,} PIXIE traces (ch {channel}) — "
                     "adjust the gate markers to compute the PSD and MCA")

    def _show_pixie_stats(self):
        """Pop up per-channel event statistics for the loaded PIXIE run."""
        if self._pixie_df is None:
            self._status("Load a PIXIE run first")
            return
        dlg = PixieStatsDialog(self.app)
        dlg.populate(self._pixie_df, self._pixie_desc)
        dlg.exec_()

    def _send_to_spectrum(self):
        """Histogram the current MCA selection and load it on the Spectrum tab."""
        if self.nt is None or not self._analysis_ready:
            self._status("Adjust the gate markers to build a spectrum first")
            return
        nt = self.nt
        mask = self._selection_mask()
        energies = nt.energy[mask]
        if energies.size == 0:
            self._status("No pulses in the current selection to send")
            return
        bins = self.opts.spin_bins_mca.value()
        rng = (float(energies.min()), float(energies.max()))
        cts, edges = np.histogram(energies, bins=bins, range=rng)
        centers = 0.5 * (edges[1:] + edges[:-1])
        # Uncalibrated pulse-integral axis: carry the real values on adc_channels
        # (a pure coordinate) while counts stay on the 0..N index, matching how
        # the API tab sends an uncalibrated spectrum.
        try:
            spect = sp.Spectrum(counts=cts, adc_channels=centers)
        except Exception as exc:  # noqa: BLE001
            self._status(f"Could not build spectrum: {exc}")
            return
        name = "Neutrons spectrum"
        if self._pixie_df is not None:
            name = f"Neutrons {self._pixie_desc}"
        self.app.load_external_spectrum(spect, name, switch_tab=False)
        self._flash_button(self.opts.btn_send_spec)
        self._status(f"Sent {int(cts.sum()):,}-count spectrum to the Spectrum tab")

    # -- marker drag (recompute) ----------------------------------------------
    def _on_params(self, threshold_v, gate, prompt, tail):
        if self.nt is None:
            return
        self.nt.set_params(threshold_v=threshold_v, gate_start_ns=gate,
                           prompt_end_ns=prompt, tail_end_ns=tail)
        # A Traces-panel selection: from now on populate the MCA/PSD panels.
        self._analysis_ready = True
        self._status("Recomputing PSD…")
        self.app.repaint()
        self.nt.compute()
        # New gates → new PSD values, so any standing FOM slice must be re-fitted.
        if self.page.fom_armed and self._fom_region is not None:
            self._fit_fom()
            self._show_fom_dialog(raise_=False)
        self._redraw()
        self._status("PSD recomputed")

    # -- selection -------------------------------------------------------------
    @property
    def _armed(self):
        return self.opts.btn_psd_select.isChecked()

    def _on_arm(self, checked):
        """Toggle multi-box 'PSD selections' mode. Arming clears the single-mode
        filters so selections are made fresh, only on the PSD panel."""
        # The two armed modes both own the PSD rubber band, so only one at a
        # time (unchecking fires _on_fom_arm, which closes the FOM pop-out).
        if checked and self.opts.btn_fom.isChecked():
            self.opts.btn_fom.setChecked(False)
        self.page.armed = checked
        # Selecting on the PSD panel needs it populated, so arming commits the
        # analysis if the gates haven't been touched yet.
        if checked:
            self._analysis_ready = True
        # Make the armed state unmistakable: the button flips to its filled
        # amber :checked style and its label calls out that it's active.
        self.opts.btn_psd_select.setText(
            "● PSD selections ON" if checked else "PSD selections")
        self._energy_range = (None, None)
        self._psd_region = None
        self._psd_selections = []
        if self.nt is not None:
            self._redraw()
        self._status("PSD selections armed — draw coloured boxes on the PSD panel"
                     if checked else "PSD selections disarmed")

    def _on_energy_range(self, lo, hi):
        if self._armed:
            return  # the MCA span is disabled while armed
        self._energy_range = (lo, hi)
        self._redraw()

    def _on_psd_region(self, region):
        self._psd_region = region
        self._redraw()

    def _on_psd_add(self, region):
        """Append a new coloured selection box (armed mode)."""
        color = T.OVERLAY_COLORS[len(self._psd_selections) % len(T.OVERLAY_COLORS)]
        self._psd_selections.append({"region": region, "color": color})
        self._redraw()
        self._status(f"Added PSD selection {len(self._psd_selections)}")

    # -- figure of merit -------------------------------------------------------
    def _on_fom_arm(self, checked):
        """Toggle figure-of-merit mode: the PSD rubber band then selects the
        slice to project into the pop-out FOM window instead of cross-filtering.
        Disarming closes that window."""
        if checked and self.opts.btn_psd_select.isChecked():
            self.opts.btn_psd_select.setChecked(False)   # mutually exclusive
        self.opts.btn_fom.setText(
            "● Figure of merit ON" if checked else "Figure of merit")
        self._clear_fom()
        if not checked and self._fom_dialog is not None:
            self._fom_dialog.close()
        # Selecting on the PSD panel needs it populated, so arming commits the
        # analysis if the gates haven't been touched yet.
        if checked and self.nt is not None:
            self._analysis_ready = True
        self.page.set_fom_mode(checked)
        if self.nt is None:
            self.page.show_empty()
        else:
            self._redraw()
        self._status(
            "Figure of merit armed — drag a box across both PSD bands"
            if checked else "Figure of merit disarmed")

    def _clear_fom(self):
        self._fom_region = None
        self._fom_result = None
        self._fom_error = None
        self.opts.lbl_fom.setText("—")
        # An open pop-out would otherwise keep showing the dropped fit.
        if self._fom_dialog is not None and self._fom_dialog.isVisible():
            self._fom_dialog.show_message(
                "Drag a box on the PSD panel spanning\n"
                "both the gamma and the neutron band")

    def _on_fom_region(self, region):
        """A slice was dragged on the PSD panel: project it, fit it and pop the
        FOM window up (or refresh it if it is already open)."""
        if region is None or self.nt is None:
            self._clear_fom()
            self._redraw()
            return
        self._fom_region = region
        self._fit_fom()
        self._show_fom_dialog()
        self._redraw()

    def _refit_fom(self):
        """Re-fit with the current bin count (FOM bins spin box)."""
        if self._fom_region is None or self.nt is None:
            return
        self._fit_fom()
        self._show_fom_dialog(raise_=False)
        self._redraw()

    def _show_fom_dialog(self, raise_=True):
        """Draw the current fit (or its failure message) in the pop-out window,
        creating and showing it the first time. *raise_* brings it to the front,
        which a fresh selection does but a background re-fit does not."""
        if self._fom_dialog is None:
            self._fom_dialog = FOMDialog(self.app)
        dlg = self._fom_dialog
        if self._fom_result is not None:
            dlg.plot(self._fom_result, self._fom_region, self.page.charge_unit)
        else:
            dlg.show_message(self._fom_error or "No fit for the current slice")
        if raise_ or dlg.isVisible():
            dlg.show(); dlg.raise_()

    def _fit_fom(self):
        """Fit the double Gaussian for the stored slice, recording any failure."""
        from wara.neutron_psd import figure_of_merit

        nt = self.nt
        elo, ehi, plo, phi = self._fom_region
        mask = (nt.valid & (nt.energy >= elo) & (nt.energy <= ehi)
                & (nt.psd >= plo) & (nt.psd <= phi))
        if int(mask.sum()) < 10:
            # Usually a slice left over from before a gate drag: the PSD axis
            # has moved and the old box no longer covers any pulses.
            self._fom_result = None
            self._fom_error = ("No pulses in the selected slice.\n"
                               "Drag a new box across both PSD bands.")
            self.opts.lbl_fom.setText("empty slice")
            self._status("FOM: no pulses in the selected slice")
            return
        try:
            self._fom_result = figure_of_merit(
                nt.psd[mask], bins=self.opts.spin_bins_fom.value(),
                psd_range=(plo, phi))
            self._fom_error = None
        except Exception as exc:  # noqa: BLE001 — surface fit failures in the UI
            self._fom_result = None
            self._fom_error = f"Could not fit this slice:\n{exc}"
            self.opts.lbl_fom.setText("fit failed")
            self._status(f"FOM fit failed: {exc}")
            return
        r = self._fom_result
        self.opts.lbl_fom.setText(f"{r.fom:.3f} · {r.n_events:,} pulses")
        self._status(
            f"FOM = {r.fom:.3f} (S = {r.separation:.4g}, "
            f"ΣFWHM = {r.fwhm_gamma + r.fwhm_n:.4g}) from {r.n_events:,} pulses")

    def _clear_selection(self):
        self._energy_range = (None, None)
        self._psd_region = None
        self._psd_selections = []
        self._clear_fom()
        self._redraw()
        self._status("Selection cleared")

    # -- selection masks -------------------------------------------------------
    def _region_mask(self, region):
        nt = self.nt
        elo, ehi, plo, phi = region
        return ((nt.energy >= elo) & (nt.energy <= ehi)
                & (nt.psd >= plo) & (nt.psd <= phi))

    def _selection_mask(self):
        """Valid pulses surviving the current filters. In armed mode this is the
        union of every coloured PSD selection; otherwise the MCA-span AND
        PSD-box cross-filter."""
        nt = self.nt
        if self._armed:
            mask = np.zeros(nt.n_traces, bool)
            for s in self._psd_selections:
                mask |= self._region_mask(s["region"])
            return mask & nt.valid
        mask = nt.valid.copy()
        lo, hi = self._energy_range
        if lo is not None:
            mask &= (nt.energy >= lo) & (nt.energy <= hi)
        if self._psd_region is not None:
            mask &= self._region_mask(self._psd_region)
        return mask

    # -- drawing ---------------------------------------------------------------
    def _redraw(self):
        if self.nt is None:
            self.page.show_empty()
            return
        nt = self.nt
        self.opts.lbl_thresh.setText(
            f"{nt.threshold_v * self.page.thresh_scale:.0f} {self.page.thresh_unit}")
        self.opts.lbl_gate.setText(f"{nt.gate_start_ns:.1f} ns")
        self.opts.lbl_prompt.setText(f"{nt.prompt_end_ns:.1f} ns")
        self.opts.lbl_tail.setText(f"{nt.tail_end_ns:.1f} ns")

        # Before the first Traces-panel selection: draw the traces + gate
        # markers only and leave the MCA/PSD panels blank with a prompt.
        if not self._analysis_ready:
            self.opts.lbl_sel.setText("—")
            self.opts.btn_send_spec.setEnabled(False)
            self._draw_traces(self._selection_groups())
            self.page.draw_pending(
                "Adjust the gate markers on the\n"
                "Traces panel to populate this plot")
            self.page.finish_draw()
            return
        self.opts.btn_send_spec.setEnabled(True)

        valid = nt.valid
        energy_all = nt.energy[valid]

        # Per-selection masks drive the MCA overlays, PSD boxes and trace groups.
        groups = self._selection_groups()   # [(mask, color, label), ...]
        total_sel = int(self._selection_mask().sum())
        if self._armed:
            self.opts.lbl_sel.setText(f"{total_sel:,} · {len(self._psd_selections)} sel")
        else:
            self.opts.lbl_sel.setText(f"{total_sel:,} of {int(valid.sum()):,}")

        overlays = [(nt.energy[m], color or T.ACCENT_AMBER, label)
                    for m, color, label in groups]
        self.page.draw_mca(energy_all, overlays, self.opts.spin_bins_mca.value(),
                           self._energy_range, log_y=self.opts.cb_mca_log.isChecked())
        self.page.draw_psd(energy_all, nt.psd[valid], self.opts.spin_bins_psd.value(),
                           self._energy_range, self._psd_rects(),
                           log_counts=self.opts.cb_psd_log.isChecked())
        self._draw_traces(groups)
        self.page.finish_draw()

    def _selection_groups(self):
        """Return ``[(mask, color, label)]`` for the active selections.

        Armed: one entry per coloured PSD box. Otherwise a single group of the
        cross-filtered pulses, coloured by amplitude (``color`` is ``None``)."""
        nt = self.nt
        if self._armed:
            return [(self._region_mask(s["region"]) & nt.valid, s["color"],
                     f"sel {i + 1}")
                    for i, s in enumerate(self._psd_selections)]
        return [(self._selection_mask(), None, "selected")]

    def _psd_rects(self):
        """Rectangles to draw on the PSD panel: colored solid boxes when armed,
        else the single dashed cross-filter box."""
        if self._armed:
            return [(*s["region"], s["color"], "solid") for s in self._psd_selections]
        # FOM mode: keep the projected slice outlined after the rubber band goes.
        if self.page.fom_armed:
            return ([(*self._fom_region, T.ACCENT_GREEN, "--")]
                    if self._fom_region is not None else [])
        if self._psd_region is not None:
            return [(*self._psd_region, T.ACCENT_RED, "--")]
        return []

    def _redraw_traces_only(self):
        if self.nt is None:
            return
        self._draw_traces(self._selection_groups())
        self.page.finish_draw()

    def _draw_traces(self, groups):
        nt = self.nt
        average = self.opts.cb_avg_trace.isChecked()
        n_max = self.opts.spin_shown.value()
        # Share the trace budget across groups so a busy multi-selection stays
        # legible (at least a few traces per group).
        per_group = max(5, n_max // max(1, len(groups)))
        n_total = 0
        n_used = 0          # pulses actually drawn / averaged
        draw_groups = []
        for mask, color, _label in groups:
            idx = np.flatnonzero(mask)
            n_total += idx.size
            # Averaging while armed uses *every* pulse in the PSD box; otherwise
            # it averages exactly the random sample that would have been drawn.
            subsample = not (average and self._armed)
            if subsample and idx.size > per_group:
                idx = self._rng.choice(idx, size=per_group, replace=False)
                idx.sort()
            n_used += idx.size
            if average:
                # Mean in float64 straight from the float32 store (no full copy).
                traces = (np.mean(nt.corrected[idx], axis=0, dtype=np.float64,
                                  keepdims=True) if idx.size
                          else np.empty((0, nt.time_ns.size)))
            else:
                traces = (nt.corrected[idx].astype(float) if idx.size
                          else np.empty((0, nt.time_ns.size)))
            draw_groups.append((traces, color))
        n_shown = n_used if average else sum(len(tr) for tr, _c in draw_groups)
        self.page.draw_traces(
            nt.time_ns, draw_groups, nt.threshold_v, nt.gate_start_ns,
            nt.prompt_end_ns, nt.tail_end_ns, n_shown=n_shown, n_total=n_total,
            log_y=self.opts.cb_traces_log.isChecked(), average=average)
