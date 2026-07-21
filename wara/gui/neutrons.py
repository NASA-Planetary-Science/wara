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
    QFileDialog, QSizePolicy,
)
from PyQt5.QtCore import Qt, QSize

from wara.neutron_psd import NeutronTraces

from . import theme as T
from .widgets import hsep, header, stat_row, SpinBox

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

        # Controller callbacks (no-ops until wired).
        self.on_params = None        # (threshold_v, gate, prompt, tail) on release
        self.on_energy_range = None  # (lo, hi) or (None, None) to clear
        self.on_psd_region = None    # (elo, ehi, plo, phi) or None to clear
        self.on_psd_add = None       # (elo, ehi, plo, phi) — armed additive box

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
                    prompt_end, tail_end, n_shown, n_total):
        """Draw the sampled traces plus the draggable markers.

        *groups* is a list of ``(traces, color)`` tuples, each an already-
        subsampled (k, n_samples) array. When *color* is ``None`` the pulses are
        coloured by their own peak amplitude (default single-selection mode);
        otherwise every pulse in the group is drawn in that solid colour (the
        armed "PSD selections" mode). *n_total* is how many pulses passed the
        filter across all groups.
        """
        ax = self.ax_traces
        ax.clear()
        self._marker_lines = {}
        drawn = sum(len(tr) for tr, _c in groups)
        if drawn:
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
            label=f"threshold ({threshold_v * 1000:.0f} mV)")
        self._marker_lines["gate"] = ax.axvline(
            gate_start, color=T.ACCENT_CYAN, ls="--", lw=1.4, zorder=6, label="gate start")
        self._marker_lines["prompt"] = ax.axvline(
            prompt_end, color=T.ACCENT_AMBER, ls="--", lw=1.4, zorder=6, label="prompt end")
        self._marker_lines["tail"] = ax.axvline(
            tail_end, color=T.ACCENT_GREEN, ls="--", lw=1.4, zorder=6, label="tail end")

        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude (V)")
        ax.set_xlim(float(time_ns[0]), float(time_ns[-1]))
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        self._style(ax)

    # -- drawing: MCA panel ----------------------------------------------------
    def draw_mca(self, energy_all, overlays, bins, energy_range):
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
            ax.set_yscale("log")
        if energy_range[0] is not None:
            for x in energy_range:
                ax.axvline(x, color=T.ACCENT_RED, ls=":", lw=1.3, zorder=6)
        ax.set_xlabel("Pulse integral / energy (V·ns)")
        ax.set_ylabel("Counts")
        ax.set_title("MCA spectrum — drag to select an energy window")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
        self._style(ax)
        self._attach_span()

    # -- drawing: PSD panel ----------------------------------------------------
    def draw_psd(self, energy, psd, bins, energy_range, rects):
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
                      norm=LogNorm(), cmap="jet")
        if energy_range[0] is not None:
            for x in energy_range:
                ax.axvline(x, color=T.TEXT_PRIMARY, ls=":", lw=1.2, zorder=6)
        for elo, ehi, plo_, phi_, color, ls in rects:
            ax.add_patch(Rectangle(
                (elo, plo_), ehi - elo, phi_ - plo_, fill=False,
                edgecolor=color, ls=ls, lw=1.8, zorder=7))
        ax.set_xlabel("Energy / pulse integral (V·ns)")
        ax.set_ylabel("PSD = 1 - Q_prompt / Q_tail")
        ax.set_title("PSD vs. energy — draw coloured selection boxes" if self.armed
                     else "PSD vs. energy — drag a box to select")
        self._style(ax, grid=False)
        self._attach_rect(self.armed)

    def finish_draw(self):
        self.canvas.draw_idle()

    # -- selectors -------------------------------------------------------------
    def _attach_span(self):
        if self._span is not None:
            try:
                self._span.disconnect_events()
            except Exception:  # noqa: BLE001
                pass
            self._span = None
        # While armed for PSD selections the MCA span is disabled so selections
        # are made only on the PSD panel.
        if self.armed:
            return
        self._span = SpanSelector(
            self.ax_mca, self._span_done, "horizontal", useblit=True,
            interactive=True, drag_from_anywhere=True,
            props=dict(alpha=0.25, facecolor=T.ACCENT_AMBER))

    def _attach_rect(self, armed=False):
        if self._rect is not None:
            try:
                self._rect.disconnect_events()
            except Exception:  # noqa: BLE001
                pass
        # Armed: each drag commits a new box (non-interactive, no lingering
        # handles). Normal: a single interactive, re-draggable box.
        # The armed rubber-band needs a bold, high-contrast outline so it reads
        # against the bright jet histogram while being dragged.
        if armed:
            props = dict(facecolor=T.TEXT_PRIMARY, edgecolor=T.TEXT_PRIMARY,
                         linewidth=2.5, linestyle="--", alpha=0.35, fill=True)
        else:
            props = dict(facecolor=T.TEXT_PRIMARY, edgecolor=T.ACCENT_RED,
                         alpha=0.15, fill=True)
        self._rect = RectangleSelector(
            self.ax_psd, self._rect_done, useblit=True, button=[1],
            minspanx=0, minspany=0, spancoords="pixels", interactive=not armed,
            props=props)

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
        # Armed → additive coloured selection; otherwise the single filter box.
        if self.armed and self.on_psd_add is not None:
            if region is not None:
                self.on_psd_add(region)
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

        lay.addWidget(hsep()); lay.addWidget(header("GATING"))
        hint = QLabel(
            "Drag the coloured markers on the Traces panel. PSD is computed per "
            "pulse from the charge integrated over each gate window:\n"
            "• Q_prompt = ∫ gate start → prompt end\n"
            "• Q_tail = ∫ prompt end → tail end\n"
            "• Energy = ∫ gate start → tail end\n"
            "• PSD = 1 − Q_prompt / Q_tail\n"
            "Pulses peaking below the threshold are excluded.")
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
        brow = QHBoxLayout()
        self.spin_bins = SpinBox(); self.spin_bins.setRange(32, 1024)
        self.spin_bins.setValue(512); self.spin_bins.setSingleStep(32)
        self.spin_bins.setMaximumWidth(84)
        self.spin_bins.setToolTip("Histogram bins for the MCA and PSD panels")
        brow.addWidget(QLabel("Bins:")); brow.addStretch(1); brow.addWidget(self.spin_bins)
        lay.addLayout(brow)

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

        self.btn_reset = QPushButton("Clear selection")
        self.btn_reset.setObjectName("danger_btn")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setToolTip("Clear the MCA span and all PSD box selections")
        lay.addWidget(self.btn_reset)

        # Let full-width buttons shrink with the fixed-width panel instead of
        # forcing the scroll content wider than the viewport (which would clip
        # their right edge). They still stretch to fill the available width.
        for b in (self.btn_load, self.btn_psd_select, self.btn_reset):
            b.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        lay.addStretch(1)
        self.setWidget(inner)


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
        # On load only the Traces panel is drawn; the MCA and PSD panels stay
        # blank until the user makes their first Traces-panel selection (i.e.
        # commits the gate markers), which flips this True.
        self._analysis_ready = False
        self._rng = np.random.default_rng(0)
        self._wire()

    def _wire(self):
        self.opts.btn_load.clicked.connect(self._load)
        self.opts.btn_reset.clicked.connect(self._clear_selection)
        self.opts.btn_psd_select.toggled.connect(self._on_arm)
        self.opts.spin_shown.valueChanged.connect(lambda *_: self._redraw_traces_only())
        self.opts.spin_bins.valueChanged.connect(lambda *_: self._redraw())
        self.page.on_params = self._on_params
        self.page.on_energy_range = self._on_energy_range
        self.page.on_psd_region = self._on_psd_region
        self.page.on_psd_add = self._on_psd_add

    def _status(self, msg):
        self.app.statusBar().showMessage(f"  {msg}")

    # -- loading ---------------------------------------------------------------
    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self.app, "Load neutron trace file",
            filter="Trace files (*.npz *.npy *.txt *.csv *.dat);;All files (*)")
        if not path:
            return
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
        self._energy_range = (None, None)
        self._psd_region = None
        self._psd_selections = []
        # Fresh load: show traces only until the user adjusts the gates.
        self._analysis_ready = False
        if self.opts.btn_psd_select.isChecked():
            self.opts.btn_psd_select.setChecked(False)  # fires _on_arm to reset
        import os
        self.opts.lbl_file.setText(
            f"{os.path.basename(path)}\n{nt.n_traces:,} traces · "
            f"{'flipped' if nt.sign < 0 else 'positive'} polarity")
        self._redraw()
        self._status(f"Loaded {nt.n_traces:,} traces — adjust the gate markers "
                     "to compute the PSD and MCA")

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
        self._redraw()
        self._status("PSD recomputed")

    # -- selection -------------------------------------------------------------
    @property
    def _armed(self):
        return self.opts.btn_psd_select.isChecked()

    def _on_arm(self, checked):
        """Toggle multi-box 'PSD selections' mode. Arming clears the single-mode
        filters so selections are made fresh, only on the PSD panel."""
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

    def _clear_selection(self):
        self._energy_range = (None, None)
        self._psd_region = None
        self._psd_selections = []
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
        self.opts.lbl_thresh.setText(f"{nt.threshold_v * 1000:.0f} mV")
        self.opts.lbl_gate.setText(f"{nt.gate_start_ns:.1f} ns")
        self.opts.lbl_prompt.setText(f"{nt.prompt_end_ns:.1f} ns")
        self.opts.lbl_tail.setText(f"{nt.tail_end_ns:.1f} ns")

        # Before the first Traces-panel selection: draw the traces + gate
        # markers only and leave the MCA/PSD panels blank with a prompt.
        if not self._analysis_ready:
            self.opts.lbl_sel.setText("—")
            self._draw_traces(self._selection_groups())
            self.page.draw_pending(
                "Adjust the gate markers on the\n"
                "Traces panel to populate this plot")
            self.page.finish_draw()
            return

        bins = self.opts.spin_bins.value()
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
        self.page.draw_mca(energy_all, overlays, bins, self._energy_range)
        self.page.draw_psd(energy_all, nt.psd[valid], bins,
                           self._energy_range, self._psd_rects())
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
        n_max = self.opts.spin_shown.value()
        # Share the trace budget across groups so a busy multi-selection stays
        # legible (at least a few traces per group).
        per_group = max(5, n_max // max(1, len(groups)))
        n_total = 0
        draw_groups = []
        for mask, color, _label in groups:
            idx = np.flatnonzero(mask)
            n_total += idx.size
            if idx.size > per_group:
                idx = self._rng.choice(idx, size=per_group, replace=False)
                idx.sort()
            traces = (nt.corrected[idx].astype(float) if idx.size
                      else np.empty((0, nt.time_ns.size)))
            draw_groups.append((traces, color))
        n_shown = sum(len(tr) for tr, _c in draw_groups)
        self.page.draw_traces(
            nt.time_ns, draw_groups, nt.threshold_v, nt.gate_start_ns,
            nt.prompt_end_ns, nt.tail_end_ns, n_shown=n_shown, n_total=n_total)
