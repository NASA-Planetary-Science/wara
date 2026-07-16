"""Energy/time shifts dialog for the API tab (``ShiftsDialog``)."""
import numpy as np

from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

from PyQt5.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSizePolicy, QTabWidget, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt, QSize, QTimer

from wara import apicalc

from . import theme as T
from .widgets import hsep, header, labeled_row
from .api_common import API_PLOT_BG, _clear_axis


class ShiftsDialog(QDialog):
    """Drift-correction window: gain-shift the energy axis and shift/align the
    time axis, independently. Two tabs (Energy, Time), each showing the raw
    segment overlay above the aligned overlay (Preview), with controls to apply
    the correction to the API dataframe via the controller."""

    def __init__(self, controller):
        super().__init__(controller.app)
        self.c = controller
        self.setWindowTitle("Drift correction  -- shifts")
        self.setStyleSheet(T.STYLESHEET)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.resize(1000, 700)

        # Per-tab widget handles, keyed by "energy"/"time".
        self.ax_raw = {}
        self.ax_aligned = {}
        self.canvas = {}
        self.toolbar = {}
        self.fig = {}
        self.f_nseg = {}
        self.f_method = {}
        self.f_bins = {}
        self.f_xmin = {}
        self.f_xmax = {}
        self.lbl_state = {}
        self.btn_clear = {}
        self.btn_apply = {}
        # When the user previews a time-segment alignment, cache the per-event
        # aligned dt (and its description) so a constant Δt can be applied
        # straight on top of the *previewed* data -- no re-alignment, no separate
        # "Apply alignment" step.
        self._time_preview_pending = False
        self._previewed_aligned = None
        self._previewed_seg_desc = ""

        lay = QVBoxLayout(self)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)
        self.tabs.addTab(self._build_tab("energy"), "Energy")
        self.tabs.addTab(self._build_tab("time"), "Time")

    # ── construction ──────────────────────────────────────────────────────────
    def _build_tab(self, kind):
        tab = QWidget()
        row = QHBoxLayout(tab); row.setContentsMargins(8, 8, 8, 8); row.setSpacing(8)

        fig = Figure(figsize=(6, 6), facecolor=API_PLOT_BG)
        self.fig[kind] = fig
        self.ax_raw[kind] = fig.add_subplot(211)
        # Share the x-axis so zooming/panning the raw overlay moves the aligned
        # one in lockstep (they are the same axis, before vs after correction).
        self.ax_aligned[kind] = fig.add_subplot(212, sharex=self.ax_raw[kind],
                                                       sharey=self.ax_raw[kind])
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.setMinimumWidth(540)
        self.canvas[kind] = canvas
        # Pan/zoom toolbar above the plot  -- styled to match the API page's.
        toolbar = NavToolbar(canvas, self)
        toolbar.setObjectName("plot_toolbar")
        toolbar.setIconSize(QSize(22, 22))
        T.recolor_toolbar_icons(toolbar, T.TEXT_PRIMARY)
        self.toolbar[kind] = toolbar
        plot_col = QVBoxLayout(); plot_col.setContentsMargins(0, 0, 0, 0); plot_col.setSpacing(2)
        plot_col.addWidget(toolbar)
        plot_col.addWidget(canvas, 1)
        plot_w = QWidget(); plot_w.setLayout(plot_col)
        row.addWidget(plot_w, 1)

        side = QVBoxLayout(); side.setSpacing(6)
        axis = "energy channels" if kind == "energy" else "time (dt)"
        side.addWidget(header(f"{kind.upper()} DRIFT"))
        note = QLabel(f"Split the run into time segments and align each segment's "
                      f"{axis} onto the first (earliest) one.")
        note.setObjectName("stat_key"); note.setWordWrap(True)
        side.addWidget(note)

        nseg = QLineEdit("20"); nseg.setFixedWidth(70)
        r, _ = labeled_row("Segments", nseg); side.addWidget(r)
        self.f_nseg[kind] = nseg

        method = QComboBox(); method.addItems(["shift", "linear"])
        method.setToolTip("shift: additive slide (robust, single peak ok)\n"
                          "linear: slope + offset (true gain drift; needs ≥2 peaks)")
        r, _ = labeled_row("Method", method); side.addWidget(r)
        self.f_method[kind] = method

        default_bins = self.c.ebins if kind == "energy" else self.c.tbins
        bins = QLineEdit(str(default_bins)); bins.setFixedWidth(70)
        bins.setToolTip("Number of histogram bins used to align the segments"
                        + (" (time range focuses on the 1%–99% dt span)"
                           if kind == "time" else ""))
        r, _ = labeled_row("Bins", bins); side.addWidget(r)
        self.f_bins[kind] = bins

        xmin = QLineEdit(); xmin.setPlaceholderText("min"); xmin.setFixedWidth(70)
        xmax = QLineEdit(); xmax.setPlaceholderText("max"); xmax.setFixedWidth(70)
        xr = QHBoxLayout(); xr.setContentsMargins(0, 0, 0, 0)
        xr.addWidget(QLabel("Window")); xr.addWidget(xmin); xr.addWidget(xmax)
        xrw = QWidget(); xrw.setLayout(xr); side.addWidget(xrw)
        xmin.setToolTip("Optional: restrict the alignment to this axis window")
        self.f_xmin[kind] = xmin; self.f_xmax[kind] = xmax

        btn_prev = QPushButton("Preview"); btn_prev.setObjectName("action_btn")
        btn_prev.setCursor(Qt.PointingHandCursor)
        btn_prev.clicked.connect(lambda _=False, k=kind: self._preview(k, prospective=True))
        side.addWidget(btn_prev)

        btn_apply = QPushButton("Apply alignment")
        # Distinct colour per axis (green = energy panel, cyan = dt panel). Idle =
        # outlined (transparent fill); pressing fills it solid, and it returns to
        # the outline once the alignment is applied  -- so the user sees exactly
        # when the action takes place.
        btn_apply._color = T.ACCENT_GREEN if kind == "energy" else T.ACCENT_CYAN
        btn_apply._base_css = self._apply_css(btn_apply._color, solid=False)
        btn_apply.setStyleSheet(btn_apply._base_css)
        btn_apply.setCursor(Qt.PointingHandCursor)
        btn_apply.clicked.connect(lambda _=False, k=kind: self._apply_segments(k))
        self.btn_apply[kind] = btn_apply
        side.addWidget(btn_apply)

        if kind == "time":
            # The Time axis additionally keeps the simple manual constant shift.
            side.addWidget(hsep())
            side.addWidget(header("CONSTANT SHIFT"))
            const = QLineEdit(); const.setPlaceholderText("Δt (ns)"); const.setFixedWidth(70)
            r, b = labeled_row("Δt (ns)", const, apply_btn=True)
            const.setToolTip("Add a constant to every dt value (negative shifts left)")
            b.clicked.connect(self._apply_constant)
            const.returnPressed.connect(self._apply_constant)
            self.f_const = const
            side.addWidget(r)

        side.addWidget(hsep())
        clear = QPushButton("Clear correction"); clear.setObjectName("mini_btn")
        clear.setCursor(Qt.PointingHandCursor); clear.setEnabled(False)
        clear.clicked.connect(lambda _=False, k=kind: self._clear(k))
        self.btn_clear[kind] = clear
        side.addWidget(clear)

        state = QLabel("Not applied"); state.setObjectName("stat_key")
        state.setWordWrap(True); self.lbl_state[kind] = state
        side.addWidget(state)
        side.addStretch(1)

        holder = QWidget(); holder.setFixedWidth(250); holder.setLayout(side)
        row.addWidget(holder, 0)
        return tab

    # ── parameter reading ──────────────────────────────────────────────────────
    def _params(self, kind):
        """Return (n_segments, method, xrange) from the tab controls."""
        try:
            nseg = max(2, int(float(self.f_nseg[kind].text().strip())))
        except ValueError:
            nseg = 20
        method = self.f_method[kind].currentText()
        try:
            xr = (float(self.f_xmin[kind].text()), float(self.f_xmax[kind].text()))
            if xr[0] >= xr[1]:
                xr = None
        except ValueError:
            xr = None
        return nseg, method, xr

    def _bins(self, kind):
        """User-defined histogram bin count for the tab (falls back to the main
        view's energy/time bins on a bad entry)."""
        default = self.c.ebins if kind == "energy" else self.c.tbins
        try:
            return max(2, int(float(self.f_bins[kind].text().strip())))
        except ValueError:
            return default

    def _axis_cfg(self, kind):
        """(base column, out column, bins, base erange) for the tab's axis."""
        c = self.c
        if kind == "energy":
            return (c._chan_base or "energy_orig", "energy_drift",
                    self._bins("energy"), c._chan_erange())
        return ("dt", "dt_cal", self._bins("time"), c._dt_erange())

    def _segment_spectra(self, col, n_segments, bins, erange):
        """Per-segment histograms of *col* (no alignment, nothing committed)."""
        gs = apicalc.GainShift(self.c.df_api, n_segments=n_segments, bins=bins,
                               erange=erange, col=col, out_col="_preview_tmp")
        return gs._spectra

    # ── actions ─────────────────────────────────────────────────────────────────
    def _preview(self, kind, prospective=False):
        """Redraw the tab: raw segments on top; on the bottom the *committed*
        correction (constant or segment) when one is applied. The prospective
        segment alignment for the current controls is computed only when the user
        asks for it (``prospective=True``, the Preview button) -- opening the
        window must not run any shift computation, just show the raw segments."""
        c = self.c
        if c.df_api is None:
            return
        nseg, method, xr = self._params(kind)
        base, out_col, bins, base_erange = self._axis_cfg(kind)
        applied = c._dt_corrected if kind == "time" else c._egain_applied
        bottom = None
        try:
            raw_spectra = self._segment_spectra(base, nseg, bins, base_erange)
            if applied and out_col in c.df_api.columns:
                # Histogram the corrected column over the SAME kind of range as the
                # raw panel (time: 0.2%–99.5% percentile; energy: [0, max]) so the
                # shared-x view keeps the percentile zoom instead of snapping back
                # to the full dt span after applying.
                vals = c.df_api[out_col].astype(float).to_numpy()
                if kind == "time":
                    lo, hi = np.percentile(vals, [0.2, 99.5])
                    erange = ((float(lo), float(hi)) if hi > lo
                              else (float(vals.min()), float(vals.max())))
                else:
                    erange = (0.0, float(vals.max()))
                bottom = self._segment_spectra(out_col, nseg, bins, erange)
                bottom_title = "Corrected"
            elif prospective:
                gs = apicalc.GainShift(c.df_api, n_segments=nseg, bins=bins,
                                       erange=base_erange, col=base, out_col=out_col)
                gs.align(method=method, xrange=xr)
                bottom = gs._aligned_spectra
                bottom_title = f"Aligned ({method})"
                # Cache the per-event aligned dt so a constant Δt can be applied
                # directly on top of this preview (see _apply_constant).
                if kind == "time":
                    self._time_preview_pending = True
                    self._previewed_aligned = gs.df[out_col].astype(float).to_numpy()
                    self._previewed_seg_desc = f"{nseg} seg · {method}"
            else:
                bottom_title = "Press Preview to compute the alignment"
        except Exception as exc:  # noqa: BLE001  -- surface to the state label
            self.lbl_state[kind].setText(f"Preview failed: {exc}")
            return
        xlabel = "Raw channel" if kind == "energy" else "dt (ns)"
        yscale = "log" if kind == "energy" else "linear"
        # Draw the bottom panel first, then the raw panel last: the axes share y,
        # so the raw overlay's autoscale sets the final (positive) limits.
        if bottom is not None:
            self._draw_overlay(self.ax_aligned[kind], bottom, nseg,
                               bottom_title, xlabel, yscale)
        else:
            self._draw_placeholder(self.ax_aligned[kind], bottom_title, xlabel, yscale)
        self._draw_overlay(self.ax_raw[kind], raw_spectra, nseg,
                           f"Raw segments ({nseg})", xlabel, yscale)
        self.fig[kind].tight_layout()
        self.canvas[kind].draw_idle()
        # Reset the pan/zoom history so "Home" returns to this fresh view.
        self.toolbar[kind].update()

    def _draw_placeholder(self, ax, message, xlabel, yscale="log"):
        """Empty aligned panel with a hint, shown until the user runs a preview."""
        _clear_axis(ax)
        ax.set_ylim(0.1, 1.0)          # positive, so a log scale doesn't warn
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, color=T.TEXT_DIM, fontsize=12)
        ax.set_facecolor(API_PLOT_BG)
        ax.tick_params(colors=T.TEXT_DIM, which="both", length=3, labelsize=11)
        for sp_ in ax.spines.values():
            sp_.set_color(T.BORDER)
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
                color=T.TEXT_DIM, fontsize=12)

    def _draw_overlay(self, ax, spectra, n_seg, title, xlabel, yscale="log"):
        _clear_axis(ax)
        colors = cm.viridis(np.linspace(0, 1, n_seg))
        for i, spe in enumerate(spectra):
            ax.step(spe.energies, spe.counts, where="mid",
                    color=colors[i], lw=0.8, alpha=0.8)
        ax.set_yscale(yscale)
        ax.set_title(title, color=T.TEXT_PRIMARY, fontsize=12)
        ax.set_xlabel(xlabel, color=T.TEXT_DIM, fontsize=12)
        ax.set_facecolor(API_PLOT_BG)
        ax.tick_params(colors=T.TEXT_DIM, which="both", length=3, labelsize=11)
        for sp_ in ax.spines.values():
            sp_.set_color(T.BORDER)
        handles = [
            Line2D([0], [0], color=cm.viridis(0.0), lw=1.5, label="earlier"),
            Line2D([0], [0], color=cm.viridis(1.0), lw=1.5, label="later"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=11,
                  facecolor=API_PLOT_BG, edgecolor=T.BORDER,
                  labelcolor=T.TEXT_DIM, framealpha=0.7)

    @staticmethod
    def _apply_css(color, solid):
        """Stylesheet for an Apply button. ``solid`` fills it with *color* (the
        pressed/active state); otherwise it is an outline with a transparent
        fill (the idle state)."""
        bg = color if solid else "transparent"
        fg = T.BG_DARK if solid else color
        weight = 800 if solid else 700
        return (f"background-color:{bg}; color:{fg}; "
                f"border:2px solid {color}; border-radius:5px; "
                f"padding:8px 13px; font-size:14px; font-weight:{weight};")

    def _apply_segments(self, kind):
        # Fill the button solid for the (blocking) alignment so the press is
        # visible, then revert to the outline once it is applied.
        b = self.btn_apply[kind]
        b.setStyleSheet(self._apply_css(b._color, solid=True))
        b.repaint()
        nseg, method, xr = self._params(kind)
        recal = False
        if kind == "energy":
            recal = self.c.apply_energy_gainshift(nseg, method=method, xrange=xr,
                                                  bins=self._bins("energy"))
        else:
            self.c.apply_dt_gainshift(nseg, method=method, xrange=xr,
                                      bins=self._bins("time"))
            self._time_preview_pending = False
            self._previewed_aligned = None
        self._preview(kind)
        QTimer.singleShot(220, lambda: b.setStyleSheet(b._base_css))
        # Confirm the action with a clear message (the controller already pushed
        # the descriptive label into lbl_state via _refresh_shift_labels).
        label = self.c._egain_label if kind == "energy" else self.c._dt_label
        axis = "Energy" if kind == "energy" else "Time"
        self.lbl_state[kind].setText(f"✓ {axis} alignment applied  --  {label}")
        # The energy shift recomputed the energy axis, dropping a file-provided
        # calibration  -- tell the user it must be redone.
        if recal:
            QMessageBox.information(
                self, "Re-calibrate after shifting",
                "The energy gain-shift recomputed the energy axis from the raw "
                "channels, so the run's previous energy calibration was "
                "removed.\n\nRe-calibrate the energy before any quantitative "
                "analysis.")

    def _apply_constant(self):
        txt = self.f_const.text().strip()
        try:
            shift = float(txt)
        except ValueError:
            self.lbl_state["time"].setText("Constant shift must be a number (ns)")
            return
        # If a segment alignment was previewed (and not yet committed), apply the
        # constant directly on top of that previewed data -- using the exact
        # aligned values the user saw, not a fresh re-alignment.
        prev = self._previewed_aligned
        if (self._time_preview_pending and prev is not None
                and not self.c._dt_segments_applied
                and self.c.df_api is not None and len(prev) == len(self.c.df_api)):
            self.c.apply_dt_preview_shift(prev, shift, self._previewed_seg_desc)
        else:
            self.c.apply_dt_shift(shift)
        self._preview("time")
        self.lbl_state["time"].setText(
            f"✓ Time correction applied  --  {self.c._dt_label}")

    def _clear(self, kind):
        if kind == "energy":
            self.c.clear_energy_gainshift()
        else:
            self.c._clear_dt_shift()
            self._time_preview_pending = False
            self._previewed_aligned = None
        self._preview(kind)

    # ── state sync (called by the controller) ───────────────────────────────────
    def set_energy_state(self, applied, label):
        self.lbl_state["energy"].setText(label or "Not applied")
        self.btn_clear["energy"].setEnabled(applied)

    def set_time_state(self, applied, label):
        self.lbl_state["time"].setText(label or "Not applied")
        self.btn_clear["time"].setEnabled(applied)

    def refresh_previews(self):
        if self.c.df_api is None:
            return
        for kind in ("energy", "time"):
            self._preview(kind)
