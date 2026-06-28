"""Reusable widgets for the beta GUI: canvas, option panels, pages."""
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame,
    QSizePolicy, QSpacerItem, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox,
    QScrollArea, QLineEdit,
)
from PyQt5.QtCore import Qt, pyqtSignal

from . import theme as T

# wara logo, shown faded under the "NO SPECTRUM LOADED" placeholder. Loaded
# lazily (and once) so a missing/unreadable file never blocks the GUI.
from importlib.resources import files as _files
_LOGO_PATH = str(_files("wara").joinpath("ui/wara-logo.png"))
_LOGO_ARR = None


def _logo_array():
    global _LOGO_ARR
    if _LOGO_ARR is None:
        import matplotlib.image as _mpimg
        _LOGO_ARR = _mpimg.imread(_LOGO_PATH)
    return _LOGO_ARR


# ── Scroll-safe inputs ───────────────────────────────────────────────────────
# Inside a scroll area, combo boxes and spin boxes grab the mouse wheel and
# change their value when you're just trying to scroll the panel. These ignore
# the wheel unless focused, so the wheel scrolls the panel instead.
class _NoWheelMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class SpinBox(_NoWheelMixin, QSpinBox):
    pass


class DoubleSpinBox(_NoWheelMixin, QDoubleSpinBox):
    pass


class ComboBox(_NoWheelMixin, QComboBox):
    pass


# ── Small helpers ────────────────────────────────────────────────────────────
def fmt_count(v):
    """Integer-valued counts → thousands-separated; otherwise 4 sig figs
    (so count-rate / scaled values still read cleanly)."""
    try:
        if float(v).is_integer():
            return f"{int(v):,}"
    except (TypeError, ValueError, OverflowError):
        pass
    return f"{v:,.4g}"


def hsep():
    f = QFrame(); f.setObjectName("separator"); f.setFrameShape(QFrame.NoFrame)
    return f


def header(text):
    h = QLabel(text); h.setObjectName("section_header")
    return h



def stat_row(key, accent=None):
    """Return (row_widget, value_label) so callers can update the value."""
    accent = accent or T.TEXT_PRIMARY
    row = QWidget()
    lay = QHBoxLayout(row); lay.setContentsMargins(2, 2, 2, 2)
    k = QLabel(key); k.setObjectName("stat_key")
    v = QLabel("—")
    v.setStyleSheet(f"color: {accent}; font-size: 15px; font-weight: 700; "
                    f"font-family: {T.MONO_FAMILY};")
    v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    lay.addWidget(k); lay.addStretch(1); lay.addWidget(v)
    return row, v


def labeled_row(label, widget, apply_btn=False):
    row = QWidget()
    rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(6)
    rl.addWidget(QLabel(label)); rl.addStretch(1); rl.addWidget(widget)
    btn = None
    if apply_btn:
        btn = QPushButton("Apply"); btn.setObjectName("mini_btn")
        btn.setCursor(Qt.PointingHandCursor)
        rl.addWidget(btn)
    return row, btn


def check_row(checkbox, widget):
    """A 'check  [input]' row: toggle the checkbox to apply the option live."""
    row = QWidget()
    rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(6)
    rl.addWidget(checkbox); rl.addStretch(1); rl.addWidget(widget)
    return row


# ── Matplotlib canvas ─────────────────────────────────────────────────────────
class SpectrumCanvas(FigureCanvas):
    cursor_moved = pyqtSignal(float, float)
    point_clicked = pyqtSignal(float, float)   # left-click in manual peak mode
    roi_selected = pyqtSignal(float, float)    # Drag-and-Fit region selected

    def __init__(self):
        # NB: no tight_layout=True here. That installs the *dynamic* layout
        # engine, which re-fits the axes box on every draw — including the
        # SpanSelector's blit-background capture and the post-selection redraw —
        # making the spectrum visibly shift at the start/end of a drag-and-fit.
        # We apply tight layout explicitly instead (see _tighten), only when the
        # content actually changes (_redraw / draw_empty) and on resize.
        self.fig = Figure(figsize=(10, 5), facecolor=T.BG_PLOT)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._spect = None
        self._log = False
        self._peaks = []          # list of (x, y, label)
        self._show_peaks = True
        self._manual = False
        self._snr = None          # (x_array, snr_array, threshold) or None
        self.ax2 = None           # twin axis for the SNR overlay
        self._overlays = []       # list of (x, y, label, color) kept spectra
        self._active_label = None
        self._active_visible = True
        self._xlabel = None       # axis-label overrides (None = derive from spectrum)
        self._ylabel = None
        self._ref_lines = []      # Nuclear Database reference lines: (energy, label, color)
        self.nav_toolbar = None   # set by the page; its view history is reset on redraw
        self._span = None         # Drag-and-Fit ROI selector
        self._roi_active = False
        # Isotope-ID hover: show the likely isotopes for the peak under the cursor
        # as a themed HTML tooltip (a child QLabel, so colours are easy and it
        # needs no canvas redraw).
        self._iso_on = False
        self._iso_info = {}       # {round(peak_x, 3): HTML candidate text}
        self._iso_hover = None    # peak key currently shown (None = nothing)
        self._iso_label = QLabel(self)
        self._iso_label.setObjectName("iso_tooltip")
        self._iso_label.setTextFormat(Qt.RichText)
        self._iso_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._iso_label.setStyleSheet(
            f"QLabel#iso_tooltip {{ background: {T.BG_PANEL}; color: {T.TEXT_PRIMARY};"
            f" border: 1px solid {T.ACCENT_GREEN}; border-radius: 6px; padding: 6px 9px;"
            f" font-family: {T.MONO_FAMILY}; font-size: 13px; }}")
        self._iso_label.hide()
        self.mpl_connect("motion_notify_event", self._on_move)
        self.mpl_connect("button_press_event", self._on_click)

    def _tighten(self):
        """Apply tight layout once, here rather than via the always-on layout
        engine, so plain draws (blit-background capture, post-selection redraw)
        leave the axes box exactly where it was — no shift on drag start/end."""
        try:
            self.fig.tight_layout()
        except Exception:  # noqa: BLE001 — layout is best-effort, never fatal
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._tighten()
        # Only show the placeholder when there is genuinely nothing loaded.
        # draw_empty() nulls self._spect, so calling it unconditionally here
        # destroyed the active spectrum on every resize — and because _redraw()
        # short-circuits when _spect is None, the visibility toggle could not
        # bring it back. Keep the existing artists and just repaint at the new
        # size when a spectrum is present.
        if self._spect is None:
            self.draw_empty()
        else:
            self.draw_idle()

    def draw_empty(self):
        self._spect = None
        self.ax.clear()
        self.ax.set_facecolor(T.BG_PLOT)
        self.ax.text(0.5, 0.72, "NO SPECTRUM LOADED", transform=self.ax.transAxes,
                     ha="center", va="center", fontsize=16, color=T.BORDER,
                     fontweight="bold")
        # wara logo, faded, centred just below the placeholder text.
        try:
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox
            arr = _logo_array()
            zoom = 180.0 / arr.shape[0]          # ~180-px tall, DPI-independent
            box = OffsetImage(arr, zoom=zoom, alpha=0.7)
            self.ax.add_artist(AnnotationBbox(
                box, (0.5, 0.40), xycoords="axes fraction", frameon=False))
        except Exception:                        # noqa: BLE001 — logo is optional
            pass
        self.ax.set_xticks([]); self.ax.set_yticks([])
        for sp in self.ax.spines.values():
            sp.set_color(T.BORDER)
        self._tighten()
        self.draw_idle()

    def plot_spectrum(self, spect, log_y=False, label=None, xlabel=None, ylabel=None,
                      keep_zoom=False):
        self._spect = spect
        self._log = log_y
        self._active_label = label
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._redraw(keep_zoom=keep_zoom)

    def set_overlays(self, items):
        """items: list of (x, y, label, color) for kept/overlaid spectra."""
        self._overlays = list(items)
        self._redraw()

    def set_active_visible(self, visible):
        self._active_visible = visible
        self._redraw()

    def set_peaks(self, peaks):
        self._peaks = list(peaks)
        self._redraw()

    def add_peak(self, x, y, label):
        self._peaks.append((x, y, label))
        self._redraw()

    def set_show_peaks(self, show):
        self._show_peaks = show
        self._redraw()

    def peak_xs(self):
        """X positions (energies, if calibrated) of the current peak markers."""
        return [p[0] for p in self._peaks]

    def set_manual(self, on):
        self._manual = on

    # ── Isotope-ID hover ──────────────────────────────────────────────────────
    def set_isotope_id(self, on, info=None):
        """Enable the hover-to-identify overlay. ``info`` maps ``round(peak_x, 3)``
        to the HTML candidate text shown when that peak is hovered."""
        self._iso_on = bool(on)
        self._iso_info = dict(info) if info else {}
        if not self._iso_on:
            self._iso_hover = None
            self._iso_label.hide()

    def _update_iso_hover(self, event):
        """Show/hide the isotope-ID annotation as the cursor nears a peak line."""
        if (not self._iso_on or not self._show_peaks or not self._peaks
                or event.inaxes != self.ax or event.x is None):
            self._hide_iso_annot()
            return
        # Nearest peak in screen-x (peaks are vertical lines, so y is ignored).
        best_x, best_y, best_d = None, 0.0, 1e9
        for px, py, _lbl in self._peaks:
            disp = self.ax.transData.transform((px, 0))[0]
            d = abs(disp - event.x)
            if d < best_d:
                best_d, best_x, best_y = d, px, py
        if best_x is None or best_d > 12:           # not hovering a peak
            self._hide_iso_annot()
            return
        key = round(best_x, 3)
        if key == self._iso_hover:                  # already showing this one
            return
        text = self._iso_info.get(key)
        if not text:
            self._hide_iso_annot()
            return
        self._iso_hover = key
        self._show_iso_annot(best_x, best_y, text)

    def _show_iso_annot(self, px, py, html):
        lbl = self._iso_label
        lbl.setText(html)
        lbl.adjustSize()
        # Place the tooltip just above-right of the peak. transData gives display
        # pixels (origin bottom-left, physical); convert to the widget's logical
        # top-left coordinates and clamp inside the canvas.
        disp = self.ax.transData.transform((px, py))
        dpr = self.devicePixelRatioF() or 1.0
        x = disp[0] / dpr + 14
        y = (self.figure.bbox.height - disp[1]) / dpr - lbl.height() - 14
        x = int(min(max(0, x), max(0, self.width() - lbl.width())))
        y = int(min(max(0, y), max(0, self.height() - lbl.height())))
        lbl.move(x, y)
        lbl.show()
        lbl.raise_()

    def _hide_iso_annot(self):
        if self._iso_hover is None:
            return
        self._iso_hover = None
        self._iso_label.hide()

    # ── Drag-and-Fit ROI selection ───────────────────────────────────────────
    def enable_roi(self):
        self._roi_active = True
        self._make_span()

    def disable_roi(self):
        self._roi_active = False
        if self._span is not None:
            try:
                self._span.disconnect_events()
                self._span.set_visible(False)
            except Exception:  # noqa: BLE001
                pass
            self._span = None
        self.draw_idle()

    def _make_span(self):
        if self._span is not None:
            try:
                self._span.disconnect_events()
            except Exception:  # noqa: BLE001
                pass
        # useblit=False: with the dynamic tight_layout engine removed (see the
        # Figure() comment), full redraws keep a stable axes box, so dragging an
        # edge no longer makes the opposite one wobble. Staying off the blit path
        # also means the span is always rendered into the Agg buffer — no stale
        # ghost when the always-on-top fit window overlaps the plot, and no
        # blit↔full-draw "refresh" shift at the start/end of a drag.
        self._span = SpanSelector(
            self.ax, self._on_span, "horizontal", interactive=True, useblit=False,
            props=dict(alpha=0.18, facecolor=T.LOGO_GREEN))

    def _on_span(self, xmin, xmax):
        if xmax - xmin > 0:
            self.roi_selected.emit(float(xmin), float(xmax))

    def set_span_extents(self, x0, x1):
        """Move the ROI span programmatically (e.g. from the fit window)."""
        if self._span is not None:
            try:
                self._span.extents = (x0, x1)
                self.draw_idle()
            except Exception:  # noqa: BLE001
                pass

    def add_ref_lines(self, lines, color):
        """Add Nuclear Database reference lines (each (energy, label)) in color."""
        existing = {(round(e, 3), lbl) for e, lbl, _ in self._ref_lines}
        for e, lbl in lines:
            if (round(e, 3), lbl) not in existing:
                self._ref_lines.append((e, lbl, color))
        self._redraw()

    def clear_ref_lines(self):
        self._ref_lines = []
        self._redraw()

    def set_snr(self, x, snr, threshold=None):
        self._snr = (x, snr, threshold)
        self._redraw()

    def clear_snr(self):
        self._snr = None
        self._redraw()

    def reset_view(self):
        self.ax.autoscale()
        if self.ax2 is not None:
            self.ax2.relim(); self.ax2.autoscale(axis="y"); self.ax2.set_ylim(bottom=0)
        self.draw_idle()

    def _redraw(self, keep_zoom=True):
        if self._spect is None:
            self.draw_empty()
            return
        ax = self.ax
        xlim, ylim = (ax.get_xlim(), ax.get_ylim()) if keep_zoom else (None, None)
        ax.clear()
        # Peak positions may have moved; hide the isotope-ID tooltip until the
        # next hover repositions it.
        self._iso_hover = None
        if getattr(self, "_iso_label", None) is not None:
            self._iso_label.hide()
        ax.set_facecolor(T.BG_PLOT)
        ax.patch.set_visible(True)
        # Drop any previous SNR twin axis before (maybe) recreating it.
        if self.ax2 is not None:
            self.ax2.remove()
            self.ax2 = None
        # Kept spectra, drawn behind the active one (same opacity for contrast).
        for ox, oy, olabel, ocolor in self._overlays:
            ax.step(ox, oy, where="mid", color=ocolor, linewidth=0.9,
                    alpha=0.95, label=olabel)
        x, y = self._spect.x, self._spect.counts
        if self._active_visible:
            ax.fill_between(x, y, alpha=0.12, color=T.ACCENT_CYAN, linewidth=0)
            ax.step(x, y, where="mid", color=T.ACCENT_CYAN, linewidth=0.9, alpha=0.95,
                    label=self._active_label)
        # Only switch to log if there is positive data, else matplotlib warns.
        ax.set_yscale("log" if (self._log and (y > 0).any()) else "linear")
        y_label = self._ylabel or {"Cts": "Counts", "CPS": "Counts/second"}.get(
            self._spect.y_label, self._spect.y_label)
        x_label = self._xlabel or self._spect.x_units
        ax.set_xlabel(x_label, color=T.TEXT_DIM, fontsize=12, fontweight="bold")
        ax.set_ylabel(y_label, color=T.TEXT_DIM, fontsize=12, fontweight="bold")
        for sp in ax.spines.values():
            sp.set_color(T.BORDER)
        ax.tick_params(colors=T.TEXT_DIM, which="both", length=3)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which="major", color=T.GRID, linewidth=0.6, alpha=0.9)
        if self._show_peaks and self._peaks and self._active_visible:
            for px, py, label in self._peaks:
                ax.axvline(px, color=T.ACCENT_AMBER, linewidth=1.4, linestyle="--", alpha=0.7)
                # Offset the label a fixed distance above the peak (in points, so
                # the gap is consistent on both linear and log scales). An opaque
                # background box hides the dashed line behind the digits.
                ax.annotate(label, xy=(px, py), xytext=(0, 18),
                            textcoords="offset points", color=T.ACCENT_AMBER,
                            fontsize=12, ha="center", va="bottom", rotation=90,
                            zorder=5,
                            bbox=dict(boxstyle="round,pad=0.15", facecolor=T.BG_PLOT,
                                      edgecolor="none"))
        # Show a legend whenever any labeled spectrum is on screen (active or
        # overlays), so the spectrum's name/label is always visible.
        _, labels = ax.get_legend_handles_labels()
        if labels:
            leg = ax.legend(loc="upper right", fontsize=12, framealpha=0.92)
            if leg is not None:
                leg.get_frame().set_facecolor(T.BG_PANEL)
                leg.get_frame().set_edgecolor(T.BORDER)
                for txt in leg.get_texts():
                    txt.set_color(T.TEXT_PRIMARY)
        # Nuclear Database reference lines (labels anchored at the top).
        for energy, label, color in self._ref_lines:
            ax.axvline(energy, color=color, linewidth=1.0, linestyle="-", alpha=0.7)
            if label:
                ax.annotate(label, xy=(energy, 1.0), xycoords=("data", "axes fraction"),
                            xytext=(0, -3), textcoords="offset points", color=color,
                            fontsize=12, ha="center", va="top", rotation=90, zorder=6,
                            bbox=dict(boxstyle="round,pad=0.15", facecolor=T.BG_PLOT,
                                      edgecolor="none"))
        if self._snr is not None:
            self._draw_snr_overlay()
        if self.nav_toolbar is not None:
            self.nav_toolbar.update()
            self.nav_toolbar.push_current()
        if xlim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        if self._roi_active:
            self._make_span()
        self._tighten()
        self.draw_idle()

    def _draw_snr_overlay(self):
        xs, snr, threshold = self._snr
        self.ax2 = self.ax.twinx()
        self.ax2.plot(xs, snr, color=T.SNR_PURPLE, linewidth=0.9, alpha=0.85)
        self.ax2.set_ylabel("SNR", color=T.SNR_PURPLE, fontsize=12, fontweight="bold")
        self.ax2.tick_params(axis="y", colors=T.SNR_PURPLE, length=3)
        self.ax2.set_ylim(bottom=0)
        for sp in self.ax2.spines.values():
            sp.set_color(T.BORDER)
        if threshold is not None:
            self.ax2.axhline(threshold, color=T.SNR_PURPLE, linewidth=0.7,
                             linestyle=":", alpha=0.6)
        # Keep the main axis on top (transparent) so mouse events and the
        # cursor readout stay in counts coordinates, not SNR.
        self.ax.set_zorder(self.ax2.get_zorder() + 1)
        self.ax.patch.set_visible(False)
        self.ax2.patch.set_visible(False)

    def _on_move(self, event):
        if event.inaxes == self.ax:
            self.cursor_moved.emit(event.xdata or 0.0, event.ydata or 0.0)
        if self._iso_on:
            self._update_iso_hover(event)

    def _on_click(self, event):
        if (self._manual and event.inaxes == self.ax
                and event.button == 1 and event.xdata is not None):
            self.point_clicked.emit(event.xdata, event.ydata or 0.0)


# ── Peak-find inline sub-panel (mock for now) ────────────────────────────────
class PeakFindPanel(QFrame):
    """Collapsible options revealed under Auto-Find Peaks. Mirrors wara's
    Find-Peaks dialog plus an 'Add peaks manually' toggle. Not yet wired."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("subpanel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10); lay.setSpacing(6)

        def field(label, default=""):
            e = QLineEdit(default); e.setMaximumWidth(86)
            row, _ = labeled_row(label, e)
            lay.addWidget(row)
            return e

        # Detector preset sits above SNR: choosing it fills SNR / Ref. channel /
        # Ref. FWHM, so it reads top-down as "pick a detector, then tune".
        self.detector = ComboBox()
        self.detector.addItems(["HPGe", "LaBr/CeBr", "NaI", "Plastic Scint."])
        self.detector.setCurrentText("LaBr/CeBr")
        self.detector.setMaximumWidth(150)
        row, _ = labeled_row("Detector", self.detector)
        lay.addWidget(row)

        lay.addWidget(hsep())

        self.snr = field("SNR >", "3")
        self.ref_ch = field("Ref. channel")
        self.ref_ch.setPlaceholderText("e.g. 420")
        self.ref_fwhm = field("Ref. FWHM")
        self.ref_fwhm.setPlaceholderText("e.g. 12")

        xr = QWidget(); xl = QHBoxLayout(xr); xl.setContentsMargins(0, 0, 0, 0); xl.setSpacing(5)
        xl.addWidget(QLabel("Xrange")); xl.addStretch(1)
        self.x0 = QLineEdit(); self.x0.setPlaceholderText("x0"); self.x0.setMaximumWidth(62)
        self.x1 = QLineEdit(); self.x1.setPlaceholderText("x1"); self.x1.setMaximumWidth(62)
        xl.addWidget(self.x0); xl.addWidget(self.x1)
        lay.addWidget(xr)

        lay.addWidget(hsep())
        self.cb_kernel = QCheckBox("Kernel Method"); self.cb_kernel.setChecked(True)
        self.cb_snr = QCheckBox("Plot SNR")
        for cb in (self.cb_kernel, self.cb_snr):
            lay.addWidget(cb)

        self.btn_find = QPushButton("Find Peaks"); self.btn_find.setObjectName("find_btn")
        self.btn_find.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_find)
        self.btn_clear = QPushButton("Clear Peaks"); self.btn_clear.setObjectName("danger_btn")
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.btn_clear)

        # ── Hover help ───────────────────────────────────────────────
        self.snr.setToolTip("Minimum signal-to-noise ratio — lower values find weaker peaks")
        self.ref_ch.setToolTip(
            "Channel or energy where the detector resolution is known.\n"
            "Used to estimate peak widths across the spectrum.")
        self.ref_fwhm.setToolTip(
            "FWHM (in channels or energy units) at the reference point.\n"
            "Approximate values are fine — the algorithm is forgiving.")
        self.x0.setToolTip("Optional lower bound to limit the search range")
        self.x1.setToolTip("Optional upper bound to limit the search range")
        self.detector.setToolTip(
            "Preset detector parameters — fills SNR, Ref. channel, and Ref. FWHM.\n"
            "Adjust values as needed for your specific setup.")
        self.cb_kernel.setToolTip(
            "Derivative-based peak detection — more robust but slower.\n"
            "Needs < 9000 channels, or set an Xrange to limit the region.")
        self.cb_snr.setToolTip("Overlay the SNR curve on a second axis")
        self.btn_find.setToolTip("Run the peak search with the settings above")
        self.btn_clear.setToolTip("Remove all peak markers from the plot")


class AddSubtractPanel(QFrame):
    """Inline panel under 'Add / Subtract': combine the loaded spectra. The
    active spectrum is the reference; overlays are the operands."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("subpanel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10); lay.setSpacing(6)
        hint = QLabel("Active is the reference; overlays are the operands.")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.cb_scale_lt = QCheckBox("Scale by live time")
        self.cb_scale_lt.setToolTip(
            "Before subtracting, scale each overlay by "
            "active.livetime / overlay.livetime")
        lay.addWidget(self.cb_scale_lt)
        self.btn_add = QPushButton("Add All"); self.btn_add.setObjectName("primary_btn")
        self.btn_add.setToolTip("Sum the active spectrum and all overlays into a new spectrum")
        self.btn_add.setCursor(Qt.PointingHandCursor); lay.addWidget(self.btn_add)
        self.btn_sub = QPushButton("Subtract Overlays")
        self.btn_sub.setObjectName("action_btn")
        self.btn_sub.setToolTip("Subtract the overlays from the active spectrum")
        self.btn_sub.setCursor(Qt.PointingHandCursor); lay.addWidget(self.btn_sub)


# ── Spectrum options column ──────────────────────────────────────────────────
class SpectrumOptions(QScrollArea):
    """Scrollable options for the Spectrum tab. Exposes its controls and stat
    value labels as attributes so the main window can wire them."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(9, 12, 9, 12); lay.setSpacing(6)

        title = QLabel("SPECTRUM"); title.setObjectName("opt_title")
        lay.addWidget(title); lay.addWidget(hsep())

        # Live readouts. The cursor position is shown below the plot (see
        # SpectrumPage), so only the static stats live here.
        lay.addWidget(header("STATS"))
        r, self.val_channels = stat_row("Channels", T.ACCENT_CYAN); lay.addWidget(r)
        r, self.val_max = stat_row("Max Counts", T.ACCENT_AMBER); lay.addWidget(r)
        r, self.val_total = stat_row("Total Counts"); lay.addWidget(r)

        # Display
        lay.addWidget(hsep()); lay.addWidget(header("DISPLAY"))
        self.cb_log = QCheckBox("Log Y-axis"); lay.addWidget(self.cb_log)
        self.cb_cr = QCheckBox("Set as count rate"); lay.addWidget(self.cb_cr)
        self.cb_keep = QCheckBox("Keep spectrum visible")
        self.cb_keep.setToolTip("Overlay newly loaded spectra on top of the current one")
        lay.addWidget(self.cb_keep)

        # Loaded-spectra list (active + overlays), populated by the main window.
        lay.addWidget(header("SPECTRA"))
        self.spectra_list = QVBoxLayout()
        self.spectra_list.setContentsMargins(2, 0, 2, 0)
        self.spectra_list.setSpacing(3)
        _spectra_box = QWidget(); _spectra_box.setLayout(self.spectra_list)
        lay.addWidget(_spectra_box)

        # Peaks & fitting (used most often, kept high up)
        lay.addWidget(hsep()); lay.addWidget(header("PEAKS & FITTING"))
        # Manual peak placement is its own way of getting peaks (alongside
        # Auto-Find), so it lives here on top of the action buttons rather than
        # buried in the Auto-Find options.
        self.cb_manual = QCheckBox("Add peaks manually")
        self.cb_manual.setToolTip("Click on the plot to place peaks by hand")
        lay.addWidget(self.cb_manual)
        self.cb_peaks = QCheckBox("Show Peaks"); self.cb_peaks.setChecked(True)
        self.cb_peaks.setToolTip("Show or hide the peak markers on the plot")
        lay.addWidget(self.cb_peaks)
        self.cb_isotope_id = QCheckBox("Isotope ID")
        self.cb_isotope_id.setToolTip(
            "Hover a found peak to see the most likely isotope from each nuclear "
            "database, with a probability.\nNeeds a calibrated spectrum with found peaks.")
        lay.addWidget(self.cb_isotope_id)
        self.btn_fit = QPushButton("Drag and Fit"); self.btn_fit.setObjectName("fit_btn")
        self.btn_fit.setCheckable(True)
        self.btn_fit.setToolTip(
            "Click-drag over a region on the plot to fit peaks.\n"
            "Peaks must be found first (auto-find or manual).")
        self.btn_fit.setCursor(Qt.PointingHandCursor); lay.addWidget(self.btn_fit)

        self.btn_find = QPushButton("Auto-Find Peaks  ▾"); self.btn_find.setObjectName("primary_btn")
        self.btn_find.setToolTip("Expand to configure detector parameters and run automatic peak search")
        self.btn_find.setCursor(Qt.PointingHandCursor); lay.addWidget(self.btn_find)
        self.pf_panel = PeakFindPanel(); self.pf_panel.setVisible(False)
        lay.addWidget(self.pf_panel)
        self.btn_find.clicked.connect(self._toggle_pf)

        self.btn_iso = QPushButton("Nuclear Database"); self.btn_iso.setObjectName("yellow_btn")
        self.btn_iso.setToolTip("Look up gamma-ray energies by isotope to identify peaks")
        self.btn_iso.setCursor(Qt.PointingHandCursor); lay.addWidget(self.btn_iso)

        # Spectrum actions
        lay.addWidget(hsep()); lay.addWidget(header("SPECTRUM"))
        self.btn_reset = QPushButton("Reset Spectrum"); self.btn_reset.setObjectName("action_btn")
        self.btn_reset.setToolTip("Undo all customize edits and restore the originally loaded spectrum")
        # A little red — softer than danger_btn — to flag it as a discard action.
        self.btn_reset.setStyleSheet(
            "QPushButton#action_btn { background-color: #2e1b22; color: #ffb3c1;"
            " border: 2px solid #b05a6e; }"
            "QPushButton#action_btn:hover { background-color: #4d1a2a;"
            f" border: 2px solid {T.ACCENT_RED}; color: #ff8aa0; }}")
        self.btn_remcal = QPushButton("Remove Calibration"); self.btn_remcal.setObjectName("action_btn")
        self.btn_remcal.setToolTip("Strip the energy axis and revert to channel numbers")
        self.btn_addsub = QPushButton("Add / Subtract  ▾"); self.btn_addsub.setObjectName("action_btn")
        self.btn_addsub.setToolTip("Combine the active spectrum with overlays (e.g. background subtraction)")
        for b in (self.btn_reset, self.btn_remcal, self.btn_addsub):
            b.setCursor(Qt.PointingHandCursor); lay.addWidget(b)
        self.addsub_panel = AddSubtractPanel(); self.addsub_panel.setVisible(False)
        lay.addWidget(self.addsub_panel)
        self.btn_addsub.clicked.connect(self._toggle_addsub)

        # Customize — each option applies live while its checkbox is ticked
        lay.addWidget(hsep()); lay.addWidget(header("CUSTOMIZE"))
        self.cb_smooth = QCheckBox("Smooth")
        self.cb_smooth.setToolTip("Moving-average filter — adjust the window size to the right")
        self.smooth_spin = SpinBox(); self.smooth_spin.setRange(1, 99999)
        self.smooth_spin.setSingleStep(1); self.smooth_spin.setValue(4)
        self.smooth_spin.setFixedWidth(84)
        self.smooth_spin.setToolTip("Smoothing window size (number of channels)")
        lay.addWidget(check_row(self.cb_smooth, self.smooth_spin))

        self.cb_rebin = QCheckBox("Rebin ×2")
        self.cb_rebin.setToolTip("Combine adjacent channels to reduce noise")
        self.rebin = SpinBox(); self.rebin.setRange(2, 1024); self.rebin.setSingleStep(2)
        self.rebin.setValue(2); self.rebin.setFixedWidth(84)
        self.rebin.setToolTip("Number of channels to merge")
        lay.addWidget(check_row(self.cb_rebin, self.rebin))

        self.cb_shift = QCheckBox("Shift")
        self.cb_shift.setToolTip("Translate the spectrum along the x-axis")
        self.shift_box = DoubleSpinBox(); self.shift_box.setRange(-1e6, 1e6)
        self.shift_box.setDecimals(2); self.shift_box.setFixedWidth(84)
        self.shift_box.setToolTip("Shift amount in channels or energy units")
        lay.addWidget(check_row(self.cb_shift, self.shift_box))

        self.cb_yconst = QCheckBox("y × const")
        self.cb_yconst.setToolTip("Scale all counts by a constant factor")
        self.yconst = DoubleSpinBox(); self.yconst.setRange(-1e9, 1e9)
        self.yconst.setDecimals(3); self.yconst.setValue(1.0); self.yconst.setFixedWidth(84)
        self.yconst.setToolTip("Multiplicative factor for counts")
        lay.addWidget(check_row(self.cb_yconst, self.yconst))

        self.cb_xconst = QCheckBox("x × const")
        self.cb_xconst.setToolTip("Scale the x-axis by a constant factor")
        self.xconst = DoubleSpinBox(); self.xconst.setRange(-1e9, 1e9)
        self.xconst.setDecimals(3); self.xconst.setValue(1.0); self.xconst.setFixedWidth(84)
        self.xconst.setToolTip("Multiplicative factor for the x-axis")
        lay.addWidget(check_row(self.cb_xconst, self.xconst))

        self.btn_labels = QPushButton("Axis & Legend…"); self.btn_labels.setObjectName("action_btn")
        self.btn_labels.setToolTip("Set custom axis labels and legend text")
        self.btn_labels.setCursor(Qt.PointingHandCursor); lay.addWidget(self.btn_labels)

        # ── Hover help ───────────────────────────────────────────────
        self.cb_log.setToolTip("Toggle a logarithmic Y-axis")
        self.cb_cr.setToolTip("Show count rate (counts ÷ live time); needs a live time in the file")
        self.btn_find.setToolTip("Show / hide the automatic peak-finder options")
        self.btn_iso.setToolTip("Browse gamma-ray line databases and isotopic abundances; overlay lines on the spectrum")
        self.btn_remcal.setToolTip("Drop the energy calibration and show the channel axis (Reset Spectrum restores it)")
        self.btn_addsub.setToolTip("Add the loaded spectra together, or subtract overlays from the active one")
        self.cb_smooth.setToolTip("Moving-average smooth, re-applied from the original spectrum")
        self.smooth_spin.setToolTip("Smoothing window, in channels")
        self.cb_rebin.setToolTip("Combine adjacent channels by a factor (multiples of two)")
        self.rebin.setToolTip("Rebin factor — a multiple of two")
        self.cb_shift.setToolTip("Shift the spectrum along its x-axis (energy if calibrated, else channels)")
        self.cb_yconst.setToolTip("Multiply the counts by a constant")
        self.cb_xconst.setToolTip("Multiply the x-axis values by a constant")
        self.btn_labels.setToolTip("Edit axis labels, legend and description; view the spectrum's metadata")

        lay.addStretch(1)
        self.setWidget(inner)

    def _toggle_pf(self):
        show = not self.pf_panel.isVisible()
        self.pf_panel.setVisible(show)
        self.btn_find.setText("Auto-Find Peaks  ▴" if show else "Auto-Find Peaks  ▾")

    def _toggle_addsub(self):
        show = not self.addsub_panel.isVisible()
        self.addsub_panel.setVisible(show)
        self.btn_addsub.setText("Add / Subtract  ▴" if show else "Add / Subtract  ▾")

    def set_stats(self, channels, max_counts, total):
        self.val_channels.setText(f"{channels:,}")
        self.val_max.setText(fmt_count(max_counts))
        self.val_total.setText(fmt_count(total))

    def clear_stats(self):
        for v in (self.val_channels, self.val_max, self.val_total):
            v.setText("—")


class PlaceholderOptions(QWidget):
    def __init__(self, name):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(6)
        title = QLabel(name.upper()); title.setObjectName("opt_title")
        lay.addWidget(title); lay.addWidget(hsep())
        hint = QLabel(f"{name} options\nwill live here")
        hint.setObjectName("stat_key"); hint.setWordWrap(True)
        lay.addWidget(hint); lay.addStretch(1)


# ── Pages (plot column) ──────────────────────────────────────────────────────
class PlaceholderPage(QWidget):
    def __init__(self, name):
        super().__init__()
        self.setObjectName("content")
        lay = QVBoxLayout(self)
        lab = QLabel(f"{name.upper()} PLOT AREA")
        lab.setObjectName("placeholder"); lab.setAlignment(Qt.AlignCenter)
        lay.addWidget(lab)


class ComingSoonPage(QWidget):
    """A deliberate, friendly placeholder for a section that is announced but
    not yet built (e.g. Planetary). Reads as intentional rather than unfinished."""

    def __init__(self, name, subtitle="This section is on the way."):
        super().__init__()
        self.setObjectName("content")
        lay = QVBoxLayout(self)
        lay.addStretch(1)
        title = QLabel(name.upper())
        title.setObjectName("placeholder"); title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)
        tag = QLabel("COMING SOON")
        tag.setObjectName("section_header"); tag.setAlignment(Qt.AlignCenter)
        lay.addWidget(tag)
        sub = QLabel(subtitle)
        sub.setObjectName("stat_key"); sub.setAlignment(Qt.AlignCenter)
        sub.setWordWrap(True)
        lay.addWidget(sub)
        lay.addStretch(1)
