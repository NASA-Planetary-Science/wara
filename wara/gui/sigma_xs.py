"""Sigma cross-section calculator — interactive pop-out for the API tab.

Opens from the API tab's "Sigma…" button (see :meth:`ApiController._open_sigma`).
Walks the user through the seven steps of an API neutron-source cross-section
measurement, recomputing the cross section *live* as any parameter changes:

  1. **Files**            – load sample / background / profile / flat-field runs,
     and set each one's dt / energy offset against the overlaid spectra.
  2. **Corrections**      – X-stretch (1.27), alpha-energy threshold, non-α.
  3. **Cuts**             – x / y / t / energy cuts on sample + background.
  4. **Background & Net** – scale the background (fit C or override) → net spectrum.
  5. **Fluence / Nt**     – geometry-free beam-profile fit → alpha fraction →
     neutron fluence; Nt.  The profile is flattened by solving jointly for the
     square target and a smooth gain field (:mod:`sigma.flux_profile_uniform`),
     so no detector distances / 1-over-r-squared term are needed and any run or
     channel can be used.
  6. **Cross section**    – σ (mb) with a Monte-Carlo uncertainty band.

The heavy lifting lives in the closed-source :mod:`sigma` package
(``sigma.cross_section``); this module is only the interactive Qt layer.  It is
imported lazily by the controller, so wara never depends on ``sigma``.

The dialog mirrors :class:`SelectionsDialog` in :mod:`wara.gui.api`: themed
via :mod:`wara.gui.theme`, the same ``header`` / ``hsep`` / ``labeled_row``
helpers, and embedded matplotlib canvases with navigation toolbars.
"""
from __future__ import annotations

import traceback

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

from PyQt5.QtWidgets import (
    QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLineEdit, QCheckBox, QComboBox, QTabWidget, QSizePolicy, QMessageBox,
)
from PyQt5.QtCore import Qt, QSize, QTimer

from wara import read_parquet_api, apicalc as api
from wara import spectrum as sp
from wara import peaksearch as ps

from . import theme as T
from .widgets import hsep, header, labeled_row, stat_row, fmt_count

# The engine.  Imported at module load — but this module itself is only imported
# *after* the controller has confirmed ``sigma`` is importable, so this is safe.
from sigma import cross_section as cs
from sigma import bkg_scale
# Geometry-free alpha fraction: the beam profile is flattened by a joint
# square+gain fit instead of a hand-measured 1/r² term, so tab 5 works on any
# run / channel without detector-position input.
from sigma import flux_profile_uniform as fpu

PLOT_BG = T.BG_PLOT

# Detector presets for the net-spectrum peak search: (min_snr, ref_E keV, ref_FWHM keV).
# Mirrors the Spectrum tab's presets, in energy units for the calibrated net spectrum.
DETECTOR_PRESETS = {
    "HPGe":           ("5", "420", "3"),
    "LaBr/CeBr":      ("5", "420", "12"),
    "NaI":            ("5", "420", "15"),
    "Plastic Scint.": ("5", "420", "20"),
}

# A "file role" → (default channel, is_flat) map used by the loaders.
ROLES = ["sample", "background", "profile", "flat"]
ROLE_LABELS = {
    "sample": "Sample", "background": "Background",
    "profile": "Profile", "flat": "Flat field",
}


# Preference order when an energy column is chosen automatically.  ``energy_cal``
# first: when a run has been calibrated that is the keV axis the photopeak fit
# and the profile cut both want.
EKEY_PREFERENCE = ("energy_cal", "energy", "energy_orig")


def auto_ekey(df, preferred=None):
    """The energy column to use for *df*.

    Returns *preferred* when the file actually has it, otherwise the first of
    :data:`EKEY_PREFERENCE` present.  ``None`` if the file has no energy column.
    """
    if df is None:
        return None
    if preferred and preferred in df.columns:
        return preferred
    for k in EKEY_PREFERENCE:
        if k in df.columns:
            return k
    return None


def _num(text, default=None):
    """Parse a float from a line edit, returning *default* on blank/garbage."""
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Per-file loader row
# ─────────────────────────────────────────────────────────────────────────────
class _FileRow(QWidget):
    """A 'date / run / channel  [Load]' row for one of the four datasets, plus
    its dt / energy offsets and a status line.  The loaded (raw) dataframe and
    its alpha-trigger count / live time are kept on the row for the engine.

    The offsets live here, next to the Files-tab overlay plots, because that is
    where you can see what they do: nudge one and the two spectra slide over
    each other until the peaks line up.
    """

    def __init__(self, role, on_loaded, on_offset_changed=None):
        super().__init__()
        self.role = role
        self._on_loaded = on_loaded
        self._on_offset_changed = on_offset_changed
        self.df = None            # raw dataframe (X2/Y2 reconstructed, dt in ns)
        self.alphas = None        # total alpha triggers (ch 9)
        self.t_total = None       # live time (s)
        self.date = self.runnr = self.ch = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(4)

        title = QLabel(ROLE_LABELS[role]); title.setObjectName("section_header")
        lay.addWidget(title)

        row = QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(6)
        self.ed_date = QLineEdit(); self.ed_date.setPlaceholderText("YYYY-MM-DD")
        self.ed_run = QLineEdit(); self.ed_run.setPlaceholderText("run")
        self.ed_run.setFixedWidth(58)
        self.ed_ch = QLineEdit(); self.ed_ch.setPlaceholderText("ch")
        self.ed_ch.setFixedWidth(44)
        self.btn = QPushButton("Load"); self.btn.setObjectName("open_btn")
        self.btn.setCursor(Qt.PointingHandCursor); self.btn.setFixedWidth(70)
        row.addWidget(self.ed_date, 1)
        row.addWidget(self.ed_run, 0)
        row.addWidget(self.ed_ch, 0)
        row.addWidget(self.btn, 0)
        lay.addLayout(row)

        # ── offsets: added to this dataset's dt / energy axes ─────────────
        # The flat field has neither a time nor an energy axis in play (it is
        # only ever histogrammed in X-Y), so it gets no offset row.
        self.ed_dt_off = self.ed_e_off = None
        if role != "flat":
            orow = QHBoxLayout()
            orow.setContentsMargins(0, 0, 0, 0); orow.setSpacing(6)
            self.ed_dt_off = QLineEdit("0"); self.ed_dt_off.setFixedWidth(56)
            self.ed_dt_off.setToolTip(
                f"Constant offset added to the {ROLE_LABELS[role].lower()} dt "
                f"axis (ns).  Watch the time spectra below to line the prompt "
                f"peaks up.")
            self.ed_e_off = QLineEdit("0"); self.ed_e_off.setFixedWidth(56)
            self.ed_e_off.setToolTip(
                f"Constant offset added to the {ROLE_LABELS[role].lower()} "
                f"energy axis (keV).  Watch the energy spectra below to line "
                f"known lines up.")
            for lbl, ed in (("dt +", self.ed_dt_off), ("E +", self.ed_e_off)):
                cap = QLabel(lbl); cap.setObjectName("stat_key")
                orow.addWidget(cap, 0)
                orow.addWidget(ed, 0)
                ed.editingFinished.connect(self._offset_edited)
            orow.addWidget(QLabel("ns / keV", objectName="stat_key"), 0)
            orow.addStretch(1)
            lay.addLayout(orow)

        self.lbl = QLabel("not loaded"); self.lbl.setObjectName("stat_key")
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)

        self.btn.clicked.connect(self._load)

    # ── offsets ───────────────────────────────────────────────────────────
    @property
    def dt_offset(self):
        return _num(self.ed_dt_off.text(), 0.0) if self.ed_dt_off else 0.0

    @property
    def e_offset(self):
        return _num(self.ed_e_off.text(), 0.0) if self.ed_e_off else 0.0

    def _offset_edited(self):
        if self._on_offset_changed is not None:
            self._on_offset_changed(self.role)

    def apply_offsets(self, df):
        """Add this row's dt / energy offsets to a copy-safe dataframe."""
        dt_off, e_off = self.dt_offset, self.e_offset
        if dt_off and "dt" in df.columns:
            df["dt"] = df["dt"] + dt_off
        if e_off:
            # Shift every calibrated-energy axis the file carries, so the offset
            # means the same thing whichever column is selected downstream.
            # alpha_energy / energy_ch9 are the alpha detector's — left alone.
            for col in EKEY_PREFERENCE:
                if col in df.columns:
                    df[col] = df[col] + e_off
        return df

    def seed(self, date, runnr, ch):
        # Fill only empty fields, so pre-populated / user-edited values are never
        # clobbered (e.g. by the API-page sample seeding).
        if date and not self.ed_date.text().strip():
            self.ed_date.setText(str(date))
        if runnr not in (None, "") and not self.ed_run.text().strip():
            self.ed_run.setText(str(runnr))
        if ch not in (None, "") and not self.ed_ch.text().strip():
            self.ed_ch.setText(str(ch))

    def _load(self):
        date = self.ed_date.text().strip()
        run_txt = self.ed_run.text().strip()
        ch_txt = self.ed_ch.text().strip()
        if not date or not run_txt:
            self.lbl.setText("enter a date and run number")
            return
        try:
            runnr = int(run_txt)
        except ValueError:
            self.lbl.setText("run number must be an integer")
            return
        flat = (ch_txt == "9") or self.role == "flat"
        try:
            ch = 9 if flat else int(ch_txt)
        except ValueError:
            self.lbl.setText("channel must be an integer")
            return

        self.lbl.setText("loading…")
        # Brighten the button (purple, matching the API tab's Load button) so the
        # click registers before the blocking parquet read freezes the UI; the
        # style is reverted in the finally block once the load finishes.
        self.btn.setStyleSheet(
            f"background-color:#c4a8ff; border-color:#c4a8ff; "
            f"color:{T.BG_DARK}; font-weight:800;")
        self.btn.setEnabled(False)
        self.btn.repaint()
        try:
            df = read_parquet_api.read_parquet_file(
                date=date, runnr=runnr, ch=ch, flat_field=flat)
            if df is None:
                self.lbl.setText(f"no parquet for {date}-{runnr} ch {ch}")
                return
            df = df.copy()
            if not flat:
                df["dt"] = df["dt"] * 1e9       # s → ns
            if ("X2" not in df.columns
                    and {"A", "B", "C", "D"}.issubset(df.columns)):
                df = api.calc_own_pos(df)
            self.df = df
            self.date, self.runnr, self.ch = date, runnr, ch
            # alpha triggers (ch 9) and live time for normalisation.
            try:
                self.alphas = api.get_total_counts(date, runnr, ch=9)
            except Exception:  # noqa: BLE001 — settings files may be absent
                self.alphas = None
            try:
                self.t_total = api.get_total_time(date, runnr, ch=ch, mca=False)
            except Exception:  # noqa: BLE001
                self.t_total = None
            n = df.shape[0]
            extra = ""
            if self.alphas:
                extra += f" · α={fmt_count(self.alphas)}"
            if self.t_total:
                extra += f" · t={self.t_total:.1f}s"
            self.lbl.setText(f"{n:,} events{extra}")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.lbl.setText(f"error: {exc}")
        finally:
            self.btn.setStyleSheet("")
            self.btn.setEnabled(True)
        self._on_loaded(self.role)


# ─────────────────────────────────────────────────────────────────────────────
# Main dialog
# ─────────────────────────────────────────────────────────────────────────────
class SigmaDialog(QDialog):
    """Interactive cross-section calculator."""

    def __init__(self, controller):
        super().__init__(controller.app)
        self.c = controller
        self.setWindowTitle("σ  —  Cross-section calculator")
        self.setStyleSheet(T.STYLESHEET)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.resize(1240, 780)

        # Cached intermediate results between stages (filled by _recompute).
        self._peak_gamma = 0.0
        self._sig_gamma = 0.0
        self._C = 1.0
        self._sig_C = 0.0
        self._fa_result = None
        self._last_result = None
        self._fit_win = None          # interactive photopeak FitWindow
        self._net_spectrum = None     # current net Spectrum (for the fit window/span)
        self._net_span = None         # SpanSelector on the net-spectrum panel
        self._net_search = None       # PeakSearch (fast) on the net spectrum
        self._net_peaks = []          # [(energy, counts)] found peaks to fit
        # Reference resolution for the peak search, set by the detector preset
        # (LaBr/CeBr default).  Not user-visible — the user only picks the detector.
        self._ref_x, self._ref_fwhm = 420.0, 12.0
        self._net_press_x = None      # press x for click-vs-drag discrimination
        self._applied_roi = None      # ROI (keV) of the applied fit, for the panel marker
        self._peak_source = "—"       # "interactive fit" | "mask sum" | "seeded fit"
        self._peak_applied = False    # True once a fit is applied → σ value is sticky
        self._in_recompute = False    # guards the fit-window callback re-entrancy

        # Debounce: coalesce rapid edits into one recompute.
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._recompute)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(6)

        # ── Persistent headline strip (live cross section) ────────────────
        self.headline = QLabel("XS = —")
        self.headline.setObjectName("opt_title")
        self.headline.setAlignment(Qt.AlignCenter)
        self.headline.setStyleSheet(
            f"color:{T.ACCENT_GREEN}; font-size:20px; font-weight:800; "
            f"font-family:{T.MONO_FAMILY}; padding:6px; "
            f"border:1px solid {T.BORDER}; border-radius:6px; "
            f"background:{T.BG_PANEL};")
        root.addWidget(self.headline)

        self.tabs = QTabWidget()
        self._fit_tabbar(self.tabs)
        root.addWidget(self.tabs, 1)

        self.tabs.addTab(self._build_files_tab(), "1 · Files")
        self.tabs.addTab(self._build_corrections_tab(), "2 · Corrections")
        self.tabs.addTab(self._build_cuts_tab(), "3 · Cuts")
        self.tabs.addTab(self._build_bkg_tab(), "4 · Net counts")
        self.tabs.addTab(self._build_fluence_tab(), "5 · Fluence")
        self.tabs.addTab(self._build_xs_tab(), "6 · Cross section")

        self._status_lbl = QLabel("Load the four datasets to begin.")
        self._status_lbl.setObjectName("stat_key")
        root.addWidget(self._status_lbl)

    # ── window placement ───────────────────────────────────────────────────
    def showEvent(self, event):
        """Make sure the window opens fully on-screen.  Centering on a tall main
        window can push this dialog's title bar above the top of the screen
        (then it can't be dragged); clamp it into the available screen area.
        Deferred so the frame margins (title bar height) are known."""
        super().showEvent(event)
        QTimer.singleShot(0, self._clamp_into_screen)

    def _clamp_into_screen(self):
        from PyQt5.QtWidgets import QApplication
        scr = (QApplication.screenAt(self.frameGeometry().center())
               or QApplication.primaryScreen())
        if scr is None:
            return
        avail = scr.availableGeometry()
        # Shrink if the window is larger than the usable screen area.
        if self.width() > avail.width() or self.height() > avail.height():
            self.resize(min(self.width(), avail.width() - 20),
                        min(self.height(), avail.height() - 60))
        frame = self.frameGeometry()           # includes the title bar
        nx = max(avail.left(), min(frame.x(), avail.right() - frame.width() + 1))
        ny = max(avail.top(), min(frame.y(), avail.bottom() - frame.height() + 1))
        if (nx, ny) != (frame.x(), frame.y()):
            # move() positions the client area; shift it by the frame delta so the
            # whole frame (title bar included) lands inside the available area.
            self.move(self.x() + (nx - frame.x()), self.y() + (ny - frame.y()))

    # ── seeding ───────────────────────────────────────────────────────────
    def seed_sample(self, date, runnr, ch):
        """Pre-fill the sample loader from the API tab's current/loaded run."""
        self.rows["sample"].seed(date, runnr, ch)
        # Sensible default companions seen in the FLARE config: background on the
        # same date, profile/flat left for the user.
        if date and not self.rows["background"].ed_date.text():
            self.rows["background"].ed_date.setText(str(date))

    # ── shared helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _fit_tabbar(tabw):
        bar = tabw.tabBar()
        f = bar.font(); f.setPixelSize(14); f.setBold(True)
        bar.setFont(f)
        bar.setElideMode(Qt.ElideNone)
        bar.setExpanding(False)
        bar.setUsesScrollButtons(False)
        tabw.setStyleSheet("QTabBar::tab { padding: 7px 20px; min-width: 96px; }")

    def _make_canvas(self, height=300):
        """Embedded matplotlib canvas + navigation toolbar, themed."""
        fig = Figure(facecolor=PLOT_BG)
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(height)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavToolbar(canvas, self)
        toolbar.setObjectName("plot_toolbar")
        toolbar.setIconSize(QSize(22, 22))
        T.recolor_toolbar_icons(toolbar, T.TEXT_PRIMARY)
        return fig, canvas, toolbar

    @staticmethod
    def _style_ax(ax, xlabel="", ylabel="", yscale="linear"):
        ax.set_facecolor(PLOT_BG)
        ax.set_yscale(yscale)
        if xlabel:
            ax.set_xlabel(xlabel, color=T.TEXT_DIM, fontsize=11, fontweight="bold")
        if ylabel:
            ax.set_ylabel(ylabel, color=T.TEXT_DIM, fontsize=11, fontweight="bold")
        for s in ax.spines.values():
            s.set_color(T.BORDER)
        ax.tick_params(colors=T.TEXT_DIM, which="both", length=3)
        ax.grid(True, color=T.GRID, linewidth=0.6, alpha=0.8)

    def _field(self, default, width=80, tip=""):
        ed = QLineEdit(str(default)); ed.setFixedWidth(width)
        if tip:
            ed.setToolTip(tip)
        # Any edit (Enter or focus-out) schedules a recompute.
        ed.editingFinished.connect(self._schedule)
        return ed

    def _schedule(self, *_):
        """Coalesce edits → a single debounced recompute."""
        self._timer.start()

    def _status(self, msg):
        self._status_lbl.setText(msg)

    # ══════════════════════════════════════════════════════════════════════
    # Tab 1 — Files
    # ══════════════════════════════════════════════════════════════════════
    def _build_files_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(14, 12, 14, 12); lay.setSpacing(10)

        intro = QLabel(
            "Load the four datasets the cross-section measurement needs.  Each "
            "is read from its run's parquet file; the sample is pre-filled from "
            "the run open on the API tab.  Alpha-trigger counts (ch 9) and live "
            "time are read automatically for the fluence normalisation.")
        intro.setObjectName("stat_key"); intro.setWordWrap(True)
        lay.addWidget(intro)
        lay.addWidget(hsep())

        self.rows = {}
        grid = QGridLayout(); grid.setHorizontalSpacing(24); grid.setVerticalSpacing(10)
        for i, role in enumerate(ROLES):
            r = _FileRow(role, self._on_file_loaded, self._on_offset_changed)
            self.rows[role] = r
            grid.addWidget(r, i // 2, i % 2)
        gw = QWidget(); gw.setLayout(grid)
        lay.addWidget(gw)

        # The loaders start blank; only the sample is pre-filled, from the run
        # open on the API tab (see seed_sample).

        # Raw (uncut) sample vs background: energy spectra overlaid on the left,
        # time (dt) spectra overlaid on the right.
        self.files_fig, self.files_canvas, files_tb = self._make_canvas(320)
        lay.addWidget(files_tb)
        lay.addWidget(self.files_canvas, 1)
        self.ax_files_e, self.ax_files_t = self.files_fig.subplots(1, 2)
        self._style_ax(self.ax_files_e, "Energy", "Counts", yscale="log")
        self._style_ax(self.ax_files_t, "dt (ns)", "Counts")
        return tab

    def _on_file_loaded(self, role):
        loaded = [ROLE_LABELS[r] for r in ROLES if self.rows[r].df is not None]
        self._status(f"Loaded: {', '.join(loaded) if loaded else 'none'}")
        self._draw_files()
        self._schedule()

    def _on_offset_changed(self, role):
        """A dt / energy offset was edited on the Files tab: redraw the overlay
        immediately (that is the point of putting them there) and schedule the
        downstream recompute."""
        self._draw_files()
        self._schedule()

    def _draw_files(self):
        """Overlay the sample and background energy spectra (left) and their dt
        time spectra (right), with each dataset's offsets applied — this is the
        view you tune those offsets against."""
        ax_e, ax_t = self.ax_files_e, self.ax_files_t
        ax_e.clear(); ax_t.clear()
        styles = {"sample": (T.ACCENT_CYAN, "Sample"),
                  "background": (T.ACCENT_AMBER, "Background")}
        any_e = any_t = False
        shifted = []
        for role, (color, label) in styles.items():
            row = self.rows[role]
            df = row.df
            if df is None:
                continue
            dt_off, e_off = row.dt_offset, row.e_offset
            if dt_off or e_off:
                shifted.append(f"{label} dt{dt_off:+g} ns E{e_off:+g} keV")
            ekey = auto_ekey(df)
            if ekey is not None:
                cts, edg = np.histogram(df[ekey] + e_off, bins=2048,
                                        range=[0, 10000])
                ec = (edg[1:] + edg[:-1]) / 2
                ax_e.step(ec, cts, where="mid", color=color, lw=0.9, label=label)
                any_e = True
            if "dt" in df.columns:
                ctt, edt = np.histogram(df["dt"] + dt_off, bins=512,
                                        range=[-20, 50])
                tc = (edt[1:] + edt[:-1]) / 2
                ax_t.step(tc, ctt, where="mid", color=color, lw=0.9, label=label)
                any_t = True
        self._style_ax(ax_e, "Energy", "Counts", yscale="log")
        self._style_ax(ax_t, "dt (ns)", "Counts")
        sub = ("  [" + ";  ".join(shifted) + "]") if shifted else ""
        ax_e.set_title("Energy spectra" + sub, color=T.TEXT_DIM, fontsize=10)
        ax_t.set_title("Time spectra" + sub, color=T.TEXT_DIM, fontsize=10)
        for ax, has in ((ax_e, any_e), (ax_t, any_t)):
            if has:
                leg = ax.legend(fontsize=9, loc="upper right")
                if leg:
                    leg.get_frame().set_facecolor(T.BG_PANEL)
                    for txt in leg.get_texts():
                        txt.set_color(T.TEXT_PRIMARY)
        self.files_fig.tight_layout()
        self.files_canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════════
    # Tab 2 — Corrections
    # ══════════════════════════════════════════════════════════════════════
    def _build_corrections_tab(self):
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(14, 12, 14, 12); outer.setSpacing(16)

        left = QWidget(); left.setFixedWidth(360)
        lay = QVBoxLayout(left); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(8)
        outer.addWidget(left, 0)

        lay.addWidget(header("GEOMETRY / DETECTOR CORRECTIONS"))
        self.cb_stretch = QCheckBox("Apply X-stretch factor")
        self.cb_stretch.setChecked(True)
        self.cb_stretch.setToolTip(
            "Multiply the reconstructed X2 position by the geometry factor "
            "(1.27 for FLARE) to correct the X scale.")
        self.cb_stretch.toggled.connect(self._schedule)
        lay.addWidget(self.cb_stretch)
        self.ed_stretch = self._field(1.27, tip="X-stretch geometry factor")
        row, _ = labeled_row("X stretch ×", self.ed_stretch); lay.addWidget(row)

        self.cb_ystretch = QCheckBox("Apply Y-stretch factor")
        self.cb_ystretch.setChecked(False)
        self.cb_ystretch.setToolTip(
            "Multiply the reconstructed Y2 position by a geometry factor.  Off by "
            "default (the FLARE reference only stretches X); enable it if your "
            "setup needs a Y correction too.")
        self.cb_ystretch.toggled.connect(self._schedule)
        lay.addWidget(self.cb_ystretch)
        self.ed_ystretch = self._field(1.0, tip="Y-stretch geometry factor")
        row, _ = labeled_row("Y stretch ×", self.ed_ystretch); lay.addWidget(row)

        lay.addWidget(hsep())
        lay.addWidget(header("ALPHA-ENERGY THRESHOLD"))
        self.cb_thresh = QCheckBox("Cut alpha pile-up above threshold")
        self.cb_thresh.setChecked(True)
        self.cb_thresh.setToolTip(
            "Keep only events with alpha_energy < threshold (drops pile-up).  The "
            "kept fraction scales the alpha-trigger normalisation.")
        self.cb_thresh.toggled.connect(self._schedule)
        lay.addWidget(self.cb_thresh)
        self.ed_thresh = self._field(20000, tip="alpha_energy threshold")
        row, _ = labeled_row("Threshold", self.ed_thresh); lay.addWidget(row)

        lay.addWidget(hsep())
        lay.addWidget(header("TIMING / ENERGY OFFSETS"))
        tnote = QLabel(
            "The per-dataset dt and energy offsets live on the Files tab, next "
            "to the overlaid time and energy spectra — nudge one there and "
            "watch the peaks line up.")
        tnote.setObjectName("stat_key"); tnote.setWordWrap(True)
        lay.addWidget(tnote)

        lay.addWidget(hsep())
        lay.addWidget(header("NON-ALPHA FRACTION"))
        note = QLabel("Fraction of the alpha spectrum that is not tagged alphas "
                      "(gammas + DD/TT charged particles).")
        note.setObjectName("stat_key"); note.setWordWrap(True)
        lay.addWidget(note)
        self.ed_non_a = self._field(0.0486, tip="non-α fraction (0.04 + 0.0086)")
        row, _ = labeled_row("non-α", self.ed_non_a); lay.addWidget(row)
        self.ed_sig_non_a = self._field(0.0, tip="1σ uncertainty on the non-α fraction")
        row, _ = labeled_row("σ(non-α)", self.ed_sig_non_a); lay.addWidget(row)

        lay.addStretch(1)

        # Right: live readouts on top, then the sample's alpha-energy spectrum
        # (with the threshold marked) and the X-Y map (with the stretch applied).
        right = QWidget()
        rlay = QVBoxLayout(right); rlay.setContentsMargins(0, 0, 0, 0); rlay.setSpacing(6)
        stat_grid = QGridLayout(); stat_grid.setHorizontalSpacing(18)
        for col, (key, accent, attr) in enumerate([
                ("Sample kept frac", T.ACCENT_CYAN, "val_thresh_s"),
                ("Corrected α (sample)", T.ACCENT_AMBER, "val_alphas_s")]):
            r, v = stat_row(key, accent); setattr(self, attr, v)
            stat_grid.addWidget(r, 0, col)
        for col, (key, accent, attr) in enumerate([
                ("Bkg kept frac", T.ACCENT_CYAN, "val_thresh_b"),
                ("Corrected α (bkg)", T.ACCENT_AMBER, "val_alphas_b")]):
            r, v = stat_row(key, accent); setattr(self, attr, v)
            stat_grid.addWidget(r, 1, col)
        sg = QWidget(); sg.setLayout(stat_grid)
        rlay.addWidget(sg)

        self.corr_fig, self.corr_canvas, corr_tb = self._make_canvas(380)
        rlay.addWidget(corr_tb)
        rlay.addWidget(self.corr_canvas, 1)
        self.ax_corr_a = self.corr_fig.add_subplot(1, 2, 1)
        self.ax_corr_xy = self.corr_fig.add_subplot(1, 2, 2)
        self._style_ax(self.ax_corr_a, "alpha_energy", "Counts", yscale="log")
        self._style_ax(self.ax_corr_xy, "X (stretched)", "Y")
        outer.addWidget(right, 1)
        return tab

    def _draw_corrections(self):
        """Alpha-energy spectrum (raw, with the threshold marked) and the X-Y map
        with the current stretch applied — for the sample."""
        ax_a, ax_xy = self.ax_corr_a, self.ax_corr_xy
        ax_a.clear(); ax_xy.clear()
        raw = self.rows["sample"].df
        if raw is not None and "alpha_energy" in raw.columns:
            a = raw["alpha_energy"].to_numpy()
            hi = np.nanpercentile(a, 99.9) if a.size else 1.0
            ax_a.hist(a, bins=400, range=[0, max(hi, 1.0)],
                      color=T.ACCENT_CYAN, histtype="step", lw=0.9)
            if self.cb_thresh.isChecked():
                thr = _num(self.ed_thresh.text(), 20000.0)
                ax_a.axvline(thr, color=T.ACCENT_RED, lw=1.4, ls="--")
                ax_a.axvspan(thr, max(hi, thr), alpha=0.10, color=T.ACCENT_RED)
        self._style_ax(ax_a, "alpha_energy", "Counts", yscale="log")
        ax_a.set_title("Alpha spectrum + threshold", color=T.TEXT_DIM, fontsize=10)

        # X-Y map with the stretch (and threshold) applied.
        dfc, _, _ = self._corrected_df("sample")
        if dfc is not None and {"X2", "Y2"}.issubset(dfc.columns) and dfc.shape[0]:
            ax_xy.hist2d(dfc["X2"], dfc["Y2"], bins=90,
                         range=[[-0.9, 0.9], [-0.9, 0.9]], cmap="plasma")
        self._style_ax(ax_xy, "X (stretched)", "Y")
        ax_xy.set_xlim(-0.9, 0.9); ax_xy.set_ylim(-0.9, 0.9)
        ax_xy.set_title("X-Y map (stretch applied)", color=T.TEXT_DIM, fontsize=10)
        self.corr_fig.tight_layout()
        self.corr_canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════════
    # Tab 3 — Cuts
    # ══════════════════════════════════════════════════════════════════════
    def _build_cuts_tab(self):
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(14, 12, 14, 12); outer.setSpacing(16)

        left = QWidget(); left.setFixedWidth(330)
        lay = QVBoxLayout(left); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        outer.addWidget(left, 0)

        lay.addWidget(header("X / Y / t CUTS"))
        note = QLabel(
            "Applied to both the sample and the background.  Type a range or drag "
            "directly on the panels — a rectangle on the X-Y map, a band on the "
            "gamma or time spectrum.")
        note.setObjectName("stat_key"); note.setWordWrap(True)
        lay.addWidget(note)

        self.ed_xlo = self._field(-0.68); self.ed_xhi = self._field(0.68)
        self.ed_ylo = self._field(-0.68); self.ed_yhi = self._field(0.68)
        self.ed_tlo = self._field(0.0); self.ed_thi = self._field(25.0)
        for lbl, lo, hi in [("X", self.ed_xlo, self.ed_xhi),
                            ("Y", self.ed_ylo, self.ed_yhi),
                            ("t (ns)", self.ed_tlo, self.ed_thi)]:
            lay.addWidget(self._range_row(lbl, lo, hi))

        lay.addWidget(hsep())
        lay.addWidget(header("BKG-SCALING WINDOW"))
        enote = QLabel(
            "⚠ This energy window is used ONLY to scale the background (it picks "
            "the time-spectrum band where the sample and background are matched).  "
            "It does NOT define the net photopeak counts — those come from the "
            "fit on the Net counts tab.")
        enote.setObjectName("stat_key"); enote.setWordWrap(True)
        enote.setStyleSheet(f"color:{T.ACCENT_AMBER};")
        lay.addWidget(enote)
        self.ed_elo = self._field(810.0); self.ed_ehi = self._field(900.0)
        lay.addWidget(self._range_row("E scale (keV)", self.ed_elo, self.ed_ehi))

        lay.addWidget(hsep())
        self.ekey = QComboBox()
        self.ekey.addItems(["auto", "energy_cal", "energy", "energy_orig"])
        self.ekey.setToolTip(
            "Energy column to histogram for the photopeak.  'auto' prefers "
            "energy_cal when the run has it (the calibrated keV axis), then "
            "energy, then energy_orig.")
        self.ekey.currentIndexChanged.connect(self._schedule)
        row, _ = labeled_row("Energy col", self.ekey); lay.addWidget(row)

        self.ed_ebins = self._field(4096, tip="energy bins (full 0–10000 keV histogram)")
        row, _ = labeled_row("Energy bins", self.ed_ebins); lay.addWidget(row)
        self.ed_tbins = self._field(2048, tip="dt bins over the -20–50 ns window")
        row, _ = labeled_row("dt bins", self.ed_tbins); lay.addWidget(row)

        lay.addWidget(hsep())
        r, self.val_n_sample = stat_row("Sample events (cut)", T.ACCENT_CYAN); lay.addWidget(r)
        r, self.val_n_bkg = stat_row("Bkg events (cut)", T.ACCENT_CYAN); lay.addWidget(r)
        lay.addStretch(1)

        # Right: X-Y map on the left (tall), gamma spectrum stacked on top of the
        # time spectrum on the right — three panels side by side were too cramped.
        right = QWidget()
        rlay = QVBoxLayout(right); rlay.setContentsMargins(0, 0, 0, 0); rlay.setSpacing(4)
        self.cuts_fig, self.cuts_canvas, cuts_tb = self._make_canvas(420)
        rlay.addWidget(cuts_tb)
        rlay.addWidget(self.cuts_canvas, 1)
        gs = self.cuts_fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0])
        self.ax_cut_xy = self.cuts_fig.add_subplot(gs[:, 0])   # X-Y, full height
        self.ax_cut_e = self.cuts_fig.add_subplot(gs[0, 1])    # gamma (top)
        self.ax_cut_t = self.cuts_fig.add_subplot(gs[1, 1])    # time (bottom)
        self._style_ax(self.ax_cut_xy, "X", "Y")
        self._style_ax(self.ax_cut_e, "Energy (keV)", "Counts", yscale="log")
        self._style_ax(self.ax_cut_t, "dt (ns)", "Counts")
        self._attach_cut_selectors()
        outer.addWidget(right, 1)
        return tab

    def _attach_cut_selectors(self):
        """Drag-to-cut on the Cuts-tab panels: a rectangle on the X-Y map sets the
        X/Y range; a band on the gamma or time spectrum sets the t range (time
        panel) — all of which then drive a recompute."""
        from matplotlib.widgets import RectangleSelector, SpanSelector

        def _on_xy(eclick, erelease):
            x0, x1 = sorted((eclick.xdata, erelease.xdata))
            y0, y1 = sorted((eclick.ydata, erelease.ydata))
            if None in (x0, x1, y0, y1):
                return
            self.ed_xlo.setText(f"{x0:.3g}"); self.ed_xhi.setText(f"{x1:.3g}")
            self.ed_ylo.setText(f"{y0:.3g}"); self.ed_yhi.setText(f"{y1:.3g}")
            self._schedule()
        self._cut_rect = RectangleSelector(
            self.ax_cut_xy, _on_xy, useblit=False, button=[1],
            props=dict(facecolor=T.ACCENT_CYAN, alpha=0.15, edgecolor=T.ACCENT_CYAN))

        def _on_t(lo, hi):
            if hi > lo:
                self.ed_tlo.setText(f"{lo:.3g}"); self.ed_thi.setText(f"{hi:.3g}")
                self._schedule()
        self._cut_tspan = SpanSelector(
            self.ax_cut_t, _on_t, "horizontal", useblit=False,
            props=dict(alpha=0.18, facecolor=T.ACCENT_AMBER))

    def _range_row(self, label, ed_lo, ed_hi):
        row = QWidget()
        rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(6)
        rl.addWidget(QLabel(label)); rl.addStretch(1)
        rl.addWidget(ed_lo); rl.addWidget(QLabel("→")); rl.addWidget(ed_hi)
        return row

    # ══════════════════════════════════════════════════════════════════════
    # Tab 4 — Background & Net
    # ══════════════════════════════════════════════════════════════════════
    def _build_bkg_tab(self):
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(14, 12, 14, 12); outer.setSpacing(16)

        left = QWidget(); left.setFixedWidth(320)
        lay = QVBoxLayout(left); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        outer.addWidget(left, 0)

        lay.addWidget(header("BACKGROUND SCALE  C"))
        note = QLabel("Scale the background, then subtract → net spectrum.  By "
                      "default C = α_sample / α_bkg (the corrected alpha-trigger "
                      "ratio).  Override it to least-squares match the two time "
                      "spectra within the C-fit window instead.")
        note.setObjectName("stat_key"); note.setWordWrap(True)
        lay.addWidget(note)
        self.cb_c_override = QCheckBox("Override: least-squares C in window")
        self.cb_c_override.setToolTip(
            "Off (default): C = ratio of corrected alpha triggers "
            "(α_sample / α_bkg).\n"
            "On: minimise the least-squares residual between the sample and "
            "background time spectra over the C-fit window below.")
        self.cb_c_override.toggled.connect(self._schedule)
        lay.addWidget(self.cb_c_override)
        self.ed_clo = self._field(15.0); self.ed_chi = self._field(20.0)
        lay.addWidget(self._range_row("C fit dt (ns)", self.ed_clo, self.ed_chi))

        lay.addWidget(hsep())
        lay.addWidget(header("NET PHOTOPEAK — FIND & FIT"))
        pnote = QLabel(
            "Pick a detector and SNR, find the peaks on the net spectrum (fast "
            "search), then drag over one or more peaks to fit them.  The fit's net "
            "area becomes the photopeak count once you click 'Apply to σ'.")
        pnote.setObjectName("stat_key"); pnote.setWordWrap(True)
        lay.addWidget(pnote)

        self.cmb_detector = QComboBox()
        self.cmb_detector.addItems(list(DETECTOR_PRESETS.keys()))
        self.cmb_detector.setCurrentText("LaBr/CeBr")
        self.cmb_detector.setToolTip(
            "Detector type — sets the reference resolution the peak search uses.")
        self.cmb_detector.currentTextChanged.connect(self._apply_detector_preset)
        row, _ = labeled_row("Detector", self.cmb_detector); lay.addWidget(row)

        self.ed_snr = self._field(5.0, tip="Minimum SNR — lower finds weaker peaks")
        row, _ = labeled_row("SNR >", self.ed_snr); lay.addWidget(row)

        prow = QHBoxLayout(); prow.setContentsMargins(0, 0, 0, 0); prow.setSpacing(6)
        self.btn_find_peaks = QPushButton("Find peaks")
        self.btn_find_peaks.setObjectName("find_btn")
        self.btn_find_peaks.setCursor(Qt.PointingHandCursor)
        self.btn_find_peaks.setToolTip("Run the fast peak search on the net spectrum")
        self.btn_find_peaks.clicked.connect(self._find_net_peaks)
        self.btn_clear_peaks = QPushButton("Clear")
        self.btn_clear_peaks.setObjectName("mini_btn")
        self.btn_clear_peaks.setCursor(Qt.PointingHandCursor)
        self.btn_clear_peaks.clicked.connect(self._clear_net_peaks)
        prow.addWidget(self.btn_find_peaks, 1); prow.addWidget(self.btn_clear_peaks, 0)
        prw = QWidget(); prw.setLayout(prow); lay.addWidget(prw)
        hint = QLabel("→ drag over one or more peaks to fit, then Apply to σ")
        hint.setObjectName("stat_key"); hint.setWordWrap(True); lay.addWidget(hint)

        lay.addWidget(hsep())
        r, self.val_C = stat_row("C", T.ACCENT_AMBER); lay.addWidget(r)
        r, self.val_peak = stat_row("Net photopeak", T.ACCENT_GREEN); lay.addWidget(r)
        self.val_peak_src = QLabel("source: —")
        self.val_peak_src.setObjectName("stat_key")
        lay.addWidget(self.val_peak_src)
        lay.addStretch(1)

        right = QWidget()
        rlay = QVBoxLayout(right); rlay.setContentsMargins(0, 0, 0, 0); rlay.setSpacing(4)
        self.bkg_fig, self.bkg_canvas, bkg_tb = self._make_canvas(360)
        rlay.addWidget(bkg_tb)
        rlay.addWidget(self.bkg_canvas, 1)
        self.bkg_axes = self.bkg_fig.subplots(2, 1)
        self._style_ax(self.bkg_axes[0], "Time (ns)", "Counts")
        self._style_ax(self.bkg_axes[1], "Energy (keV)", "Net counts")
        # Drag over peaks to fit a region; a click selects the nearest single
        # peak (release-based, so a click is distinguished from a drag).
        self.bkg_canvas.mpl_connect("button_press_event", self._on_net_press)
        self.bkg_canvas.mpl_connect("button_release_event", self._on_net_release)
        self._attach_net_span()
        outer.addWidget(right, 1)
        return tab

    # ── net-spectrum peak search + click-to-fit ────────────────────────────
    def _apply_detector_preset(self, name):
        preset = DETECTOR_PRESETS.get(name)
        if not preset:
            return
        snr, ref_x, ref_fwhm = preset
        self.ed_snr.setText(snr)              # SNR stays user-editable
        self._ref_x = float(ref_x)            # reference resolution is internal
        self._ref_fwhm = float(ref_fwhm)

    def _find_net_peaks(self):
        """Run the fast peak search on the net spectrum and mark the peaks."""
        if self._net_spectrum is None:
            self._status("Build the net spectrum first (load sample + background).")
            return
        try:
            search = ps.PeakSearch(
                spectrum=self._net_spectrum,
                ref_x=self._ref_x, ref_fwhm=self._ref_fwhm,
                fwhm_at_0=1.0, min_snr=_num(self.ed_snr.text(), 5.0),
                method="fast")
        except Exception as exc:  # noqa: BLE001
            self._status(f"Peak search failed: {exc}")
            return
        self._net_search = search
        idx = search.peaks_idx
        if idx is None or len(idx) == 0:
            self._net_peaks = []
            self._status("No peaks found — lower the SNR and try again.")
        else:
            x = self._net_spectrum.x
            y = self._net_spectrum.counts
            self._net_peaks = [(float(x[i]), float(y[i])) for i in idx]
            self._status(f"Found {len(self._net_peaks)} peaks — click one to fit it.")
        self._redraw_net_panel()

    def _clear_net_peaks(self):
        self._net_peaks = []
        self._net_search = None
        self._redraw_net_panel()
        self._status("Cleared found peaks.")

    def _on_net_press(self, event):
        if event.inaxes == self.bkg_axes[1] and event.xdata is not None:
            self._net_press_x = event.xdata
        else:
            self._net_press_x = None

    def _on_net_release(self, event):
        px0 = self._net_press_x
        self._net_press_x = None
        if (px0 is None or event.inaxes != self.bkg_axes[1]
                or event.xdata is None):
            return
        # A drag (wide press→release) is handled by the span selector; only a
        # click (small movement) selects a peak to fit.
        if abs(event.xdata - px0) > 15:
            return
        if not self._net_peaks:
            self._status("Find peaks first, then click one to fit it.")
            return
        # Nearest found peak to the click.
        pe = min(self._net_peaks, key=lambda p: abs(p[0] - event.xdata))[0]
        fwhm = 30.0
        if self._net_search is not None:
            try:
                fwhm = float(self._net_search.fwhm(pe))
            except Exception:  # noqa: BLE001
                pass
        self._open_or_set_roi(pe - 3 * fwhm, pe + 3 * fwhm)

    def _open_or_set_roi(self, lo, hi):
        """Open the fit window at this ROI (or move an open one's ROI)."""
        if hi <= lo:
            return
        if self._fit_win is None or not self._fit_win.isVisible():
            self._open_peak_fit(roi=(lo, hi))
        else:
            self._fit_win.set_roi(lo, hi)
            self._fit_win.raise_(); self._fit_win.activateWindow()

    # ══════════════════════════════════════════════════════════════════════
    # Tab 5 — Fluence / Nt
    # ══════════════════════════════════════════════════════════════════════
    def _build_fluence_tab(self):
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(14, 12, 14, 12); outer.setSpacing(16)

        left = QWidget(); left.setFixedWidth(330)
        lay = QVBoxLayout(left); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        outer.addWidget(left, 0)

        lay.addWidget(header("ALPHA FRACTION  fa"))
        self.cb_fa_override = QCheckBox("Override fa manually")
        self.cb_fa_override.setToolTip(
            "Use a fixed fa instead of the value from the beam-profile fit "
            "(e.g. a previously measured / cached alpha fraction).")
        self.cb_fa_override.toggled.connect(self._schedule)
        lay.addWidget(self.cb_fa_override)
        self.ed_fa_override = self._field(0.1864, tip="Manual alpha fraction fa")
        row, _ = labeled_row("fa value", self.ed_fa_override); lay.addWidget(row)
        lay.addWidget(hsep())
        lay.addWidget(header("BEAM-PROFILE FIT"))
        note = QLabel(
            "The target is a flat square, so any smooth tilt across it must be "
            "instrumental.  The fit solves for the target shape and that gain "
            "together — no detector distances, so it works on any run/channel.")
        note.setObjectName("stat_key"); note.setWordWrap(True)
        lay.addWidget(note)

        self.ed_xybins = self._field(25, tip="X-Y grid size for the profile fit")
        row, _ = labeled_row("Profile bins", self.ed_xybins); lay.addWidget(row)

        self.cmb_gain_order = QComboBox()
        self.cmb_gain_order.addItems(["0 — none (flat)", "1 — tilt (recommended)",
                                      "2 — tilt + curvature"])
        self.cmb_gain_order.setCurrentIndex(1)
        self.cmb_gain_order.setToolTip(
            "Order of the recovered gain field.\n"
            "1 is a pure exponential ramp — what a 1/r² falloff looks like "
            "across a small target, and the validated default.\n"
            "2 adds curvature that is partly degenerate with the target's own "
            "edge sharpness; use it only as a systematic check.")
        self.cmb_gain_order.currentIndexChanged.connect(self._schedule)
        row, _ = labeled_row("Gain order", self.cmb_gain_order); lay.addWidget(row)

        self.cb_center = QCheckBox("Auto-centre the target")
        self.cb_center.setChecked(True)
        self.cb_center.setToolTip(
            "Fit once to locate the target, roll it to the middle of the frame "
            "(integer pixels, lossless) and re-fit.  Removes the centre / width "
            "/ tilt degeneracy when the target is off-centre.")
        self.cb_center.toggled.connect(self._schedule)
        lay.addWidget(self.cb_center)

        self.cb_use_flat = QCheckBox("Use flat-field run for fa")
        self.cb_use_flat.setChecked(True)
        self.cb_use_flat.setToolTip(
            "fa = flat-field counts inside the fitted mask / total.  Untick to "
            "use the mask's plain area fraction instead (no flat-field run "
            "needed, but less accurate).")
        self.cb_use_flat.toggled.connect(self._schedule)
        lay.addWidget(self.cb_use_flat)

        # The profile/flat-field windows are usually a touch wider than the
        # sample cuts (the reference uses ±0.75), so give them their own X-Y range.
        self.ed_pxlo = self._field(-0.75); self.ed_pxhi = self._field(0.75)
        lay.addWidget(self._range_row("Profile X", self.ed_pxlo, self.ed_pxhi))
        self.ed_pylo = self._field(-0.75); self.ed_pyhi = self._field(0.75)
        lay.addWidget(self._range_row("Profile Y", self.ed_pylo, self.ed_pyhi))
        self.cmb_pekey = QComboBox()
        self.cmb_pekey.addItems(["auto", "energy_cal", "energy", "energy_orig"])
        self.cmb_pekey.setToolTip(
            "Energy column for the profile cut.  'auto' prefers energy_cal when "
            "the run has it, then energy, then energy_orig.")
        self.cmb_pekey.currentIndexChanged.connect(self._schedule)
        row, _ = labeled_row("Profile E col", self.cmb_pekey); lay.addWidget(row)
        self.ed_pelo = self._field(800.0)
        self.ed_pehi = self._field(1300.0)
        lay.addWidget(self._range_row("Profile E", self.ed_pelo, self.ed_pehi))
        self.ed_ptlo = self._field(4.1); self.ed_pthi = self._field(8.1)
        lay.addWidget(self._range_row("Profile t (ns)", self.ed_ptlo, self.ed_pthi))
        self.ed_namc = self._field(500, tip="MC samples for the fa uncertainty")
        row, _ = labeled_row("fa MC", self.ed_namc); lay.addWidget(row)

        lay.addWidget(hsep())
        lay.addWidget(header("SAMPLE GEOMETRY"))
        self.ed_area = self._field(100.0, tip="Sample area (cm²)")
        row, _ = labeled_row("Area (cm²)", self.ed_area); lay.addWidget(row)
        self.ed_sig_area = self._field(1.0, tip="1σ area uncertainty (cm²)")
        row, _ = labeled_row("σ(area)", self.ed_sig_area); lay.addWidget(row)

        self.btn_fit_profile = QPushButton("Fit profile → fa")
        self.btn_fit_profile.setObjectName("primary_btn")
        self.btn_fit_profile.setCursor(Qt.PointingHandCursor)
        self.btn_fit_profile.setToolTip(
            "Fit the beam profile and compute the alpha fraction.  Run after "
            "loading the profile + background (+ flat-field) files.")
        self.btn_fit_profile.clicked.connect(self._fit_profile)
        lay.addWidget(self.btn_fit_profile)

        lay.addWidget(hsep())
        r, self.val_fa = stat_row("fa", T.ACCENT_AMBER); lay.addWidget(r)
        # Fit-quality readout: how flat the target got, against the Poisson floor
        # (the irreducible scatter — reaching it means the flattening is done).
        r, self.val_flatness = stat_row("Non-uniformity", T.ACCENT_CYAN); lay.addWidget(r)
        r, self.val_floor = stat_row("Poisson floor", T.TEXT_DIM); lay.addWidget(r)
        r, self.val_fluence = stat_row("Fluence (n/cm²)", T.ACCENT_GREEN); lay.addWidget(r)
        r, self.val_flux = stat_row("Flux (n/cm²/s)", T.ACCENT_GREEN); lay.addWidget(r)
        r, self.val_flux_geo = stat_row("Flux (geometry)", T.TEXT_DIM); lay.addWidget(r)
        lay.addStretch(1)

        # Right: the geometry-free 6-panel diagnostic (raw image, recovered gain,
        # flattened image, fitted mask, X/Y profiles) — via fpu.plot_result(fig=…).
        right = QWidget()
        rlay = QVBoxLayout(right); rlay.setContentsMargins(0, 0, 0, 0); rlay.setSpacing(4)
        self.prof_fig, self.prof_canvas, prof_tb = self._make_canvas(420)
        rlay.addWidget(prof_tb)
        rlay.addWidget(self.prof_canvas, 1)
        self.prof_ax = self.prof_fig.add_subplot(111)
        self._style_ax(self.prof_ax, "")
        self.prof_ax.text(0.5, 0.5, "Fit the beam profile to see the diagnostic",
                          transform=self.prof_ax.transAxes, ha="center", va="center",
                          color=T.TEXT_DIM, fontsize=12)
        outer.addWidget(right, 1)
        return tab

    # ══════════════════════════════════════════════════════════════════════
    # Tab 6 — Cross section
    # ══════════════════════════════════════════════════════════════════════
    def _build_xs_tab(self):
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(14, 12, 14, 12); outer.setSpacing(16)

        left = QWidget(); left.setFixedWidth(330)
        lay = QVBoxLayout(left); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        outer.addWidget(left, 0)

        lay.addWidget(header("TARGET ATOMS  Nt"))
        self.ed_mass = self._field(602.47, tip="Sample mass (g)")
        row, _ = labeled_row("Mass (g)", self.ed_mass); lay.addWidget(row)
        self.ed_sig_mass = self._field(0.05, tip="1σ mass uncertainty (g)")
        row, _ = labeled_row("σ(mass)", self.ed_sig_mass); lay.addWidget(row)
        self.ed_M = self._field(55.845, tip="Molar mass (g/mol)")
        row, _ = labeled_row("M (g/mol)", self.ed_M); lay.addWidget(row)
        self.ed_fm = self._field(0.91754, tip="Isotopic abundance fraction")
        row, _ = labeled_row("Abundance", self.ed_fm); lay.addWidget(row)
        r, self.val_nt = stat_row("Nt (atoms)", T.ACCENT_CYAN); lay.addWidget(r)

        lay.addWidget(hsep())
        lay.addWidget(header("EFFICIENCY"))
        self.ed_eff = self._field(2.26e-3, tip="Photopeak efficiency at the line")
        row, _ = labeled_row("Efficiency", self.ed_eff); lay.addWidget(row)
        self.ed_eff_err = self._field(0.085, tip="Fractional efficiency uncertainty")
        row, _ = labeled_row("σ(eff) frac", self.ed_eff_err); lay.addWidget(row)

        lay.addWidget(hsep())
        lay.addWidget(header("MONTE CARLO"))
        self.ed_nmc = self._field(10000, tip="MC samples for the cross-section band")
        row, _ = labeled_row("MC samples", self.ed_nmc); lay.addWidget(row)
        self.btn_run_mc = QPushButton("Run full MC")
        self.btn_run_mc.setObjectName("find_btn")
        self.btn_run_mc.setCursor(Qt.PointingHandCursor)
        self.btn_run_mc.clicked.connect(self._recompute)
        lay.addWidget(self.btn_run_mc)

        lay.addWidget(hsep())
        lay.addWidget(header("RESULT"))
        r, self.val_xs = stat_row("σ (mb)", T.ACCENT_GREEN); lay.addWidget(r)
        r, self.val_xs_err = stat_row("σ uncertainty", T.ACCENT_AMBER); lay.addWidget(r)
        r, self.val_xs_rel = stat_row("Relative", T.ACCENT_AMBER); lay.addWidget(r)

        self.summary = QLabel("—")
        self.summary.setObjectName("stat_key"); self.summary.setWordWrap(True)
        self.summary.setStyleSheet(f"color:{T.TEXT_PRIMARY}; font-family:{T.MONO_FAMILY};"
                                   " font-size:12px;")
        lay.addWidget(self.summary)
        lay.addStretch(1)

        right = QWidget()
        rlay = QVBoxLayout(right); rlay.setContentsMargins(0, 0, 0, 0); rlay.setSpacing(4)
        self.xs_fig, self.xs_canvas, xs_tb = self._make_canvas(360)
        rlay.addWidget(xs_tb)
        rlay.addWidget(self.xs_canvas, 1)
        self.xs_ax = self.xs_fig.add_subplot(111)
        self._style_ax(self.xs_ax, "σ (mb)", "MC samples")
        outer.addWidget(right, 1)
        return tab

    # ══════════════════════════════════════════════════════════════════════
    # Parameter gathering
    # ══════════════════════════════════════════════════════════════════════
    def _cut_ranges(self):
        """The X / Y / t cut ranges from this dialog's own Cuts-tab fields
        (self-contained — no dependence on the main API page)."""
        xrange = [_num(self.ed_xlo.text(), -0.68), _num(self.ed_xhi.text(), 0.68)]
        yrange = [_num(self.ed_ylo.text(), -0.68), _num(self.ed_yhi.text(), 0.68)]
        trange = [_num(self.ed_tlo.text(), 0.0), _num(self.ed_thi.text(), 25.0)]
        return xrange, yrange, trange

    def _corrected_df(self, role):
        """Return (df, kept_fraction, alphas_corrected) for a role with the
        X-stretch, the row's dt / energy offsets and the alpha threshold
        applied.  None if the file is not loaded."""
        r = self.rows[role]
        if r.df is None:
            return None, 1.0, None
        df = r.df.copy()
        if self.cb_stretch.isChecked():
            s = _num(self.ed_stretch.text(), 1.27)
            if "X2" in df.columns:
                df["X2"] = df["X2"] * s
        if self.cb_ystretch.isChecked():
            sy = _num(self.ed_ystretch.text(), 1.0)
            if "Y2" in df.columns:
                df["Y2"] = df["Y2"] * sy
        # Per-dataset dt / energy offsets, set on the Files tab.
        df = r.apply_offsets(df)
        kept = 1.0
        if self.cb_thresh.isChecked():
            df, kept = cs.threshold_correction(
                df, threshold=_num(self.ed_thresh.text(), 20000.0))
        alphas = r.alphas * kept if r.alphas is not None else None
        return df, kept, alphas

    # ══════════════════════════════════════════════════════════════════════
    # Live recompute
    # ══════════════════════════════════════════════════════════════════════
    def _recompute(self):
        self._in_recompute = True
        try:
            self._recompute_inner()
        except Exception as exc:  # noqa: BLE001 — never let a bad edit kill the UI
            traceback.print_exc()
            self._status(f"Recompute error: {exc}")
        finally:
            self._in_recompute = False

    def _recompute_inner(self):
        # Need at least the sample to do anything meaningful.
        dfs, kept_s, alphas_s = self._corrected_df("sample")
        dfb, kept_b, alphas_b = self._corrected_df("background")
        # Corrections readouts.
        self.val_thresh_s.setText(f"{kept_s:.4f}" if dfs is not None else "—")
        self.val_thresh_b.setText(f"{kept_b:.4f}" if dfb is not None else "—")
        self.val_alphas_s.setText(fmt_count(alphas_s) if alphas_s else "—")
        self.val_alphas_b.setText(fmt_count(alphas_b) if alphas_b else "—")
        self._draw_corrections()
        if dfs is None:
            self._status("Load the sample dataset to compute a cross section.")
            return

        xrange, yrange, trange = self._cut_ranges()
        erange = [_num(self.ed_elo.text(), 810.0), _num(self.ed_ehi.text(), 900.0)]
        choice = self.ekey.currentText()
        ekey = auto_ekey(dfs, None if choice == "auto" else choice)
        if ekey is None:
            self._status("The sample file has no energy column "
                         "(energy_cal / energy / energy_orig).")
            return
        ebins = int(_num(self.ed_ebins.text(), 4096))
        tbins = int(_num(self.ed_tbins.text(), 2048))

        # ── Step 3: cuts ──────────────────────────────────────────────────
        dfsxyt = api.dftxy(dfs, xrange, yrange, trange, xkey="X2", ykey="Y2")
        self.val_n_sample.setText(f"{dfsxyt.shape[0]:,}")
        if dfb is not None:
            dfbxyt = api.dftxy(dfb, xrange, yrange, trange, xkey="X2", ykey="Y2")
            self.val_n_bkg.setText(f"{dfbxyt.shape[0]:,}")
        else:
            dfbxyt = None
            self.val_n_bkg.setText("—")

        # Full energy spectra (0–10000 keV) for the sample / background.
        cts_s, edg = np.histogram(dfsxyt[ekey], bins=ebins, range=[0, 10000])
        e_axis = (edg[1:] + edg[:-1]) / 2
        if dfbxyt is not None:
            cts_b, _ = np.histogram(dfbxyt[ekey], bins=ebins, range=[0, 10000])
        else:
            cts_b = np.zeros_like(cts_s)

        self._draw_cuts(dfs, dfsxyt, xrange, yrange, trange, e_axis, cts_s,
                        ekey, tbins)

        # ── Step 4: background scale C + net ──────────────────────────────
        self._compute_background(dfs, dfb, xrange, yrange, erange, ekey, tbins,
                                 alphas_s, alphas_b, e_axis, cts_s, cts_b)

        # ── Step 5/6: Nt, fluence, cross section ──────────────────────────
        Nt = cs.compute_nt(_num(self.ed_mass.text(), 602.47),
                           _num(self.ed_M.text(), 55.845),
                           _num(self.ed_fm.text(), 0.91754))
        self.val_nt.setText(f"{Nt:.4g}")

        fa, fa_dist = self._update_fa_readouts()

        t_sample = self.rows["sample"].t_total or 1.0
        non_a = _num(self.ed_non_a.text(), 0.0486)
        area = _num(self.ed_area.text(), 100.0)
        if alphas_s and fa:
            fluence = cs.compute_fluence(alphas_s, non_a, fa, area)
            self.val_fluence.setText(f"{fluence:.4g}")
            self.val_flux.setText(f"{fluence / t_sample:.4g}")
        else:
            self.val_fluence.setText("—")
            self.val_flux.setText("—")

        # Geometry-approximation flux for comparison.
        rs = self.rows["sample"]
        if rs.date and rs.runnr is not None:
            try:
                flux_geo = api.calculate_neutron_flux(
                    date=rs.date, runnr=rs.runnr, ch=9, L=30)
                self.val_flux_geo.setText(f"{flux_geo:.4g}")
            except Exception:  # noqa: BLE001
                self.val_flux_geo.setText("—")

        if not (alphas_s and fa):
            self._status("Fit the beam profile (tab 5) to finish the cross section.")
            self.headline.setText("XS = —   (fit profile)")
            return
        if not self._peak_gamma:
            self._status("Fit the net photopeak (tab 4) and click 'Apply to σ' "
                         "to finish the cross section.")
            self.headline.setText("XS = —   (fit the photopeak)")
            return

        # ── Step 7: MC cross section ──────────────────────────────────────
        inp = cs.CrossSectionInputs(
            peak_gamma=self._peak_gamma, sig_gamma=self._sig_gamma,
            mass=_num(self.ed_mass.text(), 602.47),
            sig_mass=_num(self.ed_sig_mass.text(), 0.05),
            M=_num(self.ed_M.text(), 55.845),
            fm=_num(self.ed_fm.text(), 0.91754),
            eff=_num(self.ed_eff.text(), 2.26e-3),
            eff_rel_err=_num(self.ed_eff_err.text(), 0.085),
            alphas=alphas_s, fa=fa, fa_dist=fa_dist,
            non_a=non_a, sig_non_a=_num(self.ed_sig_non_a.text(), 0.0),
            A_sample=area, sig_A_sample=_num(self.ed_sig_area.text(), 1.0),
            t_sample=t_sample, n_MC=int(_num(self.ed_nmc.text(), 10000)),
        )
        res = cs.cross_section_mc(inp)
        self._last_result = res
        self._update_xs_readouts(res, inp)

    # ── drawing helpers ────────────────────────────────────────────────────
    def _draw_cuts(self, dfs, dfsxyt, xrange, yrange, trange, e_axis, cts_s,
                   ekey, tbins):
        ax_xy, ax_e, ax_t = self.ax_cut_xy, self.ax_cut_e, self.ax_cut_t
        for ax in (ax_xy, ax_e, ax_t):
            ax.clear()

        # X-Y map: the t-cut data over a wide fixed plane so the user can see the
        # whole field and drag a (possibly larger) X/Y rectangle; the current X/Y
        # cut is overlaid.
        plane = [-0.9, 0.9]
        dft = api.dft(dfs, trange) if dfs is not None else dfsxyt
        if dft.shape[0]:
            ax_xy.hist2d(dft["X2"], dft["Y2"], bins=90,
                         range=[plane, plane], cmap="plasma")
        from matplotlib.patches import Rectangle
        ax_xy.add_patch(Rectangle(
            (xrange[0], yrange[0]), xrange[1] - xrange[0], yrange[1] - yrange[0],
            fill=False, edgecolor=T.ACCENT_CYAN, lw=1.6, ls="--"))
        self._style_ax(ax_xy, "X", "Y")
        ax_xy.set_xlim(*plane); ax_xy.set_ylim(*plane)

        # Gamma spectrum (top): the fully-cut sample spectrum that feeds the net.
        ax_e.step(e_axis, cts_s, where="mid", color=T.ACCENT_CYAN, lw=0.9)
        ax_e.set_xlim(max(0, e_axis.min()), e_axis.max())
        self._style_ax(ax_e, "Energy (keV)", "Counts", yscale="log")

        # Time spectrum (bottom): the X/Y-cut data over the full window, with the
        # t-cut band shown so the user can drag a new one.
        dfxy = api.dfxy(dfs, xrange, yrange) if dfs is not None else dfsxyt
        cts_t, edt = np.histogram(dfxy["dt"], bins=int(tbins), range=[-20, 50])
        et = (edt[1:] + edt[:-1]) / 2
        ax_t.step(et, cts_t, where="mid", color=T.ACCENT_AMBER, lw=0.9)
        ax_t.axvspan(trange[0], trange[1], alpha=0.14, color=T.ACCENT_GREEN)
        self._style_ax(ax_t, "dt (ns)", "Counts")

        self.cuts_fig.tight_layout()
        # Re-arm the drag selectors (cleared axes drop them).
        self._attach_cut_selectors()
        self.cuts_canvas.draw_idle()

    def _compute_background(self, dfs, dfb, xrange, yrange, erange, ekey, tbins,
                            alphas_s, alphas_b, e_axis, cts_s, cts_b):
        ax_t, ax_net = self.bkg_axes
        ax_t.clear(); ax_net.clear()

        # Time spectra within the energy window (for the C fit).
        dfsxye = api.dfxye(dfs, xrange, yrange, erange, xkey="X2", ykey="Y2",
                           ekey=ekey if ekey in dfs.columns else "energy")
        cts_ts, edt = np.histogram(dfsxye["dt"], bins=int(tbins), range=[-20, 50])
        et = (edt[1:] + edt[:-1]) / 2

        crange = [_num(self.ed_clo.text(), 15.0), _num(self.ed_chi.text(), 20.0)]
        C = 1.0; sig_C = 0.0
        if dfb is not None:
            dfbxye = api.dfxye(dfb, xrange, yrange, erange, xkey="X2", ykey="Y2",
                               ekey=ekey if ekey in dfb.columns else "energy")
            cts_tb, _ = np.histogram(dfbxye["dt"], bins=int(tbins), range=[-20, 50])
            # Least-squares C over the C-fit window (always computed; it is the
            # fallback when alpha-trigger counts are unavailable).
            try:
                C_fit, sig_fit = bkg_scale.find_background_scale(
                    cts_ts, cts_tb, et, xrange=crange)
            except Exception:  # noqa: BLE001
                C_fit, sig_fit = 1.0, 0.0
            if self.cb_c_override.isChecked():
                C, sig_C = C_fit, sig_fit                 # override: least-squares
            elif alphas_s and alphas_b:
                C, sig_C = alphas_s / alphas_b, 0.0        # default: alpha ratio
            else:
                C, sig_C = C_fit, sig_fit                  # no alphas → fall back
            # Always show the scaled background.  Dashed, so it stays visible even
            # when Bkg × C sits right on top of the Sample curve.
            ax_t.step(et, cts_tb * C, where="mid", color=T.ACCENT_AMBER,
                      lw=1.1, ls="--", label="Bkg × C")
        ax_t.step(et, cts_ts, where="mid", color=T.ACCENT_CYAN, lw=0.9, label="Sample")
        # The t-cut (applied to the spectra) is always marked.  The C-fit window
        # is only shown when Override C is checked (per the user's request).
        ax_t.axvspan(_num(self.ed_tlo.text(), 0.0), _num(self.ed_thi.text(), 25.0),
                     alpha=0.12, color=T.ACCENT_GREEN, label="t cut")
        if self.cb_c_override.isChecked():
            for xc in crange:
                ax_t.axvline(xc, color=T.ACCENT_AMBER, lw=1.0, ls=":")
            ax_t.axvspan(crange[0], crange[1], alpha=0.10, color=T.ACCENT_AMBER,
                         label="C fit window")
        self._style_ax(ax_t, "Time (ns)", "Counts")
        leg = ax_t.legend(fontsize=8, loc="upper right")
        if leg:
            leg.get_frame().set_facecolor(T.BG_PANEL)
            for txt in leg.get_texts():
                txt.set_color(T.TEXT_PRIMARY)

        self._C, self._sig_C = C, sig_C
        self.val_C.setText(f"{C:.4g}")

        # Net spectrum.  Build it by Spectrum subtraction (exactly as
        # sigma-56Fe.py does) so the per-bin uncertainty propagates to
        # √(s + C·b) — NOT the much smaller Poisson √(net) of a bare counts array.
        # Using the right errors is what makes the peak-fit reduced χ² ≈ 1; a bare
        # √(net) under-estimates the error and inflates χ² several-fold.
        cts_b_scaled = cts_b * C
        spe_s = sp.Spectrum(counts=cts_s, energies=e_axis, e_units="keV",
                            label="Sample")
        spe_b = sp.Spectrum(counts=cts_b_scaled, energies=e_axis, e_units="keV",
                            label="Bkg")
        spe_net = spe_s - spe_b          # counts_err = √(s + C·b)
        spe_net.replace_neg_vals()       # negatives → 0.1 × min positive (as in the script)
        spe_net.label = "Net"
        self._net_spectrum = spe_net
        self._draw_net_axis()

        # Photopeak counts come only from the interactive fit.  The applied value
        # is sticky: a recompute (tab change, cut, parameter edit) never overwrites
        # it — it changes only when the user clicks "Apply to σ" in the fit window.
        if self._peak_applied:
            self._peak_source = "interactive fit (applied)"
            self.val_peak.setText(f"{self._peak_gamma:.4g} ± {self._sig_gamma:.3g}")
        else:
            self._peak_source = "fit the photopeak"
            self.val_peak.setText("—  (fit the photopeak)")
        self.val_peak_src.setText(f"source: {self._peak_source}")
        self.bkg_fig.tight_layout()
        self.bkg_canvas.draw_idle()

    def _draw_net_axis(self):
        """(Re)draw the net-spectrum panel: net step, found-peak markers and the
        applied-fit ROI band.  Shared by the full recompute and the peak search."""
        ax_net = self.bkg_axes[1]
        ax_net.clear()
        if self._net_spectrum is None:
            self._style_ax(ax_net, "Energy (keV)", "Net counts")
            return
        x, y = self._net_spectrum.x, self._net_spectrum.counts
        ax_net.step(x, y, where="mid", color=T.ACCENT_GREEN, lw=0.9,
                    label="Net = Sample − Bkg")
        # Applied-fit ROI band.
        if self._peak_applied and self._applied_roi is not None:
            ax_net.axvspan(self._applied_roi[0], self._applied_roi[1],
                           alpha=0.10, color=T.ACCENT_CYAN, label="fit ROI")
        # Found peaks (click one to fit).
        for pe, pc in self._net_peaks:
            ax_net.axvline(pe, color=T.ACCENT_AMBER, lw=1.0, ls="--", alpha=0.65)
            ax_net.annotate(f"{pe:.0f}", xy=(pe, pc), xytext=(0, 4),
                            textcoords="offset points", color=T.ACCENT_AMBER,
                            fontsize=8, ha="center", rotation=90)
        self._style_ax(ax_net, "Energy (keV)", "Net counts")
        leg = ax_net.legend(fontsize=9, loc="upper right")
        if leg:
            leg.get_frame().set_facecolor(T.BG_PANEL)
            for txt in leg.get_texts():
                txt.set_color(T.TEXT_PRIMARY)
        # Clearing the axis drops the span selector — re-arm it so drag-to-fit
        # keeps working (it is always available, even before a fit is open).
        self._attach_net_span()

    def _redraw_net_panel(self):
        self._draw_net_axis()
        self.bkg_fig.tight_layout()
        self.bkg_canvas.draw_idle()

    # ── interactive photopeak fit window ───────────────────────────────────
    def _build_net_search(self):
        """A PeakSearch on the current net spectrum, seeding the fit window — uses
        the detector preset's reference energy / FWHM / SNR."""
        return ps.PeakSearch(
            spectrum=self._net_spectrum,
            ref_x=self._ref_x, ref_fwhm=self._ref_fwhm,
            fwhm_at_0=1.0, min_snr=_num(self.ed_snr.text(), 5.0),
            method="fast")

    def _open_peak_fit(self, roi=None):
        """Open the interactive peak-fit window on the net spectrum (the same
        FitWindow the Spectrum tab uses), seeded at *roi* (lo, hi) — usually a
        found peak ± 3·FWHM.  A span on the net-spectrum panel also sets the ROI."""
        if self._net_spectrum is None:
            QMessageBox.information(
                self, "No net spectrum yet",
                "Load the sample + background and let the net spectrum build "
                "first, then open the photopeak fit.")
            return
        from .fitting import FitWindow
        from PyQt5.QtWidgets import QPushButton
        if self._fit_win is None:
            self._fit_win = FitWindow(self, self._build_net_search())
            self._fit_win.setWindowTitle("Net photopeak fit")
            self._fit_win._in_refit = False     # guards the table-check signal
            self._fit_win._checked_rows = None  # remembered checked peaks (by row)
            # Send-to-Calibration/Efficiency/Resolution aren't part of measuring
            # the photopeak area — hide them to keep the bar focused (same as the
            # time-slice fit window in Selections).
            for b in (self._fit_win.btn_to_cal, self._fit_win.btn_to_eff,
                      self._fit_win.btn_to_res):
                b.hide()
            # After each refit show a cheap *preview* of the fit's area — but do
            # NOT touch the applied photopeak count or σ.  Both change only on
            # "Apply to σ", so tab/parameter changes never reset the value.
            _orig_refit = self._fit_win._refit

            def _refit_and_preview(_o=_orig_refit, _w=self._fit_win):
                _w._in_refit = True             # the table refill must not be
                try:                            # treated as a check toggle
                    _o()
                except Exception:  # noqa: BLE001 — a bad ROI must not crash the UI
                    traceback.print_exc()
                    _w._in_refit = False
                    return
                _w._in_refit = False
                self._preview_peak(_w)
            self._fit_win._refit = _refit_and_preview

            # Tick a peak's checkbox (column 0 of the results table) to include it
            # in the photopeak count; untick to drop it.
            self._fit_win.table.itemChanged.connect(
                lambda it: self._on_peak_check_changed(self._fit_win, it))

            # "Apply to σ" button: push the current fitted area into the
            # cross-section and recompute once, on demand.
            self._fit_win.btn_apply_sigma = QPushButton("✓  Apply to σ")
            self._fit_win.btn_apply_sigma.setObjectName("primary_btn")
            self._fit_win.btn_apply_sigma.setCursor(Qt.PointingHandCursor)
            self._fit_win.btn_apply_sigma.setToolTip(
                "Use the selected peak's net area as the photopeak count and "
                "recompute the cross section.")
            self._fit_win.btn_apply_sigma.clicked.connect(self._apply_peak_fit)
            self._fit_win.layout().addWidget(self._fit_win.btn_apply_sigma)
        else:
            self._fit_win.set_search(self._build_net_search())
        # Seed the ROI from the clicked peak (fallback to a central window).
        if roi is None:
            roi = (820.0, 880.0)
        self._cur_roi = (float(roi[0]), float(roi[1]))
        self._fit_win.set_roi(self._cur_roi[0], self._cur_roi[1])
        self._fit_win.show()
        self._fit_win.raise_()
        self._fit_win.activateWindow()
        self._status("Adjust the ROI, then click 'Apply to σ' to set the photopeak.")

    def _apply_peak_fit(self):
        """Apply the current photopeak fit to σ (the slow MC), on the user's
        explicit request — the only place the applied photopeak count changes."""
        if self._fit_win is None:
            return
        area, sigma = self._read_fit_area(self._fit_win)
        if not area:
            self._status("Nothing to apply — fit a peak and tick at least one row.")
            return
        self._peak_gamma, self._sig_gamma = area, sigma
        self._peak_applied = True
        n = len(self._checked_peak_rows(self._fit_win))
        self._peak_source = (f"interactive fit (applied — {n} peak"
                             f"{'s' if n != 1 else ''})")
        # Record the fit ROI so the net panel can mark the applied region.
        roi = getattr(self._fit_win, "_xrange", None)
        self._applied_roi = tuple(roi) if roi else getattr(self, "_cur_roi", None)
        self.val_peak.setText(f"{area:.4g} ± {sigma:.3g}")
        self.val_peak_src.setText(f"source: {self._peak_source}")
        self._redraw_net_panel()
        self._update_xs_from_cache()
        self._status(f"Applied photopeak {area:.4g} ± {sigma:.3g} → σ")

    def _attach_net_span(self):
        """Persistent span selector on the net-spectrum panel: dragging over one
        or more peaks opens the fit window on that region (or moves the ROI of an
        already-open one)."""
        from matplotlib.widgets import SpanSelector
        ax_net = self.bkg_axes[1]
        if self._net_span is not None:
            try:
                self._net_span.disconnect_events()
            except Exception:  # noqa: BLE001
                pass
        self._net_span = SpanSelector(
            ax_net, lambda lo, hi: self._open_or_set_roi(lo, hi), "horizontal",
            useblit=False, props=dict(alpha=0.18, facecolor=T.ACCENT_CYAN))

    def _setup_peak_checks(self, fitwin):
        """Make column 0 of each peak row a checkbox (checked = include the peak
        in the photopeak count).  Restores the remembered tick state by row so it
        survives refits; defaults to all-checked for a fresh fit."""
        tbl = getattr(fitwin, "table", None)
        fit = getattr(fitwin, "last_fit", None)
        npk = len(fit.peak_info) if (fit is not None
                                     and getattr(fit, "peak_info", None)) else 0
        if tbl is None or npk == 0 or tbl.rowCount() != npk:
            return                              # net-area method etc. — no checks
        prev = fitwin._checked_rows
        tbl.blockSignals(True)
        for r in range(npk):
            it = tbl.item(r, 0)
            if it is None:
                continue
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            checked = True if prev is None else (r in prev)
            it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        tbl.blockSignals(False)

    @staticmethod
    def _checked_peak_rows(fitwin):
        """Indices of the peak rows whose checkbox is ticked (all rows if the
        table isn't checkable yet)."""
        tbl = getattr(fitwin, "table", None)
        fit = getattr(fitwin, "last_fit", None)
        npk = len(fit.peak_info) if (fit is not None
                                     and getattr(fit, "peak_info", None)) else 0
        if tbl is None or npk == 0 or tbl.rowCount() != npk:
            return list(range(npk))
        rows = []
        for r in range(npk):
            it = tbl.item(r, 0)
            if it is None or not (it.flags() & Qt.ItemIsUserCheckable):
                rows.append(r)                  # not checkable yet → included
            elif it.checkState() == Qt.Checked:
                rows.append(r)
        return rows

    def _read_fit_area(self, fitwin):
        """Net peak area ± σ from the fit window: the sum over the *ticked* peaks
        (errors added in quadrature).  Falls back to the net-area − bkg result."""
        fit = getattr(fitwin, "last_fit", None)
        if fit is not None and getattr(fit, "peak_info", None):
            infos, errs = fit.peak_info, fit.peak_err
            rows = self._checked_peak_rows(fitwin)
            if not rows:
                return 0.0, 0.0                 # nothing ticked
            areas = [float(infos[i]["area"]) for i in rows]
            es = [float(errs[i]["area_err"]) for i in rows]
            return float(sum(areas)), float(np.sqrt(sum(e ** 2 for e in es)))
        alb = getattr(fitwin, "_last_area", None)   # net-area − linear bkg
        if alb is not None:
            return float(alb.A), float(alb.sigA)
        return None, None

    def _on_peak_check_changed(self, fitwin, item):
        """A peak checkbox was toggled: remember the ticked rows and refresh the
        preview readout (σ still only changes on Apply)."""
        if getattr(fitwin, "_in_refit", False) or item is None or item.column() != 0:
            return
        fitwin._checked_rows = set(self._checked_peak_rows(fitwin))
        area, sigma = self._read_fit_area(fitwin)
        if area is not None:
            n = len(fitwin._checked_rows)
            self.val_peak.setText(f"{area:.4g} ± {sigma:.3g}  (preview · {n} peak"
                                  f"{'s' if n != 1 else ''})")
            self.val_peak_src.setText("source: fit preview — click Apply to σ to use")

    def _preview_peak(self, fitwin):
        """Show the current fit's area as a *preview* without changing the applied
        photopeak count or σ — that only happens on 'Apply to σ'."""
        self._setup_peak_checks(fitwin)
        area, sigma = self._read_fit_area(fitwin)
        if area is None:
            return
        self.val_peak.setText(f"{area:.4g} ± {sigma:.3g}  (preview)")
        self.val_peak_src.setText("source: fit preview — click Apply to σ to use")

    def _current_fa(self):
        """The alpha fraction to use: the manual override when ticked, else the
        beam-profile fit result.  Returns (fa, fa_dist) — fa_dist is None for a
        manual value (no MC distribution)."""
        if self.cb_fa_override.isChecked():
            return _num(self.ed_fa_override.text(), 0.0), None
        if self._fa_result is not None:
            return self._fa_result.fa, self._fa_result.fa_dist
        return 0.0, None

    def _update_fa_readouts(self):
        """Refresh the tab-5 fa / fit-quality readouts.  Returns (fa, fa_dist)."""
        fa, fa_dist = self._current_fa()
        if self.cb_fa_override.isChecked():
            self.val_fa.setText(f"{fa:.4g}  (manual)")
        elif fa and fa_dist is not None and fa_dist.size > 1:
            self.val_fa.setText(f"{fa:.4g} ± {fa_dist.std():.3g}")
        else:
            self.val_fa.setText(f"{fa:.4g}" if fa else "—  (fit profile)")

        # Fit quality: the flattening is as good as it can get once the residual
        # non-uniformity reaches the Poisson floor.
        if self._fa_result is not None:
            fr = self._fa_result.flat
            self.val_flatness.setText(
                f"{100*fr.rms_raw:.1f}% → {100*fr.rms_flat:.1f}%")
            self.val_floor.setText(f"{100*fr.poisson_floor:.1f}%")
        else:
            self.val_flatness.setText("—")
            self.val_floor.setText("—")
        return fa, fa_dist

    def _update_xs_from_cache(self):
        """Re-run only the Nt/fluence/σ stages from cached fa + peak, so adjusting
        the photopeak fit updates σ without re-cutting the data."""
        fa, fa_dist = self._current_fa()
        if not fa or not self._peak_gamma:
            return
        dfs, kept_s, alphas_s = self._corrected_df("sample")
        if not alphas_s:
            return
        t_sample = self.rows["sample"].t_total or 1.0
        inp = cs.CrossSectionInputs(
            peak_gamma=self._peak_gamma, sig_gamma=self._sig_gamma,
            mass=_num(self.ed_mass.text(), 602.47),
            sig_mass=_num(self.ed_sig_mass.text(), 0.05),
            M=_num(self.ed_M.text(), 55.845), fm=_num(self.ed_fm.text(), 0.91754),
            eff=_num(self.ed_eff.text(), 2.26e-3),
            eff_rel_err=_num(self.ed_eff_err.text(), 0.085),
            alphas=alphas_s, fa=fa, fa_dist=fa_dist,
            non_a=_num(self.ed_non_a.text(), 0.0486),
            sig_non_a=_num(self.ed_sig_non_a.text(), 0.0),
            A_sample=_num(self.ed_area.text(), 100.0),
            sig_A_sample=_num(self.ed_sig_area.text(), 1.0),
            t_sample=t_sample, n_MC=int(_num(self.ed_nmc.text(), 10000)))
        res = cs.cross_section_mc(inp)
        self._last_result = res
        self._update_xs_readouts(res, inp)

    # ── profile fit (alpha fraction) ───────────────────────────────────────
    def _fit_profile(self):
        try:
            self._fit_profile_inner()
        except ValueError as exc:
            # Empty / degenerate beam spot — the engine raises an actionable
            # message; show it so the user knows which ranges to widen.
            self._status(f"Profile fit: {exc}")
            QMessageBox.warning(self, "Beam profile is empty", str(exc))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._status(f"Profile fit error: {exc}")

    def _profile_ekey(self, df):
        """Energy column for the profile cut: the user's pick, or the first one
        the file actually has (``energy_cal`` preferred)."""
        choice = self.cmb_pekey.currentText()
        ekey = auto_ekey(df, None if choice == "auto" else choice)
        if ekey is None:
            raise ValueError("The profile file has no energy column "
                             "(energy_cal / energy / energy_orig).")
        return ekey

    def _fit_profile_inner(self):
        prof_r = self.rows["profile"]; flat_r = self.rows["flat"]
        bkg_r = self.rows["background"]
        if prof_r.df is None or bkg_r.df is None:
            QMessageBox.information(
                self, "Need profile + background",
                "Load the profile and background datasets first.")
            return
        use_flat = self.cb_use_flat.isChecked()
        if use_flat and flat_r.df is None:
            QMessageBox.information(
                self, "Need a flat-field run",
                "Load the flat-field dataset, or untick 'Use flat-field run for "
                "fa' to fall back on the mask's area fraction.")
            return

        df_profile, _, alphas_p = self._corrected_df("profile")
        df_bkg, _, alphas_b = self._corrected_df("background")
        df_flat = self._corrected_df("flat")[0] if use_flat else None

        xrange = [_num(self.ed_pxlo.text(), -0.75), _num(self.ed_pxhi.text(), 0.75)]
        yrange = [_num(self.ed_pylo.text(), -0.75), _num(self.ed_pyhi.text(), 0.75)]
        trange = [_num(self.ed_ptlo.text(), 4.1), _num(self.ed_pthi.text(), 8.1)]
        erange = [_num(self.ed_pelo.text(), 800.0), _num(self.ed_pehi.text(), 1300.0)]
        xybins = int(_num(self.ed_xybins.text(), 25))

        self._status("Fitting beam profile…")
        res = fpu.alpha_fraction_from_data(
            df_profile, df_bkg, df_flat,
            xrange=xrange, yrange=yrange, trange=trange, erange=erange,
            xybins=xybins, alphas_profile=alphas_p, alphas_bkg=alphas_b,
            ekey=self._profile_ekey(df_profile),
            order=self.cmb_gain_order.currentIndex(),
            center=self.cb_center.isChecked(),
            n_MC=int(_num(self.ed_namc.text(), 500)))
        self._fa_result = res
        self._draw_profile(res)
        # Fill the fa readouts here rather than leaving it to _recompute, which
        # bails out early until the sample run is loaded.
        self._update_fa_readouts()

        fr = res.flat
        msg = (f"Alpha fraction fa = {res.fa:.4g}   "
               f"(non-uniformity {100*fr.rms_raw:.1f}% → {100*fr.rms_flat:.1f}%, "
               f"Poisson floor {100*fr.poisson_floor:.1f}%")
        if fr.was_centered:
            msg += f", re-centred by {fr.shift} px"
        self._status(msg + ")")
        self._recompute()

    def _draw_profile(self, res):
        # Render the geometry-free diagnostic (raw / gain / flattened / mask /
        # projections) onto the embedded canvas.
        fpu.plot_result(res, fig=self.prof_fig)
        self.prof_canvas.draw_idle()

    # ── cross-section readouts ─────────────────────────────────────────────
    def _update_xs_readouts(self, res, inp):
        self.val_xs.setText(f"{res.xs_mb:.4g}")
        self.val_xs_err.setText(f"± {res.xs_mb_err:.3g} mb")
        self.val_xs_rel.setText(f"{res.xs_rel_err:.2f} %")
        self.val_fluence.setText(f"{res.fluence:.4g}")
        self.val_flux.setText(f"{res.flux:.4g}")
        self.headline.setText(f"XS = {res.xs_mb:.4g} mb  ±  {res.xs_mb_err:.3g} mb"
                              f"   ({res.xs_rel_err:.1f} %)")
        self.summary.setText(
            f"Nt = {res.Nt:.4g} atoms\n"
            f"peak γ = {inp.peak_gamma:.4g} ± {inp.sig_gamma:.3g}\n"
            f"C = {self._C:.4g}\n"
            f"fa = {inp.fa:.4g}\n"
            f"fluence = {res.fluence:.4g} n/cm²\n"
            f"flux = {res.flux:.4g} n/cm²/s")
        self._status(f"σ = {res.xs_mb:.4g} ± {res.xs_mb_err:.3g} mb")

        if res.xs_dist.size > 1:
            self.xs_ax.clear()
            self.xs_ax.hist(res.xs_dist, bins=80, color=T.ACCENT_GREEN, alpha=0.85)
            self.xs_ax.axvline(res.xs_mb, color=T.ACCENT_AMBER, lw=1.4)
            self._style_ax(self.xs_ax, "σ (mb)", "MC samples")
            self.xs_fig.tight_layout()
            self.xs_canvas.draw_idle()
